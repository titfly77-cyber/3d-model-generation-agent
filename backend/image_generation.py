"""Configurable image generation for three-view reference sheets.

Official fal protocols (checked 2026-10-04):
https://fal.ai/models/fal-ai/qwen-image-2512/api
https://docs.fal.ai/model-apis/model-endpoints/queue

The default model is Qwen-Image-2512. Other fal endpoints and the common
OpenAI Images-compatible protocol are configurable. Layout and cross-view
consistency are verified by the reference stage, not asserted by this adapter.
"""

import base64
import binascii
import hashlib
import io
import ipaddress
import json
import re
import socket
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable
from urllib.parse import urlsplit

import httpx
from PIL import Image, ImageOps, UnidentifiedImageError


DEFAULT_MODEL = "fal-ai/qwen-image-2512"
MODEL = "Qwen/Qwen-Image-2512"  # Compatibility for metadata readers of old jobs.
ENDPOINT = "https://queue.fal.run/" + DEFAULT_MODEL
IMAGE_SIZE = {"width": 1664, "height": 928}
DEADLINE_SECONDS = 900.0
POLL_SECONDS = 2.0
MAX_IMAGE_BYTES = 32 * 1024 * 1024
MAX_IMAGE_PIXELS = 16_000_000
MAX_JSON_BYTES = 48 * 1024 * 1024


class ImageGenerationError(RuntimeError):
    """Safe, user-readable provider error, without upstream response bodies."""


@contextmanager
def _stream(client, method, url, **kwargs):
    """Catch transport errors without catching exceptions from caller callbacks."""
    try:
        manager = client.stream(method, url, **kwargs)
        response = manager.__enter__()
    except httpx.RequestError:
        raise ImageGenerationError("图片服务连接中断，任务记录已保留，请稍后恢复。") from None
    try:
        yield response
    except BaseException:
        try:
            manager.__exit__(None, None, None)
        except httpx.RequestError:
            pass
        raise
    else:
        try:
            manager.__exit__(None, None, None)
        except httpx.RequestError:
            raise ImageGenerationError("图片服务连接中断，任务记录已保留，请稍后恢复。") from None


def _chunks(response):
    iterator = response.iter_bytes(chunk_size=64 * 1024)
    while True:
        try:
            chunk = next(iterator)
        except StopIteration:
            return
        except httpx.RequestError:
            raise ImageGenerationError("图片服务连接中断，任务记录已保留，请稍后恢复。") from None
        yield chunk


def _write_job(path: Path, job: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _configuration(prompt, cfg):
    provider = cfg.get("image_provider", "fal")
    if provider not in ("fal", "siliconflow", "volcengine-ark", "openai-compatible"):
        raise ImageGenerationError("不支持的文生图服务协议。")
    model = cfg.get("image_model") or (DEFAULT_MODEL if provider == "fal" else "")
    if not isinstance(model, str) or not model.strip() or len(model) > 240:
        raise ImageGenerationError("请在 API 设置中填写文生图模型名称。")
    model = model.strip()
    key = cfg.get("image_api_key") or (cfg.get("generative_api_key") if provider == "fal" else "")
    if not isinstance(key, str) or not key.strip():
        raise ImageGenerationError("请先在 API 设置中配置文生图服务 API Key。")
    parameters = cfg.get("image_parameters", {})
    if not isinstance(parameters, dict) or any(not isinstance(k, str) for k in parameters):
        raise ImageGenerationError("文生图额外参数必须是 JSON 对象。")
    if any(k.lower() in ("headers", "authorization", "api_key", "api-key") for k in parameters):
        raise ImageGenerationError("请将文生图密钥填写在 API Key 字段，不要放入额外参数。")
    if parameters.get("n", 1) != 1 or parameters.get("num_images", 1) != 1:
        raise ImageGenerationError("每个任务只生成一张三视图参考图，请将图片数量设为 1。")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ImageGenerationError("三视图生成提示词不能为空。")
    if provider == "fal":
        if not re.fullmatch(r"[A-Za-z0-9_-]+/[A-Za-z0-9_-]+(?:/[A-Za-z0-9_-]+)*", model):
            raise ImageGenerationError("fal 模型名称应为组织/模型路径，不能包含网址、查询参数或相对路径。")
        endpoint = "https://queue.fal.run/" + model
        payload = ({"image_size": IMAGE_SIZE, "num_images": 1, "num_inference_steps": 50,
                    "guidance_scale": 4, "output_format": "png", "enable_safety_checker": True}
                   if model == DEFAULT_MODEL else {})
        payload.update(parameters)
        payload["prompt"] = prompt
    else:
        defaults = {"siliconflow": "https://api.siliconflow.cn/v1",
                    "volcengine-ark": "https://ark.cn-beijing.volces.com/api/v3"}
        base = cfg.get("image_base_url", "") or defaults.get(provider, "")
        if not isinstance(base, str):
            raise ImageGenerationError("请填写文生图服务的 HTTP 或 HTTPS API 地址。")
        base = base.strip().rstrip("/")
        parsed = urlsplit(base)
        if (parsed.scheme not in (("https",) if provider in ("siliconflow", "volcengine-ark") else ("http", "https")) or not parsed.hostname
                or parsed.username or parsed.password or parsed.query or parsed.fragment):
            raise ImageGenerationError("请填写文生图服务的 HTTP 或 HTTPS API 地址。")
        endpoint = base if base.endswith("/images/generations") else base + "/images/generations"
        provider_parameters = ({"size": "2K", "response_format": "url", "output_format": "png",
                                "watermark": True, "sequential_image_generation": "disabled"}
                               if provider == "volcengine-ark" else {})
        payload = {**provider_parameters, **parameters, "model": model, "prompt": prompt}
    try:
        encoded = json.dumps({"provider": provider, "model": model, "endpoint": endpoint,
                              "payload": payload}, sort_keys=True, allow_nan=False).encode()
    except (ValueError, TypeError):
        raise ImageGenerationError("文生图额外参数包含无法序列化的值。") from None
    if len(encoded) > 256 * 1024:
        raise ImageGenerationError("文生图提示词和额外参数过长。")
    return provider, model, endpoint, payload, key.strip(), hashlib.sha256(encoded).hexdigest()


def _request_url(value: str, endpoint: str, request_id: str, operation: str) -> str:
    # fal nested routes may return job URLs at their app root, e.g.
    # /fal-ai/flux/requests/id for the model /fal-ai/flux/dev.
    path = urlsplit(endpoint).path
    roots = {path, "/" + "/".join(path.strip("/").split("/")[:2])}
    suffixes = ["/" + operation] if operation else ["", "/response"]
    allowed = {root + "/requests/" + request_id + suffix for root in roots for suffix in suffixes}
    default = endpoint + "/requests/" + request_id + suffixes[0]
    url = value or default
    if not isinstance(url, str):
        raise ImageGenerationError("图片服务返回了无法验证的任务地址。")
    parsed = urlsplit(url)
    if (parsed.scheme != "https" or parsed.netloc != "queue.fal.run"
            or parsed.path not in allowed or parsed.query or parsed.fragment):
        raise ImageGenerationError("图片服务返回了无法验证的任务地址。")
    return url


def _json_response(response: httpx.Response) -> dict:
    if response.status_code >= 400:
        status = response.status_code
        if status in (401, 403):
            message = "请检查文生图 API Key 与模型访问权限"
        elif status in (402, 429):
            message = "请检查生成服务余额或请求限额"
        else:
            message = "图片生成服务暂时不可用，请稍后恢复任务"
        raise ImageGenerationError(f"图片服务 HTTP {status}：{message}。")
    if not 200 <= response.status_code < 300:
        raise ImageGenerationError("图片服务返回了意外的重定向。")
    try:
        data = response.json()
    except (ValueError, UnicodeError):
        raise ImageGenerationError("图片服务没有返回有效 JSON。") from None
    if not isinstance(data, dict):
        raise ImageGenerationError("图片服务返回的数据格式不正确。")
    return data


def _normalized_image(raw: bytes) -> Image.Image:
    if not raw or len(raw) > MAX_IMAGE_BYTES:
        raise ImageGenerationError("参考图片为空或超过 32 MB 限制。")
    try:
        with Image.open(io.BytesIO(raw)) as source:
            if source.format not in ("PNG", "JPEG", "WEBP"):
                raise ImageGenerationError("参考图片必须是 PNG、JPEG 或 WebP 格式。")
            width, height = source.size
            if width < 256 or height < 256 or width * height > MAX_IMAGE_PIXELS:
                raise ImageGenerationError("参考图片尺寸不合适，最短边需至少 256 像素且总像素不超过 1600 万。")
            if getattr(source, "is_animated", False):
                raise ImageGenerationError("参考图片必须是静态图片。")
            source.verify()
        with Image.open(io.BytesIO(raw)) as source:
            source.load()
            source = ImageOps.exif_transpose(source)
            if "A" in source.getbands() or "transparency" in source.info:
                rgba = source.convert("RGBA")
                white = Image.new("RGBA", rgba.size, "white")
                return Image.alpha_composite(white, rgba).convert("RGB")
            return source.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        raise ImageGenerationError("图片内容损坏或无法解码，请重新生成参考图。") from None


def _decode_base64(value):
    if not isinstance(value, str) or len(value) > MAX_IMAGE_BYTES * 4 // 3 + 16:
        raise ImageGenerationError("图片服务返回的内嵌图片大小不正确。")
    try:
        return base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error):
        raise ImageGenerationError("图片服务返回的内嵌图片编码不正确。") from None


def _validate_download_url(url, provider):
    try:
        parsed = urlsplit(url)
        host = parsed.hostname or ""
        if (parsed.scheme != "https" or not host or parsed.username or parsed.password
                or parsed.port not in (None, 443)):
            raise ImageGenerationError("图片服务返回了不受支持的下载地址。")
    except ValueError:
        raise ImageGenerationError("图片服务返回的下载地址无效。") from None
    if provider == "fal":
        if not (host == "fal.media" or host.endswith(".fal.media")):
            raise ImageGenerationError("fal 图片服务返回了未知图片存储地址。")
    else:
        # Compatible image services use different CDNs. Reject local/private
        # destinations; configurable local services should return b64_json.
        try:
            addresses = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
        except OSError:
            raise ImageGenerationError("无法解析参考图片下载地址，请恢复任务重试。") from None
        if not addresses or any(not ipaddress.ip_address(item[4][0]).is_global for item in addresses):
            raise ImageGenerationError("图片下载地址不能指向本机或内网。")


def _download_image(client, url, provider, guard, timeout):
    if url.startswith("data:"):
        match = re.fullmatch(r"data:image/(?:png|jpeg|webp);base64,(.*)", url, re.S)
        if not match:
            raise ImageGenerationError("图片服务返回的内嵌图片格式不正确。")
        guard()
        return _decode_base64(match[1])
    _validate_download_url(url, provider)
    guard()
    # The client has no default Authorization header. CDN receives no API key.
    with _stream(client, "GET", url, timeout=timeout(), follow_redirects=False) as response:
        if response.status_code != 200:
            raise ImageGenerationError(f"参考图片下载失败（HTTP {response.status_code}）。")
        length = response.headers.get("content-length", "")
        if length.isdigit() and int(length) > MAX_IMAGE_BYTES:
            raise ImageGenerationError("参考图片超过 32 MB 下载限制。")
        raw = bytearray()
        for chunk in _chunks(response):
            guard()
            raw.extend(chunk)
            if len(raw) > MAX_IMAGE_BYTES:
                raise ImageGenerationError("参考图片超过 32 MB 下载限制。")
        return bytes(raw)


def supports_reference_input(cfg):
    """Return whether the selected, native adapter has a documented reference field."""
    return (cfg.get('image_provider') == 'tencent-tokenhub'
            and cfg.get('image_model') in ('hy-image-v3', 'hy-image-v3.5-preview'))


def generate_sheet(prompt: str, output: Path, cfg: dict, check_cancel: Callable,
                   progress: Callable = None, job_path: Path = None, reference_image: Path | None = None) -> dict:
    """Generate/resume one sheet with bounded requests and persistent job IDs.

    Callback exceptions propagate unchanged. Ambiguous submissions are never
    silently retried because a timed-out POST may already incur a charge.
    """
    if cfg.get('image_provider') in ('aliyun-bailian', 'tencent-tokenhub'):
        from .china_image import generate_china_sheet
        return generate_china_sheet(prompt, output, cfg, check_cancel, progress, job_path, reference_image)
    if reference_image:
        raise ImageGenerationError('当前文生图平台未配置可验证的参考图输入协议；请切换到腾讯 Hy-Image，或关闭联网图片参考。')
    provider, model, endpoint, payload, key, fingerprint = _configuration(prompt, cfg)
    output = Path(output)
    job_path = Path(job_path) if job_path else output.with_name(output.stem + ".image-job.json")
    job = {}
    if job_path.exists():
        try:
            job = json.loads(job_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raise ImageGenerationError("图片任务记录损坏；为避免重复计费，未提交新的任务。") from None
        if not isinstance(job, dict) or job.get("fingerprint") != fingerprint:
            raise ImageGenerationError("已有图片任务与本次模型、服务或设计不同，请创建新的生成任务。")
        if job.get("state") == "submission_unknown":
            raise ImageGenerationError("上次图片请求的提交结果不明，请先在生成服务控制台核对；为避免重复计费，未再次提交。")
        if job.get("state") in ("cancelled", "failed"):
            raise ImageGenerationError("该图片生成请求已取消或失败，请创建新的生成任务。")
    if job.get("state") == "saved" and output.exists() and output.stat().st_size <= MAX_IMAGE_BYTES:
        check_cancel()
        raw = output.read_bytes()
        if hashlib.sha256(raw).hexdigest() == job.get("sha256"):
            _normalized_image(raw)
            return {**job["result"], "path": str(output), "reused": True}
    deadline = time.monotonic() + DEADLINE_SECONDS
    request_id = job.get("request_id")
    if provider == "fal" and request_id and not re.fullmatch(r"[A-Za-z0-9_-]{1,160}", request_id):
        raise ImageGenerationError("图片服务的任务编号格式不正确。")
    headers = {"Authorization": ("Key " if provider == "fal" else "Bearer ") + key}

    with httpx.Client(timeout=httpx.Timeout(20, connect=10), follow_redirects=False) as client:
        def notify(message):
            if progress:
                progress(message)

        def guard():
            try:
                check_cancel()
            except BaseException:
                if provider == "fal" and request_id:
                    try:
                        cancel_url = _request_url(job.get("cancel_url", ""), endpoint, request_id, "cancel")
                        cancelled = client.put(cancel_url, headers=headers, timeout=5)
                        if cancelled.status_code in (200, 202):
                            job["state"] = "cancelled"
                            _write_job(job_path, job)
                    except Exception:
                        pass  # Best-effort cancellation must preserve the callback exception.
                raise
            if time.monotonic() >= deadline:
                raise ImageGenerationError("图片生成等待超时，任务记录已保留；恢复任务不会自动重复提交已知请求。")

        def timeout(synchronous=False):
            guard()
            return max(0.1, min(240.0 if synchronous else 20.0, deadline - time.monotonic()))

        def request(method, url, synchronous=False, **kwargs):
            guard()
            # Stream even JSON to bound malformed responses and base64 outputs.
            with _stream(client, method, url, headers=headers, timeout=timeout(synchronous), **kwargs) as response:
                raw = bytearray()
                for chunk in _chunks(response):
                    # A completed POST must first be persisted before running
                    # callbacks, otherwise cancellation loses its remote ID.
                    if method != "POST":
                        guard()
                    elif time.monotonic() >= deadline:
                        raise ImageGenerationError("图片请求超时，请核对服务控制台中的请求。")
                    raw.extend(chunk)
                    if len(raw) > MAX_JSON_BYTES:
                        raise ImageGenerationError("图片服务响应超过大小限制，请核对生成服务中的请求。")
                return httpx.Response(response.status_code, content=bytes(raw), headers=response.headers)

        guard()
        if not request_id and not job.get("image_source"):
            notify(f"正在向 {model} 提交三视图生成请求")
            guard()
            job = {"model": model, "provider": provider, "endpoint": endpoint,
                   "fingerprint": fingerprint, "state": "submission_unknown"}
            _write_job(job_path, job)
            response = request("POST", endpoint, synchronous=provider != "fal", json=payload)
            if response.status_code in (400, 401, 402, 403, 404, 405, 413, 415, 422, 429):
                job["state"] = "rejected"
                _write_job(job_path, job)
            submitted = _json_response(response)
            if provider == "fal":
                request_id = submitted.get("request_id")
                if not isinstance(request_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,160}", request_id):
                    raise ImageGenerationError("图片服务没有返回有效任务编号，请先核对服务控制台中的请求。")
                job.update({"request_id": request_id, "state": "pending"})
                _write_job(job_path, job)
                for name, operation in (("status_url", "status"), ("response_url", ""), ("cancel_url", "cancel")):
                    job[name] = _request_url(submitted.get(name, ""), endpoint, request_id, operation)
            else:
                items = submitted.get("data")
                if not isinstance(items, list) or len(items) != 1 or not isinstance(items[0], dict):
                    raise ImageGenerationError("图片服务未返回约定的一张参考图，请核对服务控制台中的请求。")
                item = items[0]
                source = {k: item[k] for k in ("b64_json", "url") if isinstance(item.get(k), str)}
                if not source:
                    raise ImageGenerationError("图片服务没有返回 b64_json 或图片 URL。")
                job.update({"image_source": source, "state": "generated"})
            _write_job(job_path, job)
        else:
            notify(f"正在恢复已有 {model} 图片生成任务")

        if provider == "fal":
            status_url = _request_url(job.get("status_url", ""), endpoint, request_id, "status")
            response_url = _request_url(job.get("response_url", ""), endpoint, request_id, "")
            previous_status = None
            while True:
                data = _json_response(request("GET", status_url))
                status = data.get("status")
                guard()
                if status in ("FAILED", "CANCELLED", "CANCELED") or data.get("error"):
                    job["state"] = "failed"
                    _write_job(job_path, job)
                    raise ImageGenerationError("图片生成失败，请在 fal 控制台查看该请求详情。")
                if status == "COMPLETED":
                    break
                if status not in ("IN_QUEUE", "IN_PROGRESS"):
                    raise ImageGenerationError("图片服务返回了未知任务状态，请稍后恢复。")
                if status != previous_status:
                    notify(f"{model} 正在排队" if status == "IN_QUEUE" else f"{model} 正在生成三视图")
                    previous_status = status
                wait_until = min(deadline, time.monotonic() + POLL_SECONDS)
                while time.monotonic() < wait_until:
                    guard()
                    time.sleep(min(0.25, max(0, wait_until - time.monotonic())))
            result = _json_response(request("GET", response_url))
            guard()
            if result.get("error") or any(result.get("has_nsfw_concepts") or []):
                job["state"] = "failed"
                _write_job(job_path, job)
                raise ImageGenerationError("图片服务未提供可用参考图，请调整描述后创建新任务。")
            images = result.get("images")
            if not isinstance(images, list) or len(images) != 1 or not isinstance(images[0], dict):
                raise ImageGenerationError("图片服务未返回约定的一张三视图参考图。")
            source = {"url": images[0].get("url")}
            job["seed"] = result.get("seed")
        else:
            source = job["image_source"]
        guard()
        notify(f"正在保存并检查 {model} 三视图")
        if source.get("b64_json"):
            raw = _decode_base64(source["b64_json"])
        elif isinstance(source.get("url"), str):
            raw = _download_image(client, source["url"], provider, guard, timeout)
        else:
            raise ImageGenerationError("图片服务没有返回有效图片内容。")
        guard()
        image = _normalized_image(raw)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(output.name + ".tmp")
        image.save(temporary, format="PNG")
        guard()
        temporary.replace(output)
        metadata = {"provider": provider, "model": model, "request_id": request_id,
                    "width": image.width, "height": image.height, "path": str(output),
                    "seed": job.get("seed"), "reused": False}
        job.update({"state": "saved", "result": metadata,
                    "sha256": hashlib.sha256(output.read_bytes()).hexdigest()})
        # Keep the source reference for recovering a deleted/damaged local image.
        _write_job(job_path, job)
        return metadata
