"""Native single/multi-view Hunyuan reconstruction through fal's queue API.

Models and field names are documented at:
https://fal.ai/models/fal-ai/hunyuan3d/v2/api
https://fal.ai/models/fal-ai/hunyuan3d/v2/multi-view/api

The queue job is persisted before polling. Retrying a poll or download never
submits another paid generation. A three-view input is never reduced to one.
"""

import base64
import hashlib
import json
import re
import struct
import time
from pathlib import Path
from typing import Callable
from urllib.parse import urljoin, urlparse

import httpx
from PIL import Image, UnidentifiedImageError


SINGLE_MODEL = 'fal-ai/hunyuan3d/v2'
MULTI_MODEL = 'fal-ai/hunyuan3d/v2/multi-view'
QUEUE_ORIGIN = 'https://queue.fal.run'
MAX_IMAGE_BYTES = 15 * 1024 * 1024
MAX_GLB_BYTES = 256 * 1024 * 1024
POLL_INTERVAL = 3.0
DEFAULT_TIMEOUT = 1800.0


class ReconstructionError(RuntimeError):
    """A user-readable reconstruction failure without upstream secret data."""

    def __init__(self, message, http_status=None):
        super().__init__(message)
        self.http_status = http_status


def _save(path: Path, value: dict):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def _queue_url(value: str) -> str:
    try:
        parsed = urlparse(value)
        valid = (parsed.scheme == 'https' and parsed.hostname == 'queue.fal.run'
                 and parsed.port in (None, 443) and not parsed.username
                 and not parsed.password and not parsed.fragment)
    except (TypeError, ValueError):
        valid = False
    if not valid:
        raise ReconstructionError('3D 服务返回了无效的队列地址，已停止请求。')
    return value


def _media_url(value: str) -> str:
    try:
        parsed = urlparse(value)
        host = parsed.hostname or ''
        valid = (parsed.scheme == 'https' and parsed.port in (None, 443)
                 and not parsed.username and not parsed.password
                 and (host == 'fal.media' or host.endswith('.fal.media')
                      or host == 'storage.googleapis.com'))
    except (TypeError, ValueError):
        valid = False
    if not valid:
        raise ReconstructionError('3D 服务返回了不受支持的模型下载地址。')
    return value


def _json_request(client, method, url, headers, **kwargs):
    try:
        response = client.request(method, _queue_url(url), headers=headers, **kwargs)
    except httpx.HTTPError:
        raise ReconstructionError('无法连接 3D 云端服务；已保存的任务可重试继续查询。') from None
    if response.status_code >= 400:
        messages = {
            401: 'fal API Key 无效，请检查生成服务密钥。',
            403: 'fal API Key 没有该 3D 模型的访问权限。',
            404: '云端 3D 作业不存在或已过期，请创建新任务。',
            422: '3D 服务不接受当前输入，请检查参考图和生成参数。',
            429: '3D 服务请求受限，请稍后继续任务。',
        }
        raise ReconstructionError(messages.get(response.status_code,
            '3D 服务请求失败，请稍后继续任务。') + f'（HTTP {response.status_code}）', response.status_code)
    if response.status_code >= 300:
        raise ReconstructionError('3D 队列返回了未预期的跳转，已停止请求。')
    try:
        body = response.json()
    except (ValueError, UnicodeError):
        raise ReconstructionError('3D 服务未返回有效的 JSON 响应。') from None
    if not isinstance(body, dict):
        raise ReconstructionError('3D 服务响应结构不正确。')
    return body


def _image_uri(path: Path) -> str:
    path = Path(path)
    if not path.is_file() or path.stat().st_size > MAX_IMAGE_BYTES:
        raise ReconstructionError('重建参考图不存在或超过 15MB。')
    try:
        with Image.open(path) as picture:
            if picture.width * picture.height > 25_000_000:
                raise ReconstructionError('重建参考图超过 2500 万像素。')
            mime = Image.MIME.get(picture.format)
            if mime not in ('image/png', 'image/jpeg', 'image/webp'):
                raise ReconstructionError('重建参考图须为 PNG、JPG 或 WebP。')
            picture.verify()
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        raise ReconstructionError('重建参考图损坏或无法读取。') from None
    return f'data:{mime};base64,' + base64.b64encode(path.read_bytes()).decode('ascii')


def _valid_glb(path: Path):
    size = path.stat().st_size
    with path.open('rb') as file:
        header = file.read(12)
    if size < 20 or size > MAX_GLB_BYTES or len(header) != 12:
        raise ReconstructionError('3D 服务未返回有效大小的 GLB 文件。')
    magic, version, length = struct.unpack('<4sII', header)
    if magic != b'glTF' or version != 2 or length != size:
        raise ReconstructionError('3D 服务返回的文件不是完整的 GLB 2.0 模型。')


def _download(client, url: str, output: Path, check: Callable):
    """Stream untrusted media without API credentials and validate each redirect."""
    temporary = output.with_suffix(output.suffix + '.download')
    try:
        for _ in range(4):
            check()
            with client.stream('GET', _media_url(url)) as response:
                if response.status_code in (301, 302, 303, 307, 308):
                    url = urljoin(url, response.headers.get('location', ''))
                    continue
                if response.status_code != 200:
                    raise ReconstructionError(f'模型下载失败（HTTP {response.status_code}），可继续原任务重试。')
                content_length = response.headers.get('content-length')
                if content_length and content_length.isdigit() and int(content_length) > MAX_GLB_BYTES:
                    raise ReconstructionError('云端模型文件超过 256MB 下载限制。')
                count = 0
                with temporary.open('wb') as file:
                    for chunk in response.iter_bytes(64 * 1024):
                        check()
                        count += len(chunk)
                        if count > MAX_GLB_BYTES:
                            raise ReconstructionError('云端模型文件超过 256MB 下载限制。')
                        file.write(chunk)
                _valid_glb(temporary)
                temporary.replace(output)
                return
        raise ReconstructionError('模型下载跳转次数过多。')
    except httpx.HTTPError:
        raise ReconstructionError('模型下载连接中断，可继续原任务重新下载。') from None
    finally:
        temporary.unlink(missing_ok=True)


def reconstruct_views(
    views: dict[str, Path], output: Path, cfg: dict, check_cancel: Callable,
    progress: Callable | None = None, job_path: Path | None = None,
) -> dict:
    """Write a GLB from exactly one front view or front/left/back views.

    ``check_cancel`` is a no-argument callback; its exception is preserved.
    ``progress`` receives a Chinese message. Credentials are used only for the
    fixed fal queue origin, never for media downloads or persisted in the job.
    ``textured`` in metadata describes the request, not a material-quality claim.
    """
    check_cancel()
    if cfg.get('reconstruction_provider') in ('aliyun-bailian', 'tencent-tokenhub'):
        from .china_reconstruction import reconstruct_china_views
        return reconstruct_china_views(views, output, cfg, check_cancel, progress, job_path)
    if cfg.get('reconstruction_provider') == 'tripo-official':
        from .tripo_reconstruction import reconstruct_tripo_views
        return reconstruct_tripo_views(views, output, cfg, check_cancel, progress, job_path)
    if cfg.get('reconstruction_provider', 'fal-hunyuan') != 'fal-hunyuan':
        raise ReconstructionError('当前图生 3D 接口仅支持 fal-hunyuan。')
    if set(views) == {'front'}:
        model, used_views = SINGLE_MODEL, ['front']
        payload = {'input_image_url': _image_uri(views['front'])}
    elif set(views) == {'front', 'left', 'back'}:
        model, used_views = MULTI_MODEL, ['front', 'left', 'back']
        payload = {f'{view}_image_url': _image_uri(views[view]) for view in used_views}
    else:
        raise ReconstructionError('图生 3D 需要一张主参考图，或完整的正面、左侧、背面三张图；不会自动丢弃视角。')
    key = cfg.get('generative_api_key', '').strip()
    if not key:
        raise ReconstructionError('请在 API 设置中填写 fal 生成服务 API Key。')
    textured = cfg.get('reconstruction_textured', True)
    if not isinstance(textured, bool):
        raise ReconstructionError('纹理选项必须为布尔值。')
    seed = cfg.get('reconstruction_seed', 42)
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed <= 2**32 - 1:
        raise ReconstructionError('重建随机种子须为 0 到 4294967295 的整数。')
    payload.update(seed=seed, num_inference_steps=50, guidance_scale=7.5,
                   octree_resolution=256, textured_mesh=textured)
    fingerprint = hashlib.sha256(json.dumps({'model': model, 'input': payload},
        sort_keys=True).encode()).hexdigest()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    job_path = Path(job_path) if job_path else output.with_suffix('.job.json')
    job_path.parent.mkdir(parents=True, exist_ok=True)
    job = {}
    if job_path.exists():
        try:
            job = json.loads(job_path.read_text(encoding='utf-8'))
        except (ValueError, OSError):
            raise ReconstructionError('已保存的云端作业记录无法读取，不能安全地自动重新提交。') from None
        if not isinstance(job, dict) or job.get('fingerprint') != fingerprint:
            raise ReconstructionError('参考图或重建设置已变化，请创建新任务，避免复用错误的云端结果。')
        if job.get('state') == 'submission_unknown':
            raise ReconstructionError('上次提交连接中断，尚无法确认云端是否已接收。请在 fal 控制台核对后创建新任务，避免重复计费。')
        if job.get('state') == 'rejected':
            job = {}
        elif output.is_file() and job.get('metadata'):
            _valid_glb(output)
            return job['metadata']
        if job:
            request_id = job.get('request_id')
            if not isinstance(request_id, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,160}', request_id):
                raise ReconstructionError('已保存的云端作业编号无效。')
            for field in ('status_url', 'response_url', 'cancel_url'):
                _queue_url(job.get(field))

    try:
        timeout = min(3600., max(1., float(cfg.get('reconstruction_timeout_seconds', DEFAULT_TIMEOUT))))
    except (TypeError, ValueError):
        raise ReconstructionError('3D 生成超时设置无效。') from None
    deadline = time.monotonic() + timeout
    headers = {'Authorization': 'Key ' + key}

    with httpx.Client(timeout=httpx.Timeout(60, connect=20), follow_redirects=False) as client:
        def check():
            try:
                check_cancel()
            except Exception:
                if job.get('cancel_url'):
                    try:
                        response = client.put(_queue_url(job['cancel_url']), headers=headers, timeout=10)
                        job['cancel_requested'] = True
                        job['cancel_http_status'] = response.status_code
                        _save(job_path, job)
                    except Exception:
                        pass
                raise
            if time.monotonic() >= deadline:
                raise ReconstructionError('3D 生成等待超时；云端作业已保存，继续任务会查询原作业，不会重复生成。')

        if not job:
            check()
            # A POST interrupted after send may already incur a charge. Persist
            # ambiguity before submitting; do not automatically POST it again.
            job = {'fingerprint': fingerprint, 'model': model,
                   'used_views': used_views, 'state': 'submission_unknown'}
            _save(job_path, job)
            try:
                response = _json_request(client, 'POST', QUEUE_ORIGIN + '/' + model, headers, json=payload)
            except ReconstructionError as exc:
                # An explicit rejection cannot have accepted a generation;
                # allow retries after fixing credentials or rate limits.
                if exc.http_status in (400, 401, 402, 403, 404, 405, 413, 415, 422, 429):
                    job['state'] = 'rejected'
                    _save(job_path, job)
                raise
            request_id = response.get('request_id')
            if not isinstance(request_id, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,160}', request_id):
                raise ReconstructionError('3D 服务未返回有效作业编号，请在 fal 控制台核对提交结果。')
            # The provider may use a queue app path different from its model
            # endpoint path. Follow returned URLs, never reconstruct these paths.
            for field in ('status_url', 'response_url', 'cancel_url'):
                job[field] = _queue_url(response.get(field))
            job.update(request_id=request_id, state='submitted')
            _save(job_path, job)
            if progress:
                progress('已提交 Hunyuan 三视图联合重建' if len(used_views) == 3 else '已提交 Hunyuan 单图重建')
        elif progress:
            progress('正在继续查询已保存的 Hunyuan 云端作业')

        previous_status = None
        while True:
            check()
            result = _json_request(client, 'GET', job['status_url'], headers)
            status = result.get('status')
            if result.get('error') or status in ('FAILED', 'ERROR', 'CANCELLED', 'CANCELED'):
                job['state'] = 'failed'
                _save(job_path, job)
                raise ReconstructionError('云端 3D 作业失败或已取消；可在 fal 控制台查看该作业后创建新任务。')
            if status == 'COMPLETED':
                break
            if status not in ('IN_QUEUE', 'IN_PROGRESS'):
                raise ReconstructionError('3D 服务返回了未知的作业状态。')
            if progress and status != previous_status:
                progress('3D 云端作业正在排队' if status == 'IN_QUEUE' else 'Hunyuan 正在生成几何与纹理' if textured else 'Hunyuan 正在生成几何')
            previous_status = status
            pause_until = min(deadline, time.monotonic() + POLL_INTERVAL)
            while time.monotonic() < pause_until:
                check()
                time.sleep(min(.25, max(0., pause_until - time.monotonic())))

        check()
        body = _json_request(client, 'GET', job['response_url'], headers)
        if body.get('error'):
            raise ReconstructionError('3D 服务报告生成失败，请在 fal 控制台查看作业详情。')
        mesh = body.get('model_mesh')
        if not isinstance(mesh, dict):
            raise ReconstructionError('3D 服务没有返回模型文件。')
        if isinstance(mesh.get('file_size'), (int, float)) and mesh['file_size'] > MAX_GLB_BYTES:
            raise ReconstructionError('云端模型文件超过 256MB 下载限制。')
        url = _media_url(mesh.get('url'))
        if progress:
            progress('正在下载并检查 GLB 模型')
        _download(client, url, output, check)
        check()
        metadata = {'provider': 'fal-hunyuan', 'model': model, 'request_id': job['request_id'],
                    'used_views': used_views, 'textured': textured, 'url': url,
                    'format': 'glb', 'file_bytes': output.stat().st_size,
                    'seed': body.get('seed', seed)}
        job.update(state='completed', metadata=metadata)
        _save(job_path, job)
        return metadata
