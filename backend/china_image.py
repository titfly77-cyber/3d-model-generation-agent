"""Mainland-native text-to-image adapters for Alibaba Model Studio and Tencent TokenHub."""
import hashlib
import base64
import json
import time
from pathlib import Path
from urllib.parse import urlsplit

import httpx


class ChinaImageError(RuntimeError):
    pass


def _save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def _base(value, default):
    value = (value or default).strip().rstrip('/')
    parsed = urlsplit(value)
    if (
        parsed.scheme != 'https'
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ChinaImageError('中国云平台接口地址必须是有效的 HTTPS 地址。')
    return value


def _tokenhub_base(value):
    """
    兼容以下两种配置：
    1) https://tokenhub.tencentmaas.com
    2) https://tokenhub.tencentmaas.com/v1

    最终统一返回：
       https://tokenhub.tencentmaas.com/v1
    """
    base = _base(value, 'https://tokenhub.tencentmaas.com/v1')
    if base.endswith('/v1'):
        return base
    return base + '/v1'


def _safe_json(response, service):
    if response.status_code >= 400:
        suffix = (
            '请检查 API Key、模型权限、地域和余额。'
            if response.status_code in (400, 401, 403, 404, 429)
            else '请稍后恢复任务。'
        )
        raise ChinaImageError(f'{service}文生图返回 HTTP {response.status_code}，{suffix}')
    try:
        body = response.json()
    except ValueError:
        raise ChinaImageError(f'{service}文生图没有返回有效 JSON。') from None
    if not isinstance(body, dict):
        raise ChinaImageError(f'{service}文生图响应结构不正确。')
    if body.get('code'):
        raise ChinaImageError(
            f'{service}文生图拒绝了请求（{str(body["code"])[:80]}）。请检查模型权限、参数和额度。'
        )
    return body


def generate_china_sheet(
    prompt: str,
    output: Path,
    cfg: dict,
    check_cancel,
    progress=None,
    job_path: Path | None = None,
    reference_image: Path | None = None,
):
    from .image_generation import _download_image, _normalized_image, MAX_IMAGE_BYTES

    provider = cfg.get('image_provider')
    if provider not in ('aliyun-bailian', 'tencent-tokenhub'):
        raise ChinaImageError('不支持的中国云文生图平台。')

    model = str(
        cfg.get('image_model') or (
            'wan2.2-t2i-plus' if provider == 'aliyun-bailian' else 'hy-image-v3'
        )
    ).strip()
    key = str(cfg.get('image_api_key') or cfg.get('generative_api_key') or '').strip()

    if not model or not key:
        raise ChinaImageError('请填写文生图模型 ID 和对应平台 API Key。')

    parameters = cfg.get('image_parameters') or {}
    if not isinstance(parameters, dict):
        raise ChinaImageError('文生图额外参数必须是 JSON 对象。')

    if any(
        str(k).lower() in (
            'model', 'prompt', 'messages', 'authorization',
            'headers', 'api_key', 'n', 'num_images'
        )
        for k in parameters
    ):
        raise ChinaImageError('额外参数不能覆盖模型、提示词、数量或鉴权字段。')

    if provider == 'aliyun-bailian':
        if reference_image:
            raise ChinaImageError('当前百炼文生图模型不支持本应用的参考图协议；请改用支持图生图的模型与适配器，或关闭联网图片参考。')
        base = _base(cfg.get('image_base_url'), 'https://dashscope.aliyuncs.com/api/v1')
        endpoint = base + '/services/aigc/text2image/image-synthesis'
        defaults = {
            'size': '1440*800',
            'n': 1,
            'prompt_extend': False,
            'watermark': False,
        }
        payload = {
            'model': model,
            'input': {'prompt': prompt},
            'parameters': {**defaults, **parameters},
        }
        headers = {
            'Authorization': 'Bearer ' + key,
            'X-DashScope-Async': 'enable',
        }
        synchronous = False

    else:
        # 这里修复 TokenHub 的 /v1 重复拼接问题
        base = _tokenhub_base(cfg.get('image_base_url'))

        reference_data = None
        if reference_image:
            raw = Path(reference_image).read_bytes()
            if not raw or len(raw) > (20 if model == 'hy-image-v3.5-preview' else 10) * 1024 * 1024:
                raise ChinaImageError('联网参考图为空或超过当前腾讯模型的输入限制。')
            reference_data = base64.b64encode(raw).decode()
        if model == 'hy-image-v3':
            endpoint = base + '/wand/hunyuan-image/v3-generation'
            payload = {
                'model': model,
                'prompt': prompt,
                'size': '1280x720',
                **parameters,
            }
            if reference_data:
                payload['images'] = [reference_data]

        elif model == 'hy-image-v3.5-preview':
            endpoint = base + '/wand/hunyuan-image/v35-generation'
            content = [{'type': 'text', 'text': prompt}]
            if reference_data:
                content.append({'type': 'image_url', 'image_url': {
                    'url': 'data:image/png;base64,' + reference_data}})
            payload = {
                'model': model,
                'size': '1536x864',
                'messages': [
                    {
                        'role': 'user',
                        'content': content,
                    }
                ],
                **parameters,
            }

        else:
            raise ChinaImageError(
                '腾讯原生适配器当前支持 hy-image-v3 和 hy-image-v3.5-preview；'
                '其他模型请使用对应兼容接口。'
            )

        headers = {'Authorization': 'Bearer ' + key}
        synchronous = True

    fingerprint_payload = json.loads(json.dumps(payload))
    if model == 'hy-image-v3':
        fingerprint_payload.pop('images', None)
    elif model == 'hy-image-v3.5-preview':
        for message in fingerprint_payload.get('messages', []):
            message['content'] = [item for item in message.get('content', []) if item.get('type') != 'image_url']
    reference_sha256 = hashlib.sha256(Path(reference_image).read_bytes()).hexdigest() if reference_image else None
    fingerprint = hashlib.sha256(
        json.dumps(
            {'provider': provider, 'endpoint': endpoint, 'payload': fingerprint_payload,
             'reference_sha256': reference_sha256},
            sort_keys=True,
            ensure_ascii=False,
        ).encode()
    ).hexdigest()

    output = Path(output)
    job_path = Path(job_path) if job_path else output.with_suffix('.image-job.json')
    job = {}

    if job_path.exists():
        try:
            job = json.loads(job_path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            raise ChinaImageError('文生图作业记录损坏；为避免重复计费，未重新提交。') from None

        if job.get('fingerprint') != fingerprint:
            raise ChinaImageError('文生图模型或参数已变化，请创建新任务。')

        if job.get('state') == 'submission_unknown':
            raise ChinaImageError('上次提交结果不明，请先在云平台控制台核对，避免重复计费。')

        if job.get('state') == 'rejected':
            # The provider confirmed that no task was accepted, so corrected
            # credentials or parameters may safely reuse this local task.
            job = {}
        elif job.get('state') in ('failed', 'cancelled'):
            raise ChinaImageError('该云端文生图任务已失败或取消，请创建新的设计任务。')

        if (
            job.get('state') == 'saved'
            and output.exists()
            and hashlib.sha256(output.read_bytes()).hexdigest() == job.get('sha256')
        ):
            _normalized_image(output.read_bytes())
            return {**job['result'], 'path': str(output), 'reused': True}

    deadline = time.monotonic() + 900

    with httpx.Client(timeout=httpx.Timeout(240, connect=20), follow_redirects=False) as client:
        def guard():
            check_cancel()
            if time.monotonic() >= deadline:
                raise ChinaImageError('文生图等待超时；作业记录已保留。')

        if not job:
            guard()
            if progress:
                progress(f'正在向 {model} 提交三视图生成请求')

            job = {
                'provider': provider,
                'model': model,
                'endpoint': endpoint,
                'fingerprint': fingerprint,
                'state': 'submission_unknown',
            }
            _save(job_path, job)

            try:
                response = client.post(endpoint, headers=headers, json=payload)
            except httpx.HTTPError:
                raise ChinaImageError('文生图提交连接中断，不能确认是否已计费；请先检查平台控制台。') from None

            if response.status_code in (400, 401, 402, 403, 404, 413, 422, 429):
                job['state'] = 'rejected'
                _save(job_path, job)

            body = _safe_json(response, '阿里云百炼' if provider == 'aliyun-bailian' else '腾讯云 TokenHub')

            if synchronous:
                job.update(state='generated', result_body=body)
            else:
                task_id = (body.get('output') or {}).get('task_id')
                if not isinstance(task_id, str) or not task_id:
                    raise ChinaImageError('阿里云百炼没有返回有效 task_id，请在控制台核对请求。')
                job.update(state='submitted', task_id=task_id)

            _save(job_path, job)

        if not synchronous and job.get('state') != 'generated':
            query = base + '/tasks/' + job['task_id']
            previous = None
            while True:
                guard()
                try:
                    body = _safe_json(
                        client.get(query, headers={'Authorization': 'Bearer ' + key}),
                        '阿里云百炼'
                    )
                except httpx.HTTPError:
                    raise ChinaImageError('阿里云任务查询连接中断，恢复后会继续查询原任务。') from None

                status = (body.get('output') or {}).get('task_status')
                if status == 'SUCCEEDED':
                    job.update(state='generated', result_body=body)
                    _save(job_path, job)
                    break

                if status in ('FAILED', 'CANCELED', 'UNKNOWN'):
                    job['state'] = 'failed'
                    _save(job_path, job)
                    raise ChinaImageError('阿里云文生图任务失败、取消或已过期。')

                if status not in ('PENDING', 'RUNNING'):
                    raise ChinaImageError('阿里云返回了未知的文生图任务状态。')

                if progress and status != previous:
                    progress('阿里云文生图正在排队' if status == 'PENDING' else '阿里云正在生成三视图')

                previous = status
                time.sleep(3)

        body = job.get('result_body') or {}

        if provider == 'aliyun-bailian':
            results = (body.get('output') or {}).get('results') or []
            item = results[0] if results and isinstance(results[0], dict) else {}
        else:
            if model == 'hy-image-v3.5-preview':
                choices = body.get('choices') or []
                delta = (choices[0].get('delta') or {}) if choices and isinstance(choices[0], dict) else {}
                item = delta.get('image') if isinstance(delta.get('image'), dict) else {}
            else:
                data = body.get('data') or []
                item = data[0] if data and isinstance(data[0], dict) else {}

        source = item.get('url')
        raw = None

        if item.get('b64_json'):
            try:
                raw = base64.b64decode(item['b64_json'], validate=True)
            except Exception:
                raise ChinaImageError('图片服务返回的 Base64 图片无效。') from None
        elif source:
            raw = _download_image(client, source, 'openai-compatible', guard, lambda: 60)

        if not raw:
            raise ChinaImageError('图片服务没有返回可下载的图片。')

        image = _normalized_image(raw)
        output.parent.mkdir(parents=True, exist_ok=True)
        image.save(output, format='PNG')
        saved = output.read_bytes()

        if len(saved) > MAX_IMAGE_BYTES:
            raise ChinaImageError('标准化后的参考图超过大小限制。')

        result = {
            'provider': provider,
            'model': model,
            'request_id': job.get('task_id') or body.get('request_id'),
            'width': image.width,
            'height': image.height,
            'seed': item.get('seed'),
            'reused': False,
        }

        job.update(
            state='saved',
            result=result,
            sha256=hashlib.sha256(saved).hexdigest(),
        )
        job.pop('result_body', None)
        _save(job_path, job)

        return {**result, 'path': str(output)}
