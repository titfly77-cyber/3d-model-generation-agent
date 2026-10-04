"""Tripo official v3 single-image and multiview reconstruction adapter."""
import hashlib
import ipaddress
import json
import re
import socket
import time
from pathlib import Path
from urllib.parse import urlsplit

import httpx


class TripoReconstructionError(RuntimeError):
    pass


def _save(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def _base(value):
    value = str(value or 'https://openapi.tripo3d.com/v3').strip().rstrip('/')
    parsed = urlsplit(value)
    if (parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password
            or parsed.query or parsed.fragment):
        raise TripoReconstructionError('Tripo API 地址必须是有效的 HTTPS 地址。')
    return value


def _json(response, operation):
    if response.status_code >= 400:
        raise TripoReconstructionError(
            f'Tripo {operation}返回 HTTP {response.status_code}，请检查 API Key、模型权限、参数和额度。')
    try:
        body = response.json()
    except ValueError:
        raise TripoReconstructionError(f'Tripo {operation}没有返回有效 JSON。') from None
    if not isinstance(body, dict):
        raise TripoReconstructionError(f'Tripo {operation}响应结构不正确。')
    if body.get('code') != 0:
        code = str(body.get('code', 'unknown'))[:40]
        message = str(body.get('message') or body.get('suggestion') or '请求失败')[:160]
        raise TripoReconstructionError(f'Tripo {operation}失败（{code}）：{message}')
    data = body.get('data')
    if not isinstance(data, dict):
        raise TripoReconstructionError(f'Tripo {operation}缺少 data 对象。')
    return data


def _upload(client, base, headers, path):
    path = Path(path)
    raw = path.read_bytes()
    if len(raw) > 20 * 1024 * 1024:
        raise TripoReconstructionError('Tripo 单张参考图不能超过 20MB。')
    mime = 'image/jpeg' if path.suffix.lower() in ('.jpg', '.jpeg') else 'image/png'
    try:
        response = client.post(base + '/files', headers=headers,
                               files={'file': (path.name, raw, mime)}, timeout=120)
    except httpx.HTTPError:
        raise TripoReconstructionError('Tripo 图片上传连接中断，可恢复任务后继续。') from None
    token = _json(response, '图片上传').get('file_token')
    if not isinstance(token, str) or not token or len(token) > 240:
        raise TripoReconstructionError('Tripo 图片上传未返回有效 file_token。')
    return token


def _safe_download_url(url):
    try:
        parsed = urlsplit(url)
        host = parsed.hostname or ''
        valid = (parsed.scheme == 'https' and bool(host) and not parsed.username and not parsed.password
                 and parsed.port in (None, 443))
        addresses = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM) if valid else []
        valid = valid and bool(addresses) and all(
            ipaddress.ip_address(item[4][0]).is_global for item in addresses)
    except (ValueError, OSError):
        valid = False
    if not valid:
        raise TripoReconstructionError('Tripo 返回了不安全或无法解析的模型下载地址。')
    return url


def _download(client, url, output, guard):
    from .reconstruction import MAX_GLB_BYTES, _valid_glb
    url = _safe_download_url(url)
    temporary = output.with_suffix('.glb.download')
    try:
        with client.stream('GET', url, follow_redirects=False, timeout=120) as response:
            if response.status_code != 200:
                raise TripoReconstructionError(f'Tripo GLB 下载失败（HTTP {response.status_code}）。')
            size = 0
            with temporary.open('wb') as file:
                for chunk in response.iter_bytes(65536):
                    guard()
                    size += len(chunk)
                    if size > MAX_GLB_BYTES:
                        raise TripoReconstructionError('Tripo 模型超过 256MB 下载限制。')
                    file.write(chunk)
        _valid_glb(temporary)
        temporary.replace(output)
    except httpx.HTTPError:
        raise TripoReconstructionError('Tripo 模型下载连接中断，恢复任务会重新下载同一结果。') from None
    finally:
        temporary.unlink(missing_ok=True)


def reconstruct_tripo_views(views, output, cfg, check_cancel, progress=None, job_path=None):
    if set(views) not in ({'front'}, {'front', 'left', 'back'}):
        raise TripoReconstructionError('Tripo 需要一张正面主图，或完整的正面、左侧、背面三张图。')
    used = ['front'] if len(views) == 1 else ['front', 'left', 'back']
    key = str(cfg.get('generative_api_key') or '').strip()
    model = str(cfg.get('reconstruction_model') or 'v3.1-20260211').strip()
    if not key or not model:
        raise TripoReconstructionError('请填写 Tripo 官方 API Key 和模型版本。')
    base = _base(cfg.get('reconstruction_base_url'))
    textured = bool(cfg.get('reconstruction_textured', True))
    parameters = cfg.get('reconstruction_parameters') or {}
    if not isinstance(parameters, dict) or any(not isinstance(k, str) for k in parameters):
        raise TripoReconstructionError('3D 额外参数必须是 JSON 对象。')
    protected = {'model', 'input', 'inputs', 'texture', 'pbr', 'authorization', 'headers', 'api_key'}
    if any(key_name.lower() in protected for key_name in parameters):
        raise TripoReconstructionError('3D 额外参数不能覆盖模型、图片、纹理或鉴权字段。')
    effective_parameters = dict(parameters)
    retry_seed = cfg.get('reconstruction_seed')
    if ('model_seed' not in effective_parameters and isinstance(retry_seed, int)
            and not isinstance(retry_seed, bool) and 0 <= retry_seed <= 2**32 - 1):
        effective_parameters['model_seed'] = retry_seed
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    job_path = Path(job_path) if job_path else output.with_suffix('.job.json')
    hashes = {name: hashlib.sha256(Path(views[name]).read_bytes()).hexdigest() for name in used}
    fingerprint_data = {'provider': 'tripo-official', 'base': base, 'model': model,
                        'views': hashes, 'textured': textured, 'parameters': effective_parameters}
    fingerprint = hashlib.sha256(json.dumps(
        fingerprint_data, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    job = {}
    if job_path.exists():
        try:
            job = json.loads(job_path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            raise TripoReconstructionError('Tripo 作业记录损坏；为避免重复计费，未重新提交。') from None
        if not isinstance(job, dict) or job.get('fingerprint') != fingerprint:
            raise TripoReconstructionError('参考图、Tripo 模型或参数已变化，请创建新任务。')
        if job.get('state') == 'submission_unknown':
            raise TripoReconstructionError('上次 Tripo 提交结果不明，请先在控制台核对，避免重复计费。')
        if job.get('state') == 'rejected':
            job = {}
        elif job.get('state') in ('failed', 'cancelled'):
            raise TripoReconstructionError('该 Tripo 作业已失败或取消，请创建新的设计任务。')
        if output.exists() and job.get('metadata'):
            from .reconstruction import _valid_glb
            _valid_glb(output)
            return job['metadata']
    deadline = time.monotonic() + 3600
    headers = {'Authorization': 'Bearer ' + key}
    with httpx.Client(timeout=httpx.Timeout(120, connect=20), follow_redirects=False) as client:
        def guard():
            check_cancel()
            if time.monotonic() >= deadline:
                raise TripoReconstructionError('Tripo 生成等待超时，恢复后会继续查询原任务。')

        if not job:
            job = {'fingerprint': fingerprint, 'provider': 'tripo-official', 'model': model,
                   'used_views': used, 'state': 'preparing', 'uploads': {}}
            _save(job_path, job)
        if job.get('state') == 'preparing':
            uploads = job.get('uploads') if isinstance(job.get('uploads'), dict) else {}
            for view in used:
                if view not in uploads:
                    guard()
                    if progress:
                        progress(f'正在向 Tripo 上传{view}参考图')
                    uploads[view] = _upload(client, base, headers, views[view])
                    job['uploads'] = uploads
                    _save(job_path, job)
            common = {'model': model, 'texture': textured, 'pbr': textured, **effective_parameters}
            if len(used) == 1:
                endpoint = base + '/generation/image-to-model'
                payload = {**common, 'input': uploads['front']}
            else:
                endpoint = base + '/generation/multiview-to-model'
                payload = {**common, 'inputs': [{view: uploads[view]} for view in used]}
            job['state'] = 'submission_unknown'
            _save(job_path, job)
            guard()
            try:
                response = client.post(endpoint, headers=headers, json=payload)
            except httpx.HTTPError:
                raise TripoReconstructionError(
                    'Tripo 任务提交连接中断，请先在控制台确认是否已创建任务，避免重复计费。') from None
            if response.status_code in (400, 401, 402, 403, 404, 413, 422, 429):
                job['state'] = 'rejected'
                _save(job_path, job)
            data = _json(response, '任务提交')
            task_id = data.get('task_id')
            if not isinstance(task_id, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,240}', task_id):
                raise TripoReconstructionError('Tripo 没有返回有效 task_id，请在控制台核对请求。')
            job.update(state='submitted', task_id=task_id)
            _save(job_path, job)
            if progress:
                progress('Tripo 3D 作业已提交')
        previous = None
        while True:
            guard()
            try:
                data = _json(client.get(base + '/tasks/' + job['task_id'], headers=headers), '任务查询')
            except httpx.HTTPError:
                raise TripoReconstructionError('Tripo 作业查询连接中断，恢复后会继续查询同一任务。') from None
            status = data.get('status')
            if status == 'success':
                break
            if status in ('failed', 'cancelled'):
                job['state'] = status
                _save(job_path, job)
                message = str(data.get('error_message') or '云端任务失败')[:160]
                raise TripoReconstructionError('Tripo 3D 作业失败或取消：' + message)
            if status not in ('queued', 'running'):
                raise TripoReconstructionError('Tripo 返回了未知作业状态。')
            if progress and status != previous:
                progress('Tripo 作业正在排队' if status == 'queued' else 'Tripo 正在生成几何与纹理')
            previous = status
            time.sleep(5)
        result = data.get('output') or {}
        url = result.get('model_url') if isinstance(result, dict) else None
        if not isinstance(url, str) or not url:
            raise TripoReconstructionError('Tripo 成功任务未返回 model_url。')
        if progress:
            progress('正在下载并验证 Tripo GLB 模型')
        _download(client, url, output, guard)
        metadata = {'provider': 'tripo-official', 'model': model, 'request_id': job['task_id'],
                    'used_views': used, 'textured': textured, 'url': url, 'format': 'glb',
                    'file_bytes': output.stat().st_size, 'credits_consumed': data.get('credits_consumed')}
        job.update(state='completed', metadata=metadata)
        _save(job_path, job)
        return metadata
