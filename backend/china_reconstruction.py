"""Mainland image-to-3D adapters for Tencent TokenHub and Alibaba Model Studio."""
import base64
import hashlib
import ipaddress
import json
import re
import socket
import time
from pathlib import Path
from urllib.parse import urlsplit

import httpx


class ChinaReconstructionError(RuntimeError): pass


def _save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def _base(value, default):
    value = (value or default).strip().rstrip('/')
    parsed = urlsplit(value)
    if parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ChinaReconstructionError('3D 服务地址必须是有效的 HTTPS 地址。')
    return value


def _json(response, service):
    if response.status_code >= 400:
        if response.status_code == 404 and service == '阿里云百炼':
            raise ChinaReconstructionError(
                '阿里云百炼 Tripo 接口返回 HTTP 404：3D 地址或工作空间不存在。'
                '请填写 https://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/api/v1，'
                '不要使用 /compatible-mode/v1。')
        raise ChinaReconstructionError(f'{service} 3D 接口返回 HTTP {response.status_code}；请检查 Key、模型权限、地域和余额。')
    try: body = response.json()
    except ValueError: raise ChinaReconstructionError(f'{service} 3D 接口未返回有效 JSON。') from None
    if not isinstance(body, dict): raise ChinaReconstructionError(f'{service} 3D 响应结构不正确。')
    if body.get('code'):
        raise ChinaReconstructionError(f'{service} 3D 拒绝了请求（{str(body["code"])[:80]}）。请检查模型权限、参数和额度。')
    return body


def _data_uri(path):
    raw = Path(path).read_bytes()
    if len(raw) > 8 * 1024 * 1024: raise ChinaReconstructionError('单张 3D 参考图不能超过 8MB。')
    return 'data:image/png;base64,' + base64.b64encode(raw).decode()


def _safe_download(url):
    try:
        parsed = urlsplit(url); host = parsed.hostname or ''
        valid = parsed.scheme == 'https' and host and not parsed.username and not parsed.password and parsed.port in (None,443)
        addresses = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM) if valid else []
        valid = valid and addresses and all(ipaddress.ip_address(x[4][0]).is_global for x in addresses)
    except (ValueError, OSError): valid = False
    if not valid: raise ChinaReconstructionError('3D 服务返回了不安全或无法解析的模型下载地址。')
    return url


def _download(client, url, output, guard):
    from .reconstruction import MAX_GLB_BYTES, _valid_glb
    url = _safe_download(url); temp = output.with_suffix('.glb.download')
    try:
        with client.stream('GET', url, follow_redirects=False, timeout=120) as response:
            if response.status_code != 200: raise ChinaReconstructionError(f'GLB 下载失败（HTTP {response.status_code}）。')
            size = 0
            with temp.open('wb') as file:
                for chunk in response.iter_bytes(65536):
                    guard(); size += len(chunk)
                    if size > MAX_GLB_BYTES: raise ChinaReconstructionError('云端模型超过 256MB 限制。')
                    file.write(chunk)
        _valid_glb(temp); temp.replace(output)
    except httpx.HTTPError:
        raise ChinaReconstructionError('模型下载连接中断，恢复任务会重新下载同一结果。') from None
    finally: temp.unlink(missing_ok=True)


def _aliyun_upload(client, key, model, path):
    policy_url = 'https://dashscope.aliyuncs.com/api/v1/uploads'
    try:
        policy = _json(client.get(policy_url, headers={'Authorization':'Bearer '+key},
                                  params={'action':'getPolicy','model':model}), '阿里云百炼').get('data') or {}
    except httpx.HTTPError:
        raise ChinaReconstructionError('无法获取阿里云临时图片上传凭证。') from None
    required = ('upload_dir','upload_host','oss_access_key_id','signature','policy','x_oss_object_acl','x_oss_forbid_overwrite')
    if any(not isinstance(policy.get(k), str) for k in required): raise ChinaReconstructionError('阿里云上传凭证结构不正确。')
    host = _base(policy['upload_host'], '')
    name = hashlib.sha256(Path(path).read_bytes()).hexdigest()[:20] + '.png'
    object_key = policy['upload_dir'].rstrip('/') + '/' + name
    fields = {'OSSAccessKeyId':(None,policy['oss_access_key_id']), 'Signature':(None,policy['signature']),
              'policy':(None,policy['policy']), 'x-oss-object-acl':(None,policy['x_oss_object_acl']),
              'x-oss-forbid-overwrite':(None,policy['x_oss_forbid_overwrite']), 'key':(None,object_key),
              'success_action_status':(None,'200'), 'file':(name,Path(path).read_bytes(),'image/png')}
    try: response = client.post(host, files=fields, timeout=120)
    except httpx.HTTPError: raise ChinaReconstructionError('阿里云临时图片上传连接中断。') from None
    if response.status_code != 200: raise ChinaReconstructionError(f'阿里云临时图片上传失败（HTTP {response.status_code}）。')
    return 'oss://' + object_key


def _aliyun_parameters(model, textured, configured):
    """Build only the parameters accepted by the current Model Studio Tripo API."""
    configured = configured or {}
    if not isinstance(configured, dict):
        raise ChinaReconstructionError('阿里云 Tripo 的额外参数必须是 JSON 对象。')
    supported = {'texture_quality', 'geometry_quality'}
    unknown = sorted(set(configured) - supported)
    if unknown:
        raise ChinaReconstructionError('阿里云 Tripo 仅支持 texture_quality 和 geometry_quality；不支持：' + '、'.join(unknown[:5]) + '。')
    if 'geometry_quality' in configured and model != 'Tripo/Tripo-H3.1':
        raise ChinaReconstructionError('geometry_quality 仅适用于阿里云 Tripo/Tripo-H3.1。')
    if configured.get('texture_quality') not in (None, 'standard', 'detailed'):
        raise ChinaReconstructionError('阿里云 texture_quality 只能是 standard 或 detailed。')
    if configured.get('geometry_quality') not in (None, 'standard', 'ultra'):
        raise ChinaReconstructionError('阿里云 geometry_quality 只能是 standard 或 ultra。')
    parameters = dict(configured)
    if textured:
        parameters['texture'] = True
        parameters['pbr'] = True
        parameters.setdefault('texture_quality', 'standard')
    else:
        parameters['texture'] = False
        parameters['pbr'] = False
        # It has no meaning on an untextured result and can confuse a provider-side validator.
        parameters.pop('texture_quality', None)
    return parameters


def reconstruct_china_views(views, output, cfg, check_cancel, progress=None, job_path=None):
    provider = cfg.get('reconstruction_provider')
    if provider not in ('aliyun-bailian','tencent-tokenhub'): raise ChinaReconstructionError('不支持的中国云 3D 平台。')
    if set(views) not in ({'front'}, {'front','left','back'}):
        raise ChinaReconstructionError('图生 3D 需要一张正面主图，或完整的正面、左侧、背面三张图。')
    used = ['front'] if len(views) == 1 else ['front','left','back']
    if provider == 'tencent-tokenhub' and len(used) > 1:
        from .tencent_cos import validate_config, CosUploadError
        try: validate_config(cfg)
        except CosUploadError as exc: raise ChinaReconstructionError(str(exc)) from None
    key = str(cfg.get('generative_api_key') or '').strip()
    model = str(cfg.get('reconstruction_model') or ('Tripo/Tripo-P1.0' if provider == 'aliyun-bailian' else 'hy-3d-3.1')).strip()
    if not key or not model: raise ChinaReconstructionError('请填写 3D 平台 API Key 和模型 ID。')
    textured = bool(cfg.get('reconstruction_textured', True))
    configured_parameters = cfg.get('reconstruction_parameters') or {}
    aliyun_parameters = (_aliyun_parameters(model, textured, configured_parameters)
                         if provider == 'aliyun-bailian' else None)
    if provider == 'aliyun-bailian' and not str(cfg.get('reconstruction_base_url') or '').strip():
        raise ChinaReconstructionError('阿里云 Tripo 需要填写北京地域工作空间 API 地址（包含 WorkspaceId）。')
    base = _base(cfg.get('reconstruction_base_url'), 'https://tokenhub.tencentmaas.com')
    if provider == 'tencent-tokenhub':
        from .china_image import _tokenhub_base
        base = _tokenhub_base(base).removesuffix('/v1')
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    job_path = Path(job_path) if job_path else output.with_suffix('.job.json')
    def clean_references():
        if provider != 'tencent-tokenhub' or not job.get('cos_storage'): return
        from .tencent_cos import cleanup
        job['cos_cleanup_pending'] = not cleanup(cfg, job)
        _save(job_path, job)
        if job['cos_cleanup_pending'] and progress:
            progress('COS 临时图片清理失败，请在存储桶 forma-references/ 中清理，或配置生命周期自动删除。')
    source_hashes = {v: hashlib.sha256(Path(views[v]).read_bytes()).hexdigest() for v in used}
    fingerprint_data = {'provider':provider,'model':model,'views':source_hashes,'textured':textured}
    if configured_parameters:
        fingerprint_data['parameters'] = configured_parameters
    fingerprint = hashlib.sha256(json.dumps(fingerprint_data,sort_keys=True).encode()).hexdigest()
    job = {}
    if job_path.exists():
        try: job = json.loads(job_path.read_text(encoding='utf-8'))
        except (OSError,ValueError): raise ChinaReconstructionError('3D 作业记录损坏；为避免重复计费，未重新提交。') from None
        if job.get('fingerprint') != fingerprint:
            # A rejected request never received a cloud task ID, so it is safe to
            # discard its local submission record and use the corrected settings.
            if job.get('state') == 'rejected':
                job = {}
            else:
                raise ChinaReconstructionError('参考图、模型或 3D 参数已变化，请创建新任务。')
        if job.get('state') == 'submission_unknown': raise ChinaReconstructionError('上次 3D 提交结果不明，请先在平台控制台核对。')
        if job.get('state') == 'rejected':
            clean_references()
            job['state'] = 'preparing'
        elif job.get('state') in ('failed', 'cancelled'):
            clean_references()
            raise ChinaReconstructionError('该云端 3D 作业已失败或取消，请创建新的设计任务。')
        if output.exists() and job.get('metadata'):
            from .reconstruction import _valid_glb
            _valid_glb(output); clean_references(); return job['metadata']
    deadline = time.monotonic() + 3600
    headers = {'Authorization':'Bearer '+key}
    with httpx.Client(timeout=httpx.Timeout(120,connect=20),follow_redirects=False) as client:
        def guard():
            check_cancel()
            if time.monotonic() >= deadline: raise ChinaReconstructionError('3D 生成等待超时，恢复后会查询原作业。')
        if not job:
            job={'fingerprint':fingerprint,'provider':provider,'model':model,'used_views':used,'state':'preparing'}; _save(job_path,job)
        if job.get('state') == 'preparing':
            if provider == 'aliyun-bailian':
                uploads = job.get('uploads') if isinstance(job.get('uploads'), dict) else {}
                for view in used:
                    if view not in uploads:
                        guard(); uploads[view] = _aliyun_upload(client,key,model,views[view]); job['uploads']=uploads; _save(job_path,job)
                input_data = {'image':uploads['front']} if len(used)==1 else {'images':[
                    {'type':'png','file_token':uploads['front']}, {'type':'png','file_token':uploads['left']},
                    {'type':'png','file_token':uploads['back']}, {}]}
                payload={'model':model,'input':input_data,'parameters':aliyun_parameters}
                submit=base+'/services/aigc/video-generation/3d-generation'
                submit_headers={**headers,'X-DashScope-Async':'enable','X-DashScope-OssResourceResolve':'enable'}
            else:
                if Path(views['front']).stat().st_size > 6 * 1024 * 1024:
                    raise ChinaReconstructionError('腾讯正面参考图不能超过 6MB。')
                payload={'model':model,'image_base64':base64.b64encode(Path(views['front']).read_bytes()).decode(),
                         'enable_pbr':textured,'generate_type':'Normal' if textured else 'Geometry'}
                if len(used) > 1:
                    from .tencent_cos import upload_views, CosUploadError
                    if progress: progress('正在上传左侧与背面参考图到私有 COS，生成两小时有效的链接')
                    try: urls = upload_views(views, cfg, job, lambda: _save(job_path, job), guard)
                    except CosUploadError as exc: raise ChinaReconstructionError(str(exc)) from None
                    payload['multi_view_images'] = [{'view_type':v, 'view_image_url':urls[v]} for v in ('left','back')]
                submit=base+'/v1/api/3d/submit'; submit_headers=headers
            job['state']='submission_unknown'; _save(job_path,job); guard()
            try: response=client.post(submit,headers=submit_headers,json=payload)
            except httpx.HTTPError: raise ChinaReconstructionError('3D 提交连接中断，请先去平台控制台确认，避免重复计费。') from None
            if response.status_code in (400,401,402,403,404,413,422,429):
                job['state']='rejected'; _save(job_path,job); clean_references()
            body=_json(response,'阿里云百炼' if provider=='aliyun-bailian' else '腾讯云 TokenHub')
            task_id=((body.get('output') or {}).get('task_id') if provider=='aliyun-bailian' else body.get('id'))
            if not isinstance(task_id,str) or not task_id: raise ChinaReconstructionError('3D 服务没有返回有效任务 ID，请在平台控制台核对。')
            job.update(state='submitted',task_id=task_id); _save(job_path,job)
            if progress: progress(('阿里云 Tripo' if provider=='aliyun-bailian' else '腾讯 HY-3D')+' 作业已提交')
        query=(base+'/tasks/'+job['task_id']) if provider=='aliyun-bailian' else (base+'/v1/api/3d/query')
        previous=None
        while True:
            guard()
            try:
                response=client.get(query,headers=headers) if provider=='aliyun-bailian' else client.post(query,headers=headers,json={'model':model,'id':job['task_id']})
            except httpx.HTTPError: raise ChinaReconstructionError('3D 作业查询连接中断，恢复后会继续查询同一任务。') from None
            body=_json(response,'阿里云百炼' if provider=='aliyun-bailian' else '腾讯云 TokenHub')
            status=((body.get('output') or {}).get('task_status') if provider=='aliyun-bailian' else body.get('status'))
            if status in ('SUCCEEDED','completed'): break
            if status in ('FAILED','CANCELED','UNKNOWN','failed','cancelled','unknown'):
                job['state']='failed'; _save(job_path,job); clean_references(); raise ChinaReconstructionError('云端 3D 作业失败、取消或已过期。')
            if status not in ('PENDING','RUNNING','queued','in_progress'): raise ChinaReconstructionError('3D 服务返回了未知作业状态。')
            if progress and status!=previous: progress('3D 作业正在排队' if status in ('PENDING','queued') else '云端正在生成几何与纹理')
            previous=status; time.sleep(15 if provider == 'aliyun-bailian' else 5)
        clean_references()
        if provider=='aliyun-bailian':
            results=(body.get('output') or {}).get('results') or []; item=results[0] if results else {}
            url=item.get('pbr_model_url') if textured else item.get('base_model_url')
        else:
            files=body.get('data') or []; item=next((x for x in files if isinstance(x,dict) and str(x.get('type','')).lower()=='glb'),{})
            url=item.get('url')
        if not url: raise ChinaReconstructionError('3D 服务未返回 GLB 文件。')
        if progress: progress('正在下载并验证 GLB 模型')
        _download(client,url,output,guard)
        metadata={'provider':provider,'model':model,'request_id':job['task_id'],'used_views':used,
                  'textured':textured,'url':url,'format':'glb','file_bytes':output.stat().st_size}
        job.update(state='completed',metadata=metadata); _save(job_path,job); return metadata
