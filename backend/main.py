import asyncio
import base64
import io
import json
import mimetypes
import os
import re
import sqlite3
import shutil
import time
import uuid
import zipfile
import httpx
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlparse
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Header
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, Field
from . import db
from .config import ROOT, ASSETS, UPLOADS, settings, save_settings, public_settings
from .schema import TaskRequest
from .agent import Runner
from .llm import complete

# Windows registry can label .js as text/plain, which breaks ES module loading.
mimetypes.init()
mimetypes.add_type('text/javascript', '.js')
mimetypes.add_type('text/css', '.css')
mimetypes.add_type('image/svg+xml', '.svg')


@asynccontextmanager
async def lifespan(app):
    db.init()
    (ROOT/'data/server.pid').write_text(str(os.getpid()), encoding='utf-8')
    for task in db.list_tasks():
        if task['status'] in ('queued', 'running', 'cancelling'):
            db.update(task['id'], status='interrupted', error='上次运行中断；点击继续即可从已保存的步骤恢复。')
        elif (task.get('status') == 'failed' and task.get('stage') == 'reference_review'
              and (task.get('reference_review') or {}).get('status') != 'passed'
              and (task.get('reference_views') or task.get('reference_sheet_url'))):
            # Make reference-review failures created by an older server version
            # immediately eligible for the new explicit user override.
            db.update(task['id'], can_proceed_unverified_reference=True)
    app.state.runner = Runner()
    yield
    app.state.runner.close()


app = FastAPI(title='FORMA 3D Studio', lifespan=lifespan)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=['127.0.0.1', 'localhost', 'testserver'])


@app.middleware('http')
async def local_origin(request, call_next):
    origin = request.headers.get('origin')
    if request.method not in ('GET','HEAD','OPTIONS') and origin:
        parsed = urlparse(origin)
        if parsed.scheme != 'http' or parsed.hostname not in ('127.0.0.1','localhost') or parsed.port not in (8765,5173):
            return JSONResponse({'detail': '不允许来自其他站点的写入请求'}, status_code=403)
    response = await call_next(request)
    response.headers['X-Content-Type-Options'] = 'nosniff'
    return response


def require_task(tid):
    task = db.get(tid)
    if not task:
        raise HTTPException(404, '任务不存在')
    return task


@app.get('/api/health')
def health():
    cfg = public_settings()
    return {'status':'ok','app':'forma-studio','blender':cfg['blender_available'],'model_configured':cfg['has_api_key'], 'model':cfg['model']}


@app.get('/api/settings')
def read_settings():
    return public_settings()


class SettingsBody(BaseModel):
    provider: str = 'anthropic'
    base_url: str = Field(max_length=500)
    model: str = Field(max_length=200)
    api_key: str = Field(default='', max_length=1000)
    blender_path: str = Field(default='', max_length=1000)
    hunyuan_url: str = Field(default='', max_length=500)
    triposr_python: str = Field(default='', max_length=1000)
    triposr_model: str = Field(default='', max_length=1000)
    image_provider: str = 'fal'
    image_model: str = Field(default='fal-ai/qwen-image-2512', max_length=240)
    image_base_url: str = Field(default='', max_length=500)
    image_api_key: str = Field(default='', max_length=1000)
    image_parameters: dict = Field(default_factory=dict)
    web_reference_enabled: bool = True
    generative_api_key: str = Field(default='', max_length=1000)
    reconstruction_provider: str = 'fal-hunyuan'
    reconstruction_model: str = Field(default='fal-ai/hunyuan3d/v2', max_length=240)
    reconstruction_base_url: str = Field(default='', max_length=500)
    reconstruction_parameters: dict = Field(default_factory=dict)
    reconstruction_textured: bool = True
    tencent_cos_bucket: str = Field(default='', max_length=150)
    tencent_cos_region: str = Field(default='ap-guangzhou', max_length=80)
    tencent_cos_secret_id: str = Field(default='', max_length=1000)
    tencent_cos_secret_key: str = Field(default='', max_length=1000)
    visual_review_enabled: bool = True
    max_quality_retries: int = Field(default=1, ge=0, le=1)


@app.put('/api/settings')
def update_settings(body: SettingsBody):
    if body.provider not in ('anthropic','openai-compatible'):
        raise HTTPException(400, '不支持的接口协议')
    if not body.base_url.startswith(('https://','http://')):
        raise HTTPException(400, '模型接口需为 HTTP(S) 地址')
    if body.hunyuan_url and not body.hunyuan_url.startswith(('https://','http://')):
        raise HTTPException(400, 'Hunyuan 地址需为 HTTP(S) 地址')
    if body.image_provider not in ('fal', 'aliyun-bailian', 'tencent-tokenhub', 'siliconflow', 'volcengine-ark', 'openai-compatible'):
        raise HTTPException(400, '不支持的文生图平台')
    if body.reconstruction_provider not in ('fal-hunyuan', 'aliyun-bailian', 'tencent-tokenhub', 'tripo-official'):
        raise HTTPException(400, '不支持的 3D 重建平台')
    if body.image_provider == 'fal' and not re.fullmatch(r'[A-Za-z0-9_-]+(?:/[A-Za-z0-9_-]+)+', body.image_model):
        raise HTTPException(400, 'fal 模型 ID 应类似 fal-ai/qwen-image-2512，不能填写完整网址')
    if body.image_base_url:
        parsed = urlparse(body.image_base_url)
        if parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise HTTPException(400, '文生图地址必须是有效 HTTP(S) 地址，密钥请单独填写')
    if body.reconstruction_base_url:
        parsed = urlparse(body.reconstruction_base_url)
        if parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise HTTPException(400, '3D 服务地址必须是有效 HTTPS 地址，密钥请单独填写')
    if len(json.dumps(body.image_parameters)) > 20000:
        raise HTTPException(400, '文生图额外参数过长')
    if any(key.lower() in ('api_key', 'authorization', 'headers', 'webhook', 'webhook_url') for key in body.image_parameters):
        raise HTTPException(400, '额外参数仅填写模型参数，密钥请使用独立密钥字段')
    if len(json.dumps(body.reconstruction_parameters)) > 20000:
        raise HTTPException(400, '3D 额外参数过长')
    if any(key.lower() in ('api_key', 'authorization', 'headers', 'webhook', 'webhook_url',
                           'model', 'input', 'inputs') for key in body.reconstruction_parameters):
        raise HTTPException(400, '3D 额外参数不能包含模型、图片、回调或鉴权字段')
    if body.reconstruction_provider == 'aliyun-bailian':
        unsupported = sorted(set(body.reconstruction_parameters) - {'texture_quality', 'geometry_quality'})
        if unsupported:
            raise HTTPException(400, '阿里云 Tripo 额外参数仅支持 texture_quality、geometry_quality；不支持：' + '、'.join(unsupported[:5]))
        if 'geometry_quality' in body.reconstruction_parameters and body.reconstruction_model != 'Tripo/Tripo-H3.1':
            raise HTTPException(400, '阿里云 geometry_quality 仅适用于 Tripo/Tripo-H3.1')
        if body.reconstruction_parameters.get('texture_quality') not in (None, 'standard', 'detailed'):
            raise HTTPException(400, '阿里云 texture_quality 只能填写 standard 或 detailed')
        if body.reconstruction_parameters.get('geometry_quality') not in (None, 'standard', 'ultra'):
            raise HTTPException(400, '阿里云 geometry_quality 只能填写 standard 或 ultra')
    if any(t['status'] in ('running','queued','cancelling') for t in db.list_tasks()):
        raise HTTPException(409, '任务运行期间请先取消或等待完成，再修改设置。')
    save_settings(body.model_dump(exclude_unset=True))
    return public_settings()


@app.post('/api/settings/test')
def test_settings():
    try:
        text, _ = complete('Reply with OK.', 'Test.', max_tokens=32)
        return {'ok':True, 'message':'模型连接成功', 'model':settings()['model']}
    except Exception as exc:
        raise HTTPException(400, str(exc)[:250])


def check_provider_access(key, url, *, model, fal=False):
    if not key:
        raise HTTPException(400, '请先填写对应生成服务的 API Key 并保存')
    try:
        with httpx.Client(timeout=15, follow_redirects=False) as client:
            response = client.get(url, headers={'Authorization': ('Key ' if fal else 'Bearer ') + key},
                                  params={'endpoint_id': model} if fal else None)
        if response.status_code != 200:
            raise HTTPException(400, f'配置查询返回 HTTP {response.status_code}。请核对地址、模型 ID 和凭据权限；尚未执行付费生成。')
        data = response.json()
        if not isinstance(data, dict):
            raise ValueError()
        items = data.get('prices', []) if fal else (data.get('data') or (data.get('output') or {}).get('models', []))
        if not isinstance(items, list) or any(not isinstance(item, dict) for item in items):
            raise ValueError()
        identifier = 'endpoint_id' if fal else ('model' if (data.get('output') or {}).get('models') is not None else 'id')
        if fal and not any(item.get(identifier) == model for item in items):
            raise HTTPException(400, '服务未返回该模型的资料，请核对模型 ID；尚未执行生成。')
        if not fal and not any(item.get(identifier) == model for item in items):
            return {'ok': True, 'verified': 'connection_only', 'model': model,
                    'message': '模型列表接口可访问，但未找到此模型 ID。部分服务不列出图像模型，实际出图仍需验证。'}
        return {'ok': True, 'verified': 'metadata_only', 'model': model,
                'message': '凭据与模型资料查询通过，未执行生成；实际出图、额度和三视图能力尚待任务验证。'}
    except httpx.HTTPError:
        raise HTTPException(400, '生成服务配置查询连接失败，请检查地址与网络。') from None
    except (ValueError, TypeError):
        raise HTTPException(400, '生成服务未返回可识别的模型资料，尚未执行生成。') from None


def check_simple_access(key, url, service):
    if not key:
        raise HTTPException(400, f'请先填写 {service} API Key 并保存')
    try:
        with httpx.Client(timeout=15, follow_redirects=False) as client:
            response = client.get(url, headers={'Authorization': 'Bearer ' + key})
        if response.status_code != 200:
            raise HTTPException(400, f'{service} 配置查询返回 HTTP {response.status_code}。请核对地址、凭据和权限；尚未执行付费生成。')
        body = response.json()
        if not isinstance(body, dict) or (body.get('code') not in (None, 0)):
            raise ValueError()
        return {'ok': True, 'verified': 'account_only', 'message': f'{service} 凭据查询通过，未执行生成任务。'}
    except httpx.HTTPError:
        raise HTTPException(400, f'{service} 配置查询连接失败，请检查地址与网络。') from None
    except (ValueError, TypeError):
        raise HTTPException(400, f'{service} 配置查询返回了无法识别的结果。') from None


@app.post('/api/settings/test-generation')
def test_generation_settings():
    cfg = settings()
    provider = cfg['reconstruction_provider']
    if provider == 'fal-hunyuan':
        return check_provider_access(cfg['generative_api_key'], 'https://api.fal.ai/v1/models/pricing',
                                     model='fal-ai/hunyuan3d/v2/multi-view', fal=True)
    if provider == 'tripo-official':
        base = (cfg.get('reconstruction_base_url') or 'https://openapi.tripo3d.com/v3').rstrip('/')
        return check_simple_access(cfg['generative_api_key'], base + '/account/balance', 'Tripo 官方')
    if provider == 'aliyun-bailian' and not cfg.get('reconstruction_base_url'):
        raise HTTPException(400, '阿里云 Tripo 需要填写包含 WorkspaceId 的北京地域工作空间 API 地址')
    if provider == 'tencent-tokenhub':
        from .china_image import _tokenhub_base
        result = check_provider_access(cfg['generative_api_key'], _tokenhub_base(cfg.get('reconstruction_base_url')) + '/models', model=cfg['reconstruction_model'])
        from .tencent_cos import validate_config, CosUploadError
        try:
            validate_config(cfg)
            note = 'COS 字段完整；尚未验证上传权限或执行多视图生成。'
        except CosUploadError as exc:
            note = str(exc)
        result['message'] += ' ' + note
        return result
    base = (cfg.get('reconstruction_base_url') or 'https://tokenhub.tencentmaas.com').rstrip('/')
    return check_provider_access(cfg['generative_api_key'], base + '/models', model=cfg['reconstruction_model'])


@app.post('/api/settings/test-image')
def test_image_settings():
    cfg = settings()
    if cfg['image_provider'] == 'fal':
        return check_provider_access(cfg['image_api_key'] or cfg['generative_api_key'],
                                     'https://api.fal.ai/v1/models/pricing', model=cfg['image_model'], fal=True)
    defaults = {'aliyun-bailian': 'https://dashscope.aliyuncs.com/api/v1',
                'tencent-tokenhub': 'https://tokenhub.tencentmaas.com',
                'siliconflow': 'https://api.siliconflow.cn/v1',
                'volcengine-ark': 'https://ark.cn-beijing.volces.com/api/v3'}
    base = (cfg['image_base_url'] or defaults.get(cfg['image_provider'], '')).rstrip('/')
    if not base:
        raise HTTPException(400, '请填写文生图 API 地址')
    if base.endswith('/images/generations'):
        base = base[:-len('/images/generations')]
    if cfg['image_provider'] == 'tencent-tokenhub':
        from .china_image import _tokenhub_base
        if cfg['image_model'] not in ('hy-image-v3', 'hy-image-v3.5-preview'):
            raise HTTPException(400, '腾讯文生图当前支持 hy-image-v3 或 hy-image-v3.5-preview，请核对模型 ID')
        base = _tokenhub_base(cfg['image_base_url'])
        result = check_provider_access(cfg['image_api_key'], base + '/models', model=cfg['image_model'])
        result['message'] = '凭据与模型列表查询完成；未执行文生图，生成接口、模型权限和额度尚未验证。' + ('模型列表未找到此 ID。' if result.get('verified') == 'connection_only' else '')
        return result
    return check_provider_access(cfg['image_api_key'], base + '/models', model=cfg['image_model'])


@app.get('/api/tasks')
def tasks():
    return db.list_tasks()


@app.post('/api/tasks', status_code=202)
def create_task(body: TaskRequest, request: Request, idempotency_key: str | None = Header(default=None)):
    if idempotency_key:
        if len(idempotency_key) > 100:
            raise HTTPException(400, '请求标识过长')
        existing = db.by_key(idempotency_key)
        if existing:
            if any(existing.get(key) != value for key, value in body.model_dump().items()):
                raise HTTPException(409, '该请求标识已用于不同任务')
            return existing
    cfg = public_settings()
    if not cfg['has_api_key'] or not cfg['model']:
        raise HTTPException(400, '请先打开模型设置，填写模型接口与模型名称。')
    if not cfg['blender_available']:
        raise HTTPException(400, '请先在设置中配置有效的 Blender 路径。')
    if body.engine != 'neural':
        raise HTTPException(400, '新任务使用图片重建流程；旧程序化资产仍可查看和下载')
    if body.parent_id:
        raise HTTPException(400, '新流程请上传参考图或填写完整描述重新生成；暂不支持旧程序化资产的局部编辑')
    if body.image_id and not (UPLOADS/(body.image_id+'.png')).is_file():
        raise HTTPException(400, '参考图不存在，请重新上传')
    if not cfg.get('generation_available'):
        raise HTTPException(400, '请在设置中配置所选图生 3D 服务的 API Key')
    if not body.image_id and not cfg.get('image_generation_available'):
        raise HTTPException(400, '文字入口需要先配置文生图模型、接口与密钥，用于生成三视图')
    if not body.image_id and cfg.get('reconstruction_provider') == 'tencent-tokenhub' and not cfg.get('tencent_multiview_ready'):
        raise HTTPException(400, '腾讯三视图需要先在设置页配置 COS 存储桶、地域、SecretId 和 SecretKey；配置完成后再创建任务，避免先生成图片后才失败。')
    if sum(t['status'] in ('queued','running','cancelling') for t in db.list_tasks()) >= 8:
        raise HTTPException(429, '队列已满，请等待已有任务完成')
    tid = uuid.uuid4().hex
    (ASSETS/tid).mkdir()
    task = {'id':tid, **body.model_dump(), 'title':body.prompt[:30], 'status':'queued', 'stage':'queued', 'progress':0,
            'cancel_requested':False, 'is_example':False, 'formats':[], 'pipeline_version':2,
            'route':'image-to-3d' if body.image_id else 'text-to-image-to-3d', 'reference_views':[],
            'pipeline_options': {key: cfg.get(key) for key in ('image_provider','image_model','image_base_url','image_parameters',
                                'web_reference_enabled',
                                'reconstruction_provider','reconstruction_model','reconstruction_base_url','reconstruction_parameters','reconstruction_textured',
                                'visual_review_enabled','max_quality_retries')}}
    try:
        task = db.create(task, idempotency_key)
    except sqlite3.IntegrityError:
        existing = db.by_key(idempotency_key) if idempotency_key else None
        if existing and all(existing.get(key) == value for key, value in body.model_dump().items()):
            return existing
        raise HTTPException(409, '请求标识冲突，请创建新的请求')
    db.event(tid, 'queued', '任务已加入生成队列')
    request.app.state.runner.submit(tid)
    return task


@app.get('/api/tasks/{tid}')
def task_detail(tid: str):
    return {**require_task(tid), 'events':db.events(tid)}


@app.delete('/api/tasks/{tid}')
def delete_task(tid: str, request: Request):
    with db.LOCK, request.app.state.runner.lock:
        task = require_task(tid)
        if task['status'] in ('queued', 'running', 'cancelling') or tid in request.app.state.runner.pending:
            raise HTTPException(409, '请先取消任务并等待执行停止，再删除资产。')
        root = ASSETS.resolve()
        asset_path = ASSETS / tid
        if asset_path.is_symlink() or getattr(asset_path, 'is_junction', lambda: False)():
            raise HTTPException(400, '资产目录不能是链接，未删除任何文件。')
        target = (ASSETS / tid).resolve()
        if target.parent != root or target == root:
            raise HTTPException(400, '资产目录不合法，未删除任何文件。')
        try:
            if target.exists():
                # Verify nested links before recursive removal on Windows.
                for entry in target.rglob('*'):
                    if entry.is_symlink() or getattr(entry, 'is_junction', lambda: False)():
                        raise HTTPException(409, '资产目录含链接，请先检查目录后再删除。')
                shutil.rmtree(target)
            db.delete(tid)
        except OSError:
            raise HTTPException(409, '资产文件正在使用或清理失败，请关闭占用程序后重试。') from None
    return {'deleted': True, 'id': tid}


@app.post('/api/tasks/{tid}/cancel')
def cancel_task(tid: str):
    task = require_task(tid)
    if task['status'] not in ('queued','running','cancelling'):
        return task
    db.event(tid, 'cancelling', '已请求停止；当前模型请求返回后将终止后续步骤')
    return db.update(tid, cancel_requested=True, status='cancelling')


@app.post('/api/tasks/{tid}/retry')
def retry_task(tid: str, request: Request):
    with db.LOCK:
        return _retry_task(tid, request)


@app.post('/api/tasks/{tid}/proceed-unverified-reference')
def proceed_unverified_reference(tid: str, request: Request):
    """Resume from reference review after an explicit user risk acceptance."""
    with db.LOCK:
        task = require_task(tid)
        review = task.get('reference_review') or {}
        if task.get('status') != 'failed' or not task.get('can_proceed_unverified_reference'):
            raise HTTPException(409, '当前任务没有等待确认的参考图校验结果')
        if review.get('status') == 'passed':
            raise HTTPException(409, '参考图已经通过校验，无需强制继续')
        if _has_accepted_reconstruction_job(tid):
            raise HTTPException(409, '3D 云端任务已经提交，不能变更参考图确认状态')
        db.update(tid, allow_unverified_reference=True, reference_override_accepted=False)
        db.event(tid, 'reference_review', '使用者选择忽略参考图校验结果并继续生成 3D。')
        return _retry_task(tid, request)


RECONSTRUCTION_SNAPSHOT_KEYS = ('reconstruction_provider', 'reconstruction_model',
    'reconstruction_base_url', 'reconstruction_parameters', 'reconstruction_textured')


def _has_accepted_reconstruction_job(tid: str):
    """True only when a cloud request might already exist and must stay immutable."""
    candidate_root = ASSETS / tid / 'candidates'
    if not candidate_root.is_dir():
        return False
    for record in candidate_root.glob('*/reconstruction-job.json'):
        try:
            job = json.loads(record.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            # A corrupt record is not evidence of an accepted billable job.
            continue
        if job.get('state') in ('submitted', 'submission_unknown', 'completed') or job.get('task_id'):
            return True
    return False


def _refresh_rejected_reconstruction_settings(task):
    """Use current 3D settings only when no accepted cloud job can be queried."""
    if _has_accepted_reconstruction_job(task['id']):
        return task, False
    current = settings()
    options = dict(task.get('pipeline_options') or {})
    updated = {key: current.get(key) for key in RECONSTRUCTION_SNAPSHOT_KEYS}
    if all(options.get(key) == value for key, value in updated.items()):
        return task, False
    options.update(updated)
    return db.update(task['id'], pipeline_options=options), True


def _retry_task(tid: str, request: Request):
    task = require_task(tid)
    if task['status'] not in ('failed','cancelled','interrupted'):
        raise HTTPException(409, '当前任务无法重试')
    runner = request.app.state.runner
    with runner.lock:
        if tid in runner.pending:
            raise HTTPException(409, '上一执行正在结束，请稍后重试')
    task, refreshed = _refresh_rejected_reconstruction_settings(task)
    result = db.update(tid, status='queued', cancel_requested=False, error=None)
    db.event(tid, 'queued', '继续任务，使用当前 3D 设置并保留已完成的步骤' if refreshed else '继续任务，保留已完成的步骤')
    runner.submit(tid)
    return result


@app.get('/api/tasks/{tid}/events')
async def task_events(tid: str, request: Request, after: int = 0):
    require_task(tid)
    async def stream():
        cursor = after
        last_status = None
        while not await request.is_disconnected():
            items = await asyncio.to_thread(db.events, tid, cursor)
            for item in items:
                cursor = item['seq']
                yield f'id: {cursor}\nevent: progress\ndata: {json.dumps(item, ensure_ascii=False)}\n\n'
            task = await asyncio.to_thread(db.get, tid)
            status_key = (task['status'], task.get('stage'), task['updated_at'])
            if status_key != last_status:
                yield 'event: task\ndata: ' + json.dumps(task, ensure_ascii=False) + '\n\n'
                last_status = status_key
            if task['status'] in ('completed','needs_review','failed','cancelled','interrupted'):
                break
            yield ': heartbeat\n\n'
            await asyncio.sleep(1)
    return StreamingResponse(stream(), media_type='text/event-stream', headers={'Cache-Control':'no-cache','X-Accel-Buffering':'no'})


@app.post('/api/uploads')
async def upload_image(file: UploadFile = File(...)):
    content = await file.read(10*1024*1024 + 1)
    if len(content) > 10*1024*1024:
        raise HTTPException(413, '图片不能超过 10MB')
    try:
        with Image.open(io.BytesIO(content)) as original:
            if original.width * original.height > 25_000_000:
                raise ValueError('Image too large')
            original = ImageOps.exif_transpose(original)
            rgba = original.convert('RGBA')
            image = Image.new('RGBA', rgba.size, 'white')
            image.alpha_composite(rgba)
            image = image.convert('RGB')
            image.thumbnail((1280,1280))
            image_id = uuid.uuid4().hex
            image.save(UPLOADS/(image_id+'.png'))
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        raise HTTPException(400, '请上传有效的 PNG、JPG 或 WebP 图片（不超过 2500 万像素）')
    return {'id':image_id,'url':f'/api/uploads/{image_id}'}


@app.get('/api/uploads/{image_id}')
def image_file(image_id: str):
    if len(image_id) != 32 or any(c not in '0123456789abcdef' for c in image_id):
        raise HTTPException(404)
    path = UPLOADS/(image_id+'.png')
    if not path.is_file():
        raise HTTPException(404)
    return FileResponse(path, media_type='image/png')


FILES = {'glb':'model.glb','obj':'model.obj','stl':'model.stl','fbx':'model.fbx','blend':'scene.blend','preview':'preview.png','report':'report.json','design':'design.json','log':'blender.log'}
FILES.update({f'reference_{view}':f'reference_{view}.png' for view in ('front','left','back','sheet')})
FILES.update({f'render_{view}':f'render_{view}.png' for view in ('front','left','back')})
FILES['visual_review'] = 'visual-review.json'


@app.get('/api/tasks/{tid}/files/{kind}')
def asset_file(tid: str, kind: str, download: bool = False, transport: str = 'binary'):
    task = require_task(tid)
    if kind not in FILES:
        raise HTTPException(404)
    path = ASSETS/tid/FILES[kind]
    if not path.is_file():
        raise HTTPException(404, '文件尚未生成')
    mime = 'image/png' if path.suffix == '.png' else {'glb':'model/gltf-binary','report':'application/json','design':'application/json','visual_review':'application/json','log':'text/plain'}.get(kind,'application/octet-stream')
    if transport == 'json':
        return {'filename':f'forma-{tid[:8]}{path.suffix}', 'mime':mime, 'base64':base64.b64encode(path.read_bytes()).decode()}
    return FileResponse(path, media_type=mime, filename=f'forma-{tid[:8]}{path.suffix}' if download else None)


@app.get('/api/tasks/{tid}/download')
def download_bundle(tid: str, transport: str = 'binary'):
    task = require_task(tid)
    if task['status'] not in ('completed', 'needs_review'):
        raise HTTPException(409, '资产尚未完成')
    folder = ASSETS/tid
    bundle = folder/'asset.zip'
    if not bundle.exists():
        temporary = folder/(uuid.uuid4().hex+'.zip.tmp')
        with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as archive:
            for path in folder.iterdir():
                if path.is_file() and path.suffix in ('.glb','.obj','.mtl','.fbx','.stl','.blend','.png','.json'):
                    archive.write(path, path.name)
            textures = folder/'textures'
            if textures.is_dir():
                for path in textures.rglob('*'):
                    if path.is_file():
                        archive.write(path, path.relative_to(folder))
        temporary.replace(bundle)
    if transport == 'json':
        return {'filename':f'forma-{tid[:8]}.zip','mime':'application/zip','base64':base64.b64encode(bundle.read_bytes()).decode()}
    return FileResponse(bundle, filename=f'forma-{tid[:8]}.zip', media_type='application/zip')


dist = ROOT/'frontend/dist'
if dist.is_dir():
    app.mount('/', StaticFiles(directory=dist, html=True), name='frontend')
