import json
import hashlib
import os
import shutil
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
ASSETS = DATA / 'assets'
UPLOADS = DATA / 'uploads'
SETTINGS = DATA / 'settings.json'
for path in (DATA, ASSETS, UPLOADS, ROOT / 'tmp', ROOT / 'cache'):
    path.mkdir(parents=True, exist_ok=True)
for key, value in {
    'TEMP': ROOT / 'tmp', 'TMP': ROOT / 'tmp', 'TMPDIR': ROOT / 'tmp',
    'HF_HOME': ROOT / 'cache/huggingface', 'TORCH_HOME': ROOT / 'cache/torch',
    'XDG_CACHE_HOME': ROOT / 'cache', 'U2NET_HOME': ROOT / 'cache/u2net',
    'BLENDER_USER_CONFIG': ROOT / 'data/blender/config',
    'BLENDER_USER_SCRIPTS': ROOT / 'data/blender/scripts',
    'BLENDER_USER_DATAFILES': ROOT / 'data/blender/datafiles',
}.items():
    os.environ[key] = str(value)


def settings():
    auth = os.getenv('ANTHROPIC_API_KEY') or os.getenv('ANTHROPIC_AUTH_TOKEN', '')
    defaults = {
        'provider': 'anthropic' if auth else 'openai-compatible',
        'base_url': os.getenv('ANTHROPIC_BASE_URL', 'https://api.anthropic.com') if auth else os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1'),
        'api_key': auth or os.getenv('OPENAI_API_KEY', ''),
        'model': os.getenv('ANTHROPIC_MODEL', 'claude-sonnet-4-6') if auth else os.getenv('OPENAI_MODEL', ''),
        'blender_path': os.getenv('BLENDER_PATH', shutil.which('blender') or ('D:/blender/blender.exe' if Path('D:/blender/blender.exe').exists() else '')),
        'hunyuan_url': '', 'triposr_python': '', 'triposr_model': '',
        'image_provider': 'fal', 'image_model': 'fal-ai/qwen-image-2512',
        'image_base_url': '', 'image_api_key': os.getenv('IMAGE_API_KEY', ''),
        'image_parameters': {}, 'generative_api_key': os.getenv('FAL_KEY', ''),
        'web_reference_enabled': True,
        'reconstruction_provider': 'fal-hunyuan', 'reconstruction_textured': True,
        'reconstruction_model': 'fal-ai/hunyuan3d/v2', 'reconstruction_base_url': '',
        'reconstruction_parameters': {},
        'tencent_cos_bucket': '', 'tencent_cos_region': 'ap-guangzhou',
        'tencent_cos_secret_id': '', 'tencent_cos_secret_key': '',
        'visual_review_enabled': True, 'max_quality_retries': 1,
    }
    if SETTINGS.exists():
        defaults.update(json.loads(SETTINGS.read_text(encoding='utf-8')))
    return defaults


def save_settings(values):
    previous = json.loads(SETTINGS.read_text(encoding='utf-8')) if SETTINGS.exists() else {}
    if ('image_provider' in values and values['image_provider'] != previous.get('image_provider', 'fal')
            and not str(values.get('image_api_key') or '').strip()):
        previous.pop('image_api_key', None)
    if ('reconstruction_provider' in values
            and values['reconstruction_provider'] != previous.get('reconstruction_provider', 'fal-hunyuan')
            and not str(values.get('generative_api_key') or '').strip()):
        previous.pop('generative_api_key', None)
    allowed = {'provider', 'base_url', 'api_key', 'model', 'blender_path', 'hunyuan_url', 'triposr_python', 'triposr_model',
               'image_provider', 'image_model', 'image_base_url', 'image_api_key', 'image_parameters',
               'web_reference_enabled',
               'generative_api_key', 'reconstruction_provider', 'reconstruction_textured',
               'reconstruction_model', 'reconstruction_base_url',
               'reconstruction_parameters', 'tencent_cos_bucket', 'tencent_cos_region',
               'tencent_cos_secret_id', 'tencent_cos_secret_key',
               'visual_review_enabled', 'max_quality_retries'}
    for key, value in values.items():
        if key in allowed and not (key in ('api_key', 'image_api_key', 'generative_api_key', 'tencent_cos_secret_id', 'tencent_cos_secret_key') and not str(value or '').strip()):
            previous[key] = value.strip() if isinstance(value, str) else value
    temporary = SETTINGS.with_suffix('.tmp')
    temporary.write_text(json.dumps(previous, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(SETTINGS)


def public_settings():
    result = settings()
    result['has_tencent_cos_credentials'] = bool(result.pop('tencent_cos_secret_id', '') and result.get('tencent_cos_secret_key'))
    result.pop('tencent_cos_secret_key', None)
    result['tencent_multiview_ready'] = False
    try:
        from .tencent_cos import validate_config
        validate_config(settings())
        result['tencent_multiview_ready'] = True
    except ValueError:
        pass
    except RuntimeError:
        pass
    result['has_api_key'] = bool(result.pop('api_key', ''))
    fal_key = result.pop('generative_api_key', '')
    image_key = result.pop('image_api_key', '')
    result['has_generative_api_key'] = bool(fal_key)
    matching = {'fal':'fal-hunyuan','aliyun-bailian':'aliyun-bailian','tencent-tokenhub':'tencent-tokenhub'}
    shared_key = fal_key if matching.get(result['image_provider']) == result['reconstruction_provider'] else ''
    result['has_image_api_key'] = bool(image_key or shared_key)
    result['generation_available'] = bool(fal_key)
    result['image_generation_available'] = bool(result['has_image_api_key'] and result['image_model'] and
        (result['image_provider'] in ('fal','aliyun-bailian','tencent-tokenhub','siliconflow','volcengine-ark') or result['image_base_url']))
    result['pipeline_version'] = 2
    result['blender_available'] = bool(result['blender_path'] and Path(result['blender_path']).is_file())
    result['triposr_available'] = bool(result['triposr_python'] and Path(result['triposr_python']).is_file() and result['triposr_model'] and Path(result['triposr_model']).exists())
    result['connection_status'] = 'unknown'
    status_path = DATA/'connection.json'
    if status_path.exists():
        saved = json.loads(status_path.read_text(encoding='utf-8'))
        if saved.get('fingerprint') == connection_fingerprint():
            result['connection_status'] = saved['status']
    return result


def connection_fingerprint():
    cfg = settings()
    return hashlib.sha256(json.dumps([cfg[k] for k in ('provider','base_url','model','api_key')]).encode()).hexdigest()


def record_connection(ok):
    path = DATA/'connection.json'
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps({'fingerprint':connection_fingerprint(),'status':'connected' if ok else 'failed','checked_at':time.time()}),encoding='utf-8')
    temp.replace(path)
