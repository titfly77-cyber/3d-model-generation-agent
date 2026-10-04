"""Private, short-lived reference images for TokenHub multiview requests."""
import re
import uuid
from pathlib import Path


class CosUploadError(RuntimeError):
    pass


def validate_config(cfg):
    for name in ('bucket', 'region', 'secret_id', 'secret_key'):
        if not str(cfg.get('tencent_cos_' + name) or '').strip():
            raise CosUploadError('腾讯三视图需要在设置页填写 COS 存储桶、地域、SecretId 和 SecretKey；单图无需 COS。')
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]*-\d+', cfg['tencent_cos_bucket']):
        raise CosUploadError('COS 存储桶名称应包含 APPID，例如 studio-1250000000。')
    if not re.fullmatch(r'[a-z]+-[a-z]+(?:-\d+)?', cfg['tencent_cos_region']):
        raise CosUploadError('COS 地域应类似 ap-guangzhou。')


def client_for(cfg):
    validate_config(cfg)
    from qcloud_cos import CosConfig, CosS3Client
    return CosS3Client(CosConfig(Region=cfg['tencent_cos_region'],
        SecretId=cfg['tencent_cos_secret_id'], SecretKey=cfg['tencent_cos_secret_key'],
        Scheme='https', Timeout=60))


def upload_views(views, cfg, job, persist, guard):
    validate_config(cfg)
    storage = job.setdefault('cos_storage', {'bucket': cfg['tencent_cos_bucket'],
        'region': cfg['tencent_cos_region'], 'prefix': 'forma-references/' + uuid.uuid4().hex + '/', 'keys': {}})
    if (storage['bucket'], storage['region']) != (cfg['tencent_cos_bucket'], cfg['tencent_cos_region']):
        raise CosUploadError('恢复上传时 COS 存储桶或地域已变化，请恢复原配置。')
    persist()
    client = client_for(cfg)
    urls = {}
    for view in ('left', 'back'):
        guard()
        raw = Path(views[view]).read_bytes()
        if len(raw) >= 8 * 1024 * 1024:
            raise CosUploadError('腾讯多视图单张图片必须小于 8MB。')
        key = storage['prefix'] + view + '.png'
        # Persist intent before upload so interrupted uploads can be overwritten/cleaned.
        storage['keys'][view] = key
        persist()
        try:
            client.put_object(Bucket=storage['bucket'], Key=key, Body=raw,
                              ContentType='image/png', ACL='private')
            urls[view] = client.get_presigned_url(Bucket=storage['bucket'], Key=key,
                                                   Method='GET', Expired=7200)
        except Exception:
            raise CosUploadError('COS 图片上传或签名失败，请检查桶地域和 PutObject/GetObject 权限。') from None
    return urls


def cleanup(cfg, job):
    storage = job.get('cos_storage')
    if not storage or not storage.get('keys'):
        return True
    try:
        client = client_for({**cfg, 'tencent_cos_bucket': storage['bucket'],
                             'tencent_cos_region': storage['region']})
        for view, key in list(storage['keys'].items()):
            if not key.startswith('forma-references/') or key != storage['prefix'] + view + '.png':
                return False
            client.delete_object(Bucket=storage['bucket'], Key=key)
            del storage['keys'][view]
        return True
    except Exception:
        return False
