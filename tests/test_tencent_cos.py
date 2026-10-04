import json

import httpx
import pytest
from PIL import Image

from backend import china_reconstruction as cr, config, tencent_cos, main
from test_china_providers import GLB, mock_client


CFG = {'reconstruction_provider': 'tencent-tokenhub', 'reconstruction_model': 'hy-3d-3.1',
       'reconstruction_base_url': 'https://tokenhub.tencentmaas.com/v1/', 'generative_api_key': 'test-key',
       'tencent_cos_bucket': 'studio-1250000000', 'tencent_cos_region': 'ap-guangzhou',
       'tencent_cos_secret_id': 'cos-test-id', 'tencent_cos_secret_key': 'cos-test-secret'}


@pytest.fixture
def inputs(tmp_path):
    views = {}
    for name in ('front', 'left', 'back'):
        views[name] = tmp_path / (name + '.png')
        Image.new('RGB', (256, 256), 'red').save(views[name])
    return views


@pytest.fixture
def cos(monkeypatch):
    class Fake:
        uploaded = []
        deleted = []
        def put_object(self, **kw):
            assert kw['ACL'] == 'private' and kw['ContentType'] == 'image/png'
            self.uploaded.append(kw['Key'])
        def get_presigned_url(self, **kw):
            assert kw['Method'] == 'GET' and kw['Expired'] == 7200
            return 'https://signed.example/' + kw['Key'] + '?signature=temporary-secret'
        def delete_object(self, **kw): self.deleted.append(kw['Key'])
    fake = Fake()
    monkeypatch.setattr(tencent_cos, 'client_for', lambda cfg: fake)
    return fake


@pytest.mark.parametrize('terminal', ['completed', 'failed', 'rejected'])
def test_multiview_upload_submit_and_terminal_cleanup(monkeypatch, tmp_path, inputs, cos, terminal):
    monkeypatch.setattr(cr, '_safe_download', lambda value: value)
    def handler(req):
        assert '/v1/v1/' not in str(req.url)
        if req.url.path.endswith('/submit'):
            payload = json.loads(req.content)
            assert payload['image_base64']
            assert [v['view_type'] for v in payload['multi_view_images']] == ['left', 'back']
            assert all(v['view_image_url'].startswith('https://signed.example/') for v in payload['multi_view_images'])
            return httpx.Response(400) if terminal == 'rejected' else httpx.Response(200, json={'id': 'job-1'})
        if req.url.path.endswith('/query'):
            return httpx.Response(200, json={'status': terminal, 'data': [{'type': 'glb', 'url': 'https://cdn.example/model.glb'}]})
        return httpx.Response(200, content=GLB)
    mock_client(monkeypatch, cr, handler)
    if terminal == 'completed':
        result = cr.reconstruct_china_views(inputs, tmp_path/'model.glb', CFG, lambda: None)
        assert result['used_views'] == ['front', 'left', 'back']
    else:
        with pytest.raises(cr.ChinaReconstructionError):
            cr.reconstruct_china_views(inputs, tmp_path/'model.glb', CFG, lambda: None)
    assert len(cos.uploaded) == 2 and cos.deleted == cos.uploaded
    saved = (tmp_path/'model.job.json').read_text(encoding='utf-8')
    assert 'temporary-secret' not in saved and 'cos-test-secret' not in saved


def test_query_disconnect_resumes_without_upload_or_resubmit(monkeypatch, tmp_path, inputs, cos):
    monkeypatch.setattr(cr, '_safe_download', lambda value: value)
    submissions = []
    fail = True
    def handler(req):
        if req.url.path.endswith('/submit'):
            submissions.append(req)
            return httpx.Response(200, json={'id': 'job-1'})
        if req.url.path.endswith('/query'):
            if fail: raise httpx.ConnectError('disconnected')
            return httpx.Response(200, json={'status': 'completed', 'data': [{'type': 'glb', 'url': 'https://cdn.example/model.glb'}]})
        return httpx.Response(200, content=GLB)
    mock_client(monkeypatch, cr, handler)
    with pytest.raises(cr.ChinaReconstructionError, match='查询连接中断'):
        cr.reconstruct_china_views(inputs, tmp_path/'model.glb', CFG, lambda: None)
    assert not cos.deleted
    fail = False
    cr.reconstruct_china_views(inputs, tmp_path/'model.glb', CFG, lambda: None)
    assert len(submissions) == 1 and len(cos.uploaded) == 2 and len(cos.deleted) == 2


def test_credentials_hidden_and_blank_preserved(tmp_path, monkeypatch):
    monkeypatch.setattr(config, 'SETTINGS', tmp_path/'settings.json')
    config.save_settings(CFG)
    config.save_settings({'tencent_cos_secret_id': '', 'tencent_cos_secret_key': ''})
    assert config.settings()['tencent_cos_secret_key'] == 'cos-test-secret'
    public = config.public_settings()
    assert public['has_tencent_cos_credentials'] and public['tencent_multiview_ready']
    assert 'cos-test-secret' not in str(public) and 'cos-test-id' not in str(public)


def test_generation_check_normalizes_url_and_reports_cos_limit(monkeypatch):
    monkeypatch.setattr(main, 'settings', lambda: CFG)
    def check(key, url, **kwargs):
        assert url == 'https://tokenhub.tencentmaas.com/v1/models'
        return {'ok': True, 'message': '模型查询通过。'}
    monkeypatch.setattr(main, 'check_provider_access', check)
    assert '尚未验证上传权限' in main.test_generation_settings()['message']


def test_cleanup_failure_preserves_pending_keys(cos):
    def fail(**kwargs): raise RuntimeError('secret must not escape')
    cos.delete_object = fail
    job = {'cos_storage': {'bucket': CFG['tencent_cos_bucket'], 'region': CFG['tencent_cos_region'],
                          'prefix': 'forma-references/test/', 'keys': {'left': 'forma-references/test/left.png'}}}
    assert not tencent_cos.cleanup(CFG, job)
    assert job['cos_storage']['keys']


def test_installed_sdk_can_sign_private_get_offline():
    client = tencent_cos.client_for(CFG)
    url = client.get_presigned_url(Bucket=CFG['tencent_cos_bucket'],
        Key='forma-references/test/left.png', Method='GET', Expired=7200)
    assert url.startswith('https://studio-1250000000.cos.ap-guangzhou.myqcloud.com/')
    assert 'q-signature=' in url and CFG['tencent_cos_secret_key'] not in url
