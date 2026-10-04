import pytest
from backend import config, llm
from backend import main
from fastapi import HTTPException


@pytest.mark.parametrize('base', ['https://tokenhub.tencentmaas.com', 'https://tokenhub.tencentmaas.com/v1/'])
def test_tokenhub_check_normalizes_version_and_limits_claim(monkeypatch, base):
    monkeypatch.setattr(main, 'settings', lambda: {'image_provider': 'tencent-tokenhub',
        'image_model': 'hy-image-v3', 'image_base_url': base, 'image_api_key': 'test-key'})
    def check(key, url, **kwargs):
        assert url == 'https://tokenhub.tencentmaas.com/v1/models'
        return {'ok': True, 'verified': 'metadata_only'}
    monkeypatch.setattr(main, 'check_provider_access', check)
    assert '未执行文生图' in main.test_image_settings()['message']


def test_connection_status_tracks_exact_config(tmp_path,monkeypatch):
    monkeypatch.setattr(config,'DATA',tmp_path)
    monkeypatch.setattr(config,'SETTINGS',tmp_path/'settings.json')
    monkeypatch.setenv('ANTHROPIC_API_KEY','unit-test-secret')
    config.save_settings({'provider':'anthropic','base_url':'https://test.invalid','model':'model-a'})
    assert config.public_settings()['connection_status']=='unknown'
    config.record_connection(True)
    public=config.public_settings()
    assert public['connection_status']=='connected'
    assert 'api_key' not in public
    config.save_settings({'model':'model-b'})
    assert config.public_settings()['connection_status']=='unknown'


def test_provider_error_does_not_echo_secret(monkeypatch):
    class Response:
        status_code=401
        text='echoed secret should not appear'
    class Client:
        def __init__(self,**kwargs):pass
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def post(self,*args,**kwargs):return Response()
    monkeypatch.setattr(llm.httpx,'Client',Client)
    with pytest.raises(llm.ModelError) as error:
        llm.complete('s','u',config={'provider':'anthropic','base_url':'https://test.invalid','model':'test','api_key':'my-secret'})
    assert '401' in str(error.value)
    assert 'my-secret' not in str(error.value)
    assert 'echoed' not in str(error.value)


def test_image_provider_switch_never_reuses_fal_key_for_other_provider(tmp_path, monkeypatch):
    monkeypatch.setattr(config, 'DATA', tmp_path)
    monkeypatch.setattr(config, 'SETTINGS', tmp_path/'settings.json')
    monkeypatch.delenv('IMAGE_API_KEY', raising=False)
    config.save_settings({'generative_api_key': 'fal-secret', 'image_provider': 'fal'})
    assert config.public_settings()['image_generation_available']
    config.save_settings({'image_provider': 'openai-compatible', 'image_base_url': 'https://images.example/v1', 'image_model': 'model-x'})
    public = config.public_settings()
    assert public['generation_available'] and not public['image_generation_available']
    assert not public['has_image_api_key']
    assert 'fal-secret' not in str(public)


@pytest.mark.parametrize('fal,body,expected', [
    (True, {'prices':[{'endpoint_id':'model-x'}]}, 'metadata_only'),
    (False, {'data':[{'id':'model-x'}]}, 'metadata_only'),
    (False, {'data':[{'id':'other-model'}]}, 'connection_only'),
    (True, {'prices':[None]}, 'error'),
    (False, {'data':'not-a-list'}, 'error'),
])
def test_provider_checks_only_read_metadata(monkeypatch, fal, body, expected):
    calls = []
    class Response:
        status_code = 200
        def json(self): return body
    class Client:
        def __init__(self, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def get(self, url, **kwargs):
            calls.append((url, kwargs))
            return Response()
        def post(self, *args, **kwargs):
            pytest.fail('A configuration check must not submit generation')
    monkeypatch.setattr(main.httpx, 'Client', Client)
    if expected == 'error':
        with pytest.raises(HTTPException) as error:
            main.check_provider_access('test-key', 'https://example.invalid/models', model='model-x', fal=fal)
        assert error.value.status_code == 400
    else:
        result = main.check_provider_access('test-key', 'https://example.invalid/models', model='model-x', fal=fal)
        assert result['verified'] == expected
        assert 'test-key' not in str(result)
    assert len(calls) == 1


def test_bailian_workspace_model_list_shape_is_recognized(monkeypatch):
    class Response:
        status_code = 200
        def json(self): return {'output': {'models': [{'model': 'Tripo/Tripo-P1.0'}]}}
    class Client:
        def __init__(self, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def get(self, url, **kwargs):
            assert url.endswith('/api/v1/models')
            return Response()
    monkeypatch.setattr(main.httpx, 'Client', Client)
    result = main.check_provider_access('test-key', 'https://workspace.cn-beijing.maas.aliyuncs.com/api/v1/models',
                                        model='Tripo/Tripo-P1.0')
    assert result['verified'] == 'metadata_only'
