import json
import struct

import httpx
import pytest
from PIL import Image

from backend import reconstruction, tripo_reconstruction


KEY = 'tripo-test-key'
GLB_JSON = json.dumps({'asset': {'version': '2.0'}, 'scenes': [{}]}).encode()
GLB_JSON += b' ' * (-len(GLB_JSON) % 4)
GLB = struct.pack('<4sII', b'glTF', 2, 20 + len(GLB_JSON)) + struct.pack('<I4s', len(GLB_JSON), b'JSON') + GLB_JSON


@pytest.fixture
def views(tmp_path):
    result = {}
    for name, color in [('front', 'red'), ('left', 'green'), ('back', 'blue')]:
        result[name] = tmp_path/f'{name}.png'
        Image.new('RGB', (256, 256), color).save(result[name])
    return result


def install(monkeypatch, handler):
    original = httpx.Client
    calls = []
    def record(request):
        calls.append(request)
        return handler(request)
    monkeypatch.setattr(tripo_reconstruction.httpx, 'Client', lambda **kwargs: original(
        **kwargs, transport=httpx.MockTransport(record)))
    monkeypatch.setattr(tripo_reconstruction.time, 'sleep', lambda _: None)
    monkeypatch.setattr(tripo_reconstruction, '_safe_download_url', lambda value: value)
    return calls


@pytest.mark.parametrize('multiview', [False, True])
def test_tripo_v3_upload_submit_poll_and_download(monkeypatch, tmp_path, views, multiview):
    uploaded = []
    def handler(request):
        url = str(request.url)
        assert request.headers.get('authorization') == 'Bearer ' + KEY if 'cdn.example' not in url else True
        if url.endswith('/files'):
            token = f'file_{len(uploaded) + 1}'
            uploaded.append(token)
            return httpx.Response(200, json={'code': 0, 'data': {'file_token': token}})
        if url.endswith('/generation/image-to-model'):
            body = json.loads(request.content)
            assert body['input'] == 'file_1' and body['texture'] is True and body['geometry_quality'] == 'detailed'
            return httpx.Response(200, json={'code': 0, 'data': {'task_id': 'task_single'}})
        if url.endswith('/generation/multiview-to-model'):
            body = json.loads(request.content)
            assert body['inputs'] == [{'front': 'file_1'}, {'left': 'file_2'}, {'back': 'file_3'}]
            assert body['model'] == 'v3.1-20260211'
            return httpx.Response(200, json={'code': 0, 'data': {'task_id': 'task_multi'}})
        if '/tasks/' in url:
            return httpx.Response(200, json={'code': 0, 'data': {'status': 'success',
                'credits_consumed': 48.0, 'output': {'model_url': 'https://cdn.example/model.glb'}}})
        assert url == 'https://cdn.example/model.glb'
        assert 'authorization' not in request.headers
        return httpx.Response(200, content=GLB)
    calls = install(monkeypatch, handler)
    cfg = {'reconstruction_provider': 'tripo-official', 'reconstruction_model': 'v3.1-20260211',
           'reconstruction_base_url': 'https://openapi.tripo3d.com/v3',
           'generative_api_key': KEY, 'reconstruction_textured': True,
           'reconstruction_parameters': {'geometry_quality': 'detailed'}, 'reconstruction_seed': 43}
    chosen = views if multiview else {'front': views['front']}
    result = reconstruction.reconstruct_views(chosen, tmp_path/'model.glb', cfg, lambda: None)
    assert result['provider'] == 'tripo-official' and result['credits_consumed'] == 48.0
    assert len(uploaded) == (3 if multiview else 1)
    assert calls[-1].headers.get('authorization') is None


def test_tripo_reuses_completed_file_without_network(monkeypatch, tmp_path, views):
    calls = install(monkeypatch, lambda request: (_ for _ in ()).throw(AssertionError('network not expected')))
    output = tmp_path/'model.glb'
    output.write_bytes(GLB)
    cfg = {'reconstruction_provider': 'tripo-official', 'reconstruction_model': 'v3.1-20260211',
           'generative_api_key': KEY, 'reconstruction_textured': True, 'reconstruction_parameters': {}}
    hashes = {name: __import__('hashlib').sha256(views[name].read_bytes()).hexdigest() for name in ('front',)}
    fingerprint_data = {'provider': 'tripo-official', 'base': 'https://openapi.tripo3d.com/v3',
                        'model': 'v3.1-20260211', 'views': hashes, 'textured': True, 'parameters': {}}
    fingerprint = __import__('hashlib').sha256(json.dumps(
        fingerprint_data, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    job = output.with_suffix('.job.json')
    job.write_text(json.dumps({'fingerprint': fingerprint, 'state': 'completed',
        'metadata': {'provider': 'tripo-official', 'request_id': 'task_done'}}), encoding='utf-8')
    result = reconstruction.reconstruct_views({'front': views['front']}, output, cfg, lambda: None)
    assert result['request_id'] == 'task_done' and calls == []


def test_tripo_protects_managed_payload_fields(tmp_path, views):
    cfg = {'reconstruction_provider': 'tripo-official', 'reconstruction_model': 'v3.1-20260211',
           'generative_api_key': KEY, 'reconstruction_parameters': {'input': 'foreign'}}
    with pytest.raises(tripo_reconstruction.TripoReconstructionError, match='不能覆盖'):
        reconstruction.reconstruct_views({'front': views['front']}, tmp_path/'model.glb', cfg, lambda: None)
