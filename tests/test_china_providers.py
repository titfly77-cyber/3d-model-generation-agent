import base64
import io
import json
import struct

import httpx
import pytest
from PIL import Image

from backend import china_image, china_reconstruction, image_generation, reconstruction


KEY = 'provider-test-key'


def image_data_url():
    buffer = io.BytesIO()
    Image.new('RGB', (768, 256), 'orange').save(buffer, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode()


GLB_JSON = json.dumps({'asset': {'version': '2.0'}, 'scenes': [{}]}).encode()
GLB_JSON += b' ' * (-len(GLB_JSON) % 4)
GLB = struct.pack('<4sII', b'glTF', 2, 20 + len(GLB_JSON)) + struct.pack('<I4s', len(GLB_JSON), b'JSON') + GLB_JSON


def mock_client(monkeypatch, module, handler):
    original = httpx.Client
    calls = []
    def record(request):
        calls.append(request)
        return handler(request)
    monkeypatch.setattr(module.httpx, 'Client', lambda **kwargs: original(
        **kwargs, transport=httpx.MockTransport(record)))
    return calls


def test_siliconflow_is_first_class_images_provider(monkeypatch, tmp_path):
    def handler(request):
        assert str(request.url) == 'https://api.siliconflow.cn/v1/images/generations'
        assert request.headers['authorization'] == 'Bearer ' + KEY
        body = json.loads(request.content)
        assert body['model'] == 'example/current-image-model' and body['n'] == 1
        return httpx.Response(200, json={'data': [{'url': image_data_url()}]})
    calls = mock_client(monkeypatch, image_generation, handler)
    cfg = {'image_provider': 'siliconflow', 'image_model': 'example/current-image-model',
           'image_api_key': KEY, 'image_parameters': {'n': 1}}
    result = image_generation.generate_sheet('three views', tmp_path/'sheet.png', cfg, lambda: None)
    assert result['provider'] == 'siliconflow' and len(calls) == 1


def test_volcengine_ark_uses_seedream_single_image_defaults(monkeypatch, tmp_path):
    def handler(request):
        assert str(request.url) == 'https://ark.cn-beijing.volces.com/api/v3/images/generations'
        assert request.headers['authorization'] == 'Bearer ' + KEY
        body = json.loads(request.content)
        assert body == {'model': 'doubao-seedream-5-0-lite-260128', 'prompt': 'three views',
                        'size': '2K', 'response_format': 'url', 'output_format': 'png',
                        'watermark': True, 'sequential_image_generation': 'disabled'}
        return httpx.Response(200, json={'data': [{'url': image_data_url()}]})
    calls = mock_client(monkeypatch, image_generation, handler)
    cfg = {'image_provider': 'volcengine-ark',
           'image_model': 'doubao-seedream-5-0-lite-260128',
           'image_api_key': KEY, 'image_parameters': {}}
    result = image_generation.generate_sheet('three views', tmp_path/'sheet.png', cfg, lambda: None)
    assert result['provider'] == 'volcengine-ark' and len(calls) == 1


def test_aliyun_image_submission_and_poll(monkeypatch, tmp_path):
    monkeypatch.setattr(china_image.time, 'sleep', lambda _: None)
    def handler(request):
        url = str(request.url)
        if url.endswith('/image-synthesis'):
            assert request.headers['x-dashscope-async'] == 'enable'
            body = json.loads(request.content)
            assert body['model'] == 'wan2.2-t2i-plus'
            return httpx.Response(200, json={'output': {'task_id': 'ali-image-1'}})
        assert url.endswith('/tasks/ali-image-1')
        return httpx.Response(200, json={'output': {'task_status': 'SUCCEEDED',
            'results': [{'url': image_data_url()}]}})
    calls = mock_client(monkeypatch, china_image, handler)
    cfg = {'image_provider': 'aliyun-bailian', 'image_model': 'wan2.2-t2i-plus',
           'image_api_key': KEY, 'image_parameters': {}}
    result = image_generation.generate_sheet('three views', tmp_path/'sheet.png', cfg, lambda: None)
    assert result['request_id'] == 'ali-image-1' and len(calls) == 2


@pytest.mark.parametrize('base', ['https://tokenhub.tencentmaas.com', 'https://tokenhub.tencentmaas.com/v1/'])
def test_tencent_hy_image_35_chat_response(monkeypatch, tmp_path, base):
    def handler(request):
        body = json.loads(request.content)
        assert str(request.url).endswith('/v1/wand/hunyuan-image/v35-generation')
        assert '/v1/v1/' not in str(request.url)
        assert body['messages'][0]['content'][0] == {'type': 'text', 'text': 'three views'}
        return httpx.Response(200, json={'id': 'hy-image-35-1', 'choices': [
            {'delta': {'type': 'image', 'image': {'url': image_data_url(), 'width': 768, 'height': 256}}}]})
    mock_client(monkeypatch, china_image, handler)
    cfg = {'image_provider': 'tencent-tokenhub', 'image_model': 'hy-image-v3.5-preview',
           'image_api_key': KEY, 'image_parameters': {}, 'image_base_url': base}
    result = image_generation.generate_sheet('three views', tmp_path/'sheet.png', cfg, lambda: None)
    assert result['model'] == 'hy-image-v3.5-preview'


def test_tencent_hy_image_35_uses_local_web_reference_as_data_uri(monkeypatch, tmp_path):
    reference = tmp_path / 'web-reference.png'
    Image.new('RGB', (256, 256), 'orange').save(reference)
    def handler(request):
        body = json.loads(request.content)
        content = body['messages'][0]['content']
        assert content[0] == {'type': 'text', 'text': 'three views'}
        assert content[1]['type'] == 'image_url'
        assert content[1]['image_url']['url'].startswith('data:image/png;base64,')
        return httpx.Response(200, json={'id': 'hy-image-35-1', 'choices': [
            {'delta': {'type': 'image', 'image': {'url': image_data_url(), 'width': 768, 'height': 256}}}]})
    mock_client(monkeypatch, china_image, handler)
    cfg = {'image_provider': 'tencent-tokenhub', 'image_model': 'hy-image-v3.5-preview',
           'image_api_key': KEY, 'image_parameters': {}, 'image_base_url': 'https://tokenhub.tencentmaas.com'}
    result = image_generation.generate_sheet('three views', tmp_path/'sheet.png', cfg, lambda: None,
                                             reference_image=reference)
    assert result['model'] == 'hy-image-v3.5-preview'


@pytest.fixture
def views(tmp_path):
    result = {}
    for name, color in [('front', 'red'), ('left', 'green'), ('back', 'blue')]:
        result[name] = tmp_path/f'{name}.png'
        Image.new('RGB', (256, 256), color).save(result[name])
    return result


def test_tencent_single_image_3d_payload(monkeypatch, tmp_path, views):
    monkeypatch.setattr(china_reconstruction.time, 'sleep', lambda _: None)
    monkeypatch.setattr(china_reconstruction, '_safe_download', lambda value: value)
    def handler(request):
        url = str(request.url)
        if url.endswith('/3d/submit'):
            body = json.loads(request.content)
            assert body['model'] == 'hy-3d-3.1' and body['image_base64']
            assert 'multi_view_images' not in body
            return httpx.Response(200, json={'id': 'tencent-3d-1', 'status': 'queued'})
        if url.endswith('/3d/query'):
            return httpx.Response(200, json={'status': 'completed', 'data': [
                {'type': 'glb', 'url': 'https://cdn.example/model.glb'}]})
        assert url == 'https://cdn.example/model.glb'
        return httpx.Response(200, content=GLB)
    mock_client(monkeypatch, china_reconstruction, handler)
    cfg = {'reconstruction_provider': 'tencent-tokenhub', 'reconstruction_model': 'hy-3d-3.1',
           'generative_api_key': KEY, 'reconstruction_textured': True}
    result = reconstruction.reconstruct_views({'front': views['front']}, tmp_path/'model.glb', cfg, lambda: None)
    assert result['request_id'] == 'tencent-3d-1'


def test_tencent_rejects_local_multiview_before_network(tmp_path, views):
    cfg = {'reconstruction_provider': 'tencent-tokenhub', 'reconstruction_model': 'hy-3d-3.1',
           'generative_api_key': KEY}
    with pytest.raises(china_reconstruction.ChinaReconstructionError, match='COS'):
        reconstruction.reconstruct_views(views, tmp_path/'model.glb', cfg, lambda: None)


def test_aliyun_multiview_upload_and_submit(monkeypatch, tmp_path, views):
    monkeypatch.setattr(china_reconstruction.time, 'sleep', lambda _: None)
    monkeypatch.setattr(china_reconstruction, '_safe_download', lambda value: value)
    upload_number = 0
    def handler(request):
        nonlocal upload_number
        url = str(request.url)
        if '/uploads?' in url:
            return httpx.Response(200, json={'data': {'upload_dir': 'tmp/tests',
                'upload_host': 'https://oss.example', 'oss_access_key_id': 'id',
                'signature': 'sig', 'policy': 'policy', 'x_oss_object_acl': 'private',
                'x_oss_forbid_overwrite': 'true'}})
        if url == 'https://oss.example':
            upload_number += 1
            return httpx.Response(200)
        if url.endswith('/3d-generation'):
            assert request.headers['x-dashscope-async'] == 'enable'
            assert request.headers['x-dashscope-ossresourceresolve'] == 'enable'
            body = json.loads(request.content)
            assert len(body['input']['images']) == 4 and body['input']['images'][3] == {}
            assert all(x['file_token'].startswith('oss://') for x in body['input']['images'][:3])
            assert body['parameters'] == {'texture_quality': 'detailed', 'texture': True, 'pbr': True}
            return httpx.Response(200, json={'output': {'task_id': 'ali-3d-1'}})
        if url.endswith('/tasks/ali-3d-1'):
            return httpx.Response(200, json={'output': {'task_status': 'SUCCEEDED',
                'results': [{'pbr_model_url': 'https://cdn.example/model.glb'}]}})
        assert url == 'https://cdn.example/model.glb'
        return httpx.Response(200, content=GLB)
    mock_client(monkeypatch, china_reconstruction, handler)
    cfg = {'reconstruction_provider': 'aliyun-bailian', 'reconstruction_model': 'Tripo/Tripo-P1.0',
           'reconstruction_base_url': 'https://workspace.cn-beijing.maas.aliyuncs.com/api/v1',
           'generative_api_key': KEY, 'reconstruction_textured': True,
           'reconstruction_parameters': {'texture_quality': 'detailed'}}
    result = reconstruction.reconstruct_views(views, tmp_path/'model.glb', cfg, lambda: None)
    assert result['request_id'] == 'ali-3d-1' and upload_number == 3


def test_aliyun_h31_geometry_and_untextured_parameters():
    assert china_reconstruction._aliyun_parameters('Tripo/Tripo-H3.1', True,
        {'geometry_quality': 'ultra'}) == {'geometry_quality': 'ultra', 'texture': True, 'pbr': True,
                                           'texture_quality': 'standard'}
    assert china_reconstruction._aliyun_parameters('Tripo/Tripo-P1.0', False,
        {'texture_quality': 'detailed'}) == {'texture': False, 'pbr': False}
    with pytest.raises(china_reconstruction.ChinaReconstructionError, match='H3.1'):
        china_reconstruction._aliyun_parameters('Tripo/Tripo-P1.0', True, {'geometry_quality': 'ultra'})
    with pytest.raises(china_reconstruction.ChinaReconstructionError, match='texture_quality'):
        china_reconstruction._aliyun_parameters('Tripo/Tripo-P1.0', True, {'texture_quality': 'high'})


def test_aliyun_404_explains_workspace_api_path():
    with pytest.raises(china_reconstruction.ChinaReconstructionError, match='compatible-mode'):
        china_reconstruction._json(httpx.Response(404), '阿里云百炼')
