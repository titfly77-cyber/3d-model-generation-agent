import base64
import json
import struct

import httpx
import pytest
from PIL import Image

from backend import reconstruction as provider


# Queue URLs deliberately omit /v2/multi-view: the actual service may route all
# sub-endpoints through an app-level queue. Tests must not mask that distinction.
QUEUE_JOB = 'https://queue.fal.run/fal-ai/hunyuan3d/requests/request-123'
MESH_URL = 'https://v3.fal.media/files/unit-test/mesh.glb'
SECRET = 'secret-that-must-not-be-logged'
GLB_JSON = json.dumps({'asset': {'version': '2.0'}, 'scenes': [{}]}).encode()
GLB_JSON += b' ' * (-len(GLB_JSON) % 4)
GLB = struct.pack('<4sII', b'glTF', 2, 20 + len(GLB_JSON)) + struct.pack('<I4s', len(GLB_JSON), b'JSON') + GLB_JSON


@pytest.fixture
def views(tmp_path):
    result = {}
    for name, color in [('front', 'red'), ('left', 'green'), ('back', 'blue')]:
        result[name] = tmp_path / f'{name}.png'
        Image.new('RGBA', (32, 32), color).save(result[name])
    return result


@pytest.fixture
def cfg():
    return {'reconstruction_provider': 'fal-hunyuan', 'generative_api_key': SECRET}


def mock_transport(monkeypatch, handler):
    original_client = httpx.Client
    monkeypatch.setattr(provider.httpx, 'Client',
                        lambda **kwargs: original_client(transport=httpx.MockTransport(handler), **kwargs))
    monkeypatch.setattr(provider, 'POLL_INTERVAL', 0)


def normal_response(request):
    if request.method == 'POST':
        return httpx.Response(200, json={'request_id': 'request-123',
            'status_url': QUEUE_JOB + '/status', 'response_url': QUEUE_JOB,
            'cancel_url': QUEUE_JOB + '/cancel'})
    if str(request.url) == QUEUE_JOB + '/status':
        return httpx.Response(200, json={'status': 'COMPLETED'})
    if str(request.url) == QUEUE_JOB:
        return httpx.Response(200, json={'model_mesh': {'url': MESH_URL, 'file_size': len(GLB)}})
    if str(request.url) == MESH_URL:
        return httpx.Response(200, content=GLB)
    if request.method == 'PUT' and str(request.url) == QUEUE_JOB + '/cancel':
        return httpx.Response(202, json={'status': 'CANCELLATION_REQUESTED'})
    raise AssertionError(f'Unexpected request: {request.method} {request.url}')


@pytest.mark.parametrize('multiview', [True, False])
def test_exact_native_routes_payloads_and_credentials(monkeypatch, tmp_path, views, cfg, multiview):
    calls, progress = [], []
    def handler(request):
        calls.append(request)
        if request.url.host == 'queue.fal.run':
            assert request.headers['authorization'] == 'Key ' + SECRET
        else:
            assert 'authorization' not in request.headers
        return normal_response(request)
    mock_transport(monkeypatch, handler)
    chosen = views if multiview else {'front': views['front']}
    cfg['reconstruction_seed'] = 43
    cfg['reconstruction_textured'] = multiview
    output = tmp_path / 'raw.glb'
    result = provider.reconstruct_views(chosen, output, cfg, lambda: None, progress.append)
    model = provider.MULTI_MODEL if multiview else provider.SINGLE_MODEL
    assert str(calls[0].url) == provider.QUEUE_ORIGIN + '/' + model
    payload = json.loads(calls[0].content)
    fields = ['front_image_url', 'left_image_url', 'back_image_url'] if multiview else ['input_image_url']
    assert sorted(k for k in payload if k.endswith('_image_url')) == sorted(fields)
    for field, view in zip(fields, chosen):
        prefix, encoded = payload[field].split(',', 1)
        assert prefix == 'data:image/png;base64'
        assert base64.b64decode(encoded) == chosen[view].read_bytes()
    assert payload['seed'] == result['seed'] == 43
    assert payload['textured_mesh'] == result['textured'] == multiview
    assert result['model'] == model
    assert result['used_views'] == list(chosen)
    assert result['request_id'] == 'request-123' and result['url'] == MESH_URL
    assert output.read_bytes() == GLB
    assert any('下载' in message for message in progress)
    persisted = output.with_suffix('.job.json').read_text(encoding='utf-8')
    assert SECRET not in persisted and 'base64' not in persisted


def test_poll_failure_resume_and_completed_cache_do_not_submit_again(monkeypatch, tmp_path, views, cfg):
    calls, fail_once = [], [True]
    def handler(request):
        calls.append(request)
        if fail_once and str(request.url) == QUEUE_JOB + '/status' and fail_once.pop():
            raise httpx.ReadTimeout('private upstream details ' + SECRET, request=request)
        return normal_response(request)
    mock_transport(monkeypatch, handler)
    output = tmp_path / 'raw.glb'
    with pytest.raises(provider.ReconstructionError) as caught:
        provider.reconstruct_views(views, output, cfg, lambda: None)
    assert SECRET not in str(caught.value)
    persisted = json.loads(output.with_suffix('.job.json').read_text(encoding='utf-8'))
    assert persisted['request_id'] == 'request-123'
    result = provider.reconstruct_views(views, output, cfg, lambda: None)
    call_count = len(calls)
    assert provider.reconstruct_views(views, output, cfg, lambda: None) == result
    assert len(calls) == call_count
    assert sum(r.method == 'POST' for r in calls) == 1


def test_cancel_preserves_exception_and_requests_remote_cancellation(monkeypatch, tmp_path, views, cfg):
    class UserCancelled(Exception):
        pass
    calls, cancel = [], [False]
    signal = UserCancelled('stop')
    def check():
        if cancel[0]:
            raise signal
    def handler(request):
        calls.append(request)
        return normal_response(request)
    mock_transport(monkeypatch, handler)
    def progress(_):
        cancel[0] = True
    with pytest.raises(UserCancelled) as caught:
        provider.reconstruct_views(views, tmp_path / 'raw.glb', cfg, check, progress)
    assert caught.value is signal
    assert [str(r.url) for r in calls if r.method == 'PUT'] == [QUEUE_JOB + '/cancel']
    assert not (tmp_path / 'raw.glb').exists()


def test_deadline_preserves_queue_job_without_cancelling_or_resubmitting(monkeypatch, tmp_path, views, cfg):
    calls, clock = [], [0.0]
    def handler(request):
        calls.append(request)
        if str(request.url) == QUEUE_JOB + '/status':
            return httpx.Response(200, json={'status': 'IN_QUEUE'})
        return normal_response(request)
    def tick():
        clock[0] += .2
        return clock[0]
    mock_transport(monkeypatch, handler)
    monkeypatch.setattr(provider.time, 'monotonic', tick)
    cfg['reconstruction_timeout_seconds'] = 1
    with pytest.raises(provider.ReconstructionError, match='等待超时'):
        provider.reconstruct_views(views, tmp_path / 'raw.glb', cfg, lambda: None)
    assert sum(r.method == 'POST' for r in calls) == 1
    assert not any(r.method == 'PUT' for r in calls)
    saved = json.loads((tmp_path / 'raw.job.json').read_text(encoding='utf-8'))
    assert saved['request_id'] == 'request-123'


@pytest.mark.parametrize('bad_views', [{}, {'front', 'back'}, {'front', 'left', 'top'}, {'front', 'left', 'back', 'right'}])
def test_unsupported_view_sets_are_rejected_without_silent_fallback(tmp_path, views, cfg, bad_views):
    chosen = {name: views.get(name, views['front']) for name in bad_views}
    with pytest.raises(provider.ReconstructionError, match='不会自动丢弃视角'):
        provider.reconstruct_views(chosen, tmp_path / 'raw.glb', cfg, lambda: None)


def test_untrusted_queue_url_never_receives_key(monkeypatch, tmp_path, views, cfg):
    calls = []
    def handler(request):
        calls.append(request)
        result = normal_response(request).json()
        result['status_url'] = 'https://attacker.invalid/status'
        return httpx.Response(200, json=result)
    mock_transport(monkeypatch, handler)
    with pytest.raises(provider.ReconstructionError, match='队列地址'):
        provider.reconstruct_views(views, tmp_path / 'raw.glb', cfg, lambda: None)
    assert len(calls) == 1 and calls[0].url.host == 'queue.fal.run'


@pytest.mark.parametrize('bad_content', [b'not a GLB model' * 5, GLB[:-4], b'<html>error</html>' * 5])
def test_corrupted_model_rejected_without_publishing_partial_file(monkeypatch, tmp_path, views, cfg, bad_content):
    def handler(request):
        if str(request.url) == MESH_URL:
            return httpx.Response(200, content=bad_content)
        return normal_response(request)
    mock_transport(monkeypatch, handler)
    with pytest.raises(provider.ReconstructionError, match='GLB'):
        provider.reconstruct_views(views, tmp_path / 'raw.glb', cfg, lambda: None)
    assert not (tmp_path / 'raw.glb').exists()
    assert not (tmp_path / 'raw.glb.download').exists()


@pytest.mark.parametrize('oversize_kind', ['metadata', 'header', 'stream'])
def test_model_download_size_limits(monkeypatch, tmp_path, views, cfg, oversize_kind):
    monkeypatch.setattr(provider, 'MAX_GLB_BYTES', 1024)
    def handler(request):
        if str(request.url) == QUEUE_JOB and oversize_kind == 'metadata':
            return httpx.Response(200, json={'model_mesh': {'url': MESH_URL, 'file_size': 2048}})
        if str(request.url) == MESH_URL:
            if oversize_kind == 'header':
                return httpx.Response(200, content=GLB, headers={'content-length': '2048'})
            return httpx.Response(200, stream=httpx.ByteStream(b'x' * 2048))
        return normal_response(request)
    mock_transport(monkeypatch, handler)
    with pytest.raises(provider.ReconstructionError, match='下载限制'):
        provider.reconstruct_views(views, tmp_path / 'raw.glb', cfg, lambda: None)
    assert not (tmp_path / 'raw.glb').exists()
    assert not (tmp_path / 'raw.glb.download').exists()


def test_redirect_to_localhost_is_blocked(monkeypatch, tmp_path, views, cfg):
    calls = []
    def handler(request):
        calls.append(request)
        if str(request.url) == MESH_URL:
            return httpx.Response(302, headers={'location': 'https://127.0.0.1/private'})
        return normal_response(request)
    mock_transport(monkeypatch, handler)
    with pytest.raises(provider.ReconstructionError, match='下载地址'):
        provider.reconstruct_views(views, tmp_path / 'raw.glb', cfg, lambda: None)
    assert all(r.url.host != '127.0.0.1' for r in calls)


def test_cloud_failure_does_not_echo_error_text_or_secret(monkeypatch, tmp_path, views, cfg):
    def handler(request):
        if str(request.url) == QUEUE_JOB + '/status':
            return httpx.Response(200, json={'status': 'COMPLETED', 'error': 'echo ' + SECRET})
        return normal_response(request)
    mock_transport(monkeypatch, handler)
    with pytest.raises(provider.ReconstructionError) as caught:
        provider.reconstruct_views(views, tmp_path / 'raw.glb', cfg, lambda: None)
    assert SECRET not in str(caught.value)
    assert SECRET not in (tmp_path / 'raw.job.json').read_text(encoding='utf-8')


def test_input_change_does_not_reuse_or_duplicate_remote_job(monkeypatch, tmp_path, views, cfg):
    calls = []
    def handler(request):
        calls.append(request)
        return normal_response(request)
    mock_transport(monkeypatch, handler)
    provider.reconstruct_views(views, tmp_path / 'raw.glb', cfg, lambda: None)
    cfg['reconstruction_seed'] = 43
    with pytest.raises(provider.ReconstructionError, match='设置已变化'):
        provider.reconstruct_views(views, tmp_path / 'raw.glb', cfg, lambda: None)
    assert sum(r.method == 'POST' for r in calls) == 1


def test_uncertain_submit_is_not_repeated(monkeypatch, tmp_path, views, cfg):
    calls = []
    def handler(request):
        calls.append(request)
        raise httpx.ReadTimeout('response lost', request=request)
    mock_transport(monkeypatch, handler)
    with pytest.raises(provider.ReconstructionError):
        provider.reconstruct_views(views, tmp_path / 'raw.glb', cfg, lambda: None)
    with pytest.raises(provider.ReconstructionError, match='尚无法确认'):
        provider.reconstruct_views(views, tmp_path / 'raw.glb', cfg, lambda: None)
    assert len(calls) == 1


def test_explicit_unauthorized_submission_can_retry_with_corrected_key(monkeypatch, tmp_path, views, cfg):
    calls, reject = [], [True]
    def handler(request):
        calls.append(request)
        if request.method == 'POST' and reject[0]:
            reject[0] = False
            return httpx.Response(401, text='echo ' + SECRET)
        return normal_response(request)
    mock_transport(monkeypatch, handler)
    with pytest.raises(provider.ReconstructionError, match='401') as caught:
        provider.reconstruct_views(views, tmp_path / 'raw.glb', cfg, lambda: None)
    assert SECRET not in str(caught.value)
    cfg['generative_api_key'] = 'corrected-key'
    result = provider.reconstruct_views(views, tmp_path / 'raw.glb', cfg, lambda: None)
    assert result['request_id'] == 'request-123'
    assert sum(r.method == 'POST' for r in calls) == 2
