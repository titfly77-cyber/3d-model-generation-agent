import io
import base64
import json
import threading
import zipfile
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from backend import config, db, main
from backend.schema import SceneSpec, TaskRequest
from backend.llm import extract_json, ModelError
from backend.examples import robot, chair, lamp


class FakeRunner:
    def __init__(self):
        self.pending=set()
        self.lock=threading.Lock()
    def submit(self, tid):
        self.pending.add(tid)
        return True
    def close(self):
        pass


@pytest.fixture
def available_settings():
    return {'has_api_key': True, 'model': 'test-planner', 'blender_available': True,
            'hunyuan_url': '', 'triposr_available': False,
            'has_generative_api_key': True, 'has_image_api_key': True,
            'generation_available': True, 'image_generation_available': True,
            'image_provider': 'fal', 'image_model': 'fal-ai/qwen-image-2512',
            'image_base_url': '', 'image_parameters': {},
            'reconstruction_provider': 'fal-hunyuan', 'reconstruction_textured': True,
            'visual_review_enabled': True, 'max_quality_retries': 1,
            'pipeline_version': 2}


@pytest.fixture
def client(tmp_path, monkeypatch, available_settings):
    monkeypatch.setattr(db,'DATA',tmp_path)
    monkeypatch.setattr(config,'DATA',tmp_path)
    monkeypatch.setattr(config,'SETTINGS',tmp_path/'settings.json')
    monkeypatch.setattr(main,'ROOT',tmp_path)
    (tmp_path/'data').mkdir()
    assets=tmp_path/'assets';assets.mkdir()
    uploads=tmp_path/'uploads';uploads.mkdir()
    monkeypatch.setattr(main,'ASSETS',assets)
    monkeypatch.setattr(main,'UPLOADS',uploads)
    monkeypatch.setattr(main,'Runner',FakeRunner)
    monkeypatch.setattr(main,'public_settings',lambda: dict(available_settings))
    with TestClient(main.app) as value:
        yield value


def test_examples_are_valid():
    for factory in (robot,chair,lamp):
        scene=SceneSpec.model_validate(factory())
        assert len(scene.parts)>0
        assert len(scene.parts)==len(set(p.name for p in scene.parts))


def test_delete_asset_cleans_only_selected_task(client):
    import sqlite3
    for tid in ('delete-me', 'keep-me'):
        db.create({'id': tid, 'status': 'completed', 'title': tid})
        db.event(tid, 'done', 'test')
        folder = main.ASSETS / tid
        folder.mkdir()
        (folder / 'model.glb').write_bytes(b'test')
    with sqlite3.connect(db.DATA / 'checkpoints.db') as conn:
        conn.executescript('CREATE TABLE checkpoints(thread_id TEXT); CREATE TABLE writes(thread_id TEXT);')
        for table in ('checkpoints', 'writes'):
            conn.executemany(f'INSERT INTO {table} VALUES (?)', [('delete-me',), ('keep-me',)])
    response = client.delete('/api/tasks/delete-me')
    assert response.status_code == 200
    assert db.get('delete-me') is None and db.events('delete-me') == []
    assert not (main.ASSETS / 'delete-me').exists()
    assert db.get('keep-me') and (main.ASSETS / 'keep-me/model.glb').exists()
    with sqlite3.connect(db.DATA / 'checkpoints.db') as conn:
        for table in ('checkpoints', 'writes'):
            assert conn.execute(f'SELECT thread_id FROM {table}').fetchall() == [('keep-me',)]
    assert client.delete('/api/tasks/delete-me').status_code == 404


@pytest.mark.parametrize('status', ['queued', 'running', 'cancelling'])
def test_delete_active_asset_is_rejected(client, status):
    db.create({'id': 'busy', 'status': status})
    assert client.delete('/api/tasks/busy').status_code == 409
    assert db.get('busy')


def test_delete_pending_or_missing_files(client):
    db.create({'id': 'ending', 'status': 'cancelled'})
    main.app.state.runner.pending.add('ending')
    assert client.delete('/api/tasks/ending').status_code == 409
    main.app.state.runner.pending.clear()
    assert client.delete('/api/tasks/ending').status_code == 200


@pytest.mark.parametrize('text',['```json\n{"a":1}\n```','prefix {"a":1} suffix','<think>private text</think>{"a":1}'])
def test_model_json_wrappers(text):
    assert extract_json(text)=={'a':1}


def test_truncated_model_output_rejected():
    with pytest.raises(ModelError):extract_json('{"a":')


def test_geometry_guards():
    scene=chair()
    scene['parts'][0]['scale']=[float('nan'),1,1]
    with pytest.raises(ValueError):SceneSpec.model_validate(scene)
    scene=chair();scene['parts'][0]['material']='missing'
    with pytest.raises(ValueError):SceneSpec.model_validate(scene)
    scene=chair();scene['parts'][1]['name']=scene['parts'][0]['name']
    with pytest.raises(ValueError):SceneSpec.model_validate(scene)


def test_invalid_asset_id_rejected():
    with pytest.raises(ValueError):TaskRequest(prompt='robot',parent_id='../../settings')


def test_idempotency(client):
    a=client.post('/api/tasks',json={'prompt':'A green robot'},headers={'Idempotency-Key':'same'})
    b=client.post('/api/tasks',json={'prompt':'A green robot'},headers={'Idempotency-Key':'same'})
    assert a.status_code==b.status_code==202
    assert a.json()['id']==b.json()['id']
    c=client.post('/api/tasks',json={'prompt':'A green robot','quality':'high'},headers={'Idempotency-Key':'same'})
    assert c.status_code==409
    assert len(client.get('/api/tasks').json())==1


def test_cancel_and_resume_guards(client):
    task=client.post('/api/tasks',json={'prompt':'A chair'}).json()
    assert client.post(f"/api/tasks/{task['id']}/retry").status_code==409
    result=client.post(f"/api/tasks/{task['id']}/cancel").json()
    assert result['cancel_requested'] and result['status']=='cancelling'
    assert len(db.events(task['id']))==2


def test_explicit_reference_override_endpoint_requires_waiting_review(client):
    tid = 'u' * 32
    db.create({'id': tid, 'status': 'failed', 'stage': 'reference_review',
               'can_proceed_unverified_reference': True,
               'reference_review': {'status': 'needs_review', 'summary': '视角不一致', 'issues': ['视角不一致']}})
    response = client.post(f'/api/tasks/{tid}/proceed-unverified-reference')
    assert response.status_code == 200
    task = response.json()
    assert task['status'] == 'queued'
    assert task['allow_unverified_reference'] is True
    assert tid in main.app.state.runner.pending
    assert client.post(f'/api/tasks/{tid}/proceed-unverified-reference').status_code == 409


def test_reference_override_endpoint_rejects_unrelated_failure(client):
    tid = 'v' * 32
    db.create({'id': tid, 'status': 'failed', 'stage': 'reconstruct', 'error': 'cloud failed'})
    assert client.post(f'/api/tasks/{tid}/proceed-unverified-reference').status_code == 409


def test_retry_rejected_reconstruction_uses_corrected_current_settings(client, monkeypatch):
    tid = 'r' * 32
    db.create({'id': tid, 'status': 'failed', 'title': 'Bad endpoint', 'pipeline_options': {
        'reconstruction_provider': 'aliyun-bailian', 'reconstruction_model': 'Tripo/Tripo-P1.0',
        'reconstruction_base_url': 'https://workspace.cn-beijing.maas.aliyuncs.com/compatible-mode/v1',
        'reconstruction_parameters': {}, 'reconstruction_textured': True}})
    record = main.ASSETS / tid / 'candidates' / '1'
    record.mkdir(parents=True)
    (record / 'reconstruction-job.json').write_text(json.dumps({'state': 'rejected'}), encoding='utf-8')
    monkeypatch.setattr(main, 'settings', lambda: {'reconstruction_provider': 'aliyun-bailian',
        'reconstruction_model': 'Tripo/Tripo-P1.0',
        'reconstruction_base_url': 'https://workspace.cn-beijing.maas.aliyuncs.com/api/v1',
        'reconstruction_parameters': {'texture_quality': 'detailed'}, 'reconstruction_textured': True})
    response = client.post(f'/api/tasks/{tid}/retry')
    assert response.status_code == 200
    options = response.json()['pipeline_options']
    assert options['reconstruction_base_url'].endswith('/api/v1')
    assert options['reconstruction_parameters'] == {'texture_quality': 'detailed'}
    assert '当前 3D 设置' in db.events(tid)[-1]['message']


def test_retry_accepted_reconstruction_keeps_frozen_settings(client, monkeypatch):
    tid = 's' * 32
    old = 'https://old.cn-beijing.maas.aliyuncs.com/api/v1'
    db.create({'id': tid, 'status': 'interrupted', 'title': 'Submitted', 'pipeline_options': {
        'reconstruction_provider': 'aliyun-bailian', 'reconstruction_model': 'Tripo/Tripo-P1.0',
        'reconstruction_base_url': old, 'reconstruction_parameters': {}, 'reconstruction_textured': True}})
    record = main.ASSETS / tid / 'candidates' / '1'
    record.mkdir(parents=True)
    (record / 'reconstruction-job.json').write_text(json.dumps({'state': 'submitted', 'task_id': 'cloud-task'}), encoding='utf-8')
    monkeypatch.setattr(main, 'settings', lambda: {'reconstruction_provider': 'fal-hunyuan',
        'reconstruction_model': 'new', 'reconstruction_base_url': '', 'reconstruction_parameters': {},
        'reconstruction_textured': False})
    response = client.post(f'/api/tasks/{tid}/retry')
    assert response.status_code == 200
    assert response.json()['pipeline_options']['reconstruction_base_url'] == old


def test_upload_validation(client):
    assert client.post('/api/uploads',files={'file':('bad.png',b'not a picture','image/png')}).status_code==400
    content=io.BytesIO();Image.new('RGB',(32,32),'green').save(content,format='PNG')
    upload=client.post('/api/uploads',files={'file':('safe.png',content.getvalue(),'image/png')})
    assert upload.status_code==200
    assert client.get(upload.json()['url']).headers['content-type']=='image/png'
    assert client.get('/api/uploads/not-an-id').status_code==404


def test_provider_preconditions(client):
    assert client.post('/api/tasks',json={'prompt':'Robot','engine':'hunyuan'}).status_code==400
    assert client.post('/api/tasks',json={'prompt':'Robot','image_id':'0'*32}).status_code==400
    parent = client.post('/api/tasks',json={'prompt':'Robot','parent_id':'0'*32})
    assert parent.status_code == 400
    assert '局部编辑' in parent.json()['detail']


def test_origin_and_host_guards(client):
    assert client.post('/api/tasks',json={'prompt':'Robot'},headers={'Origin':'https://foreign.example'}).status_code==403
    assert client.get('/api/health',headers={'Host':'foreign.example'}).status_code==400


def test_asset_download_and_missing_file(client):
    task=client.post('/api/tasks',json={'prompt':'Robot'}).json()
    tid=task['id']
    assert client.get(f'/api/tasks/{tid}/download').status_code==409
    assert client.get(f'/api/tasks/{tid}/files/glb').status_code==404
    assert client.get(f'/api/tasks/{tid}/files/settings').status_code==404
    folder=main.ASSETS/tid
    (folder/'model.glb').write_bytes(b'glTF-test-binary')
    db.update(tid,status='completed')
    envelope=client.get(f'/api/tasks/{tid}/files/glb?transport=json').json()
    assert base64.b64decode(envelope['base64'])==b'glTF-test-binary'
    archive=client.get(f'/api/tasks/{tid}/download?transport=json').json()
    assert base64.b64decode(archive['base64'])[:2]==b'PK'


def test_startup_marks_interrupted(tmp_path,monkeypatch):
    monkeypatch.setattr(db,'DATA',tmp_path)
    monkeypatch.setattr(main,'ROOT',tmp_path)
    (tmp_path/'data').mkdir()
    monkeypatch.setattr(main,'Runner',FakeRunner)
    db.init();db.create({'id':'a'*32,'status':'running'})
    with TestClient(main.app):
        assert db.get('a'*32)['status']=='interrupted'


def test_web_settings_save_persists_even_when_cloud_test_fails(client, tmp_path, monkeypatch):
    monkeypatch.setattr(config, 'SETTINGS', tmp_path/'settings.json')
    monkeypatch.setattr(config, 'DATA', tmp_path)
    monkeypatch.setattr(main, 'public_settings', config.public_settings)
    payload = {'provider': 'openai-compatible', 'base_url': 'https://test.invalid/v1',
               'model': 'test-model', 'api_key': 'settings-test-secret',
               'blender_path': 'D:/blender/blender.exe'}
    headers = {'Origin': 'http://127.0.0.1:8765'}
    saved = client.put('/api/settings', json=payload, headers=headers)
    assert saved.status_code == 200
    assert saved.json()['has_api_key']
    assert 'api_key' not in saved.json()
    assert 'settings-test-secret' not in saved.text
    assert json.loads(config.SETTINGS.read_text(encoding='utf-8'))['api_key'] == payload['api_key']

    def reject_cloud(*args, **kwargs):
        raise ModelError('模型接口返回 HTTP 401。请检查地址、模型名、额度和授权。')

    monkeypatch.setattr(main, 'complete', reject_cloud)
    tested = client.post('/api/settings/test', headers=headers)
    assert tested.status_code == 400
    assert '401' in tested.json()['detail']
    assert client.get('/api/settings').json()['model'] == 'test-model'
    payload.update(api_key='', model='test-model-updated')
    assert client.put('/api/settings', json=payload, headers=headers).status_code == 200
    assert config.settings()['api_key'] == 'settings-test-secret'
    assert client.get('/api/settings').json()['model'] == 'test-model-updated'


def upload_reference(client):
    content = io.BytesIO()
    Image.new('RGB', (512, 512), 'white').save(content, format='PNG')
    response = client.post('/api/uploads', files={'file': ('reference.png', content.getvalue(), 'image/png')})
    assert response.status_code == 200
    return response.json()['id']


def test_text_task_defaults_to_neural_and_records_image_pipeline(client):
    response = client.post('/api/tasks', json={'prompt': 'A compact agricultural robot'})
    assert response.status_code == 202
    task = response.json()
    assert task['engine'] == 'neural'
    assert task['route'] == 'text-to-image-to-3d'
    assert task['pipeline_version'] == 2
    assert task['pipeline_options']['image_model'] == 'fal-ai/qwen-image-2512'
    assert task['pipeline_options']['reconstruction_provider'] == 'fal-hunyuan'
    assert task['id'] in main.app.state.runner.pending


def test_uploaded_image_can_be_submitted_without_text_or_image_generation(client, available_settings):
    image_id = upload_reference(client)
    available_settings.update(image_generation_available=False, has_image_api_key=False)
    response = client.post('/api/tasks', json={'image_id': image_id, 'prompt': '   '})
    assert response.status_code == 202
    task = response.json()
    assert task['route'] == 'image-to-3d' and task['engine'] == 'neural'
    assert task['image_id'] == image_id and '参考图片' in task['prompt']
    repeated = client.post('/api/tasks', json={'image_id': image_id})
    assert repeated.status_code == 202


@pytest.mark.parametrize('prompt', ['', ' ', 'x'])
def test_task_without_useful_text_or_picture_is_rejected(client, prompt):
    response = client.post('/api/tasks', json={'prompt': prompt})
    assert response.status_code == 422
    assert client.get('/api/tasks').json() == []
    assert not main.app.state.runner.pending


@pytest.mark.parametrize('with_image', [False, True])
def test_reconstruction_configuration_required_for_both_routes(client, available_settings, with_image):
    payload = {'image_id': upload_reference(client)} if with_image else {'prompt': 'A green robot'}
    available_settings.update(generation_available=False, has_generative_api_key=False)
    response = client.post('/api/tasks', json=payload)
    assert response.status_code == 400 and '3D' in response.json()['detail']
    assert not main.app.state.runner.pending
    assert client.get('/api/tasks').json() == []


def test_text_route_requires_image_generation_configuration(client, available_settings):
    available_settings['image_generation_available'] = False
    response = client.post('/api/tasks', json={'prompt': 'A green robot'})
    assert response.status_code == 400 and '文生图' in response.json()['detail']
    assert client.get('/api/tasks').json() == []


@pytest.mark.parametrize('engine', ['procedural', 'hunyuan', 'triposr'])
def test_new_tasks_cannot_silently_use_legacy_procedural_route(client, engine):
    response = client.post('/api/tasks', json={'prompt': 'A green robot', 'engine': engine})
    assert response.status_code == 400 and '图片重建' in response.json()['detail']
    assert not main.app.state.runner.pending
    assert client.get('/api/tasks').json() == []


def test_old_procedural_assets_remain_readable_and_downloadable(client):
    tid = 'a' * 32
    folder = main.ASSETS / tid
    folder.mkdir()
    (folder / 'model.glb').write_bytes(b'old-procedural-glb')
    db.create({'id': tid, 'status': 'completed', 'engine': 'procedural', 'title': 'Legacy model'})
    assert client.get(f'/api/tasks/{tid}').json()['engine'] == 'procedural'
    archive = client.get(f'/api/tasks/{tid}/download?transport=json')
    assert archive.status_code == 200
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(archive.json()['base64']))) as bundle:
        assert bundle.read('model.glb') == b'old-procedural-glb'


def test_needs_review_assets_include_visual_evidence_and_end_sse(client):
    task = client.post('/api/tasks', json={'prompt': 'A green robot'}).json()
    tid = task['id']
    folder = main.ASSETS / tid
    (folder / 'model.glb').write_bytes(b'glTF-review-model')
    (folder / 'visual-review.json').write_text(json.dumps({'passed': False, 'issues': ['Wrong wheel count']}), encoding='utf-8')
    for name in ('reference_front.png', 'reference_left.png', 'reference_back.png',
                 'reference_sheet.png', 'render_front.png', 'render_left.png', 'render_back.png'):
        Image.new('RGB', (32, 32), 'white').save(folder / name)
    first_seq = db.events(tid)[-1]['seq']
    db.update(tid, status='needs_review', stage='review', progress=100)
    db.event(tid, 'review', '请检查生成结果')
    evidence = client.get(f'/api/tasks/{tid}/files/visual_review')
    assert evidence.status_code == 200 and evidence.json()['passed'] is False
    for view in ('reference_front', 'reference_left', 'reference_back', 'reference_sheet',
                 'render_front', 'render_left', 'render_back'):
        response = client.get(f'/api/tasks/{tid}/files/{view}')
        assert response.status_code == 200 and response.headers['content-type'] == 'image/png'
    archive = client.get(f'/api/tasks/{tid}/download?transport=json')
    assert archive.status_code == 200
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(archive.json()['base64']))) as bundle:
        assert bundle.read('model.glb') == b'glTF-review-model'
        assert 'visual-review.json' in bundle.namelist()
        assert 'reference_front.png' in bundle.namelist()
    events = client.get(f'/api/tasks/{tid}/events?after={first_seq}')
    assert events.status_code == 200 and 'text/event-stream' in events.headers['content-type']
    assert 'event: progress' in events.text and 'event: task' in events.text
    assert 'needs_review' in events.text
    assert '任务已加入生成队列' not in events.text
    assert ': heartbeat' not in events.text


def test_idempotency_keeps_route_and_snapshot_of_configured_image_model(client, available_settings):
    available_settings['image_model'] = 'fal-ai/first-image-model'
    payload = {'prompt': 'A green robot'}
    first = client.post('/api/tasks', json=payload, headers={'Idempotency-Key': 'config-snapshot'})
    available_settings['image_model'] = 'fal-ai/second-image-model'
    repeated = client.post('/api/tasks', json=payload, headers={'Idempotency-Key': 'config-snapshot'})
    assert repeated.status_code == 202 and repeated.json()['id'] == first.json()['id']
    assert repeated.json()['pipeline_options']['image_model'] == 'fal-ai/first-image-model'
    conflict = client.post('/api/tasks', json={**payload, 'engine': 'procedural'}, headers={'Idempotency-Key': 'config-snapshot'})
    assert conflict.status_code == 409


def test_web_settings_can_switch_image_model_and_preserve_three_keys(client, monkeypatch):
    monkeypatch.setattr(main, 'public_settings', config.public_settings)
    payload = {'provider': 'openai-compatible', 'base_url': 'https://planner.example/v1',
               'model': 'planner-model', 'api_key': 'planner-test-secret',
               'image_provider': 'fal', 'image_model': 'fal-ai/qwen-image-2512',
               'image_api_key': 'image-test-secret', 'generative_api_key': 'reconstruction-test-secret'}
    saved = client.put('/api/settings', json=payload)
    assert saved.status_code == 200
    for secret in ('planner-test-secret', 'image-test-secret', 'reconstruction-test-secret'):
        assert secret not in saved.text and secret not in client.get('/api/settings').text
    for key in ('api_key', 'image_api_key', 'generative_api_key'):
        assert key not in saved.json()
    payload.update(api_key='', image_api_key=' ', generative_api_key='', image_model='fal-ai/alternative-image-model',
                   image_parameters={'num_inference_steps': 12})
    assert client.put('/api/settings', json=payload).status_code == 200
    stored = config.settings()
    assert stored['api_key'] == 'planner-test-secret'
    assert stored['image_api_key'] == 'image-test-secret'
    assert stored['generative_api_key'] == 'reconstruction-test-secret'
    assert stored['image_model'] == 'fal-ai/alternative-image-model'
    payload.update(image_provider='openai-compatible', image_model='another-provider-model',
                   image_base_url='https://images.example/v1', image_parameters={'size': '1536x1024'})
    switched = client.put('/api/settings', json=payload)
    assert switched.status_code == 200
    assert switched.json()['image_model'] == 'another-provider-model'
    assert not switched.json()['image_generation_available']
    payload['image_api_key'] = 'new-provider-secret'
    switched = client.put('/api/settings', json=payload)
    assert switched.json()['image_generation_available']
    assert switched.json()['image_parameters'] == {'size': '1536x1024'}
    # A caller updating only the original planner fields must not reset the new image settings.
    assert client.put('/api/settings', json={'base_url': payload['base_url'], 'model': 'planner-2'}).status_code == 200
    assert config.settings()['image_model'] == 'another-provider-model'


@pytest.mark.parametrize('image_fields', [
    {'image_provider': 'unsupported'},
    {'image_model': 'https://foreign.example/model'},
    {'image_model': 'fal-ai/../model'},
    {'image_base_url': 'https://username:password@images.example/v1'},
    {'image_parameters': {'headers': {'Authorization': 'test-secret'}}},
])
def test_invalid_image_service_settings_do_not_persist(client, image_fields):
    response = client.put('/api/settings', json={'base_url': 'https://planner.example/v1',
                                                'model': 'planner', **image_fields})
    assert response.status_code == 400
    assert not config.SETTINGS.exists()


def test_settings_mutation_is_blocked_during_generation(client):
    assert client.post('/api/tasks', json={'prompt': 'A green robot'}).status_code == 202
    response = client.put('/api/settings', json={'base_url': 'https://planner.example/v1', 'model': 'planner'})
    assert response.status_code == 409
    assert not config.SETTINGS.exists()
