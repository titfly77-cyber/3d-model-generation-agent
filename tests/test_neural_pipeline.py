"""Real LangGraph/checkpoint/GLB tests with scripted paid services and rendering.

Cloud generation, LLM and Blender execution are replaced by deterministic
fixtures; network APIs and GPU model inference are never called by these tests.
"""
import json
import shutil
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
import trimesh
from PIL import Image, ImageDraw

from backend import agent, db, neural_pipeline as pipeline
from backend.llm import ModelError


def reference_picture(path, color='#537891'):
    picture = Image.new('RGB', (320, 384), 'white')
    draw = ImageDraw.Draw(picture)
    draw.rounded_rectangle((80, 40, 240, 340), radius=20, fill=color)
    draw.ellipse((112, 80, 152, 120), fill='#222222')
    picture.save(path)


def reference_sheet(path, invalid=False):
    if invalid:
        sheet = Image.new('RGB', (960, 384), 'white')
        ImageDraw.Draw(sheet).rounded_rectangle((30, 50, 930, 330), radius=30, fill='#537891')
        sheet.save(path)
        return
    sheet = Image.new('RGB', (960, 384), 'white')
    draw = ImageDraw.Draw(sheet)
    for index in range(3):
        x = 320 * index
        draw.rounded_rectangle((x+80, 40, x+240, 340), radius=20, fill='#537891')
        draw.ellipse((x+112, 80, x+152, 120), fill='#222222')
    sheet.save(path)


def verdict(passed=True, score=90, summary='fixture visual review'):
    return {'passed': passed, 'score': score, 'summary': summary,
            'issues': [] if passed else ['fixture mismatch']}


def test_split_accepts_grey_background_and_shifted_gutters(tmp_path):
    sheet = Image.new('RGB', (960, 384), (215, 215, 215))
    draw = ImageDraw.Draw(sheet)
    for bounds in [(40, 60, 335, 325), (380, 60, 660, 325), (715, 60, 925, 325)]:
        draw.rectangle(bounds, fill=(40, 50, 65))
    path = tmp_path / 'sheet.png'
    sheet.save(path)
    views = pipeline.split_sheet(path, tmp_path)
    assert 335 < views['front']['crop'][2] < 380
    assert 660 < views['left']['crop'][2] < 715
    assert views['front']['crop'][2] == views['left']['crop'][0]


def test_split_rejects_foreground_crossing_entire_gutter_search(tmp_path):
    sheet = Image.new('RGB', (960, 384), 'white')
    ImageDraw.Draw(sheet).rectangle((30, 80, 930, 300), fill='black')
    path = tmp_path / 'sheet.png'
    sheet.save(path)
    with pytest.raises(ModelError, match='截断主体|背景过暗'):
        pipeline.split_sheet(path, tmp_path)


@pytest.fixture
def harness(tmp_path, monkeypatch):
    assets, uploads = tmp_path/'assets', tmp_path/'uploads'
    assets.mkdir()
    uploads.mkdir()
    monkeypatch.setattr(db, 'DATA', tmp_path)
    monkeypatch.setattr(agent, 'DATA', tmp_path)
    monkeypatch.setattr(agent, 'ASSETS', assets)
    monkeypatch.setattr(agent, 'UPLOADS', uploads)
    cfg = {'model': 'scripted-test-language-model', 'blender_path': 'scripted-blender',
           'visual_review_enabled': True, 'max_quality_retries': 1,
           'web_reference_enabled': False,
           'image_model': 'scripted-image-model', 'reconstruction_provider': 'fal-hunyuan',
           'generative_api_key': 'never-used-for-network'}
    monkeypatch.setattr(agent, 'settings', lambda: dict(cfg))
    db.init()
    h = SimpleNamespace(assets=assets, uploads=uploads, cfg=cfg,
        analysis_calls=[], image_calls=[], mesh_calls=[], process_calls=[],
        reference_review_calls=[], mesh_review_calls=[], reference_verdicts=[], mesh_verdicts=[],
        invalid_sheet_attempts=set(), process_failures={}, reconstruction_failures={},
        cancel_after_analysis=False, task_id=None, runner=None)

    def complete(system, prompt, image_path=None, **kwargs):
        if system.startswith('Plan a 3D asset.'):
            h.analysis_calls.append(prompt)
            if h.cancel_after_analysis:
                h.cancel_after_analysis = False
                db.update(h.task_id, cancel_requested=True)
            response = {'title': 'Scripted asset', 'summary': 'A test object',
                        'requirements': ['one object', 'blue body'],
                        'optimized_prompt': 'One blue device with a dark circular feature.'}
        else:
            assert image_path and Path(image_path).is_file()
            if 'first row' in prompt:
                h.mesh_review_calls.append(prompt)
                index = len(h.mesh_review_calls)-1
                response = h.mesh_verdicts[index] if index < len(h.mesh_verdicts) else verdict()
            else:
                h.reference_review_calls.append(prompt)
                index = len(h.reference_review_calls)-1
                response = h.reference_verdicts[index] if index < len(h.reference_verdicts) else verdict()
            if isinstance(response, Exception):
                raise response
        return json.dumps(response), {'input_tokens': 1, 'output_tokens': 1}

    def generate_sheet(prompt, output, provider_cfg, check_cancel, progress=None, job_path=None, reference_image=None):
        check_cancel()
        assert job_path and Path(job_path).parent == output.parent
        if not output.exists():
            h.image_calls.append({'prompt': prompt, 'output': output, 'cfg': provider_cfg, 'reference_image': reference_image})
            reference_sheet(output, invalid=int(output.parent.name) in h.invalid_sheet_attempts)
        return {'model': provider_cfg['image_model'], 'request_id': 'image-fixture'}

    def reconstruct_views(views, output, provider_cfg, check_cancel, progress=None, job_path=None):
        check_cancel()
        attempt = int(output.parent.name)
        if h.reconstruction_failures.get(attempt, 0):
            h.reconstruction_failures[attempt] -= 1
            raise ModelError('scripted cloud reconstruction unavailable')
        if not output.exists():
            h.mesh_calls.append({'views': dict(views), 'output': output, 'seed': provider_cfg['reconstruction_seed']})
            mesh = trimesh.creation.box(extents=[1 + attempt * .1, .4, 2])
            output.write_bytes(mesh.export(file_type='glb'))
        return {'provider': 'scripted', 'model': 'scripted-multiview' if len(views) == 3 else 'scripted-single',
                'used_views': list(views), 'seed': provider_cfg['reconstruction_seed'],
                'request_id': f'fixture-mesh-{attempt}', 'textured': True}

    def process_command(tid, command, logfile, **kwargs):
        output = Path(command[command.index('--output')+1])
        assert '--review-views' in command
        assert '--import-model' in command
        attempt = int(output.name)
        h.process_calls.append(output)
        if h.process_failures.get(attempt, 0):
            h.process_failures[attempt] -= 1
            raise ModelError('scripted Blender interruption')
        shutil.copy2(output/'raw.glb', output/'model.glb')
        geometry = trimesh.load(output/'model.glb', force='scene')
        report = {'objects': len(geometry.geometry),
                  'vertices': sum(len(m.vertices) for m in geometry.geometry.values()),
                  'triangles': sum(len(m.faces) for m in geometry.geometry.values()),
                  'checks': [], 'warnings': []}
        agent.write_json(output/'report.json', report)
        for view in pipeline.VIEWS:
            reference_picture(output/f'render_{view}.png')
        reference_picture(output/'preview.png')

    monkeypatch.setattr(pipeline, 'complete', complete)
    monkeypatch.setattr(pipeline, 'generate_sheet', generate_sheet)
    monkeypatch.setattr(pipeline, 'reconstruct_views', reconstruct_views)
    monkeypatch.setattr(agent, 'process_command', process_command)
    monkeypatch.setattr(agent, 'complete', lambda *args, **kwargs: pytest.fail('New task incorrectly entered procedural LLM path'))

    def create(image=False, options=None):
        tid = uuid.uuid4().hex
        (assets/tid).mkdir()
        image_id = uuid.uuid4().hex if image else None
        if image_id:
            reference_picture(uploads/(image_id+'.png'))
        frozen = {key: cfg[key] for key in ('image_model', 'visual_review_enabled', 'max_quality_retries')}
        frozen.update(options or {})
        db.create({'id': tid, 'status': 'queued', 'prompt': 'Make one blue device.', 'style': 'industrial',
                   'quality': 'draft', 'engine': 'neural', 'image_id': image_id,
                   'cancel_requested': False, 'pipeline_options': frozen})
        h.task_id = tid
        h.runner = agent.Runner()
        return tid
    h.create = create
    h.run = lambda: h.runner._run(h.task_id)
    h.result = lambda: db.get(h.task_id)
    yield h
    if h.runner:
        h.runner.close()
        h.runner.connection.close()


@pytest.mark.parametrize('image_input', [False, True])
def test_text_and_image_route_into_correct_native_reconstruction(harness, image_input):
    h = harness
    tid = h.create(image=image_input)
    h.run()
    result = h.result()
    assert result['status'] == 'completed', result.get('error')
    assert len(h.analysis_calls) == 1
    assert len(h.image_calls) == (0 if image_input else 1)
    assert len(h.mesh_calls) == 1
    assert list(h.mesh_calls[0]['views']) == (['front'] if image_input else ['front', 'left', 'back'])
    assert result['report']['verified']
    assert result['report']['used_views'] == list(h.mesh_calls[0]['views'])
    assert (h.assets/tid/'model.glb').read_bytes().startswith(b'glTF')
    if image_input:
        assert (h.assets/tid/'reference_front.png').read_bytes() == (h.uploads/(result['image_id']+'.png')).read_bytes()
    else:
        assert all(p.name != 'sheet.png' for p in h.mesh_calls[0]['views'].values())


def test_bad_sheet_gets_one_repair_before_any_reconstruction(harness):
    h = harness
    h.invalid_sheet_attempts = {1}
    h.create()
    h.run()
    assert h.result()['status'] == 'completed', h.result().get('error')
    assert len(h.image_calls) == 2 and len(h.mesh_calls) == 1
    assert 'Correct these problems' in h.image_calls[1]['prompt']
    assert all(p.parent.name == '2' for p in h.mesh_calls[0]['views'].values())


def test_verified_web_reference_is_injected_into_supported_image_model(harness, monkeypatch):
    h = harness
    h.cfg.update(web_reference_enabled=True, image_provider='tencent-tokenhub', image_model='hy-image-v3.5-preview')
    def discovered(query, output_dir, config, choose, check_cancel, progress):
        reference = Path(output_dir) / 'selected.png'
        reference.parent.mkdir(parents=True, exist_ok=True)
        reference_picture(reference, '#ef8b2c')
        return {'status': 'selected', 'used': True, 'path': str(reference), 'candidate_count': 3,
                'confidence': 94, 'reason': '已核验', 'candidate': {'title': 'Exact fixture',
                'page_url': 'https://commons.wikimedia.org/wiki/File:fixture'}}
    monkeypatch.setattr(pipeline, 'discover_web_reference', discovered)
    h.create()
    h.run()
    result = h.result()
    assert result['status'] == 'completed', result.get('error')
    assert h.image_calls[0]['reference_image'].name == 'selected.png'
    assert 'REFERENCE IMAGE:' in h.image_calls[0]['prompt']
    assert result['web_reference']['used'] is True


def test_twice_invalid_reference_stops_before_paid_3d(harness):
    h = harness
    h.invalid_sheet_attempts = {1, 2, 3}
    h.create()
    h.run()
    assert h.result()['status'] == 'failed'
    assert len(h.image_calls) == 2 and not h.mesh_calls
    assert '参考图检查未通过' in h.result()['error']


def test_user_can_explicitly_continue_after_reference_validation_failure(harness):
    h = harness
    h.invalid_sheet_attempts = {1, 2, 3}
    tid = h.create()
    h.run()
    stopped = h.result()
    assert stopped['status'] == 'failed'
    assert stopped['can_proceed_unverified_reference'] is True
    assert not h.mesh_calls
    db.update(tid, status='queued', error=None, allow_unverified_reference=True)
    h.run()
    result = h.result()
    assert result['status'] == 'needs_review', result.get('error')
    assert len(h.mesh_calls) == 1
    assert result['reference_override_accepted'] is True
    assert result['report']['reference_override_accepted'] is True
    assert any('使用者已明确选择继续生成 3D' in item for item in result['report']['warnings'])


def test_uploaded_image_is_not_regenerated_when_reference_review_fails(harness):
    h = harness
    h.reference_verdicts = [verdict(False, 20)]
    h.create(image=True)
    h.run()
    assert h.result()['status'] == 'failed'
    assert not h.image_calls and not h.mesh_calls


def test_one_mesh_retry_selects_passing_candidate_and_keeps_both(harness):
    h = harness
    h.mesh_verdicts = [verdict(False, 80), verdict(True, 75)]
    tid = h.create()
    h.run()
    assert h.result()['status'] == 'completed', h.result().get('error')
    assert len(h.image_calls) == 1 and len(h.mesh_calls) == 2
    assert [call['seed'] for call in h.mesh_calls] == [42, 43]
    assert h.result()['selected_candidate'] == 2
    assert (h.assets/tid/'model.glb').read_bytes() == (h.assets/tid/'candidates/2/model.glb').read_bytes()
    assert (h.assets/tid/'candidates/1/model.glb').exists()


def test_failed_visual_candidates_deliver_the_better_first_version_as_needs_review(harness):
    h = harness
    h.mesh_verdicts = [verdict(False, 80), verdict(False, 40), verdict(True, 95)]
    tid = h.create()
    h.run()
    result = h.result()
    assert result['status'] == 'needs_review', result.get('error')
    assert result['selected_candidate'] == 1
    assert len(h.mesh_calls) == 2 and len(h.mesh_review_calls) == 2
    assert (h.assets/tid/'model.glb').read_bytes() == (h.assets/tid/'candidates/1/model.glb').read_bytes()
    assert result['report']['verified'] is True
    assert any(c['label'] == '外观一致性复核' and c['passed'] is False for c in result['report']['checks'])


def test_disabled_review_does_not_claim_visual_acceptance_or_create_candidates(harness):
    h = harness
    h.create(options={'visual_review_enabled': False})
    h.run()
    assert h.result()['status'] == 'needs_review'
    assert len(h.mesh_calls) == 1
    assert not h.reference_review_calls and not h.mesh_review_calls
    assert h.result()['visual_review']['status'] == 'skipped'


def test_zero_retry_budget_delivers_first_candidate(harness):
    h = harness
    h.mesh_verdicts = [verdict(False, 50)]
    h.create(options={'max_quality_retries': 0})
    h.run()
    assert h.result()['status'] == 'needs_review'
    assert len(h.image_calls) == len(h.mesh_calls) == 1


def test_unavailable_vision_api_never_claims_success(harness):
    h = harness
    h.reference_verdicts = [ModelError('vision not supported')]
    h.mesh_verdicts = [ModelError('vision not supported')]
    h.create()
    h.run()
    assert h.result()['status'] == 'needs_review'
    assert h.result()['visual_review']['status'] == 'unavailable'
    assert len(h.mesh_calls) == 1


def test_checkpoint_after_cloud_before_blender_resumes_without_generation(harness):
    h = harness
    h.process_failures = {1: 1}
    tid = h.create()
    h.run()
    assert h.result()['status'] == 'failed'
    snapshot = h.runner.neural_graph.get_state({'configurable': {'thread_id': tid}})
    assert snapshot.next == ('generate',)
    assert len(h.image_calls) == len(h.mesh_calls) == 1
    h.run()
    assert h.result()['status'] == 'completed', h.result().get('error')
    assert len(h.image_calls) == len(h.mesh_calls) == len(h.analysis_calls) == 1
    assert len(h.process_calls) == 2


def test_cancel_between_analysis_and_paid_calls_then_resume(harness):
    h = harness
    h.cancel_after_analysis = True
    tid = h.create()
    h.run()
    assert h.result()['status'] == 'cancelled'
    assert not h.image_calls and not h.mesh_calls
    db.update(tid, cancel_requested=False, status='queued')
    h.run()
    assert h.result()['status'] == 'completed'
    assert len(h.analysis_calls) == len(h.image_calls) == len(h.mesh_calls) == 1


@pytest.mark.parametrize('vision_enabled', [True, False])
def test_replaying_delivered_graph_cannot_charge_for_new_generation(harness, vision_enabled):
    h = harness
    h.create(options={'visual_review_enabled': vision_enabled})
    h.run()
    expected_status = 'completed' if vision_enabled else 'needs_review'
    assert h.result()['status'] == expected_status
    counts = (len(h.analysis_calls), len(h.image_calls), len(h.mesh_calls))
    h.run()
    assert h.result()['status'] == expected_status
    assert (len(h.analysis_calls), len(h.image_calls), len(h.mesh_calls)) == counts


@pytest.mark.parametrize('failing_stage', ['reconstruction', 'blender'])
def test_optional_second_candidate_failure_still_delivers_valid_first_candidate(harness, failing_stage):
    h = harness
    h.mesh_verdicts = [verdict(False, 70)]
    if failing_stage == 'reconstruction':
        h.reconstruction_failures = {2: 1}
    else:
        h.process_failures = {2: 1}
    tid = h.create()
    h.run()
    result = h.result()
    assert result['status'] == 'needs_review', result.get('error')
    assert result['selected_candidate'] == 1
    assert (h.assets/tid/'model.glb').read_bytes() == (h.assets/tid/'candidates/1/model.glb').read_bytes()


def test_uniform_colored_image_is_empty_for_reconstruction(tmp_path):
    path = tmp_path/'uniform-red.png'
    Image.new('RGB', (256, 256), 'red').save(path)
    with pytest.raises(ModelError, match='空白'):
        pipeline.image_check(path)
