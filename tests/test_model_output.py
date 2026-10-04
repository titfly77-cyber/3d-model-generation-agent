import json

import pytest

from backend import agent, db, llm
from backend.examples import chair


@pytest.mark.parametrize('base,json_mode,expected_options', [
    ('https://api.deepseek.com', True, {'thinking': {'type': 'disabled'}, 'response_format': {'type': 'json_object'}}),
    ('https://api.deepseek.com/v1', False, {'thinking': {'type': 'disabled'}}),
    ('https://another-provider.invalid/v1', True, {}),
    ('https://api.deepseek.com.other.invalid', True, {}),
])
def test_deepseek_options_are_scoped_to_its_official_endpoint(monkeypatch, base, json_mode, expected_options):
    requests = []
    class Response:
        status_code = 200
        def json(self):
            return {'choices': [{'message': {'content': '{"ok":true}'}, 'finish_reason': 'stop'}]}
    class Client:
        def __init__(self, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def post(self, url, **kwargs):
            requests.append(kwargs['json'])
            return Response()
    monkeypatch.setattr(llm.httpx, 'Client', Client)
    output, usage = llm.complete('Return JSON.', 'Test', json_output=json_mode,
        config={'provider': 'openai-compatible', 'base_url': base, 'api_key': 'fake-test-key', 'model': 'deepseek-flash'})
    assert json.loads(output) == {'ok': True}
    assert {k: requests[0][k] for k in ('thinking', 'response_format') if k in requests[0]} == expected_options


def test_thinking_only_response_reports_exhausted_budget_without_exposing_reasoning():
    body = {'choices': [{'finish_reason': 'length', 'message': {'content': '', 'reasoning_content': 'private-test-reasoning'}}],
            'usage': {'completion_tokens': 14000}}
    with pytest.raises(llm.ModelOutputError) as caught:
        llm.final_response(body, 'openai-compatible')
    exc = caught.value
    assert exc.code == 'output_limit'
    assert exc.usage['completion_tokens'] == 14000
    assert exc.usage['_response']['final_characters'] == 0
    assert '思考内容耗尽' in str(exc)
    assert 'private-test-reasoning' not in str(exc) + repr(exc.usage) + exc.output


@pytest.mark.parametrize('provider,body', [
    ('openai-compatible', {'choices': [{'finish_reason': 'length', 'message': {'content': '{"parts":['}}]}),
    ('anthropic', {'stop_reason': 'max_tokens', 'content': [{'type': 'text', 'text': '{"parts":['}]}),
])
def test_truncated_output_rejected_before_scene_parsing(provider, body):
    with pytest.raises(llm.ModelOutputError) as caught:
        llm.final_response(body, provider)
    assert caught.value.code == 'output_limit'
    assert caught.value.output == '{"parts":['


def test_empty_final_output_is_not_replaced_by_reasoning():
    with pytest.raises(llm.ModelOutputError) as caught:
        llm.final_response({'choices': [{'finish_reason': 'stop', 'message': {'content': None, 'reasoning_content': '{"not_a_scene":true}'}}]}, 'openai-compatible')
    assert caught.value.code == 'empty_output'
    output, usage = llm.final_response({'choices': [{'finish_reason': 'stop', 'message': {'content': '{"ok":true}', 'reasoning_content': 'ignored'}}]}, 'openai-compatible')
    assert output == '{"ok":true}'


@pytest.fixture
def design_task(tmp_path, monkeypatch):
    monkeypatch.setattr(db, 'DATA', tmp_path)
    monkeypatch.setattr(agent, 'ASSETS', tmp_path)
    monkeypatch.setattr(agent, 'settings', lambda: {'model': 'test-provider'})
    db.init()
    tid = 'd' * 32
    (tmp_path / tid).mkdir()
    db.create({'id': tid, 'status': 'running', 'engine': 'procedural', 'prompt': 'A wooden chair', 'cancel_requested': False})
    return {'task_id': tid, 'analysis': {'requirements': ['chair']}}, tmp_path / tid


def test_scene_retry_uses_specific_validation_errors(design_task, monkeypatch):
    state, folder = design_task
    prompts = []
    invalid = chair()
    invalid['parts'][0]['scale'] = [0, 1, 1]
    def complete(system, prompt, *args, **kwargs):
        prompts.append(prompt)
        assert kwargs['json_output']
        return json.dumps(invalid if len(prompts) == 1 else chair()), {'completion_tokens': 50}
    monkeypatch.setattr(agent, 'complete', complete)
    scene = agent.design(state)['scene']
    assert len(scene['parts']) == len(chair()['parts'])
    assert 'Scale must be positive' in prompts[1]
    assert (folder / 'design.json').exists()
    diagnostics = [json.loads(p.read_text(encoding='utf-8')) for p in sorted((folder/'diagnostics').glob('design-*.json'))]
    assert [d['valid'] for d in diagnostics] == [False, True]
    assert 'input' not in diagnostics[0]['errors'][0]
    assert db.get(state['task_id'])['usage']['completion_tokens'] == 100


def test_output_limit_retry_preserves_usage_and_recovers(design_task, monkeypatch):
    state, folder = design_task
    prompts = []
    def complete(system, prompt, *args, **kwargs):
        prompts.append(prompt)
        if len(prompts) == 1:
            raise llm.ModelOutputError('模型输出达到长度上限。', code='output_limit', usage={'completion_tokens': 14000})
        return json.dumps(chair()), {'completion_tokens': 500}
    monkeypatch.setattr(agent, 'complete', complete)
    assert agent.design(state)['scene']['parts']
    assert 'complete compact JSON' in prompts[1]
    assert db.get(state['task_id'])['usage']['completion_tokens'] == 14500


def test_repeated_output_failure_has_actionable_diagnostics(design_task, monkeypatch):
    state, folder = design_task
    def complete(*args, **kwargs):
        raise llm.ModelOutputError('模型输出达到长度上限：最终内容未完整输出。', code='output_limit', usage={'completion_tokens': 14000})
    monkeypatch.setattr(agent, 'complete', complete)
    with pytest.raises(llm.ModelError, match='输出达到长度上限'):
        agent.design(state)
    assert not (folder/'design.json').exists()
    assert len(list((folder/'diagnostics').glob('design-*.json'))) == 2
    assert db.get(state['task_id'])['usage']['completion_tokens'] == 28000
    assert any(e.get('detail', {}).get('errors') for e in db.events(state['task_id']) if e.get('detail'))


def test_authentication_failure_is_not_retried_as_bad_geometry(design_task, monkeypatch):
    state, folder = design_task
    calls = []
    def complete(*args, **kwargs):
        calls.append(1)
        raise llm.ModelError('模型接口返回 HTTP 401。')
    monkeypatch.setattr(agent, 'complete', complete)
    with pytest.raises(llm.ModelError, match='401'):
        agent.design(state)
    assert len(calls) == 1
