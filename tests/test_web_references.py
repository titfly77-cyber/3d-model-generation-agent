from pathlib import Path

from PIL import Image

from backend import web_references


def _candidate(index):
    return {'title': f'Exact model {index}', 'image_url': f'https://upload.wikimedia.org/example-{index}.png',
            'page_url': f'https://commons.wikimedia.org/wiki/File:example-{index}.png'}


def test_selects_only_verified_reference_from_multiple_candidates(monkeypatch, tmp_path):
    monkeypatch.setattr(web_references, '_search', lambda client, query: [_candidate(i) for i in range(1, 5)])
    def download(client, candidate, path):
        Image.new('RGB', (300, 300), 'orange').save(path)
        return {**candidate, 'path': str(path), 'sha256': 'fixture-' + candidate['title']}
    monkeypatch.setattr(web_references, '_download', download)
    seen = {}
    def choose(board, candidates):
        seen['board'] = board
        assert len(candidates) == 4 and board.exists()
        return {'use_reference': True, 'candidate_index': 2, 'confidence': 92, 'reason': '同一真实型号。'}
    result = web_references.discover('Exact Model', tmp_path, {'web_reference_enabled': True}, choose, lambda: None)
    assert result['used'] and result['status'] == 'selected'
    assert result['candidate']['title'] == 'Exact model 2'
    assert Path(result['path']).exists() and Path(result['board']).exists()
    assert seen['board'].is_file()


def test_does_not_use_candidate_when_visual_match_is_not_confident(monkeypatch, tmp_path):
    monkeypatch.setattr(web_references, '_search', lambda client, query: [_candidate(i) for i in range(1, 4)])
    def download(client, candidate, path):
        Image.new('RGB', (300, 300), 'orange').save(path)
        return {**candidate, 'path': str(path), 'sha256': 'fixture'}
    monkeypatch.setattr(web_references, '_download', download)
    result = web_references.discover('Exact Model', tmp_path, {'web_reference_enabled': True},
                                     lambda board, candidates: {'use_reference': True, 'candidate_index': 1,
                                                                'confidence': 65, 'reason': 'Only similar.'}, lambda: None)
    assert not result['used'] and result['status'] == 'not_used'
    assert '80' in result['reason']


def test_disabled_search_never_makes_network_call(monkeypatch, tmp_path):
    monkeypatch.setattr(web_references, '_search', lambda *args: (_ for _ in ()).throw(AssertionError('must not search')))
    result = web_references.discover('Exact Model', tmp_path, {'web_reference_enabled': False}, lambda *_: {}, lambda: None)
    assert result['status'] == 'disabled' and not result['used']


def test_search_rotates_through_three_sources_after_403(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(web_references, '_search', lambda client, query: (_ for _ in ()).throw(
        web_references.WebReferenceError('公开图片检索返回 HTTP 403。')))
    def route(client, query, api_url, route_type):
        calls.append(api_url)
        if len(calls) == 1:
            raise web_references.WebReferenceError('公开图片检索返回 HTTP 403。')
        return [_candidate(i) for i in range(1, 4)]
    monkeypatch.setattr(web_references, '_search_route', route)
    def download(client, candidate, path):
        Image.new('RGB', (300, 300), 'orange').save(path)
        return {**candidate, 'path': str(path), 'sha256': 'fixture'}
    monkeypatch.setattr(web_references, '_download', download)
    result = web_references.discover(
        'Exact Model', tmp_path, {'web_reference_enabled': True},
        lambda board, candidates: {'use_reference': False, 'confidence': 0, 'reason': 'fixture'},
        lambda: None)
    assert len(result['search_attempts']) == 3
    assert [item['status'] for item in result['search_attempts']] == ['failed', 'failed', 'ok']
    assert result['candidate_count'] == 3
