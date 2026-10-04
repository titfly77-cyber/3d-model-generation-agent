"""Conservative public-image discovery for real, named 3D subjects.

Three Wikimedia-backed search routes are tried in order (Commons, English
Wikipedia, Chinese Wikipedia). They require no user key and expose stable
source pages. A candidate is never used merely because search found it; a
configured vision model must select it from at least three candidates.
"""
import hashlib
import ipaddress
import json
import socket
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from PIL import Image, ImageDraw, ImageOps


COMMONS_API = 'https://commons.wikimedia.org/w/api.php'
EN_WIKIPEDIA_API = 'https://en.wikipedia.org/w/api.php'
ZH_WIKIPEDIA_API = 'https://zh.wikipedia.org/w/api.php'
SEARCH_ROUTES = (
    ('wikimedia-commons', COMMONS_API, 'file'),
    ('english-wikipedia', EN_WIKIPEDIA_API, 'article'),
    ('chinese-wikipedia', ZH_WIKIPEDIA_API, 'article'),
)
MAX_CANDIDATES = 6
MIN_CANDIDATES = 3
MAX_DOWNLOAD_BYTES = 8 * 1024 * 1024
HTTP_HEADERS = {
    'User-Agent': 'FORMA-3D-Studio/1.0 (local reference discovery; contact: local-user)',
    'Accept': 'application/json,image/*;q=0.9,*/*;q=0.5',
}


class WebReferenceError(RuntimeError):
    pass


def _safe_wikimedia_url(value):
    try:
        parsed = urlsplit(value)
        host = parsed.hostname or ''
        valid_host = (host == 'upload.wikimedia.org' or host.endswith('.wikimedia.org')
                      or host.endswith('.wikipedia.org'))
        addresses = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM) if valid_host else []
        if (parsed.scheme != 'https' or not valid_host or parsed.username or parsed.password
                or parsed.port not in (None, 443) or not addresses
                or any(not ipaddress.ip_address(item[4][0]).is_global for item in addresses)):
            raise ValueError()
    except (OSError, ValueError):
        raise WebReferenceError('公开图片来源返回了不安全或无法解析的地址。') from None
    return value


def _search(client, query):
    """Search Commons files. Kept as a small seam for deterministic tests."""
    return _search_route(client, query, COMMONS_API, 'file')


def _search_route(client, query, api_url, route_type):
    if route_type == 'file':
        params = {
        'action': 'query', 'format': 'json', 'generator': 'search', 'gsrnamespace': '6',
        'gsrlimit': str(MAX_CANDIDATES), 'gsrsearch': query, 'prop': 'imageinfo|info',
        'iiprop': 'url', 'iiurlwidth': '900', 'inprop': 'url',
        }
    else:
        params = {
            'action': 'query', 'format': 'json', 'generator': 'search', 'gsrnamespace': '0',
            'gsrlimit': str(MAX_CANDIDATES), 'gsrsearch': query, 'prop': 'pageimages|info',
            'piprop': 'thumbnail|original', 'pithumbsize': '900', 'inprop': 'url',
        }
    response = client.get(api_url, params=params)
    if response.status_code != 200:
        raise WebReferenceError(f'公开图片检索返回 HTTP {response.status_code}。')
    try:
        pages = (response.json().get('query') or {}).get('pages') or {}
    except (ValueError, AttributeError):
        raise WebReferenceError('公开图片检索没有返回有效数据。') from None
    candidates = []
    for page in pages.values():
        info = (page.get('imageinfo') or [{}])[0]
        image = (info.get('thumburl') or info.get('url')) if route_type == 'file' else (
            (page.get('thumbnail') or {}).get('source') or (page.get('original') or {}).get('source'))
        page_url = page.get('canonicalurl')
        if not all(isinstance(value, str) and value for value in (image, page_url)):
            continue
        title = str(page.get('title') or '').removeprefix('File:')[:180]
        try:
            _safe_wikimedia_url(image)
            _safe_wikimedia_url(page_url)
        except WebReferenceError:
            continue
        candidates.append({'title': title, 'image_url': image, 'page_url': page_url})
    return candidates[:MAX_CANDIDATES]


def _download(client, candidate, path):
    from .image_generation import _normalized_image
    response = client.get(candidate['image_url'], follow_redirects=False)
    if response.status_code != 200:
        raise WebReferenceError(f"候选图片下载失败（HTTP {response.status_code}）。")
    raw = response.content
    if not raw or len(raw) > MAX_DOWNLOAD_BYTES:
        raise WebReferenceError('候选图片为空或超过 8MB 限制。')
    image = _normalized_image(raw)
    image.save(path, format='PNG')
    return {**candidate, 'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def _board(candidates, output):
    cell_w, cell_h = 360, 330
    columns = 3
    rows = (len(candidates) + columns - 1) // columns
    board = Image.new('RGB', (cell_w * columns, cell_h * rows), 'white')
    draw = ImageDraw.Draw(board)
    for index, candidate in enumerate(candidates):
        with Image.open(candidate['path']) as source:
            image = ImageOps.contain(source.convert('RGB'), (cell_w - 24, cell_h - 55))
        x, y = (index % columns) * cell_w, (index // columns) * cell_h
        board.paste(image, (x + (cell_w - image.width) // 2, y + 30 + (cell_h - 55 - image.height) // 2))
        draw.text((x + 10, y + 10), f'[{index + 1}] {candidate["title"][:42]}', fill='black')
    board.save(output)


def discover(query, output_dir, config, choose, check_cancel, progress=None):
    """Return a selected, locally saved image only after visual verification.

    ``choose`` receives a contact sheet and metadata, and must return a dict
    with use_reference, candidate_index, confidence, and reason.
    """
    result = {'status': 'not_found', 'source': 'wikimedia', 'query': query,
              'candidate_count': 0, 'used': False, 'search_attempts': []}
    if not config.get('web_reference_enabled', True):
        return {**result, 'status': 'disabled', 'reason': '已在设置中关闭联网图片参考。'}
    if not isinstance(query, str) or len(query.strip()) < 3:
        return {**result, 'status': 'not_found', 'reason': '需求中没有可检索的具体对象名称。'}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = []
    attempts = []
    seen_images = set()
    with httpx.Client(timeout=httpx.Timeout(20, connect=10), follow_redirects=False,
                      headers=HTTP_HEADERS) as client:
        for attempt_number, (source_name, api_url, route_type) in enumerate(SEARCH_ROUTES, 1):
            if progress:
                progress(f'正在检索公开图片参考（来源 {attempt_number}/3）')
            check_cancel()
            try:
                raw = (_search(client, query.strip()[:220]) if attempt_number == 1 else
                       _search_route(client, query.strip()[:220], api_url, route_type))
                downloaded = 0
                for candidate in raw:
                    check_cancel()
                    if candidate['image_url'] in seen_images:
                        continue
                    seen_images.add(candidate['image_url'])
                    try:
                        item = _download(client, candidate, output_dir / f'candidate-{len(candidates) + 1}.png')
                    except (httpx.HTTPError, WebReferenceError):
                        continue
                    item['source'] = source_name
                    candidates.append(item)
                    downloaded += 1
                    if len(candidates) >= MAX_CANDIDATES:
                        break
                attempts.append({'source': source_name, 'status': 'ok', 'candidate_count': downloaded})
            except (httpx.HTTPError, WebReferenceError) as exc:
                attempts.append({'source': source_name, 'status': 'failed', 'reason': str(exc)[:300]})
            if len(candidates) >= MIN_CANDIDATES:
                break
    result['search_attempts'] = attempts
    result['candidate_count'] = len(candidates)
    if len(candidates) < MIN_CANDIDATES:
        failures = [item['reason'] for item in attempts if item['status'] == 'failed']
        reason = ('三个公开图片来源均不可用：' + '；'.join(failures)) if failures and not candidates else (
            '已轮换三个公开图片来源，但未找到至少三张可用候选图，改由模型根据文字生成。')
        return {**result, 'status': 'unavailable' if failures and not candidates else 'not_found', 'reason': reason[:900]}
    board = output_dir / 'candidates.png'
    _board(candidates, board)
    if progress: progress('正在核验公开图片是否与目标对象准确对应')
    verdict = choose(board, candidates)
    if not isinstance(verdict, dict) or not verdict.get('use_reference'):
        return {**result, 'status': 'not_used', 'reason': str((verdict or {}).get('reason') or '候选图片不足以证明是目标对象。')[:500],
                'confidence': float((verdict or {}).get('confidence') or 0)}
    index = verdict.get('candidate_index')
    if not isinstance(index, int) or not 1 <= index <= len(candidates):
        return {**result, 'status': 'not_used', 'reason': '视觉核验未选择有效候选图。'}
    confidence = float(verdict.get('confidence') or 0)
    if confidence < 80:
        return {**result, 'status': 'not_used', 'reason': '候选图匹配度不足 80，改由模型根据文字生成。', 'confidence': confidence}
    selected = candidates[index - 1]
    selected_path = output_dir / 'selected.png'
    Path(selected['path']).replace(selected_path)
    return {**result, 'status': 'selected', 'source': selected.get('source', 'wikimedia'),
            'used': True, 'confidence': confidence,
            'reason': str(verdict.get('reason') or '')[:500], 'candidate': {k: v for k, v in selected.items() if k != 'path'},
            'path': str(selected_path), 'sha256': hashlib.sha256(selected_path.read_bytes()).hexdigest(),
            'board': str(board)}
