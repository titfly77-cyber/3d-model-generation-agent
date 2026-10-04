import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import httpx
from backend.config import settings, save_settings
from backend.llm import endpoint, complete

cfg = settings()
headers = {'Authorization': 'Bearer ' + cfg['api_key'], 'x-api-key': cfg['api_key'], 'anthropic-version': '2023-06-01'}
try:
    r = httpx.get(endpoint(cfg['base_url'], '/models'), headers=headers, timeout=25)
    print('Models endpoint status:', r.status_code)
    if r.status_code == 200:
        ids = [x['id'] for x in r.json().get('data', [])]
        print('Model IDs:', json.dumps(ids[:40]))
        choices = [m for m in ids if 'sonnet' in m.lower()]
        if choices and cfg['model'] not in ids:
            cfg['model'] = next((m for m in choices if '4-6' in m or '4.6' in m), choices[0])
            save_settings({'model': cfg['model']})
except Exception as exc:
    print('Model listing unavailable:', type(exc).__name__)
try:
    text, usage = complete('Reply with exactly OK.', 'Connection test.', max_tokens=32, config=cfg)
    print('Connection:', text[:80], 'Model:', cfg['model'])
except Exception as exc:
    print('Connection failed:', str(exc))
    sys.exit(1)
