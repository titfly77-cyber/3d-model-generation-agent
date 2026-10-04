"""Reproduce one design response, retaining only final output and safe metadata."""
import argparse
import json
from pathlib import Path
import sys
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import httpx
from pydantic import ValidationError
from backend import agent, db
from backend.config import ASSETS, settings
from backend.llm import complete, extract_json
from backend.schema import SceneSpec

parser = argparse.ArgumentParser()
parser.add_argument('task_id')
args = parser.parse_args()
task = db.get(args.task_id)
if not task or task['engine'] != 'procedural':
    raise SystemExit('An existing procedural task is required.')
folder = ASSETS / task['id'] / 'diagnostics'
folder.mkdir(exist_ok=True)
stem = 'probe-' + uuid.uuid4().hex[:12]
meta = {'model': settings()['model'], 'max_tokens': 14000}
original_post = httpx.Client.post


def inspect_response(client, *arguments, **kwargs):
    payload = kwargs.get('json') or {}
    meta['request_options'] = {key: payload[key] for key in ('thinking', 'response_format') if key in payload}
    response = original_post(client, *arguments, **kwargs)
    meta['http_status'] = response.status_code
    if response.status_code == 200:
        data = response.json()
        choice = (data.get('choices') or [{}])[0]
        message = choice.get('message') or {}
        final = message.get('content') or ''
        meta.update(finish_reason=choice.get('finish_reason'),
                    final_characters=len(final),
                    reasoning_characters=len(message.get('reasoning_content') or ''),
                    usage=data.get('usage', {}))
        # Do not store or print reasoning text, credentials, or HTTP headers.
        (folder / (stem + '.txt')).write_text(final, encoding='utf-8')
    return response


httpx.Client.post = inspect_response
try:
    prompt = json.dumps(task['analysis'], ensure_ascii=False) + '\nUser request: ' + task['prompt']
    output, usage = complete(agent.DESIGN_SYSTEM, prompt, max_tokens=14000, json_output=True)
    scene = SceneSpec.model_validate(extract_json(output))
    meta.update(valid=True, parts=len(scene.parts))
except ValidationError as exc:
    meta.update(valid=False, errors=exc.errors(include_input=False, include_context=False, include_url=False))
except Exception as exc:
    meta.update(valid=False, error=str(exc)[:500], error_type=type(exc).__name__)
finally:
    httpx.Client.post = original_post
    (folder / (stem + '.json')).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(meta, ensure_ascii=False))
