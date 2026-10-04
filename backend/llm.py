import base64
import json
import re
import time
from urllib.parse import urlparse
import httpx
from .config import settings, record_connection


class ModelError(Exception):
    pass


class ModelOutputError(ModelError):
    """A completed HTTP request that did not deliver usable final output."""
    def __init__(self, message, *, code, usage=None, output=''):
        super().__init__(message)
        self.code = code
        self.usage = usage or {}
        self.output = output


def final_response(data, provider):
    # Only final content is usable as a scene. Never substitute reasoning text.
    usage = data.get('usage') if isinstance(data.get('usage'), dict) else {}
    usage = {key: usage[key] for key in ('input_tokens', 'output_tokens', 'prompt_tokens', 'completion_tokens')
             if isinstance(usage.get(key), int)}
    if provider == 'anthropic':
        blocks = data.get('content') or []
        output = '\n'.join(c.get('text', '') for c in blocks if c.get('type') == 'text')
        reason = data.get('stop_reason')
        reasoning_chars = 0
    else:
        choices = data.get('choices') or []
        if not choices:
            raise ModelOutputError('模型接口没有返回候选结果，请检查接口协议。', code='missing_choices', usage=usage)
        choice = choices[0]
        message = choice.get('message') or {}
        output = message.get('content') or ''
        if isinstance(output, list):
            output = '\n'.join(c.get('text', '') for c in output if c.get('type') == 'text')
        reason = choice.get('finish_reason')
        reasoning_chars = len(message.get('reasoning_content') or '')
    if not isinstance(output, str):
        raise ModelOutputError('模型接口的最终内容格式不正确。', code='invalid_content', usage=usage)
    usage['_response'] = {'finish_reason': reason, 'final_characters': len(output),
                          'reasoning_characters': reasoning_chars}
    if reason in ('length', 'max_tokens'):
        detail = '思考内容耗尽了输出额度，未返回场景 JSON' if reasoning_chars and not output.strip() else '最终内容未完整输出'
        raise ModelOutputError('模型输出达到长度上限：' + detail + '。',
                               code='output_limit', usage=usage, output=output)
    if not output.strip():
        raise ModelOutputError('模型没有返回最终内容，无法读取场景 JSON。',
                               code='empty_output', usage=usage)
    return output, usage


def endpoint(base, suffix):
    base = base.rstrip('/')
    return base + (suffix if base.endswith('/v1') else '/v1' + suffix)


def extract_json(text):
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.S).strip()
    if text.startswith('```'):
        text = re.sub(r'^```(?:json)?\s*|\s*```$', '', text)
    start = text.find('{')
    if start < 0:
        raise ModelError('模型未返回设计 JSON，请重试或切换模型。')
    try:
        return json.JSONDecoder().raw_decode(text[start:])[0]
    except json.JSONDecodeError as exc:
        raise ModelError('模型返回的设计数据不完整，请尝试更简洁的描述或更高输出上限的模型。') from exc


def complete(system, text, image_path=None, max_tokens=10000, config=None, json_output=False):
    cfg = config or settings()
    if not cfg['api_key']:
        raise ModelError('请先在模型设置中配置 API Key。示例资产仍可预览与下载。')
    headers = {'Authorization': 'Bearer ' + cfg['api_key']}
    image_data = base64.b64encode(image_path.read_bytes()).decode() if image_path else None
    if cfg['provider'] == 'anthropic':
        headers.update({'x-api-key': cfg['api_key'], 'anthropic-version': '2023-06-01'})
        content = [{'type': 'text', 'text': text}]
        if image_data:
            content.insert(0, {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png', 'data': image_data}})
        payload = {'model': cfg['model'], 'max_tokens': max_tokens, 'system': system, 'messages': [{'role': 'user', 'content': content}]}
        url = endpoint(cfg['base_url'], '/messages')
    else:
        content = text if not image_data else [{'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + image_data}}, {'type': 'text', 'text': text}]
        payload = {'model': cfg['model'], 'max_tokens': max_tokens, 'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': content}]}
        if urlparse(cfg['base_url']).hostname == 'api.deepseek.com':
            # DeepSeek defaults to thinking mode; its tokens share max_tokens
            # with the final JSON. Scene generation needs a complete final object.
            payload['thinking'] = {'type': 'disabled'}
            if json_output:
                payload['response_format'] = {'type': 'json_object'}
        url = endpoint(cfg['base_url'], '/chat/completions')
    for attempt in range(3):
        try:
            with httpx.Client(timeout=httpx.Timeout(240, connect=20)) as client:
                response = client.post(url, headers=headers, json=payload)
            if response.status_code in (429, 500, 502, 503, 504) and attempt < 2:
                time.sleep(2 ** attempt)
                continue
            if response.status_code >= 400:
                if config is None:
                    record_connection(False)
                # Never persist raw upstream errors: gateways can echo credentials.
                raise ModelError(f'模型接口返回 HTTP {response.status_code}。请检查地址、模型名、额度和授权。')
            try:
                data = response.json()
            except ValueError as exc:
                raise ModelOutputError('模型接口返回的响应不是有效 JSON。', code='invalid_response') from exc
            if not isinstance(data, dict):
                raise ModelOutputError('模型接口返回的响应结构不正确。', code='invalid_response')
            if config is None:
                record_connection(True)
            return final_response(data, cfg['provider'])
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            if attempt == 2:
                raise ModelError('模型接口连接超时或网络不可用，请检查连接后重试。') from exc
            time.sleep(2 ** attempt)
    raise ModelError('模型调用失败')
