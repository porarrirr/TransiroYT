"""User-selected text API protocols; presets are stored by task_routing."""
import hashlib
import ipaddress
import json
from types import SimpleNamespace
from typing import Literal
from urllib.parse import parse_qsl, quote, urlsplit

import httpx
from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator

from .api_requests import APIRequests
from .credentials import CredentialError
from .text_response import validate_text_response


class CustomAPIConfig(BaseModel):
    enabled: bool = False
    protocol: Literal['chat', 'responses', 'anthropic', 'gemini'] = 'chat'
    base_url: str = 'https://api.openai.com/v1'
    inference_url: str = ''
    models_url: str = ''
    model: str = ''
    chat_token_parameter: Literal['max_completion_tokens', 'max_tokens'] = 'max_completion_tokens'
    reasoning_effort: Literal['', 'none', 'low', 'medium', 'high'] = ''
    reasoning_token_budget: int = Field(default=4096, ge=0, le=32768)
    api_key: SecretStr | None = Field(default=None, repr=False)
    credential_id: str | None = Field(default=None, pattern=r'^[0-9a-f]{32}$', repr=False)

    @field_validator('base_url', 'inference_url', 'models_url')
    @classmethod
    def endpoint(cls, value):
        value = value.strip().rstrip('/')
        if value:
            url = urlsplit(value)
            if url.scheme not in ('http', 'https') or not url.netloc or url.username or url.password or url.fragment:
                raise ValueError('HTTP(S)のURLを指定してください（認証情報・フラグメント不可）。')
            _ = url.port  # Reject invalid ports at the settings boundary.
            try:
                local = ipaddress.ip_address(url.hostname).is_loopback
            except ValueError:
                local = url.hostname == 'localhost'
            if url.scheme == 'http' and not local:
                raise ValueError('外部APIにはHTTPSが必要です。HTTPはこのMacのループバック接続でのみ使えます。')
            if any(key.lower().replace('-', '_') in {'key', 'api_key', 'apikey', 'token', 'access_token', 'authorization'}
                   for key, _ in parse_qsl(url.query)):
                raise ValueError('URLにAPIキーやトークンを含めず、APIキー欄に入力してください。')
        return value

    @model_validator(mode='after')
    def ready(self):
        if not self.base_url:
            raise ValueError('ベースURLを指定してください。')
        if self.enabled and not self.model.strip():
            raise ValueError('モデルを選択または入力してください。')
        if self.reasoning_effort and self.protocol not in ('chat', 'responses'):
            raise ValueError('reasoning effortの指定はChat Completions・Responsesでのみ使えます。')
        return self

    def public(self):
        return {**self.model_dump(exclude={'api_key', 'credential_id'}), 'has_key': bool(self.api_key and self.api_key.get_secret_value())}

    @property
    def identity(self):
        recipe = [self.protocol, self.base_url, self.inference_url, self.model, self.chat_token_parameter,
                  self.reasoning_effort, self.reasoning_token_budget]
        return 'custom-api-v1:' + hashlib.sha256(json.dumps(recipe).encode()).hexdigest()

    @property
    def headers(self):
        if self.credential_id and not self.api_key:
            raise CredentialError('保存したAPIキーがキーチェーンに見つかりません。接続の設定でキーを再入力してください。')
        headers = {}
        key = self.api_key.get_secret_value() if self.api_key else ''
        if self.protocol == 'anthropic':
            headers['anthropic-version'] = '2023-06-01'
            if key:
                headers['x-api-key'] = key
        elif self.protocol == 'gemini':
            if key:
                headers['x-goog-api-key'] = key
        elif key:
            headers['Authorization'] = f'Bearer {key}'
        return headers


def credential_scope(config):
    origins = []
    for value in (config.base_url, config.inference_url or config.base_url, config.models_url or config.base_url):
        url = urlsplit(value)
        origins.append((url.scheme, url.hostname, url.port or (443 if url.scheme == 'https' else 80)))
    return config.protocol, *origins


async def request_json(client, method, url, *, requests=None, **kwargs):
    try:
        response = await (requests or APIRequests()).request(client, method, url, **kwargs)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as exc:
        # Provider error bodies and URLs can contain credentials or submitted text.
        status = exc.response.status_code
        reason = ('APIキーと利用権限を確認してください。' if status in (401, 403)
                  else '利用上限・残高・レート制限を確認し、時間をおいて再試行してください。' if status == 429
                  else '送信URLとモデルIDを確認してください。' if status == 404
                  else '接続先で障害が発生しています。時間をおいて再試行してください。' if status >= 500
                  else 'URL・APIキー・モデルを確認してください。')
        raise RuntimeError(f'カスタムAPIがHTTP {status}を返しました。{reason}') from None
    except (httpx.HTTPError, ValueError):
        raise RuntimeError('カスタムAPIへ接続できないか、JSON応答を読み取れませんでした。') from None


async def list_models(config, *, transport=None):
    models = []
    params = {}
    requests = APIRequests()
    url = config.models_url or config.base_url + '/models'
    async with httpx.AsyncClient(timeout=30, follow_redirects=False, transport=transport) as client:
        while True:
            payload = await request_json(client, 'GET', httpx.URL(url).copy_merge_params(params),
                                         requests=requests, headers=config.headers)
            try:
                rows = payload['models'] if config.protocol == 'gemini' else payload['data']
                for row in rows:
                    if config.protocol == 'gemini':
                        if 'generateContent' not in row.get('supportedGenerationMethods', ['generateContent']):
                            continue
                        name = row['name'].removeprefix('models/')
                    else:
                        name = row['id']
                    if not isinstance(name, str) or not name:
                        raise ValueError()
                    models.append(name)
                token = payload.get('nextPageToken') if config.protocol == 'gemini' else (
                    payload.get('last_id') if payload.get('has_more') else None)
                if not token:
                    break
                key = 'pageToken' if config.protocol == 'gemini' else 'after_id'
                if params.get(key) == token:
                    raise ValueError()
                params[key] = token
            except (KeyError, TypeError, ValueError, AttributeError):
                raise RuntimeError('モデル一覧の形式が不正です。モデルIDは手入力もできます。') from None
    return sorted(set(models))


class CustomTextAPI:
    def __init__(self, config, *, transport=None):
        self.config, self.transport = config, transport
        self.model = SimpleNamespace(identity=config.identity)
        self.text_model = self
        self.requests = APIRequests()

    async def close(self):
        pass

    async def complete(self, content, *, schema=None, max_tokens=1024, system=None):
        config = self.config
        prompt = '\n\n'.join(part['text'] for part in content)
        if schema:
            prompt += '\n次のJSON Schemaに一致するJSONだけを返してください。\n' + json.dumps(schema, ensure_ascii=False)
        messages = ([{'role': 'system', 'content': system}] if system else []) + [{'role': 'user', 'content': prompt}]
        limit = max(1, max_tokens)
        if config.reasoning_effort not in ('', 'none'):
            limit += config.reasoning_token_budget
        if config.protocol == 'chat':
            path = '/chat/completions'
            body = {'model': config.model, 'messages': messages, config.chat_token_parameter: limit, 'stream': False}
            if config.reasoning_effort:
                body['reasoning_effort'] = config.reasoning_effort
        elif config.protocol == 'responses':
            path = '/responses'
            body = {'model': config.model, 'input': messages, 'max_output_tokens': limit, 'store': False, 'stream': False}
            if config.reasoning_effort:
                body['reasoning'] = {'effort': config.reasoning_effort}
        elif config.protocol == 'anthropic':
            path = '/messages'
            body = {'model': config.model, 'messages': [{'role': 'user', 'content': prompt}], 'max_tokens': limit}
            if system:
                body['system'] = system
        else:
            path = '/models/' + quote(config.model.removeprefix('models/'), safe='') + ':generateContent'
            body = {'contents': [{'role': 'user', 'parts': [{'text': prompt}]}],
                    'generationConfig': {'maxOutputTokens': limit}}
            if schema:
                body['generationConfig']['responseMimeType'] = 'application/json'
            if system:
                body['systemInstruction'] = {'parts': [{'text': system}]}
        async with httpx.AsyncClient(timeout=180, follow_redirects=False, transport=self.transport) as client:
            payload = await request_json(client, 'POST', config.inference_url or config.base_url + path,
                                         requests=self.requests, headers=config.headers, json=body)
        try:
            if config.protocol == 'chat':
                choice = payload['choices'][0]
                if choice.get('finish_reason') == 'length':
                    raise RuntimeError('カスタムAPIの出力がトークン上限に達しました。')
                raw = choice['message']['content']
            elif config.protocol == 'responses':
                if payload.get('status') == 'incomplete':
                    raise RuntimeError('カスタムAPIの応答が未完了です。')
                raw = ''.join(part['text'] for item in payload['output'] if item['type'] == 'message'
                              for part in item['content'] if part['type'] == 'output_text')
            elif config.protocol == 'anthropic':
                if payload.get('stop_reason') == 'max_tokens':
                    raise RuntimeError('カスタムAPIの出力がトークン上限に達しました。')
                raw = ''.join(part['text'] for part in payload['content'] if part['type'] == 'text')
            else:
                candidate = payload['candidates'][0]
                if candidate.get('finishReason') == 'MAX_TOKENS':
                    raise RuntimeError('カスタムAPIの出力がトークン上限に達しました。')
                raw = ''.join(part['text'] for part in candidate['content']['parts']
                              if 'text' in part and not part.get('thought', False))
            if not isinstance(raw, str):
                raise ValueError()
        except (KeyError, IndexError, TypeError, ValueError):
            raise RuntimeError('カスタムAPIから有効なテキスト応答を取得できませんでした。') from None
        raw = raw.strip()
        if raw.startswith('```') and raw.endswith('```'):
            raw = raw.split('\n', 1)[1].rsplit('```', 1)[0].strip()
        return validate_text_response(raw, schema)

    async def complete_json(self, content, schema, **options):
        return json.loads(await self.complete(content, schema=schema, **options))
