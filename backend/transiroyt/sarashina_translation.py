"""Sarashina Translate API (English to Japanese) and its private API key."""
import uuid

import httpx
from pydantic import BaseModel, Field, SecretStr

from . import credentials
from .api_requests import APIRequests


class SarashinaSettings(BaseModel):
    api_key: SecretStr | None = Field(default=None, repr=False)
    credential_id: str | None = Field(default=None, pattern=r'^[0-9a-f]{32}$', repr=False)

    def public(self):
        return {'has_key': bool(self.api_key and self.api_key.get_secret_value().strip())}


@credentials.serialized
def load_settings(data):
    path = data / 'sarashina-translation.json'
    config = SarashinaSettings.model_validate_json(path.read_text()) if path.exists() else SarashinaSettings()
    if config.api_key and config.api_key.get_secret_value():
        return save_settings(data, config, previous=SarashinaSettings())
    if config.credential_id:
        value = credentials.Keychain(data).get(config.credential_id)
        config.api_key = SecretStr(value) if value else None
    return config


@credentials.serialized
def save_settings(data, settings, *, previous=None):
    previous = previous if previous is not None else load_settings(data)
    if settings.api_key is None:
        return previous
    value = settings.api_key.get_secret_value().strip()
    store = credentials.Keychain(data)
    account = uuid.uuid4().hex if value else None
    try:
        if account:
            store.put(account, value)
        credentials.write_private_json(data / 'sarashina-translation.json', {'credential_id': account})
    except Exception:
        if account:
            store.delete(account)
        raise
    if previous.credential_id:
        store.delete(previous.credential_id)
    return SarashinaSettings(api_key=value, credential_id=account)


async def translate_text(client, text, language, api_key, *, requests=None):
    if language.lower().split('-')[0] != 'en':
        raise RuntimeError('Sarashina翻訳は英語から日本語への翻訳にのみ対応しています。')
    if not api_key:
        raise RuntimeError('設定でSarashinaのAPIキーを入力してください。')
    if len(text) > 5000:
        raise RuntimeError('Sarashina翻訳の文字数上限（5000文字）を超えました。')
    try:
        response = await (requests or APIRequests()).request(
            client, 'POST', 'https://api.sarashina.ai/v1/translate',
            headers={'Authorization': f'Bearer {api_key}'},
            json={'text': text, 'source_lang': 'en', 'target_lang': 'ja'})
        response.raise_for_status()
        result = response.json()['translated_text']
        if not isinstance(result, str) or not result.strip():
            raise ValueError()
        return result
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(f'Sarashina翻訳がHTTP {exc.response.status_code}を返しました。APIキー・利用上限を確認してください。') from None
    except (httpx.HTTPError, ValueError, KeyError, TypeError):
        raise RuntimeError('Sarashina翻訳へ接続できないか、有効な翻訳応答を取得できませんでした。') from None
