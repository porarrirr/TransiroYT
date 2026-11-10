import json
import stat
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from transiroyt.anonymous_translation import AnonymousDubbingTranslator
from transiroyt.app import create_app
from transiroyt.config import Settings
from transiroyt.dubbing_translation import TranslationItem
from transiroyt.sarashina_translation import (
    SarashinaSettings,
    load_settings,
    save_settings,
    translate_text,
)
from transiroyt.task_routing import Engines, TaskRoutingConfig, save_routes


async def test_translate_protocol():
    def handle(request):
        assert request.method == 'POST'
        assert str(request.url) == 'https://api.sarashina.ai/v1/translate'
        assert request.headers['authorization'] == 'Bearer private-key'
        assert json.loads(request.content) == {'text': '15%, not 50%.', 'source_lang': 'en', 'target_lang': 'ja'}
        return httpx.Response(200, json={'translated_text': '50%ではなく15%です。', 'future_field': 1})
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        assert await translate_text(client, '15%, not 50%.', 'en-US', 'private-key') == '50%ではなく15%です。'


@pytest.mark.parametrize('language,key,text', [('zh', 'key', 'hello'), ('auto', 'key', 'hello'),
                                              ('en', '', 'hello'), ('en', 'key', 'a' * 5001)])
async def test_invalid_input_does_not_send(language, key, text):
    def handle(request):
        pytest.fail('Invalid input must not be sent')
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(RuntimeError):
            await translate_text(client, text, language, key)


async def test_codepoint_limit_accepts_supplementary_characters():
    async with httpx.AsyncClient(transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={'translated_text': 'こんにちは'}))) as client:
        assert await translate_text(client, '😀' * 5000, 'en', 'key') == 'こんにちは'


@pytest.mark.parametrize('payload', [{}, [], {'translated_text': ''}, {'translated_text': 1}])
async def test_invalid_response(payload):
    async with httpx.AsyncClient(transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json=payload))) as client:
        with pytest.raises(RuntimeError):
            await translate_text(client, 'hello', 'en', 'private-key')


async def test_http_error_does_not_leak_key():
    async with httpx.AsyncClient(transport=httpx.MockTransport(
            lambda request: httpx.Response(401, text='private-key'))) as client:
        with pytest.raises(RuntimeError, match='HTTP 401') as error:
            await translate_text(client, 'hello', 'en', 'private-key')
        assert 'private-key' not in str(error.value)


async def test_translator_routing(monkeypatch):
    gemma = SimpleNamespace(model=SimpleNamespace(identity='test'))
    translate = AsyncMock(return_value='こんにちは。')
    monkeypatch.setattr('transiroyt.anonymous_translation.sarashina_translate', translate)
    translator = AnonymousDubbingTranslator(gemma, provider='sarashina', sarashina_key='private-key',
                                            title=None, mora_per_second=6)
    assert await translator.translate_batch('en', [TranslationItem(0, 'Hello.', None, 3)], [], []) == {0: 'こんにちは。'}
    assert translate.call_args.args[1:] == ('Hello.', 'en', 'private-key')
    assert 'private-key' not in translator.identity


async def test_private_settings_api(tmp_path):
    config = SarashinaSettings(api_key='private-key')
    save_settings(tmp_path, config)
    assert stat.S_IMODE((tmp_path / 'sarashina-translation.json').stat().st_mode) == 0o600
    assert load_settings(tmp_path).api_key.get_secret_value() == 'private-key'
    assert 'private-key' not in repr(config)
    app = create_app(Settings(data=tmp_path, token='token'))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://127.0.0.1') as client:
        assert (await client.get('/api/sarashina-translation')).status_code == 401
        headers = {'Authorization': 'Bearer token'}
        result = await client.get('/api/sarashina-translation', headers=headers)
        assert result.json() == {'has_key': True}
        assert 'private-key' not in result.text
        assert (await client.put('/api/sarashina-translation', headers=headers, json={})).json() == {'has_key': True}
        assert load_settings(tmp_path).api_key.get_secret_value() == 'private-key'
        assert (await client.put('/api/sarashina-translation', headers=headers, json={'api_key': ''})).json() == {'has_key': False}


def test_missing_key_rejects_job_before_processing(tmp_path, monkeypatch):
    from transiroyt.dubbing_options import FileDubbingOptions
    from transiroyt.errors import FileDubbingError
    from transiroyt.file_jobs import FileDubbingJobs

    monkeypatch.setattr('transiroyt.file_jobs.check_consumer_runtime', lambda settings: None)
    save_routes(tmp_path, TaskRoutingConfig(engines=Engines(sarashina=True)))
    jobs = FileDubbingJobs(Settings(data=tmp_path))
    options = FileDubbingOptions(video=tmp_path / 'video.mp4', subtitles=tmp_path / 'source.srt',
                                 output=tmp_path / 'out.mp4')
    with pytest.raises(FileDubbingError, match='APIキー'):
        jobs.start(options)
    assert jobs.task is None
