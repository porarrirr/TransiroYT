from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from transiroyt.anonymous_translation import AnonymousDubbingTranslator, translate_text
from transiroyt.dubbing_translation import TranslationItem


@pytest.mark.parametrize('provider', ['google', 'bing'])
async def test_protocol(provider):
    def handle(request):
        assert request.method == 'POST'
        if provider == 'google':
            assert b'client=gtx' in request.content
            assert b'sl=zh-CN' in request.content
            return httpx.Response(200, json=[[['こんにちは', 'hello'], ['。', '.']]])
        assert request.url.params['from'] == 'zh-Hans'
        return httpx.Response(200, json=[{'translations': [{'text': 'こんにちは。'}]}])
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        assert await translate_text(client, provider, 'hello.', 'zh') == 'こんにちは。'


@pytest.mark.parametrize('payload', [[], {}, [None], [['bad']]])
async def test_invalid_response(payload):
    async with httpx.AsyncClient(transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json=payload))) as client:
        with pytest.raises(RuntimeError):
            await translate_text(client, 'bing', 'hello', 'en')


async def test_external_translator_sends_each_line(monkeypatch):
    gemma = SimpleNamespace(model=SimpleNamespace(identity='test'))
    external = AsyncMock(return_value='こんにちは。')
    monkeypatch.setattr('transiroyt.anonymous_translation.translate_text', external)
    translator = AnonymousDubbingTranslator(gemma, provider='bing', title=None, mora_per_second=6)
    assert await translator.translate_batch('en', [TranslationItem(0, 'Hello.', None, 3)], [], []) == {0: 'こんにちは。'}
    assert external.await_count == 1
