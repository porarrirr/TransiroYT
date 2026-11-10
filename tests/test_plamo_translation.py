import json
from unittest.mock import AsyncMock

import httpx
import pytest
from pydantic import ValidationError
from transiroyt.config import Settings
from transiroyt.dubbing_options import FileDubbingOptions
from transiroyt.dubbing_translation import TranslationItem
from transiroyt.file_dubbing import FileDubber
from transiroyt.plamo_translation import PLAMO, PlamoTranslationModel
from transiroyt.task_routing import (
    Engines,
    TaskDubbingTranslator,
    TaskRoute,
    TaskRouter,
    TaskRoutingConfig,
)


async def test_native_plamo_request_and_context(tmp_path, monkeypatch):
    router = TaskRouter(Settings(data=tmp_path), TaskRoutingConfig(enabled=True, translation=TaskRoute(chain=['plamo']), shortening=TaskRoute(chain=[])))
    router.plamo.start = AsyncMock()
    router.plamo.endpoint = 'http://local.test'
    requests = []
    original = httpx.AsyncClient

    def handle(request):
        body = json.loads(request.content)
        assert request.url.path == '/v1/completions'
        assert request.headers['authorization'] == f'Bearer {router.plamo.key}'
        assert 'messages' not in body and 'response_format' not in body
        assert body['temperature'] == 0 and body['max_tokens'] == 1024
        assert body['stop'] == ['<|plamo:tag|>', '<|plamo:bos|>']
        requests.append(body['prompt'])
        return httpx.Response(200, json={'choices': [{'text': 'こんにちは。', 'finish_reason': 'stop'}]})

    monkeypatch.setattr('transiroyt.plamo_translation.httpx.AsyncClient', lambda **kw: original(transport=httpx.MockTransport(handle)))
    translator = TaskDubbingTranslator(router, title='Veritasium', mora_per_second=6)
    items = [TranslationItem(0, 'Hello.', None, 3), TranslationItem(1, 'Hi.', None, 3)]
    assert await translator.translate_batch('en', items, [(None, 'Before')], [(None, 'After')]) == {0: 'こんにちは。', 1: 'こんにちは。'}
    assert '<|plamo:tag|>context<|plamo:msg|>Before\nHi.\nAfter' in requests[0]
    assert '<|plamo:tag|>context<|plamo:msg|>Before\nHello.\nAfter' in requests[1]
    assert requests[0].endswith('<|plamo:val|>polite<|plamo:msg|>')
    assert PLAMO.identity in str(router.config.identity('translation'))
    await router.close()


async def test_plamo_failure_uses_explicit_fallback(tmp_path):
    config = TaskRoutingConfig(enabled=True, translation=TaskRoute(chain=['plamo', 'gemma']), shortening=TaskRoute(chain=[]))
    router = TaskRouter(Settings(data=tmp_path), config)
    router.plamo.translate = AsyncMock(side_effect=RuntimeError('offline'))
    router.local.complete_json = AsyncMock(return_value={'T0': 'こんにちは。'})
    translator = TaskDubbingTranslator(router, title=None, mora_per_second=6)
    assert await translator.translate_batch('en', [TranslationItem(0, 'Hello', None, 3)], [], []) == {0: 'こんにちは。'}
    router.local.complete_json.assert_awaited_once()


def test_default_translation_and_task_restrictions(tmp_path):
    routes = TaskRoutingConfig()
    assert routes.chain('translation') == ['plamo', 'gemma']
    assert routes.chain('language') == routes.chain('shortening') == ['gemma']
    for task in ('language', 'shortening'):
        with pytest.raises(ValidationError):
            TaskRoutingConfig(enabled=True, **{task: TaskRoute(chain=['plamo'])})
    video, subtitles = tmp_path / "input.mp4", tmp_path / "input.srt"
    video.touch()
    subtitles.touch()
    dubber = FileDubber(Settings(data=tmp_path), FileDubbingOptions(video, subtitles, tmp_path / "output.mp4"))
    assert dubber.task_router.config.chain('translation')[0] == 'plamo'
    assert isinstance(dubber.task_router.plamo, PlamoTranslationModel)


@pytest.mark.parametrize('enabled', [False, True])
async def test_plamo_and_sarashina_translate_only_english_sources(tmp_path, enabled):
    config = TaskRoutingConfig(enabled=enabled, engines=Engines(sarashina=True),
                               translation=TaskRoute(chain=['plamo', 'sarashina', 'gemma']))
    assert [p for p in config.translation_chain('en-US') if p in ('plamo', 'sarashina')] == ['plamo', 'sarashina']
    assert 'plamo' not in config.translation_chain('fr') and 'sarashina' not in config.translation_chain('fr')
    router = TaskRouter(Settings(data=tmp_path), config)
    router.plamo.translate = AsyncMock(side_effect=AssertionError('PLaMo must not translate French'))
    translator = TaskDubbingTranslator(router, title=None, mora_per_second=6)
    translator._request_from = AsyncMock(return_value={0: 'こんにちは。'})
    assert await translator._request('fr', [TranslationItem(0, 'Bonjour.', None, 3)], [], []) == {0: 'こんにちは。'}
    assert [call.args[0] for call in translator._request_from.call_args_list] == ['gemma']


async def test_english_only_chain_rejects_other_languages(tmp_path, monkeypatch):
    from transiroyt.errors import FileDubbingError
    from transiroyt.file_jobs import FileDubbingJobs
    from transiroyt.task_routing import save_routes

    config = TaskRoutingConfig(enabled=True, translation=TaskRoute(chain=['plamo']))
    translator = TaskDubbingTranslator(TaskRouter(Settings(data=tmp_path), config), title=None, mora_per_second=6)
    with pytest.raises(RuntimeError, match='英語の原文だけ'):
        await translator._request('de', [TranslationItem(0, 'Hallo.', None, 3)], [], [])
    save_routes(tmp_path, config)
    monkeypatch.setattr('transiroyt.file_jobs.check_consumer_runtime', lambda settings: None)
    jobs = FileDubbingJobs(Settings(data=tmp_path))
    options = FileDubbingOptions(tmp_path / 'video.mp4', tmp_path / 'source.srt', tmp_path / 'out.mp4', source_language='de')
    with pytest.raises(FileDubbingError, match='英語の原文だけ'):
        jobs.start(options)
    assert jobs.task is None


async def test_english_format_markers_do_not_reach_model(tmp_path, monkeypatch):
    model = PlamoTranslationModel(Settings(data=tmp_path))
    model.start = AsyncMock()
    model.endpoint = 'http://local.test'
    prompts = []
    original = httpx.AsyncClient

    def handle(request):
        prompts.append(json.loads(request.content)['prompt'])
        return httpx.Response(200, json={'choices': [{'text': '電池が2本必要です。', 'finish_reason': 'stop'}]})

    monkeypatch.setattr('transiroyt.plamo_translation.httpx.AsyncClient', lambda **kw: original(transport=httpx.MockTransport(handle)))
    await model.translate('Two\u200b batteries.', 'Not\ufeff broken.', 'plain', 'en-US')
    assert 'Two batteries.' in prompts[0] and 'Not broken.' in prompts[0]
    assert '\u200b' not in prompts[0] and '\ufeff' not in prompts[0]
    await model.translate('คำ\u200bไทย', '', 'plain', 'th')
    assert 'คำ\u200bไทย' in prompts[1]
    await model.close()
