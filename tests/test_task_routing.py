import json
import stat
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from pydantic import SecretStr, ValidationError
from transiroyt.app import create_app
from transiroyt.config import Settings
from transiroyt.dubbing_captions import detect_language
from transiroyt.dubbing_translation import TranslationItem
from transiroyt.file_dubbing import FileDubber
from transiroyt.task_routing import (
    APIPreset,
    Engines,
    RoutedText,
    TaskDubbingTranslator,
    TaskRoute,
    TaskRouter,
    TaskRoutingConfig,
    load_routes,
    save_routes,
)

FIRST, SECOND = 'a' * 32, 'b' * 32


def presets(**options):
    return [APIPreset(id=FIRST, name='無料枠1', model='first', base_url='https://first.example/v1', enabled=True, **options),
            APIPreset(id=SECOND, name='無料枠2', model='second', base_url='https://second.example/v1', enabled=True, **options)]


def test_unsupported_assignments_are_rejected():
    for config in [dict(language=TaskRoute(chain=['bing'])), dict(shortening=TaskRoute(chain=['sarashina'])),
                   dict(translation=TaskRoute(chain=[])), dict(language=TaskRoute(chain=['api:' + FIRST])),
                   dict(language=TaskRoute(chain=['pcc']))]:
        with pytest.raises(ValidationError):
            TaskRoutingConfig(enabled=True, **config)
    with pytest.raises(ValidationError, match='モデル'):
        TaskRoutingConfig(enabled=True, presets=[APIPreset(id=FIRST, name='空')],
                          language=TaskRoute(chain=['api:' + FIRST]))
    assert TaskRoutingConfig(presets=[APIPreset(name='下書き')]).presets[0].model == ''
    # Simple mode needs a general model for language detection and shortening.
    with pytest.raises(ValidationError):
        TaskRoutingConfig(engines=Engines(gemma=False))
    assert TaskRoutingConfig(enabled=True, shortening=TaskRoute(chain=[])).chain('shortening') == []


def test_simple_mode_orders_enabled_engines():
    config = TaskRoutingConfig(presets=presets(), engines=Engines(google=True))
    apis = ['api:' + FIRST, 'api:' + SECOND]
    assert config.chain('translation') == ['plamo', *apis, 'google', 'gemma']
    assert config.chain('language') == config.chain('shortening') == [*apis, 'gemma']
    assert config.translation_chain('ko') == [*apis, 'google', 'gemma']
    config.presets[0].enabled = False
    assert config.chain('language') == ['api:' + SECOND, 'gemma']


async def test_rate_limited_presets_then_gemma(tmp_path):
    config = TaskRoutingConfig(presets=presets(), engines=Engines(plamo=False))
    router = TaskRouter(Settings(data=tmp_path), config)
    hosts = []

    def handle(request):
        hosts.append(request.url.host)
        return httpx.Response(429, headers={'Retry-After': '120'}, text='rate limited')
    for client in router.apis.values():
        client.transport = httpx.MockTransport(handle)
    router.local.complete = AsyncMock(return_value='{"T0":"こんにちは。"}')
    translator = TaskDubbingTranslator(router, title=None, mora_per_second=6)
    assert await translator._request('en', [TranslationItem(0, 'Hello.', None, 3)], [], []) == {0: 'こんにちは。'}
    assert hosts == ['first.example', 'second.example']
    assert router.local.complete.await_count == 1
    hosts.clear()
    router.local.complete = AsyncMock(return_value='{"language":"en"}')
    assert await router.text('language').complete_json([{'text': 'hello'}], {}) == {'language': 'en'}
    assert hosts == []  # Both presets are still in their shared cooldown.


async def test_language_chain_and_its_end(tmp_path):
    config = TaskRoutingConfig(enabled=True, presets=presets(), language=TaskRoute(chain=['api:' + FIRST, 'gemma']))
    router = TaskRouter(Settings(data=tmp_path), config)
    router.apis['api:' + FIRST].complete = AsyncMock(side_effect=RuntimeError('offline'))
    router.local.complete = AsyncMock(return_value='{"language":"en"}')
    text = router.text('language')
    assert await text.complete_json([{'text': 'hello'}], {}) == {'language': 'en'}
    assert router.local.complete.await_count == 1
    config.language.chain = ['api:' + FIRST]
    router.local.complete.reset_mock()
    with pytest.raises(RuntimeError, match='offline'):
        await text.complete([{'text': 'hello'}])
    router.local.complete.assert_not_awaited()


async def test_tasks_use_their_own_presets(tmp_path):
    third = APIPreset(id='c' * 32, name='短縮', model='short-model', base_url='https://short.example/v1')
    config = TaskRoutingConfig(enabled=True, presets=[*presets(), third],
        language=TaskRoute(chain=['api:' + FIRST]), translation=TaskRoute(chain=['api:' + SECOND]),
        shortening=TaskRoute(chain=['api:' + third.id]))
    router = TaskRouter(Settings(data=tmp_path), config)
    requests = []
    def handle(request):
        body = json.loads(request.content)
        requests.append((request.url.host, body['model']))
        answer = '{"language":"en"}' if request.url.host == 'first.example' else '{"T0":"こんにちは。"}'
        return httpx.Response(200, json={'choices': [{'message': {'content': answer}}]})
    for client in router.apis.values():
        client.transport = httpx.MockTransport(handle)
    from transiroyt.subtitle_files import Cue
    assert await detect_language(router.text('language'), [Cue(0, 1, 'Hello')]) == 'en'
    translator = TaskDubbingTranslator(router, title=None, mora_per_second=6)
    items = [TranslationItem(0, 'Hello.', None, 3)]
    assert await translator.translate_batch('en', items, [], []) == {0: 'こんにちは。'}
    assert await translator._condense('en', items, {0: '非常に長い訳文です。'}, [], []) == {0: 'こんにちは。'}
    assert requests == [('first.example', 'first'), ('second.example', 'second'), ('short.example', 'short-model')]
    assert router.shortening_translator(title=None, mora_per_second=6).gemma.task == 'shortening'


@pytest.mark.parametrize('failure', ['timeout', 'malformed', 'truncated', 'unauthorized', 'server'])
@pytest.mark.parametrize('task', ['language', 'translation', 'shortening'])
async def test_api_failures_fallback_for_every_text_task(tmp_path, failure, task):
    config = TaskRoutingConfig(presets=presets(), engines=Engines(plamo=False, gemma=False))
    router = TaskRouter(Settings(data=tmp_path), config)
    calls = []
    answer = '{"language":"en"}' if task == 'language' else '{"T0":"こんにちは。"}'
    def handle(request):
        calls.append(request.url.host)
        if request.url.host == 'first.example':
            if failure == 'timeout':
                raise httpx.ReadTimeout('private body and URL must not escape', request=request)
            if failure == 'malformed':
                return httpx.Response(200, json={})
            if failure == 'truncated':
                return httpx.Response(200, json={'choices': [{'finish_reason': 'length'}]})
            return httpx.Response(401 if failure == 'unauthorized' else 503)
        return httpx.Response(200, json={'choices': [{'message': {'content': answer}}]})
    for api in router.apis.values():
        api.transport = httpx.MockTransport(handle)
    if task == 'language':
        from transiroyt.subtitle_files import Cue
        assert await detect_language(router.text(task), [Cue(0, 3, 'Hello')]) == 'en'
    else:
        translator = TaskDubbingTranslator(router, title=None, mora_per_second=6)
        items = [TranslationItem(0, 'Hello', None, 3)]
        result = (await translator._request('en', items, [], []) if task == 'translation'
                  else await translator._condense('en', items, {0: 'とても長い現在の訳文です。'}, [], []))
        assert result == {0: 'こんにちは。'}
    assert calls == ['first.example', 'second.example']


async def test_external_translation_failure_uses_next_engine(tmp_path, monkeypatch):
    config = TaskRoutingConfig(enabled=True, translation=TaskRoute(chain=['bing', 'gemma']),
                               shortening=TaskRoute(chain=[]))
    answer = AsyncMock(return_value='{"T0":"こんにちは。"}')
    router = TaskRouter(Settings(data=tmp_path), config)
    router.local.complete = answer
    external = AsyncMock(side_effect=RuntimeError('HTTP failure'))
    monkeypatch.setattr('transiroyt.anonymous_translation.translate_text', external)
    translator = TaskDubbingTranslator(router, title=None, mora_per_second=6)
    assert await translator.translate_batch('en', [TranslationItem(0, 'Hello', None, 3)], [], []) == {0: 'こんにちは。'}
    assert router.local.complete.await_count == 1
    assert await translator._condense('en', [], {}, [], []) == {}
    assert router.shortening_translator(title=None, mora_per_second=6) is None


async def test_external_partial_success_is_kept_for_fallback(tmp_path, monkeypatch):
    config = TaskRoutingConfig(enabled=True, translation=TaskRoute(chain=['sarashina', 'gemma']),
                               shortening=TaskRoute(chain=[]))
    answer = AsyncMock(return_value='{"T1":"ありがとう。"}')
    router = TaskRouter(Settings(data=tmp_path), config)
    router.local.complete = answer
    external = AsyncMock(side_effect=['こんにちは。', RuntimeError('HTTP 429')])
    monkeypatch.setattr('transiroyt.anonymous_translation.sarashina_translate', external)
    translator = TaskDubbingTranslator(router, title=None, mora_per_second=6)
    items = [TranslationItem(0, 'Hello', None, 3), TranslationItem(1, 'Thanks', None, 3)]
    assert await translator._request('en', items, [], []) == {0: 'こんにちは。', 1: 'ありがとう。'}
    assert external.await_count == 2
    prompt = router.local.complete.call_args.args[0][0]['text']
    assert 'T0 [' not in prompt and 'T1 [' in prompt


def test_routes_and_cache_identity(tmp_path):
    config = TaskRoutingConfig(enabled=True, shortening=TaskRoute(chain=[]))
    save_routes(tmp_path, config)
    dubber = FileDubber(Settings(data=tmp_path), Mock(allow_download=False))
    assert isinstance(dubber._translator(Mock()), TaskDubbingTranslator)
    before = dubber._translator(Mock()).identity
    config.translation.chain = ['bing']
    save_routes(tmp_path, config)
    other = FileDubber(Settings(data=tmp_path), Mock(allow_download=False))
    assert other._translator(Mock()).identity != before
    config.shortening.chain = ['gemma']
    third = TaskDubbingTranslator(TaskRouter(Settings(data=tmp_path), config), title=None, mora_per_second=6)
    assert third.identity != other._translator(Mock()).identity
    # Renaming or reordering an unused preset does not invalidate translations.
    renamed = TaskRoutingConfig(presets=presets())
    identity = renamed.identity('translation')
    renamed.presets[0].name = '別名'
    assert renamed.identity('translation') == identity


async def test_private_preset_storage_and_models(tmp_path, monkeypatch):
    config = TaskRoutingConfig(presets=presets(api_key='private-key'))
    save_routes(tmp_path, config)
    assert stat.S_IMODE((tmp_path / 'task-routing.json').stat().st_mode) == 0o600
    assert 'private-key' not in json.dumps(config.public())
    app = create_app(Settings(data=tmp_path, token='token'))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://127.0.0.1') as client:
        assert (await client.get('/api/task-routing')).status_code == 401
        headers = {'Authorization': 'Bearer token'}
        response = await client.get('/api/task-routing', headers=headers)
        assert [preset['has_key'] for preset in response.json()['presets']] == [True, True]
        assert 'private-key' not in response.text
        # Saving a masked GET response must preserve each preset's own key.
        saved = await client.put('/api/task-routing', headers=headers, json=response.json())
        assert saved.status_code == 200
        assert load_routes(tmp_path).presets[1].api_key.get_secret_value() == 'private-key'
        async def models(api):
            assert api.api_key.get_secret_value() == 'private-key'
            return ['test']
        monkeypatch.setattr('transiroyt.app.list_models', models)
        response = await client.post('/api/task-routing/presets/models', headers=headers,
                                     json={'id': FIRST, 'base_url': 'https://first.example/v1'})
        assert response.json() == {'models': ['test']}
    draft = load_routes(tmp_path)
    draft.presets[0] = draft.presets[0].model_copy(update={'base_url': 'https://other.example/v1', 'api_key': None})
    from transiroyt.credentials import CredentialError
    with pytest.raises(CredentialError, match='再入力'):
        save_routes(tmp_path, draft)
    draft.presets[0].api_key = SecretStr('')
    save_routes(tmp_path, draft)
    assert not load_routes(tmp_path).presets[0].api_key


async def test_cancellation_does_not_fallback(tmp_path):
    import asyncio
    router = TaskRouter(Settings(data=tmp_path), TaskRoutingConfig(presets=presets()))
    router.apis['api:' + FIRST].complete = AsyncMock(side_effect=asyncio.CancelledError())
    router.local.complete = AsyncMock()
    with pytest.raises(asyncio.CancelledError):
        await RoutedText(router, 'language').complete([{'text': 'hello'}])
    router.local.complete.assert_not_awaited()


async def test_gemma_uses_local_server(tmp_path, monkeypatch):
    from transiroyt import gemma_audio
    from transiroyt.gemma_audio import GemmaAudio

    gemma = GemmaAudio(Settings(data=tmp_path))
    gemma.start = AsyncMock()
    gemma.endpoint = 'http://local.test'
    original = httpx.AsyncClient
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={
        'choices': [{'finish_reason': 'stop', 'message': {'content': 'こんにちは'}}]}))
    monkeypatch.setattr(gemma_audio.httpx, 'AsyncClient', lambda **kwargs: original(transport=transport))
    assert await gemma.complete([{'type': 'text', 'text': 'hello'}]) == 'こんにちは'


async def test_unanswered_lines_use_next_engine(tmp_path):
    config = TaskRoutingConfig(enabled=True, presets=presets(), translation=TaskRoute(chain=['api:' + FIRST, 'gemma']))
    router = TaskRouter(Settings(data=tmp_path), config)
    translator = TaskDubbingTranslator(router, title=None, mora_per_second=6)
    translator._request_from = AsyncMock(side_effect=[{0: 'こんにちは。'}, {1: 'ありがとう。'}])
    items = [TranslationItem(0, 'Hello', None, 3), TranslationItem(1, 'Thanks', None, 3)]
    assert await translator._request('en', items, [], []) == {0: 'こんにちは。', 1: 'ありがとう。'}
    assert translator._request_from.call_args_list[1].args[0] == 'gemma'
    assert translator._request_from.call_args_list[1].args[2] == [items[1]]


async def test_language_detection_sends_first_150_characters():
    from transiroyt.subtitle_files import Cue
    text = SimpleNamespace(complete_json=AsyncMock(return_value={'language': 'en'}))
    assert await detect_language(text, [Cue(0, 1, 'a' * 100), Cue(1, 2, 'b' * 100)]) == 'en'
    prompt = text.complete_json.call_args.args[0][0]['text']
    sample = json.loads(prompt.split('\n', 1)[1])['subtitles']
    assert sample == 'a' * 100 + ' ' + 'b' * 49


def test_primary_fallback_files_become_chains(tmp_path):
    api = {'protocol': 'chat', 'base_url': 'https://old.example/v1', 'model': 'old-model', 'api_key': 'private-key'}
    (tmp_path / 'task-routing.json').write_text(json.dumps({
        'enabled': True,
        'language': {'primary': 'gemma', 'fallback': 'pcc', 'api': {}},
        'translation': {'primary': 'plamo', 'fallback': 'bing', 'api': {}},
        'shortening': {'primary': 'api', 'fallback': 'none', 'api': api}}))
    config = load_routes(tmp_path)
    assert config.chain('language') == ['gemma']
    assert config.chain('translation') == ['plamo', 'bing']
    [preset] = config.presets
    assert config.chain('shortening') == [f'api:{preset.id}']
    assert (preset.model, preset.api_key.get_secret_value()) == ('old-model', 'private-key')
    save_routes(tmp_path, config)
    assert load_routes(tmp_path).chain('shortening') == [f'api:{preset.id}']
    with pytest.raises(ValidationError):
        TaskRoute.model_validate({'primary': 'pcc'})


def test_saved_cloud_routes_migrate_without_losing_api_or_disabling_shortening(tmp_path):
    path = tmp_path / 'task-routing.json'
    payload = {
        'enabled': True, 'engines': {'pcc': True},
        'presets': [preset.model_dump(mode='json') for preset in presets()],
        'language': {'chain': ['pcc', 'gemma']},
        'translation': {'chain': ['plamo', 'api:' + FIRST, 'pcc', 'gemma']},
        'shortening': {'chain': []},
    }
    path.write_text(json.dumps(payload))
    config = load_routes(tmp_path)
    assert config.chain('language') == ['gemma']
    assert config.chain('translation') == ['plamo', 'api:' + FIRST, 'gemma']
    assert config.chain('shortening') == []
    assert [preset.id for preset in config.presets] == [FIRST, SECOND]
    assert path.read_text() == json.dumps(payload)
    save_routes(tmp_path, config)
    assert 'pcc' not in path.read_text()
    with pytest.raises(ValidationError):
        TaskRoutingConfig(enabled=True, language=TaskRoute(chain=['pcc']))
