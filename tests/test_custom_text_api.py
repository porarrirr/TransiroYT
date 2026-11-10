import json

import httpx
import pytest
from pydantic import ValidationError
from transiroyt.custom_text_api import CustomAPIConfig, CustomTextAPI, list_models
from transiroyt.dubbing_translation import GemmaDubbingTranslator, TranslationItem


@pytest.mark.parametrize('protocol', ['chat', 'responses', 'anthropic', 'gemini'])
async def test_protocol_and_translation(protocol):
    def handle(request):
        body = json.loads(request.content)
        text = '{"T0":"50%ではなく15%増えました。"}'
        if protocol == 'chat':
            assert request.url.path == '/v1/chat/completions'
            assert request.headers['authorization'] == 'Bearer private-key'
            assert 'JSON Schema' in body['messages'][0]['content']
            return httpx.Response(200, json={'choices': [{'message': {'content': text}}]})
        if protocol == 'responses':
            assert request.url.path == '/v1/responses'
            assert body['store'] is False
            return httpx.Response(200, json={'output': [{'type': 'reasoning'}, {'type': 'message', 'content': [{'type': 'output_text', 'text': text}]}]})
        if protocol == 'anthropic':
            assert request.url.path == '/v1/messages'
            assert request.headers['x-api-key'] == 'private-key'
            assert request.headers['anthropic-version'] == '2023-06-01'
            return httpx.Response(200, json={'content': [{'type': 'thinking', 'thinking': 'hidden'}, {'type': 'text', 'text': text}]})
        assert request.url.path == '/v1/models/test:generateContent'
        assert request.headers['x-goog-api-key'] == 'private-key'
        assert body['generationConfig']['responseMimeType'] == 'application/json'
        return httpx.Response(200, json={'candidates': [{'content': {'parts': [{'text': 'hidden', 'thought': True}, {'text': text}]}}]})
    api = CustomTextAPI(CustomAPIConfig(protocol=protocol, base_url='https://example.com/v1',
                                       model='test', api_key='private-key'), transport=httpx.MockTransport(handle))
    translator = GemmaDubbingTranslator(api, title=None, mora_per_second=6)
    result = await translator.translate_batch('en', [TranslationItem(0, '15 percent, not 50.', None, 10)], [], [])
    assert result == {0: '50%ではなく15%増えました。'}
    assert 'private-key' not in translator.identity


@pytest.mark.parametrize('protocol', ['chat', 'responses', 'anthropic', 'gemini'])
async def test_model_listing(protocol):
    def handle(request):
        assert request.url.path == '/model'
        assert request.url.params['custom'] == '1'
        if protocol == 'gemini':
            if request.url.params.get('pageToken') == 'next':
                return httpx.Response(200, json={'models': [{'name': 'models/b'}]})
            return httpx.Response(200, json={'models': [{'name': 'models/a', 'supportedGenerationMethods': ['generateContent']},
                                                       {'name': 'models/embed', 'supportedGenerationMethods': ['embedContent']}], 'nextPageToken': 'next'})
        return httpx.Response(200, json={'data': [{'id': 'b'}, {'id': 'a'}]})
    config = CustomAPIConfig(protocol=protocol, models_url='https://example.com/model?custom=1')
    assert await list_models(config, transport=httpx.MockTransport(handle)) == ['a', 'b']


async def test_errors_do_not_expose_provider_body_or_key():
    config = CustomAPIConfig(model='test', api_key='private-key')
    api = CustomTextAPI(config, transport=httpx.MockTransport(lambda request: httpx.Response(401, text='private-key echoed')))
    with pytest.raises(RuntimeError, match='HTTP 401') as error:
        await api.complete([{'text': 'hello'}])
    assert 'private-key' not in str(error.value)


@pytest.mark.parametrize('url', ['ftp://example.com', 'https://user:password@example.com', 'https://example.com/#fragment'])
def test_endpoint_validation(url):
    with pytest.raises(ValidationError):
        CustomAPIConfig(base_url=url)


def test_identity_changes_with_model_or_endpoint():
    config = CustomAPIConfig(model='first')
    assert config.identity != config.model_copy(update={'model': 'second'}).identity
    assert config.identity != config.model_copy(update={'inference_url': 'https://example.com/custom'}).identity


async def test_custom_inference_and_legacy_token_limit():
    def handle(request):
        assert str(request.url) == 'https://example.com/generate?route=1'
        body = json.loads(request.content)
        assert body['max_tokens'] == 64
        assert 'max_completion_tokens' not in body
        return httpx.Response(200, json={'choices': [{'message': {'content': 'こんにちは'}}]})
    api = CustomTextAPI(CustomAPIConfig(model='test', inference_url='https://example.com/generate?route=1',
                                       chat_token_parameter='max_tokens'), transport=httpx.MockTransport(handle))
    assert await api.complete([{'text': 'hello'}], max_tokens=64) == 'こんにちは'


@pytest.mark.parametrize('protocol', ['chat', 'responses'])
async def test_reasoning_budget_is_added_to_task_output_limit(protocol):
    def handle(request):
        body = json.loads(request.content)
        if protocol == 'chat':
            assert body['reasoning_effort'] == 'medium' and body['max_tokens'] == 4112
            return httpx.Response(200, json={'choices': [{'message': {'content': 'OK'}}]})
        assert body['reasoning'] == {'effort': 'medium'} and body['max_output_tokens'] == 4112
        return httpx.Response(200, json={'output': [{'type': 'message', 'content': [{'type': 'output_text', 'text': 'OK'}]}]})
    config = CustomAPIConfig(protocol=protocol, model='test', chat_token_parameter='max_tokens', reasoning_effort='medium')
    api = CustomTextAPI(config, transport=httpx.MockTransport(handle))
    assert await api.complete([{'text': 'Reply OK'}], max_tokens=16) == 'OK'
    assert config.identity != config.model_copy(update={'reasoning_effort': ''}).identity
    assert config.identity != config.model_copy(update={'reasoning_token_budget': 1024}).identity


@pytest.mark.parametrize('protocol', ['anthropic', 'gemini'])
def test_unsupported_reasoning_protocol_is_rejected(protocol):
    with pytest.raises(ValidationError, match='effort'):
        CustomAPIConfig(protocol=protocol, reasoning_effort='medium')
