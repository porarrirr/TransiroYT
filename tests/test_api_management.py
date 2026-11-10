import io
import json
import logging
import stat
from unittest.mock import AsyncMock

import httpx
import pytest
from pydantic import SecretStr, ValidationError
from transiroyt import cli, credentials
from transiroyt.app import create_app
from transiroyt.config import Settings
from transiroyt.custom_text_api import CustomAPIConfig, credential_scope
from transiroyt.sarashina_translation import SarashinaSettings
from transiroyt.sarashina_translation import load_settings as load_sarashina
from transiroyt.sarashina_translation import save_settings as save_sarashina
from transiroyt.task_routing import (
    APIPreset,
    TaskRoutingConfig,
    load_routes,
    resolve_preset,
    save_routes,
)

DUMMY = 'dummy-secret-for-tests-only'
FIRST = 'a' * 32


def test_routes_move_plaintext_to_keychain_and_keep_json_free_of_secrets(tmp_path, isolated_keychain):
    path = tmp_path / 'task-routing.json'
    path.write_text(json.dumps({'presets': [{'id': FIRST, 'model': 'm', 'api_key': DUMMY}]}))
    config = load_routes(tmp_path)
    assert config.presets[0].api_key.get_secret_value() == DUMMY
    account = config.presets[0].credential_id
    assert isolated_keychain[str(tmp_path.resolve()), account] == DUMMY
    assert DUMMY not in path.read_text() and 'api_key' not in path.read_text()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert DUMMY not in json.dumps(config.public()) and account not in json.dumps(config.public())
    public = TaskRoutingConfig.model_validate(config.public())
    public.presets[0].name = 'Renamed'
    save_routes(tmp_path, public)
    assert load_routes(tmp_path).presets[0].credential_id == account
    assert load_routes(tmp_path).presets[0].api_key.get_secret_value() == DUMMY
    save_routes(tmp_path, TaskRoutingConfig())
    assert not isolated_keychain


def test_sarashina_migration_rotation_and_delete(tmp_path, isolated_keychain):
    path = tmp_path / 'sarashina-translation.json'
    path.write_text(json.dumps({'api_key': DUMMY}))
    original = load_sarashina(tmp_path)
    assert original.api_key.get_secret_value() == DUMMY and DUMMY not in path.read_text()
    replaced = save_sarashina(tmp_path, SarashinaSettings(api_key='new-dummy'))
    assert original.credential_id != replaced.credential_id
    assert len(isolated_keychain) == 1
    assert load_sarashina(tmp_path).api_key.get_secret_value() == 'new-dummy'
    save_sarashina(tmp_path, SarashinaSettings(api_key=''))
    assert not isolated_keychain and not load_sarashina(tmp_path).public()['has_key']


def test_keychain_failure_preserves_legacy_plaintext_without_silently_falling_back(tmp_path, monkeypatch):
    path = tmp_path / 'task-routing.json'
    path.write_text(json.dumps({'presets': [{'model': 'm', 'api_key': DUMMY}]}))
    before = path.read_bytes()
    class LockedKeychain:
        def __init__(self, data): pass
        def put(self, account, value): raise credentials.CredentialError('locked')
        def delete(self, account): pass
    monkeypatch.setattr(credentials, 'Keychain', LockedKeychain)
    with pytest.raises(credentials.CredentialError, match='locked'):
        load_routes(tmp_path)
    assert path.read_bytes() == before


def test_json_write_failure_keeps_original_credential_and_rolls_back_new_item(tmp_path, monkeypatch, isolated_keychain):
    config = save_routes(tmp_path, TaskRoutingConfig(presets=[APIPreset(model='m', api_key=DUMMY)]))
    original = (tmp_path / 'task-routing.json').read_bytes()
    keys = dict(isolated_keychain)
    config.presets[0].api_key = SecretStr('replacement')
    def fail(*args): raise OSError('disk full')
    monkeypatch.setattr(credentials, 'write_private_json', fail)
    with pytest.raises(OSError):
        save_routes(tmp_path, config)
    assert (tmp_path / 'task-routing.json').read_bytes() == original
    assert isolated_keychain == keys


@pytest.mark.parametrize('field,value', [('base_url', 'https://other.test/v1'),
    ('inference_url', 'https://other.test/infer'), ('models_url', 'https://other.test/models'),
    ('protocol', 'anthropic')])
def test_changed_credential_destination_never_reuses_key(tmp_path, field, value):
    config = save_routes(tmp_path, TaskRoutingConfig(presets=[APIPreset(id=FIRST, model='m', api_key=DUMMY)]))
    draft = config.presets[0].model_copy(update={field: value, 'api_key': None})
    with pytest.raises(credentials.CredentialError, match='再入力'):
        resolve_preset(tmp_path, draft)
    draft.api_key = SecretStr('')
    assert resolve_preset(tmp_path, draft).credential_id is None


def test_same_origin_endpoint_path_keeps_key(tmp_path):
    config = save_routes(tmp_path, TaskRoutingConfig(presets=[APIPreset(model='m', api_key=DUMMY)]))
    draft = config.presets[0].model_copy(update={'inference_url': 'https://api.openai.com:443/custom', 'api_key': None})
    assert credential_scope(draft) == credential_scope(config.presets[0])
    assert resolve_preset(tmp_path, draft).api_key.get_secret_value() == DUMMY


@pytest.mark.parametrize('url', ['http://external.test/v1', 'http://192.168.1.4/v1',
                                'https://example.test/api?api_key=hidden', 'https://example.test/api?token=hidden'])
def test_unsafe_endpoint_is_rejected(url):
    with pytest.raises(ValidationError):
        CustomAPIConfig(base_url=url)


@pytest.mark.parametrize('url', ['http://127.0.0.1:8080/v1', 'http://localhost:8080/v1', 'http://[::1]:8080/v1',
                                'https://example.test/api?route=custom'])
def test_loopback_and_https_are_accepted(url):
    assert CustomAPIConfig(base_url=url).base_url == url


async def test_validation_response_does_not_echo_any_secret(tmp_path):
    app = create_app(Settings(data=tmp_path, token='test-token'))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://127.0.0.1') as client:
        for payload in [{'presets': [{'enabled': True, 'model': '', 'api_key': DUMMY}]},
                        {'presets': [{'base_url': 'http://external.test/' + DUMMY, 'api_key': DUMMY}]}]:
            result = await client.put('/api/task-routing', headers={'Authorization': 'Bearer test-token'}, json=payload)
            assert result.status_code == 422 and DUMMY not in result.text and 'input' not in result.text


async def test_connection_test_uses_saved_key_without_exposing_answer(tmp_path, monkeypatch):
    save_routes(tmp_path, TaskRoutingConfig(presets=[APIPreset(id=FIRST, model='m', api_key=DUMMY)]))
    complete = AsyncMock(return_value=DUMMY)
    monkeypatch.setattr('transiroyt.app.CustomTextAPI.complete', complete)
    app = create_app(Settings(data=tmp_path, token='test-token'))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://127.0.0.1') as client:
        draft = {'id': FIRST, 'model': 'm'}
        assert (await client.post('/api/task-routing/presets/test', json=draft)).status_code == 401
        result = await client.post('/api/task-routing/presets/test', headers={'Authorization': 'Bearer test-token'}, json=draft)
        assert result.status_code == 200 and result.json()['ok'] and DUMMY not in result.text
        assert complete.call_args.kwargs['max_tokens'] == 16


def test_http_logging_does_not_record_query_secrets():
    from transiroyt.logging_utils import configure_logging
    configure_logging()
    capture = io.StringIO()
    logger = logging.getLogger('httpx')
    handler = logging.StreamHandler(capture)
    logger.addHandler(handler)
    try:
        logger.info('HTTP request https://example.test?custom=%s', DUMMY)
    finally:
        logger.removeHandler(handler)
    assert DUMMY not in capture.getvalue()


def test_cli_key_uses_stdin_and_does_not_print_secret(tmp_path, monkeypatch, capsys):
    config = TaskRoutingConfig(presets=[APIPreset(id=FIRST, name='Example', model='m')]).public()
    saved = []
    class Client:
        def __init__(self, *args): self.http = self
        def connect(self): pass
        def close(self): pass
        def request(self, method, path, **kwargs):
            if method == 'GET':
                return config
            saved.append(kwargs['json'])
            return {'has_key': True}
    monkeypatch.setattr(cli, 'DesktopClient', Client)
    monkeypatch.setattr('sys.stdin', io.StringIO(DUMMY))
    assert cli.main(['api', 'key', 'Example', '--stdin'], support=tmp_path) == 0
    assert saved[0]['presets'][0]['api_key'] == DUMMY
    captured = capsys.readouterr()
    assert DUMMY not in captured.out + captured.err
