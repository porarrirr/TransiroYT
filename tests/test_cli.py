import hashlib
import io
import json
import subprocess
import sys
from pathlib import Path

import httpx
import pytest
from transiroyt import cli


def test_preferences_are_shared_and_invalid_update_preserves_file(tmp_path, capsys):
    assert cli.main(['config', 'set', 'originalVolume', '0.25'], support=tmp_path) == 0
    assert cli.main(['config', 'set', 'steps', '16'], support=tmp_path) == 0
    assert cli.main(['config', 'set', 'youtubeQuality', '720'], support=tmp_path) == 0
    assert cli.load_preferences(tmp_path).youtubeQuality == '720'
    assert json.loads((tmp_path / 'product.json').read_text())['originalVolume'] == 0.25
    previous = (tmp_path / 'product.json').read_bytes()
    for key, value in [('steps', '17'), ('speakers', '1.5'), ('dubVolume', 'NaN'), ('typo', '1')]:
        assert cli.main(['config', 'set', key, value], support=tmp_path) != 0
        assert (tmp_path / 'product.json').read_bytes() == previous
    assert cli.main(['config', 'get'], support=tmp_path) == 0
    assert '0.25' in capsys.readouterr().out


def test_dub_request_uses_preferences_and_resolves_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    prefs = cli.Preferences(outputDirectory=str(tmp_path), originalVolume=0.3, speakers=2, steps=12,
                            youtubeQuality='720')
    args = cli.parser().parse_args(['dub', 'film.mp4', '--subtitles', 'captions.srt', '--dub-volume', '0.5'])
    body = cli.dub_request(args, prefs)
    assert body['video'] == str(tmp_path / 'film.mp4')
    assert body['subtitles'] == str(tmp_path / 'captions.srt')
    assert (body['original_volume'], body['dub_volume'], body['speakers'], body['steps']) == (0.3, 0.5, 2, 12)
    assert Path(body['output']).parent == tmp_path
    assert body['allow_download'] and not body['overwrite']
    args = cli.parser().parse_args(['dub', 'https://youtu.be/abcdefghijk', '--no-download'])
    body = cli.dub_request(args, prefs)
    assert body['youtube_url'] == args.source and body['video'] is None
    assert body['youtube_quality'] == '720' and not body['allow_download']


def connected_client(tmp_path, handler):
    data = tmp_path / '.local'
    data.mkdir(exist_ok=True)
    (data / 'token').write_text('test-private-token')
    def respond(request):
        if request.url.path == '/health':
            return httpx.Response(200, json={'service': 'transiroyt', 'data_id': hashlib.sha256(str(data.resolve()).encode()).hexdigest()})
        assert request.headers['Authorization'] == 'Bearer test-private-token'
        if request.url.path == '/api/status':
            return httpx.Response(200, json={'ready': True})
        return handler(request)
    client = cli.DesktopClient(tmp_path)
    client.http.close()
    client.http = httpx.Client(base_url=cli.API, transport=httpx.MockTransport(respond))
    return client


def test_ai_setting_preserves_masked_keys_and_import_is_secret_safe(tmp_path, monkeypatch, capsys):
    calls = []
    def handler(request):
        if request.method == 'GET':
            return httpx.Response(200, json={'engines': {'bing': False}, 'presets': [{'has_key': True}]})
        calls.append(json.loads(request.content))
        return httpx.Response(200, json={'engines': {'bing': True}, 'presets': [{'has_key': True}]})
    client = connected_client(tmp_path, handler)
    monkeypatch.setattr(cli, 'DesktopClient', lambda *args: client)
    assert cli.main(['config', 'set', 'ai.engines.bing', 'true'], support=tmp_path) == 0
    assert calls[0]['engines']['bing'] is True
    assert 'api_key' not in calls[0]['presets'][0]
    client = connected_client(tmp_path, lambda request: httpx.Response(422, json={'detail': [{'input': 'secret-key'}]}))
    monkeypatch.setattr(cli, 'DesktopClient', lambda *args: client)
    monkeypatch.setattr(sys, 'stdin', io.StringIO('{"api_key":"secret-key"}'))
    assert cli.main(['config', 'import', 'sarashina', '-'], support=tmp_path) == 1
    captured = capsys.readouterr()
    assert 'secret-key' not in captured.out + captured.err


def test_wrong_backend_is_never_sent_token(tmp_path):
    client = cli.DesktopClient(tmp_path)
    client.http.close()
    def handler(request):
        assert 'Authorization' not in request.headers
        return httpx.Response(200, json={'service': 'transiroyt', 'data_id': 'different'})
    client.http = httpx.Client(base_url=cli.API, transport=httpx.MockTransport(handler))
    with pytest.raises(RuntimeError, match='別のTransiroYT'):
        client.connect()
    client.http.close()


def test_auto_launch_waits_for_authenticated_backend(tmp_path, monkeypatch):
    client = connected_client(tmp_path, lambda request: httpx.Response(200, json={}))
    ready_transport = client.http._transport
    attempts = []
    def handler(request):
        attempts.append(request.url.path)
        if len(attempts) == 1:
            raise httpx.ConnectError('not running', request=request)
        return ready_transport.handle_request(request)
    client.http = httpx.Client(base_url=cli.API, transport=httpx.MockTransport(handler))
    app = tmp_path / 'TransiroYT.app'
    client.app = app
    launched = []
    monkeypatch.setattr(cli.subprocess, 'run', lambda command, **kwargs: launched.append(command))
    monkeypatch.setattr(cli.time, 'sleep', lambda seconds: None)
    client.connect()
    assert launched == [['/usr/bin/open', '-g', str(app)]]
    assert attempts == ['/health', '/health', '/api/status']
    client.http.close()


@pytest.mark.parametrize('state,code', [('done', 0), ('error', 1), ('cancelled', 1)])
def test_dub_wait_exit_codes_and_authenticated_request(tmp_path, monkeypatch, state, code):
    job = {'id': 'a' * 32, 'state': 'running', 'stage': 'prepare', 'fraction': 0, 'message': 'start'}
    submitted = []
    def handler(request):
        if request.method == 'POST':
            submitted.append(json.loads(request.content))
            assert Path(submitted[-1]['output']).parent.is_dir()
            return httpx.Response(202, json={'job': job})
        return httpx.Response(200, json={'job': {**job, 'state': state}})
    client = connected_client(tmp_path, handler)
    monkeypatch.setattr(cli, 'DesktopClient', lambda *args: client)
    monkeypatch.setattr(cli.time, 'sleep', lambda seconds: None)
    output = tmp_path / 'new directory/out.mp4'
    assert cli.main(['dub', str(tmp_path / 'video.mp4'), '-o', str(output)], support=tmp_path) == code
    assert submitted[0]['output'] == str(output)


def test_launcher_resolves_relative_symlink_and_uses_bundled_python(tmp_path):
    runtime = tmp_path / "App with spaces.app/Contents/Resources/Runtime"
    launcher = runtime / 'backend/transiroyt/cli-launcher'
    launcher.parent.mkdir(parents=True)
    source = Path(__file__).resolve().parents[1] / 'backend/transiroyt/cli-launcher'
    launcher.write_bytes(source.read_bytes())
    launcher.chmod(0o755)
    python = runtime / 'python-api/bin/python3.12'
    python.parent.mkdir(parents=True)
    python.symlink_to(sys.executable)
    module = launcher.parent / 'cli.py'
    module.write_text('import os,sys,json; print(json.dumps([os.environ["PYTHONPATH"],sys.argv[1:]]))')
    command = tmp_path / 'transiroyt'
    command.symlink_to(launcher.relative_to(tmp_path))
    result = subprocess.run([str(command), 'config', 'value with spaces'], capture_output=True, text=True, check=True)
    assert json.loads(result.stdout) == [str(runtime / 'backend'), ['config', 'value with spaces']]
    assert not list(runtime.rglob('__pycache__'))


def test_real_ai_routes_roundtrip_preserves_key_and_rejects_invalid_config(tmp_path, monkeypatch, capsys):
    from fastapi.testclient import TestClient
    from transiroyt.app import create_app
    from transiroyt.config import Settings
    from transiroyt.task_routing import load_routes

    data = tmp_path / '.local'
    data.mkdir()
    (data / 'token').write_text('test-token')
    app = create_app(Settings(data=data, token='test-token'))
    client = cli.DesktopClient(tmp_path)
    client.http.close()
    client.http = TestClient(app, base_url=cli.API)
    client.connect()
    config = client.request('GET', '/api/task-routing')
    preset = {'id': 'a' * 32, 'name': 'Example', 'model': 'model-id', 'enabled': True,
              'api_key': 'very-private-test-key'}
    config['presets'] = [preset]
    args = cli.parser().parse_args(['config', 'import', 'ai', '-'])
    monkeypatch.setattr(sys, 'stdin', io.StringIO(json.dumps(config)))
    public = cli.configure(args, tmp_path, client)
    assert public['presets'][0]['has_key']
    assert 'very-private-test-key' not in json.dumps(public)
    args = cli.parser().parse_args(['config', 'set', 'ai.engines.bing', 'true'])
    cli.configure(args, tmp_path, client)
    assert load_routes(data).presets[0].api_key.get_secret_value() == 'very-private-test-key'
    assert load_routes(data).engines.bing
    before = (data / 'task-routing.json').read_bytes()
    config['enabled'] = True
    config['language'] = {'chain': ['bing']}
    monkeypatch.setattr(sys, 'stdin', io.StringIO(json.dumps(config)))
    args = cli.parser().parse_args(['config', 'import', 'ai', '-'])
    with pytest.raises(RuntimeError, match='設定値'):
        cli.configure(args, tmp_path, client)
    assert (data / 'task-routing.json').read_bytes() == before
    assert (data / 'task-routing.json').stat().st_mode & 0o777 == 0o600
    client.http.close()
