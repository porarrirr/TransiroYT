"""Installed app CLI. Jobs and AI settings use the desktop's authenticated API."""
from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

SUPPORT = Path.home() / 'Library/Application Support/TransiroYT'
API = 'http://127.0.0.1:8765'


class Preferences(BaseModel):
    # Match ProductPreferences, including the optional fields in older saved files.
    model_config = ConfigDict(extra='forbid')
    outputDirectory: str = Field(default='~/Downloads/TransiroYT', min_length=1)
    originalVolume: float = Field(default=0, ge=0, le=2, allow_inf_nan=False)
    dubVolume: float = Field(default=1, gt=0, le=2, allow_inf_nan=False)
    speakers: int | None = Field(default=None, ge=1, le=20, strict=True)
    youtubeQuality: Literal['best', '2160', '1440', '1080', '720', '480', '360'] = 'best'
    steps: int | None = Field(default=None, ge=4, le=16, strict=True)


def load_preferences(support):
    path = support / 'product.json'
    return Preferences.model_validate_json(path.read_text()) if path.exists() else Preferences()


class DesktopClient:
    def __init__(self, support, app=None):
        self.support, self.app = support, app
        self.http = httpx.Client(base_url=API, timeout=10, trust_env=False)

    def connect(self):
        deadline = time.monotonic() + 30
        launched = False
        expected = hashlib.sha256(str((self.support / '.local').resolve()).encode()).hexdigest()
        while True:
            try:
                health = self.http.get('/health', timeout=1)
                health.raise_for_status()
                identity = health.json()
                if identity.get('service') != 'transiroyt' or identity.get('data_id') != expected:
                    raise RuntimeError('別のTransiroYTがポート8765を使用しています。終了してから再実行してください。')
                token = (self.support / '.local/token').read_text().strip()
                self.http.headers['Authorization'] = f'Bearer {token}'
                self.request('GET', '/api/status')
                return
            except (httpx.TransportError, FileNotFoundError):
                if not launched:
                    if self.app is None:
                        raise RuntimeError('TransiroYTアプリを起動してから再実行してください。') from None
                    subprocess.run(['/usr/bin/open', '-g', str(self.app)], check=True)
                    launched = True
                if time.monotonic() >= deadline:
                    raise RuntimeError('アプリに接続できませんでした。アプリのログを確認してください。') from None
                time.sleep(0.25)

    def request(self, method, path, **kwargs):
        response = self.http.request(method, path, **kwargs)
        if response.is_error:
            # Validation responses can echo secrets from config imports.
            if response.status_code == 422:
                raise RuntimeError('設定値または入力が不正です。--helpとガイドを確認してください。')
            detail = response.json().get('detail', f'HTTP {response.status_code}')
            raise RuntimeError(str(detail))
        return response.json()


def parser():
    result = argparse.ArgumentParser(prog='transiroyt', description='TransiroYTの吹き替え・設定（アプリと共有）')
    commands = result.add_subparsers(dest='command', required=True)
    dub = commands.add_parser('dub', help='動画またはYouTube URLを吹き替え')
    dub.add_argument('source', help='動画ファイル、またはYouTube URL')
    dub.add_argument('--subtitles', type=Path, help='SRT字幕（省略可）')
    dub.add_argument('-o', '--output', type=Path, help='出力MP4（省略時は設定の保存先）')
    dub.add_argument('--original-volume', type=float)
    dub.add_argument('--dub-volume', type=float)
    dub.add_argument('--speakers', type=int)
    dub.add_argument('--steps', type=int)
    dub.add_argument('--youtube-quality', choices=['best', '2160', '1440', '1080', '720', '480', '360'])
    dub.add_argument('--language', default='auto')
    dub.add_argument('--no-download', action='store_true', help='モデルの自動取得を無効化')
    dub.add_argument('--overwrite', action='store_true')
    dub.add_argument('--no-wait', action='store_true', help='開始したジョブのJSONを返して終了')
    commands.add_parser('status', help='現在のジョブ・モデル状態をJSONで表示')
    cancel = commands.add_parser('cancel', help='指定したジョブを中止')
    cancel.add_argument('job_id')
    api = commands.add_parser('api', help='AI API接続の管理・接続テスト・キー設定')
    actions = api.add_subparsers(dest='action', required=True)
    actions.add_parser('list', help='接続を一覧表示（キーは非表示）')
    add = actions.add_parser('add', help='接続を追加。キーはapi keyで別途設定')
    add.add_argument('name')
    add.add_argument('--url', required=True)
    add.add_argument('--protocol', choices=['chat', 'responses', 'anthropic', 'gemini'], default='chat')
    add.add_argument('--model', default='')
    add.add_argument('--chat-token-parameter', choices=['max_completion_tokens', 'max_tokens'], default='max_completion_tokens')
    add.add_argument('--reasoning-effort', choices=['none', 'low', 'medium', 'high'], default='')
    add.add_argument('--reasoning-token-budget', type=int, default=4096)
    add.add_argument('--enable', action='store_true', help='簡易形式（enabled: false）で使用する。作業別の順番は設定画面で指定')
    for action in ['test', 'models', 'remove', 'key']:
        command = actions.add_parser(action, help={'test': '短いテスト文を送信（API利用料金が発生する場合あり）',
                                                  'models': 'モデル一覧を取得', 'remove': '接続と作業の割り当てを削除',
                                                  'key': 'APIキーをキーチェーンに保存'}[action])
        command.add_argument('connection', help='接続のIDまたは一意な名前')
        if action == 'key':
            source = command.add_mutually_exclusive_group()
            source.add_argument('--stdin', action='store_true', help='キーを標準入力から読む')
            source.add_argument('--delete', action='store_true', help='保存済みキーを削除')
    config = commands.add_parser('config', help='設定を表示・変更')
    actions = config.add_subparsers(dest='action', required=True)
    get = actions.add_parser('get', help='設定を表示（APIキーは表示しない）')
    get.add_argument('section', nargs='?', choices=['product', 'ai', 'sarashina'], default='product')
    edit = actions.add_parser('set', help='設定を1項目変更。値はJSONまたは文字列')
    edit.add_argument('key', help='例: originalVolume、steps、ai.engines.bing')
    edit.add_argument('value', help='APIキーはimportを使って標準入力から渡してください')
    imp = actions.add_parser('import', help='JSON設定を読み込み（APIキーは省略で保持、空文字で削除）')
    imp.add_argument('section', choices=['product', 'ai', 'sarashina'])
    imp.add_argument('file', help='JSONファイル、または - で標準入力')
    return result


def manage_api(args, client):
    client.connect()
    config = client.request('GET', '/api/task-routing')
    if args.action == 'list':
        return config['presets']
    if args.action == 'add':
        config['presets'].append({'id': uuid.uuid4().hex, 'name': args.name, 'base_url': args.url,
                                  'protocol': args.protocol, 'model': args.model, 'enabled': args.enable,
                                  'chat_token_parameter': args.chat_token_parameter,
                                  'reasoning_effort': args.reasoning_effort,
                                  'reasoning_token_budget': args.reasoning_token_budget})
    else:
        matches = [preset for preset in config['presets'] if args.connection in (preset['id'], preset['name'])]
        if len(matches) != 1:
            raise RuntimeError('接続を一意に指定できません。api listでIDを確認してください。')
        preset = matches[0]
        if args.action in ('test', 'models'):
            preset['enabled'] = False
            return client.request('POST', f'/api/task-routing/presets/{args.action}', json=preset, timeout=35)
        if args.action == 'key':
            if not args.stdin and not args.delete and not sys.stdin.isatty():
                raise RuntimeError('APIキーを標準入力から設定する場合は--stdinを指定してください。')
            value = '' if args.delete else (sys.stdin.read().strip() if args.stdin else getpass.getpass('APIキー（画面には表示しません）: ').strip())
            if not value and not args.delete:
                raise RuntimeError('空のAPIキーは保存できません。削除する場合は--deleteを指定してください。')
            preset['api_key'] = value
        else:
            config['presets'].remove(preset)
            for task in ('language', 'translation', 'shortening'):
                config[task]['chain'] = [provider for provider in config[task]['chain'] if provider != 'api:' + preset['id']]
    return client.request('PUT', '/api/task-routing', json=config)


def configure(args, support, client):
    section = args.section if args.action != 'set' else 'product'
    if args.action == 'set' and '.' in args.key:
        section, args.key = args.key.split('.', 1)
    paths = {'ai': '/api/task-routing', 'sarashina': '/api/sarashina-translation'}
    if section not in ('product', *paths):
        raise RuntimeError('設定の種類はproduct、ai、sarashinaから指定してください。')
    if args.action == 'set' and args.key.split('.')[-1] in ('api_key', 'has_key'):
        raise RuntimeError('APIキーはconfig importでJSONファイルまたは標準入力から設定してください。')
    if section == 'product':
        current = load_preferences(support).model_dump()
    else:
        client.connect()
        current = client.request('GET', paths[section])
    if args.action == 'get':
        return current
    if args.action == 'import':
        payload = json.loads(sys.stdin.read() if args.file == '-' else Path(args.file).expanduser().read_text())
        if not isinstance(payload, dict):
            raise RuntimeError('設定のJSONオブジェクトを指定してください。')
    else:
        try:
            value = json.loads(args.value)
        except ValueError:
            value = args.value
        payload = current
        target = payload
        keys = args.key.split('.')
        for key in keys[:-1]:
            if key not in target or not isinstance(target[key], dict):
                raise RuntimeError('設定名を確認してください。')
            target = target[key]
        if keys[-1] not in target:
            raise RuntimeError('設定名を確認してください。config getで一覧を確認できます。')
        if isinstance(target[keys[-1]], str) and not isinstance(value, str):
            value = args.value
        target[keys[-1]] = value
    if section == 'product':
        preferences = Preferences.model_validate(payload)
        support.mkdir(parents=True, exist_ok=True)
        temporary = support / 'product.cli.tmp'
        temporary.write_text(preferences.model_dump_json(indent=2))
        temporary.replace(support / 'product.json')
        return preferences.model_dump()
    return client.request('PUT', paths[section], json=payload)


def dub_request(args, preferences):
    youtube = args.source.startswith(('https://', 'http://'))
    name = 'YouTube' if youtube else Path(args.source).stem
    output = args.output or Path(preferences.outputDirectory).expanduser() / f'{name}-{uuid.uuid4().hex[:8]}.ja.mp4'
    payload = {'output': str(output.expanduser().resolve()), 'language': args.language,
               'overwrite': args.overwrite, 'allow_download': not args.no_download}
    if youtube:
        if args.subtitles:
            raise RuntimeError('YouTube URLと字幕ファイルは同時に指定できません。')
        payload['youtube_url'] = args.source
    else:
        payload['video'] = str(Path(args.source).expanduser().resolve())
        if args.subtitles:
            payload['subtitles'] = str(args.subtitles.expanduser().resolve())
    for key, default in [('original_volume', preferences.originalVolume), ('dub_volume', preferences.dubVolume),
                         ('speakers', preferences.speakers), ('steps', preferences.steps or 10),
                         ('youtube_quality', preferences.youtubeQuality)]:
        value = getattr(args, key)
        payload[key] = value if value is not None else default
    from .api_models import StartFileDubbing
    return StartFileDubbing.model_validate(payload).model_dump()


def wait_for_job(client, job):
    job_id, previous = job['id'], None
    while job['state'] == 'running':
        progress = (job['stage'], job['fraction'], job['message'])
        if progress != previous:
            print(f"[{job['stage']} {job['fraction']:.0%}] {job['message']}", file=sys.stderr, flush=True)
            previous = progress
        time.sleep(1)
        job = client.request('GET', '/api/file-dubbing')['job']
        if job is None or job['id'] != job_id:
            raise RuntimeError(f'ジョブ{job_id}の結果を取得できません。statusとライブラリを確認してください。')
    return job


def main(argv=None, *, support=SUPPORT, app=None):
    args = parser().parse_args(argv)
    client = DesktopClient(support, app)
    try:
        if args.command == 'api':
            value = manage_api(args, client)
        elif args.command == 'config':
            value = configure(args, support, client)
        else:
            payload = dub_request(args, load_preferences(support)) if args.command == 'dub' else None
            client.connect()
            if args.command == 'status':
                value = client.request('GET', '/api/file-dubbing')
            elif args.command == 'cancel':
                # IDs are generated as UUID hex by FileDubbingJobs.
                job_id = uuid.UUID(args.job_id).hex
                value = client.request('DELETE', f'/api/file-dubbing/{job_id}')
            else:
                Path(payload['output']).parent.mkdir(parents=True, exist_ok=True)
                value = client.request('POST', '/api/file-dubbing', json=payload)['job']
                if not args.no_wait:
                    value = wait_for_job(client, value)
                    if value['state'] != 'done':
                        raise RuntimeError(value.get('error') or '吹き替えが中止されました。')
        print(json.dumps(value, ensure_ascii=False, indent=2))
        return 0
    except KeyboardInterrupt:
        print('待機を終了しました。処理はアプリで継続します。中止はtransiroyt cancel JOB_ID。'
              if args.command == 'dub' else '操作を中止しました。', file=sys.stderr)
        return 130
    except ValidationError:
        print('エラー: 入力または設定値が不正です。--helpとガイドを確認してください。', file=sys.stderr)
        return 2
    except (OSError, ValueError, RuntimeError, httpx.HTTPError, subprocess.CalledProcessError) as exc:
        # JSON decoding errors may contain input fragments; never print config inputs.
        message = '設定ファイルを読み取れません。JSON形式・権限を確認してください。' if isinstance(exc, (ValueError, OSError)) else str(exc)
        print(f'エラー: {message}', file=sys.stderr)
        return 1
    finally:
        client.http.close()


if __name__ == '__main__':
    runtime = Path(__file__).resolve().parents[2]
    app_path = runtime.parents[2] if (runtime / 'distribution.json').exists() else None
    sys.exit(main(app=app_path))
