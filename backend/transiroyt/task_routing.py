"""Text engines per task: saved API presets, tried in order until one answers.

Simple mode (``enabled`` false) only switches engines on or off and the order
comes from ``TaskRoutingConfig.chain``; per-task mode stores an explicit chain.
"""
import hashlib
import json
import logging
import uuid
from types import SimpleNamespace

from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator

from . import credentials
from .anonymous_translation import AnonymousDubbingTranslator
from .api_requests import APIRequests
from .custom_text_api import CustomAPIConfig, CustomTextAPI, credential_scope
from .dubbing_translation import GemmaDubbingTranslator
from .gemma_audio import GemmaAudio
from .plamo_translation import PLAMO, PROTOCOL, PlamoDubbingTranslator, PlamoTranslationModel
from .sarashina_translation import load_settings as load_sarashina

logger = logging.getLogger(__name__)

TASKS = ('language', 'translation', 'shortening')
ENGINES = ('plamo', 'gemma', 'sarashina', 'google', 'bing')
EXTERNAL = ('google', 'bing', 'sarashina')
TRANSLATION_ONLY = ('plamo', *EXTERNAL)
# PLaMo was adopted from English→Japanese measurements only; Sarashina's API is English→Japanese.
ENGLISH_ONLY = ('plamo', 'sarashina')
LABELS = {'plamo': 'PLaMo 2B', 'gemma': 'Gemma', 'sarashina': 'Sarashina翻訳',
          'google': 'Google翻訳', 'bing': 'Bing翻訳'}


def is_english(language):
    return language.lower().split('-')[0] == 'en'


class APIPreset(CustomAPIConfig):
    """One saved API connection; ``enabled`` means "use it" in simple mode."""
    id: str = Field(default_factory=lambda: uuid.uuid4().hex, pattern=r'^[0-9a-f]{32}$')
    name: str = Field(default='API', min_length=1, max_length=80)


class Engines(BaseModel):
    plamo: bool = True
    gemma: bool = True
    sarashina: bool = False
    google: bool = False
    bing: bool = False


class TaskRoute(BaseModel):
    # Unknown keys would silently fall back to the default chain.
    model_config = ConfigDict(extra='forbid')
    # Engine names or ``api:<preset id>``, tried first to last.
    chain: list[str] = Field(default_factory=lambda: ['gemma'], max_length=24)


class TaskRoutingConfig(BaseModel):
    enabled: bool = False
    presets: list[APIPreset] = Field(default_factory=list, max_length=16)
    engines: Engines = Field(default_factory=Engines)
    language: TaskRoute = Field(default_factory=TaskRoute)
    translation: TaskRoute = Field(default_factory=lambda: TaskRoute(chain=['plamo', 'gemma']))
    shortening: TaskRoute = Field(default_factory=TaskRoute)

    @model_validator(mode='before')
    @classmethod
    def primary_fallback_format(cls, data):
        """Files saved before presets: each task had primary/fallback and its own API."""
        if not isinstance(data, dict) or not any('primary' in data.get(task, {}) for task in TASKS):
            return data
        data = {**data, 'presets': list(data.get('presets', []))}
        for task in TASKS:
            old = data.get(task)
            if not old or 'primary' not in old:
                continue
            chain = [p for p in dict.fromkeys((old['primary'], old['fallback'])) if p != 'none']
            if 'api' in chain:
                preset_id = hashlib.md5(task.encode()).hexdigest()
                data['presets'].append({**old['api'], 'id': preset_id, 'enabled': False,
                                        'name': {'language': '言語判定のAPI', 'translation': '翻訳のAPI',
                                                 'shortening': '短縮のAPI'}[task]})
                chain = [f'api:{preset_id}' if p == 'api' else p for p in chain]
            data[task] = {'chain': chain}
        return data

    @model_validator(mode='after')
    def valid_chains(self):
        ids = [preset.id for preset in self.presets]
        if len(set(ids)) != len(ids):
            raise ValueError('APIプリセットのIDが重複しています。')
        for preset in self.presets:
            in_use = (any(f'api:{preset.id}' in getattr(self, task).chain for task in TASKS)
                      if self.enabled else preset.enabled)
            if in_use and not preset.model.strip():
                raise ValueError(f'APIプリセット「{preset.name}」のモデルを指定してください。')
        known = {*ENGINES, *(f'api:{i}' for i in ids)}
        for task in TASKS:
            chain = getattr(self, task).chain
            if any(provider not in known for provider in chain):
                raise ValueError('存在しないAPIプリセットまたは使用先が指定されています。')
            if len(set(chain)) != len(chain):
                raise ValueError('同じ使用先を1つの作業に重複して指定できません。')
            if task != 'translation' and any(provider in TRANSLATION_ONLY for provider in chain):
                raise ValueError('翻訳専用モデル・サービスは翻訳の作業にだけ指定できます。')
        for task in ('language', 'translation'):
            if not self.chain(task):
                raise ValueError('言語判定・翻訳に使うAPIプリセット・Gemmaのいずれかを有効にしてください。'
                                 if not self.enabled else '言語判定・翻訳には使用先を指定してください。')
        return self

    def chain(self, task):
        if self.enabled:
            return getattr(self, task).chain
        apis = [f'api:{preset.id}' for preset in self.presets if preset.enabled]
        order = (['plamo', *apis, 'sarashina', 'google', 'bing', 'gemma'] if task == 'translation'
                 else [*apis, 'gemma'])
        return [p for p in order if p.startswith('api:') or getattr(self.engines, p)]

    def translation_chain(self, language):
        return [p for p in self.chain('translation') if is_english(language) or p not in ENGLISH_ONLY]

    def identity(self, task):
        presets = {f'api:{preset.id}': preset.identity for preset in self.presets}
        chain = self.chain(task)
        return [[presets.get(p, p) for p in chain], [PLAMO.identity, PROTOCOL] if 'plamo' in chain else None]

    def public(self):
        return {**self.model_dump(exclude={'presets'}), 'presets': [preset.public() for preset in self.presets]}


def migrate_saved_routes(payload):
    """Replace the retired cloud engine in existing files, keeping the user's order."""
    for task in TASKS:
        route = payload.get(task, {})
        if 'chain' in route:
            route['chain'] = list(dict.fromkeys('gemma' if p == 'pcc' else p for p in route['chain']))
        for key in ('primary', 'fallback'):
            if route.get(key) == 'pcc':
                route[key] = 'gemma'
    payload.get('engines', {}).pop('pcc', None)
    return payload


@credentials.serialized
def load_routes(data):
    path = data / 'task-routing.json'
    config = TaskRoutingConfig.model_validate(migrate_saved_routes(json.loads(path.read_text()))) if path.exists() else TaskRoutingConfig()
    if any(preset.api_key and preset.api_key.get_secret_value() for preset in config.presets):
        # Only remove legacy plaintext after every Keychain write was verified.
        return _persist_routes(data, config, config)
    store = None
    for preset in config.presets:
        if preset.credential_id:
            store = store or credentials.Keychain(data)
            value = store.get(preset.credential_id)
            preset.api_key = SecretStr(value) if value else None
    return config


def _resolve_preset(preset, saved):
    if preset.api_key is None and saved:
        if credential_scope(preset) == credential_scope(saved):
            return preset.model_copy(update={'api_key': saved.api_key, 'credential_id': saved.credential_id})
        if saved.api_key:
            raise credentials.CredentialError('接続先またはAPI形式が変わりました。APIキーを再入力するか、明示的に削除してください。')
    return preset.model_copy(update={'credential_id': None})


@credentials.serialized
def resolve_preset(data, preset):
    saved = next((p for p in load_routes(data).presets if p.id == preset.id), None)
    return _resolve_preset(preset, saved)


def _persist_routes(data, config, previous):
    saved = {preset.id: preset for preset in previous.presets}
    resolved = [_resolve_preset(preset, saved.get(preset.id)) for preset in config.presets]
    created, store = [], None
    try:
        for preset in resolved:
            value = preset.api_key.get_secret_value().strip() if preset.api_key else ''
            preset.api_key = SecretStr(value) if value else None
            if not value:
                preset.credential_id = None
            elif not preset.credential_id:
                store = store or credentials.Keychain(data)
                preset.credential_id = uuid.uuid4().hex
                created.append(preset.credential_id)
                store.put(preset.credential_id, value)
        payload = config.model_dump(exclude={'presets'})
        payload['presets'] = [preset.model_dump(exclude={'api_key'}) for preset in resolved]
        credentials.write_private_json(data / 'task-routing.json', payload)
    except Exception:
        for account in created:
            try:
                store.delete(account)
            except credentials.CredentialError:
                logger.warning('APIキーの一時保存の整理に失敗しました。キーチェーンを確認してください。')
        raise
    active = {preset.credential_id for preset in resolved}
    for account in {p.credential_id for p in previous.presets if p.credential_id} - active:
        try:
            (store or credentials.Keychain(data)).delete(account)
        except credentials.CredentialError:
            logger.warning('古いAPIキーの削除に失敗しました。キーチェーンを確認してください。')
    return config.model_copy(update={'presets': resolved})


@credentials.serialized
def save_routes(data, config):
    return _persist_routes(data, config, load_routes(data))


class RoutedText:
    def __init__(self, router, task):
        self.router, self.task = router, task
        identity = router.config.identity(task)
        self.model = SimpleNamespace(identity='task-v2:' + hashlib.sha256(json.dumps(identity).encode()).hexdigest())

    async def complete(self, content, **options):
        chain = self.router.config.chain(self.task)
        for number, provider in enumerate(chain):
            try:
                return await self.router.client(provider).complete(content, **options)
            except RuntimeError as exc:
                if number == len(chain) - 1:
                    raise
                self.router.log_next(self.task, provider, chain[number + 1], exc)

    async def complete_json(self, content, schema, **options):
        return json.loads(await self.complete(content, schema=schema, **options))


class TaskRouter:
    def __init__(self, settings, config, *, allow_download=False, download_progress=None):
        self.settings, self.config = settings, config
        self.local = GemmaAudio(settings, allow_download=allow_download, download_progress=download_progress)
        self.plamo = PlamoTranslationModel(settings, allow_download=allow_download, download_progress=download_progress)
        self.apis = {f'api:{preset.id}': CustomTextAPI(preset) for preset in config.presets}
        self.names = {f'api:{preset.id}': preset.name for preset in config.presets}
        sarashina = load_sarashina(settings.data)
        self.sarashina_key = sarashina.api_key.get_secret_value() if sarashina.api_key else ''
        self.sarashina_requests = APIRequests()

    def client(self, provider):
        return self.local if provider == 'gemma' else self.apis[provider]

    def label(self, provider):
        return self.names.get(provider) or LABELS[provider]

    def describe(self, chain):
        first, *rest = [self.label(p) for p in chain]
        return first + (f'（失敗時：{" → ".join(rest)}）' if rest else '')

    def log_next(self, task, provider, following, exc):
        logger.warning('task routing %s: %s failed, trying %s: %s',
                       task, self.label(provider), self.label(following), exc)

    async def close(self):
        await self.local.close()
        await self.plamo.close()

    def text(self, task):
        return RoutedText(self, task)

    def shortening_translator(self, *, title, mora_per_second):
        if not self.config.chain('shortening'):
            return None
        return GemmaDubbingTranslator(self.text('shortening'), title=title, mora_per_second=mora_per_second)


class TaskDubbingTranslator(GemmaDubbingTranslator):
    def __init__(self, router, **options):
        super().__init__(router.text('translation'), **options)
        self.router = router
        self.shortener = router.shortening_translator(**options)

    @property
    def identity(self):
        shortening = self.shortener.identity if self.shortener else 'disabled'
        return f'{super().identity}:shortening:{shortening}'

    async def _request_from(self, provider, language, items, before, after, note):
        if provider in EXTERNAL:
            translator = AnonymousDubbingTranslator(self.gemma, provider=provider,
                                                     sarashina_key=self.router.sarashina_key,
                                                     requests=self.router.sarashina_requests,
                                                     title=self.title, mora_per_second=self.rate)
        elif provider == 'plamo':
            translator = PlamoDubbingTranslator(self.router.plamo, title=self.title, mora_per_second=self.rate)
        else:
            translator = GemmaDubbingTranslator(self.router.client(provider),
                                                 title=self.title, mora_per_second=self.rate)
        return await translator._request(language, items, before, after, note=note)

    async def _request(self, language, items, before, after, *, note=''):
        chain = self.router.config.translation_chain(language)
        if not chain:
            raise RuntimeError(f'原文の言語（{language}）に使える翻訳先がありません。'
                               'PLaMo・Sarashinaは英語の原文だけに使います。')
        result, error = {}, None
        for number, provider in enumerate(chain):
            missing = [item for item in items if item.key not in result]
            if not missing:
                break
            if number:
                self.router.log_next('translation', chain[number - 1], provider,
                                     error or f'{len(missing)} lines unanswered')
            try:
                result.update(await self._request_from(provider, language, missing, before, after, note))
                error = None
            except RuntimeError as exc:
                error = exc
        if error is not None and not result:
            raise error
        return result

    async def _condense(self, language, items, current, before, after):
        if self.shortener is None:
            return {}
        return await self.shortener._condense(language, items, current, before, after)
