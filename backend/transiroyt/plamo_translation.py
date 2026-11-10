"""PLaMo 3 Translate 2B, using its native PFML3 text protocol."""
import asyncio

import httpx

from .dubbing_translation import GemmaDubbingTranslator, clean_translation
from .gemma_audio import GemmaAudio, GemmaOutputLimit, TextModel

PLAMO = TextModel('pfnet/plamo-3-translate-2b-GGUF', 'f2b4599dff9b3e45c62567eac15e83186b4cb64e',
                  'plamo-3-translate-2b-Q4_K_M.gguf', 'PLaMo 3 Translate 2B Q4_K_M')
PROTOCOL = 'pfml3-context-v2-en-visible'


class PlamoTranslationModel(GemmaAudio):
    def __init__(self, settings, *, parallel=2, **options):
        super().__init__(settings, model=PLAMO, download_resource='plamo', parallel=parallel, **options)

    async def translate(self, text, context, form, language):
        language = language.split('-')[0].lower()
        if language == 'en':
            # YouTube captions can contain hundreds of word-break markers. PLaMo
            # copies them into its answer, spending tokens on invisible output.
            text = text.replace('\u200b', '').replace('\ufeff', '')
            context = context.replace('\u200b', '').replace('\ufeff', '')
        # PFML3 is the model's trained format; generic JSON/system prompts degrade it.
        prompt = ('<|plamo:tag|>dataset<|plamo:msg|>translation'
                  + ('<|plamo:tag|>context<|plamo:msg|>' + context if context else '')
                  + '<|plamo:tag|>input<|plamo:key|>lang<|plamo:val|>' + language
                  + '<|plamo:msg|>' + text
                  + '<|plamo:tag|>output<|plamo:key|>lang<|plamo:val|>ja'
                  + '<|plamo:key|>form<|plamo:val|>' + form + '<|plamo:msg|>')
        async with self._admit(min(16384, len(prompt.encode()) + 1536)):
            await self.start()
            try:
                async with httpx.AsyncClient(timeout=600, trust_env=False) as client:
                    response = await client.post(self.endpoint + '/v1/completions', headers=self._headers, json={
                        'prompt': prompt, 'temperature': 0, 'max_tokens': 1024,
                        'stop': ['<|plamo:tag|>', '<|plamo:bos|>'], 'cache_prompt': False})
                    response.raise_for_status()
                choice = response.json()['choices'][0]
                if choice['finish_reason'] == 'length':
                    raise GemmaOutputLimit('PLaMoの翻訳が出力上限に達しました。')
                return choice['text']
            except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
                raise RuntimeError('PLaMoの翻訳応答を取得できませんでした。') from exc


class PlamoDubbingTranslator(GemmaDubbingTranslator):
    async def _request(self, language, items, before, after, *, note=''):
        form = 'polite' if self.title and 'Veritasium' in self.title else 'plain'

        async def one(index, item):
            preceding = before + [(i.speaker, i.source) for i in items[:index]]
            following = [(i.speaker, i.source) for i in items[index + 1:]] + after
            context = ('\n'.join(text for _, text in preceding[-6:]) + '\n'
                       + '\n'.join(text for _, text in following[:3])).strip()
            text = await self.gemma.translate(item.source, context, form, language)
            return item.key, clean_translation(text, item.source)

        return dict(await asyncio.gather(*(one(n, item) for n, item in enumerate(items))))
