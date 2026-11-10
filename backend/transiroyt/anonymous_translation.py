"""Anonymous Google/Bing protocols, based on deepseek-harness translator/provider.ts.

Independent implementation; no harness runtime or account is required.
"""
import json

import httpx

from .api_requests import APIRequests
from .dubbing_translation import GemmaDubbingTranslator, clean_translation
from .sarashina_translation import translate_text as sarashina_translate


async def translate_text(client, provider, text, language):
    if len(text.encode("utf-16-le")) // 2 > 4000:
        raise RuntimeError("外部翻訳の文字数上限（4000文字）を超えました。")
    lower = language.lower()
    if lower in ("zh", "zh-cn", "zh-sg", "zh-hans"):
        language = "zh-CN" if provider == "google" else "zh-Hans"
    elif lower in ("zh-tw", "zh-hk", "zh-mo", "zh-hant"):
        language = "zh-TW" if provider == "google" else "zh-Hant"
    if provider == "google":
        url = "https://translate.googleapis.com/translate_a/single"
        kwargs = {"data": {"client": "gtx", "sl": language, "tl": "ja", "dt": "t", "q": text}}
    else:
        url = "https://edge.microsoft.com/translate/translatetext"
        params = {"to": "ja", "isEnterpriseClient": "false"}
        if language != "auto":
            params["from"] = language
        kwargs = {"params": params, "json": [text]}
    try:
        async with client.stream("POST", url, **kwargs) as response:
            response.raise_for_status()
            body = bytearray()
            async for chunk in response.aiter_bytes():
                body.extend(chunk)
                if len(body) > 1024 * 1024:
                    raise RuntimeError("外部翻訳の応答が上限を超えました。")
        payload = json.loads(body)
        if provider == "google":
            result = "".join(segment[0] for segment in payload[0])
        else:
            result = payload[0]["translations"][0]["text"]
        if not isinstance(result, str) or not result.strip():
            raise ValueError("empty translation")
        return result
    except (httpx.HTTPError, ValueError, TypeError, KeyError, IndexError) as exc:
        raise RuntimeError(f"{provider}翻訳に失敗しました（{type(exc).__name__}）。") from exc


class AnonymousDubbingTranslator(GemmaDubbingTranslator):
    def __init__(self, gemma, *, provider, sarashina_key="", requests=None, **kwargs):
        super().__init__(gemma, **kwargs)
        self.provider = provider
        self.sarashina_key = sarashina_key
        self.requests = requests or APIRequests()

    @property
    def identity(self):
        return f"anonymous-v2:{self.provider}:{super().identity}"

    async def _request(self, language, items, before, after, *, note=""):
        result = {}
        async with httpx.AsyncClient(timeout=60 if self.provider == "sarashina" else 10, follow_redirects=False) as client:
            for item in items:
                try:
                    translated = (await sarashina_translate(client, item.source, language, self.sarashina_key,
                                                           requests=self.requests)
                                  if self.provider == "sarashina"
                                  else await translate_text(client, self.provider, item.source, language))
                except RuntimeError:
                    if not result:
                        raise
                    # The router sends only the remaining lines to the next engine.
                    return result
                result[item.key] = clean_translation(translated, item.source)
        return result

    async def _condense(self, language, items, current, before, after):
        # Anonymous translation accepts text only, not instructions to shorten it.
        return {}
