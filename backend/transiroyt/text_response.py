"""Shared validation for local models and API text responses."""
import json
import re

_JAPANESE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")


class UntranslatedLines(RuntimeError):
    """A well-formed translation answer in which some lines are not Japanese.

    A local model can copy an occasional long English line (LFM3 did); the other lines of the batch
    are still usable, and only the listed keys need another request.
    """

    def __init__(self, result: dict[str, str], keys: list[str]):
        super().__init__("テキストモデルが日本語の訳文を返しませんでした。")
        self.result, self.keys = result, keys


def validate_text_response(raw, schema):
    """Validate structured text once at each inference boundary."""
    if not raw.strip():
        raise RuntimeError("テキストモデルから応答を取得できませんでした。")
    if "<think>" in raw or "</think>" in raw:
        raise RuntimeError("テキストモデルの応答に思考文が混ざりました。")
    if schema is not None:
        try:
            result = json.loads(raw)
        except ValueError as exc:
            raise RuntimeError("テキストモデルの応答をJSONとして解釈できません。") from exc
        properties = schema["properties"]
        if (not isinstance(result, dict) or set(result) != set(properties)
                or any(not isinstance(value, str) or not value.strip() for value in result.values())):
            raise RuntimeError("テキストモデルの応答が要求したテキストの形式と一致しません。")
        if properties and all(re.fullmatch(r"T\d+", key) for key in properties):
            # Numbers and one-letter codes ("550?", Battleship's "B2.") have no
            # Japanese spelling to give; an English word copied over does.
            untranslated = [key for key, value in result.items()
                            if not _JAPANESE.search(value) and re.search(r"[A-Za-z]{2}", value)]
            if untranslated:
                raise UntranslatedLines(result, untranslated)
    return raw
