"""Translation review: copies of the source or the prompt are retried alone, then dropped; long lines condensed."""
import pytest
from transiroyt.dubbing_translation import GemmaDubbingTranslator, TranslationItem
from transiroyt.gemma_audio import GEMMA
from transiroyt.text_response import UntranslatedLines


class TextEngine:
    """Answers by request kind: the batch, a line asked alone, and the condensed text."""
    model = GEMMA

    def __init__(self, first, alone=None, condensed=None):
        self.first, self.alone, self.condensed = first, alone or {}, condensed or {}
        self.prompts = []

    async def complete_json(self, content, schema, **_):
        prompt = content[0]["text"]
        self.prompts.append(prompt)
        keys = [int(key[1:]) for key in schema["properties"]]
        table = self.condensed if "読み切れません" in prompt else self.alone if len(self.prompts) > 1 else self.first
        return {f"T{key}": table[key] for key in keys if key in table}


def translator(engine):
    return GemmaDubbingTranslator(engine, title=None, mora_per_second=6.4)


@pytest.mark.parametrize("copy", [
    "我们总算也是解释清楚了为什么8EE6的中核同频性能会开倒车而它的功耗又为什么会变高了",
    "(3.2秒以内・21文字以内): ステップ1、メモリとGPUが高い。",
    # LFM3 joined neighbouring turns or the context listing into one answer.
    "ようやく説明できた。 [S1] そうね、速く良くなったから。",
    "[T1]: ようやく説明できた。",
    "直後の台詞（参考、訳さない）: [S1] It's actually also hard to control.",
])
async def test_a_copy_in_a_batch_is_asked_alone_and_the_real_translation_kept(copy):
    source = "我们总算也是解释清楚了 为什么8EE6的中核 同频性能会开倒车 而它的功耗又为什么会变高了"
    engine = TextEngine({1: copy, 2: "次の行。"}, alone={1: "ようやく説明できた。"})
    t = translator(engine)
    items = [TranslationItem(1, source, "S1", 9.0), TranslationItem(2, "Next.", "S1", 2.0)]
    assert await t.translate_batch("zh", items, [], []) == {1: "ようやく説明できた。", 2: "次の行。"}
    # Only the copied line is asked again; the batch's other line is kept.
    assert len(engine.prompts) == 2 and "T1 " in engine.prompts[1] and "T2 " not in engine.prompts[1]


async def test_a_line_that_copies_again_alone_is_dropped_with_its_reason():
    echo = "もう少しだけ。 [S1] そうね、速く良くなったから。 [S1] もっと理解したんだよね。"
    t = translator(TextEngine({1: echo}))
    assert await t.translate_batch("en", [TranslationItem(1, "Maybe a little more.", "S1", 2.5)], [], []) == {}
    assert t.checks[1] == {"copied_source": True}


@pytest.mark.parametrize(("condensed", "kept"), [
    ("短い。", "短い。"),
    # Not shorter, or an echo of the prompt: the full translation stays.
    ("これはとても長い訳文で読み切れないほどの長さがあって、さらに長くなりました。", None),
    ("(1.0秒以内・6文字以内): 短い。", None),
])
async def test_condensing_replaces_a_long_line_only_with_a_shorter_translation(condensed, kept):
    full = "これはとても長い訳文で読み切れないほどの長さがあります。"
    t = translator(TextEngine({1: full}, condensed={1: condensed}))
    result = await t.translate_batch("en", [TranslationItem(1, "This is long.", "S1", 1.0)], [], [])
    assert result == {1: kept or full}
    assert t.checks.get(1, {}).get("condensed", False) is (kept is not None)


async def test_a_line_without_any_japanese_answer_records_why():
    class English(TextEngine):
        async def complete_json(self, content, schema, **options):
            raise UntranslatedLines({"T1": "Right."}, ["T1"])

    t = translator(English({}))
    assert await t.translate_batch("en", [TranslationItem(1, "Right.", "S1", 2.0)], [], []) == {}
    assert t.checks[1] == {"error": "no Japanese answer"}
