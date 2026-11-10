"""Caption metadata is used, not discarded: speaker markers, reactions, sounds."""
import pytest
from test_file_dubbing import (
    SCRIPT,
    FakeEmbeddings,
    FakeGemma,
    FakeReferenceConverter,
    FakeTranscriber,
    FakeTTS,
    exact_reference_check,  # noqa: F401  (fixture)
    fake_reference_conversion,  # noqa: F401  (autouse fixture)
    fake_separation,  # noqa: F401  (autouse fixture)
    write_srt,
    write_video,
)
from transiroyt.caption_annotations import (
    classify,
    strip_non_speech_annotations,
    voiced_annotations,
    without_reaction_emojis,
)
from transiroyt.captions import Cue, voiced_cues
from transiroyt.config import Settings
from transiroyt.dubbing_options import FileDubbingOptions
from transiroyt.dubbing_translation import (
    GemmaDubbingTranslator,
    TranslationItem,
    clean_translation,
)
from transiroyt.file_dubbing import LAUGHING_VOICE, FileDubber
from transiroyt.sentence_captions import group_sentences
from transiroyt.session_models import Segment
from transiroyt.speech_length import REACTION_SECONDS, natural_seconds
from transiroyt.subtitle_files import caption_events, parse_srt, spoken_cues


@pytest.mark.parametrize("text,expected", [
    ("[laughs] Hello", "🤭 Hello"),
    ("I can't believe that [laughter] worked.", "I can't believe that 🤭 worked."),
    ("Tails. [laughter]", "Tails. 🤭"),
    ("hundreds [snorts] of", "hundreds 🤧 of"),
    ("[sighs] Fine.", "😮‍💨 Fine."),
    ("Wow. [screaming]", "Wow. 😱"),
    ("[man chuckles] Sure.", "🤭 Sure."),
    ("[laughter, applause] Hi", "🤭 Hi"),
    # Others' reactions, music and sounds are never the speaker's voice.
    ("[audience laughs] Welcome back.", "Welcome back."),
    ("Thank you [applause].", "Thank you."),
    ("[Music] Hello", "Hello"),
    ("[door slams] Who's there?", "Who's there?"),
    # Unknown brackets are still words.
    ("[whispering] Quiet", "[whispering] Quiet"),
    ("[Alice] Hello", "[Alice] Hello"),
])
def test_speaker_reactions_become_irodori_emojis(text, expected):
    assert voiced_annotations(text)[0] == expected
    # Recognized annotations are never read; the unvoiced form has no emoji.
    assert strip_non_speech_annotations(text) == without_reaction_emojis(expected)


@pytest.mark.parametrize("text", ["[laughter]", "[snorts]", "[clears throat]", "[sighs]", "LAUGHTER"])
def test_reaction_only_captions_are_not_lines(text):
    spoken, annotations = voiced_annotations(text)
    assert spoken == "" and annotations
    assert strip_non_speech_annotations(text) == ""


def test_classification_kinds():
    assert [(a.label, a.kind, a.emoji) for a in classify("laughs")] == [("laughter", "vocal", "🤭")]
    assert [(a.label, a.kind) for a in classify("audience laughs")] == [("audience laughs", "audience")]
    assert [(a.label, a.kind) for a in classify("Applause")] == [("applause", "audience")]
    assert [(a.label, a.kind) for a in classify("soft piano music playing")] == [("music", "music")]
    assert [(a.label, a.kind) for a in classify("thunder")] == [("thunder", "sound")]
    assert [(a.label, a.kind) for a in classify("笑い")] == [("laughter", "vocal")]
    assert classify("I love music") is None and classify("Music and dialogue") is None


def test_marker_before_a_reaction_still_changes_speaker_and_events_are_kept():
    srt = ("1\n00:00:00,000 --> 00:00:02,000\nIt's Brad cuz you're a villain.\n\n"
           "2\n00:00:02,000 --> 00:00:03,000\n>> [laughter]\n\n"
           "3\n00:00:03,000 --> 00:00:04,000\nNext [snorts] game.\n\n"
           "4\n00:00:04,000 --> 00:00:06,000\nThank you [applause].\n\n"
           "5\n00:00:06,000 --> 00:00:07,000\n[Music]\n")
    cues = spoken_cues(parse_srt(srt))
    assert [(c.text, c.speaker_change) for c in cues] == [
        ("It's Brad cuz you're a villain.", False), ("Next 🤧 game.", True), ("Thank you.", False)]
    events = caption_events(parse_srt(srt))
    # The marker's own share of the caption is proportional, as for words.
    assert [(e.start, e.end, e.labels, e.kinds) for e in events] == [
        (pytest.approx(2 + 1 / 11), 3.0, ["laughter"], ["vocal"]), (4.0, 6.0, ["applause"], ["audience"]),
        (6.0, 7.0, ["music"], ["music"])]


def test_cues_split_speakers_and_keep_sound_boundaries():
    cues = voiced_cues([Cue(0, 2, "Hello there >> Hi"), Cue(2, 3, "[applause]"),
                        Cue(3, 4, ">> [laughs]"), Cue(4, 5, "okay then")])
    assert [(c.text, c.speaker_change) for c in cues] == [
        ("Hello there", False), ("Hi", True), ("[applause]", False), ("[laughs]", False),
        ("okay then", True)]
    # Words of two speakers never merge into one sentence.
    assert [c.text for c in group_sentences([Cue(0, 1, "So the"), Cue(1, 2, "answer", speaker_change=True)])] == [
        "So the", "answer"]


def test_a_reaction_after_a_full_stop_stays_with_its_sentence():
    assert group_sentences([Cue(0, 2, "Tails. 🤭")]) == [Cue(0, 2, "Tails. 🤭")]
    # Between two sentences it leads the next one, as captioned.
    assert [c.text for c in group_sentences([Cue(0, 4, "Yes. 🤭 Next one.")])] == ["Yes.", "🤭 Next one."]


def test_reaction_time():
    assert natural_seconds("はい。🤭", 6.4) == pytest.approx(natural_seconds("はい。", 6.4) + REACTION_SECONDS)


def test_translation_keeps_only_the_source_reactions_and_no_labels():
    assert clean_translation("[S22] 裏。🤭 〔laughter〕", "Tails. 🤭") == "裏。🤭"
    # A reaction the source does not have is not invented; 😮‍💨 is not 😮.
    assert clean_translation("えっ😮‍💨😱", "What? 😮") == "えっ"
    assert clean_translation("えっ😮", "What? 😮") == "えっ😮"


async def test_translator_sees_reactions_and_what_is_heard_after_a_line():
    class Gemma:
        prompts = []

        async def complete_json(self, content, schema, **_):
            self.prompts.append(content[0]["text"])
            return {key: "裏だ。🤭" for key in schema["properties"]}

    gemma = Gemma()
    translator = GemmaDubbingTranslator(gemma, title=None, mora_per_second=6.4)
    item = TranslationItem(3, "Tails. 🤭", "S2", 1.0 + REACTION_SECONDS, ("applause",))
    assert await translator.translate_batch("en", [item], [], []) == {3: "裏だ。🤭"}
    prompt = gemma.prompts[0]
    assert "T3 [S2] (1.4秒以内・6文字以内): Tails. 🤭 〔applause〕" in prompt
    assert "🤭=笑い" in prompt and "〔 〕" in prompt


@pytest.mark.usefixtures("exact_reference_check")
async def test_file_dubbing_voices_reactions_and_gives_events_to_translation(tmp_path):
    script = [
        ("A", 0.5, 3.0, "[laughs] Welcome to the show, everyone."),
        ("B", 3.2, 5.6, ">> Thanks, it is great to be here. [applause]"),
        ("A", 5.8, 8.2, ">> Tell us about your new project."),
        ("B", 8.4, 11.4, ">> We built a small boat out of wood."),
        ("B", 11.8, 13.9, "It took three years to finish. [sighs]"),
        ("A", 14.2, 16.6, ">> That sounds like hard work."),
        ("B", 16.6, 17.2, ">> [laughter]"),
    ]
    video, srt, output = tmp_path / "in.mp4", tmp_path / "in.srt", tmp_path / "out.mp4"
    write_video(video)
    write_srt(srt, script)

    class Translator:
        identity = "reaction-test"
        items = []

        async def translate_batch(self, language, items, before, after):
            self.items += items
            return {item.key: "訳。" + "".join(e for e in ("🤭", "😮‍💨") if e in item.source) for item in items}

    tts, translator = FakeTTS(), Translator()
    report = await FileDubber(
        Settings(data=tmp_path / "data"), FileDubbingOptions(video, srt, output),
        gemma=FakeGemma(tts=tts), embeddings=FakeEmbeddings(SCRIPT), translator=translator, tts=tts,
        transcriber=FakeTranscriber(tts),
        reference_converter=FakeReferenceConverter(failure="このテストでは参照変換しない")).run()

    sources = {item.source: item.after for item in translator.items}
    assert sources["🤭 Welcome to the show, everyone."] == ()
    assert sources["Thanks, it is great to be here."] == ("applause",)
    assert sources["It took three years to finish. 😮‍💨"] == ()
    assert sources["That sounds like hard work."] == ("laughter",)
    assert not any(">>" in source or "[" in source for source in sources)
    assert {call["text"] for call in tts.calls} >= {"訳。🤭", "訳。😮‍💨"}
    # The reaction has its own time on top of the words.
    laugh = next(call for call in tts.calls if call["text"] == "訳。🤭")
    assert laugh["seconds"] == pytest.approx(natural_seconds("訳。", 6.4) + REACTION_SECONDS, abs=1e-3)
    # The default calm-narration caption is replaced only for the laughing line.
    assert laugh["voice"] == LAUGHING_VOICE
    assert all(call["voice"] is None for call in tts.calls if "🤭" not in call["text"])
    assert report["segments"][0]["style_caption"] == LAUGHING_VOICE
    assert report["summary"]["source_reactions"] == 2 and report["summary"]["dub_reactions"] == 2
    first = report["segments"][0]
    assert first["source_reactions"] == ["🤭"] and first["dub_reactions"] == ["🤭"]
    assert report["segments"][1]["heard_after"] == ["applause"]
    # A clip with a captioned laugh is not used as the speaker's reference voice.
    rejected = {item["line"]: item["reason"] for ref in report["speakers"].values() for item in ref["rejected"]}
    assert rejected[0] == "nonverbal_sound"


def test_a_voice_chosen_by_the_user_is_never_replaced_for_laughter(tmp_path):
    segment = Segment(0, 0, 2, "Tails. 🤭", text="裏だ。🤭")
    default = FileDubber.__new__(FileDubber)
    default.settings = Settings(data=tmp_path)
    assert default._voice(segment) == LAUGHING_VOICE
    assert default._voice(Segment(0, 0, 2, "Tails.", text="裏だ。")) == Settings.voice
    custom = FileDubber.__new__(FileDubber)
    custom.settings = Settings(data=tmp_path, voice="元気な若い女性の声。")
    assert custom._voice(segment) == "元気な若い女性の声。"
