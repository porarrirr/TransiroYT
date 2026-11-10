import array
import asyncio
import base64
import hashlib
import json
import logging
import math
import os
import shutil
import subprocess
import wave
from pathlib import Path

import pytest
from transiroyt import file_dubbing, speaker_diarization, speech_check
from transiroyt.captions import Cue
from transiroyt.config import Settings
from transiroyt.dubbing_captions import speaker_segments
from transiroyt.dubbing_media import place_segments
from transiroyt.dubbing_options import FileDubbingOptions
from transiroyt.errors import FileDubbingError
from transiroyt.file_dubbing import FileDubber
from transiroyt.gemma_audio import GEMMA, GemmaOutputLimit
from transiroyt.japanese_reading import align_latin_spellings
from transiroyt.reference_conversion import CONVERSION_TEXT, REFERENCE_CONVERSION_VERSION
from transiroyt.session_models import Segment
from transiroyt.speaker_diarization import (
    AnalysisAudio,
    Similarity,
    SpeakerActivity,
    SpeakerDiarizer,
    SpeakerLine,
    cluster,
)
from transiroyt.speech_check import (
    compare,
    numbers_heard,
    same_reading,
    speech_verified,
    transcript_coverage,
    transcript_ratio,
)
from transiroyt.speech_length import mora_count, natural_seconds
from transiroyt.subtitle_files import SubtitleError, parse_srt, spoken_cues
from transiroyt.whisper_transcription import TranscriptOutputLimit, WhisperUnavailable

# Two voices, with lines in seconds; the fake embedding model knows who speaks when.
# One speaker's two lines stay 0.4 s apart, above MERGE_GAP_SECONDS: how short
# sentences join is covered on its own below, and these tests count six lines.
SCRIPT = [
    ("A", 0.5, 3.0, "Welcome to the show, everyone."),
    ("B", 3.2, 5.6, "Thanks, it is great to be here."),
    ("A", 5.8, 8.2, "Tell us about your new project."),
    ("B", 8.4, 11.4, "We built a small boat out of wood."),
    ("B", 11.8, 13.9, "It took three years to finish."),
    ("A", 14.2, 16.6, "That sounds like hard work."),
]


def srt_time(seconds):
    ms = round(seconds * 1000)
    return f"{ms // 3600000:02}:{ms // 60000 % 60:02}:{ms // 1000 % 60:02},{ms % 1000:03}"


def write_srt(path, script=SCRIPT):
    path.write_text("\n".join(f"{i + 1}\n{srt_time(a)} --> {srt_time(b)}\n{text}\n"
                              for i, (_, a, b, text) in enumerate(script)), encoding="utf-8")


def write_video(path, seconds=18.0, frequency=440):
    # A 440 Hz tone at a known level stands in for the original soundtrack.
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y",
                    "-f", "lavfi", "-i", f"color=c=black:s=160x90:r=10:d={seconds}",
                    "-f", "lavfi", "-i", f"sine=f={frequency}:d={seconds}:sample_rate=48000",
                    "-filter_complex", "[1:a]volume=0.5[a]", "-map", "0:v", "-map", "[a]",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
                    "-shortest", str(path)], check=True)


def voice_at(script, start, end):
    middle = (start + end) / 2
    return min(script, key=lambda line: 0 if line[1] <= middle <= line[2]
               else min(abs(line[1] - middle), abs(line[2] - middle)))[0]


class FakeEmbeddings:
    def __init__(self, script=SCRIPT, overrides=None):
        self.script, self.overrides, self.calls = script, overrides or {}, []

    async def similarity(self, audio, items, work):
        self.calls.append(items)
        voices = [voice_at(self.script, a, b) for a, b in items]
        values = array.array("f")
        for i, x in enumerate(voices):
            for j, y in enumerate(voices):
                values.append(self.overrides.get((i, j), self.overrides.get((j, i), 1.0 if i == j
                              else 0.8 if x == y else 0.05)))
        stats = [{"rms": 0.1, "peak": 0.5, "clipped": 0.0, "speech_db": -20.0, "floor_db": -55.0}
                 for _ in items]
        return Similarity(values, len(items)), stats

    async def activity(self, audio, work):
        """Nemotron stand-in: each script voice is active (0.9) over its own lines, in order of arrival."""
        order = list(dict.fromkeys(voice for voice, *_ in self.script))
        frames = round(max(end for _, _, end, _ in self.script) * 100) + 100
        values = array.array("f", [0.0] * frames * len(order))
        for voice, start, end, _ in self.script:
            for frame in range(round(start * 100), round(end * 100)):
                values[frame * len(order) + order.index(voice)] = 0.9
        return SpeakerActivity(values, len(order))


class FakeGemma:
    model = GEMMA

    def __init__(self, tts=None, context=None):
        self.tts, self.context = tts, context or {}
        self.prompts = []

    async def complete_json(self, content, schema, **_):
        prompt = next(part["text"] for part in content if part["type"] == "text")
        self.prompts.append(prompt)
        if set(schema['properties']) == {'language'}:
            return {"language": "en"}
        assert not prompt.startswith("Transcribe the speech exactly"), "Reference ASR must use Whisper"
        if "吹き替え台本の翻訳者" in prompt:
            self.translations = getattr(self, "translations", 0) + 1
            return {key: f"訳{key}。" for key in schema["properties"]}
        return {key: "test" if key.endswith("_basis") else self.context.get(key, "unknown")
                for key in schema["properties"]}


class FakeTranscriber:
    def __init__(self, tts=None, garbled_seeds=(), looping_seeds=()):
        self.tts, self.garbled_seeds, self.looping_seeds = tts, set(garbled_seeds), set(looping_seeds)

    async def transcribe(self, wav, expected, *, language="ja"):
        if isinstance(wav, bytes):
            return expected
        # Identify the actual audio under test, including when the next
        # turn has already been synthesized while this check was waiting.
        with wave.open(str(wav)) as audio:
            samples = array.array("h", audio.readframes(audio.getnframes()))
        amplitude = math.sqrt(2 * sum(x * x for x in samples) / len(samples))
        call = min(self.tts.calls, key=lambda c: abs(c["amplitude"] - amplitude))
        if call["seed"] in self.looping_seeds:
            raise TranscriptOutputLimit("looping transcript")
        return "まったく別の内容" if call["seed"] in self.garbled_seeds else call["text"]


class FakeTranslator:
    identity = "fake-dub-translator"

    def __init__(self):
        self.batches = []

    async def translate_batch(self, language, items, before, after):
        assert language == "en"
        self.batches.append((items, before, after))
        return {item.key: f"訳{len(item.source)}。" for item in items}


class FakeSeparator:
    """The whole test soundtrack counts as speech; the background is silent."""

    def __init__(self):
        self.calls = 0

    async def separate(self, video, directory):
        self.calls += 1
        directory.mkdir(parents=True, exist_ok=True)
        vocals, background = directory / "vocals.wav", directory / "background.wav"
        if not vocals.is_file():
            for target, gain in ((vocals, 1), (background, 0)):
                subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(video),
                                "-vn", "-ac", "2", "-ar", "44100", "-af", f"volume={gain}", str(target)], check=True)
        return vocals, background


@pytest.fixture(autouse=True)
def fake_separation(monkeypatch):
    separator = FakeSeparator()
    monkeypatch.setattr(file_dubbing, "VocalSeparator", lambda *args, **kwargs: separator)
    return separator


class FakeTTS:
    def __init__(self):
        self.calls = []

    async def synthesize(self, text, duration, output, reference=None, steps=None, seconds=None, seed=None,
                         voice=None):
        amplitude = 8000 + 100 * len(self.calls)
        self.calls.append({"text": text, "reference": reference, "steps": steps, "seconds": seconds,
                           "seed": seed, "amplitude": amplitude, "voice": voice})
        frames = round(seconds * 48000)
        with wave.open(str(output), "wb") as wav:
            wav.setparams((1, 2, 48000, 0, "NONE", "not compressed"))
            wav.writeframes(array.array("h", (round(amplitude * math.sin(i * 0.05))
                                              for i in range(frames))).tobytes())
        return {"inference_seconds": 0.01}

    async def close(self):
        pass


CONVERSION_SECONDS = 12.0


def write_conversion_tone(path, seconds=CONVERSION_SECONDS, rate=48000):
    with wave.open(str(path), "wb") as wav:
        wav.setparams((1, 2, rate, 0, "NONE", "not compressed"))
        wav.writeframes(array.array("h", (round(6000 * math.sin(i * 0.05))
                                          for i in range(round(seconds * rate)))).tobytes())


class FakeReferenceConverter:
    """IndexTTS-2.5 stand-in: a tone in place of the fixed sentence, or a worker failure."""

    def __init__(self, *, failure=None, seconds=CONVERSION_SECONDS, rate=48000):
        self.failure, self.seconds, self.rate = failure, seconds, rate
        self.converted, self.closed = [], 0

    async def convert(self, reference, output):
        if self.failure:
            raise RuntimeError(self.failure)
        write_conversion_tone(output, self.seconds, self.rate)
        self.converted.append((reference, output))
        return {"seconds": self.seconds}

    async def close(self):
        self.closed += 1


@pytest.fixture(autouse=True)
def fake_reference_conversion(monkeypatch):
    """No test loads IndexTTS-2.5 or its weights: the stand-in fails unless a test injects one."""
    def factory(settings):
        return FakeReferenceConverter(failure="参照変換モデルを読み込めません")

    monkeypatch.setattr(file_dubbing, "ReferenceConverter", factory)
    monkeypatch.setattr(file_dubbing, "model_ready", lambda settings: True)
    return factory


class ConversionTranscriber(FakeTranscriber):
    """Whisper stand-in that hears the fixed conversion sentence, and the rest as FakeTranscriber."""

    async def transcribe(self, wav, expected, *, language="ja"):
        if expected == CONVERSION_TEXT:
            return expected
        return await super().transcribe(wav, expected, language=language)


class ConversionEmbeddings(FakeEmbeddings):
    """Fake ECAPA that hears the converted reference as the same voice as the original."""

    conversion_similarity = 0.9

    async def similarity(self, audio, items, work):
        if audio.parent.name == file_dubbing.CONVERSION_CACHE_DIRECTORY:
            values = array.array("f")
            for i in range(2):
                for j in range(2):
                    values.append(1.0 if i == j else self.conversion_similarity)
            return Similarity(values, 2), [{} for _ in items]
        return await super().similarity(audio, items, work)


async def test_identical_audio_reuses_all_models_across_copied_and_remuxed_inputs(tmp_path, exact_reference_check, monkeypatch):
    video, subtitles = tmp_path / "first.mp4", tmp_path / "first.srt"
    write_video(video)
    write_srt(subtitles)
    settings = Settings(data=tmp_path / "data")
    tts = FakeTTS()
    gemma = FakeGemma(tts)
    embeddings = FakeEmbeddings()
    fitted = []
    original_fit = file_dubbing.fit_audio

    async def fit(*args):
        fitted.append(args[1])
        return await original_fit(*args)

    monkeypatch.setattr(file_dubbing, "fit_audio", fit)
    first = await FileDubber(settings, FileDubbingOptions(video, subtitles, tmp_path / "first-out.mp4"),
                             tts=tts, gemma=gemma, translator=FakeTranslator(), embeddings=embeddings,
                             transcriber=FakeTranscriber(tts)).run()
    first_fit_count = len(fitted)
    original_waveform = fitted[0].read_bytes()

    class NoModels:
        identity = FakeTranslator.identity

        async def unavailable(self, *args, **kwargs):
            raise AssertionError("A cached rerun must not call any model")

        translate_batch = synthesize = complete_json = similarity = separate = transcribe = unavailable

    no_models = NoModels()
    for remux in (False, True):
        folder = tmp_path / ("remux" if remux else "copy")
        folder.mkdir()
        second = folder / "source.mp4"
        if remux:
            subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(video), "-map", "0", "-c", "copy",
                            "-metadata", "title=remuxed input", str(second)], check=True)
            assert second.read_bytes() != video.read_bytes()
        else:
            shutil.copyfile(video, second)
        srt = folder / "source.srt"
        shutil.copyfile(subtitles, srt)
        result = await FileDubber(settings, FileDubbingOptions(second, srt, folder / "out.mp4"),
                                  tts=no_models, gemma=no_models, translator=no_models, embeddings=no_models,
                                  transcriber=no_models).run()
        assert result["segments"] == first["segments"]
        assert len(fitted) == first_fit_count

    damaged = bytearray(fitted[0].read_bytes())
    damaged[-1] ^= 1
    fitted[0].write_bytes(damaged)
    result = await FileDubber(settings, FileDubbingOptions(video, subtitles, tmp_path / "repaired.mp4"),
                              tts=no_models, gemma=no_models, translator=no_models, embeddings=no_models,
                              transcriber=no_models).run()
    assert len(fitted) == first_fit_count + 1
    assert fitted[0].read_bytes() == original_waveform
    assert result["segments"] == first["segments"]

    # A style change reuses the speaker analysis and translation, but must
    # generate new speech even though all other inputs are identical.
    previous_calls = len(tts.calls)
    custom = Settings(data=settings.data, voice="違う音声スタイル")
    await FileDubber(custom, FileDubbingOptions(video, subtitles, tmp_path / "custom-style.mp4"),
                     tts=tts, gemma=gemma, translator=no_models, embeddings=no_models,
                     transcriber=FakeTranscriber(tts)).run()
    assert len(tts.calls) == previous_calls + first["summary"]["ready"]


async def test_content_cache_does_not_trust_equal_size_and_mtime(tmp_path):
    video, changed, subtitles = tmp_path / "source.mp4", tmp_path / "changed.mp4", tmp_path / "source.srt"
    write_video(video, seconds=2, frequency=440)
    write_video(changed, seconds=2, frequency=880)
    # MP4 allows trailing bytes; make metadata identical while audio differs.
    size = max(video.stat().st_size, changed.stat().st_size)
    for path in (video, changed):
        with path.open("ab") as handle:
            handle.write(b"\0" * (size - path.stat().st_size))
    subtitles.write_text("subtitle identity")
    dubber = FileDubber(Settings(data=tmp_path / "data"),
                       FileDubbingOptions(video, subtitles, tmp_path / "out.mp4"))
    saved = video.stat()
    before = await dubber._work_directory("same-subtitles")
    shutil.copyfile(changed, video)
    os.utime(video, ns=(saved.st_atime_ns, saved.st_mtime_ns))
    assert video.stat().st_size == saved.st_size and video.stat().st_mtime_ns == saved.st_mtime_ns
    after = await dubber._work_directory("same-subtitles")
    assert before.resolve() != after.resolve()
    assert file_dubbing._audio_identity(before / "audio48k.wav") != file_dubbing._audio_identity(after / "audio48k.wav")


async def test_exact_content_cache_does_not_decode_again(tmp_path, monkeypatch):
    video, subtitles = tmp_path / "source.mp4", tmp_path / "source.srt"
    write_video(video, seconds=2)
    subtitles.touch()
    dubber = FileDubber(Settings(data=tmp_path / "data"),
                       FileDubbingOptions(video, subtitles, tmp_path / "out.mp4"))
    directory = await dubber._work_directory("same-subtitles")

    async def forbidden_decode(*args):
        raise AssertionError("An exact repeat must use its byte index without decoding again")

    monkeypatch.setattr(file_dubbing, "extract_audio", forbidden_decode)
    assert await dubber._work_directory("same-subtitles") == directory


async def test_audio_cache_keeps_different_subtitles_and_speakers_separate_but_shares_style_analysis(tmp_path):
    video, subtitles = tmp_path / "source.mp4", tmp_path / "source.srt"
    write_video(video, seconds=2)
    subtitles.touch()
    directories = []
    for digest, speakers, voice in [("a", None, Settings.voice), ("b", None, Settings.voice),
                                    ("a", 2, Settings.voice), ("a", None, "違う音声スタイル")]:
        dubber = FileDubber(Settings(data=tmp_path / "data", voice=voice),
                           FileDubbingOptions(video, subtitles, tmp_path / "out.mp4", speakers=speakers))
        directories.append((await dubber._work_directory(digest)).resolve())
    assert len(set(directories)) == 3
    assert directories[0] == directories[3]


@pytest.mark.parametrize("cancel", [False, True])
async def test_synthesis_lookahead_is_bounded_and_drained(tmp_path, monkeypatch, cancel):
    video, subtitles = tmp_path / "v.mp4", tmp_path / "v.srt"
    video.touch()
    subtitles.touch()
    dubber = FileDubber(Settings(data=tmp_path),
                       FileDubbingOptions(video, subtitles, tmp_path / "out.mp4"), tts=FakeTTS())
    limit = file_dubbing.SYNTHESIS_LOOKAHEAD
    segments = [Segment(i, i * 3, i * 3 + 2, str(i), text=f"発話{i}") for i in range(limit * 2)]
    entered, release, drained = asyncio.Event(), asyncio.Event(), asyncio.Event()
    started, finished = [], []

    async def generate(tts, segment, reference, steps, stem):
        started.append(segment.index)
        if len(started) == limit:
            entered.set()
        try:
            await release.wait()
            await tts.synthesize(segment.text, 1, stem.with_name(stem.name + ".natural.wav"),
                                 seconds=1, seed=42)
            return {"ratio": 1.0, "coverage": 1.0, "attempts": [], "seconds": 1, "transcript": segment.text, "tail": {}}
        finally:
            finished.append(segment.index)
            if len(finished) == limit:
                drained.set()

    monkeypatch.setattr(dubber, "_generate_verified", generate)
    task = asyncio.create_task(dubber._synthesize(segments, ["S1"] * len(segments),
                                                 {"S1": tmp_path / "reference.wav"}, 10, tmp_path))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        assert started == list(range(limit)) and all(s.state == "pending" for s in segments)
        if cancel:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert drained.is_set() and sorted(finished) == list(range(limit))
        else:
            release.set()
            await task
            assert started == list(range(limit * 2)) and all(s.state == "ready" for s in segments)
            assert all(a.speech_end <= b.speech_start for a, b in zip(segments, segments[1:], strict=False))
    finally:
        release.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def test_synthesis_lookahead_grows_with_mac_generators_and_paired_devices(tmp_path, monkeypatch):
    from transiroyt.mobile_workers import Device, MobileWorkers

    video, subtitles = tmp_path / "v.mp4", tmp_path / "v.srt"
    video.touch()
    subtitles.touch()
    tts = FakeTTS()
    tts.capacity = 2
    pool = MobileWorkers(tmp_path)
    for name in ("phone", "pad"):
        pool.devices[name] = Device(name, name, f"{name}-token")
    dubber = FileDubber(Settings(data=tmp_path), FileDubbingOptions(video, subtitles, tmp_path / "out.mp4"),
                        tts=tts, mobile=pool)
    segments = [Segment(i, i * 3, i * 3 + 2, str(i), text=f"発話{i}") for i in range(30)]
    started, release = [], asyncio.Event()

    async def generate(tts, segment, reference, steps, stem):
        started.append(segment.index)
        await release.wait()

    monkeypatch.setattr(dubber, "_generate_verified", generate)
    task = asyncio.create_task(dubber._synthesize(segments, ["S1"] * len(segments),
                                                 {"S1": tmp_path / "reference.wav"}, 10, tmp_path))
    try:
        for _ in range(20):
            await asyncio.sleep(0)
        # Two Mac generators and two devices: three lines queued for each.
        assert started == list(range(file_dubbing.LOOKAHEAD_PER_EXECUTOR * 4))
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def test_synthesis_duplicate_turns_share_generation_and_failure_does_not_abort(tmp_path, monkeypatch):
    video, subtitles = tmp_path / "v.mp4", tmp_path / "v.srt"
    video.touch()
    subtitles.touch()
    dubber = FileDubber(Settings(data=tmp_path),
                       FileDubbingOptions(video, subtitles, tmp_path / "out.mp4"), tts=FakeTTS())
    segments = [Segment(i, i * 3, i * 3 + 2, text, text=text)
                for i, text in enumerate(["同じ", "同じ", "失敗", "続き"])]
    calls = []

    async def generate(tts, segment, reference, steps, stem):
        calls.append(segment.text)
        await asyncio.sleep(0)
        if segment.text == "失敗":
            raise RuntimeError("test generation error")
        await tts.synthesize(segment.text, 1, stem.with_name(stem.name + ".natural.wav"), seconds=1, seed=42)
        return {"ratio": 1.0, "coverage": 1.0, "attempts": [], "seconds": 1, "transcript": segment.text, "tail": {}}

    monkeypatch.setattr(dubber, "_generate_verified", generate)
    await dubber._synthesize(segments, ["S1"] * 4, {"S1": tmp_path / "reference.wav"}, 10, tmp_path)
    assert calls == ["同じ", "失敗", "続き"]
    assert [s.state for s in segments] == ["ready", "ready", "skipped", "ready"]
    assert segments[2].skip_reason == "synthesis_failed"


async def test_missing_transcript_model_stops_synthesis_instead_of_skipping_each_turn(tmp_path):
    video, subtitles = tmp_path / "v.mp4", tmp_path / "v.srt"
    video.touch()
    subtitles.touch()

    class Transcriber:
        async def transcribe(self, raw, expected):
            raise WhisperUnavailable("model missing")

    tts = FakeTTS()
    dubber = FileDubber(Settings(data=tmp_path), FileDubbingOptions(video, subtitles, tmp_path / "out.mp4"),
                        tts=tts, transcriber=Transcriber())
    segments = [Segment(i, i * 4, i * 4 + 3, text, text=text) for i, text in enumerate(["最初。", "次。"])]
    with pytest.raises(WhisperUnavailable):
        await dubber._synthesize(segments, ["S1"] * 2, {"S1": tmp_path / "reference.wav"}, 10, tmp_path)
    assert all(s.state == "pending" for s in segments)


async def test_next_turn_is_generated_while_first_transcript_is_pending(tmp_path, monkeypatch):
    video, subtitles = tmp_path / "v.mp4", tmp_path / "v.srt"
    video.touch()
    subtitles.touch()
    checking, generated_next = asyncio.Event(), asyncio.Event()

    class TTS(FakeTTS):
        async def synthesize(self, text, *args, **kwargs):
            if text == "次の発話です。":
                await checking.wait()
                generated_next.set()
            return await super().synthesize(text, *args, **kwargs)

    class Transcriber:
        async def transcribe(self, raw, expected):
            if expected == "最初の発話です。":
                checking.set()
                await asyncio.wait_for(generated_next.wait(), 2)
            return expected

    tts = TTS()
    dubber = FileDubber(Settings(data=tmp_path),
                       FileDubbingOptions(video, subtitles, tmp_path / "out.mp4"), tts=tts,
                       transcriber=Transcriber())
    segments = [Segment(i, i * 4, i * 4 + 3, text, text=text)
                for i, text in enumerate(["最初の発話です。", "次の発話です。"])]
    await asyncio.wait_for(dubber._synthesize(segments, ["S1"] * 2,
                                             {"S1": tmp_path / "reference.wav"}, 10, tmp_path), 5)
    assert all(s.state == "ready" and s.audio["verified"] for s in segments)
    assert [call["seed"] for call in tts.calls] == [42, 42]


@pytest.fixture
def exact_reference_check(monkeypatch):
    original = transcript_ratio

    def ratio(expected, heard):
        return 1.0 if heard == "REFERENCE_OK" else original(expected, heard)
    monkeypatch.setattr(speaker_diarization, "transcript_ratio", ratio)


def test_srt_parsing_tags_dialogue_dashes_and_errors():
    cues = parse_srt("﻿1\r\n00:00:01,000 --> 00:00:03,000\r\n<i>Hello</i> {\\an8}there\r\n\r\n"
                     "2\n00:00:04.000 --> 00:00:08.000\n- How are you?\n- Fine, thanks.\n\n"
                     "3\n00:00:09,000 --> 00:00:10,500\n[Music]\n")
    assert [(c.start, c.end, c.text) for c in cues[:1]] == [(1.0, 3.0, "Hello there")]
    # Dialogue dashes are one line per speaker: split, proportional inside the cue.
    assert [c.text for c in cues[1:3]] == [">> How are you?", ">> Fine, thanks."]
    assert cues[1].start == 4.0 and cues[2].end == 8.0 and 4.0 < cues[1].end == cues[2].start < 8.0
    spoken = spoken_cues(cues)
    assert [c.text for c in spoken] == ["Hello there", "How are you?", "Fine, thanks."]
    assert [c.speaker_change for c in spoken] == [False, True, True]
    with pytest.raises(SubtitleError):
        parse_srt("1\n00:00:05,000 --> 00:00:04,000\nBackwards\n")
    with pytest.raises(SubtitleError):
        parse_srt("not a subtitle file")


def test_closed_caption_speaker_markers_split_cues_and_are_never_spoken():
    cues = spoken_cues(parse_srt(
        "1\n00:00:00,000 --> 00:00:02,000\nThese are some of the\n\n"
        "2\n00:00:02,000 --> 00:00:04,000\nmost popular games. >> How?\n\n"
        "3\n00:00:04,000 --> 00:00:05,000\n>> Like this. >> [laughs]\n\n"
        "4\n00:00:05,000 --> 00:00:06,000\nreally.\n"))
    # ">> [laughs]" is someone else laughing: "really." is that person's words.
    assert [(c.text, c.speaker_change) for c in cues] == [
        ("These are some of the", False), ("most popular games.", False), ("How?", True),
        ("Like this.", True), ("really.", True)]
    # The split inside a cue is proportional to the text; the cue's end is exact.
    assert cues[1].start == 2.0 and 2.0 < cues[1].end == cues[2].start < 4.0 and cues[2].end == 4.0
    assert all(">>" not in c.text for c in cues)
    # An unmarked caption continues the previous speaker's turn.
    assert [[c.text for c in turn] for turn in speaker_diarization.speaker_turns(cues)] == [
        ["These are some of the", "most popular games."], ["How?"], ["Like this."], ["really."]]
    plain = [Cue(0, 1, "a"), Cue(1, 2, "b")]
    assert speaker_diarization.speaker_turns(plain) == [[plain[0]], [plain[1]]]


def test_rolling_auto_caption_windows_are_normalized():
    cues = parse_srt("1\n00:00:01,000 --> 00:00:04,000\nhello there\n\n"
                     "2\n00:00:02,000 --> 00:00:05,000\nhello there general kenobi\n")
    assert [(c.start, c.end, c.text) for c in spoken_cues(cues)] == [
        (1.0, 2.0, "hello there"), (2.0, 5.0, "general kenobi")]


def matrix(groups, size, same=0.7, other=0.1):
    values = array.array("f", [1.0 if i == j else same if groups[i] == groups[j] else other
                               for i in range(size) for j in range(size)])
    return Similarity(values, size)


def test_clustering_cuts_by_threshold_or_known_count():
    groups = [0, 0, 1, 2, 1, 0, 2, 2]
    similarity = matrix(groups, len(groups))
    assert cluster(list(range(8)), similarity, 0.3) == [[0, 1, 5], [2, 4], [3, 6, 7]]
    assert len(cluster(list(range(8)), similarity, 0.3, speakers=2)) == 2
    assert cluster(list(range(8)), similarity, 0.95) == [[i] for i in range(8)]
    assert cluster([], similarity, 0.3) == []


def test_transcript_ratio_and_mora_estimates():
    assert transcript_ratio("Hello, I'm Sian.", "hello im sian") == 1.0
    assert transcript_ratio("Hello, I'm Sian.", "Hello, I'm Sean.") > 0.75
    assert transcript_ratio("お会いできて光栄です。", "おはよう、いってこーです。") < 0.6
    assert mora_count("きょう") == 2 and mora_count("今日") == 3.6
    # Long text keeps Irodori's measured pace; punctuation adds pauses.
    assert natural_seconds("はい、そうです。", 6.4) == pytest.approx(6 / 6.4 + 0.15)


@pytest.mark.parametrize("expected,heard", [
    ("だから、どこからでもアマゾンを使える。", "だからどこからでもAmazonを使える"),
    ("だから、どこからでもAmazonを使える。", "だからどこからでもアマゾンを使える"),
    ("Ａｍａｚｏｎを使える。", "ｱﾏｿﾞﾝを使える。"),
])
def test_latin_and_kana_spellings_align_only_on_exact_readings(expected, heard):
    comparison = align_latin_spellings(expected, heard)
    assert transcript_ratio(*comparison) == 1.0
    assert transcript_coverage(*comparison) == 1.0


def test_spelling_alignment_preserves_wrong_words_numbers_and_missing_phrases():
    for expected, heard in [
        ("アマゾンを使える。", "Googleを使える。"),
        ("ですから、会社を作ろうとしているなら、stripe.com/atlasを見てください。",
         "ですから、会社を作ろうとしているなら、spardy.comはあってですよ。見てください。"),
    ]:
        comparison = align_latin_spellings(expected, heard)
        assert transcript_ratio(*comparison) < 0.75
    comparison = align_latin_spellings("Amazonで1000円。", "アマゾンで9000円。")
    assert "1000" in comparison[0] and "9000" in comparison[1]
    assert transcript_ratio(*comparison) < 1.0
    expected = "Amazonを使える。価格を必ず確認してください。"
    comparison = align_latin_spellings(expected, "アマゾンを使える。")
    assert transcript_coverage(*comparison) < 0.5


@pytest.mark.parametrize("expected,heard", [
    ("アマゾン・ドット・コムをクリックする体験とは関係ない。", "amazon.comをクリックする体験とは関係ない"),
    ("はい、私は Software を修正する必要があると感じている。", "はい、私はソフトウェアを修正する必要があると感じている"),
    ("ナヴィエ・ストークスで、1回1000万ドルくらいかな。", "Naviya Stokesで一回1000万ドルくらいかな"),
    ("OpenAIの資本配分も、 basically 計算資源の配分だ。", "オープンAIの資本配分もベーシカリー計算資源の配分だ"),
])
def test_whisper_spellings_of_names_and_english_words_pass(expected, heard):
    assert speech_verified(*compare(expected, heard))


@pytest.mark.parametrize("expected,heard", [
    # Another word, not another spelling.
    ("アマゾンを使える。", "Googleを使える。"),
    ("これは最初の非Guinnessだが、最初のIPAだ.", "これは最初の非ニューブスだが最初のIPAだ"),
    # A respelled name must not carry a wrong number through.
    ("つまり、 Navier-Stokes を1年で当時のコストの1/100に再計算できるだろうと思うよ。",
     "つまりナビアストックスを1年で当時のコストの100万円に再計算できるだろうと思うよ"),
])
def test_other_words_and_wrong_numbers_still_fail(expected, heard):
    assert not speech_verified(*compare(expected, heard))


@pytest.mark.parametrize("expected,heard", [
    ("次。じゃんけん、シュート。パンク。じゃんけん、シュート。", "次、ジャンケンシュート、パンク、ジャンケンシュート。"),
    ("モロイーさん、レジェンドだね。", "もろいさん、レジェンドだね。"),
    ("哨戒艇沈めたよ。", "紹介亭沈めたよ"),
    ("うわあ。", "うわぁ"),
])
def test_a_transcript_with_the_lines_exact_reading_passes(expected, heard):
    assert not speech_verified(transcript_ratio(expected, heard), 1.0)
    assert speech_verified(*compare(expected, heard))


@pytest.mark.parametrize("expected,heard", [
    ("ヘッズ。", "ヘッツ"),
    ("表がいいな。", "おもてがいいな"),
])
def test_a_different_reading_is_not_a_spelling(expected, heard):
    assert not speech_verified(*compare(expected, heard))


def test_readings_drop_digits_so_the_numbers_must_agree_both_ways():
    assert same_reading("3ターン。", "3ターン")
    assert not same_reading("3ターン。", "5ターン")
    assert not same_reading("ターン。", "3ターン")


def test_numbers_are_heard_in_kanji_and_digits_of_names_belong_to_the_word():
    assert numbers_heard("1000倍賢い。10,000回も。", "千倍賢い。1万回も。")
    assert numbers_heard("LPDDR6版は4.9GHzまで。", "LPDシック版は4.9ギガヘルツまで")
    assert not numbers_heard("1,000,000倍の賢いもの", "1000倍の賢い者")


async def test_diarization_labels_from_the_voice_alone(tmp_path):
    analysis = tmp_path / "a.wav"
    with wave.open(str(analysis), "wb") as wav:
        wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        wav.writeframes(b"\0\0" * 16000 * 20)
    script = [*SCRIPT, ("C", 17.0, 19.0, "And I am the third guest.")]
    cues = [Cue(a, b, text) for _, a, b, text in script] + [Cue(19.2, 19.5, "Yes.")]
    lines = await SpeakerDiarizer(FakeEmbeddings(script)).run(cues, AnalysisAudio(analysis), tmp_path)
    assert [line.speaker for line in lines[:7]] == ["S1", "S2", "S1", "S2", "S2", "S1", "S3"]
    assert {line.source for line in lines[:7]} == {"acoustic"}
    # Too short to embed: no speaker; the dub uses the line's own voice (FileDubber._voices).
    assert (lines[7].speaker, lines[7].source) == (None, "unassigned")



async def test_overlapped_speech_keeps_its_speaker_but_is_marked_uncertain(tmp_path):
    analysis = tmp_path / "a.wav"
    with wave.open(str(analysis), "wb") as wav:
        wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        wav.writeframes(b"\0\0" * 16000 * 18)

    class Overlap(FakeEmbeddings):
        async def activity(self, audio, work):
            activity = await super().activity(audio, work)
            for frame in range(600, 700):  # B speaks over the start of A's "Tell us about your new project."
                activity.values[frame * activity.speakers + 1] = 0.7
            return activity

    lines = await SpeakerDiarizer(Overlap()).run([Cue(a, b, text) for _, a, b, text in SCRIPT],
                                                 AnalysisAudio(analysis), tmp_path)
    assert [line.speaker for line in lines] == ["S1", "S2", "S1", "S2", "S2", "S1"]
    assert lines[2].source == "acoustic_uncertain"
    assert {line.source for line in lines if line.index != 2} == {"acoustic"}


async def test_a_requested_speaker_count_cuts_the_voice_clusters_without_the_activity_model(tmp_path):
    analysis = tmp_path / "a.wav"
    with wave.open(str(analysis), "wb") as wav:
        wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        wav.writeframes(b"\0\0" * 16000 * 18)

    class NoActivity(FakeEmbeddings):
        async def activity(self, audio, work):
            raise AssertionError("an explicit speaker count must not run the activity model")

    lines = await SpeakerDiarizer(NoActivity(), speakers=1).run([Cue(a, b, text) for _, a, b, text in SCRIPT],
                                                                AnalysisAudio(analysis), tmp_path)
    assert {line.speaker for line in lines} == {"S1"}


def test_speaker_activity_sums_and_peaks_over_an_interval():
    values = array.array("f", [0.9, 0.0] * 50 + [0.2, 0.6] * 50)  # 1 s: speaker 0, then speaker 1
    activity = SpeakerActivity(values, 2)
    (total0, peak0), (total1, peak1) = activity.over(0.4, 0.8)
    assert peak0 == pytest.approx(0.9) and peak1 == pytest.approx(0.6)
    assert total0 == pytest.approx(0.9 * 10 + 0.2 * 30) and total1 == pytest.approx(0.6 * 30)

def write_voice(path, spans, seconds):
    """16 kHz speech stand-in: a 200 Hz tone over ``spans``, silence elsewhere."""
    samples = array.array("h", bytes(16000 * seconds * 2))
    for a, b in spans:
        for i in range(round(a * 16000), round(b * 16000)):
            samples[i] = round(8000 * math.sin(2 * math.pi * 200 * i / 16000))
    with wave.open(str(path), "wb") as wav:
        wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        wav.writeframes(samples.tobytes())


def test_voiced_spans_drop_lead_in_trailing_silence_and_long_pauses(tmp_path):
    analysis = tmp_path / "a.wav"
    write_voice(analysis, [(1.0, 2.0), (2.2, 3.0), (4.0, 5.0), (5.5, 5.52)], 7)
    spans = AnalysisAudio(analysis).voiced(0.5, 6.5)
    # A 0.2 s gap stays inside a span; a 1 s pause splits it; a 20 ms click is dropped.
    assert [(round(a, 2), round(b, 2)) for a, b in spans] == [(0.95, 3.05), (3.95, 5.05)]
    assert AnalysisAudio(analysis).voiced(5.8, 6.5) == []


async def test_reference_selection_rejects_overlap_noise_and_wrong_words(tmp_path, exact_reference_check):
    analysis = tmp_path / "a.wav"
    write_voice(analysis, [(0, 18)], 18)
    script = [*SCRIPT[:5], ("A", 13.5, 16.6, "That sounds like hard work.")]  # Overlaps line 4.

    class WrongWords(FakeTranscriber):
        calls = 0

        async def transcribe(self, wav, expected, *, language):
            self.calls += 1
            return "something else entirely" if self.calls == 1 else expected

    cues = [Cue(a, b, text) for _, a, b, text in script]
    diarizer = SpeakerDiarizer(FakeEmbeddings(script), transcriber=WrongWords())
    lines = await diarizer.run(cues, AnalysisAudio(analysis), tmp_path)
    references = await diarizer.references(lines, AnalysisAudio(analysis))
    reasons = {item["line"]: item["reason"] for ref in references.values() for item in ref["rejected"]}
    assert reasons[5] == "overlaps_other_speaker"
    assert "transcript_mismatch" in reasons.values()
    accepted = {item["line"] for ref in references.values() for item in ref["clips"]}
    assert accepted and not accepted & set(reasons)
    # Touching neighbours are cut 80ms inside, never across the other voice.
    clip = next(c for ref in references.values() for c in ref["clips"] if c["line"] == 1)
    assert clip["start"] == pytest.approx(3.2) and clip["end"] == pytest.approx(5.6)


@pytest.mark.parametrize("speakers", [None, 1])
async def test_halves_are_embedded_only_for_possible_reference_candidates(tmp_path, speakers):
    from transiroyt.caption_annotations import EMOJI

    analysis = tmp_path / "a.wav"
    with wave.open(str(analysis), "wb") as wav:
        wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        wav.writeframes(b"\0\0" * 16000 * 27)
    script = [("A", 0.5, 3.5, "Welcome to the show, everyone."),
              ("B", 3.7, 19.7, "A turn longer than any reference candidate."),
              ("A", 20.0, 23.0, f"That is funny {EMOJI['laughter']}"),
              ("B", 23.2, 26.2, "We built a small boat out of wood.")]
    embeddings = FakeEmbeddings(script)
    diarizer = SpeakerDiarizer(embeddings, speakers=speakers)
    await diarizer.run([Cue(a, b, text) for _, a, b, text in script], AnalysisAudio(analysis), tmp_path)
    halves = {} if speakers == 1 else {0: (4, 5), 3: (6, 7)}
    assert diarizer.halves == halves
    assert len(embeddings.calls[0]) == len(script) + 2 * len(halves)


def test_segments_are_sentences_of_one_turn_and_never_cross_a_speaker_change():
    lines = [SpeakerLine(0, 0.5, 2, "Hi.", "S1"), SpeakerLine(1, 2.2, 4, "How are you?", "S1"),
             SpeakerLine(2, 4.1, 5, "Fine.", "S2"), SpeakerLine(3, 7, 8, "Good.", "S2")]
    # Short neighbours of one speaker join; the other speaker is never joined.
    assert [(speaker, cue.start, cue.end, cue.text) for speaker, cue in speaker_segments(lines)] == [
        ("S1", 0.5, 4, "Hi. How are you?"), ("S2", 4.1, 5, "Fine."), ("S2", 7, 8, "Good.")]
    # A long turn keeps one anchor per sentence, from its captions.
    turn = SpeakerLine(0, 0, 9, "x", "S1", parts=[(0, 3, "We built a boat out"), (3, 5, "of wood."),
                                                    (5, 9, "It took three long years.")])
    assert [(cue.start, cue.end, cue.text) for _, cue in speaker_segments([turn])] == [
        (0, 5, "We built a boat out of wood."), (5, 9, "It took three long years.")]
    # Sentences of one speaker join until the run reaches MIN_DUB_SECONDS: a
    # Japanese line long enough for the voice to read clearly (26+ characters
    # passed 93 of 95 takes against 32 of 81 for 16-25) needs about 4 s of window.
    lines = [SpeakerLine(0, 0, 3, "We built a boat out of wood.", "S1"),
             SpeakerLine(1, 3.1, 6, "It took three long years.", "S1"),
             SpeakerLine(2, 6.1, 11.5, "Then we sailed it across the bay.", "S1")]
    assert [(cue.start, cue.end, cue.text) for _, cue in speaker_segments(lines)] == [
        (0, 6, "We built a boat out of wood. It took three long years."),
        (6.1, 11.5, "Then we sailed it across the bay.")]


# A 57-minute dub accepted this take (whole-line ratio 0.837); heard as garbled by ear.
GARBLED_LINE = ("ここでTLBヒットが起きたとすると、まず組番号で第358組を見つけて、そのアドレスタグを4ウェイと一つずつ比較する。"
                "BJウェイのタグが同じなら、その物理ページ番号を取り出して、そのままのページ内オフセットをくっつければ物理アドレスになる。")
GARBLED_HEARD = ("ここでTLBヒットが起きたとするとまず組番号で第358組を見つけてそのアドレスタグを4ウェイと一つずつ比較する"
                 "BJウェイのアドレスタグだらそのぶつけページ番号を取り出してそのまま上手に変形を押せとくっつければぶつけアドレスになる")


def test_transcript_coverage_finds_a_garbled_phrase_in_a_long_line():
    assert transcript_ratio(GARBLED_LINE, GARBLED_HEARD) > 0.75
    assert transcript_coverage(GARBLED_LINE, GARBLED_HEARD) < 0.5
    assert transcript_coverage(GARBLED_LINE, GARBLED_LINE) == 1.0
    # Whisper's own spelling of the same sounds is no error: readings are compared.
    assert transcript_coverage("今年はアンドロイド旗艦機で玄界O3の2026年モデルを試せた",
                               "今年はAndroid旗艦機で限界O3の二〇二六年モデルを試せた") >= 0.5
    assert transcript_coverage("短い。", "") == 1.0


async def test_take_with_a_garbled_phrase_is_regenerated_despite_a_high_line_ratio(tmp_path):
    video, srt = tmp_path / "in.mp4", tmp_path / "in.srt"
    write_video(video)
    write_srt(srt)
    tts = FakeTTS()

    class Transcriber(FakeTranscriber):
        async def transcribe(self, wav, expected, *, language="ja"):
            return GARBLED_HEARD if len(tts.calls) == 1 else expected

    dubber = FileDubber(Settings(data=tmp_path / "data"), FileDubbingOptions(video, srt, tmp_path / "out.mp4"),
                        tts=tts, transcriber=Transcriber(tts))
    segment = Segment(0, 0.0, 12.0, "source", text=GARBLED_LINE)
    result = await dubber._generate_verified(tts, segment, tmp_path / "reference.wav", 10, tmp_path / "clip")
    first, second = result["attempts"]
    assert first["ratio"] >= 0.75 and first["coverage"] < 0.5
    assert (result["ratio"], result["coverage"]) == (1.0, 1.0) and len(tts.calls) == 2


async def test_latin_spelling_does_not_trigger_repeated_synthesis(tmp_path):
    (tmp_path / "v.mp4").touch()
    (tmp_path / "v.srt").touch()
    tts = FakeTTS()

    class Transcriber:
        async def transcribe(self, wav, expected):
            return "だからどこからでもAmazonを使える"

    dubber = FileDubber(Settings(data=tmp_path),
                       FileDubbingOptions(tmp_path / "v.mp4", tmp_path / "v.srt", tmp_path / "out.mp4"),
                       tts=tts, transcriber=Transcriber())
    segment = Segment(0, 0.0, 3.0, "source", text="だから、どこからでもアマゾンを使える。")
    result = await dubber._generate_verified(tts, segment, tmp_path / "ref.wav", 10, tmp_path / "clip")
    assert result["ratio"] == 1.0 and result["coverage"] == 1.0
    assert len(tts.calls) == 1


async def test_cached_latin_spelling_is_rechecked_without_regenerating_audio(tmp_path):
    (tmp_path / "v.mp4").touch()
    (tmp_path / "v.srt").touch()
    tts = FakeTTS()
    dubber = FileDubber(Settings(data=tmp_path),
                       FileDubbingOptions(tmp_path / "v.mp4", tmp_path / "v.srt", tmp_path / "out.mp4"),
                       tts=tts)
    segment = Segment(0, 0.0, 3.0, "source", text="だから、どこからでもアマゾンを使える。")
    reference = tmp_path / "ref.wav"
    clips = tmp_path / "clips"
    clips.mkdir()
    stem = clips / dubber._synthesis_key(segment, reference.parent.name, 10)
    natural = stem.with_suffix(".natural.wav")
    await tts.synthesize(segment.text, 3, natural, seconds=3)
    audio_hash = hashlib.sha256(natural.read_bytes()).hexdigest()
    stem.with_suffix(".json").write_text(json.dumps({
        "ratio": 0.722, "coverage": 1.0, "attempts": [], "seconds": 3,
        "transcript": "だからどこからでもAmazonを使える", "tail": {},
    }))
    await dubber._synthesize([segment], ["S1"], {"S1": reference}, 10, tmp_path)
    assert segment.state == "ready" and segment.audio["transcript_ratio"] == 1.0
    assert len(tts.calls) == 1
    assert hashlib.sha256(natural.read_bytes()).hexdigest() == audio_hash

    verification = json.loads(stem.with_suffix(".json").read_text())
    verification["transcript"] = "Amazon" + "あ" * 294
    stem.with_suffix(".json").write_text(json.dumps(verification))
    assert dubber._cached_verification(stem, dubber._voice(segment), segment.text) is None


async def test_three_locally_garbled_takes_are_never_placed_or_reused_as_ready(tmp_path):
    (tmp_path / "v.mp4").touch()
    (tmp_path / "v.srt").touch()
    tts = FakeTTS()

    class Transcriber:
        async def transcribe(self, wav, expected):
            return GARBLED_HEARD

    dubber = FileDubber(Settings(data=tmp_path),
                       FileDubbingOptions(tmp_path / "v.mp4", tmp_path / "v.srt", tmp_path / "out.mp4"),
                       tts=tts, transcriber=Transcriber())
    for _ in range(2):
        segment = Segment(0, 0.0, 12.0, "source", text=GARBLED_LINE)
        await dubber._synthesize([segment], ["S1"], {"S1": tmp_path / "ref.wav"}, 10, tmp_path)
        assert segment.state == "skipped"
        assert segment.skip_reason == "speech_verification_failed"
        assert len(segment.audio["verification"]) == 3
        assert all(a["ratio"] > 0.75 and a["coverage"] < 0.5 for a in segment.audio["verification"])
        assert not (tmp_path / "clips/segment-0.wav").exists()
    assert len(tts.calls) == 3


@pytest.mark.parametrize("ratio,coverage,ready", [(0.7, 1.0, False), (0.7499, 1.0, False),
                                                  (1.0, 0.4999, False), (1.0, 0.49, False),
                                                  (0.75, 0.5, True)])
async def test_final_speech_gate_uses_both_generation_thresholds(tmp_path, ratio, coverage, ready):
    (tmp_path / "v.mp4").touch()
    (tmp_path / "v.srt").touch()
    tts = FakeTTS()
    dubber = FileDubber(Settings(data=tmp_path),
                       FileDubbingOptions(tmp_path / "v.mp4", tmp_path / "v.srt", tmp_path / "out.mp4"),
                       tts=tts)
    segment = Segment(0, 0.0, 2.0, "source", text="こんにちは。")
    reference = tmp_path / "ref.wav"
    clips = tmp_path / "clips"
    clips.mkdir()
    stem = clips / dubber._synthesis_key(segment, reference.parent.name, 10)
    await tts.synthesize(segment.text, 1, stem.with_suffix(".natural.wav"), seconds=1)
    stem.with_suffix(".json").write_text(json.dumps({
        "ratio": ratio, "coverage": coverage, "attempts": [{}] * len(file_dubbing.SHORT_LINE_SEEDS), "seconds": 1,
        "transcript": "こんばんは。", "tail": {},
    }))
    await dubber._synthesize([segment], ["S1"], {"S1": reference}, 10, tmp_path)
    assert (segment.state == "ready") == ready
    assert (clips / "segment-0.wav").exists() == ready
    if ready:
        assert segment.audio["verified"]
    else:
        assert segment.skip_reason == "speech_verification_failed"


def test_speech_cache_is_separate_from_the_permissive_final_gate(tmp_path, monkeypatch):
    (tmp_path / "v.mp4").touch()
    (tmp_path / "v.srt").touch()
    dubber = FileDubber(Settings(data=tmp_path),
                       FileDubbingOptions(tmp_path / "v.mp4", tmp_path / "v.srt", tmp_path / "out.mp4"))
    segment = Segment(0, 0.0, 2.0, "source", text="こんにちは。")
    current = dubber._synthesis_key(segment, "reference-id", 10)
    monkeypatch.setattr(file_dubbing, "SYNTHESIS_VERSION", "coreml-mf-verified-v2")
    assert current != dubber._synthesis_key(segment, "reference-id", 10)


def test_unpunctuated_transcript_segments_stay_within_what_the_tts_can_speak():
    # A transcribed monologue without sentence ends: 40 clauses of 1 s each.
    lines = [SpeakerLine(i, i, i + 1, "这个是它的超大核", "S1") for i in range(40)]
    lengths = [cue.end - cue.start for _, cue in speaker_segments(lines)]
    assert sum(lengths) == 40 and max(lengths) <= 20


def test_dub_track_rejects_overlapping_clips(tmp_path):
    (tmp_path / "clips").mkdir()
    for index in (0, 1):
        with wave.open(str(tmp_path / "clips" / f"segment-{index}.wav"), "wb") as wav:
            wav.setparams((1, 2, 48000, 0, "NONE", "not compressed"))
            wav.writeframes(b"\x01\x00" * 48000)
    first, second = Segment(0, 0.0, 1.0, "a"), Segment(1, 0.5, 1.5, "b")
    with pytest.raises(FileDubbingError):
        place_segments([first, second], tmp_path, tmp_path / "track.wav", 3)


@pytest.mark.parametrize("skipped", [False, True])
async def test_synthesis_places_clips_on_sample_clock_across_skipped_turns(tmp_path, skipped, monkeypatch):
    video, subtitles = tmp_path / "v.mp4", tmp_path / "v.srt"
    video.touch()
    subtitles.touch()
    tts = FakeTTS()
    dubber = FileDubber(Settings(data=tmp_path),
                       FileDubbingOptions(video, subtitles, tmp_path / "out.mp4"),
                       tts=tts, gemma=FakeGemma(tts))
    # Anchors and natural length from the failing export (with whole seconds removed).
    if skipped:
        first = Segment(0, 0.6531327160494, 1.2, "a", text="こんにちは。")
        middle = Segment(1, 1.2, 2.72, "b", text="はい。")
        last = Segment(2, 2.72, 4.0, "c", text="こんばんは。")
    else:
        first = Segment(0, 0.0562059711197, 1.6302800451938, "a", text="こんにちは。")
        middle = Segment(1, 1.6302800451938, 2.0, "b", text="はい。")
        last = Segment(2, 2.0, 4.0, "c", text="こんばんは。")

    async def verified(tts, segment, reference, steps, stem):
        natural = stem.with_name(f"{stem.name}.natural.wav")
        await tts.synthesize(segment.text, 2.8, natural, seconds=2.8, seed=42)
        return {"ratio": 1.0, "coverage": 1.0, "attempts": [], "seconds": 2.8, "transcript": segment.text, "tail": {}}

    monkeypatch.setattr(dubber, "_generate_verified", verified)
    segments = [first, middle, last]
    if skipped:
        middle.state = "skipped"
    file_dubbing.prepare_speech_limits(segments, 6)
    await dubber._synthesize(segments, ["S1"] * 3,
                             {"S1": tmp_path / "reference.wav"}, 10, tmp_path)
    ready = [s for s in segments if s.state == "ready"]
    cursor = 0
    for segment in ready:
        start = round(segment.speech_start * 48000)
        assert start >= cursor
        with wave.open(str(tmp_path / "clips" / f"segment-{segment.index}.wav")) as wav:
            cursor = start + wav.getnframes()
        assert cursor == round(segment.speech_end * 48000)
    place_segments(ready, tmp_path, tmp_path / "track.wav", 6)


def test_options_validation(tmp_path):
    video, srt = tmp_path / "v.mp4", tmp_path / "v.srt"
    video.write_bytes(b"x")
    srt.write_text("")
    for kwargs in ({"original_volume": -0.1}, {"dub_volume": 0}, {"dub_volume": 2.5},
                   {"output": tmp_path / "out.mov"}, {"output": video}, {"speakers": 0},
                   {"source_language": "english!"}, {"output": tmp_path / "missing" / "o.mp4"}):
        options = FileDubbingOptions(video, srt, tmp_path / "out.mp4")
        for key, value in kwargs.items():
            setattr(options, key, value)
        with pytest.raises(FileDubbingError):
            options.validate()


def rms(samples):
    return math.sqrt(sum(x * x for x in samples) / len(samples)) / 32768


def decode(path, start, end):
    raw = subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-ss", str(start), "-to", str(end),
                          "-i", str(path), "-ac", "1", "-ar", "48000", "-f", "s16le", "-"],
                         check=True, capture_output=True).stdout
    return array.array("h", raw)


async def test_file_dubbing_end_to_end_with_volumes_and_verified_speech(tmp_path, exact_reference_check):
    video, srt, output = tmp_path / "in.mp4", tmp_path / "in.srt", tmp_path / "out.mp4"
    write_video(video)
    write_srt(srt)
    tts = FakeTTS()
    gemma = FakeGemma(tts=tts)
    progress = []
    options = FileDubbingOptions(video, srt, output, original_volume=0.5, dub_volume=1.0,
                                 uploader="Test Channel", thumbnail_url="https://i.ytimg.com/vi/abcdefghijk/hqdefault.jpg")
    report = await FileDubber(Settings(data=tmp_path / "data"), options,
                              lambda *event: progress.append(event), gemma=gemma,
                              embeddings=FakeEmbeddings(), translator=FakeTranslator(), tts=tts,
                              transcriber=FakeTranscriber(tts, garbled_seeds={42}, looping_seeds={7})).run()

    assert report["uploader"] == "Test Channel"
    assert report["thumbnail_url"] == options.thumbnail_url
    saved = json.loads(output.with_suffix(".dubbing.json").read_text())
    assert saved["uploader"] == report["uploader"] and saved["thumbnail_url"] == options.thumbnail_url
    assert report["source_language"] == "en"
    assert report["summary"] == {**report["summary"], "segments": 6, "ready": 6, "skipped": 0}
    assert [s["speaker"] for s in report["segments"]] == ["S1", "S2", "S1", "S2", "S2", "S1"]
    # Two full sentences of B keep their own anchors on the source clock.
    assert [report["segments"][i]["source"] for i in (3, 4)] == [
        "We built a small boat out of wood.", "It took three years to finish."]
    references = {name: item["reference"] for name, item in report["speakers"].items()}
    assert len(set(references.values())) == 2
    assert {str(call["reference"]) for call in tts.calls} == set(references.values())
    # Explicit length from the text (at most 1.25x faster when the window is
    # short); the garbled first take is regenerated.
    for call in tts.calls:
        natural = max(0.6, natural_seconds(call["text"], 6.4))
        assert natural / 1.25 - 1e-3 <= call["seconds"] <= natural + 1e-3
    # A take whose transcript never finishes counts as failed; the next seed is tried.
    first_call = tts.calls[0]
    assert [call["seed"] for call in tts.calls
            if (call["text"], call["reference"]) == (first_call["text"], first_call["reference"])][:3] == [42, 7, 1234]
    attempts = report["segments"][0]["verification"]
    assert [a["ratio"] for a in attempts] == [0, 0, 1.0]
    assert attempts[1]["transcript_error"] == "no_complete_transcript"
    assert report["segments"][0]["transcript_ratio"] == 1.0 and report["segments"][0]["verified"]
    assert progress[-1][0] == "done"
    assert json.loads(output.with_suffix(".dubbing.json").read_text())["output"] == str(output)

    streams = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,codec_name",
                              "-of", "csv=p=0", str(output)], check=True, capture_output=True, text=True).stdout
    assert streams.split() == ["h264,video", "aac,audio"]
    # Outside dubbed lines the original soundtrack plays unchanged; under a
    # dubbed line the (here: all-speech) original is lowered to the requested level.
    source_level = rms(decode(video, 16.9, 17.6))
    assert rms(decode(output, 16.9, 17.6)) == pytest.approx(source_level, rel=0.08)
    assert rms(decode(output, 1.0, 2.0)) > source_level * 0.5 * 1.3

    # A rerun reuses speaker analysis and verified speech instead of recomputing.
    before = len(tts.calls)
    embeddings = FakeEmbeddings()
    await FileDubber(Settings(data=tmp_path / "data"), FileDubbingOptions(video, srt, output),
                     gemma=gemma, embeddings=embeddings, translator=FakeTranslator(), tts=tts).run()
    assert len(tts.calls) == before and embeddings.calls == []


async def test_unusable_speech_keeps_the_original_audio(tmp_path, exact_reference_check):
    video, srt, output = tmp_path / "in.mp4", tmp_path / "in.srt", tmp_path / "out.mp4"
    write_video(video)
    write_srt(srt)
    tts = FakeTTS()
    gemma = FakeGemma(tts=tts)
    with pytest.raises(FileDubbingError, match="1つも"):
        await FileDubber(Settings(data=tmp_path / "data"), FileDubbingOptions(video, srt, output),
                         gemma=gemma, embeddings=FakeEmbeddings(), translator=FakeTranslator(), tts=tts,
                         transcriber=FakeTranscriber(tts, garbled_seeds=set(file_dubbing.SHORT_LINE_SEEDS))).run()
    # Every line is short, so each gets the six takes of a short line.
    assert len(tts.calls) == 6 * len(file_dubbing.SHORT_LINE_SEEDS)
    assert not output.exists()


async def test_reference_conversion_replaces_the_references_and_records_the_gates(
        tmp_path, exact_reference_check):
    video, srt = tmp_path / "in.mp4", tmp_path / "in.srt"
    write_video(video)
    write_srt(srt)
    settings = Settings(data=tmp_path / "data")
    converter, progress = FakeReferenceConverter(), []

    class ClosedFirstTTS(FakeTTS):
        async def synthesize(self, *args, **kwargs):
            # IndexTTS's memory is released before Irodori generates.
            assert converter.closed == 1
            return await super().synthesize(*args, **kwargs)

    tts = ClosedFirstTTS()
    report = await FileDubber(
        settings, FileDubbingOptions(video, srt, tmp_path / "out.mp4"),
        lambda *event: progress.append(event), tts=tts, gemma=FakeGemma(tts),
        embeddings=ConversionEmbeddings(), translator=FakeTranslator(),
        transcriber=ConversionTranscriber(tts), reference_converter=converter).run()

    assert report["summary"]["ready"] == 6
    cache = settings.data / file_dubbing.CONVERSION_CACHE_DIRECTORY
    converted = {}
    for name, item in report["speakers"].items():
        conversion = item["conversion"]
        assert conversion["state"] == "converted"
        assert Path(conversion["path"]) == cache / f"{REFERENCE_CONVERSION_VERSION}-{item['digest']}.wav"
        assert Path(conversion["path"]).is_file()
        assert json.loads(Path(conversion["path"]).with_suffix(".json").read_text())["text"] == CONVERSION_TEXT
        assert conversion["seconds"] == CONVERSION_SECONDS
        assert conversion["similarity"] == pytest.approx(0.9)
        assert conversion["ratio"] == pytest.approx(1.0)
        # The report keeps the programme's own reference; only synthesis uses the converted take.
        assert Path(item["reference"]).parent.name != file_dubbing.CONVERSION_CACHE_DIRECTORY
        converted[name] = Path(conversion["path"])
    assert len(converted) == 2 and len(converter.converted) == 2
    assert converter.closed == 1
    assert {call["reference"] for call in tts.calls} == set(converted.values())
    stages = [event[0] for event in progress]
    assert "reference_conversion" in stages and progress[-1][0] == "done"
    fractions = [event[1] for event in progress if event[0] == "reference_conversion"]
    assert fractions[0] == 0.0 and fractions[-1] == 1.0


async def test_failed_reference_conversion_keeps_the_original_and_warns(tmp_path, caplog):
    video, srt = tmp_path / "in.mp4", tmp_path / "in.srt"
    write_video(video)
    write_srt(srt)
    settings, tts = Settings(data=tmp_path / "data"), FakeTTS()
    converter = FakeReferenceConverter(failure="worker exited")
    with caplog.at_level(logging.WARNING, logger="transiroyt.file_dubbing"):
        report = await FileDubber(
            settings, FileDubbingOptions(video, srt, tmp_path / "out.mp4"),
            tts=tts, gemma=FakeGemma(tts), embeddings=FakeEmbeddings(), translator=FakeTranslator(),
            transcriber=FakeTranscriber(tts), reference_converter=converter).run()

    assert report["summary"]["ready"] == 6
    original = {Path(item["reference"]) for item in report["speakers"].values()}
    for item in report["speakers"].values():
        assert item["conversion"] == {"state": "fallback", "reason": "worker exited"}
    assert {call["reference"] for call in tts.calls} == original
    assert converter.converted == [] and converter.closed == 1
    assert "file dubbing reference conversion fallback speaker=S1 reason=worker exited" in caplog.text
    assert not list((settings.data / file_dubbing.CONVERSION_CACHE_DIRECTORY).glob("*.wav"))


async def test_cached_reference_conversion_skips_the_gates_and_the_models(tmp_path):
    video, srt = tmp_path / "in.mp4", tmp_path / "in.srt"
    write_video(video)
    write_srt(srt)
    settings = Settings(data=tmp_path / "data")
    tts = FakeTTS()
    first = await FileDubber(
        settings, FileDubbingOptions(video, srt, tmp_path / "out.mp4"),
        tts=tts, gemma=FakeGemma(tts), embeddings=ConversionEmbeddings(),
        translator=FakeTranslator(), transcriber=ConversionTranscriber(tts),
        reference_converter=FakeReferenceConverter()).run()

    class NoConversion(FakeReferenceConverter):
        async def convert(self, reference, output):
            raise AssertionError("a cached conversion must not run the worker")

    class NoModels:
        identity = FakeTranslator.identity

        async def forbidden(self, *args, **kwargs):
            raise AssertionError("a cached rerun must not call any model")

        translate_batch = synthesize = complete_json = similarity = transcribe = forbidden

        def start(self):
            raise AssertionError("a cached rerun must not start the transcript model")

    converter, no_models = NoConversion(), NoModels()
    second = await FileDubber(
        settings, FileDubbingOptions(video, srt, tmp_path / "out.mp4"),
        tts=no_models, gemma=no_models, embeddings=no_models, translator=no_models,
        transcriber=no_models, reference_converter=converter).run()
    assert second["segments"] == first["segments"]
    assert {item["conversion"]["state"] for item in second["speakers"].values()} == {"cached"}
    assert all({"seconds", "similarity", "ratio"} <= set(item["conversion"])
               for item in second["speakers"].values())
    assert converter.converted == [] and converter.closed == 1


async def test_successful_conversion_does_not_reuse_takes_made_from_the_original_reference(tmp_path):
    video, srt = tmp_path / "in.mp4", tmp_path / "in.srt"
    write_video(video)
    write_srt(srt)
    settings = Settings(data=tmp_path / "data")
    original_tts = FakeTTS()
    fallback = await FileDubber(
        settings, FileDubbingOptions(video, srt, tmp_path / "first.mp4"),
        tts=original_tts, gemma=FakeGemma(original_tts), embeddings=FakeEmbeddings(),
        translator=FakeTranslator(), transcriber=FakeTranscriber(original_tts),
        reference_converter=FakeReferenceConverter(failure="worker crashed")).run()
    assert {item["conversion"]["state"] for item in fallback["speakers"].values()} == {"fallback"}
    assert len(original_tts.calls) == fallback["summary"]["ready"]

    converted_tts = FakeTTS()
    converted = await FileDubber(
        settings, FileDubbingOptions(video, srt, tmp_path / "second.mp4"),
        tts=converted_tts, gemma=FakeGemma(converted_tts), embeddings=ConversionEmbeddings(),
        translator=FakeTranslator(), transcriber=ConversionTranscriber(converted_tts),
        reference_converter=FakeReferenceConverter()).run()
    assert {item["conversion"]["state"] for item in converted["speakers"].values()} == {"converted"}
    # The converted reference is a different voice source: every take is generated again.
    assert len(converted_tts.calls) == converted["summary"]["ready"]
    original = {Path(item["reference"]) for item in converted["speakers"].values()}
    assert {call["reference"] for call in converted_tts.calls}.isdisjoint(original)


def test_reference_conversion_cache_requires_the_current_version_and_text(tmp_path):
    target = tmp_path / "converted.wav"
    write_conversion_tone(target)
    metadata = target.with_suffix(".json")
    metadata.write_text(json.dumps({"version": REFERENCE_CONVERSION_VERSION,
                                    "text": CONVERSION_TEXT, "seconds": 12.0,
                                    "similarity": 0.8, "ratio": 1.0}))
    assert FileDubber._cached_conversion(target)["similarity"] == 0.8
    metadata.write_text(json.dumps({"version": REFERENCE_CONVERSION_VERSION,
                                    "text": "別の固定文", "seconds": 12.0}))
    assert FileDubber._cached_conversion(target) is None
    metadata.write_text(json.dumps({"version": "indextts25-ja-87mora-v0",
                                    "text": CONVERSION_TEXT, "seconds": 12.0}))
    assert FileDubber._cached_conversion(target) is None
    metadata.write_text(json.dumps({"version": REFERENCE_CONVERSION_VERSION,
                                    "text": CONVERSION_TEXT, "seconds": 40.0}))
    assert FileDubber._cached_conversion(target) is None


async def test_reference_conversion_needs_the_model_or_permission(tmp_path, monkeypatch):
    video, subtitles = tmp_path / "v.mp4", tmp_path / "v.srt"
    video.touch()
    subtitles.touch()
    reference = tmp_path / "reference.wav"
    write_conversion_tone(reference, seconds=5)
    monkeypatch.setattr(file_dubbing, "model_ready", lambda settings: False)
    dubber = FileDubber(Settings(data=tmp_path / "data"),
                        FileDubbingOptions(video, subtitles, tmp_path / "out.mp4"),
                        reference_converter=FakeReferenceConverter())
    with pytest.raises(FileDubbingError, match="IndexTTS-2.5"):
        await dubber._convert_references(tmp_path, {"S1": reference}, {"S1": "digest"})
    assert not (tmp_path / "data" / file_dubbing.CONVERSION_CACHE_DIRECTORY).exists()


async def test_reference_conversion_downloads_the_model_when_allowed(tmp_path, monkeypatch):
    from transiroyt import model_downloads

    video, subtitles = tmp_path / "v.mp4", tmp_path / "v.srt"
    video.touch()
    subtitles.touch()
    reference = tmp_path / "reference.wav"
    write_conversion_tone(reference, seconds=5)
    ready, progress, downloads = [False], [], []

    monkeypatch.setattr(file_dubbing, "model_ready", lambda settings: ready[0])

    async def download(settings, resource, callback):
        downloads.append(resource)
        ready[0] = True

    monkeypatch.setattr(model_downloads, "download_model", download)
    dubber = FileDubber(
        Settings(data=tmp_path / "data"),
        FileDubbingOptions(video, subtitles, tmp_path / "out.mp4", allow_download=True),
        lambda *event: progress.append(event), reference_converter=FakeReferenceConverter(),
        transcriber=ConversionTranscriber(None), embeddings=ConversionEmbeddings())
    references, conversions = await dubber._convert_references(
        tmp_path, {"S1": reference}, {"S1": "digest"})
    assert downloads == ["indextts_model"]
    assert ("reference_conversion", 0.0, "参照音声モデル（IndexTTS-2.5）を取得しています") in progress
    assert conversions["S1"]["state"] == "converted"
    assert references["S1"] != reference and references["S1"].is_file()


@pytest.mark.parametrize("failure,reason", [
    ("converter", "worker crashed"),
    ("transcript", "固定文どおりに聞き取れません"),
    ("similarity", "元の声と似ていません"),
    ("length", "秒でした"),
    ("format", "48kHz"),
])
async def test_reference_conversion_gate_keeps_the_original_reference(tmp_path, failure, reason):
    video, subtitles = tmp_path / "v.mp4", tmp_path / "v.srt"
    video.touch()
    subtitles.touch()
    reference = tmp_path / "reference.wav"
    write_conversion_tone(reference, seconds=5)

    class WrongTranscriber:
        async def transcribe(self, wav, expected):
            return "まったく違う文章"

    class LowEmbeddings:
        async def similarity(self, audio, items, work):
            values = array.array("f", [1.0, 0.2, 0.2, 1.0])
            return Similarity(values, 2), [{} for _ in items]

    converter = FakeReferenceConverter(
        failure="worker crashed" if failure == "converter" else None,
        seconds=2 if failure == "length" else CONVERSION_SECONDS,
        rate=44100 if failure == "format" else 48000)
    dubber = FileDubber(
        Settings(data=tmp_path / "data"),
        FileDubbingOptions(video, subtitles, tmp_path / "out.mp4"),
        reference_converter=converter,
        transcriber=WrongTranscriber() if failure == "transcript" else ConversionTranscriber(None),
        embeddings=LowEmbeddings() if failure == "similarity" else ConversionEmbeddings())
    references, conversions = await dubber._convert_references(
        tmp_path, {"S1": reference}, {"S1": "digest"})
    assert references["S1"] == reference
    assert conversions["S1"]["state"] == "fallback" and reason in conversions["S1"]["reason"]
    cache = tmp_path / "data" / file_dubbing.CONVERSION_CACHE_DIRECTORY
    assert not list(cache.glob("*.wav")) and not list(cache.glob("*.json"))


@pytest.mark.parametrize("encoder,codec", [("libsvtav1", "av1"), ("libvpx-vp9", "vp9"), ("libx264", "h264")])
async def test_mux_writes_apple_playable_video(tmp_path, encoder, codec):
    video, output = tmp_path / "source.mp4", tmp_path / "dubbed.mp4"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "lavfi", "-i",
                    "testsrc2=s=160x90:r=10:d=1", "-c:v", encoder, "-pix_fmt", "yuv420p",
                    str(video)], check=True, capture_output=True)
    track = tmp_path / "track.wav"
    with wave.open(str(track), "wb") as wav:
        wav.setparams((1, 2, 48000, 0, "NONE", "not compressed"))
        wav.writeframes(array.array("h", [1000] * 48000).tobytes())
    await file_dubbing.mux(video, track, output, 1.0, vocals=track, background=track, speech_gain=track)
    info = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(output)]))
    picture, sound = info["streams"]
    assert (picture["codec_name"], picture["codec_tag_string"], picture["pix_fmt"]) == ("h264", "avc1", "yuv420p")
    assert (picture["width"], picture["height"], picture["nb_frames"]) == (160, 90, "10")
    assert sound["codec_name"] == "aac"
    subprocess.run(["ffmpeg", "-v", "error", "-xerror", "-i", str(output), "-f", "null", "-"], check=True, capture_output=True)
    if codec == "h264":
        def packets(path):
            return subprocess.check_output(["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v:0",
                                            "-c:v", "copy", "-f", "hash", "-hash", "sha256", "-"])
        assert packets(video) == packets(output)


async def test_video_without_audio_is_refused(tmp_path):
    video, srt = tmp_path / "in.mp4", tmp_path / "in.srt"
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                    "color=c=black:s=160x90:r=10:d=2", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(video)],
                   check=True)
    write_srt(srt, SCRIPT[:1])
    with pytest.raises(FileDubbingError, match="音声トラック"):
        await FileDubber(Settings(data=tmp_path / "data"), FileDubbingOptions(video, srt, tmp_path / "o.mp4"),
                         gemma=FakeGemma(), embeddings=FakeEmbeddings()).run()


async def test_file_dubbing_api_runs_one_job_and_cancels(tmp_path):
    import asyncio

    import httpx
    from transiroyt.app import create_app

    video, srt = tmp_path / "in.mp4", tmp_path / "in.srt"
    video.write_bytes(b"video")
    write_srt(srt)
    release = asyncio.Event()
    started = []

    class Dubber:
        def __init__(self, options, progress):
            options.validate()
            self.options, self.progress = options, progress

        async def run(self):
            started.append(self.options)
            self.progress("synthesis", 0.5, "生成中")
            await release.wait()
            return {"summary": {"segments": 2, "ready": 2, "skipped": 0},
                    "speakers": {"S1": {"duration": 9.5, "reference": "/r.wav", "error": None}}}

    app = create_app(Settings(data=tmp_path / "data", token="t"), file_dubber_factory=Dubber)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1") as client:
            body = {"video": str(video), "subtitles": str(srt), "output": str(tmp_path / "out.mp4"),
                    "original_volume": 0.3, "dub_volume": 1.2, "speakers": 2, "steps": 10}
            assert (await client.post("/api/file-dubbing", json=body)).status_code == 401
            client.headers["Authorization"] = "Bearer t"
            status = (await client.get("/api/file-dubbing")).json()
            assert status["job"] is None and set(status["models"]) == {"coreml", "coreml_tool", "llama_server", "gemma", "plamo", "speaker_model", "diarization_model", "separation_model", "speech_check_model", "indextts_model"}
            assert (await client.post("/api/file-dubbing", json={**body, "dub_volume": 0})).status_code == 422
            assert (await client.post("/api/file-dubbing", json={**body, "steps": 0})).status_code == 422
            assert (await client.post("/api/file-dubbing", json={**body, "output": str(video)})).status_code == 409
            job = (await client.post("/api/file-dubbing", json=body)).json()["job"]
            await asyncio.sleep(0)
            assert started[0].original_volume == 0.3 and started[0].dub_volume == 1.2 and started[0].speakers == 2
            assert started[0].steps == 10
            current = (await client.get("/api/file-dubbing")).json()["job"]
            assert (current["state"], current["stage"], current["message"]) == ("running", "synthesis", "生成中")
            assert (await client.post("/api/file-dubbing", json=body)).status_code == 409
            release.set()
            for _ in range(50):
                current = (await client.get("/api/file-dubbing")).json()["job"]
                if current["state"] != "running":
                    break
                await asyncio.sleep(0.01)
            assert current["state"] == "done" and current["summary"]["ready"] == 2
            assert current["speakers"]["S1"]["reference"] is True

            (tmp_path / "out.mp4").write_bytes(b"existing")
            assert (await client.post("/api/file-dubbing", json=body)).status_code == 409
            release.clear()
            job = (await client.post("/api/file-dubbing", json={**body, "overwrite": True})).json()["job"]
            cancelled = (await client.delete(f"/api/file-dubbing/{job['id']}")).json()["job"]
            assert cancelled["state"] == "cancelled"
            assert (await client.delete("/api/file-dubbing/unknown")).status_code == 404


async def test_file_preloads_once_and_cached_export_loads_nothing(
        tmp_path, exact_reference_check, monkeypatch):
    video, subtitles = tmp_path / "source.mp4", tmp_path / "source.srt"
    write_video(video)
    write_srt(subtitles)
    loaded, generated = asyncio.Event(), asyncio.Event()

    class PreloadingTTS(FakeTTS):
        preparations = 0

        def preload(self):
            self.preparations += 1

            async def prepare():
                loaded.set()
            return asyncio.create_task(prepare())

        async def synthesize(self, *args, **kwargs):
            assert loaded.is_set()
            result = await super().synthesize(*args, **kwargs)
            generated.set()
            return result

    class Translator(FakeTranslator):
        async def translate_batch(self, *args):
            # Translation runs with the reference checks, before any speech.
            assert not generated.is_set()
            return await super().translate_batch(*args)

    monkeypatch.setattr(file_dubbing, "BATCH_SIZE", 1)
    tts = PreloadingTTS()
    settings = Settings(data=tmp_path / "data")
    options = FileDubbingOptions(video, subtitles, tmp_path / "out.mp4")
    first = await FileDubber(settings, options, tts=tts, gemma=FakeGemma(tts),
                             embeddings=FakeEmbeddings(), translator=Translator(),
                             transcriber=FakeTranscriber(tts)).run()
    assert tts.preparations == 1
    assert first["summary"]["ready"] == 6

    class NoModels:
        identity = FakeTranslator.identity

        def preload(self):
            raise AssertionError("Cached export must not even preload TTS")

        def start(self):
            raise AssertionError("Cached export must not load the transcript model")

        async def forbidden(self, *args, **kwargs):
            raise AssertionError("Cached export must not call models")

        translate_batch = synthesize = complete_json = similarity = transcribe = forbidden

    model = NoModels()
    second = await FileDubber(settings, options, tts=model, gemma=model,
                              embeddings=model, translator=model, transcriber=model).run()
    assert first["segments"] == second["segments"]


async def test_cancel_translation_reclaims_file_tts_preparation(
        tmp_path, exact_reference_check, monkeypatch):
    video, subtitles = tmp_path / "source.mp4", tmp_path / "source.srt"
    write_video(video)
    write_srt(subtitles)
    started, cancelled = asyncio.Event(), asyncio.Event()

    class PreloadingTTS(FakeTTS):
        closed = False

        def preload(self):
            async def prepare():
                started.set()
                try:
                    await asyncio.Future()
                finally:
                    cancelled.set()
            return asyncio.create_task(prepare())

        async def close(self):
            self.closed = True

    class Translator(FakeTranslator):
        calls = 0

        async def translate_batch(self, *args):
            self.calls += 1
            if self.calls > 1:
                await asyncio.Future()
            return await super().translate_batch(*args)

    monkeypatch.setattr(file_dubbing, "BATCH_SIZE", 1)
    tts = PreloadingTTS()
    monkeypatch.setattr(file_dubbing, "CoreMLTTS", lambda _, **kwargs: tts)
    dubber = FileDubber(Settings(data=tmp_path / "data"),
                       FileDubbingOptions(video, subtitles, tmp_path / "out.mp4"),
                       gemma=FakeGemma(tts), embeddings=FakeEmbeddings(), translator=Translator(),
                       transcriber=FakeTranscriber(tts))
    task = asyncio.create_task(dubber.run())
    await asyncio.wait_for(started.wait(), 3)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cancelled.is_set() and tts.closed and dubber.tts is None


@pytest.mark.parametrize("failure", [False, True])
async def test_reference_batches_are_bounded_and_reclaimed_on_cancel_or_failure(failure):
    diarizer = SpeakerDiarizer(None)
    lines = [SpeakerLine(i, i * 3, i * 3 + 2, "hello", f"S{i}") for i in range(6)]
    active = set()
    started = asyncio.Event()
    fail = asyncio.Event()
    peak = 0

    async def select(speaker, *_):
        nonlocal peak
        active.add(speaker)
        peak = max(peak, len(active))
        if len(active) == 1:
            started.set()
        try:
            await fail.wait()
            if speaker == "S0":
                raise RuntimeError("reference check failed")
            await asyncio.Future()
        finally:
            active.remove(speaker)

    diarizer._reference_for = select
    task = asyncio.create_task(diarizer.references(lines, None))
    await asyncio.wait_for(started.wait(), 1)
    if failure:
        fail.set()
        with pytest.raises(RuntimeError, match="reference check failed"):
            await task
    else:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert peak == 1 and not active


async def test_reference_checks_share_one_whisper_worker_and_keep_candidate_order(monkeypatch):
    from types import SimpleNamespace

    # Two 6 s clips reach a 10 s target; the third clip of each speaker is never checked.
    monkeypatch.setattr("transiroyt.speaker_diarization.REFERENCE_TARGET_SECONDS", 10.0)
    monkeypatch.setattr("transiroyt.speaker_diarization.REFERENCE_MAX_SECONDS", 15.0)

    lines = [SpeakerLine(i, i * 7, i * 7 + 6, f"words {i}", f"S{i // 3}",
                         score=0.9 - (i % 3) * 0.01, source="acoustic") for i in range(9)]
    by_start = {line.start: line for line in lines}
    audio = SimpleNamespace(wav=lambda start, end: str(start).encode(), voiced=lambda start, end: [(start, end)])

    class Transcriber:
        def __init__(self):
            self.calls = []

        async def transcribe(self, wav, expected, *, language):
            assert language == "en"
            start = float(wav)
            self.calls.append(by_start[start].index)
            await asyncio.sleep(0)
            return by_start[start].text

    transcriber = Transcriber()
    diarizer = SpeakerDiarizer(None, transcriber=transcriber)
    diarizer.groups = {}
    diarizer.halves = {}
    diarizer.item_of = {line.index: line.index for line in lines}
    diarizer.stats = [{"rms": 0.1, "clipped": 0, "speech_db": -20, "floor_db": -50}] * len(lines)
    await diarizer.references(lines, audio)
    assert sorted(transcriber.calls) == [0, 1, 3, 4, 6, 7]
    assert diarizer.reference_transcriptions == 6
    for speaker in range(3):
        assert [i for i in transcriber.calls if i // 3 == speaker] == [speaker * 3, speaker * 3 + 1]


async def test_untranslatable_line_is_skipped_once_and_releases_synthesis_waiters(
        tmp_path, exact_reference_check, monkeypatch):
    video, subtitles = tmp_path / "source.mp4", tmp_path / "source.srt"
    write_video(video)
    write_srt(subtitles)
    checking, translating = asyncio.Event(), asyncio.Event()
    original = SpeakerDiarizer.references

    async def references(self, *args):
        checking.set()
        await asyncio.wait_for(translating.wait(), 1)
        return await original(self, *args)

    class Translator(FakeTranslator):
        async def translate_batch(self, language, items, before, after):
            # Translation runs while the reference candidates are being checked.
            await asyncio.wait_for(checking.wait(), 1)
            translating.set()
            result = await super().translate_batch(language, items, before, after)
            if 1 in result:
                self.checks = {1: {"error": "no Japanese answer"}}
            return {key: text for key, text in result.items() if key != 1}

    monkeypatch.setattr(SpeakerDiarizer, "references", references)
    monkeypatch.setattr(file_dubbing, "BATCH_SIZE", 1)
    translator, tts = Translator(), FakeTTS()
    report = await asyncio.wait_for(FileDubber(
        Settings(data=tmp_path / "data"), FileDubbingOptions(video, subtitles, tmp_path / "out.mp4"),
        tts=tts, gemma=FakeGemma(tts), embeddings=FakeEmbeddings(), translator=translator,
        transcriber=FakeTranscriber(tts)).run(), 6)
    assert report["summary"]["ready"] == 5 and len(tts.calls) == 5
    assert report["segments"][1]["skip_reason"] == "translation_failed"
    assert report["segments"][1]["translation_check"] == {"error": "no Japanese answer"}
    # Each line is asked once in the job, including the one that failed.
    assert len(translator.batches) == 6
    # Context is the neighbouring source lines with their speakers.
    items, before, after = translator.batches[2]
    assert [item.key for item in items] == [2]
    assert before == [("S1", SCRIPT[0][3]), ("S2", SCRIPT[1][3])]
    assert after[0] == ("S2", SCRIPT[3][3])


async def test_fresh_file_synthesizes_translated_lines_while_the_rest_translate(
        tmp_path, exact_reference_check, monkeypatch):
    video, subtitles = tmp_path / "source.mp4", tmp_path / "source.srt"
    write_video(video)
    write_srt(subtitles)
    synthesizing = asyncio.Event()

    class TTS(FakeTTS):
        async def synthesize(self, *args, **kwargs):
            synthesizing.set()
            return await super().synthesize(*args, **kwargs)

    class Translator(FakeTranslator):
        async def translate_batch(self, language, items, before, after):
            # The last line is translated only after synthesis has begun.
            if items[0].key == len(SCRIPT) - 1:
                await asyncio.wait_for(synthesizing.wait(), 3)
            return await super().translate_batch(language, items, before, after)

    monkeypatch.setattr(file_dubbing, "BATCH_SIZE", 1)
    translator, tts = Translator(), TTS()
    report = await asyncio.wait_for(FileDubber(
        Settings(data=tmp_path / "data"), FileDubbingOptions(video, subtitles, tmp_path / "out.mp4"),
        tts=tts, gemma=FakeGemma(tts), embeddings=FakeEmbeddings(), translator=translator,
        transcriber=FakeTranscriber(tts)).run(), 6)
    assert report["summary"]["ready"] == len(SCRIPT) == len(tts.calls)
    assert len(translator.batches) == len(SCRIPT)


async def test_translation_failure_during_reference_checks_stops_the_job(
        tmp_path, exact_reference_check, monkeypatch):
    video, subtitles = tmp_path / "source.mp4", tmp_path / "source.srt"
    write_video(video)
    write_srt(subtitles)
    cancelled = asyncio.Event()

    async def references(self, *args):
        try:
            await asyncio.Future()
        finally:
            cancelled.set()

    class Translator(FakeTranslator):
        async def translate_batch(self, *args):
            raise RuntimeError("engine unavailable")

    monkeypatch.setattr(SpeakerDiarizer, "references", references)
    tts = FakeTTS()
    with pytest.raises(RuntimeError, match="engine unavailable"):
        await asyncio.wait_for(FileDubber(
            Settings(data=tmp_path / "data"), FileDubbingOptions(video, subtitles, tmp_path / "out.mp4"),
            tts=tts, gemma=FakeGemma(tts), embeddings=FakeEmbeddings(), translator=Translator(),
            transcriber=FakeTranscriber(tts)).run(), 6)
    assert cancelled.is_set() and tts.calls == []


@pytest.mark.parametrize("language", ["en", "auto"])
async def test_fresh_file_prepares_models_during_audio_extraction_and_closes_once(
        tmp_path, exact_reference_check, monkeypatch, language):
    video, subtitles = tmp_path / "source.mp4", tmp_path / "source.srt"
    write_video(video)
    write_srt(subtitles)
    model_started = [asyncio.Event() for _ in range(2)]

    class TTS(FakeTTS):
        closed = 0

        def preload(self):
            async def prepare():
                model_started[0].set()
            return asyncio.create_task(prepare())

        async def close(self):
            self.closed += 1

    class Gemma(FakeGemma):
        started = 0

        async def start(self):
            self.started += 1
            raise AssertionError("Local LLM must remain lazy until text inference is needed")

    class Transcriber(FakeTranscriber):
        closed = 0

        def start(self):
            async def prepare():
                model_started[1].set()
            return asyncio.create_task(prepare())

        async def close(self):
            self.closed += 1

    class Embeddings(FakeEmbeddings):
        async def similarity(self, *args):
            await asyncio.wait_for(asyncio.gather(*(event.wait() for event in model_started)), 1)
            return await super().similarity(*args)

    original = file_dubbing.extract_audio

    async def extract(*args):
        if args[2] == file_dubbing.ANALYSIS_RATE:
            await asyncio.wait_for(asyncio.gather(*(event.wait() for event in model_started)), 1)
        await original(*args)

    tts = TTS()
    gemma, transcriber = Gemma(tts), Transcriber(tts)
    monkeypatch.setattr(file_dubbing, "CoreMLTTS", lambda _, **kwargs: tts)
    monkeypatch.setattr(file_dubbing, "WhisperTranscriber", lambda _, **kwargs: transcriber)
    monkeypatch.setattr(file_dubbing, "extract_audio", extract)
    dubber = FileDubber(
        Settings(data=tmp_path / "data"),
        FileDubbingOptions(video, subtitles, tmp_path / "out.mp4", source_language=language),
        gemma=gemma, embeddings=Embeddings())
    report = await dubber.run()
    # Without an injected translator the job's Gemma translates, in one batch.
    assert report["summary"]["ready"] == 6 and tts.closed == 1 and gemma.translations == 1
    assert transcriber.closed == 1 and gemma.started == 0
    assert [s["text"] for s in report["segments"]] == [f"訳T{i}。" for i in range(6)]
    assert dubber.tts is None and dubber.transcriber is None


def test_speech_gain_lowers_only_dubbed_lines_and_keeps_skipped_speech(tmp_path):
    from transiroyt.dubbing_media import GAIN_UNITY, write_speech_gain

    dubbed = Segment(0, 1.0, 2.0, "a", state="ready")
    dubbed.playback_start, dubbed.playback_end = 1.0, 2.5  # The dub runs past its line.
    later = Segment(2, 5.0, 6.0, "c", state="ready")
    target = tmp_path / "gain.wav"
    write_speech_gain([dubbed, later], target, 8.0, 0.25)
    with wave.open(str(target)) as wav:
        assert (wav.getframerate(), wav.getnframes()) == (48000, 8 * 48000)
        gain = array.array("h", wav.readframes(wav.getnframes()))

    def at(seconds):
        return gain[round(seconds * 48000)] / GAIN_UNITY

    assert at(0.5) == 1.0 and at(3.5) == 1.0 and at(7.5) == 1.0  # Skipped line 3-4 s, silence.
    assert at(1.5) == at(2.55) == at(5.5) == pytest.approx(0.25, abs=1e-4)
    # Short linear ramps at the edges, never a step.
    assert max(abs(gain[i + 1] - gain[i]) for i in range(len(gain) - 1)) < GAIN_UNITY / 100


async def test_speaker_heard_briefly_is_cloned_from_their_own_unverified_clips(tmp_path):
    from types import SimpleNamespace

    lines = [SpeakerLine(0, 0, 10, "host " * 10, "S1", score=0.9, source="acoustic"),
             SpeakerLine(1, 10.5, 11.3, "Jenga.", "S2", score=0.9, source="acoustic"),
             SpeakerLine(2, 12, 13.8, "Is that even legal?", "S2", score=0.8, source="acoustic"),
             SpeakerLine(3, 14, 14.3, "Oh.", "S3", score=0.9, source="acoustic")]
    by_start = {line.start: line for line in lines}

    class Transcriber:
        parallel = 1

        async def transcribe(self, wav, expected, *, language):
            start = float(wav)
            # The brief speaker's only long clip fails the transcript check.
            return by_start[start].text if start == 0 else "something else"

    diarizer = SpeakerDiarizer(None, transcriber=Transcriber())
    diarizer.groups = {"S1": [0], "S2": [1, 2], "S3": [3]}
    diarizer.halves = {}
    diarizer.item_of = {line.index: line.index for line in lines}
    diarizer.stats = [{"rms": 0.1, "clipped": 0, "speech_db": -20, "floor_db": -50}] * len(lines)
    audio = SimpleNamespace(wav=lambda start, end: str(start).encode(), voiced=lambda start, end: [(start, end)])
    references = await diarizer.references(lines, audio)
    assert references["S1"]["verified"] and references["S1"]["usable"]
    brief = references["S2"]
    assert not brief["verified"] and brief["usable"]
    assert {clip["line"] for clip in brief["clips"]} == {1, 2}
    assert all(clip["verified"] is False for clip in brief["clips"])
    # 0.3 s of one word is too little of anyone's voice; that line stays original.
    assert not references["S3"]["usable"] and references["S3"]["clips"] == []


async def test_brief_speakers_use_their_short_atypical_and_unconfirmed_lines(tmp_path):
    from types import SimpleNamespace

    lines = [SpeakerLine(0, 0, 10, "host " * 10, "S1", score=0.9, source="acoustic"),
             # One-off speakers score low against their own tiny cluster.
             SpeakerLine(1, 11, 12.3, "Bro, how are you doing this?", "S2", score=0.54, source="acoustic"),
             # Context could not confirm the voice label; the line keeps it.
             SpeakerLine(2, 13, 14.2, "Take it. You win.", "S3", score=0.4, source="acoustic_uncertain"),
             SpeakerLine(3, 15, 17, "I want heads.", "S3", score=0.7, source="acoustic_uncertain"),
             SpeakerLine(4, 18, 19.2, "Good game.", "S3", score=0.7, source="acoustic"),
             SpeakerLine(5, 20, 21.2, "Heads.", "S4", score=0.7, source="acoustic")]

    class Transcriber:
        async def transcribe(self, wav, expected, *, language):
            return lines[0].text if float(wav) == 0 else "something else"

    diarizer = SpeakerDiarizer(None, transcriber=Transcriber())
    diarizer.groups = {"S1": [0], "S2": [1], "S3": [2, 3, 4], "S4": [5]}
    diarizer.halves = {}
    diarizer.item_of = {line.index: line.index for line in lines}
    diarizer.stats = [{"rms": 0.001 if i == 5 else 0.1, "clipped": 0, "speech_db": -20, "floor_db": -50}
                      for i in range(len(lines))]
    audio = SimpleNamespace(wav=lambda start, end: str(start).encode(), voiced=lambda start, end: [(start, end)])
    references = await diarizer.references(lines, audio)
    assert references["S2"]["usable"] and [c["line"] for c in references["S2"]["clips"]] == [1]
    s3 = references["S3"]
    assert s3["usable"] and not s3["verified"] and {c["line"] for c in s3["clips"]} == {2, 3, 4}
    # An unconfirmed line is never transcript-checked as verified reference.
    assert s3["rejected"] == [{"line": 3, "start": 15, "end": 17, "reason": "uncertain_speaker"}]
    # A short clip rejected for another reason is recorded, not silently dropped.
    assert not references["S4"]["usable"]
    assert references["S4"]["rejected"] == [{"line": 5, "start": 20, "end": 21.2, "reason": "too_quiet"}]


def test_segments_without_a_speaker_reference_use_their_own_voice(tmp_path):
    from types import SimpleNamespace

    speech = tmp_path / "vocals48k.wav"
    with wave.open(str(speech), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(48000)
        wav.writeframes(array.array("h", [1000, -1000] * 48000 * 3).tobytes())
    lines = [SpeakerLine(0, 0.0, 1.0, "Hello.", "S1"), SpeakerLine(1, 1.2, 1.5, "Hit.", None),
             SpeakerLine(2, 1.6, 1.8, "Huh.", "S2"), SpeakerLine(3, 2.0, 2.6, "Heads.", "S3")]
    # The voice inside each line; Huh. is shorter than a usable reference.
    voiced = {0.0: [(0.0, 1.0)], 1.2: [(1.2, 1.5)], 1.6: [(1.65, 1.8)], 2.0: [(2.05, 2.55)]}
    analysis = SimpleNamespace(voiced=lambda start, end: voiced[start])
    write_video(tmp_path / "in.mp4")
    write_srt(tmp_path / "in.srt")
    dubber = FileDubber(Settings(data=tmp_path / "data"),
                        FileDubbingOptions(tmp_path / "in.mp4", tmp_path / "in.srt", tmp_path / "out.mp4"),
                        lambda *_: None)
    segments, owners = dubber._segments(lines)
    references, reference_ids = {"S1": tmp_path / "s1.wav"}, {"S1": "original:s1"}
    voices = dubber._voices(segments, owners, references, reference_ids, analysis, speech, tmp_path)
    assert voices == ["S1", "segment-1", None, "segment-3"]
    # The unknown speaker and the speaker without a reference keep their own voice.
    with wave.open(str(references["segment-1"])) as wav:
        assert wav.getnframes() == round(0.3 * 48000)
    assert reference_ids["segment-3"].startswith("own-segment:")
    assert dubber._own_voices == {1: 0.3, 3: 0.5}
    assert [segment.state for segment in segments] == ["pending", "pending", "skipped", "pending"]
    assert segments[2].skip_reason == "speaker_reference_unavailable"


async def test_reference_skips_a_clip_far_from_the_speakers_voice(tmp_path):
    from types import SimpleNamespace

    lines = [SpeakerLine(0, 0, 3, "first line here", "S1", score=0.85, source="acoustic"),
             # The cleanest recording, but far from the speaker's other clips.
             SpeakerLine(1, 4, 6.4, "between two primes", "S1", score=0.49, source="acoustic"),
             SpeakerLine(2, 7, 10, "third line here", "S1", score=0.8, source="acoustic"),
             SpeakerLine(3, 11, 13, "only this one", "S2", score=0.5, source="acoustic")]
    by_start = {line.start: line for line in lines}

    class Transcriber:
        async def transcribe(self, wav, expected, *, language):
            return by_start[float(wav)].text

    diarizer = SpeakerDiarizer(None, transcriber=Transcriber())
    diarizer.groups = {"S1": [0, 1, 2], "S2": [3]}
    diarizer.halves = {}
    diarizer.item_of = {line.index: line.index for line in lines}
    diarizer.stats = [{"rms": 0.1, "clipped": 0, "speech_db": -20, "floor_db": -50 - 25 * (i == 1)}
                      for i in range(len(lines))]
    audio = SimpleNamespace(wav=lambda start, end: str(start).encode(), voiced=lambda start, end: [(start, end)])
    references = await diarizer.references(lines, audio)
    assert {clip["line"] for clip in references["S1"]["clips"]} == {0, 2}
    assert references["S1"]["rejected"] == [{"line": 1, "start": 4, "end": 6.4, "reason": "atypical_voice"}]
    # Someone with nothing else is still cloned from their own clip, marked unverified.
    assert [(clip["line"], clip["verified"]) for clip in references["S2"]["clips"]] == [(3, False)]


async def test_gemma_dubbing_translation_condenses_long_lines_and_retries_alone():
    from transiroyt.dubbing_translation import GemmaDubbingTranslator, TranslationItem

    class Gemma:
        def __init__(self):
            self.prompts = []

        async def complete_json(self, content, schema, **_):
            prompt = content[0]["text"]
            self.prompts.append(prompt)
            keys = list(schema["properties"])
            if len(keys) > 1:
                raise GemmaOutputLimit("batch too long")
            if "次の吹き替え用の訳は" in prompt:
                return {keys[0]: "短い訳です。"}
            if keys == ["T1"]:
                return {"T1": "とても長い訳になってしまいました、本当に長い訳です。>>"}
            return {keys[0]: ""}

    gemma = Gemma()
    translator = GemmaDubbingTranslator(gemma, title="Board Games", mora_per_second=6.4)
    items = [TranslationItem(1, "It works.", "S1", 1.0), TranslationItem(2, "Next.", "S2", 1.0)]
    result = await translator.translate_batch("en", items, [("S1", "Hi.")], [("S2", "Bye.")])
    # The failed batch is retried per line; an empty answer is a missing line, not text.
    assert result == {1: "短い訳です。"}
    first_single = next(p for p in gemma.prompts if "T1 [S1]" in p and "T2" not in p)
    assert "「Board Games」" in first_single and "[S1] Hi." in first_single and "[S2] Next." in first_single
    assert "翻訳する台詞:\nT1 [S1] (1.0秒以内・6文字以内): It works." in first_single



async def test_gemma_dubbing_translation_keeps_japanese_lines_and_retries_only_untranslated():
    from transiroyt.dubbing_translation import GemmaDubbingTranslator, TranslationItem
    from transiroyt.text_response import UntranslatedLines, validate_text_response

    class Gemma:
        def __init__(self):
            self.requests = []

        async def complete_json(self, content, schema, **_):
            keys = list(schema["properties"])
            self.requests.append(keys)
            answers = {"T1": "動きます。", "T2": "The long line copied in English.", "T3": "次です。"}
            if len(keys) == 1:
                answers["T2"] = "長い台詞です。"
            raw = json.dumps({key: answers[key] for key in keys}, ensure_ascii=False)
            return json.loads(validate_text_response(raw, schema))

    gemma = Gemma()
    translator = GemmaDubbingTranslator(gemma, title=None, mora_per_second=6.4)
    items = [TranslationItem(1, "It works.", "S1", 2.0), TranslationItem(2, "The long line.", "S1", 2.0),
             TranslationItem(3, "Next.", "S2", 2.0)]
    with pytest.raises(UntranslatedLines):
        validate_text_response('{"T1":"動きます。","T2":"Hello"}',
                               {"properties": {"T1": {}, "T2": {}}})
    # Coordinates and numbers have no Japanese spelling: the take check decides them.
    validate_text_response('{"T1":"B2。","T2":"550？"}', {"properties": {"T1": {}, "T2": {}}})
    result = await translator.translate_batch("en", items, [], [])
    assert result == {1: "動きます。", 2: "長い台詞です。", 3: "次です。"}
    # One English copy costs one extra request, not one per line of the batch.
    assert gemma.requests == [["T1", "T2", "T3"], ["T2"]]

def test_dense_lines_ask_irodori_to_speak_faster_but_only_up_to_its_limit():
    segment = Segment(0, 10.0, 12.0, "x", text="あ" * 32)  # 5 s at 6.4 mora/s.
    segment.speech_limit = 13.0
    assert FileDubber._speech_seconds(segment) == pytest.approx(5.0 / file_dubbing.MODEL_SPEEDUP_MAX)
    segment.speech_limit = 14.5
    assert FileDubber._speech_seconds(segment) == pytest.approx(4.5)
    segment.speech_limit = 16.0  # Fits: natural pace.
    assert FileDubber._speech_seconds(segment) == pytest.approx(5.0)


@pytest.mark.parametrize('remote_error', [False, True])
async def test_file_generation_uses_mobile_result_or_mac_recovery(tmp_path, monkeypatch, remote_error):
    import httpx
    from transiroyt.mobile_workers import TTS_IDENTITY, Device, MobileWorkers

    video, subtitles, reference = tmp_path / 'v.mp4', tmp_path / 'v.srt', tmp_path / 'reference.wav'
    video.touch()
    subtitles.touch()
    generated = tmp_path / 'take.wav'
    await FakeTTS().synthesize('テストです。', 2, generated, seconds=2)
    reference.write_bytes(generated.read_bytes())
    pool = MobileWorkers(tmp_path)
    device = Device('phone', 'iPhone', 'worker-token', accepting=True, capabilities={'tts': TTS_IDENTITY})
    pool.devices[device.id] = device
    pool.local_costs['tts'] = 2
    device.costs['tts'] = 1

    class FileTTS(FakeTTS):
        is_coreml = True

    class Transcriber:
        async def transcribe(self, path, expected):
            assert path.is_file()
            return expected

    local = FileTTS()
    dubber = FileDubber(Settings(data=tmp_path), FileDubbingOptions(video, subtitles, tmp_path / 'out.mp4'),
                       tts=local, mobile=pool, transcriber=Transcriber())
    segment = Segment(0, 0, 2, 'Test.', text='テストです。')
    task = asyncio.create_task(dubber._generate_verified(local, segment, reference, 10, tmp_path / 'result'))
    await asyncio.sleep(0)
    for _ in range(3):
        await asyncio.sleep(0)
    [assignment] = device.assignments.values()
    assert base64.b64decode(assignment.payload['reference']) == reference.read_bytes()
    assert assignment.payload['reference_sha256'] == hashlib.sha256(reference.read_bytes()).hexdigest()
    assert 'reference_path' not in assignment.payload
    assert assignment.payload['text'] == segment.text
    # The device times the line like the Mac (FileDubber._speech_seconds).
    assert assignment.payload['seconds'] == FileDubber._speech_seconds(segment)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=pool.worker_app()), base_url='https://mac') as client:
        result = await client.post('/result', headers={'Authorization': 'Bearer worker-token'}, json={
            'task_id': assignment.id, **({'error': 'model failed'} if remote_error else
                                         {'wav': base64.b64encode(generated.read_bytes()).decode(), 'elapsed_ms': 900})})
        assert result.status_code == 200
    verification = await task
    assert verification['ratio'] == 1 and len(verification['attempts']) == 1
    assert verification['attempts'][0]['executor'] == ('mac' if remote_error else 'mobile-coreml')
    assert len(local.calls) == int(remote_error)
    assert (tmp_path / 'result.natural.wav').is_file()


async def test_coreml_retries_until_old_acceptance_threshold(tmp_path, monkeypatch):
    class CoreMLFake(FakeTTS):
        is_coreml = True

    tts = CoreMLFake()
    ratios = iter([0.59, 0.74, 0.8])
    class Transcriber:
        async def transcribe(self, path, expected):
            return 'heard'

    monkeypatch.setattr(speech_check, 'transcript_ratio', lambda *args: next(ratios))
    video, subtitles, reference = (tmp_path / name for name in ('v.mp4', 'v.srt', 'ref.wav'))
    for path in (video, subtitles, reference):
        path.touch()
    dubber = FileDubber(Settings(data=tmp_path), FileDubbingOptions(video, subtitles, tmp_path / 'out.mp4'), tts=tts,
                        transcriber=Transcriber())
    result = await dubber._generate_verified(tts, Segment(0, 0, 2, 'x', text='テスト'), reference, 10, tmp_path / 'result')
    assert len(tts.calls) == 3
    assert [attempt['ratio'] for attempt in result['attempts']] == [0.59, 0.74, 0.8]
    assert [attempt['attempt'] for attempt in result['attempts']] == [1, 2, 3]
    assert result['ratio'] == 0.8


def whisper_words(start, end, text, leading=" "):
    words = text.split()
    step = (end - start) / len(words)
    return [{"start": start + i * step, "end": start + (i + 1) * step, "word": leading + word}
            for i, word in enumerate(words)]


def test_transcribed_segments_become_one_subtitle_per_sentence_or_pause():
    from transiroyt.subtitle_files import format_srt, transcribed_cues

    segments = [
        # Two sentences of different speakers in one Whisper segment.
        {"words": whisper_words(0.5, 3.0, "Welcome to the show, everyone.")
         + whisper_words(3.2, 5.6, "Thanks, it is great to be here.")},
        # A long pause without punctuation.
        {"words": whisper_words(6.0, 7.0, "so anyway") + whisper_words(9.0, 10.0, "we started")},
        {"words": [{"start": 11.0, "end": 11.4, "word": "こんにちは"}, {"start": 11.4, "end": 12.0, "word": "。"}]},
        # A word Whisper could not place in time.
        {"words": [{"start": 13.0, "end": 13.0, "word": " uh"}]},
    ]
    cues = transcribed_cues(segments)
    assert [(c.start, round(c.end, 3), c.text) for c in cues] == [
        (0.5, 3.0, "Welcome to the show, everyone."), (3.2, 5.6, "Thanks, it is great to be here."),
        (6.0, 7.0, "so anyway"), (9.0, 10.0, "we started"), (11.0, 12.0, "こんにちは。")]
    assert [(c.start, c.end, c.text) for c in parse_srt(format_srt(cues))] == [
        (c.start, round(c.end, 3), c.text) for c in cues]


def apple_words(script):
    """Apple's timed words: contiguous, a pause given to the word after it."""
    words, previous = [], 0.0
    for _, start, end, text in script:
        timed = whisper_words(start, end, text)
        timed[0]["start"] = previous
        words += timed
        previous = end
    return words


class ReviewTranscriber(FakeTranscriber):
    def __init__(self, tts, heard):
        super().__init__(tts)
        self.heard, self.calls = heard, []

    async def detect_language(self, wav, windows):
        with wave.open(str(wav)) as audio:
            assert (audio.getnchannels(), audio.getframerate()) == (1, 16000)
        self.calls.append(("detect", windows))
        return "en", {"en": 0.99}

    async def pauses(self, wav, minimum):
        return [[0.0, SCRIPT[0][1]]] + [[a[2], b[1]] for a, b in zip(SCRIPT, SCRIPT[1:], strict=False)]

    async def transcribe_windows(self, wav, windows, language):
        self.calls.append(("review", windows, language))
        return [(self.heard, True) for _ in windows]


async def test_video_without_subtitles_is_dubbed_from_reviewed_system_transcript(
        tmp_path, exact_reference_check, fake_separation):
    video = tmp_path / "source.mp4"
    write_video(video)
    settings = Settings(data=tmp_path / "data")
    tts = FakeTTS()
    # macOS mishears a word; Whisper skips a whole sentence.
    misheard = [(s, a, b, text.replace("boat", "goat")) for s, a, b, text in SCRIPT]
    heard = " ".join(text for _, _, _, text in SCRIPT if not text.startswith("It took"))
    transcriber = ReviewTranscriber(tts, heard)
    recognized = []

    async def system(wav, language, *, timeout):
        recognized.append(language)
        return {"locale": "en-US", "segments": [{"start": 0.0, "end": 16.6, "text": "",
                                                 "words": apple_words(misheard)}]}

    def dubber(subtitles, output):
        return FileDubber(settings, FileDubbingOptions(video, subtitles, output, transcribe=True), tts=tts,
                          gemma=FakeGemma(tts), translator=FakeTranslator(), embeddings=FakeEmbeddings(),
                          transcriber=transcriber, system_transcriber=system)

    subtitles = tmp_path / "job" / "transcribed.srt"
    report = await dubber(subtitles, tmp_path / "out.mp4").run()
    assert recognized == ["en"] and [call[0] for call in transcriber.calls] == ["detect", "review"]
    assert transcriber.calls[1][1:] == ([(0.3, 16.8)], "en")
    # Whisper's word replaces the misheard one, the skipped sentence keeps
    # macOS's words, and each subtitle spans its speech between the pauses.
    assert [(c.start, round(c.end, 3), c.text) for c in parse_srt(subtitles.read_text())] == [
        (a, b, text) for _, a, b, text in SCRIPT]
    assert report["source_language"] == "en"
    assert "apple-speechtranscriber" in report["subtitle_source"] and "reviewed by" in report["subtitle_source"]
    assert report["summary"]["ready"] > 0 and len(report["speakers"]) == 2
    # Separation runs once and serves both the transcript and the dub.
    assert fake_separation.calls == 1

    # A rerun reuses the transcript of the same video bytes.
    again = await dubber(tmp_path / "job2" / "transcribed.srt", tmp_path / "again.mp4").run()
    assert recognized == ["en"] and len(transcriber.calls) == 2 and again["segments"] == report["segments"]


async def test_language_macos_cannot_transcribe_is_written_by_whisper(tmp_path, exact_reference_check):
    from transiroyt.speech_transcript import SpeechTranscriptionUnsupported

    video = tmp_path / "source.mp4"
    write_video(video)
    tts = FakeTTS()
    timed = []

    class Transcriber(ReviewTranscriber):
        async def transcribe_timed(self, wav, language, *, timeout):
            timed.append(language)
            return {"language": "en", "segments": [
                {"start": a, "end": b, "text": text, "words": whisper_words(a, b, text)}
                for _, a, b, text in SCRIPT]}

    async def system(wav, language, *, timeout):
        raise SpeechTranscriptionUnsupported("enはmacOS標準の文字起こしに対応していません。")

    subtitles = tmp_path / "job" / "transcribed.srt"
    report = await FileDubber(Settings(data=tmp_path / "data"),
                              FileDubbingOptions(video, subtitles, tmp_path / "out.mp4", transcribe=True),
                              tts=tts, gemma=FakeGemma(tts), translator=FakeTranslator(), embeddings=FakeEmbeddings(),
                              transcriber=Transcriber(tts, ""), system_transcriber=system).run()
    assert timed == ["en"]
    assert [(c.start, round(c.end, 3), c.text) for c in parse_srt(subtitles.read_text())] == [
        (a, b, text) for _, a, b, text in SCRIPT]
    assert report["subtitle_source"].startswith("mlx-community/whisper-large-v3-turbo")


async def test_synthesis_prefetch_does_not_wait_for_untranslated_neighbour(tmp_path, monkeypatch):
    video, subtitles = tmp_path / 'v.mp4', tmp_path / 'v.srt'
    video.touch()
    subtitles.touch()
    dubber = FileDubber(Settings(data=tmp_path), FileDubbingOptions(video, subtitles, tmp_path / 'out.mp4'), tts=FakeTTS())
    segments = [Segment(i, i * 3, i * 3 + 2, str(i), text=f'発話{i}') for i in range(4)]
    translated = {i: asyncio.Event() for i in range(4)}
    started = {i: asyncio.Event() for i in range(4)}
    translated[0].set()
    translated[2].set()
    async def generate(tts, segment, reference, steps, stem):
        started[segment.index].set()
        await asyncio.Event().wait()
    monkeypatch.setattr(dubber, '_generate_verified', generate)
    task = asyncio.create_task(dubber._synthesize(segments, ['S1'] * 4, {'S1': tmp_path / 'ref.wav'}, 10, tmp_path, translated=translated))
    try:
        await asyncio.wait_for(started[0].wait(), 1)
        await asyncio.wait_for(started[2].wait(), 1)
        assert not started[1].is_set()
        translated[3].set()
        await asyncio.wait_for(started[3].wait(), 1)
        assert not started[1].is_set()
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def test_translation_runs_at_most_translation_parallel_batches(tmp_path, monkeypatch):
    video, subtitles = tmp_path / 'v.mp4', tmp_path / 'v.srt'
    video.touch()
    subtitles.touch()
    active, peak, release = 0, [], asyncio.Event()

    class Translator(FakeTranslator):
        async def translate_batch(self, language, items, before, after):
            nonlocal active
            active += 1
            peak.append(active)
            try:
                await release.wait()
                return await super().translate_batch(language, items, before, after)
            finally:
                active -= 1

    monkeypatch.setattr(file_dubbing, 'BATCH_SIZE', 1)
    dubber = FileDubber(Settings(data=tmp_path), FileDubbingOptions(video, subtitles, tmp_path / 'out.mp4'),
                        translator=Translator())
    segments = [Segment(i, i * 3, i * 3 + 2, f'line {i}') for i in range(6)]
    task = asyncio.create_task(dubber._translate(None, segments, ['S1'] * 6, 'en', tmp_path))
    for _ in range(10):
        await asyncio.sleep(0)
    assert max(peak) == file_dubbing.TRANSLATION_PARALLEL
    release.set()
    await asyncio.wait_for(task, 1)
    assert all(segment.text for segment in segments)


@pytest.mark.parametrize('replacement', ['短い訳。', None])
async def test_measured_slow_take_is_retranslated_once_and_never_extremely_sped_up(tmp_path, monkeypatch, replacement):
    video, subtitles = tmp_path / 'v.mp4', tmp_path / 'v.srt'
    video.touch()
    subtitles.touch()
    dubber = FileDubber(Settings(data=tmp_path),
                       FileDubbingOptions(video, subtitles, tmp_path / 'out.mp4'), tts=FakeTTS())
    original = '元の長い翻訳です。'
    segment = Segment(0, 0, 2, 'Original source.', text=original)
    segment.speech_limit = 2
    generated, budgets = [], []

    async def verified(tts, segment, reference, steps, stem):
        generated.append(segment.text)
        duration = 8 if segment.text == original else 2
        with wave.open(str(stem.with_suffix('.natural.wav')), 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(48000)
            wav.writeframes(b'\0\0' * (duration * 48000))
        return {'ratio': 1, 'coverage': 1, 'attempts': [], 'seconds': duration,
                'transcript': segment.text, 'tail': {}}

    async def retime(turn, seconds):
        assert turn.source == 'Original source.'
        budgets.append(seconds)
        return replacement

    monkeypatch.setattr(dubber, '_generate_verified', verified)
    await dubber._synthesize([segment], ['S1'], {'S1': tmp_path / 'reference.wav'},
                             10, tmp_path, retime=retime)
    assert len(budgets) == 1 and 0 < budgets[0] < 2
    assert segment.audio['slow_take'] == {'measured_seconds': 8, 'available_seconds': 2,
                                          'retranslation': 'used' if replacement else 'no_answer'}
    if replacement:
        assert generated == [original, replacement]
        assert segment.state == 'ready' and segment.audio['tempo'] <= 1.25
    else:
        assert generated == [original]
        assert segment.state == 'skipped' and segment.skip_reason == 'speech_too_fast'
        assert segment.audio['generated_seconds'] == 8 and segment.audio['allowed_seconds'] == 2
        assert not (tmp_path / 'clips' / 'segment-0.wav').exists()


def test_a_reply_too_long_for_its_slot_may_start_in_the_silence_before_it():
    from transiroyt.speech_timing import earliest_start

    segments = [Segment(0, 0.0, 1.0, "D8."), Segment(1, 1.1, 1.4, "Hit."), Segment(2, 1.6, 2.0, "B9.")]
    assert earliest_start(segments, segments[1], 1.1) == pytest.approx(1.0)  # Not over D8.
    segments[0].end = 0.5
    assert earliest_start(segments, segments[1], 1.1) == pytest.approx(0.9)
    assert earliest_start(segments, segments[0], 0.0) == 0.0


def test_edge_silence_is_not_counted_as_speaking_time(tmp_path):
    from transiroyt.audio import trim_edge_silence

    path = tmp_path / "take.wav"
    voice = [8000 * (1 if i % 40 < 20 else -1) for i in range(round(0.35 * 48000))]
    samples = [0] * round(0.1 * 48000) + voice + [0] * round(0.2 * 48000)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(48000)
        wav.writeframes(array.array("h", samples).tobytes())
    assert trim_edge_silence(path) == pytest.approx(0.15, abs=0.021)
    with wave.open(str(path)) as wav:
        assert wav.getnframes() / 48000 == pytest.approx(0.35 + 0.03 + 0.12, abs=0.021)
    assert trim_edge_silence(path) == 0.0


async def test_separation_change_gets_new_analysis_cache_without_removing_old(tmp_path, monkeypatch):
    video, subtitles = tmp_path / "source.mp4", tmp_path / "source.srt"
    write_video(video, seconds=2)
    subtitles.touch()
    dubber = FileDubber(Settings(data=tmp_path / "data"),
                       FileDubbingOptions(video, subtitles, tmp_path / "out.mp4"))
    previous = await dubber._work_directory("same-subtitles")
    marker = previous / "speakers.json"
    marker.write_text("previous separation analysis")
    monkeypatch.setattr(file_dubbing, "SEPARATION_VERSION", "different-windows")
    current = await dubber._work_directory("same-subtitles")
    assert current.resolve() != previous.resolve()
    assert not (current / "speakers.json").exists()
    assert marker.read_text() == "previous separation analysis"
