"""Dub a local MP4 with its SRT: per-speaker references, then a mixed MP4.

A video without subtitles is transcribed first: macOS speech recognition writes
the SRT, Whisper reviews it on the separated speech (speech_transcript), and the
rest of the pipeline runs on it unchanged.

The soundtrack is first split into speech and background. Speakers are found
per caption turn on the speech, every line is translated by the configured text engine with its
context and time budget, and the mix keeps the background while replacing
only the speech that was dubbed.

Stages are cached in a work directory keyed by the inputs, so an interrupted job
resumes without repeating separation, speaker analysis, translation or synthesis.
"""
from __future__ import annotations

import array
import asyncio
import hashlib
import json
import logging
import math
import tempfile
import time
import wave
from collections import Counter
from contextlib import AsyncExitStack
from functools import partial
from pathlib import Path

from .audio import SAMPLE_RATE, fit_audio, trim_edge_silence, trim_generated_tail
from .caption_annotations import EMOJI, reaction_emojis
from .captions import Cue
from .config import Settings
from .coreml_tts import CoreMLTTS
from .dubbing_captions import detect_language, reactions_after, speaker_segments
from .dubbing_media import (
    extract_audio,
    mux,
    place_segments,
    playable_video,
    probe,
    write_speech_gain,
)
from .dubbing_options import LANGUAGE_CODE, FileDubbingOptions
from .dubbing_translation import (
    BATCH_SIZE,
    CONTEXT_AFTER,
    CONTEXT_BEFORE,
    SHORTEN_TEMPO,
    GemmaDubbingTranslator,
    TranslationItem,
    with_events,
)
from .errors import FileDubbingError
from .gemma_audio import GemmaAudio
from .japanese_reading import align_latin_spellings
from .mobile_workers import TTS_IDENTITY
from .process import ProcessError, run_process
from .reference_conversion import (
    CONVERSION_TEXT,
    REFERENCE_CONVERSION_VERSION,
    ReferenceConverter,
    model_ready,
)
from .session_models import Segment
from .speaker_diarization import (
    ACTIVITY_MODEL_REPO,
    ACTIVITY_MODEL_REVISION,
    ANALYSIS_RATE,
    AnalysisAudio,
    EmbeddingWorker,
    SpeakerDiarizer,
    SpeakerLine,
)
from .speech_check import (
    SPEECH_ACCEPT_RATIO,
    SPEECH_COVERAGE,
    check_take,
    compare,
    same_reading,
    speech_verified,
)
from .speech_length import natural_seconds
from .speech_timing import earliest_start, prepare_speech_limits, speech_end, speech_window
from .speech_transcript import (
    SYSTEM_IDENTITY,
    language_windows,
    reviewed_transcript,
    system_transcript,
)
from .subtitle_files import (
    caption_events,
    format_srt,
    parse_srt,
    spoken_cues,
)
from .vocal_separation import MODEL_IDENTITY as SEPARATION_IDENTITY
from .vocal_separation import SEPARATION_VERSION, VocalSeparator
from .whisper_transcription import MODEL_IDENTITY as SPEECH_CHECK_IDENTITY
from .whisper_transcription import WhisperTranscriber, WhisperUnavailable

logger = logging.getLogger(__name__)

PIPELINE_VERSION = "file-dubbing-v1"
# The reference conversion changed every reference: never reuse takes made from the old one.
SYNTHESIS_VERSION = "coreml-mf-verified-v4-reference-conversion"
DIARIZATION_VERSION = "turns-vocals-nemotron-ecapa-v9-caption-speaker-boundaries"
TRANSCRIPT_VERSION = "apple-whisper-review-v2"
SPEAKER_IDENTITY = f"{ACTIVITY_MODEL_REPO}@{ACTIVITY_MODEL_REVISION}+speechbrain/spkrec-ecapa-voxceleb"
DEFAULT_STEPS = 10
# Irodori's own pace on sentences of 24+ mora (median of 6.0-6.9) in the test runs.
MORA_PER_SECOND = 6.4
MIN_SPEECH_SECONDS = 0.6
# When a line does not fit its window, Irodori itself speaks up to 1.25x faster
# than its own pace; atempo fits only the rest. On 16 dense lines of the host,
# 0.8x of the natural length passed the transcript check as often as 1.0x
# (14/16 at ratio >= 0.75); 0.7x passed 12/16.
MODEL_SPEEDUP_MAX = 1.25
SEEDS = (42, 7, 1234)
# A line of a few morae ("うん。", "表。") is often heard as another short word
# (おー, 思って) on its first takes and costs about a second per take. Of 7 short
# lines that failed three takes on C782A726, takes 4-6 passed 4 under the same
# gate.
SHORT_LINE_SEEDS = (*SEEDS, 3, 5, 11)
SHORT_LINE_SECONDS = 2.0
# Style caption for lines with a captioned laugh (with the default voice only).
# On five host lines it raised AudioSet laughter scores in four, words kept.
LAUGHING_VOICE = "明瞭で自然な日本語。笑いながら楽しそうに話している。"
REFERENCE_GAP_SECONDS = 0.2
# A segment without a speaker reference is dubbed from its own voice. On
# C782A726, 17 of 19 such lines (0.27-2.9 s of voice) passed the speech check
OWN_SEGMENT_MIN_SECONDS = 0.25
# A converted reference reads one fixed Japanese sentence in the speaker's own
# voice. Too short, misread, or another voice must keep the original reference
# (benchmarks: CER 16.8% -> 4.0%, but ECAPA drops a little even then).
REFERENCE_CONVERSION_MIN_SECONDS = 4.0
REFERENCE_CONVERSION_MAX_SECONDS = 30.0
REFERENCE_SIMILARITY_MIN = 0.3
REFERENCE_SIMILARITY_GAP_SECONDS = 0.3
CONVERSION_CACHE_DIRECTORY = "reference-conversion"
SYNTHESIS_LOOKAHEAD = 4
# Lines queued per executor (Mac generator or paired device). The line being
# placed waits for its Whisper check, retakes and retiming; with a 4-line window
# and four executors the queue was empty, and in a 53-minute job the two Mac
# generators were idle 63% and 76% of the time.
LOOKAHEAD_PER_EXECUTOR = 3
TRANSLATION_PARALLEL = 2


def _digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _audio_identity(path: Path) -> str | None:
    """Identify decoded programme audio for sharing work across remuxes."""
    try:
        with wave.open(str(path), "rb") as wav:
            digest = hashlib.sha256(json.dumps([
                wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getnframes(),
            ]).encode())
            for block in iter(lambda: wav.readframes(65536), b""):
                digest.update(block)
            return digest.hexdigest()
    except (OSError, EOFError, wave.Error):
        return None


def _cue_dict(cue: Cue) -> dict:
    return {"start": cue.start, "end": cue.end, "text": cue.text}


def _write_json(path: Path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=1))
    temporary.replace(path)


def cut_reference(source: Path, clips: list[dict], directory: Path) -> tuple[Path, str]:
    """Concatenate the voiced spans of verified clips from the 48 kHz speech, short pauses between."""
    with wave.open(str(source), "rb") as wav:
        if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) != (1, 2, SAMPLE_RATE):
            raise FileDubbingError("参照音声の元データ形式が不正です。")
        frames = wav.getnframes()
        parts = []
        for start, end in (span for clip in clips for span in clip["voiced"]):
            a = max(0, round(start * SAMPLE_RATE))
            b = min(frames, round(end * SAMPLE_RATE))
            wav.setpos(a)
            data = array.array("h", wav.readframes(b - a))
            fade = min(len(data) // 2, round(SAMPLE_RATE * 0.01))
            for i in range(fade):  # Remove clicks at cut points.
                data[i] = data[i] * i // fade
                data[-1 - i] = data[-1 - i] * i // fade
            if parts:
                parts.append(b"\0\0" * round(REFERENCE_GAP_SECONDS * SAMPLE_RATE))
            parts.append(data.tobytes())
    content = b"".join(parts)
    digest = hashlib.sha256(content).hexdigest()
    target_directory = directory / digest
    target_directory.mkdir(parents=True, exist_ok=True)
    target = target_directory / "reference.wav"
    if not target.is_file():
        temporary = target_directory / "reference.tmp.wav"
        with wave.open(str(temporary), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(SAMPLE_RATE)
            wav.writeframes(content)
        temporary.replace(target)
    return target, digest


def _converted_seconds(path: Path) -> float:
    """Length of a converted reference; the worker must have written 48 kHz mono PCM16."""
    try:
        with wave.open(str(path), "rb") as wav:
            if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) != (1, 2, SAMPLE_RATE):
                raise FileDubbingError("変換後の参照音声が48kHzモノラルの16bit PCMではありません。")
            return wav.getnframes() / wav.getframerate()
    except (OSError, EOFError, wave.Error) as exc:
        raise FileDubbingError(f"変換後の参照音声を読み取れませんでした: {exc}") from exc


class _EarlyTranslation:
    """Lines translated before synthesis listens, replayed to it when it does."""

    def __init__(self):
        self.task, self.finished, self.listener = None, [], None

    def __call__(self, segment):
        self.finished.append(segment)
        if self.listener:
            self.listener(segment)

    def listen(self, listener):
        self.listener = listener
        for segment in self.finished:
            listener(segment)


class FileDubber:
    def __init__(self, settings, options: FileDubbingOptions, progress=None, *,
                 gemma=None, embeddings=None, translator=None, tts=None, separator=None,
                 transcriber=None, system_transcriber=None, download_progress=None,
                 mobile=None, reference_converter=None):
        options.validate()
        self.mobile = mobile
        self.settings, self.options = settings, options
        self._progress = progress or (lambda *_: None)
        self.gemma = gemma
        from .task_routing import TaskRouter, load_routes
        # Injected text models (tests) replace the routed engines.
        self.task_router = None if gemma or translator else TaskRouter(
            settings, load_routes(settings.data), allow_download=options.allow_download,
            download_progress=download_progress)
        self.text_identity = (self.task_router.text('language').model.identity if self.task_router
                              else "local-text-v1")
        self.embeddings = embeddings
        self.translator = translator
        self.tts = tts
        self.separator = separator
        self.transcriber = transcriber
        self.system_transcriber = system_transcriber or partial(system_transcript, settings.system_transcribe_binary)
        self.download_progress = download_progress
        # IndexTTS-2.5 stays loaded for the job and is stopped with the other
        # resources (run); a test passes a stand-in here.
        self.reference_converter = reference_converter
        self._reference_conversions = {}
        # Segment index -> seconds of its own voice used as its reference (_voices).
        self._own_voices = {}
        self._synthesis_lock = asyncio.Semaphore(getattr(tts, "capacity", 1))
        self._translation_cache_lock = asyncio.Lock()
        # Greedy decoding repeats a failure: within one job, ask once per line.
        self._failed_translations = set()
        # Translation rejections and shortening decisions, for the report.
        self._translation_checks = {}
        # Started during the reference checks; synthesis takes it over (_EarlyTranslation).
        self._translation = None
        self._duration = None
        self._separated = None
        self._picture = None
        # Caption annotations that are not voiced: reactions, sounds (subtitle_files).
        self._events = []
        self._subtitle_source = "subtitle file"

    @property
    def title(self):
        # Only a YouTube job knows the title. A local file name is not used: the
        # same programme under another name must reuse the same translations.
        return self.options.title

    def progress(self, stage: str, fraction: float, message: str = ""):
        self._progress(stage, max(0.0, min(1.0, fraction)), message)

    async def _work_directory(self, subtitles_digest: str) -> Path:
        video_digest = await asyncio.to_thread(_digest_file, self.options.video)
        identity = json.dumps([PIPELINE_VERSION, SEPARATION_IDENTITY, SEPARATION_VERSION, "input-v2", video_digest,
                               subtitles_digest, self.options.speakers])
        base = self.settings.data / "file-dubbing"
        base.mkdir(parents=True, exist_ok=True)
        index = base / ("input-" + hashlib.sha256(identity.encode()).hexdigest()[:24])
        if index.is_dir():
            return index

        def link(path, target):
            try:
                path.symlink_to(target.name, target_is_directory=True)
            except FileExistsError:
                if not path.is_dir():
                    raise

        # MP4 headers and video encodings can differ between downloads. The
        # voice models depend on decoded audio and subtitles, not those bytes.
        # Keep a byte-key index for fast exact repeats, and a PCM-key directory
        # that also shares results across remuxes of the same programme audio.
        with tempfile.TemporaryDirectory(prefix=".input-", dir=base) as temporary:
            programme = Path(temporary) / "audio48k.wav"
            await extract_audio(self.options.video, programme, SAMPLE_RATE)
            expected = await asyncio.to_thread(_audio_identity, programme)
            if expected is None:
                raise FileDubbingError("動画から取り出した音声を確認できませんでした。")
            identity = json.dumps([PIPELINE_VERSION, SEPARATION_IDENTITY, SEPARATION_VERSION, "audio-content-v2", expected,
                                   subtitles_digest, self.options.speakers])
            directory = base / hashlib.sha256(identity.encode()).hexdigest()[:24]
            if not directory.is_dir():
                directory.mkdir(exist_ok=True)
                programme.replace(directory / "audio48k.wav")
            link(index, directory)
            return index

    async def run(self) -> dict:
        started = time.monotonic()
        options = self.options
        self.progress("prepare", 0.0, "動画と字幕を確認しています")
        media = await probe(options.video)
        self._duration = media["duration"]
        owns_gemma = self.gemma is None
        gemma = self.gemma or GemmaAudio(self.settings, allow_download=options.allow_download,
                                          download_progress=self.download_progress,
                                          parallel=SYNTHESIS_LOOKAHEAD)
        original_tts, original_transcriber = self.tts, self.transcriber
        preparations = []
        try:
            async with AsyncExitStack() as resources:
                if self.task_router:
                    resources.push_async_callback(self.task_router.close)
                if self.transcriber is None:
                    # Loads nothing until a take is checked or start() is called.
                    self.transcriber = WhisperTranscriber(self.settings, allow_download=options.allow_download,
                                                          download_progress=self.download_progress)
                    resources.push_async_callback(self.transcriber.close)
                # IndexTTS-2.5 is started on the first reference and stopped
                # after the last; this also stops it if the job ends mid-conversion.
                resources.push_async_callback(self._close_reference_converter)
                # The picture does not depend on the dub: converting it overlaps
                # separation, speaker analysis and synthesis instead of the final mux.
                converted = options.output.with_name(f".{options.output.stem}.video.mp4")
                self._picture = asyncio.create_task(playable_video(options.video, converted))
                resources.push_async_callback(self._discard_picture, converted)
                if owns_gemma:
                    resources.push_async_callback(gemma.close)
                try:
                    separator = self.separator or VocalSeparator(
                        self.settings, allow_download=options.allow_download,
                        download_progress=self.download_progress)
                    language = options.source_language
                    if options.transcribe:
                        language = await self._transcribe_speech(separator, media["duration"])
                    try:
                        text = options.subtitles.read_text(encoding="utf-8-sig")
                    except UnicodeDecodeError as exc:
                        raise FileDubbingError("SRTはUTF-8で保存してください。") from exc
                    parsed = parse_srt(text)
                    cues = spoken_cues(parsed)
                    self._events = caption_events(parsed)
                    if not cues:
                        raise FileDubbingError("SRTに発話の字幕がありません。")
                    work = await self._work_directory(hashlib.sha256(text.encode()).hexdigest())
                    # An existing generation directory may be a completely cached
                    # export. Only a fresh work area starts models speculatively.
                    fresh = not (work / "clips").exists()
                    if fresh:
                        if self.tts is None:
                            self.tts = CoreMLTTS(self.settings, allow_download=self.options.allow_download, download_progress=self.download_progress)
                            resources.push_async_callback(self.tts.close)
                            preparations.append(self.tts.preload())
                        if hasattr(self.transcriber, "start"):
                            preparations.append(self.transcriber.start())
                    if self._separated is None:
                        self.progress("prepare", 0.2, "声と背景音を分離しています")
                        self._separated = await separator.separate(options.video, work / "separation")
                    vocals, background = self._separated
                    # Speaker analysis and references use the separated speech: the
                    # TTS copies background music and noise from its reference.
                    analysis_path, speech_path = work / "vocals16k.wav", work / "vocals48k.wav"
                    self.progress("prepare", 0.5, "分離した声を準備しています")
                    await extract_audio(vocals, analysis_path, ANALYSIS_RATE)
                    await extract_audio(vocals, speech_path, SAMPLE_RATE)
                    analysis = AnalysisAudio(analysis_path)
                    if language == "auto":
                        language_cache = work / "language.json"
                        try:
                            saved = json.loads(language_cache.read_text()) if language_cache.is_file() else {}
                        except (OSError, ValueError):
                            saved = {}
                        if (isinstance(saved, dict) and saved.get("model") == self.text_identity
                                and saved.get("version") == "subtitle-language-v3"
                                and LANGUAGE_CODE.fullmatch(str(saved.get("language", "")))):
                            language = saved["language"]
                        else:
                            self.progress("prepare", 0.6, "字幕の言語を判定しています")
                            language = await detect_language(self.task_router.text('language') if self.task_router else gemma, cues)
                            _write_json(language_cache, {"model": self.text_identity, "version": "subtitle-language-v3",
                                                         "language": language})
                    speakers = await self._speakers(gemma, cues, analysis, speech_path, work, language=language)
                    return await self._dub(gemma, media, language, speakers, work, started, analysis, speech_path)
                finally:
                    if self._translation is not None:
                        preparations.append(self._translation.task)
                        self._translation = None
                    for task in preparations:
                        if task is not None and not task.done():
                            task.cancel()
                    await asyncio.gather(*(t for t in preparations if t is not None), return_exceptions=True)
        finally:
            self.tts, self.transcriber = original_tts, original_transcriber

    async def _transcribe_speech(self, separator, duration: float) -> str:
        """Write the SRT from the speech; return the source language.

        macOS transcribes the soundtrack on the Neural Engine while separation
        runs on the GPU; Whisper then reviews every window of it on the
        separated speech (speech_transcript). Kept per video bytes, so a rerun
        reuses the separation and transcript.
        """
        options = self.options
        directory = (self.settings.data / "file-dubbing-transcripts"
                     / (await asyncio.to_thread(_digest_file, options.video))[:24])
        requested = None if options.source_language == "auto" else options.source_language
        identity = {"model": SPEECH_CHECK_IDENTITY, "system": SYSTEM_IDENTITY, "version": TRANSCRIPT_VERSION,
                    "language": requested, "separation": [SEPARATION_IDENTITY, SEPARATION_VERSION]}
        cache = directory / "transcript.json"
        try:
            transcript = json.loads(cache.read_text())
        except (OSError, ValueError):
            transcript = {}
        system = None
        if transcript.get("identity") != identity:
            soundtrack = directory / "soundtrack16k.wav"
            directory.mkdir(parents=True, exist_ok=True)
            await extract_audio(options.video, soundtrack, ANALYSIS_RATE)
            language = requested or await self._spoken_language(soundtrack, duration)
            system = asyncio.create_task(self.system_transcriber(soundtrack, language, timeout=600 + duration))
        self.progress("prepare", 0.05, "声と背景音を分離しています" if system is None
                      else "声と背景音を分離しながら、macOS標準の音声認識で字幕を作成しています")
        try:
            self._separated = await separator.separate(options.video, directory / "separation")
            speech = directory / "vocals16k.wav"
            await extract_audio(self._separated[0], speech, ANALYSIS_RATE)
            if system is not None:
                transcript = {"identity": identity, **await self._reviewed_transcript(system, speech, language, duration)}
                _write_json(cache, transcript)
        finally:
            if system is not None and not system.done():
                system.cancel()
                await asyncio.gather(system, return_exceptions=True)
        cues = [Cue(**cue) for cue in transcript["cues"]]
        if not cues:
            raise FileDubbingError("動画の音声から発話を書き起こせませんでした。")
        self._subtitle_source = transcript["source"]
        options.subtitles.parent.mkdir(parents=True, exist_ok=True)
        options.subtitles.write_text(format_srt(cues), encoding="utf-8")
        logger.info("file dubbing transcribed %d subtitles language=%s source=%s review=%s", len(cues),
                    transcript["language"], transcript["source"], transcript.get("review"))
        return requested or transcript["language"]

    async def _spoken_language(self, soundtrack: Path, duration: float) -> str:
        self.progress("prepare", 0.02, "話されている言語を判定しています")
        language, probabilities = await self.transcriber.detect_language(soundtrack, language_windows(duration))
        logger.info("file dubbing spoken language=%s probabilities=%s", language, probabilities)
        return language

    async def _reviewed_transcript(self, system, speech: Path, language: str, duration: float) -> dict:
        try:
            transcript = await reviewed_transcript(self.transcriber, system, speech, language, duration,
                                                   lambda fraction, message: self.progress("prepare", fraction, message))
        except (ProcessError, OSError) as exc:
            raise FileDubbingError(f"macOS標準の音声認識で字幕を作成できませんでした: {str(exc).strip()[-300:]}") from exc
        return {**transcript, "cues": [_cue_dict(cue) for cue in transcript["cues"]]}

    async def _discard_picture(self, converted: Path):
        self._picture.cancel()
        await asyncio.gather(self._picture, return_exceptions=True)
        self._picture = None
        converted.unlink(missing_ok=True)

    async def _dub(self, gemma, media, language, speakers, work, started, analysis, speech_path) -> dict:
        options = self.options
        lines = [SpeakerLine(**{k: v for k, v in line.items()
                                if k in SpeakerLine.__dataclass_fields__})
                 for line in speakers["lines"]]
        references = {name: Path(item["path"]) for name, item in speakers["references"].items()
                      if item.get("path")}
        # The source digest names both caches: a converted reference lives under
        # a constant directory name, so synthesis identity must not read the path.
        reference_ids = {name: item["digest"] for name, item in speakers["references"].items()
                         if name in references}
        references, self._reference_conversions = await self._convert_references(
            work, references, reference_ids)
        # The conversion outcome is part of a take's identity: a job that fell
        # back to the original reference must not reuse its takes after a later
        # run converted the same source digest.
        converted = {name for name, item in self._reference_conversions.items()
                     if item["state"] in {"converted", "cached"}}
        reference_ids = {name: (f"{REFERENCE_CONVERSION_VERSION}:{digest}" if name in converted
                                else f"original:{digest}")
                         for name, digest in reference_ids.items()}
        segments, owners = self._segments(lines)
        # Added after the conversion: IndexTTS is not given a fraction of a second.
        voices = self._voices(segments, owners, references, reference_ids, analysis, speech_path, work)
        prepare_speech_limits(segments, media["duration"])
        steps = options.steps or max(DEFAULT_STEPS, self.settings.steps)
        return await self._render_dub(gemma, media, language, speakers, work, started,
                                      segments, owners, voices, references, reference_ids, steps)

    async def _convert_references(self, work: Path, references: dict[str, Path],
                                  reference_ids: dict[str, str]):
        """Read the fixed Japanese sentence in each speaker's voice for the TTS reference.

        A converted take replaces the reference only after the format, speech and
        voice-identity gates pass; otherwise the original reference stays and the
        job continues with a warning (the report records why).
        """
        if not references:
            return references, {}
        if not model_ready(self.settings):
            if not self.options.allow_download:
                raise FileDubbingError(
                    "参照音声モデル（IndexTTS-2.5）を取得できません。"
                    "モデルのダウンロードを許可して再試行してください。")
            self.progress("reference_conversion", 0.0, "参照音声モデル（IndexTTS-2.5）を取得しています")
            from .model_downloads import download_model

            await download_model(self.settings, "indextts_model", self.download_progress)
        if self.reference_converter is None:
            self.reference_converter = ReferenceConverter(self.settings)
        cache = self.settings.data / CONVERSION_CACHE_DIRECTORY
        cache.mkdir(parents=True, exist_ok=True)
        converted, conversions = {}, {}
        total = len(references)
        for number, (speaker, reference) in enumerate(references.items()):
            self.progress("reference_conversion", number / total,
                          f"参照音声を作り直しています（{number + 1}/{total}）")
            target = cache / f"{REFERENCE_CONVERSION_VERSION}-{reference_ids[speaker]}.wav"
            cached = self._cached_conversion(target)
            if cached is not None:
                converted[speaker] = target
                conversions[speaker] = {"state": "cached", "path": str(target), **cached}
                logger.info("file dubbing reference conversion cached speaker=%s", speaker)
                continue
            try:
                converted[speaker] = target
                conversions[speaker] = {"state": "converted", "path": str(target),
                                        **await self._convert_reference(reference, target, work)}
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                converted.pop(speaker, None)
                reason = str(exc) or type(exc).__name__
                logger.warning("file dubbing reference conversion fallback speaker=%s reason=%s",
                               speaker, reason)
                conversions[speaker] = {"state": "fallback", "reason": reason}
        # Every conversion is done: free IndexTTS's MPS memory before synthesis
        # loads Irodori (C782A726 kept ~25 GiB of it through synthesis).
        await self._close_reference_converter()
        self.progress("reference_conversion", 1.0, "参照音声を用意しました")
        return {speaker: converted.get(speaker, path)
                for speaker, path in references.items()}, conversions

    @staticmethod
    def _cached_conversion(target: Path):
        """A cached take skips the gates; its version and fixed text identify it."""
        metadata = target.with_suffix(".json")
        if not target.is_file() or not metadata.is_file():
            return None
        try:
            saved = json.loads(metadata.read_text())
        except (OSError, ValueError):
            return None
        if (not isinstance(saved, dict) or saved.get("version") != REFERENCE_CONVERSION_VERSION
                or saved.get("text") != CONVERSION_TEXT):
            return None
        seconds = saved.get("seconds")
        if (not isinstance(seconds, int | float)
                or not REFERENCE_CONVERSION_MIN_SECONDS <= seconds <= REFERENCE_CONVERSION_MAX_SECONDS):
            return None
        return {name: saved[name] for name in ("seconds", "similarity", "ratio") if name in saved}

    async def _convert_reference(self, reference: Path, target: Path, work: Path) -> dict:
        """Convert one reference; raise unless every gate passes, so the original stays."""
        temporary = target.with_name(f".{target.stem}.tmp.wav")
        try:
            await self.reference_converter.convert(reference, temporary)
            seconds = _converted_seconds(temporary)
            if not REFERENCE_CONVERSION_MIN_SECONDS <= seconds <= REFERENCE_CONVERSION_MAX_SECONDS:
                raise FileDubbingError(
                    f"変換後の参照音声が{seconds:.1f}秒でした"
                    f"（{REFERENCE_CONVERSION_MIN_SECONDS:.0f}〜{REFERENCE_CONVERSION_MAX_SECONDS:.0f}秒"
                    "に収まりません）。")
            check = await check_take(self.transcriber, temporary, CONVERSION_TEXT)
            if not check["verified"]:
                raise FileDubbingError(
                    "変換後の参照音声が固定文どおりに聞き取れません"
                    f"（一致率{check['ratio']:.2f}、局所一致率{check['coverage']:.2f}）。")
            similarity = await self._reference_similarity(reference, temporary, work)
            if similarity < REFERENCE_SIMILARITY_MIN:
                raise FileDubbingError(f"変換後の参照音声が元の声と似ていません（類似度{similarity:.2f}）。")
            temporary.replace(target)
            _write_json(target.with_suffix(".json"), {
                "version": REFERENCE_CONVERSION_VERSION, "text": CONVERSION_TEXT,
                "seconds": round(seconds, 3), "similarity": round(similarity, 3),
                "ratio": round(check["ratio"], 3), "coverage": round(check["coverage"], 3)})
            logger.info("file dubbing reference converted seconds=%.2f similarity=%.3f ratio=%.3f",
                        seconds, similarity, check["ratio"])
            return {"seconds": round(seconds, 3), "similarity": round(similarity, 3),
                    "ratio": round(check["ratio"], 3)}
        finally:
            temporary.unlink(missing_ok=True)

    async def _reference_similarity(self, original: Path, converted: Path, work: Path) -> float:
        """ECAPA cosine of the two references, in one pass on one concatenated 16 kHz file."""
        with wave.open(str(original), "rb") as wav:
            original_seconds = wav.getnframes() / wav.getframerate()
        with wave.open(str(converted), "rb") as wav:
            converted_seconds = wav.getnframes() / wav.getframerate()
        directory = work / CONVERSION_CACHE_DIRECTORY
        directory.mkdir(exist_ok=True)
        combined = directory / "similarity-input.wav"
        try:
            await run_process(
                "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                "-i", str(original), "-i", str(converted), "-filter_complex",
                f"[0:a]aresample={ANALYSIS_RATE},aformat=sample_fmts=s16:channel_layouts=mono,"
                f"apad=pad_dur={REFERENCE_SIMILARITY_GAP_SECONDS}[a0];"
                f"[1:a]aresample={ANALYSIS_RATE},aformat=sample_fmts=s16:channel_layouts=mono[a1];"
                "[a0][a1]concat=n=2:v=0:a=1[out]",
                "-map", "[out]", "-c:a", "pcm_s16le", str(combined), timeout=300)
            worker = self.embeddings or EmbeddingWorker(
                self.settings, allow_download=self.options.allow_download,
                download_progress=self.download_progress)
            start = original_seconds + REFERENCE_SIMILARITY_GAP_SECONDS
            similarity, _ = await worker.similarity(
                combined, [(0.0, round(original_seconds, 3)),
                           (round(start, 3), round(start + converted_seconds, 3))], directory)
            return similarity(0, 1)
        finally:
            combined.unlink(missing_ok=True)

    async def _close_reference_converter(self):
        converter, self.reference_converter = self.reference_converter, None
        if converter is not None:
            await converter.close()

    @staticmethod
    def _segments(lines):
        segments, owners = [], []
        for speaker, sentence in speaker_segments(lines):
            segment = Segment(len(segments), sentence.start, sentence.end, sentence.text)
            segment.boundary_before = sentence.boundary_before
            segments.append(segment)
            owners.append(speaker)
        return segments, owners

    def _voices(self, segments, owners, references, reference_ids, analysis, speech_path, work):
        """The reference key of each segment: its speaker's, or else the segment's own voice.

        A one-word reaction (C782A726: "Hit.", 0.3 s) has no speaker label or too
        little clean audio for a speaker reference. Its own voice is the speaker's
        voice by definition, so it is never dubbed in another speaker's voice.
        """
        voices = []
        for segment, owner in zip(segments, owners, strict=True):
            if owner in references:
                voices.append(owner)
                continue
            voiced = analysis.voiced(segment.start, segment.end)
            seconds = sum(b - a for a, b in voiced)
            if seconds < OWN_SEGMENT_MIN_SECONDS:
                voices.append(None)
                segment.state = "skipped"
                if owner is None:
                    segment.skip_reason = "speaker_unknown"
                    segment.error = "話者を判定できず、区間の声も短すぎるため、元の音声のままにしました。"
                else:
                    segment.skip_reason = "speaker_reference_unavailable"
                    segment.error = f"{owner}の参照音声に使える区間がなく、区間の声も短すぎるため、元の音声のままにしました。"
                logger.info("file dubbing segment=%d kept original reason=%s own_voice=%.2f",
                            segment.index, segment.skip_reason, seconds)
                continue
            path, digest = cut_reference(speech_path, [{"voiced": voiced}], work / "speakers")
            key = f"segment-{segment.index}"
            references[key], reference_ids[key] = path, f"own-segment:{digest}"
            self._own_voices[segment.index] = round(seconds, 3)
            voices.append(key)
        return voices

    async def _render_dub(self, gemma, media, language, speakers, work, started,
                          segments, owners, voices, references, reference_ids, steps):
        options = self.options
        # Loading the file's workers can overlap translation. Wait until
        # a translated turn actually needs new audio: cached exports load no model.
        original_tts = self.tts
        self.tts = original_tts or CoreMLTTS(self.settings, allow_download=self.options.allow_download, download_progress=self.download_progress)
        preparation = None
        translated_turns = [asyncio.Event() for _ in segments]

        def translated(segment):
            nonlocal preparation
            if (segment.state == "pending" and preparation is None
                    and hasattr(self.tts, "preload")):
                stem = work / "clips" / self._synthesis_key(
                    segment, reference_ids[voices[segment.index]], steps)
                if self._cached_verification(stem, self._voice(segment), segment.text) is None:
                    preparation = self.tts.preload()
                    if hasattr(self.transcriber, "start"):
                        self.transcriber.start()
            translated_turns[segment.index].set()

        def adopt(source):
            # The early translation's segments share indices but were made before
            # the references were known (_references_and_translations).
            segment = segments[source.index]
            if segment.state == "pending" and not segment.text:
                segment.text = source.text
                if source.state != "pending":
                    segment.state, segment.skip_reason, segment.error = (
                        source.state, source.skip_reason, source.error)
            translated(segment)

        async def retime(segment, seconds):
            translator = (self.task_router.shortening_translator(title=self.title, mora_per_second=MORA_PER_SECOND)
                          if self.task_router else self._translator(gemma))
            if translator is None:
                return None
            index = segment.index
            result = await translator.translate_batch(
                language, [TranslationItem(index, segment.source, owners[index], seconds)],
                [(owners[i], segments[i].source) for i in range(max(0, index - CONTEXT_BEFORE), index)],
                [(owners[i], segments[i].source)
                 for i in range(index + 1, min(len(segments), index + 1 + CONTEXT_AFTER))])
            if index in result:
                # The slower-take retranslation, used only if it is shorter (_synthesize).
                self._translation_checks[index] = {**(self._translation_checks.get(index) or {}),
                                                   "retime": getattr(translator, "checks", {}).get(index)}
            return result.get(index)

        early, self._translation = self._translation, None
        tasks = [early.task] if early is not None else []
        try:
            # Translate in source order, while the existing bounded lookahead
            # generates already translated turns.
            if early is None:
                tasks.append(asyncio.create_task(self._translate(
                    gemma, segments, owners, language, work, on_translated=translated)))
            else:
                early.listen(adopt)
            tasks.append(asyncio.create_task(self._synthesize(
                segments, voices, references, steps, work, reference_ids=reference_ids,
                translated=translated_turns, retime=retime)))
            await asyncio.gather(*tasks)
            if preparation is not None:
                # A failed preparation still uses the normal per-turn retry and
                # error reporting, rather than losing every remaining turn.
                await asyncio.gather(preparation, return_exceptions=True)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            if preparation is not None:
                if not preparation.done():
                    preparation.cancel()
                await asyncio.gather(preparation, return_exceptions=True)
            try:
                if original_tts is None:
                    await self.tts.close()
            finally:
                self.tts = original_tts
        ready = [segment for segment in segments if segment.state == "ready"]
        if not ready:
            raise FileDubbingError("吹き替え音声を1つも生成できませんでした。レポートの各区間のエラーを確認してください。")
        dub_track, speech_gain = work / "dub-track.wav", work / "speech-gain.wav"
        self.progress("mix", 0.0, "音声を合成しています")
        place_segments(ready, work, dub_track, media["duration"])
        write_speech_gain(ready, speech_gain, media["duration"], options.original_volume)
        vocals, background = self._separated
        await mux(await self._picture, dub_track, options.output, options.dub_volume, media["channels"],
                  vocals=vocals, background=background, speech_gain=speech_gain)
        report = self._report(gemma, media, language, steps, speakers, segments, owners, started)
        _write_json(options.output.with_suffix(".dubbing.json"), report)
        logger.info("file dubbing finished segments=%d ready=%d skipped=%s elapsed=%.0fs", len(segments), len(ready),
                    dict(Counter(segment.skip_reason for segment in segments if segment.state == "skipped")),
                    report["summary"]["elapsed_seconds"])
        self.progress("done", 1.0, "完了しました")
        return report

    async def _speakers(self, gemma, cues, analysis, speech_path, work, *, language=None) -> dict:
        saved = work / "speakers.json"
        if saved.is_file():
            result = json.loads(saved.read_text())
            if (result.get("model") == SPEAKER_IDENTITY and result.get("version") == DIARIZATION_VERSION
                    and result.get("reference_transcriber") == SPEECH_CHECK_IDENTITY
                    and result.get("source_language") == language
                    and all(Path(item["path"]).is_file() for item in result["references"].values()
                            if item.get("path"))):
                self.progress("speakers", 1.0, "話者の判定結果を再利用しました")
                return result
        messages = {"speaker_embedding": "声の特徴を分析しています",
                    "reference_selection": "Whisperで参照音声の候補を検証しています"}
        diarizer = SpeakerDiarizer(
            self.embeddings or EmbeddingWorker(self.settings, allow_download=self.options.allow_download,
                                               download_progress=self.download_progress),
            transcriber=self.transcriber, language=language, speakers=self.options.speakers,
            progress=lambda stage, fraction: self.progress("speakers", fraction, messages.get(stage, "")))
        lines = await diarizer.run(cues, analysis, work)
        selected = await self._references_and_translations(gemma, diarizer, lines, analysis, language, work)
        references = {}
        for speaker, item in selected.items():
            path, digest = None, None
            if item["usable"]:
                path, digest = cut_reference(speech_path, item["clips"], work / "speakers")
            references[speaker] = {
                **item, "path": str(path) if path else None, "digest": digest,
                "error": None if path else "参照音声に使える本人の発話が短すぎます。"}
            logger.info("file dubbing speaker=%s reference_seconds=%.2f verified=%s clips=%d rejected=%d",
                        speaker, item["duration"], item["verified"], len(item["clips"]), len(item["rejected"]))
        result = {"model": SPEAKER_IDENTITY, "version": DIARIZATION_VERSION,
                  "reference_transcriber": SPEECH_CHECK_IDENTITY, "source_language": language,
                  "reference_transcriptions": diarizer.reference_transcriptions,
                  "speakers": list(diarizer.groups),
                  "lines": [line.public() for line in lines], "references": references}
        _write_json(saved, result)
        return result

    async def _references_and_translations(self, gemma, diarizer, lines, analysis, language, work):
        """Translation needs only the speaker turns, so it runs during reference checks.

        It is not awaited here: once the references are chosen, synthesis starts
        on the lines translated so far while the text engine translates the rest. run()
        reclaims the translation if the job stops before synthesis takes it over.
        """
        segments, owners = self._segments(lines)
        prepare_speech_limits(segments, self._duration)
        translation = _EarlyTranslation()
        translation.task = asyncio.create_task(self._translate(
            gemma, segments, owners, language, work, on_translated=translation))
        self._translation = translation
        references = asyncio.create_task(diarizer.references(lines, analysis))
        try:
            # A failed translation (engine unavailable) stops the job without
            # waiting for the remaining reference checks.
            done, _ = await asyncio.wait((references, translation.task),
                                         return_when=asyncio.FIRST_COMPLETED)
            if translation.task in done:
                translation.task.result()
            return await references
        finally:
            if not references.done():
                references.cancel()
            await asyncio.gather(references, return_exceptions=True)

    def _translator(self, gemma):
        if self.task_router and self.translator is None:
            from .task_routing import TaskDubbingTranslator
            self.translator = TaskDubbingTranslator(self.task_router, title=self.title, mora_per_second=MORA_PER_SECOND)
        return self.translator or GemmaDubbingTranslator(gemma, title=self.title,
                                                         mora_per_second=MORA_PER_SECOND)

    async def _translate(self, gemma, segments, owners, language, work, *, on_translated=None, show_progress=True):
        """Batches of consecutive lines with surrounding source context and time budgets.

        The context is source text only, so batches are independent: their
        results and cache keys do not depend on the order batches finish in.
        """
        pending = [segment for segment in segments if segment.state == "pending" and not segment.text]
        if on_translated:
            for segment in segments:
                if segment.state != "pending" or segment.text:
                    on_translated(segment)
        if not pending:
            return
        if language.lower().startswith("ja"):
            for segment in pending:
                segment.text = segment.source
                if on_translated:
                    on_translated(segment)
            return
        if show_progress:
            chain = self.task_router.config.translation_chain(language) if self.task_router else []
            model = self.task_router.describe(chain) if chain else "Gemma"
            self.progress("translation", 0.0, f"{model}で字幕を翻訳しています（0/{len(pending)}区間完了）")
        cache_path, checks_path = work / "translations.json", work / "translation-checks.json"
        translator = self._translator(gemma)
        cache = json.loads(cache_path.read_text()) if cache_path.is_file() else {}
        checks = json.loads(checks_path.read_text()) if checks_path.is_file() else {}
        done = 0

        def budget(segment):
            limit = segment.speech_limit if segment.speech_limit is not None else segment.end
            return max(MIN_SPEECH_SECONDS, limit - segment.start)

        heard_after = reactions_after(segments, self._events)

        def context(indices):
            return [(owners[i], with_events(segments[i].source, heard_after[i]))
                    for i in indices if 0 <= i < len(segments)]

        def key(segment):
            before = context(range(segment.index - CONTEXT_BEFORE, segment.index))
            after = context(range(segment.index + 1, segment.index + 1 + CONTEXT_AFTER))
            return hashlib.sha256(json.dumps([translator.identity, language, self.title, before,
                                              owners[segment.index], segment.source,
                                              heard_after[segment.index], after,
                                              round(budget(segment), 1)]).encode()).hexdigest()

        slots = asyncio.Semaphore(TRANSLATION_PARALLEL)

        async def run(batch):
            nonlocal done
            keys = {segment.index: key(segment) for segment in batch}
            missing = [segment for segment in batch if keys[segment.index] not in cache
                       and keys[segment.index] not in self._failed_translations]
            if missing:
                async with slots:
                    first, last = missing[0].index, missing[-1].index
                    result = await translator.translate_batch(
                        language,
                        [TranslationItem(segment.index, segment.source, owners[segment.index], budget(segment),
                                         heard_after[segment.index])
                         for segment in missing],
                        context(range(first - CONTEXT_BEFORE, first)),
                        context(range(last + 1, last + 1 + CONTEXT_AFTER)))
                failed = [segment.index for segment in missing if segment.index not in result]
                if failed:
                    logger.warning("file dubbing translation failed segments=%s", failed)
                async with self._translation_cache_lock:
                    for segment in missing:
                        if segment.index in result:
                            cache[keys[segment.index]] = result[segment.index]
                            checks[keys[segment.index]] = getattr(translator, "checks", {}).get(segment.index)
                        else:
                            self._failed_translations.add(keys[segment.index])
                            # Why the line kept its original voice, for the report; not cached.
                            self._translation_checks[segment.index] = getattr(translator, "checks", {}).get(segment.index)
                    _write_json(cache_path, cache)
                    _write_json(checks_path, checks)
            for segment in batch:
                text = cache.get(keys[segment.index])
                if text:
                    segment.text = text
                    self._translation_checks[segment.index] = checks.get(keys[segment.index])
                else:
                    segment.state, segment.skip_reason = "skipped", "translation_failed"
                    segment.error = "翻訳モデルから訳文を得られなかったため、元の音声のままにしました。"
                if on_translated:
                    on_translated(segment)
            done += len(batch)
            if show_progress:
                self.progress("translation", done / len(pending), f"字幕の翻訳を処理しました（{done}/{len(pending)}区間）")

        batches = [pending[i:i + BATCH_SIZE] for i in range(0, len(pending), BATCH_SIZE)]
        tasks = [asyncio.create_task(run(batch)) for batch in batches]
        try:
            await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        if show_progress:
            self.progress("translation", 1.0, "字幕を翻訳しました")

    @staticmethod
    def _speech_seconds(segment):
        natural = max(MIN_SPEECH_SECONDS, natural_seconds(segment.text, MORA_PER_SECOND))
        available = (segment.speech_limit if segment.speech_limit is not None else segment.end) - segment.start
        if natural <= available:
            return round(natural, 3)
        return round(max(natural / MODEL_SPEEDUP_MAX, available, MIN_SPEECH_SECONDS), 3)

    def _seeds(self, segment):
        return SHORT_LINE_SEEDS if self._speech_seconds(segment) <= SHORT_LINE_SECONDS else SEEDS

    def _voice(self, segment):
        """Irodori's style caption for a line.

        The default calm-narration caption contradicts a captioned laugh. With
        the default voice only, a laughing line asks for laughter instead; a
        voice chosen by the user is never replaced.
        """
        if self.settings.voice == Settings.voice and EMOJI["laughter"] in (segment.text or ""):
            return LAUGHING_VOICE
        return self.settings.voice

    def _synthesis_key(self, segment, reference_id, steps):
        # A converted reference sits under a constant directory; the source
        # digest keeps takes of the same voice together across runs.
        identity = [SYNTHESIS_VERSION, TTS_IDENTITY, segment.text,
                    reference_id, steps, self._speech_seconds(segment)]
        voice = self._voice(segment)
        if voice != Settings.voice:
            identity.append(["voice", voice])
        return hashlib.sha256(json.dumps(identity).encode()).hexdigest()[:32]

    def _cached_verification(self, stem, voice, expected):
        natural, meta = stem.with_suffix(".natural.wav"), stem.with_suffix(".json")
        if not natural.is_file() or not meta.is_file():
            return None
        try:
            verification = json.loads(meta.read_text())
        except (OSError, ValueError):
            return None
        if (isinstance(verification, dict)
                and verification.get("voice", Settings.voice) == voice):
            heard = verification["transcript"]
            if same_reading(expected, heard) or align_latin_spellings(expected, heard) != (expected, heard):
                # Older caches truncated the transcript. Recheck a full take
                # before changing its score, without discarding other audio.
                if len(heard) == 300:
                    return None
                ratio, coverage = compare(expected, heard)
                verification.update(ratio=ratio, coverage=coverage)
            return verification
        return None

    async def _synthesize(self, segments, voices, references, steps, work, *,
                          reference_ids=None, translated=None, retime=None):
        clips = work / "clips"
        clips.mkdir(exist_ok=True)
        owns = self.tts is None
        tts = self.tts or CoreMLTTS(self.settings, allow_download=self.options.allow_download, download_progress=self.download_progress)
        capacity = getattr(tts, "capacity", 1)
        self._synthesis_lock = asyncio.Semaphore(capacity)
        if self.mobile:
            self.mobile.tts.local_workers = capacity
        pending = [segment for segment in segments if segment.state == "pending"]
        previous_end_frame = 0
        tasks = {}
        prefetch = []

        def schedule(segment):
            reference = references[voices[segment.index]]
            source = (reference_ids[voices[segment.index]] if reference_ids is not None
                      else reference.parent.name)
            key = self._synthesis_key(segment, source, steps)
            if key not in tasks:
                async def prepare():
                    stem = clips / key
                    verification = self._cached_verification(stem, self._voice(segment), segment.text)
                    if (getattr(tts, "is_coreml", False) and verification is not None
                            and verification["ratio"] < SPEECH_ACCEPT_RATIO
                            and len(verification["attempts"]) == 1
                            and "seed" in verification["attempts"][0]):
                        stem = clips / f"{key}-retry"
                        verification = self._cached_verification(stem, self._voice(segment), segment.text)
                    if (verification is not None and not speech_verified(verification["ratio"], verification["coverage"])
                            and len(verification["attempts"]) < len(self._seeds(segment))):
                        verification = None  # Failed with fewer takes than this line now gets.
                    natural, meta = stem.with_suffix(".natural.wav"), stem.with_suffix(".json")
                    if verification is None:
                        verification = await self._generate_verified(
                            tts, segment, reference, steps, stem)
                        verification["voice"] = self._voice(segment)
                        _write_json(meta, verification)
                    return natural, verification
                tasks[key] = asyncio.create_task(prepare())
            return key, tasks[key]

        async def prefetch_segment(segment):
            if translated is not None:
                await translated[segment.index].wait()
            if segment.state == "pending":
                schedule(segment)

        try:
            for number, segment in enumerate(pending):
                for waiter in prefetch:
                    waiter.cancel()
                await asyncio.gather(*prefetch, return_exceptions=True)
                executors = capacity + (len(self.mobile.devices) if self.mobile else 0)
                lookahead = max(SYNTHESIS_LOOKAHEAD, LOOKAHEAD_PER_EXECUTOR * executors)
                prefetch = [asyncio.create_task(prefetch_segment(upcoming))
                            for upcoming in pending[number + 1:number + lookahead]]
                waiting = translated is not None and not translated[segment.index].is_set()
                self.progress("synthesis", number / max(1, len(pending)),
                              f"字幕の翻訳結果を待っています（音声処理 {number}/{len(pending)}区間完了）" if waiting else
                              f"吹き替え音声を生成・検証しています（{number + 1}/{len(pending)}）")
                if translated is not None:
                    await translated[segment.index].wait()
                if segment.state != "pending":
                    continue
                self.progress("synthesis", number / max(1, len(pending)),
                              f"吹き替え音声を生成・検証しています（{number + 1}/{len(pending)}）")
                key, task = schedule(segment)
                try:
                    natural, verification = await task
                except (asyncio.CancelledError, WhisperUnavailable):
                    # Without the transcript model no take can be verified: stop the job.
                    raise
                except Exception as exc:
                    logger.exception("file dubbing synthesis failed segment=%d", segment.index)
                    tasks.pop(key, None)
                    segment.state, segment.skip_reason, segment.error = "skipped", "synthesis_failed", str(exc)
                    continue
                # Text-based duration estimates miss slow takes and delayed starts.
                # Retry the script once against the measured voice, before fitting.
                start, limit = speech_window(segments, segment)
                start = max(start, previous_end_frame / SAMPLE_RATE)
                trim_edge_silence(natural)
                with wave.open(str(natural), "rb") as wav:
                    measured = wav.getnframes() / wav.getframerate()
                available = max(0.05, limit - start)
                slow_take = {}
                if retime is not None and measured > available * 1.25:
                    # The voice's measured pace, read up to SHORTEN_TEMPO faster.
                    budget = available * SHORTEN_TEMPO * min(1.0, natural_seconds(segment.text, MORA_PER_SECOND) / measured)
                    replacement = await retime(segment, budget)
                    shorter = bool(replacement) and natural_seconds(
                        replacement, MORA_PER_SECOND) < natural_seconds(segment.text, MORA_PER_SECOND)
                    slow_take = {"slow_take": {"measured_seconds": round(measured, 3),
                                               "available_seconds": round(available, 3),
                                               "retranslation": "used" if shorter else
                                               "not_shorter" if replacement else "no_answer"}}
                    logger.info("file dubbing segment=%d slow take measured=%.2f available=%.2f retranslation=%s",
                                segment.index, measured, available, slow_take["slow_take"]["retranslation"])
                    if shorter:
                        segment.text = replacement
                        key, task = schedule(segment)
                        natural, verification = await task
                        trim_edge_silence(natural)
                segment.audio = {"verification": verification["attempts"], **slow_take}
                if not speech_verified(verification["ratio"], verification["coverage"]):
                    # A garbled dub is worse than the original voice; keep the original here.
                    segment.state, segment.skip_reason = "skipped", "speech_verification_failed"
                    logger.info("file dubbing segment=%d kept original reason=speech_verification_failed "
                                "ratio=%.2f coverage=%.2f", segment.index, verification["ratio"], verification["coverage"])
                    segment.error = ("生成音声の書き起こしに不一致または局所的な崩れがあるため、"
                                     f"元の音声のままにしました（一致率{verification['ratio']:.2f}、"
                                     f"局所一致率{verification['coverage']:.2f}）。")
                    continue
                with wave.open(str(natural), "rb") as wav:
                    generated_seconds = wav.getnframes() / wav.getframerate()
                # Place each line within its bounded window on the source clock.
                start, limit = speech_window(segments, segment)
                # Skipped turns must not forget the last emitted clip. Align both
                # boundaries to the PCM clock: round(end - start) can otherwise
                # differ by one sample from round(end) - round(start).
                start_frame = max(previous_end_frame, round(start * SAMPLE_RATE))
                start = start_frame / SAMPLE_RATE
                end_frame = max(start_frame + 1, round(
                    speech_end(segment, start, limit, generated_seconds, preferred_tempo=1.1) * SAMPLE_RATE))
                needed_frames = math.ceil(generated_seconds / 1.25 * SAMPLE_RATE)
                if end_frame - start_frame < needed_frames:
                    start_frame = max(previous_end_frame, end_frame - needed_frames,
                                      round(earliest_start(segments, segment, start) * SAMPLE_RATE))
                    start = start_frame / SAMPLE_RATE
                if generated_seconds > (end_frame - start_frame) / SAMPLE_RATE * 1.25 + 1 / SAMPLE_RATE:
                    segment.audio.update(generated_seconds=round(generated_seconds, 3),
                                         allowed_seconds=round((end_frame - start_frame) / SAMPLE_RATE, 3))
                    segment.state, segment.skip_reason = "skipped", "speech_too_fast"
                    logger.info("file dubbing segment=%d kept original reason=speech_too_fast generated=%.2f allowed=%.2f",
                                segment.index, generated_seconds, (end_frame - start_frame) / SAMPLE_RATE)
                    segment.error = "訳文の音声が区間内に収まらず早口になるため、元の音声のままにしました。"
                    continue
                segment.playback_start = start
                segment.playback_end = end_frame / SAMPLE_RATE
                target = clips / f"segment-{segment.index}.wav"
                fitted = await self._fit_clip(natural, target, end_frame - start_frame)
                previous_end_frame = end_frame
                segment.audio.update(fitted)
                segment.audio.update({"requested_seconds": verification["seconds"],
                                      "transcript": verification["transcript"],
                                      "transcript_ratio": verification["ratio"],
                                      "transcript_coverage": verification["coverage"],
                                      "verified": speech_verified(verification["ratio"], verification["coverage"]),
                                      **verification["tail"]})
                segment.state = "ready"
        finally:
            for waiter in prefetch:
                waiter.cancel()
            await asyncio.gather(*prefetch, return_exceptions=True)
            for task in tasks.values():
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks.values(), return_exceptions=True)
            if owns:
                await tts.close()
        self.progress("synthesis", 1.0, "吹き替え音声を生成しました")

    async def _fit_clip(self, natural: Path, target: Path, frames: int) -> dict:
        """Reuse an exact fitted waveform, never a stale file with equal length."""
        identity = ["fit-audio-v1", await asyncio.to_thread(_digest_file, natural), frames, SAMPLE_RATE]
        metadata = target.with_suffix(".fit.json")
        try:
            saved = json.loads(metadata.read_text()) if metadata.is_file() else {}
            if (isinstance(saved, dict) and saved.get("identity") == identity
                    and isinstance(saved.get("audio"), dict)
                    and saved["audio"].get("duration") == frames / SAMPLE_RATE
                    and target.is_file()
                    and saved.get("sha256") == await asyncio.to_thread(_digest_file, target)):
                return saved["audio"]
        except (OSError, ValueError):
            pass
        fitted = await fit_audio(natural, target, frames / SAMPLE_RATE)
        _write_json(metadata, {"identity": identity, "audio": fitted,
                               "sha256": await asyncio.to_thread(_digest_file, target)})
        return fitted

    async def _generate_verified(self, tts, segment, reference, steps, stem: Path) -> dict:
        """Generate and verify up to three takes (six for a short line) before fitting the best one."""
        seconds = self._speech_seconds(segment)
        voice = self._voice(segment)
        style = {} if voice == self.settings.voice else {"voice": voice}
        attempts, best = [], None
        for seed in self._seeds(segment):
            raw = stem.with_name(f"{stem.name}.seed{seed}.wav")
            async def local(raw=raw, seed=seed):
                async with self._synthesis_lock:
                    await tts.synthesize(segment.text, segment.end - segment.start, raw,
                                         reference=reference, steps=steps, seconds=seconds, seed=seed, **style)

            remote = None
            if self.mobile:
                # Earlier lines first, so a retake of the line being placed
                # does not wait behind the lookahead.
                remote = await self.mobile.tts.run({
                    "text": segment.text, "voice": voice, "steps": steps, "seconds": seconds,
                    "reference_path": str(reference),
                }, local, priority=segment.index)
            else:
                await local()
            if remote is not None:
                raw.write_bytes(remote)
            tail = trim_generated_tail(raw)
            check = await check_take(self.transcriber, raw, segment.text)
            heard, ratio, coverage = check["heard"], check["ratio"], check["coverage"]
            attempts.append({"executor": "mobile-coreml" if remote is not None else "mac",
                             **({"attempt": len(attempts) + 1} if getattr(tts, "is_coreml", False) else {"seed": seed}),
                             "transcript": heard[:300], "ratio": round(ratio, 3), "coverage": round(coverage, 3),
                             **({} if heard else {"transcript_error": "no_complete_transcript"})})
            # A take without a garbled phrase wins over a higher whole-line ratio.
            score = (speech_verified(ratio, coverage), coverage >= SPEECH_COVERAGE, ratio)
            if best is None or score > best[0]:
                if best is not None:
                    best[1].unlink(missing_ok=True)
                best = (score, raw, heard, tail, ratio, coverage)
            else:
                raw.unlink(missing_ok=True)
            if speech_verified(ratio, coverage):
                break
        _, raw, heard, tail, ratio, coverage = best
        raw.replace(stem.with_name(f"{stem.name}.natural.wav"))
        if len(attempts) > 1:
            logger.info("file dubbing segment=%d speech attempts=%s", segment.index,
                        [(a.get("attempt", a.get("seed")), a["ratio"]) for a in attempts])
        # Preserve gate precision when the result is read back from the cache.
        return {"seconds": round(seconds, 3), "ratio": ratio, "coverage": coverage,
                "transcript": heard, "attempts": attempts, "tail": tail}

    def _report(self, gemma, media, language, steps, speakers, segments, owners, started):
        heard_after = reactions_after(segments, self._events)
        voiced = [segment for segment in segments if reaction_emojis(segment.source)]
        return {
            "version": PIPELINE_VERSION, "video": str(self.options.video),
            "subtitles": str(self.options.subtitles), "output": str(self.options.output),
            "subtitle_source": self._subtitle_source,
            "duration": media["duration"], "source_language": language,
            "volumes": {"original_speech_under_dub": self.options.original_volume,
                        "background": 1.0, "dub": self.options.dub_volume},
            "steps": steps, "synthesis_model": TTS_IDENTITY, "translator": self._translator(gemma).identity,
            "translation_attribution": ("Built with PLaMo; PLaMoによる翻訳を含む場合があります"
                                        if self.task_router and "plamo" in self.task_router.config.translation_chain(language)
                                        else None),
            "title": self.title, "uploader": self.options.uploader,
            "thumbnail_url": self.options.thumbnail_url,
            "separation_model": f"{SEPARATION_IDENTITY} (speech / background)",
            "speaker_model": f"{SPEAKER_IDENTITY} (who speaks when; ECAPA scores reference clips)",
            "speech_check_model": f"{SPEECH_CHECK_IDENTITY} (reference and dubbed speech transcript check)",
            "sound_events": "captions",
            "reference_transcriptions": speakers.get("reference_transcriptions"),
            "speakers": {name: {k: v for k, v in item.items() if k != "path"} | {"reference": item.get("path")}
                         | ({"conversion": self._reference_conversions[name]}
                            if name in self._reference_conversions else {})
                         for name, item in speakers["references"].items()},
            "lines": speakers["lines"],
            "segments": [{**segment.public(), "speaker": owners[segment.index],
                          **({"own_voice_reference_seconds": self._own_voices[segment.index]}
                             if segment.index in self._own_voices else {}),
                          "heard_after": list(heard_after[segment.index]),
                          "translation_check": self._translation_checks.get(segment.index),
                          "source_reactions": reaction_emojis(segment.source),
                          "dub_reactions": reaction_emojis(segment.text or ""),
                          "style_caption": self._voice(segment)} for segment in segments],
            "summary": {"segments": len(segments),
                        "ready": sum(segment.state == "ready" for segment in segments),
                        "skipped": sum(segment.state == "skipped" for segment in segments),
                        # Captioned laughs, sighs... of the speaker, and how many the dub voices.
                        "source_reactions": sum(len(reaction_emojis(s.source)) for s in voiced),
                        "dub_reactions": sum(len(reaction_emojis(s.text)) for s in voiced
                                             if s.state == "ready" and s.text),
                        "elapsed_seconds": round(time.monotonic() - started, 1)},
        }
