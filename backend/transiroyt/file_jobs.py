"""One background file-dubbing job at a time, for the desktop app."""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import replace
from pathlib import Path

from .coreml_tts import coreml_binary
from .dubbing_options import FileDubbingOptions
from .errors import FileDubbingError
from .file_dubbing import FileDubber
from .gemma_audio import MODEL_FILES, MODEL_REPO, MODEL_REVISION, GemmaUnavailable
from .installation import check_consumer_runtime
from .library import DubbingLibrary
from .reference_conversion import model_ready as reference_conversion_ready
from .speaker_diarization import ACTIVITY_MODEL_FILES, ACTIVITY_MODEL_REPO, ACTIVITY_MODEL_REVISION
from .subtitle_files import SubtitleError
from .vocal_separation import SeparationUnavailable
from .vocal_separation import model_ready as separation_ready
from .whisper_transcription import MODEL_FILES as WHISPER_FILES
from .whisper_transcription import MODEL_REPO as WHISPER_REPO
from .whisper_transcription import MODEL_REVISION as WHISPER_REVISION
from .whisper_transcription import WhisperUnavailable
from .youtube_file import download_youtube, youtube_video_id

logger = logging.getLogger(__name__)

SPEAKER_MODEL_REPO = "speechbrain/spkrec-ecapa-voxceleb"
SPEAKER_MODEL_REVISION = "0f99f2d0ebe89ac095bcc5903c4dd8f72b367286"


def model_status(settings) -> dict:
    """What a file job needs locally; never downloads."""
    import shutil

    from huggingface_hub import try_to_load_from_cache

    cache = settings.data / "models/huggingface"

    def cached(repo, revision, filename):
        return isinstance(try_to_load_from_cache(repo, filename, cache_dir=cache, revision=revision), str)

    from .gemma_audio import llama_server_binary
    from .plamo_translation import PLAMO
    try:
        server = llama_server_binary()
    except GemmaUnavailable:
        server = None
    return {
        "coreml_tool": coreml_binary(settings).is_file(),
        "coreml": (settings.data / "models/irodori-coreml-light-int8-v0.2.0/manifest.json").is_file(),
        "llama_server": bool(server and shutil.which(server)),
        "gemma": all(cached(MODEL_REPO, MODEL_REVISION, name) for name in MODEL_FILES),
        "plamo": cached(PLAMO.repo, PLAMO.revision, PLAMO.file),
        "speaker_model": cached(SPEAKER_MODEL_REPO, SPEAKER_MODEL_REVISION, "embedding_model.ckpt"),
        "diarization_model": all(cached(ACTIVITY_MODEL_REPO, ACTIVITY_MODEL_REVISION, name)
                                 for name in ACTIVITY_MODEL_FILES),
        "separation_model": separation_ready(settings),
        "speech_check_model": all(cached(WHISPER_REPO, WHISPER_REVISION, name) for name in WHISPER_FILES),
        "indextts_model": reference_conversion_ready(settings),
    }


class FileDubbingJobs:
    def __init__(self, settings, dubber_factory=None, *, mobile=None):
        self.settings = settings
        self.dubber_factory = dubber_factory or (lambda options, progress: FileDubber(
            settings, options, progress, download_progress=self._download_progress, mobile=mobile))
        self.job: dict | None = None
        self.task: asyncio.Task | None = None
        self.clearing_cache = False

    @property
    def running(self):
        return self.task is not None and not self.task.done()

    def start(self, options: FileDubbingOptions, *, youtube_url: str | None = None, youtube_quality: str = "best",
              embedded_subtitles: bool = False) -> dict:
        if self.running:
            raise FileDubbingError("別のファイル吹き替えを実行中です。完了または中止してから開始してください。")
        if self.clearing_cache:
            raise FileDubbingError("キャッシュの削除が終わってから開始してください。")
        try:
            check_consumer_runtime(self.settings)
        except RuntimeError as exc:
            raise FileDubbingError(str(exc)) from exc
        from .task_routing import load_routes
        routes = load_routes(self.settings.data)
        if 'sarashina' in routes.chain('translation'):
            from .sarashina_translation import load_settings
            if not load_settings(self.settings.data).public()['has_key']:
                raise FileDubbingError("設定でSarashinaのAPIキーを入力してください。")
        language = options.source_language
        if language != "auto" and not language.lower().startswith("ja") and not routes.translation_chain(language):
            raise FileDubbingError(f"原文の言語（{language}）に使える翻訳先がありません。"
                                   "PLaMo・Sarashinaは英語の原文だけに使います。設定で他の使用先を追加してください。")
        job_id = uuid.uuid4().hex
        if youtube_url:
            youtube_video_id(youtube_url)
            directory = self.settings.data / "file-dubbing-downloads" / job_id
            options = replace(options, video=directory / "source.mp4", subtitles=directory / "source.srt")
            options.validate(require_inputs=False)
            dubber = None
        elif embedded_subtitles:
            directory = self.settings.data / "file-dubbing-downloads" / job_id
            options = replace(options, subtitles=directory / "embedded.srt")
            options.validate(require_inputs=False)
            if not options.video.is_file():
                raise FileDubbingError(f"動画ファイルが見つかりません: {options.video}")
            dubber = None
        else:
            dubber = self.dubber_factory(options, self._progress)  # Validates the request first.
        self.job = {"id": job_id, "state": "running", "stage": "prepare", "fraction": 0.0,
                    "message": "開始しています", "video": str(options.video), "subtitles": str(options.subtitles),
                    "output": str(options.output), "started_at": time.time(), "error": None, "summary": None,
                    "speakers": None, "downloads": {}, "activities": {}}
        self.task = asyncio.create_task(self._run(dubber, self.job, options, youtube_url, youtube_quality, embedded_subtitles))
        return self.public()

    def _progress(self, stage, fraction, message):
        if self.job and self.job["state"] == "running":
            self.job["activities"][stage] = {"fraction": round(fraction, 4), "message": message,
                                             "updated_at": time.time()}
            # Reference checks and translation can finish during synthesis.
            # Update their activity rows without moving the main stage backwards.
            stages = {"download": 0, "prepare": 0, "speakers": 1, "reference_conversion": 1, "translation": 2,
                      "synthesis": 3, "mix": 4, "done": 5}
            if stages[stage] < stages[self.job["stage"]]:
                return
            if stage != self.job["stage"]:
                logger.info("File dubbing stage=%s: %s", stage, message)
            self.job.update({"stage": stage, "fraction": round(fraction, 4)})
            if message:
                self.job["message"] = message

    def _translation_note(self, message):
        """Tell the translation row why it waits: a local model is being fetched."""
        if not self.job or self.job["state"] != "running":
            return
        activity = self.job["activities"].get("translation")
        if activity and activity["fraction"] < 1:
            activity.update(message=message, updated_at=time.time())
            if self.job["stage"] == "translation":
                self.job["message"] = message

    def _download_progress(self, update):
        if not self.job or self.job["state"] != "running":
            return
        resource = update["resource"]
        self.job["downloads"][resource] = update
        if resource in ("gemma", "plamo") and update.get("total"):
            model = "PLaMo 2B" if resource == "plamo" else "Gemma"
            self._translation_note(f"{model}のモデルを取得しています（{update['downloaded'] / update['total']:.0%}）")
        # Model preparation runs alongside analysis and translation. Its events
        # belong in the resource rows and must not replace the active pipeline.
        if resource != "youtube":
            return
        total = update.get("total")
        fraction = min(1.0, update["downloaded"] / total) if total else 0.0
        message = "YouTubeの動画・字幕をダウンロードしています"
        if update["state"] == "checking":
            message = "YouTubeの動画・字幕の取得情報を確認しています"
        elif update["state"] == "processing":
            message = "YouTubeの動画と音声を結合しています"
        elif update["state"] == "complete":
            message = "YouTubeの動画・字幕を取得しました"
        self._progress("download", fraction, message)

    async def _run(self, dubber, job, options, youtube_url=None, youtube_quality="best", embedded_subtitles=False):
        try:
            if embedded_subtitles:
                from .embedded_subtitles import extract_subtitles
                self._progress("prepare", 0.0, "動画内の字幕を取り出しています")
                if not await extract_subtitles(options.video, options.subtitles):
                    options = replace(options, subtitles=options.subtitles.with_name("transcribed.srt"), transcribe=True)
                dubber = self.dubber_factory(options, self._progress)
            if youtube_url:
                self._progress("download", 0.0, "YouTubeの動画と字幕をダウンロードしています")
                metadata = await download_youtube(self.settings, youtube_url, options.video.parent,
                                                  options.source_language, quality=youtube_quality, progress=self._download_progress)
                if not metadata["subtitles"]:
                    options = replace(options, transcribe=True)
                elif options.source_language == "auto":
                    options = replace(options, source_language=metadata["language"])
                if metadata.get("title"):
                    options = replace(options, title=metadata["title"])
                options = replace(options, uploader=metadata.get("uploader"),
                                  thumbnail_url=metadata.get("thumbnail_url"))
                dubber = self.dubber_factory(options, self._progress)
            job.update(subtitles=str(options.subtitles), transcribed=options.transcribe)
            report = await dubber.run()
        except asyncio.CancelledError:
            job.update({"state": "cancelled", "message": "中止しました"})
            raise
        except (FileDubbingError, SubtitleError, GemmaUnavailable, SeparationUnavailable,
                WhisperUnavailable) as exc:
            logger.warning("File dubbing failed stage=%s: %s", job["stage"], exc)
            job.update({"state": "error", "error": str(exc)})
        except Exception as exc:
            logger.exception("File dubbing failed")
            job.update({"state": "error", "error": f"ファイル吹き替えに失敗しました: {exc}"})
        else:
            try:
                DubbingLibrary(self.settings.data).remember(Path(job["output"]).parent)
            except (OSError, ValueError):
                logger.exception("Could not register completed export in library")
            job.update({"state": "done", "fraction": 1.0, "message": "完了しました",
                        "summary": report["summary"],
                        "report": str(Path(job["output"]).with_suffix(".dubbing.json")),
                        "speakers": {name: {"reference_seconds": item["duration"],
                                            "reference": bool(item.get("reference")),
                                            "error": item.get("error")}
                                     for name, item in report["speakers"].items()}})
        finally:
            job["finished_at"] = time.time()
            await self._prune_cache()

    async def maintain_cache(self):
        if not self.running and not self.clearing_cache:
            await self._prune_cache()

    async def _prune_cache(self):
        self.clearing_cache = True
        try:
            await asyncio.to_thread(DubbingLibrary(self.settings.data).prune_cache)
        except (OSError, ValueError):
            logger.exception("Could not automatically clean working cache")
        finally:
            self.clearing_cache = False

    def public(self):
        if not self.job:
            return None
        result = dict(self.job)
        if self.running and self.clearing_cache:
            result.update(state="running", message="キャッシュを整理しています")
        return result

    async def cancel(self, job_id):
        if not self.job or self.job["id"] != job_id:
            raise KeyError(job_id)
        if self.running:
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
            if self.job["state"] == "running":
                self.job.update({"state": "cancelled", "message": "中止しました", "finished_at": time.time()})
        return self.public()

    async def shutdown(self):
        if self.running:
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
