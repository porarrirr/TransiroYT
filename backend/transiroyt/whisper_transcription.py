"""Transcribe reference and generated speech with a persistent MLX Whisper worker.

The worker also identifies languages and reviews or creates timed subtitles."""
from __future__ import annotations

import asyncio
import base64
import json
import os
from pathlib import Path

from .logging_utils import log_model_start
from .process import run_process
from .tts import tts_python

MODEL_REPO = "mlx-community/whisper-large-v3-turbo"
MODEL_REVISION = "a4aaeec0636e6fef84abdcbe3544cb2bf7e9f6fb"
MODEL_FILES = ("config.json", "weights.safetensors")
MODEL_IDENTITY = f"{MODEL_REPO}@{MODEL_REVISION}"
SAMPLE_RATE = 16000
# The decoder has 448 positions; the prompt takes 4 of them.
MAX_TOKENS = 444
# A timed reply holds every word of a programme on one JSON line (about 1 MB for an hour);
# asyncio's default 64 KiB line limit rejects it.
REPLY_LINE_LIMIT = 256 * 1024 * 1024


def _whisper_language(language: str) -> str:
    return language.lower().replace("_", "-").split("-")[0]


class WhisperUnavailable(RuntimeError):
    pass


class TranscriptOutputLimit(RuntimeError):
    """The transcript did not finish, e.g. looping on unintelligible audio."""


class WhisperTranscriber:
    def __init__(self, settings, *, allow_download=False, download_progress=None):
        self.settings, self.allow_download = settings, allow_download
        self.download_progress = download_progress
        self.process = None
        self.log = None
        self.lock = asyncio.Lock()
        self.preparation = None

    def start(self):
        """Load the model once; concurrent callers share the same loading task."""
        if self.preparation is None:
            self.preparation = asyncio.create_task(self._start())
        return self.preparation

    async def transcribe(self, wav: Path | bytes, expected: str, *, language="ja") -> str:
        source, stdin = ("pipe:0", wav) if isinstance(wav, bytes) else (str(wav), None)
        pcm = await run_process("ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-i", source,
                                "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "s16le", "-", input=stdin, timeout=60)
        # Room for the expected words and some slack; a looping transcript stops early.
        reply = await self._request({"pcm": base64.b64encode(pcm).decode(),
                                     "max_tokens": min(MAX_TOKENS, 32 + 2 * len(expected)),
                                     "language": _whisper_language(language)}, 120)
        if not reply["complete"]:
            raise TranscriptOutputLimit(reply["text"][:200])
        return reply["text"]

    async def transcribe_timed(self, wav: Path, language: str | None, *, timeout: float) -> dict:
        """Segments with word times of a 16 kHz mono WAV, and the language (detected when None)."""
        reply = await self._request({"audio": str(wav), "timestamps": True,
                                     "language": language and _whisper_language(language)}, timeout)
        return {"language": reply["language"], "segments": reply["segments"]}

    async def detect_language(self, wav: Path, windows: list[tuple[float, float]]) -> tuple[str, dict]:
        """Spoken language of a 16 kHz mono WAV from up to 30 s windows, with the top probabilities."""
        reply = await self._request({"audio": str(wav), "detect": windows}, 120)
        return reply["language"], reply["probabilities"]

    async def transcribe_windows(self, wav: Path, windows: list[tuple[float, float]], language: str
                                 ) -> list[tuple[str, bool]]:
        """Text of each window (at most 30 s) of a 16 kHz mono WAV, and whether it ended normally."""
        reply = await self._request({"audio": str(wav), "windows": windows,
                                     "language": _whisper_language(language)}, 60 + 10 * len(windows))
        return list(zip(reply["texts"], reply["complete"], strict=True))

    async def pauses(self, wav: Path, minimum: float) -> list[list[float]]:
        """[start, end] of silences at least ``minimum`` seconds long in separated speech."""
        return (await self._request({"audio": str(wav), "pauses": minimum}, 120))["pauses"]

    async def _request(self, request: dict, timeout: float) -> dict:
        async with self.lock:
            # A worker stopped after a failed take is started again here.
            await asyncio.shield(self.start())
            try:
                self.process.stdin.write((json.dumps(request) + "\n").encode())
                await self.process.stdin.drain()
                async with asyncio.timeout(timeout):
                    raw = await self.process.stdout.readline()
                if not raw:
                    raise RuntimeError("音声確認プロセスが終了しました。whisper.logを確認してください。")
                reply = json.loads(raw)
            except BaseException:
                # The next take starts a new worker instead of reusing a broken one.
                await self._stop()
                raise
        if not reply.get("ok"):
            raise RuntimeError(f"音声の書き起こしに失敗しました: {reply.get('error')}")
        return reply

    async def _start(self):
        directory = await self._model_directory()
        self.log = (self.settings.data / "whisper.log").open("a")
        log_model_start(self.log, MODEL_IDENTITY)
        env = {**os.environ, "TRANSIROYT_WHISPER_MODEL": str(directory),
               **({"PYTHONPATH": os.environ["TRANSIROYT_TTS_SITE_PACKAGES"]}
                  if "TRANSIROYT_TTS_SITE_PACKAGES" in os.environ else {})}
        self.process = await asyncio.create_subprocess_exec(
            str(tts_python(self.settings)), "-B", "-u", str(self.settings.root / "scripts/whisper_worker.py"),
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=self.log, env=env,
            limit=REPLY_LINE_LIMIT)
        try:
            async with asyncio.timeout(300):
                raw = await self.process.stdout.readline()
            ready = json.loads(raw) if raw else {"error": "process exited"}
            if not ready.get("ok"):
                raise RuntimeError(f"音声確認モデル（Whisper）を読み込めませんでした: {ready.get('error')}")
        except BaseException:
            await self._stop()
            raise

    async def _model_directory(self) -> Path:
        if self.allow_download and self.download_progress:
            from .model_downloads import download_model
            await download_model(self.settings, "speech_check_model", self.download_progress)
        return await asyncio.to_thread(self._resolve, self.allow_download and not self.download_progress)

    def _resolve(self, download: bool) -> Path:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import LocalEntryNotFoundError

        try:
            paths = [hf_hub_download(MODEL_REPO, name, revision=MODEL_REVISION,
                                     cache_dir=self.settings.data / "models/huggingface",
                                     local_files_only=not download) for name in MODEL_FILES]
        except LocalEntryNotFoundError as exc:
            raise WhisperUnavailable(
                "音声確認モデル（Whisper large-v3-turbo、約1.6GB）が未取得です。"
                "ダウンロードを許可して再実行してください。") from exc
        return Path(paths[0]).parent

    async def _stop(self):
        if self.process and self.process.returncode is None:
            self.process.kill()
            await self.process.wait()
        self.process, self.preparation = None, None
        if self.log:
            self.log.close()
            self.log = None

    async def close(self):
        if self.preparation is not None and not self.preparation.done():
            self.preparation.cancel()
            await asyncio.gather(self.preparation, return_exceptions=True)
        async with self.lock:
            await self._stop()
