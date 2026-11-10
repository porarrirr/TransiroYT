"""Speech / background separation for file dubbing (Kim Mel-Band RoFormer, pinned weights).

Mixing the dub over the whole original soundtrack kept the English voice under
every Japanese line and lowered music and effects with it. Separating speech
lets the mix keep the background at its original level, replace only the
speech that is dubbed, and gives the speaker analysis and voice references
speech without background music (the TTS reproduces a reference's noise).
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
from pathlib import Path

from .gemma_audio import model_cache
from .process import run_process
from .tts import tts_python

MODEL_URL = "https://huggingface.co/KimberleyJSN/melbandroformer/resolve/ac9b0614ab3cd7f77219e18ba494dfd93956c348/MelBandRoformer.ckpt"
MODEL_FILE = "MelBandRoformer-kim-ac9b0614.ckpt"
MODEL_SHA256 = "87201f4d31afb5bc79993230fc49446918425574db48c01c405e44f365c7559e"
MODEL_SIZE = 913_106_900
MODEL_IDENTITY = f"kim-melband-roformer:{MODEL_SHA256[:16]}"
SEPARATION_VERSION = "kim-mlx-fp16-7.7s-overlap0.25-v1"
INPUT_RATE = 44100


class SeparationUnavailable(RuntimeError):
    pass


def separation_model_path(cache: Path) -> Path:
    """Beside the Hugging Face cache; ``cache`` is ``model_cache(settings)``."""
    return Path(cache).parent / "separation" / MODEL_FILE


def model_ready(settings) -> bool:
    path = separation_model_path(model_cache(settings))
    return path.is_file() and path.stat().st_size == MODEL_SIZE


def fetch_model(cache: Path, report) -> Path:
    """Download the pinned weights; the SHA-256 must match before the file is used."""
    import httpx

    target = separation_model_path(cache)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and target.stat().st_size == MODEL_SIZE:
        report("complete", MODEL_SIZE, MODEL_SIZE)
        return target
    temporary = target.with_name(f".{MODEL_FILE}.{os.getpid()}.part")
    digest = hashlib.sha256()
    downloaded = 0
    try:
        with httpx.stream("GET", MODEL_URL, follow_redirects=True, timeout=60, trust_env=False) as response:
            response.raise_for_status()
            with temporary.open("wb") as handle:
                for block in response.iter_bytes(1 << 20):
                    handle.write(block)
                    digest.update(block)
                    downloaded += len(block)
                    report("downloading", downloaded, MODEL_SIZE)
        if downloaded != MODEL_SIZE or digest.hexdigest() != MODEL_SHA256:
            raise SeparationUnavailable("音源分離モデルの内容が固定したハッシュと一致しません。")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    report("complete", MODEL_SIZE, MODEL_SIZE)
    return target


class VocalSeparator:
    def __init__(self, settings, *, allow_download=False, download_progress=None):
        self.settings, self.allow_download = settings, allow_download
        self.download_progress = download_progress

    async def separate(self, video: Path, directory: Path) -> tuple[Path, Path]:
        """Return (vocals, background), 44.1 kHz stereo WAV, cached in ``directory``."""
        directory.mkdir(parents=True, exist_ok=True)
        vocals, background = directory / "vocals.wav", directory / "background.wav"
        marker = directory / "separation.json"
        identity = {"version": SEPARATION_VERSION, "model": MODEL_IDENTITY}
        try:
            if json.loads(marker.read_text()) == identity and vocals.is_file() and background.is_file():
                return vocals, background
        except (OSError, ValueError):
            pass
        model = await self._model()
        mixture = directory / "mixture.tmp.wav"
        try:
            # Mono programmes are duplicated to the two channels the model expects.
            await run_process("ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                              "-i", str(video), "-map", "0:a:0", "-vn", "-ac", "2", "-ar", str(INPUT_RATE),
                              "-c:a", "pcm_s16le", str(mixture), timeout=3600)
            partial = [directory / "vocals.tmp.wav", directory / "background.tmp.wav"]
            request = {"audio": str(mixture), "model": str(model),
                       "vocals": str(partial[0]), "background": str(partial[1])}
            env = {**os.environ, **({"PYTHONPATH": os.environ["TRANSIROYT_TTS_SITE_PACKAGES"]}
                                    if "TRANSIROYT_TTS_SITE_PACKAGES" in os.environ else {})}
            process = await asyncio.create_subprocess_exec(
                str(tts_python(self.settings)), "-B", str(self.settings.root / "scripts/separate_vocals.py"),
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE, env=env, start_new_session=True)
            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(json.dumps(request).encode()), 7200)
            except BaseException:
                if process.returncode is None:
                    process.kill()
                    await process.wait()
                raise
            try:
                response = json.loads(stdout.decode().strip().splitlines()[-1])
            except (ValueError, IndexError):
                response = {"ok": False, "error": stderr.decode(errors="replace")[-600:]}
            if process.returncode or not response.get("ok"):
                raise SeparationUnavailable(f"音声と背景音の分離に失敗しました: {response.get('error')}")
            partial[0].replace(vocals)
            partial[1].replace(background)
            marker.write_text(json.dumps(identity))
            return vocals, background
        finally:
            mixture.unlink(missing_ok=True)
            for path in (directory / "vocals.tmp.wav", directory / "background.tmp.wav"):
                path.unlink(missing_ok=True)

    async def _model(self) -> Path:
        path = separation_model_path(model_cache(self.settings))
        if path.is_file() and path.stat().st_size == MODEL_SIZE:
            return path
        if not self.allow_download:
            raise SeparationUnavailable(
                "音源分離モデル（約913MB）が未取得です。ダウンロードを許可して再実行してください。")
        if self.download_progress:
            from .model_downloads import download_model
            await download_model(self.settings, "separation_model", self.download_progress)
        else:
            await asyncio.to_thread(fetch_model, model_cache(self.settings), lambda *_: None)
        return path
