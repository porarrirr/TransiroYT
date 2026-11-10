"""Convert speaker references to Japanese with a persistent IndexTTS-2.5 worker.

Pinned weights run in a dedicated Python 3.11 environment. Converted audio
is returned as 48 kHz mono PCM16 and falls back to the original reference
when transcript or voice checks fail."""
from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

from .errors import FileDubbingError
from .logging_utils import log_model_start
from .process import run_process

CONVERSION_TEXT = (
    "昨日は久しぶりに家族と一緒に近所の公園へ出かけて、満開の桜を眺めながらゆっくりと散歩を楽しみ、"
    "そのあとは駅前のカフェで温かいコーヒーを飲みました。"
)  # 87 morae; 10.5-17.0 s from the benchmark's two references.
REFERENCE_CONVERSION_VERSION = "indextts25-ja-87mora-v1"
IDENTITY = "IndexTTS-2.5 (0.8B, MPS fp32, 25 steps)"
MODEL_REPO = "IndexTeam/IndexTTS-2.5"
MODEL_REVISION = "c39ce5ba981572cb187443877ff559dfb246ce63"
W2V_BERT_REPO = "facebook/w2v-bert-2.0"
W2V_BERT_REVISION = "da985ba0987f70aaeb84a80f2851cfac8c697a7b"
BIGVGAN_REPO = "nvidia/bigvgan_v2_22khz_80band_256x"
BIGVGAN_REVISION = "633ff708ed5b74903e86ff1298cf4a98e921c513"
MASKGCT_REPO = "amphion/MaskGCT"
MASKGCT_REVISION = "265c6cef07625665d0c28d2faafb1415562379dc"
CAMPPLUS_REPO = "funasr/campplus"
CAMPPLUS_REVISION = "e4b6ede7ce16997aff4ae69fbca1f0175e2afede"
OUTPUT_RATE = 48000
WORKER_START_TIMEOUT = 1800.0
CONVERSION_TIMEOUT = 900.0


@dataclass(frozen=True)
class ModelFile:
    """A pinned file: where it comes from, its LFS size and where the worker reads it."""
    repo: str
    revision: str
    remote: str          # path in the repository
    size: int            # bytes; of the LFS object where the repository stores one
    directory: str       # hf_hub_download local_dir, relative to checkpoints
    name: str            # final path, relative to checkpoints


MODEL_FILES = (
    ModelFile(MODEL_REPO, MODEL_REVISION, "config.yaml", 2_860, "", "config.yaml"),
    ModelFile(MODEL_REPO, MODEL_REVISION, "gpt.pth", 3_259_599_833, "", "gpt.pth"),
    ModelFile(MODEL_REPO, MODEL_REVISION, "codec.pth", 607_290_935, "", "codec.pth"),
    ModelFile(MODEL_REPO, MODEL_REVISION, "s2mel.pth", 414_908_601, "", "s2mel.pth"),
    ModelFile(MODEL_REPO, MODEL_REVISION, "feat1.pt", 57_170, "", "feat1.pt"),
    ModelFile(MODEL_REPO, MODEL_REVISION, "feat2.pt", 374_866, "", "feat2.pt"),
    ModelFile(MODEL_REPO, MODEL_REVISION, "multilingual_zh_ja_yue_char_del.tiktoken", 907_395, "",
              "multilingual_zh_ja_yue_char_del.tiktoken"),
    ModelFile(MODEL_REPO, MODEL_REVISION, "wav2vec2bert_stats.pt", 9_343, "", "wav2vec2bert_stats.pt"),
    ModelFile(W2V_BERT_REPO, W2V_BERT_REVISION, "model.safetensors", 2_322_063_736,
              "hf_cache/w2v-bert-2.0", "hf_cache/w2v-bert-2.0/model.safetensors"),
    ModelFile(W2V_BERT_REPO, W2V_BERT_REVISION, "config.json", 1_874,
              "hf_cache/w2v-bert-2.0", "hf_cache/w2v-bert-2.0/config.json"),
    ModelFile(W2V_BERT_REPO, W2V_BERT_REVISION, "preprocessor_config.json", 275,
              "hf_cache/w2v-bert-2.0", "hf_cache/w2v-bert-2.0/preprocessor_config.json"),
    ModelFile(BIGVGAN_REPO, BIGVGAN_REVISION, "bigvgan_generator.pt", 449_228_171,
              "hf_cache/bigvgan", "hf_cache/bigvgan/bigvgan_generator.pt"),
    ModelFile(BIGVGAN_REPO, BIGVGAN_REVISION, "config.json", 1_405,
              "hf_cache/bigvgan", "hf_cache/bigvgan/config.json"),
    # The model loads both auxiliary files flat from hf_cache: keep the repository's
    # subdirectory only for the download, then move the file into place.
    ModelFile(MASKGCT_REPO, MASKGCT_REVISION, "semantic_codec/model.safetensors", 177_183_712,
              "hf_cache", "hf_cache/semantic_codec_model.safetensors"),
    ModelFile(CAMPPLUS_REPO, CAMPPLUS_REVISION, "campplus_cn_common.bin", 28_036_335,
              "hf_cache", "hf_cache/campplus_cn_common.bin"),
)


def model_dir(settings) -> Path:
    return settings.data / "models/indextts-2.5"


def checkpoints_dir(settings) -> Path:
    return model_dir(settings) / "checkpoints"

def _present(path: Path, size: int) -> bool:
    return path.is_file() and path.stat().st_size == size


def model_ready(settings) -> bool:
    """All pinned weights are in place at their expected size; the model is not loaded."""
    checkpoints = checkpoints_dir(settings)
    return all(_present(checkpoints / spec.name, spec.size) for spec in MODEL_FILES)


def converter_python(settings) -> Path:
    """The dedicated Python 3.11 environment that runs the worker."""
    configured = os.getenv("TRANSIROYT_INDEXTTS_PYTHON")
    candidates = ([Path(configured)] if configured else []) + [
        settings.root / "third_party/index-tts/.venv/bin/python",  # development checkout
        settings.root / "python-indextts/bin/python3.11",          # bundled runtime
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileDubbingError(
        "参照音声モデル（IndexTTS-2.5）用のPython環境が見つかりません。アプリを再インストールしてください。")


async def fetch_model(settings, report) -> None:
    """Download every pinned weight into checkpoints/, reporting total byte progress.

    ``report(state, downloaded, total, filename)`` gets "checking", "downloading"
    and "complete". Files already present at their pinned size are skipped.
    """
    await asyncio.to_thread(_fetch_model, settings, report)


def _fetch_model(settings, report):
    from huggingface_hub import hf_hub_download
    from tqdm.auto import tqdm

    checkpoints = checkpoints_dir(settings)
    checkpoints.mkdir(parents=True, exist_ok=True)
    total = sum(spec.size for spec in MODEL_FILES)
    # Bytes accounted for: a present file counts fully, a missing one grows with its
    # transfer. The progress bar's n already includes a resumed file's earlier bytes.
    pieces = {spec.name: spec.size if _present(checkpoints / spec.name, spec.size) else 0
              for spec in MODEL_FILES}
    last_report = 0.0

    def report_now(state, filename, *, force=False):
        nonlocal last_report
        if not force and time.monotonic() - last_report < 0.2:
            return
        report(state, sum(pieces.values()), total, filename)
        last_report = time.monotonic()

    def progress_class(spec):
        class DownloadProgress(tqdm):
            def __init__(self, *args, **kwargs):
                kwargs["disable"] = False
                super().__init__(*args, **kwargs)

            def display(self, *args, **kwargs):
                # Xet downloads report network and reconstructed bytes separately;
                # keep the larger so the total never steps backwards.
                received = min(spec.size, int(getattr(self, "n", 0) or 0))
                pieces[spec.name] = max(pieces[spec.name], received)
                report_now("downloading", spec.remote)

        return DownloadProgress

    report_now("checking", None, force=True)
    for spec in MODEL_FILES:
        target = checkpoints / spec.name
        if pieces[spec.name] == spec.size:
            continue
        source = checkpoints / spec.directory / spec.remote
        report_now("downloading", spec.remote, force=True)
        if source != target and _present(source, spec.size):
            # An interrupted earlier download left the repository path behind.
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source, target)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            path = Path(hf_hub_download(spec.repo, spec.remote, revision=spec.revision,
                                        local_dir=checkpoints / spec.directory,
                                        tqdm_class=progress_class(spec)))
            if path.resolve() != target.resolve():
                os.replace(path, target)
        pieces[spec.name] = spec.size
        report_now("downloading", spec.remote, force=True)
    if not model_ready(settings):
        raise RuntimeError(
            "参照音声モデル（IndexTTS-2.5）を取得しましたが、ファイルのサイズが固定値と一致しません。"
            "再試行してください。")
    report_now("complete", None, force=True)


class ReferenceConverter:
    """One persistent IndexTTS-2.5 worker; requests are serialised by ``convert``."""

    def __init__(self, settings):
        self.settings = settings
        self.process = None
        self.log = None
        self.lock = asyncio.Lock()

    @property
    def log_path(self) -> Path:
        return self.settings.data / "indextts.log"

    async def convert(self, reference: Path, output: Path) -> dict:
        """Read CONVERSION_TEXT in the reference's timbre, as a 48 kHz mono PCM16 WAV."""
        async with self.lock:
            await self._start()
            return await self._convert(reference, output)

    async def _convert(self, reference: Path, output: Path) -> dict:
        reference, output = Path(reference).resolve(), Path(output).resolve()
        if not reference.is_file():
            raise FileDubbingError(f"参照音声ファイルが見つかりません: {reference}")
        output.parent.mkdir(parents=True, exist_ok=True)
        take = output.with_name(f".{output.name}.indextts.wav")
        converted = output.with_name(f".{output.name}.converted.wav")
        take.unlink(missing_ok=True)
        try:
            reply = await self._request(
                {"text": CONVERSION_TEXT, "reference": str(reference), "output": str(take), "lang": "ja"},
                CONVERSION_TIMEOUT)
            try:
                await run_process("ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                                  "-i", str(take), "-ac", "1", "-ar", str(OUTPUT_RATE),
                                  "-c:a", "pcm_s16le", str(converted), timeout=120)
            except (RuntimeError, TimeoutError) as exc:
                raise RuntimeError(
                    f"参照音声の変換に失敗しました: {exc}。{self.log_path}を確認してください。") from exc
            os.replace(converted, output)
            return {"seconds": float(reply["seconds"])}
        finally:
            take.unlink(missing_ok=True)
            converted.unlink(missing_ok=True)

    async def _start(self):
        if self.process is not None and self.process.returncode is None:
            return
        if not model_ready(self.settings):
            raise FileDubbingError(
                "参照音声モデル（IndexTTS-2.5）が未取得です。ダウンロードを許可して再実行してください。")
        python = converter_python(self.settings)
        script = self.settings.root / "scripts/indextts_worker.py"
        if not script.is_file():
            raise FileDubbingError(f"参照音声モデルのworkerが見つかりません: {script}")
        self.settings.data.mkdir(parents=True, exist_ok=True)
        self.log = self.log_path.open("a")
        log_model_start(self.log, IDENTITY)
        env = {**os.environ, "PYTORCH_ENABLE_MPS_FALLBACK": "1", "USE_MODELSCOPE": "false",
               "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1"}
        try:
            self.process = await asyncio.create_subprocess_exec(
                str(python), "-B", "-u", str(script),
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=self.log,
                cwd=str(model_dir(self.settings)), env=env)
        except BaseException:
            await self._stop()
            raise
        try:
            async with asyncio.timeout(WORKER_START_TIMEOUT):
                raw = await self.process.stdout.readline()
        except TimeoutError as exc:
            await self._stop()
            raise RuntimeError(
                f"参照音声モデル（IndexTTS-2.5）の起動が{WORKER_START_TIMEOUT:g}秒を超えました。"
                f"{self.log_path}を確認してください。") from exc
        except BaseException:
            await self._stop()
            raise
        try:
            ready = json.loads(raw) if raw else {"error": "プロセスが終了しました"}
        except ValueError:
            ready = {"error": raw[:300].decode(errors="replace")}
        if not ready.get("ready"):
            await self._stop()
            raise RuntimeError(
                f"参照音声モデル（IndexTTS-2.5）を読み込めませんでした: {ready.get('error')}。"
                f"{self.log_path}を確認してください。")

    async def _request(self, request: dict, timeout: float) -> dict:
        if self.process is None or self.process.returncode is not None:
            await self._stop()
            raise RuntimeError(
                f"参照音声の生成プロセスが起動していません。{self.log_path}を確認してください。")
        try:
            self.process.stdin.write((json.dumps(request, ensure_ascii=False) + "\n").encode())
            await self.process.stdin.drain()
            async with asyncio.timeout(timeout):
                raw = await self.process.stdout.readline()
        except TimeoutError as exc:
            await self._stop()
            raise RuntimeError(
                f"参照音声の生成が{timeout:g}秒を超えました。{self.log_path}を確認してください。") from exc
        except BaseException:
            await self._stop()
            raise
        if not raw:
            await self._stop()
            raise RuntimeError(
                f"参照音声の生成プロセスが終了しました。{self.log_path}を確認してください。")
        try:
            reply = json.loads(raw)
        except ValueError:
            await self._stop()
            raise RuntimeError(
                f"参照音声の生成プロセスが不正な応答を返しました: {raw[:300]!r}。"
                f"{self.log_path}を確認してください。") from None
        if not reply.get("ok"):
            raise RuntimeError(f"参照音声の生成に失敗しました: {reply.get('error')}")
        return reply

    async def _stop(self):
        process, self.process = self.process, None
        if process is not None and process.returncode is None:
            process.terminate()
            try:
                async with asyncio.timeout(10):
                    await process.wait()
            except TimeoutError:
                process.kill()
                await process.wait()
        if self.log is not None:
            self.log.close()
            self.log = None

    async def close(self):
        """Stop the worker; a conversion still running fails instead of holding the shutdown."""
        await self._stop()
