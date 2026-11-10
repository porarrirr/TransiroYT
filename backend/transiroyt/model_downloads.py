"""Pinned file-dubbing models, with byte progress and cancellable workers."""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .download_progress import DownloadRate, run_download, worker_args, write_progress
from .gemma_audio import (
    MODEL_FILES,
    MODEL_REPO,
    MODEL_REVISION,
    GemmaUnavailable,
    model_cache,
)
from .plamo_translation import PLAMO
from .process import ProcessError
from .speaker_diarization import ACTIVITY_MODEL_FILES, ACTIVITY_MODEL_REPO, ACTIVITY_MODEL_REVISION
from .whisper_transcription import MODEL_FILES as WHISPER_FILES
from .whisper_transcription import MODEL_REPO as WHISPER_REPO
from .whisper_transcription import MODEL_REVISION as WHISPER_REVISION

# IndexTTS-2.5 plus its auxiliary weights are about 7.3 GB. Allow a slow first
# download to finish without keeping every other model resource on a long leash.
INDEXTTS_TIMEOUT = 8 * 60 * 60

SPECS = {
    "gemma": (MODEL_REPO, MODEL_REVISION, MODEL_FILES),
    "plamo": (PLAMO.repo, PLAMO.revision, (PLAMO.file, "LICENSE/en", "LICENSE/ja", "README.md")),
    "speaker_model": ("speechbrain/spkrec-ecapa-voxceleb", "0f99f2d0ebe89ac095bcc5903c4dd8f72b367286",
                      ("embedding_model.ckpt", "hyperparams.yaml", "README.md")),
    "speech_check_model": (WHISPER_REPO, WHISPER_REVISION, WHISPER_FILES),
    "diarization_model": (ACTIVITY_MODEL_REPO, ACTIVITY_MODEL_REVISION, ACTIVITY_MODEL_FILES),
}


async def download_model(settings, resource, callback):
    directory = settings.data / "file-model-downloads"
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=directory) as temporary:
        progress_path = Path(temporary) / "progress.json"
        args = worker_args("transiroyt.model_downloads", resource, model_cache(settings), progress_path)
        overrides = {"timeout": INDEXTTS_TIMEOUT} if resource == "indextts_model" else {}
        try:
            await run_download(args, progress_path, callback, **overrides)
        except ProcessError as exc:
            raise GemmaUnavailable("モデルを取得できませんでした。接続や空き容量を確認して再試行してください。") from exc
        except TimeoutError as exc:
            raise GemmaUnavailable("モデルの取得が制限時間を超えました。再試行してください。取得済みファイルは再利用します。") from exc


def fetch(resource, cache, progress_path):
    rate = DownloadRate()
    if resource == "separation_model":
        from .vocal_separation import MODEL_FILE, fetch_model

        last = 0.0

        def report(state, downloaded, total):
            nonlocal last
            if state == "downloading" and time.monotonic() - last < 0.2:
                return
            write_progress(progress_path, {"resource": resource, "state": state, "filename": MODEL_FILE,
                                           "downloaded": downloaded, "total": total,
                                           **(rate.measure(downloaded, total) if state == "downloading" else {})})
            last = time.monotonic()

        fetch_model(cache, report)
        return
    if resource == "indextts_model":
        from .config import Settings
        from .reference_conversion import fetch_model

        # The worker's cache argument is the shared HF cache (data/models/huggingface);
        # IndexTTS keeps its weights under data/models/indextts-2.5 instead.
        last = 0.0

        def report(state, downloaded, total, filename):
            nonlocal last
            if state == "downloading" and time.monotonic() - last < 0.2:
                return
            write_progress(progress_path, {"resource": resource, "state": state, "filename": filename,
                                           "downloaded": downloaded, "total": total,
                                           **(rate.measure(downloaded, total) if state == "downloading" else {})})
            last = time.monotonic()

        asyncio.run(fetch_model(Settings(data=cache.parents[1]), report))
        return
    from huggingface_hub import get_hf_file_metadata, hf_hub_download, hf_hub_url
    from huggingface_hub.errors import LocalEntryNotFoundError
    from tqdm.auto import tqdm

    repo, revision, filenames = SPECS[resource]
    sizes, downloaded, cached, transferred = {}, {}, {}, {}
    last_report = 0.0
    lock = threading.RLock()

    def report(state, filename=None, *, force=False):
        nonlocal last_report
        with lock:
            if not force and time.monotonic() - last_report < 0.2:
                return
            total = sum(sizes.values()) if all(sizes.get(name) is not None for name in filenames) else None
            write_progress(progress_path, {"resource": resource, "state": state, "filename": filename,
                           "downloaded": sum(downloaded.values()), "total": total,
                           **(rate.measure(sum(downloaded.values()), total, transferred=sum(transferred.values())) if state == "downloading" else {})})
            last_report = time.monotonic()

    report("checking", force=True)
    for filename in filenames:
        try:
            path = Path(hf_hub_download(repo, filename, revision=revision, cache_dir=cache, local_files_only=True))
        except LocalEntryNotFoundError:
            path = None
        if path:
            cached[filename] = path
            sizes[filename] = downloaded[filename] = path.stat().st_size
        else:
            try:
                sizes[filename] = get_hf_file_metadata(hf_hub_url(repo, filename, revision=revision)).size
            except Exception:
                sizes[filename] = None
        report("checking", filename, force=True)

    def download_file(filename):
        class DownloadProgress(tqdm):
            def __init__(self, *args, **kwargs):
                kwargs["disable"] = False
                super().__init__(*args, **kwargs)

            def display(self, *args, **kwargs):
                with lock:
                    if self.total is not None:
                        sizes[filename] = int(self.total)
                    downloaded[filename] = int(self.n)
                    transferred[filename] = max(0, int(self.n - self.initial))
                    report("downloading", filename)

        report("downloading", filename, force=True)
        path = Path(hf_hub_download(repo, filename, revision=revision, cache_dir=cache, tqdm_class=DownloadProgress))
        with lock:
            sizes[filename] = downloaded[filename] = path.stat().st_size
            report("downloading", filename, force=True)

    pending = [name for name in filenames if name not in cached]
    if pending:
        with ThreadPoolExecutor(max_workers=min(2, len(pending))) as pool:
            list(pool.map(download_file, pending))
    report("complete", force=True)


def main():
    # Applied in the fresh worker before importing huggingface_hub/hf_xet.
    # Explicit user tuning still takes precedence.
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    resource, cache, progress = sys.argv[1:]
    try:
        fetch(resource, Path(cache), Path(progress))
    except Exception:
        print("モデルのダウンロードに失敗しました。", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
