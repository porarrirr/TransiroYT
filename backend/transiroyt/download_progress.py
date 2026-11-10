"""Atomic progress reports shared with cancellable download workers."""
from __future__ import annotations

import asyncio
import json
import sys
import time
from collections import deque
from pathlib import Path

from .process import run_process


class DownloadRate:
    """Recent byte throughput; the first sample includes resumed/cached bytes."""

    def __init__(self):
        self.samples = deque()

    def measure(self, downloaded, total, *, transferred=None, now=None):
        now = time.monotonic() if now is None else now
        received = downloaded if transferred is None else transferred
        if self.samples and received < self.samples[-1][1]:
            self.samples.clear()
        self.samples.append((now, received))
        while len(self.samples) > 2 and self.samples[1][0] <= now - 3:
            self.samples.popleft()
        elapsed = now - self.samples[0][0]
        speed = (received - self.samples[0][1]) / elapsed if elapsed > 0 else None
        eta = max(0, total - downloaded) / speed if total and speed else None
        return {"speed": speed, "eta": eta}


def write_progress(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({**value, "updated_at": time.time()}, ensure_ascii=False))
    temporary.replace(path)


def worker_args(module, *args):
    return [sys.executable, "-B", "-s", "-c",
            "import runpy,sys; sys.path.insert(0,sys.argv.pop(1)); "
            f"runpy.run_module('{module}',run_name='__main__')",
            str(Path(__file__).resolve().parents[1]), *map(str, args)]


async def run_download(args, progress_path, callback=None, *, timeout=7200):
    task = asyncio.create_task(run_process(*args, timeout=timeout, process_group=True))
    previous = None

    def report():
        nonlocal previous
        try:
            value = json.loads(progress_path.read_text())
        except (OSError, ValueError):
            return
        if callback and value != previous:
            callback(value)
            previous = value

    try:
        while not task.done():
            report()
            await asyncio.wait({task}, timeout=0.25)
        await task
        report()
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
