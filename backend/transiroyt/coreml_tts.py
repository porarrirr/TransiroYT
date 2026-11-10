"""Core ML Irodori worker pool for file dubbing."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import subprocess
import time
from pathlib import Path

from .download_progress import DownloadRate
from .mobile_workers import TTS_IDENTITY

logger = logging.getLogger(__name__)

# Two ~3 GiB workers, ~2 GiB for Whisper, and ~2 GiB of headroom.
PARALLEL_MEMORY_BUDGET = 10 * 1024**3


def coreml_binary(settings):
    resources = os.getenv('TRANSIROYT_RESOURCE_ROOT')
    return (Path(resources) / 'bin/transiro-coreml' if resources else
            settings.root / '.local/bin/transiro-coreml')


def coreml_worker_capacity():
    """Sample reclaimable memory and pressure once for this generation job."""
    try:
        stats = subprocess.check_output(['/usr/bin/vm_stat'], text=True, timeout=2,
                                        env={**os.environ, 'LC_ALL': 'C'})
        page_size = re.search(r'page size of (\d+) bytes', stats)
        if page_size is None:
            raise ValueError('vm_stat page size is missing')
        pages = {name: int(count) for name, count in re.findall(
            r'^(Pages free|Pages inactive|Pages speculative):\s+(\d+)\.', stats, re.MULTILINE)}
        # vm_stat subtracts speculative pages from its displayed free count.
        # Do not add purgeable pages: they can overlap the inactive count.
        available = sum(pages[name] for name in ('Pages free', 'Pages inactive', 'Pages speculative')) * int(page_size[1])
        pressure = int(subprocess.check_output(
            ['/usr/sbin/sysctl', '-n', 'kern.memorystatus_vm_pressure_level'], text=True, timeout=2))
    except (OSError, subprocess.SubprocessError, ValueError, KeyError) as exc:
        logger.warning('Core ML memory status unavailable; using one worker: %s', exc)
        return 1
    # The sysctl reports the notification mask: 1=normal, 2=warning, 4=critical.
    capacity = 2 if pressure == 1 and available >= PARALLEL_MEMORY_BUDGET else 1
    logger.info('Core ML memory budget available_mib=%.0f required_mib=%.0f pressure=%d workers=%d',
                available / 1024**2, PARALLEL_MEMORY_BUDGET / 1024**2, pressure, capacity)
    return capacity


class CoreMLTTS:
    is_coreml = True
    identity = TTS_IDENTITY

    def __init__(self, settings, *, allow_download=False, download_progress=None, workers=None):
        self.capacity = workers if workers is not None else coreml_worker_capacity()
        self.workers = [_Worker(settings, index, allow_download=allow_download,
                                download_progress=download_progress if index == 0 else None)
                        for index in range(self.capacity)]
        self.available = asyncio.Queue()
        for worker in self.workers:
            self.available.put_nowait(worker)
        self.settings = settings
        self.preparation = None

    def preload(self):
        if self.preparation is None:
            self.preparation = asyncio.create_task(self._prepare())
        return self.preparation

    async def _prepare(self):
        try:
            # Complete a possible model download before a second process opens it.
            result = await self.workers[0]._request({'op': 'prepare'})
            await asyncio.gather(*(worker._request({'op': 'prepare'}) for worker in self.workers[1:]))
            logger.info('Core ML synthesis ready workers=%d', self.capacity)
            return result
        except BaseException:
            self.preparation = None
            raise

    async def synthesize(self, text, duration, output, reference=None, *, seconds, **kwargs):
        if reference is None:
            raise ValueError('ファイル吹き替えには本人の参照音声が必要です')
        await asyncio.shield(self.preload())
        return await self._request({'op': 'synthesize', 'text': text,
                                    'voice': kwargs.get('voice') or self.settings.voice,
                                    'reference': str(reference), 'output': str(output), 'seconds': seconds,
                                    'steps': kwargs.get('steps') or self.settings.steps})

    async def _request(self, request):
        queued = time.monotonic()
        worker = await self.available.get()
        started = time.monotonic()
        try:
            result = await worker._request(request)
            if request['op'] == 'synthesize':
                logger.info('Core ML synthesis worker=%d queue_ms=%.0f elapsed_ms=%.0f chars=%d steps=%d',
                            worker.index, (started - queued) * 1000,
                            (time.monotonic() - started) * 1000, len(request['text']), request['steps'])
            return result
        finally:
            self.available.put_nowait(worker)

    async def close(self):
        if self.preparation and not self.preparation.done():
            self.preparation.cancel()
            await asyncio.gather(self.preparation, return_exceptions=True)
        await asyncio.gather(*(worker._stop() for worker in self.workers))
        self.preparation = None


class _Worker:
    def __init__(self, settings, index, *, allow_download, download_progress):
        self.download_progress = download_progress
        self.settings = settings
        self.allow_download = allow_download
        self.binary = coreml_binary(settings)
        self.models = settings.data / 'models/irodori-coreml-light-int8-v0.2.0'
        self.process = None
        self.log = None
        self.lock = asyncio.Lock()
        self.index = index

    async def _request(self, request):
        async with self.lock:
            if not self.binary.is_file():
                raise RuntimeError('Core ML音声生成ツールがありません。scripts/build_coreml.sh でビルドしてください。')
            try:
                if self.process is None:
                    self.log = (self.settings.data / 'coreml-tts.log').open('a')
                    self.process = await asyncio.create_subprocess_exec(
                        str(self.binary), stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                        stderr=self.log)
                body = request | {'models': str(self.models), 'download': self.allow_download}
                if self.download_progress and request['op'] == 'prepare':
                    self.download_progress({'resource': 'coreml', 'state': 'preparing',
                                            'downloaded': 0, 'total': None})
                self.process.stdin.write((json.dumps(body, ensure_ascii=False) + '\n').encode())
                await self.process.stdin.drain()
                rate = DownloadRate()
                byte_progress = {}
                while True:
                    raw = await asyncio.wait_for(self.process.stdout.readline(), timeout=1800)
                    if not raw:
                        raise RuntimeError('Core ML音声生成が終了しました。coreml-tts.log を確認してください。')
                    result = json.loads(raw)
                    if result.get('event') != 'download':
                        break
                    if self.download_progress:
                        if "downloaded" in result:
                            byte_progress = {"downloaded": result["downloaded"], "total": result["total_bytes"],
                                             **rate.measure(result["downloaded"], result["total_bytes"],
                                                            transferred=result["transferred"])}
                        complete = result.get('completed') == result.get('total') if 'completed' in result else False
                        self.download_progress({'resource': 'coreml', 'state': 'preparing' if complete else 'downloading',
                                                'completed_files': result.get('completed'), 'total_files': result.get('total'),
                                                'downloaded': 0, 'total': None, **byte_progress,
                                                'updated_at': time.time(), 'filename': result['filename']})
                if not result.get('ok'):
                    raise RuntimeError(result.get('error', 'Core ML音声生成に失敗しました'))
                if self.download_progress and request['op'] == 'prepare':
                    self.download_progress({'resource': 'coreml', 'state': 'ready',
                                            'downloaded': 0, 'total': None})
                return result
            except BaseException:
                await self._stop()
                raise

    async def _stop(self):
        if self.process and self.process.returncode is None:
            self.process.kill()
            await self.process.wait()
        self.process = None
        if self.log:
            self.log.close()
            self.log = None
