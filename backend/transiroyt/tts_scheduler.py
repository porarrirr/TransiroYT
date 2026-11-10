"""Replan the bounded file-synthesis queue whenever work or device state changes."""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from functools import partial

logger = logging.getLogger(__name__)


@dataclass
class SynthesisJob:
    id: str
    payload: dict
    local: Callable[[], Awaitable[None]]
    future: asyncio.Future
    queued: float
    priority: int = 0
    attempts: int = 0


@dataclass
class RunningSynthesis:
    job: SynthesisJob
    started: float
    # Seconds this line takes on its executor, not counting the lines before it.
    estimate: float
    task: asyncio.Task


class TTSScheduler:
    def __init__(self, pool):
        self.pool = pool
        self.pending = []
        # Per executor, in the order it runs them: a device holds up to DEVICE_SLOTS.
        self.running: dict[str, list[RunningSynthesis]] = {}
        self.scheduled = False
        self.local_workers = 1

    async def run(self, payload, local, *, priority=0):
        """Lower ``priority`` runs first: file dubbing passes the line number."""
        job = SynthesisJob(uuid.uuid4().hex, payload, local,
                           asyncio.get_running_loop().create_future(), time.monotonic(), priority)
        self.pending.append(job)
        self.changed()
        try:
            return await job.future
        finally:
            if job in self.pending:
                self.pending.remove(job)
            active = next((r.task for queue in self.running.values() for r in queue if r.job is job), None)
            if active is not None:
                active.cancel()
                await asyncio.gather(active, return_exceptions=True)
            self.changed()

    def changed(self):
        if not self.scheduled:
            self.scheduled = True
            asyncio.get_running_loop().call_soon(self._dispatch)

    def _dispatch(self):
        self.scheduled = False
        self.pending = sorted((job for job in self.pending if not job.future.done()),
                              key=lambda job: (job.priority, job.queued))
        if not self.pending:
            return
        now = time.monotonic()
        # A device holding work this plan did not give it has an unknown backlog.
        devices = {device.id: device for device in self.pool.tts_devices()
                   if len(device.assignments) <= len(self.running.get(device.id, []))}
        local_workers = ['mac' if index == 0 else f'mac-{index + 1}'
                         for index in range(self.local_workers)]
        workers = [*local_workers, *devices]
        local_cost = self.pool.local_costs.get('tts', 0.1)
        costs = [*[local_cost] * self.local_workers,
                 *(d.costs.get('tts', local_cost) for d in devices.values())]
        finish = [self._backlog(w, now) for w in workers]
        # Earliest-completion list scheduling over the whole queue, in line
        # order: each job goes where it would finish first, after the work
        # already planned there. A slower phone takes a job once the faster
        # executors' backlog would finish it later; a lone job waits for a
        # faster busy executor. A job starts when its executor has room: the
        # Mac generator runs one line, a device holds a few so it never waits
        # for the next between lines.
        started = []
        for job in list(self.pending):
            durations = [max(0.01, costs[w] * self.pool.work_units('tts', job.payload)) for w in range(len(workers))]
            estimates = [finish[w] + durations[w] for w in range(len(workers))]
            worker = min(range(len(workers)), key=estimates.__getitem__)
            name = workers[worker]
            queue = self.running.setdefault(name, [])
            room = self.pool.has_slot(devices[name]) if name in devices else not queue
            if room:
                self.pending.remove(job)
                task = asyncio.create_task(self._execute(job, name, devices.get(name), estimates[worker]))
                entry = RunningSynthesis(job, now, durations[worker], task)
                queue.append(entry)
                task.add_done_callback(partial(self._finished, name, entry))
                started.append(name)
            finish[worker] = estimates[worker]
        logger.info('TTS plan pending=%d started=%s predicted_finish_ms=%.0f seconds_per_char=%s',
                    len(self.pending) + len(started), started, max(finish) * 1000,
                    dict(zip(workers, costs, strict=True)))

    def _backlog(self, worker, now):
        """Seconds until the executor finishes what it holds: the line under way, then the queued ones."""
        queue = self.running.get(worker, [])
        if not queue:
            return 0
        return max(0.01, queue[0].estimate - (now - queue[0].started)) + sum(r.estimate for r in queue[1:])

    def _finished(self, worker, entry, task):
        queue = self.running[worker]
        head = queue[0] is entry
        queue.remove(entry)
        if not queue:
            del self.running[worker]
        elif head:
            # The next queued line starts generating now.
            queue[0].started = time.monotonic()
        self.changed()

    async def _execute(self, job, worker, device, estimate):
        started = time.monotonic()
        job.attempts += 1
        task_id = f'{job.id}-{job.attempts}'
        logger.info('TTS assignment task=%s executor=%s queue_ms=%.0f predicted_ms=%.0f chars=%d',
                    task_id, worker, (started - job.queued) * 1000, estimate * 1000,
                    len(job.payload['text']))
        outcome = 'failed'
        try:
            if device is None:
                async with self.pool.local_work('tts', job.payload):
                    await job.local()
                value = None
            else:
                # The device may have filled up or stopped accepting between
                # planning and this task's first event-loop turn.
                if not self.pool.has_slot(device) or device not in self.pool.tts_devices():
                    outcome = 'replan'
                    if not job.future.done():
                        self.pending.insert(0, job)
                    return
                value = await self.pool.execute_on(device, 'tts', job.payload, task_id=task_id)
                if value is None:
                    outcome = 'retry'
                    if not job.future.done():
                        self.pending.insert(0, job)
                    return
            outcome = 'success'
            if not job.future.done():
                job.future.set_result(value)
        except asyncio.CancelledError:
            outcome = 'cancelled'
            job.future.cancel()
            raise
        except Exception as exc:
            if not job.future.done():
                job.future.set_exception(exc)
        finally:
            logger.info('TTS completion task=%s executor=%s elapsed_ms=%.0f outcome=%s',
                        task_id, worker, (time.monotonic() - started) * 1000, outcome)
