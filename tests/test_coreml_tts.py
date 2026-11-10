import asyncio
import sys
from types import SimpleNamespace

import pytest
from transiroyt import coreml_tts


async def test_synthesis_request_carries_the_step_count_and_length(tmp_path, monkeypatch):
    worker = tmp_path / "worker"
    worker.write_text(f"#!{sys.executable}\n" + '''import json, sys
for line in sys.stdin:
    request = json.loads(line)
    print(json.dumps({"ok": True, "steps": request.get("steps"), "seconds": request.get("seconds")}), flush=True)
''')
    worker.chmod(0o700)
    monkeypatch.setattr(coreml_tts, "coreml_binary", lambda settings: worker)
    tts = coreml_tts.CoreMLTTS(SimpleNamespace(data=tmp_path, voice="ナレーション", steps=4))
    try:
        reference = tmp_path / "reference.wav"
        # File dubbing passes its own setting; other callers get the configured default.
        assert (await tts.synthesize("台詞", 0, tmp_path / "a.wav", reference=reference, steps=10,
                                     seconds=1.5)) == {"ok": True, "steps": 10, "seconds": 1.5}
        assert (await tts.synthesize("台詞", 0, tmp_path / "b.wav", reference=reference, seconds=1.5))["steps"] == 4
    finally:
        await tts.close()


async def test_two_workers_overlap_and_cancellation_releases_only_its_worker(tmp_path, monkeypatch):
    worker = tmp_path / "worker"
    worker.write_text(f"#!{sys.executable}\n" + '''import json, os, sys, time
from pathlib import Path
for line in sys.stdin:
    request = json.loads(line)
    if request["op"] == "synthesize":
        output = Path(request["output"])
        output.write_text(str(os.getpid()))
        while not output.with_suffix(".release").exists():
            time.sleep(.01)
    print(json.dumps({"ok": True, "pid": os.getpid()}), flush=True)
''')
    worker.chmod(0o700)
    monkeypatch.setattr(coreml_tts, "coreml_binary", lambda settings: worker)
    tts = coreml_tts.CoreMLTTS(SimpleNamespace(data=tmp_path, voice="", steps=10), workers=2)
    outputs = [tmp_path / f"{index}.wav" for index in range(3)]
    tasks = [asyncio.create_task(tts.synthesize("台詞", 0, output, reference=tmp_path / "ref.wav", seconds=1.0))
             for output in outputs]

    async def entered(output):
        while not output.exists():
            await asyncio.sleep(.01)

    try:
        await asyncio.wait_for(asyncio.gather(*(entered(p) for p in outputs[:2])), 3)
        assert outputs[0].read_text() != outputs[1].read_text()
        assert not outputs[2].exists()
        cancelled_process = tts.workers[0].process
        tasks[0].cancel()
        await asyncio.gather(tasks[0], return_exceptions=True)
        await asyncio.wait_for(entered(outputs[2]), 3)
        assert cancelled_process.returncode is not None
        assert not tasks[1].done()
        for output in outputs[1:]:
            output.with_suffix(".release").touch()
        await asyncio.wait_for(asyncio.gather(*tasks[1:]), 3)
        assert tts.available.qsize() == 2
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await tts.close()
    assert all(worker.process is None and worker.log is None for worker in tts.workers)


@pytest.mark.parametrize("available_gib, pressure, expected", [
    (4, 1, 1), (9, 1, 1), (10, 1, 2), (16, 1, 2), (16, 2, 1), (16, 4, 1),
])
def test_worker_count_uses_available_memory_and_pressure(tmp_path, monkeypatch, available_gib, pressure, expected):
    # Only a small part is completely free; most reusable memory is inactive.
    page_size = 16384
    pages = available_gib * 1024**3 // page_size
    stats = f"""Mach Virtual Memory Statistics: (page size of {page_size} bytes)
Pages free: 1024.
Pages inactive: {pages - 2048}.
Pages speculative: 1024.
Pages purgeable: 9999999.
"""
    def read(command, **kwargs):
        return stats if command == ['/usr/bin/vm_stat'] else str(pressure)
    monkeypatch.setattr(coreml_tts.subprocess, "check_output", read)
    assert coreml_tts.CoreMLTTS(SimpleNamespace(data=tmp_path, root=tmp_path)).capacity == expected


@pytest.mark.parametrize("stats", ["unreadable", "page size of 16384 bytes\nPages free: 20.\n"])
def test_missing_memory_statistics_use_one_worker(monkeypatch, caplog, stats):
    monkeypatch.setattr(coreml_tts.subprocess, "check_output", lambda *args, **kwargs: stats)
    assert coreml_tts.coreml_worker_capacity() == 1
    assert "using one worker" in caplog.text


def test_new_job_resamples_memory_but_explicit_workers_skip_probe(tmp_path, monkeypatch):
    snapshots = iter([2, 1])
    monkeypatch.setattr(coreml_tts, "coreml_worker_capacity", lambda: next(snapshots))
    settings = SimpleNamespace(data=tmp_path, root=tmp_path)
    assert coreml_tts.CoreMLTTS(settings).capacity == 2
    assert coreml_tts.CoreMLTTS(settings).capacity == 1
    assert coreml_tts.CoreMLTTS(settings, workers=2).capacity == 2
