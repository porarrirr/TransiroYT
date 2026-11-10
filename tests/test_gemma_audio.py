import asyncio
from types import SimpleNamespace

import httpx
import pytest
from transiroyt.config import Settings
from transiroyt.gemma_audio import CONTEXT_TOKENS, GemmaAudio


@pytest.mark.parametrize("fallback", [False, True])
async def test_concurrent_requests_share_one_ready_server_and_keep_each_slots_context(tmp_path, monkeypatch, fallback):
    from transiroyt import gemma_audio
    health, requests_ready, finish = asyncio.Event(), asyncio.Event(), asyncio.Event()
    spawns, active, peaks = [], [], []

    class Process:
        returncode = None
        pid = 12345

        def kill(self):
            self.returncode = -9

        async def wait(self):
            return self.returncode

    async def spawn(*args, **kwargs):
        spawns.append(args)
        process = Process()
        if fallback and "--kv-unified" in args:
            process.returncode = 1
        return process

    class Client:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def get(self, *args, **kwargs):
            await health.wait()
            return SimpleNamespace(status_code=200)

        async def post(self, *args, **kwargs):
            assert health.is_set()
            active.append(1)
            peaks.append(len(active))
            if len(active) == 4:
                requests_ready.set()
            await finish.wait()
            active.pop()
            return SimpleNamespace(status_code=200, json=lambda: {
                "choices": [{"finish_reason": "stop", "message": {"content": "ok"}}]})

    monkeypatch.setattr(gemma_audio, "llama_server_binary", lambda: "test-llama-server")
    monkeypatch.setattr(gemma_audio, "model_file", lambda *a, **k: tmp_path / "model")
    monkeypatch.setattr(gemma_audio.asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(gemma_audio.httpx, "AsyncClient", Client)
    gemma = GemmaAudio(Settings(data=tmp_path), parallel=4)
    tasks = [asyncio.create_task(gemma.complete([])) for _ in range(8)]
    try:
        # Let the first request enter readiness polling. Other callers must
        # neither start another server nor post to the unready first process.
        expected_spawns = 4 if fallback else 2
        for _ in range(100):
            if len(spawns) == expected_spawns:
                break
            await asyncio.sleep(.01)
        assert len(spawns) == expected_spawns and not active  # server and parent watchdog
        args = spawns[-2]
        assert args[args.index("--ctx-size") + 1] == str(CONTEXT_TOKENS * (4 if fallback else 2))
        if not fallback:
            assert args[args.index("--kv-unified-per-slot") + 1] == str(CONTEXT_TOKENS)
        assert args[args.index("--parallel") + 1] == "4"
        assert args[args.index("--cache-ram") + 1] == "0"
        assert args[args.index("--spec-type") + 1] == "draft-mtp" and "--mmproj" not in args
        assert args[args.index("--spec-draft-n-max") + 1] == "3" and "--model-draft" in args
        health.set()
        await asyncio.wait_for(requests_ready.wait(), 2)
        assert max(peaks) == 4 and len(spawns) == expected_spawns
        finish.set()
        assert await asyncio.gather(*tasks) == ["ok"] * 8
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await gemma.close()


async def test_speculative_model_starts_with_its_draft_and_thinking_off(tmp_path, monkeypatch):
    from transiroyt import gemma_audio
    from transiroyt.gemma_audio import TextModel

    model = TextModel("org/model-GGUF", "rev", "model.gguf", "Test", reasoning_tokens=0,
                      spec_type="draft-mtp", draft=("org/drafts", "rev2", "mtp.gguf"), draft_max=3)
    spawns, bodies = [], []

    class Process:
        returncode, pid = None, 12345

        def kill(self):
            self.returncode = -9

        async def wait(self):
            return self.returncode

    async def spawn(*args, **kwargs):
        spawns.append(args)
        return Process()

    class Client:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def get(self, *args, **kwargs):
            return SimpleNamespace(status_code=200)

        async def post(self, url, *, json, **kwargs):
            bodies.append(json)
            return SimpleNamespace(status_code=200, json=lambda: {
                "choices": [{"finish_reason": "stop", "message": {"content": "ok"}}]})

    monkeypatch.setattr(gemma_audio, "llama_server_binary", lambda: "test-llama-server")
    monkeypatch.setattr(gemma_audio, "model_file", lambda *a, draft=False, **k: tmp_path / ("mtp.gguf" if draft else "model.gguf"))
    monkeypatch.setattr(gemma_audio.asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(gemma_audio.httpx, "AsyncClient", Client)
    gemma = GemmaAudio(Settings(data=tmp_path), model=model)
    try:
        assert await gemma.complete([gemma_audio.text_part("x")], max_tokens=100) == "ok"
    finally:
        await gemma.close()
    args = spawns[0]
    assert args[args.index("--spec-type") + 1] == "draft-mtp"
    assert args[args.index("--model-draft") + 1] == str(tmp_path / "mtp.gguf")
    assert args[args.index("--spec-draft-n-max") + 1] == "3"
    assert args[args.index("--reasoning-budget") + 1] == "0"
    # No thinking budget: the answer gets only its own tokens and thinking is switched off.
    assert bodies[0]["max_tokens"] == 100
    assert bodies[0]["chat_template_kwargs"] == {"enable_thinking": False}


@pytest.mark.parametrize(("max_tokens", "expected_limit"), [(100, 676), (-1, -1), (0, 0)])
async def test_local_thinking_budget_preserves_unlimited_outputs(monkeypatch, max_tokens, expected_limit):
    from transiroyt import gemma_audio

    model = gemma_audio.TextModel("org/model", "rev", "model.gguf", "Test", reasoning_tokens=64)
    gemma = GemmaAudio(Settings(), model=model)

    async def start():
        gemma.endpoint = "http://local.test"

    def answer(request):
        import json
        body = json.loads(request.content)
        assert body["max_tokens"] == expected_limit
        assert body["chat_template_kwargs"] == {"enable_thinking": True}
        assert body["messages"][0]["content"].endswith("\n\n追加の指示")
        assert request.headers["authorization"] == f"Bearer {gemma.key}"
        return httpx.Response(200, json={
            "choices": [{"finish_reason": "stop", "message": {"content": "応答"}}]})

    client = httpx.AsyncClient
    monkeypatch.setattr(gemma, "start", start)
    monkeypatch.setattr(gemma_audio.httpx, "AsyncClient",
                        lambda **kwargs: client(transport=httpx.MockTransport(answer), **kwargs))
    assert await gemma.complete([gemma_audio.text_part("本文")], max_tokens=max_tokens,
                                system="追加の指示") == "応答"
    assert gemma.context_available == gemma.context_capacity


@pytest.mark.parametrize(("status", "finish_reason", "error"), [
    (503, "stop", "HTTP 503"),
    (200, "length", "出力が上限"),
])
async def test_failed_local_inference_releases_context(monkeypatch, status, finish_reason, error):
    from transiroyt import gemma_audio

    gemma = GemmaAudio(Settings(), parallel=2)

    async def start():
        gemma.endpoint = "http://local.test"

    def answer(request):
        return httpx.Response(status, json={
            "choices": [{"finish_reason": finish_reason, "message": {"content": "途中"}}]})

    client = httpx.AsyncClient
    monkeypatch.setattr(gemma, "start", start)
    monkeypatch.setattr(gemma_audio.httpx, "AsyncClient",
                        lambda **kwargs: client(transport=httpx.MockTransport(answer), **kwargs))
    with pytest.raises(RuntimeError, match=error):
        await gemma.complete([gemma_audio.text_part("本文")])
    assert gemma.context_available == gemma.context_capacity


@pytest.mark.parametrize("parallel", [0, 5])
def test_gemma_rejects_unbounded_parallelism(parallel):
    with pytest.raises(ValueError):
        GemmaAudio(Settings(), parallel=parallel)


@pytest.mark.parametrize("download", [False, True])
def test_pinned_text_model_preserves_existing_cache(tmp_path, monkeypatch, download):
    import huggingface_hub
    from transiroyt.gemma_audio import MODEL_FILE, MODEL_REPO, MODEL_REVISION, model_file
    calls = []
    def fetch(repo, filename, **kwargs):
        calls.append((repo, filename, kwargs))
        return str(tmp_path / "model.gguf")
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fetch)
    old = tmp_path / "old-gemma.gguf"
    old.write_text("preserve")
    assert model_file(Settings(data=tmp_path), download=download) == tmp_path / "model.gguf"
    assert calls == [(MODEL_REPO, MODEL_FILE, {"revision": MODEL_REVISION,
        "cache_dir": tmp_path / "models/huggingface", "local_files_only": not download})]
    assert old.read_text() == "preserve"


async def test_shared_context_pool_queues_large_requests_and_drains_cancelled_waiters():
    gemma = GemmaAudio(Settings(), parallel=4)
    two_entered, release = asyncio.Event(), asyncio.Event()
    active, peak = [], []

    async def request():
        async with gemma._admit(CONTEXT_TOKENS):
            active.append(1)
            peak.append(len(active))
            if len(active) == 2:
                two_entered.set()
            try:
                await release.wait()
            finally:
                active.pop()

    tasks = [asyncio.create_task(request()) for _ in range(4)]
    try:
        await asyncio.wait_for(two_entered.wait(), 2)
        assert len(active) == 2 and gemma.context_available == 0
        tasks[-1].cancel()
        await asyncio.gather(tasks[-1], return_exceptions=True)
        release.set()
        await asyncio.gather(*tasks[:-1])
        assert max(peak) == 2
        assert gemma.context_available == gemma.context_capacity
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


def test_shared_context_budget_covers_text_and_unlimited_outputs():
    gemma = GemmaAudio(Settings(), parallel=4)
    assert gemma._context_budget([{"type": "text", "text": "あ" * 100}], 10, None, None) == 822
    assert gemma._context_budget([], -1, None, None) == CONTEXT_TOKENS


def test_packaged_llama_uses_only_the_bundled_server(tmp_path, monkeypatch):
    from transiroyt.gemma_audio import GemmaUnavailable, llama_server_binary

    monkeypatch.setenv("TRANSIROYT_RESOURCE_ROOT", str(tmp_path))
    monkeypatch.setenv("TRANSIROYT_LLAMA_SERVER", "/opt/homebrew/bin/llama-server")
    monkeypatch.setenv("PATH", "/opt/homebrew/bin:/usr/bin:/bin")
    with pytest.raises(GemmaUnavailable, match="再インストール"):
        llama_server_binary()
    binary = tmp_path / "bin/llama-server"
    binary.parent.mkdir()
    binary.write_text("#!/bin/sh\nexit 0\n")
    binary.chmod(0o755)
    assert llama_server_binary() == str(binary)


@pytest.mark.parametrize("raw", ['{"T1":"English source echoed"}', '{"T1":""}', '{"wrong":"訳"}',
                                 '{"T1":"訳","extra":"追加"}', '{"T1":"<think>思考</think>訳"}'])
def test_local_fallback_rejects_unusable_translation(raw):
    from transiroyt.gemma_audio import validate_local_response
    schema = {"type": "object", "properties": {"T1": {"type": "string"}}}
    with pytest.raises(RuntimeError):
        validate_local_response(raw, schema)


def test_local_fallback_accepts_japanese_translation_and_context_labels():
    from transiroyt.gemma_audio import validate_local_response
    assert validate_local_response('{"T1":"断ります"}', {"properties": {"T1": {}}}) == '{"T1":"断ります"}'
    assert validate_local_response('{"L2":"unknown"}', {"properties": {"L2": {}}}) == '{"L2":"unknown"}'
