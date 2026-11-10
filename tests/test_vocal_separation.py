"""Pinned model acquisition and cache migration without model downloads."""
from __future__ import annotations

import hashlib
from contextlib import nullcontext
from types import SimpleNamespace

import httpx
import pytest
from transiroyt import vocal_separation as separation


def test_download_verifies_before_publishing(tmp_path, monkeypatch):
    content = b'Kim weights'
    monkeypatch.setattr(separation, 'MODEL_SIZE', len(content))
    monkeypatch.setattr(separation, 'MODEL_SHA256', hashlib.sha256(content).hexdigest())
    monkeypatch.setattr(httpx, 'stream', lambda *a, **k: nullcontext(httpx.Response(200, content=content, request=httpx.Request('GET', separation.MODEL_URL))))
    progress = []
    path = separation.fetch_model(tmp_path / 'hub', lambda *args: progress.append(args))
    assert path.read_bytes() == content
    assert progress[-1] == ('complete', len(content), len(content))
    path.unlink()
    monkeypatch.setattr(separation, 'MODEL_SHA256', '0' * 64)
    with pytest.raises(separation.SeparationUnavailable, match='ハッシュ'):
        separation.fetch_model(tmp_path / 'hub', lambda *_: None)
    assert not path.exists()
    assert list(path.parent.iterdir()) == []


@pytest.mark.asyncio
async def test_old_cache_is_not_reused_and_remains_intact(tmp_path):
    directory = tmp_path / 'separation'
    directory.mkdir()
    (directory / 'separation.json').write_text('{"version":"obsolete-separation-version"}')
    for name in ('vocals.wav', 'background.wav'):
        (directory / name).write_bytes(b'old audio')
    separator = separation.VocalSeparator(SimpleNamespace())

    async def unavailable():
        raise separation.SeparationUnavailable('new model required')

    separator._model = unavailable
    with pytest.raises(separation.SeparationUnavailable, match='new model required'):
        await separator.separate(tmp_path / 'input.mp4', directory)
    assert (directory / 'vocals.wav').read_bytes() == b'old audio'


@pytest.mark.asyncio
async def test_current_cache_does_not_require_loading_model(tmp_path):
    import json

    (tmp_path / 'separation.json').write_text(json.dumps(
        {'version': separation.SEPARATION_VERSION, 'model': separation.MODEL_IDENTITY}))
    for name in ('vocals.wav', 'background.wav'):
        (tmp_path / name).touch()
    separator = separation.VocalSeparator(SimpleNamespace())
    assert await separator.separate(tmp_path / 'input.mp4', tmp_path) == (
        tmp_path / 'vocals.wav', tmp_path / 'background.wav')
