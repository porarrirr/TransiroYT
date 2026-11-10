"""Build the same Rust PCM kernel that is shipped in the app."""
import runpy
from pathlib import Path

import pytest


def pytest_sessionstart(session):
    runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/build_native_audio.py"))["build"]()


@pytest.fixture(autouse=True)
def isolated_keychain(monkeypatch):
    """Unit tests exercise persistence without writing the user's real Keychain."""
    from transiroyt import credentials
    items = {}
    class MemoryKeychain:
        def __init__(self, data):
            self.scope = str(data.resolve())
        def get(self, account):
            return items.get((self.scope, account))
        def put(self, account, value):
            items[self.scope, account] = value
        def delete(self, account):
            items.pop((self.scope, account), None)
    monkeypatch.setattr(credentials, 'Keychain', MemoryKeychain)
    return items
