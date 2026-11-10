"""Local macOS Keychain credentials; JSON files contain only opaque item IDs."""
from __future__ import annotations

import ctypes
import hashlib
import json
import os
import threading
from contextlib import contextmanager
from functools import wraps

_SETTINGS_LOCK = threading.RLock()


def serialized(action):
    @wraps(action)
    def call(*args, **kwargs):
        # The API moves blocking Keychain operations into worker threads.
        # Keep JSON migration and replacement serialized within this backend.
        with _SETTINGS_LOCK:
            return action(*args, **kwargs)
    return call


class CredentialError(RuntimeError):
    pass


class Keychain:
    def __init__(self, data):
        self.service = 'com.porarri.transiroyt.api.' + hashlib.sha256(str(data.resolve()).encode()).hexdigest()[:24]
        self.cf = ctypes.CDLL('/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation')
        self.sec = ctypes.CDLL('/System/Library/Frameworks/Security.framework/Security')
        pointer = ctypes.c_void_p
        self.cf.CFStringCreateWithCString.argtypes = [pointer, ctypes.c_char_p, ctypes.c_uint32]
        self.cf.CFStringCreateWithCString.restype = pointer
        self.cf.CFDataCreate.argtypes = [pointer, ctypes.c_char_p, ctypes.c_long]
        self.cf.CFDataCreate.restype = pointer
        self.cf.CFDataGetLength.argtypes = [pointer]
        self.cf.CFDataGetLength.restype = ctypes.c_long
        self.cf.CFDataGetBytePtr.argtypes = [pointer]
        self.cf.CFDataGetBytePtr.restype = pointer
        self.cf.CFDictionaryCreateMutable.argtypes = [pointer, ctypes.c_long, pointer, pointer]
        self.cf.CFDictionaryCreateMutable.restype = pointer
        self.cf.CFDictionarySetValue.argtypes = [pointer, pointer, pointer]
        self.cf.CFRelease.argtypes = [pointer]
        self.sec.SecItemCopyMatching.argtypes = [pointer, ctypes.POINTER(pointer)]
        self.sec.SecItemCopyMatching.restype = ctypes.c_int32
        self.sec.SecItemAdd.argtypes = [pointer, pointer]
        self.sec.SecItemAdd.restype = ctypes.c_int32
        self.sec.SecItemDelete.argtypes = [pointer]
        self.sec.SecItemDelete.restype = ctypes.c_int32

    @contextmanager
    def query(self, account, *, value=None, returning=False):
        # The dictionary has no retain callbacks. Own its values until the
        # synchronous Security call completes, then release each exactly once.
        owned = []
        query = self.cf.CFDictionaryCreateMutable(None, 0, None, None)
        def constant(name):
            return ctypes.c_void_p.in_dll(self.sec, name).value
        def string(text):
            result = self.cf.CFStringCreateWithCString(None, text.encode(), 0x08000100)
            owned.append(result)
            return result
        def add(key, item):
            self.cf.CFDictionarySetValue(query, constant(key), item)
        try:
            add('kSecClass', constant('kSecClassGenericPassword'))
            add('kSecAttrService', string(self.service))
            add('kSecAttrAccount', string(account))
            if value is not None:
                encoded = value.encode()
                blob = self.cf.CFDataCreate(None, encoded, len(encoded))
                owned.append(blob)
                add('kSecValueData', blob)
                add('kSecAttrLabel', string('TransiroYT APIキー'))
            if returning:
                add('kSecReturnData', ctypes.c_void_p.in_dll(self.cf, 'kCFBooleanTrue').value)
                add('kSecMatchLimit', constant('kSecMatchLimitOne'))
            yield query
        finally:
            self.cf.CFRelease(query)
            for item in reversed(owned):
                self.cf.CFRelease(item)

    @staticmethod
    def check(status):
        if status:
            raise CredentialError(f'キーチェーンにアクセスできません（OSStatus {status}）。Macのロックを解除し、キーチェーンへのアクセスを許可してから再試行してください。')

    def get(self, account):
        result = ctypes.c_void_p()
        with self.query(account, returning=True) as query:
            status = self.sec.SecItemCopyMatching(query, ctypes.byref(result))
        if status == -25300:
            return None
        self.check(status)
        try:
            return ctypes.string_at(self.cf.CFDataGetBytePtr(result), self.cf.CFDataGetLength(result)).decode()
        finally:
            self.cf.CFRelease(result)

    def put(self, account, value):
        # Callers allocate a new random account, so updating settings never
        # overwrites the credential referenced by the last committed JSON file.
        with self.query(account, value=value) as query:
            self.check(self.sec.SecItemAdd(query, None))
        if self.get(account) != value:
            raise CredentialError('キーチェーンへの保存を確認できません。設定ファイルは変更していません。')

    def delete(self, account):
        with self.query(account) as query:
            status = self.sec.SecItemDelete(query)
        if status != -25300:
            self.check(status)


def write_private_json(path, payload):
    temporary = path.with_suffix('.tmp')
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, 'w') as output:
        os.fchmod(output.fileno(), 0o600)
        json.dump(payload, output, ensure_ascii=False, indent=2)
    temporary.replace(path)
