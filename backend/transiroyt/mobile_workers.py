"""Opt-in, isolated HTTPS worker service; never exposes the desktop API on the LAN."""
from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import logging
import math
import os
import secrets
import socket
import ssl
import time
import uuid
import wave
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Literal

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from .tts_scheduler import TTSScheduler
from .usb_workers import USBWorkers

logger = logging.getLogger(__name__)
TTS_IDENTITY = "irodori-coreml-948f17da-light-int8-v0.2.0-unwatermarked-v2-timed"
HEARTBEAT_TIMEOUT = 20
TASK_TIMEOUT = 600
MAX_BODY = 20 * 1024 * 1024
DEVICE_LIMIT = 2
# Lines a device holds: one generating and two waiting, so it never waits for
# the Mac between lines (1-2 s per 3 s line with one at a time, 2026-10-31).
DEVICE_SLOTS = 3
# Result ids kept so a resent result is acknowledged without a second completion.
RECEIVED_KEPT = 16


class PairRequest(BaseModel):
    code: str = Field(max_length=100)
    name: str = Field(min_length=1, max_length=80)
    device_id: uuid.UUID


Digest = Annotated[str, Field(min_length=64, max_length=64)]


class WorkerStatus(BaseModel):
    capabilities: dict[Literal['tts'], str] = Field(default_factory=dict)
    state: str = Field(max_length=100)
    accepting: bool = True
    # The tasks the app holds, generating first; a delivered task missing here was lost.
    tasks: list[Annotated[str, Field(max_length=64)]] = Field(default_factory=list, max_length=DEVICE_SLOTS)
    # SHA-256 of the reference WAVs the app keeps for this connection. Apps
    # without this field receive the reference with every task.
    references: list[Digest] = Field(default_factory=list, max_length=64)


class WorkerResult(BaseModel):
    task_id: str = Field(max_length=64)
    wav: str | None = Field(default=None, max_length=16 * 1024 * 1024)
    error: str | None = Field(default=None, max_length=500)
    # Generation time on the device; the wait in its queue is not speed.
    elapsed_ms: int | None = Field(default=None, ge=0)
    # The app may drop an old reference while it runs a task.
    references: list[Digest] | None = Field(default=None, max_length=64)


@dataclass
class Assignment:
    id: str
    kind: str
    payload: dict
    future: asyncio.Future
    delivered: bool = False


@dataclass
class Device:
    id: str
    name: str
    token: str
    seen: float = field(default_factory=time.monotonic)
    capabilities: dict = field(default_factory=dict)
    state: str = 'モデルを準備しています'
    accepting: bool = False
    # In the order the device runs them.
    assignments: dict[str, Assignment] = field(default_factory=dict)
    held: list[str] = field(default_factory=list)
    completed: int = 0
    failures: int = 0
    seconds: dict = field(default_factory=dict)
    costs: dict = field(default_factory=dict)
    retry_after: float = 0
    received: list[str] = field(default_factory=list)
    task_ready: asyncio.Event = field(default_factory=asyncio.Event)
    references: set = field(default_factory=set)


class WorkerAccessFilter(logging.Filter):
    """Keep worker polling out of the desktop API's access log.

    Both servers write to uvicorn's process-wide ``uvicorn.access`` logger, and
    uvicorn's own ``access_log``/``log_level`` options reconfigure that logger
    for the desktop API as well.
    """

    paths = frozenset({'/pair', '/heartbeat', '/task', '/result', '/connection'})

    def filter(self, record):
        return record.args[2].partition('?')[0] not in self.paths


WORKER_ACCESS_FILTER = WorkerAccessFilter()


class WorkerServer(uvicorn.Server):
    @contextmanager
    def capture_signals(self):
        # The desktop API server owns process termination and lifespan cleanup.
        yield


class MobileWorkers:
    def __init__(self, data: Path):
        self.data = data
        self.directory = data / 'mobile-workers'
        self.devices: dict[str, Device] = {}
        self.port = 0
        self.code = ''
        self.expires = 0.0
        self.url = None
        self.transport = None
        self.usb = None
        self.server = None
        self.server_task = None
        self.socket = None
        self.local_costs = {}
        self.local_active = {'tts': 0}
        self.local_references = set()
        self.reference_digests = {}
        self.tts = TTSScheduler(self)

    def public(self):
        now = time.monotonic()
        return {'enabled': self.server_task is not None, 'pairing_url': self.url if now < self.expires else None,
                'pairing_seconds': max(0, int(self.expires - now)), 'pairing_transport': self.transport, 'limit': DEVICE_LIMIT,
                'devices': [{'id': d.id, 'name': d.name, 'online': now - d.seen < HEARTBEAT_TIMEOUT,
                             'state': d.state, 'accepting': d.accepting,
                             'task': next(iter(d.assignments.values())).kind if d.assignments else None,
                             'completed': d.completed, 'failures': d.failures,
                             'seconds': d.seconds, 'capabilities': list(d.capabilities)}
                            for d in self.devices.values()]}

    async def enable(self, transport="wifi"):
        address = '127.0.0.1' if transport == 'usb' else await self._pairing_address()
        await self._listen(self.port)
        self.transport = transport
        self.code = secrets.token_urlsafe(32)
        self.expires = time.monotonic() + 120
        self.url = f'https://{address}:{self.port}#pin={self.fingerprint}&code={self.code}&transport={transport}'
        self._save()
        return self.public()

    async def resume(self):
        """Listen again for the devices paired before this launch; they reconnect with their saved tokens."""
        paired = self.directory / 'devices.json'
        if not paired.exists():
            return
        saved = json.loads(paired.read_text())
        if not saved['devices']:
            return
        for entry in saved['devices']:
            self.devices[entry['id']] = Device(entry['id'], entry['name'], entry['token'], seen=-math.inf)
        await self._listen(saved['port'])
        self.transport = saved['transport']
        self._save()
        logger.info('Mobile pairing resumed devices=%d transport=%s', len(self.devices), self.transport)

    async def _listen(self, port):
        if self.server_task is not None:
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        cert, key = self.directory / 'certificate.pem', self.directory / 'private-key.pem'
        if not cert.exists() or not key.exists():
            process = await asyncio.create_subprocess_exec(
                '/usr/bin/openssl', 'req', '-x509', '-newkey', 'rsa:2048', '-nodes',
                '-keyout', str(key), '-out', str(cert), '-days', '3650',
                '-subj', '/CN=TransiroYT Mobile', stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL)
            if await process.wait():
                raise RuntimeError('モバイル接続の証明書を作成できませんでした。')
            key.chmod(0o600)
        self.socket = socket.socket()
        # Connections from before a restart linger in TIME_WAIT on the saved port.
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self.socket.bind(('0.0.0.0', port))
        except OSError as exc:
            # Paired devices know only this port; they need a new QR code.
            logger.warning('Mobile port %d unavailable; paired devices must scan a new QR code: %s', port, exc)
            self.socket.bind(('0.0.0.0', 0))
        self.socket.listen(16)
        self.socket.setblocking(False)
        self.port = self.socket.getsockname()[1]
        self.fingerprint = hashlib.sha256(ssl.PEM_cert_to_DER_cert(cert.read_text())).hexdigest()
        logging.getLogger('uvicorn.access').addFilter(WORKER_ACCESS_FILTER)
        config = uvicorn.Config(self.worker_app(), ssl_certfile=str(cert), ssl_keyfile=str(key),
                                log_config=None, ws='none', lifespan='off')
        self.server = WorkerServer(config)
        self.server_task = asyncio.create_task(self.server.serve(sockets=[self.socket]))
        while not self.server.started:
            if self.server_task.done():
                await self.server_task
                raise RuntimeError('モバイル接続を開始できませんでした。')
            await asyncio.sleep(0.02)
        # Paired devices keep the method of their own QR code, while `transport`
        # holds only the latest one, so USB tunnels run whenever the service does.
        self.usb = USBWorkers(self.port)

    def _save(self):
        """Device tokens survive restarts so paired devices reconnect without a new QR code."""
        self.directory.mkdir(parents=True, exist_ok=True)
        temporary = self.directory / 'devices.json.tmp'
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(descriptor, 'w') as file:
            json.dump({'port': self.port, 'transport': self.transport,
                       'devices': [{'id': d.id, 'name': d.name, 'token': d.token} for d in self.devices.values()]}, file)
        temporary.replace(self.directory / 'devices.json')

    async def _pairing_address(self):
        interfaces = ('en0', 'en1')
        for interface in interfaces:
            process = await asyncio.create_subprocess_exec('/usr/sbin/ipconfig', 'getifaddr', interface,
                                                           stdout=asyncio.subprocess.PIPE,
                                                           stderr=asyncio.subprocess.DEVNULL)
            output, _ = await process.communicate()
            if process.returncode == 0 and output.strip():
                return output.decode().strip()
        raise RuntimeError('Wi-FiのIPアドレスを確認できません。MacをWi-Fiに接続してください。')

    def remove(self, device_id):
        self._drop(device_id)
        self._save()

    def _drop(self, device_id):
        device = self.devices.pop(device_id, None)
        if device:
            device.task_ready.set()
            self._fail(device, ConnectionError('端末との接続が解除されました'))
        self.tts.changed()

    async def stop(self):
        """Stop sharing work and forget every paired device."""
        await self.close()
        (self.directory / 'devices.json').unlink(missing_ok=True)

    async def close(self):
        """Shut the listener down; paired devices stay saved for the next launch."""
        if self.usb:
            await self.usb.close()
            self.usb = None
        for device_id in list(self.devices):
            self._drop(device_id)
        if self.server:
            self.server.should_exit = True
        if self.server_task:
            await self.server_task
        if self.socket:
            self.socket.close()
        self.server = self.server_task = self.socket = self.url = None
        self.code = ''
        self.expires = 0

    @staticmethod
    def work_units(kind, payload):
        return max(1, len(payload.get('text', '')))

    @asynccontextmanager
    async def local_work(self, kind, payload):
        self.local_active[kind] += 1
        started = time.monotonic()
        # The Mac's first line of a voice also loads the model or the reference
        # (54.5 s for 86 characters, then 2.8 s for 26); like a device's first
        # task, it is not a speed sample.
        reference = payload.get('reference_path')
        first = reference is not None and reference not in self.local_references
        try:
            yield
            if first:
                self.local_references.add(reference)
            else:
                cost = (time.monotonic() - started) / self.work_units(kind, payload)
                self.local_costs[kind] = 0.7 * self.local_costs.get(kind, cost) + 0.3 * cost
        finally:
            self.local_active[kind] -= 1

    def tts_devices(self):
        """Connected devices for speech; a device still running work the Mac withdrew is left out."""
        now = time.monotonic()
        return [d for d in self.devices.values() if d.accepting and set(d.held) <= set(d.assignments)
                and now >= d.retry_after
                and now - d.seen < HEARTBEAT_TIMEOUT and d.capabilities.get('tts') == TTS_IDENTITY]

    @staticmethod
    def has_slot(device):
        return len(device.assignments) < DEVICE_SLOTS

    @staticmethod
    def _fail(device, error):
        for assignment in device.assignments.values():
            if not assignment.future.done():
                assignment.future.set_exception(error)

    async def execute(self, kind: str, payload: dict):
        """Reserve an idle compatible device atomically. Busy/absent devices cost no wait."""
        now = time.monotonic()
        available = [d for d in self.devices.values() if d.accepting and not d.held and not d.assignments
                     and now >= d.retry_after
                     and now - d.seen < HEARTBEAT_TIMEOUT and d.capabilities.get(kind) == TTS_IDENTITY]
        if not available:
            return None
        local_cost = self.local_costs.get(kind)
        if local_cost is not None:
            threshold = local_cost * (1 + self.local_active[kind])
            available = [d for d in available if kind not in d.costs or d.costs[kind] <= threshold]
        if not available:
            return None
        device = min(available, key=lambda d: d.costs.get(kind, 0))
        return await self.execute_on(device, kind, payload)

    async def execute_on(self, device, kind, payload, *, task_id=None):
        task_id = task_id or uuid.uuid4().hex
        payload = self._tts_payload(device, payload)
        future = asyncio.get_running_loop().create_future()
        device.assignments[task_id] = Assignment(task_id, kind, payload, future)
        device.task_ready.set()
        try:
            async with asyncio.timeout(TASK_TIMEOUT):
                while not future.done():
                    await asyncio.wait([future], timeout=1)
                    if time.monotonic() - device.seen >= HEARTBEAT_TIMEOUT:
                        raise ConnectionError('端末の応答が途切れました')
                result = future.result()
            if result.error:
                raise RuntimeError(result.error)
            raw = base64.b64decode(result.wav or '', validate=True)
            with wave.open(io.BytesIO(raw), 'rb') as wav:
                if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) != (1, 2, 48000):
                    raise ValueError('端末の音声形式が一致しません')
                frames = wav.getnframes()
                if not 0 < frames <= 48000 * 120 or len(wav.readframes(frames)) != frames * 2:
                    raise ValueError('端末の音声データが不完全です')
            if result.elapsed_ms is None:
                raise ValueError('端末の処理時間がありません')
            device.completed += 1
            elapsed = result.elapsed_ms / 1000
            logger.info('Mobile %s completion task=%s executor=%s elapsed_ms=%.0f units=%.1f',
                        kind, task_id, device.id, elapsed * 1000, self.work_units(kind, payload))
            device.seconds[kind] = round(elapsed, 2)
            # A task that carried the reference also paid its transfer and the
            # device's first use of it: 22.9 s for 44 characters, then 3.7 s for
            # 55. As a speed sample it kept a fast iPad out of the plan.
            if 'reference' not in payload:
                cost = elapsed / self.work_units(kind, payload)
                device.costs[kind] = 0.7 * device.costs.get(kind, cost) + 0.3 * cost
            return raw
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            device.failures += 1
            device.retry_after = time.monotonic() + HEARTBEAT_TIMEOUT
            logger.warning('Mobile %s failed; falling back to Mac: %s', kind, exc)
            return None
        finally:
            future.cancel()
            self._release(device, task_id)
            self.tts.changed()

    @staticmethod
    def _release(device, task_id):
        """Withdraw an assignment; a reference it alone was carrying moves to the next for that voice."""
        assignment = device.assignments.pop(task_id, None)
        if assignment is None or assignment.delivered or 'reference' not in assignment.payload:
            return
        digest = assignment.payload['reference_sha256']
        heir = next((a for a in device.assignments.values()
                     if not a.delivered and a.payload.get('reference_sha256') == digest), None)
        if heir is not None:
            heir.payload['reference'] = assignment.payload['reference']

    def _tts_payload(self, device, payload):
        """Send a reference's bytes only to a device that does not hold it yet.

        A 60 s reference is 7.7 MB of base64 in every task otherwise.
        """
        path = payload['reference_path']
        if path not in self.reference_digests:
            self.reference_digests[path] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        digest = self.reference_digests[path]
        wire = {key: value for key, value in payload.items() if key != 'reference_path'}
        wire['reference_sha256'] = digest
        # The app saves a reference when it receives a task, so one queued task carries it.
        carried = any(a.payload['reference_sha256'] == digest and 'reference' in a.payload
                      for a in device.assignments.values())
        if digest not in device.references and not carried:
            wire['reference'] = base64.b64encode(Path(path).read_bytes()).decode()
        return wire

    def worker_app(self):
        app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

        @app.middleware('http')
        async def body_limit(request, call_next):
            chunks, size = [], 0
            async for chunk in request.stream():
                size += len(chunk)
                if size > MAX_BODY:
                    from fastapi.responses import JSONResponse
                    return JSONResponse({'detail': 'Request too large'}, status_code=413)
                chunks.append(chunk)
            request._body = b''.join(chunks)
            response = await call_next(request)
            response.headers['Cache-Control'] = 'no-store'
            return response

        async def device_for(request: Request):
            token = request.headers.get('authorization', '')
            for device in self.devices.values():
                if secrets.compare_digest(token, f'Bearer {device.token}'):
                    device.seen = time.monotonic()
                    return device
            raise HTTPException(401, 'Macで接続し直してください')

        authenticated_device = Depends(device_for)

        @app.post('/pair')
        async def pair(body: PairRequest):
            if time.monotonic() >= self.expires or not secrets.compare_digest(body.code, self.code):
                raise HTTPException(401, '接続コードの有効期限が切れています')
            if len(self.devices) >= DEVICE_LIMIT and body.device_id.hex not in self.devices:
                raise HTTPException(409, '追加できる端末は最大2台です')
            device_id = body.device_id.hex
            self._drop(device_id)
            device = Device(device_id, body.name, secrets.token_urlsafe(32))
            self.devices[device.id] = device
            self._save()
            return {'token': device.token, 'id': device.id, 'tts_identity': TTS_IDENTITY}

        @app.post('/heartbeat')
        async def heartbeat(body: WorkerStatus, device: Device = authenticated_device):
            device.held = body.tasks
            for assignment in device.assignments.values():
                if assignment.delivered and assignment.id not in body.tasks and not assignment.future.done():
                    assignment.future.set_exception(ConnectionError('端末で処理が終了しましたが結果を受信できませんでした'))
            device.capabilities = body.capabilities
            device.references = set(body.references)
            device.state, device.accepting = body.state, body.accepting
            if not body.accepting:
                self._fail(device, ConnectionError('端末が一時停止しました'))
            self.tts.changed()
            # A resumed device skips /pair, so it checks the model versions here.
            return {'task_ids': list(device.assignments), 'completed': device.completed,
                    'tts_identity': TTS_IDENTITY, 'slots': DEVICE_SLOTS}

        @app.get('/task')
        async def task(wait: bool = False, device: Device = authenticated_device):
            def undelivered():
                return [a for a in device.assignments.values() if not a.delivered]

            if wait and not undelivered() and device.accepting:
                device.task_ready.clear()
                try:
                    async with asyncio.timeout(5):
                        await device.task_ready.wait()
                except TimeoutError:
                    pass
            if device.id not in self.devices:
                raise HTTPException(401, 'Macで接続し直してください')
            tasks = undelivered()
            for assignment in tasks:
                assignment.delivered = True
                logger.info('Mobile task received task=%s executor=%s', assignment.id, device.id)
            return {'tasks': [{'id': a.id, 'kind': a.kind, 'payload': a.payload} for a in tasks]}

        @app.post('/result')
        async def result(body: WorkerResult, device: Device = authenticated_device):
            if body.task_id in device.received:
                return {'ok': True}
            assignment = device.assignments.get(body.task_id)
            if assignment is None:
                raise HTTPException(409, 'この処理は既にMacで回収しました')
            if assignment.future.done():
                raise HTTPException(409, '結果は受信済みです')
            device.received = [*device.received, body.task_id][-RECEIVED_KEPT:]
            if body.references is not None:
                device.references = set(body.references)
            # The app has finished this task; it need not report it first.
            device.held = [task_id for task_id in device.held if task_id != body.task_id]
            assignment.future.set_result(body)
            return {'ok': True}

        @app.delete('/connection')
        async def disconnect(device: Device = authenticated_device):
            self.remove(device.id)
            return {'ok': True}

        return app
