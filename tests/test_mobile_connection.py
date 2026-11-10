"""Pairing must advertise the selected network, including direct USB."""
import asyncio
from unittest.mock import AsyncMock, Mock

import httpx
from transiroyt import mobile_workers
from transiroyt.app import create_app
from transiroyt.config import Settings
from transiroyt.mobile_workers import MobileWorkers


async def test_transport_api_validation_and_usb_error(monkeypatch, tmp_path):
    enable = AsyncMock(side_effect=RuntimeError('USB接続を確認できません。'))
    monkeypatch.setattr(MobileWorkers, 'enable', enable)
    app = create_app(Settings(data=tmp_path, token='test'))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://127.0.0.1',
                                 headers={'Authorization': 'Bearer test'}) as client:
        assert (await client.post('/api/mobile?transport=other')).status_code == 422
        response = await client.post('/api/mobile?transport=usb')
        assert response.status_code == 409
        enable.assert_awaited_once_with('usb')


async def test_usb_pairing_https_service(monkeypatch, tmp_path):
    import hashlib
    import ssl
    from urllib.parse import parse_qs, urlsplit

    pool = MobileWorkers(tmp_path)
    wifi = AsyncMock(side_effect=AssertionError('USB must not query Wi-Fi'))
    monkeypatch.setattr(pool, '_pairing_address', wifi)
    try:
        status = await pool.enable('usb')
        wifi.assert_not_called()
        assert status['pairing_transport'] == 'usb'
        link = urlsplit(status['pairing_url'])
        credentials = parse_qs(link.fragment)
        cert = tmp_path / 'mobile-workers/certificate.pem'
        assert credentials['pin'] == [hashlib.sha256(ssl.PEM_cert_to_DER_cert(cert.read_text())).hexdigest()]
        context = ssl.create_default_context(cafile=str(cert))
        context.check_hostname = False
        async with httpx.AsyncClient(base_url=f'https://127.0.0.1:{link.port}', verify=context) as client:
            import uuid
            response = await client.post('/pair', json={'code': credentials['code'][0],
                                                       'name': 'USB iPhone', 'device_id': str(uuid.uuid4())})
            assert response.status_code == 200
            assert pool.public()['devices'][0]['name'] == 'USB iPhone'
    finally:
        await pool.close()


async def test_usbmux_message_and_usb_only_discovery(monkeypatch):
    import plistlib
    import struct

    from transiroyt.usb_workers import usb_devices

    reply = plistlib.dumps({'DeviceList': [
        {'DeviceID': 3, 'Properties': {'ConnectionType': 'USB'}},
        {'DeviceID': 4, 'Properties': {'ConnectionType': 'Network'}}]})
    reader = asyncio.StreamReader()
    reader.feed_data(struct.pack('<IIII', 16 + len(reply), 1, 8, 1) + reply)
    writer = AsyncMock()
    writer.write = Mock()
    writer.close = Mock()
    monkeypatch.setattr(asyncio, 'open_unix_connection', AsyncMock(return_value=(reader, writer)))
    assert await usb_devices() == [3]
    request = writer.write.call_args.args[0]
    assert struct.unpack('<IIII', request[:16]) == (len(request), 1, 8, 1)
    assert plistlib.loads(request[16:])['MessageType'] == 'ListDevices'
    writer.wait_closed.assert_awaited_once()


async def test_reverse_usb_tunnel_transfers_large_payload_and_closes(monkeypatch):
    import transiroyt.usb_workers as usb_module
    from transiroyt.usb_workers import USBWorkers, copy_stream

    # Simulate the iOS USB listener, then connect a local client through it.
    accepted = asyncio.Queue()
    server = await asyncio.start_server(lambda r, w: accepted.put_nowait((r, w)), '127.0.0.1', 0)
    async def connect(device):
        return await asyncio.open_connection('127.0.0.1', server.sockets[0].getsockname()[1])
    monkeypatch.setattr(usb_module, 'usb_connect', connect)
    monkeypatch.setattr(usb_module, 'usb_devices', AsyncMock(return_value=[7]))
    async def echo(reader, writer):
        try:
            await copy_stream(reader, writer)
        finally:
            writer.close()
    target = await asyncio.start_server(echo, '127.0.0.1', 0)
    bridge = USBWorkers(target.sockets[0].getsockname()[1])
    try:
        reader, writer = await asyncio.wait_for(accepted.get(), 2)
        writer.write(b'\x01')
        payload = bytes(range(256)) * 4096
        writer.write(payload)
        await writer.drain()
        assert await asyncio.wait_for(reader.readexactly(len(payload)), 5) == payload
        writer.close()
        await writer.wait_closed()
        await bridge.close()
        assert not bridge.tasks
    finally:
        await bridge.close()
        while not accepted.empty():
            _, writer = accepted.get_nowait()
            writer.close()
            await writer.wait_closed()
        server.close()
        target.close()
        await server.wait_closed()
        await target.wait_closed()


async def test_worker_server_keeps_desktop_access_log(monkeypatch, tmp_path):
    import logging
    import ssl
    import uuid
    from urllib.parse import parse_qs, urlsplit

    access = logging.getLogger('uvicorn.access')
    records = []
    handler = logging.Handler()
    handler.emit = records.append
    monkeypatch.setattr(access, 'handlers', [handler])
    monkeypatch.setattr(access, 'propagate', False)
    level = access.level
    access.setLevel(logging.INFO)
    pool = MobileWorkers(tmp_path)
    try:
        link = urlsplit((await pool.enable('usb'))['pairing_url'])
        context = ssl.create_default_context(cafile=str(tmp_path / 'mobile-workers/certificate.pem'))
        context.check_hostname = False
        async with httpx.AsyncClient(base_url=f'https://127.0.0.1:{link.port}', verify=context) as client:
            response = await client.post('/pair', json={'code': parse_qs(link.fragment)['code'][0],
                                                       'name': 'iPhone', 'device_id': str(uuid.uuid4())})
            assert response.status_code == 200
        assert access.handlers == [handler] and access.getEffectiveLevel() == logging.INFO
        access.info('%s - "%s %s HTTP/%s" %d', '127.0.0.1:1', 'GET', '/api/mobile', '1.1', 200)
        assert [record.args[2] for record in records] == ['/api/mobile']
    finally:
        await pool.close()
        access.removeFilter(mobile_workers.WORKER_ACCESS_FILTER)
        access.setLevel(level)


async def test_paired_device_reconnects_after_restart_until_stopped(tmp_path):
    import ssl
    import uuid
    from urllib.parse import parse_qs, urlsplit

    from transiroyt.mobile_workers import TTS_IDENTITY

    pool = MobileWorkers(tmp_path)
    link = urlsplit((await pool.enable('usb'))['pairing_url'])
    context = ssl.create_default_context(cafile=str(tmp_path / 'mobile-workers/certificate.pem'))
    context.check_hostname = False
    base = f'https://127.0.0.1:{link.port}'
    try:
        async with httpx.AsyncClient(base_url=base, verify=context) as client:
            response = await client.post('/pair', json={'code': parse_qs(link.fragment)['code'][0],
                                                       'name': 'iPhone', 'device_id': str(uuid.uuid4())})
        headers = {'Authorization': f"Bearer {response.json()['token']}"}
    finally:
        await pool.close()
    assert (tmp_path / 'mobile-workers/devices.json').stat().st_mode & 0o777 == 0o600

    restarted = MobileWorkers(tmp_path)
    try:
        await restarted.resume()
        assert restarted.public()['enabled'] and restarted.port == link.port
        assert restarted.public()['devices'][0]['online'] is False
        async with httpx.AsyncClient(base_url=base, verify=context) as client:
            heartbeat = await client.post('/heartbeat', headers=headers, json={
                'capabilities': {'tts': TTS_IDENTITY}, 'state': '待機中', 'accepting': True})
        assert heartbeat.status_code == 200 and heartbeat.json()['tts_identity'] == TTS_IDENTITY
        assert restarted.public()['devices'][0]['online'] is True
    finally:
        await restarted.stop()

    stopped = MobileWorkers(tmp_path)
    await stopped.resume()
    assert not stopped.public()['enabled'] and stopped.public()['devices'] == []


async def test_usb_tunnels_run_after_a_later_wifi_qr_code(monkeypatch, tmp_path):
    """A device paired over USB keeps working after the latest QR code was for Wi-Fi."""
    import transiroyt.usb_workers as usb_module

    monkeypatch.setattr(usb_module, 'usb_devices', AsyncMock(return_value=[]))
    pool = MobileWorkers(tmp_path)
    monkeypatch.setattr(pool, '_pairing_address', AsyncMock(return_value='192.168.0.2'))
    try:
        await pool.enable('usb')
        pool.devices['usb'] = mobile_workers.Device('usb', 'USB iPhone', 'token')
        await pool.enable('wifi')
        assert pool.usb is not None
    finally:
        await pool.close()
    assert pool.usb is None

    restarted = MobileWorkers(tmp_path)
    try:
        await restarted.resume()
        assert restarted.transport == 'wifi' and restarted.usb is not None
    finally:
        await restarted.close()
