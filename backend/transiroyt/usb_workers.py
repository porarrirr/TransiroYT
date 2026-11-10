"""Reverse TCP tunnels through macOS usbmuxd; TLS remains end to end."""
import asyncio
import plistlib
import socket
import struct
from contextlib import suppress

USB_PORT = 48765


async def mux_message(reader, writer, message):
    body = plistlib.dumps({'ClientVersionString': 'TransiroYT', 'ProgName': 'TransiroYT',
                          'kLibUSBMuxVersion': 3, **message})
    writer.write(struct.pack('<IIII', len(body) + 16, 1, 8, 1) + body)
    await writer.drain()
    length, version, kind, tag = struct.unpack('<IIII', await reader.readexactly(16))
    if not 16 <= length <= 1024 * 1024 or (version, kind, tag) != (1, 8, 1):
        raise ConnectionError('Invalid usbmuxd response')
    return plistlib.loads(await reader.readexactly(length - 16))


async def usb_devices():
    reader, writer = await asyncio.open_unix_connection('/var/run/usbmuxd')
    try:
        response = await mux_message(reader, writer, {'MessageType': 'ListDevices'})
        return [d['DeviceID'] for d in response['DeviceList'] if d['Properties']['ConnectionType'] == 'USB']
    finally:
        writer.close()
        await writer.wait_closed()


async def usb_connect(device):
    reader, writer = await asyncio.open_unix_connection('/var/run/usbmuxd')
    try:
        response = await mux_message(reader, writer, {'MessageType': 'Connect', 'DeviceID': device,
                                                     'PortNumber': socket.htons(USB_PORT)})
        if response.get('Number') != 0:
            raise ConnectionError('Mobile USB listener is not ready')
        return reader, writer
    except BaseException:
        writer.close()
        await writer.wait_closed()
        raise


async def copy_stream(reader, writer):
    while data := await reader.read(65536):
        writer.write(data)
        await writer.drain()


class USBWorkers:
    def __init__(self, port):
        self.port = port
        self.tasks = {}
        self.monitor = asyncio.create_task(self.run())

    async def run(self):
        try:
            while True:
                try:
                    devices = await asyncio.wait_for(usb_devices(), 5)
                except (TimeoutError, OSError, ValueError, KeyError, asyncio.IncompleteReadError):
                    devices = []
                for device in devices:
                    for slot in range(4):
                        key = (device, slot)
                        if key not in self.tasks or self.tasks[key].done():
                            self.tasks[key] = asyncio.create_task(self.tunnel(device))
                for key in list(self.tasks):
                    if key[0] not in devices:
                        task = self.tasks.pop(key)
                        task.cancel()
                        await asyncio.gather(task, return_exceptions=True)
                await asyncio.sleep(1)
        finally:
            for task in self.tasks.values():
                task.cancel()
            await asyncio.gather(*self.tasks.values(), return_exceptions=True)
            self.tasks.clear()

    async def tunnel(self, device):
        writers, pumps = [], []
        try:
            remote_reader, remote_writer = await asyncio.wait_for(usb_connect(device), 5)
            writers.append(remote_writer)
            # Wait until iOS pairs this idle tunnel with a local HTTPS connection.
            # This byte is outside TLS and is consumed here, before forwarding.
            if await remote_reader.readexactly(1) != b'\x01':
                return
            local_reader, local_writer = await asyncio.open_connection('127.0.0.1', self.port)
            writers.append(local_writer)
            pumps = [asyncio.create_task(copy_stream(remote_reader, local_writer)),
                     asyncio.create_task(copy_stream(local_reader, remote_writer))]
            await asyncio.wait(pumps, return_when=asyncio.FIRST_COMPLETED)
        except (TimeoutError, OSError, asyncio.IncompleteReadError):
            pass
        finally:
            for task in pumps:
                task.cancel()
            await asyncio.gather(*pumps, return_exceptions=True)
            for writer in writers:
                writer.close()
            for writer in writers:
                with suppress(OSError):
                    await writer.wait_closed()

    async def close(self):
        self.monitor.cancel()
        await asyncio.gather(self.monitor, return_exceptions=True)
