"""Exercise the real worker protocol, scheduler and lease recovery without models."""
import asyncio
import base64
import hashlib
import io
import tempfile
import time
import uuid
import wave
from pathlib import Path

import httpx
import pytest
from transiroyt.mobile_workers import (
    DEVICE_SLOTS,
    HEARTBEAT_TIMEOUT,
    TTS_IDENTITY,
    Assignment,
    MobileWorkers,
)


async def connect(pool, client, name='iPhone'):
    response = await client.post('/pair', json={'code': pool.code, 'name': name, 'device_id': str(uuid.uuid4())})
    assert response.status_code == 200
    headers = {'Authorization': f"Bearer {response.json()['token']}"}
    await client.post('/heartbeat', headers=headers, json={
        'capabilities': {'tts': TTS_IDENTITY}, 'state': '待機中', 'accepting': True})
    return response.json()['id'], headers


@pytest.fixture
async def workers(tmp_path):
    pool = MobileWorkers(tmp_path)
    pool.code = 'test-code'
    pool.expires = time.monotonic() + 120
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=pool.worker_app()), base_url='https://mac') as client:
        yield pool, client
    await pool.close()


async def next_task(client, auth, wait=False):
    """The first task the Mac hands the app, or None."""
    tasks = (await client.get('/task' + ('?wait=true' if wait else ''), headers=auth)).json()['tasks']
    return tasks[0] if tasks else None


def done(task_id, **extra):
    return {'task_id': task_id, 'wav': base64.b64encode(wav_data()).decode(), 'elapsed_ms': 100, **extra}


def occupy(device, count=DEVICE_SLOTS):
    """Fill a device with other lines' assignments, as during lookahead."""
    loop = asyncio.get_running_loop()
    for n in range(count):
        device.assignments[f'busy{n}'] = Assignment(f'busy{n}', 'tts', {'reference_sha256': 'x'}, loop.create_future())


def speech(text):
    return {'text': text, 'reference_path': str(REFERENCE)}


def wav_data():
    output = io.BytesIO()
    with wave.open(output, 'wb') as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(48000)
        audio.writeframes(bytes(9600))
    return output.getvalue()


REFERENCE = Path(tempfile.mkdtemp()) / 'reference.wav'
REFERENCE.write_bytes(wav_data())


async def test_pair_limit_auth_and_revoke(workers):
    pool, client = workers
    assert (await client.get('/task')).status_code == 401
    first, auth = await connect(pool, client)
    await connect(pool, client, 'iPad')
    assert (await client.post('/pair', json={'code': pool.code, 'name': 'third', 'device_id': str(uuid.uuid4())})).status_code == 409
    pool.remove(first)
    assert (await client.get('/task', headers=auth)).status_code == 401
    await connect(pool, client, 'replacement')
    pool.expires = 0
    assert (await client.post('/pair', json={'code': pool.code, 'name': 'expired', 'device_id': str(uuid.uuid4())})).status_code == 401


async def test_two_remote_jobs_and_local_overflow(workers):
    pool, client = workers
    _, auth1 = await connect(pool, client)
    _, auth2 = await connect(pool, client, 'iPad')
    tasks = [asyncio.create_task(pool.execute('tts', speech(f'line{i}'))) for i in range(3)]
    await asyncio.sleep(0)
    assert await tasks[2] is None
    ids = []
    for auth in (auth1, auth2):
        assignment = await next_task(client, auth)
        ids.append(assignment['id'])
        result = await client.post('/result', headers=auth, json=done(assignment['id']))
        assert result.status_code == 200
    assert ids[0] != ids[1]
    assert await asyncio.gather(*tasks[:2]) == [wav_data(), wav_data()]
    assert sum(d.completed for d in pool.devices.values()) == 2


@pytest.mark.parametrize('failure', ['offline', 'pause', 'removed', 'invalid_audio', 'error', 'cancel'])
async def test_reclaims_failed_or_cancelled_lease_and_rejects_late_result(workers, failure):
    pool, client = workers
    device_id, auth = await connect(pool, client)
    task = asyncio.create_task(pool.execute('tts', speech('x')))
    await asyncio.sleep(0)
    assignment = await next_task(client, auth)
    if failure == 'offline':
        pool.devices[device_id].seen -= HEARTBEAT_TIMEOUT + 1
    elif failure == 'pause':
        await client.post('/heartbeat', headers=auth, json={'state': 'paused', 'accepting': False})
    elif failure == 'removed':
        pool.remove(device_id)
    elif failure == 'cancel':
        task.cancel()
    else:
        await client.post('/result', headers=auth, json={
            'task_id': assignment['id'], **({'wav': 'bad'} if failure == 'invalid_audio' else {'error': 'model failed'})})
    if failure == 'cancel':
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        assert await asyncio.wait_for(task, 2) is None
    late = await client.post('/result', headers=auth, json={'task_id': assignment['id'], 'wav': ''})
    assert late.status_code == (401 if failure == 'removed' else
                                200 if failure in ('invalid_audio', 'error') else 409)
    assert not any(d.assignments for d in pool.devices.values())


async def test_incompatible_or_busy_device_is_not_selected(workers):
    pool, client = workers
    device_id, _ = await connect(pool, client)
    pool.devices[device_id].capabilities['tts'] = 'different-model'
    assert await pool.execute('tts', speech('x')) is None


async def test_measured_slow_worker_yields_to_free_mac(workers):
    pool, client = workers
    device_id, _ = await connect(pool, client)
    pool.local_costs['tts'] = 1
    pool.devices[device_id].costs['tts'] = 1.5
    assert await pool.execute('tts', speech('テスト')) is None
    # A busy Mac raises the completion-time estimate, allowing the same worker.
    pool.local_active['tts'] = 1
    task = asyncio.create_task(pool.execute('tts', speech('テスト')))
    await asyncio.sleep(0)
    assert pool.devices[device_id].assignments
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_local_measurement_releases_on_cancellation(workers):
    pool, _ = workers
    async with pool.local_work('tts', {'text': 'テスト'}):
        assert pool.local_active['tts'] == 1
    cost = pool.local_costs['tts']
    assert pool.local_active['tts'] == 0 and cost > 0
    with pytest.raises(asyncio.CancelledError):
        async with pool.local_work('tts', {'text': 'テスト'}):
            raise asyncio.CancelledError
    assert pool.local_active['tts'] == 0 and pool.local_costs['tts'] == cost


async def test_result_retry_is_acknowledged_without_duplicate_completion(workers):
    pool, client = workers
    device_id, auth = await connect(pool, client)
    task = asyncio.create_task(pool.execute('tts', speech('test')))
    await asyncio.sleep(0)
    assignment = await next_task(client, auth)
    body = done(assignment['id'])
    assert (await client.post('/result', headers=auth, json=body)).status_code == 200
    assert await task == wav_data()
    assert (await client.post('/result', headers=auth, json=body)).status_code == 200
    assert pool.devices[device_id].completed == 1


async def test_failure_cooldown_survives_heartbeat_and_then_recovers(workers):
    pool, client = workers
    device_id, auth = await connect(pool, client)
    task = asyncio.create_task(pool.execute('tts', speech('x')))
    await asyncio.sleep(0)
    assignment = await next_task(client, auth)
    await client.post('/result', headers=auth, json={'task_id': assignment['id'], 'error': 'failed'})
    assert await task is None
    await client.post('/heartbeat', headers=auth, json={
        'capabilities': {'tts': TTS_IDENTITY}, 'state': 'ready', 'accepting': True})
    assert await pool.execute('tts', speech('x')) is None
    pool.devices[device_id].retry_after = time.monotonic() - 1
    retry = asyncio.create_task(pool.execute('tts', speech('x')))
    await asyncio.sleep(0)
    assert pool.devices[device_id].assignments
    retry.cancel()
    with pytest.raises(asyncio.CancelledError):
        await retry


async def test_tts_queue_moves_to_phone_when_it_becomes_free(workers):
    pool, client = workers
    device_id, auth = await connect(pool, client)
    device = pool.devices[device_id]
    # Other lines fill the phone, as happens during lookahead.
    occupy(device)
    entered, release = asyncio.Event(), asyncio.Event()
    local_calls = []

    async def local_first():
        local_calls.append('first')
        entered.set()
        await release.wait()

    async def local_second():
        local_calls.append('second')

    first = asyncio.create_task(pool.tts.run(speech('first'), local_first))
    await entered.wait()
    second = asyncio.create_task(pool.tts.run(speech('second'), local_second))
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert not second.done()
    device.assignments.clear()
    await client.post('/heartbeat', headers=auth, json={
        'capabilities': {'tts': TTS_IDENTITY}, 'state': 'ready', 'accepting': True})
    await asyncio.sleep(0)
    assignment = await next_task(client, auth)
    assert assignment['payload']['text'] == 'second'
    await client.post('/result', headers=auth, json=done(assignment['id']))
    assert await second == wav_data()
    assert local_calls == ['first']
    release.set()
    assert await first is None


async def test_tts_plan_uses_backlog_instead_of_rejecting_slower_phone(workers):
    pool, client = workers
    device_id, auth = await connect(pool, client)
    pool.local_costs['tts'] = 1
    pool.devices[device_id].costs['tts'] = 2.5
    release = asyncio.Event()
    local_calls = []

    async def local():
        local_calls.append(True)
        await release.wait()

    tasks = [asyncio.create_task(pool.tts.run(speech('x' * 10), local)) for _ in range(4)]
    for _ in range(4):
        await asyncio.sleep(0)
    # Three jobs on Mac take 30s; one on the phone takes 25s. The old 2x
    # threshold rejected this phone despite the three jobs waiting on Mac.
    assignment = await next_task(client, auth)
    assert assignment is not None
    assert len(local_calls) == 1
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    await asyncio.sleep(0)
    assert not pool.tts.pending and not pool.tts.running


async def test_tts_single_job_chooses_faster_mac(workers):
    pool, client = workers
    device_id, _ = await connect(pool, client)
    pool.local_costs['tts'] = 1
    pool.devices[device_id].costs['tts'] = 2
    calls = []

    async def local():
        calls.append(True)

    assert await pool.tts.run(speech('hello'), local) is None
    assert calls == [True]
    assert pool.devices[device_id].completed == 0


async def test_tts_two_mac_slots_and_two_phones_share_four_jobs(workers):
    pool, client = workers
    _, auth1 = await connect(pool, client)
    _, auth2 = await connect(pool, client, 'iPad')
    pool.tts.local_workers = 2
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def local():
        calls.append(True)
        if len(calls) == 2:
            entered.set()
        await release.wait()

    tasks = [asyncio.create_task(pool.tts.run(speech('test'), local)) for _ in range(4)]
    try:
        await asyncio.wait_for(entered.wait(), 2)
        assert set(pool.tts.running) == {'mac', 'mac-2', *pool.devices}
        for auth in (auth1, auth2):
            assignment = await next_task(client, auth)
            assert assignment is not None
            await client.post('/result', headers=auth, json=done(assignment['id']))
        release.set()
        await asyncio.wait_for(asyncio.gather(*tasks), 2)
        assert len(calls) == 2
        assert sum(d.completed for d in pool.devices.values()) == 2
        assert not pool.tts.running and not pool.tts.pending
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def test_tts_failure_replans_to_mac(workers):
    pool, client = workers
    device_id, auth = await connect(pool, client)
    pool.local_costs['tts'] = 2
    pool.devices[device_id].costs['tts'] = 1
    calls = []

    async def local():
        calls.append(True)

    task = asyncio.create_task(pool.tts.run(speech('hello'), local))
    for _ in range(4):
        await asyncio.sleep(0)
    assignment = await next_task(client, auth)
    await client.post('/result', headers=auth, json={'task_id': assignment['id'], 'error': 'failed'})
    assert await asyncio.wait_for(task, 2) is None
    assert calls == [True] and pool.devices[device_id].failures == 1


async def test_tts_waits_for_fast_mac_instead_of_starting_slow_phone(workers):
    pool, client = workers
    device_id, _ = await connect(pool, client)
    pool.local_costs['tts'] = 1
    pool.devices[device_id].costs['tts'] = 10
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def local():
        calls.append(True)
        entered.set()
        await release.wait()

    first = asyncio.create_task(pool.tts.run(speech('x'), local))
    await entered.wait()
    second = asyncio.create_task(pool.tts.run(speech('x'), local))
    for _ in range(4):
        await asyncio.sleep(0)
    assert not pool.devices[device_id].assignments
    assert len(calls) == 1
    release.set()
    await asyncio.gather(first, second)
    assert len(calls) == 2


async def test_tts_cancel_before_execution_releases_slot(workers):
    pool, _ = workers
    calls = []

    async def local():
        calls.append(True)

    task = asyncio.create_task(pool.tts.run(speech('x'), local))
    await asyncio.sleep(0)
    # The dispatcher runs next; cancel the caller before the newly created
    # executor gets its first turn.
    asyncio.get_running_loop().call_soon(task.cancel)
    await asyncio.gather(task, return_exceptions=True)
    for _ in range(4):
        await asyncio.sleep(0)
    assert not pool.tts.running and not pool.tts.pending
    await pool.tts.run(speech('next'), local)
    assert calls


async def test_long_poll_wakes_on_assignment(workers):
    pool, client = workers
    _, auth = await connect(pool, client)
    waiting = asyncio.create_task(client.get('/task?wait=true', headers=auth))
    await asyncio.sleep(0)
    task = asyncio.create_task(pool.execute('tts', speech('test')))
    response = await asyncio.wait_for(waiting, 1)
    assert response.json()['tasks'][0]['payload']['text'] == 'test'
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)


async def test_tts_two_phones_share_the_queue(workers):
    pool, client = workers
    _, auth1 = await connect(pool, client)
    _, auth2 = await connect(pool, client, 'iPad')
    release = asyncio.Event()

    async def local():
        await release.wait()

    tasks = [asyncio.create_task(pool.tts.run(speech('test'), local)) for _ in range(3)]
    for _ in range(4):
        await asyncio.sleep(0)
    for auth in (auth1, auth2):
        assignment = await next_task(client, auth)
        assert assignment is not None
        await client.post('/result', headers=auth, json=done(assignment['id']))
    release.set()
    await asyncio.wait_for(asyncio.gather(*tasks), 2)
    assert sum(d.completed for d in pool.devices.values()) == 2


async def test_tts_cancellation_waits_for_local_cleanup(workers):
    pool, _ = workers
    entered, cleaned = asyncio.Event(), asyncio.Event()

    async def local():
        try:
            entered.set()
            await asyncio.Event().wait()
        finally:
            await asyncio.sleep(0)
            cleaned.set()

    task = asyncio.create_task(pool.tts.run(speech('test'), local))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cleaned.is_set()
    assert not pool.tts.running and not pool.tts.pending


async def test_tts_replans_if_another_task_claims_phone_after_planning(workers):
    pool, client = workers
    device_id, _ = await connect(pool, client)
    pool.local_costs['tts'] = 2
    pool.devices[device_id].costs['tts'] = 1
    calls = []

    async def local():
        calls.append(True)

    task = asyncio.create_task(pool.tts.run(speech('test'), local))
    await asyncio.sleep(0)
    # Run after planning, before the scheduled TTS executor starts.
    asyncio.get_running_loop().call_soon(occupy, pool.devices[device_id])
    assert await asyncio.wait_for(task, 2) is None
    assert set(pool.devices[device_id].assignments) == {f'busy{n}' for n in range(DEVICE_SLOTS)}
    assert calls == [True]


async def test_heartbeat_reclaims_delivered_job_abandoned_by_worker(workers):
    pool, client = workers
    device_id, auth = await connect(pool, client)
    task = asyncio.create_task(pool.execute('tts', speech('test')))
    await asyncio.sleep(0)
    assignment = await next_task(client, auth)
    await client.post('/heartbeat', headers=auth, json={
        'state': 'working', 'capabilities': {'tts': TTS_IDENTITY}, 'tasks': [assignment['id']]})
    assert not task.done()
    # The result upload failed and the worker returned to its idle loop.
    await client.post('/heartbeat', headers=auth, json={
        'state': 'idle', 'capabilities': {'tts': TTS_IDENTITY}, 'tasks': []})
    assert await asyncio.wait_for(task, 1) is None
    assert pool.devices[device_id].failures == 1
    assert (await client.post('/result', headers=auth, json=done(assignment['id']))).status_code == 409


async def test_heartbeat_before_delivery_does_not_reclaim_assignment(workers):
    pool, client = workers
    _, auth = await connect(pool, client)
    task = asyncio.create_task(pool.execute('tts', speech('test')))
    await asyncio.sleep(0)
    await client.post('/heartbeat', headers=auth, json={
        'state': 'idle', 'capabilities': {'tts': TTS_IDENTITY}, 'tasks': []})
    assert not task.done()
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)


async def test_cancelled_phone_is_not_reused_until_native_job_stops(workers):
    pool, client = workers
    device_id, auth = await connect(pool, client)
    task = asyncio.create_task(pool.execute('tts', speech('test')))
    await asyncio.sleep(0)
    assignment = await next_task(client, auth)
    await client.post('/heartbeat', headers=auth, json={
        'state': 'working', 'capabilities': {'tts': TTS_IDENTITY}, 'tasks': [assignment['id']]})
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    assert await pool.execute('tts', speech('next')) is None
    assert pool.devices[device_id] not in pool.tts_devices()
    await client.post('/heartbeat', headers=auth, json={
        'state': 'idle', 'capabilities': {'tts': TTS_IDENTITY}, 'tasks': []})
    assert pool.devices[device_id] in pool.tts_devices()


async def test_first_use_of_a_reference_is_not_a_speed_sample(workers, tmp_path):
    pool, client = workers
    device_id, auth = await connect(pool, client)
    device = pool.devices[device_id]
    payload = speech('テスト')
    for _ in range(2):
        task = asyncio.create_task(pool.execute_on(device, 'tts', payload))
        await asyncio.sleep(0)
        assignment = await next_task(client, auth)
        await client.post('/result', headers=auth, json=done(
            assignment['id'], references=[assignment['payload']['reference_sha256']]))
        assert await task == wav_data()
        # The first task carried the reference; only the second measures speed.
        assert ('tts' in device.costs) is ('reference' not in assignment['payload'])
    assert 'tts' in device.costs

    for _ in range(2):
        async with pool.local_work('tts', payload):
            pass
    assert pool.local_costs['tts'] >= 0
    other = {**payload, 'reference_path': str(tmp_path / 'other.wav')}
    before = pool.local_costs['tts']
    async with pool.local_work('tts', other):
        await asyncio.sleep(0.05)
    assert pool.local_costs['tts'] == before


async def test_task_cancelled_after_delivery_is_dropped_by_the_app(workers):
    pool, client = workers
    device_id, auth = await connect(pool, client)
    task = asyncio.create_task(pool.execute('tts', speech('first')))
    await asyncio.sleep(0)
    assignment = await next_task(client, auth)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    # The app still holds it: no new work until the Mac's list tells it to drop it.
    reply = await client.post('/heartbeat', headers=auth, json={
        'capabilities': {'tts': TTS_IDENTITY}, 'state': 'working', 'accepting': True, 'tasks': [assignment['id']]})
    assert reply.json()['task_ids'] == [] and pool.devices[device_id] not in pool.tts_devices()
    await client.post('/heartbeat', headers=auth, json={
        'capabilities': {'tts': TTS_IDENTITY}, 'state': 'ready', 'accepting': True, 'tasks': []})
    assert pool.devices[device_id] in pool.tts_devices()


async def test_reference_is_sent_once_to_a_device_that_keeps_it(workers, tmp_path):
    pool, client = workers
    device_id, auth = await connect(pool, client)
    reference = tmp_path / 'reference.wav'
    reference.write_bytes(wav_data())
    digest = hashlib.sha256(wav_data()).hexdigest()
    payload = {'text': 'テスト', 'voice': 'v', 'steps': 10, 'reference_path': str(reference)}

    async def local():
        raise AssertionError('the phone is idle and faster')

    pool.local_costs['tts'] = 10
    pool.devices[device_id].costs['tts'] = 1
    for expected in (True, False):
        task = asyncio.create_task(pool.tts.run(payload, local))
        for _ in range(3):
            await asyncio.sleep(0)
        assignment = await next_task(client, auth)
        assert assignment['payload']['reference_sha256'] == digest
        assert ('reference' in assignment['payload']) is expected
        assert 'reference_path' not in assignment['payload']
        await client.post('/result', headers=auth, json=done(assignment['id']))
        assert await task == wav_data()
        # The app lists the reference it now keeps.
        await client.post('/heartbeat', headers=auth, json={
            'capabilities': {'tts': TTS_IDENTITY}, 'state': 'ready', 'accepting': True,
            'tasks': [], 'references': [digest]})
    assert pool.devices[device_id].references == {digest}


async def test_result_makes_reporting_device_available_without_idle_heartbeat(workers):
    pool, client = workers
    device_id, auth = await connect(pool, client)
    await client.post('/heartbeat', headers=auth, json={
        'capabilities': {'tts': TTS_IDENTITY}, 'state': 'ready', 'accepting': True, 'tasks': []})
    task = asyncio.create_task(pool.execute('tts', speech('x')))
    await asyncio.sleep(0)
    assignment = await next_task(client, auth)
    await client.post('/heartbeat', headers=auth, json={
        'capabilities': {'tts': TTS_IDENTITY}, 'state': 'working', 'accepting': True, 'tasks': [assignment['id']]})
    await client.post('/result', headers=auth, json=done(assignment['id']))
    assert pool.devices[device_id].held == []
    await task
    assert pool.devices[device_id] in pool.tts_devices()


async def test_tts_queue_runs_earlier_lines_first(workers):
    pool, _ = workers
    release = asyncio.Event()
    order = []

    def local(name):
        async def run():
            order.append(name)
            await release.wait()
        return run

    busy = asyncio.create_task(pool.tts.run(speech('x'), local('busy'), priority=0))
    await asyncio.sleep(0)
    later = asyncio.create_task(pool.tts.run(speech('x'), local('line 9'), priority=9))
    earlier = asyncio.create_task(pool.tts.run(speech('x'), local('line 2'), priority=2))
    for _ in range(3):
        await asyncio.sleep(0)
    assert order == ['busy']
    release.set()
    await asyncio.gather(busy, later, earlier)
    assert order == ['busy', 'line 2', 'line 9']


async def test_tts_plan_sees_the_whole_queue_beyond_four_jobs(workers):
    pool, client = workers
    _, auth = await connect(pool, client)
    pool.local_costs['tts'] = 1
    pool.devices[next(iter(pool.devices))].costs['tts'] = 5
    release = asyncio.Event()

    async def local():
        await release.wait()

    # Four equal jobs finish sooner on the Mac alone (4 s against 5 s), but
    # the fifth and later ones would wait longer there than on the phone.
    tasks = [asyncio.create_task(pool.tts.run(speech('x'), local)) for _ in range(8)]
    for _ in range(3):
        await asyncio.sleep(0)
    assert await next_task(client, auth) is not None
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


async def test_device_holds_three_lines_and_receives_them_together(workers):
    pool, client = workers
    device_id, auth = await connect(pool, client)
    device = pool.devices[device_id]
    tasks = [asyncio.create_task(pool.execute_on(device, 'tts', speech(f'line{n}'))) for n in range(DEVICE_SLOTS)]
    await asyncio.sleep(0)
    assert not pool.has_slot(device)
    received = (await client.get('/task', headers=auth)).json()['tasks']
    assert [t['payload']['text'] for t in received] == [f'line{n}' for n in range(DEVICE_SLOTS)]
    # Delivered once: the next poll has nothing new.
    assert (await client.get('/task', headers=auth)).json()['tasks'] == []
    for item in received:
        await client.post('/result', headers=auth, json=done(item['id']))
    assert await asyncio.gather(*tasks) == [wav_data()] * DEVICE_SLOTS
    assert pool.has_slot(device) and device.completed == DEVICE_SLOTS


async def test_scheduler_keeps_a_fast_phone_queued_while_it_generates(workers):
    pool, client = workers
    device_id, auth = await connect(pool, client)
    # Lines take 1 s on the phone and 3.5 s on the Mac: the fourth would end later on the phone.
    pool.local_costs['tts'] = 3.5
    pool.devices[device_id].costs['tts'] = 1
    local_calls = []

    async def local():
        local_calls.append(True)
        await asyncio.Event().wait()

    tasks = [asyncio.create_task(pool.tts.run(speech('x'), local)) for _ in range(5)]
    for _ in range(4):
        await asyncio.sleep(0)
    # Three lines wait on the phone; the Mac takes the next one.
    assert len(pool.devices[device_id].assignments) == DEVICE_SLOTS
    assert len((await client.get('/task', headers=auth)).json()['tasks']) == DEVICE_SLOTS
    assert local_calls == [True]
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


async def test_speed_comes_from_the_devices_generation_time_not_its_queue(workers):
    pool, client = workers
    device_id, auth = await connect(pool, client)
    device = pool.devices[device_id]
    device.references.add(hashlib.sha256(wav_data()).hexdigest())
    task = asyncio.create_task(pool.execute_on(device, 'tts', speech('テスト')))
    await asyncio.sleep(0)
    assignment = await next_task(client, auth)
    await asyncio.sleep(0.05)
    await client.post('/result', headers=auth, json=done(assignment['id'], elapsed_ms=2000))
    assert await task == wav_data()
    assert device.seconds['tts'] == 2.0 and device.costs['tts'] == pytest.approx(2.0 / 3)


async def test_result_without_generation_time_is_a_failure(workers):
    pool, client = workers
    device_id, auth = await connect(pool, client)
    task = asyncio.create_task(pool.execute('tts', speech('x')))
    await asyncio.sleep(0)
    assignment = await next_task(client, auth)
    body = done(assignment['id'])
    del body['elapsed_ms']
    await client.post('/result', headers=auth, json=body)
    assert await task is None and pool.devices[device_id].failures == 1


async def test_one_queued_line_carries_a_new_reference_and_hands_it_on_if_withdrawn(workers):
    pool, client = workers
    device_id, auth = await connect(pool, client)
    device = pool.devices[device_id]
    first = asyncio.create_task(pool.execute_on(device, 'tts', speech('first')))
    second = asyncio.create_task(pool.execute_on(device, 'tts', speech('second')))
    await asyncio.sleep(0)
    payloads = [a.payload for a in device.assignments.values()]
    assert ['reference' in p for p in payloads] == [True, False]
    # Withdrawn before delivery: the next line for that voice carries it instead.
    first.cancel()
    await asyncio.gather(first, return_exceptions=True)
    assert 'reference' in next(iter(device.assignments.values())).payload
    second.cancel()
    await asyncio.gather(second, return_exceptions=True)
