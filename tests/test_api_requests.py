import asyncio
from types import SimpleNamespace

import httpx
import pytest
from transiroyt import api_requests
from transiroyt.api_requests import APIRequests


@pytest.fixture
def clock(monkeypatch):
    state = SimpleNamespace(now=100.0, waits=[])
    async def sleep(seconds):
        state.waits.append(seconds)
        state.now += seconds
    monkeypatch.setattr(api_requests, 'time', SimpleNamespace(monotonic=lambda: state.now, time=lambda: 0))
    monkeypatch.setattr(api_requests.asyncio, 'sleep', sleep)
    return state


@pytest.mark.parametrize('retry_after,wait', [('7', 7), ('Thu, 01 Jan 1970 00:00:09 GMT', 9),
                                           ('invalid', 1), ('NaN', 1), ('-2', 1), ('inf', 1)])
async def test_retry_after_then_recovery(clock, retry_after, wait):
    calls = []
    def handle(request):
        calls.append(request)
        return httpx.Response(429, headers={'Retry-After': retry_after}) if len(calls) == 1 else httpx.Response(200)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        response = await APIRequests().request(client, 'POST', 'https://example.test')
    assert response.status_code == 200 and len(calls) == 2
    assert clock.waits == [wait]


async def test_retry_limit_and_cooldown_across_requests(clock):
    calls = []
    def handle(request):
        calls.append(request)
        return httpx.Response(429)
    requests = APIRequests()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        assert (await requests.request(client, 'GET', 'https://example.test')).status_code == 429
        assert len(calls) == 3 and clock.waits == [1, 2]
        with pytest.raises(RuntimeError, match='解除待ち'):
            await requests.request(client, 'GET', 'https://example.test')
        assert len(calls) == 3
        clock.now += 4
        assert (await requests.request(client, 'GET', 'https://example.test')).status_code == 429
        assert len(calls) == 6


async def test_long_wait_falls_back_without_retrying(clock):
    requests = APIRequests()
    async with httpx.AsyncClient(transport=httpx.MockTransport(
            lambda request: httpx.Response(429, headers={'Retry-After': '3600'}))) as client:
        assert (await requests.request(client, 'POST', 'https://example.test')).status_code == 429
        assert clock.waits == [] and requests.retry_at == 3700


@pytest.mark.parametrize('status', [200, 400, 401, 403, 404, 500, 503])
async def test_other_statuses_are_returned_without_retry(clock, status):
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(status))) as client:
        assert (await APIRequests().request(client, 'GET', 'https://example.test')).status_code == status
        assert clock.waits == []


async def test_cancellation_during_retry_propagates(monkeypatch):
    async def cancelled(seconds):
        raise asyncio.CancelledError()
    monkeypatch.setattr(api_requests.asyncio, 'sleep', cancelled)
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(429))) as client:
        with pytest.raises(asyncio.CancelledError):
            await APIRequests().request(client, 'POST', 'https://example.test')
