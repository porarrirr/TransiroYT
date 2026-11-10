"""Bounded retries and shared cooldown for one API connection."""
import asyncio
import logging
import math
import time
from email.utils import parsedate_to_datetime

logger = logging.getLogger(__name__)


class APIRequests:
    def __init__(self):
        self.retry_at = 0.0

    async def request(self, client, method, url, **kwargs):
        if time.monotonic() < self.retry_at:
            raise RuntimeError('APIのレート制限の解除待ちです。次の使用先で処理するか、時間をおいて再試行してください。')
        waited = 0.0
        for attempt in range(3):
            response = await client.request(method, url, **kwargs)
            if response.status_code != 429:
                return response
            delay = 2 ** attempt
            value = response.headers.get('Retry-After')
            if value is not None:
                try:
                    delay = float(value)
                except ValueError:
                    try:
                        delay = parsedate_to_datetime(value).timestamp() - time.time()
                    except (ValueError, TypeError, OverflowError):
                        pass
            if not math.isfinite(delay) or delay < 0:
                delay = 2 ** attempt
            self.retry_at = max(self.retry_at, time.monotonic() + delay)
            # A long quota wait must not stall every caption. Keep the cooldown
            # even when this request goes on to the next configured engine.
            if attempt == 2 or waited + delay > 60:
                return response
            logger.warning('API rate limited; retry %d/2 after %.1f seconds', attempt + 1, delay)
            await asyncio.sleep(delay)
            waited += delay
        raise AssertionError('unreachable')
