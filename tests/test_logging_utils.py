import logging

import pytest
from transiroyt.logging_utils import POLLING_ACCESS_FILTER


@pytest.mark.parametrize(("method", "path", "status", "kept"), [
    ("GET", "/health", 200, False),
    ("GET", "/api/status", 200, False),
    ("GET", "/api/file-dubbing?job=1", 200, False),
    ("GET", "/api/status", 500, True),
    ("POST", "/api/file-dubbing", 200, True),
    ("GET", "/api/mobile", 200, True),
])
def test_only_successful_status_polls_leave_the_access_log(method, path, status, kept):
    record = logging.LogRecord("uvicorn.access", logging.INFO, __file__, 0, '%s - "%s %s HTTP/%s" %d',
                               ("127.0.0.1:1", method, path, "1.1", status), None)
    assert POLLING_ACCESS_FILTER.filter(record) is kept
