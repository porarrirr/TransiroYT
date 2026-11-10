"""Keep API and model lifecycle messages on the same wall clock."""
from __future__ import annotations

import logging
from datetime import datetime

FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class PollingAccessFilter(logging.Filter):
    """Keep the app's successful 3-second status polls out of the access log.

    They were 96% of a 58 MB log after one job, and the in-app log view shows
    only the last 32 KB, which then held about six minutes of polls.
    """

    paths = frozenset({"/health", "/api/status", "/api/file-dubbing"})

    def filter(self, record):
        _, method, path, _, status = record.args
        return not (method == "GET" and path.partition("?")[0] in self.paths and status < 400)


POLLING_ACCESS_FILTER = PollingAccessFilter()


def configure_logging():
    logging.basicConfig(format=FORMAT, datefmt=DATE_FORMAT, level=logging.INFO)
    # httpx INFO messages contain complete URLs, including custom query values.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").addFilter(POLLING_ACCESS_FILTER)
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        for handler in logging.getLogger(name).handlers:
            handler.setFormatter(logging.Formatter(FORMAT, datefmt=DATE_FORMAT))


def log_model_start(log, name):
    log.write(f"\n{datetime.now().astimezone().isoformat(timespec='seconds')} [lifecycle] {name} starting\n")
    log.flush()
