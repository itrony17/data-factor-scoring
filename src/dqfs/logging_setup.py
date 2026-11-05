"""Structured logging setup for DQFS.

Provides consistent console logging with two modes:

* human-readable ``KEY=value`` records (default);
* one-JSON-object-per-line records (``--log-json``) for log shippers.

Levels: default ``WARNING``, ``-v`` -> ``INFO``, ``-vv`` -> ``DEBUG``.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s %(message)s"
_DATEFMT = "%Y-%m-%dT%H:%M:%S%z"


class JsonFormatter(logging.Formatter):
    """Format log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, _DATEFMT),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def setup_logging(verbose: int = 0, log_json: bool = False) -> None:
    """Configure the root logger for the CLI.

    Args:
        verbose: 0 -> WARNING, 1 -> INFO, >=2 -> DEBUG.
        log_json: Emit one JSON object per record instead of plain text.
    """
    level = (logging.WARNING, logging.INFO, logging.DEBUG)[min(verbose, 2)]
    handler = logging.StreamHandler(sys.stderr)
    if log_json:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
    # Third-party noise down.
    for noisy in ("urllib3", "matplotlib", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
