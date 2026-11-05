"""Structured logging setup for DQFS.

Provides consistent console logging with two modes:

* human-readable ``KEY=value`` records (default);
* one-JSON-object-per-line records (``--log-json``) for log shippers.

JSON formatting uses ``python-json-logger`` when installed and falls back
to an equivalent built-in formatter, so the CLI never hard-fails on a
trimmed dependency set.

Levels: default ``WARNING``, ``-v`` -> ``INFO``, ``-vv`` -> ``DEBUG``.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

try:  # python-json-logger >= 3.1 exposes the formatter from the root package
    from pythonjsonlogger.json import JsonFormatter as _PjlFormatter
except ImportError:  # pragma: no cover - exercised on trimmed installs
    try:  # older layouts
        from pythonjsonlogger.jsonlogger import JsonFormatter as _PjlFormatter
    except ImportError:
        _PjlFormatter = None  # type: ignore[assignment,misc]

_JSON_FIELDS = ("asctime", "levelname", "name", "message")
_FORMAT = "%(asctime)s %(levelname)-7s %(name)s %(message)s"
_DATEFMT = "%Y-%m-%dT%H:%M:%S%z"


#: Structured fields merged into every JSON record when present in the
#: record's ``extra`` payload (see scoring.py / publisher.py).
_STRUCTURED_KEYS = ("run_id", "dataset", "verdict", "dqs", "duration_seconds",
                    "rows", "repo", "branch", "commit_sha")


class JsonFormatter(logging.Formatter):
    """Fallback JSON formatter matching python-json-logger's field names."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "asctime": self.formatTime(record, _DATEFMT),
            "levelname": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        for key in _STRUCTURED_KEYS:
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def build_json_formatter() -> logging.Formatter:
    """Return the JSON log formatter.

    Prefers :mod:`pythonjsonlogger` (battle-tested, timestamp/level fields
    standardised) and falls back to the built-in equivalent.
    """
    if _PjlFormatter is not None:
        return _PjlFormatter(fmt=_JSON_FIELDS, datefmt=_DATEFMT)
    return JsonFormatter()


def setup_logging(verbose: int = 0, log_json: bool = False) -> None:
    """Configure the root logger for the CLI.

    Args:
        verbose: 0 -> WARNING, 1 -> INFO, >=2 -> DEBUG.
        log_json: Emit one JSON object per record instead of plain text.
    """
    level = (logging.WARNING, logging.INFO, logging.DEBUG)[min(verbose, 2)]
    handler = logging.StreamHandler(sys.stderr)
    if log_json:
        handler.setFormatter(build_json_formatter())
    else:
        handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
    # Third-party noise down.
    for noisy in ("urllib3", "matplotlib", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
