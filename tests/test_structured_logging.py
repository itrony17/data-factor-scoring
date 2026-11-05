"""Tests for boundary validation, structured logging and doctor --json."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from dqfs.cli import main as cli_main
from dqfs.logging_setup import build_json_formatter, setup_logging


# ------------------------------------------------------- structured logging
def _records_to_json(record: logging.LogRecord, fmt: logging.Formatter) -> str:
    return fmt.format(record)


def test_json_formatter_includes_structured_extras() -> None:
    setup_logging(verbose=1, log_json=True)
    fmt = build_json_formatter()
    rec = logging.LogRecord(
        "dqfs.test", logging.INFO, __file__, 1,
        "scoring done", (), None,
    )
    rec.run_id = "20260101_000000"  # type: ignore[attr-defined]
    rec.verdict = "WARN"  # type: ignore[attr-defined]
    rec.dqs = 92.0  # type: ignore[attr-defined]
    payload = json.loads(fmt.format(rec))
    assert payload["levelname"] == "INFO"
    assert payload["message"] == "scoring done"
    assert payload["run_id"] == "20260101_000000"
    assert payload["verdict"] == "WARN"
    assert payload["dqs"] == 92.0


def test_json_logging_via_cli_flag(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """--log-json must emit one valid JSON object per stderr line."""
    demo_dir = tmp_path / "demo"
    assert cli_main(["demo", "--out", str(demo_dir)]) == 0
    assert cli_main([
        "--log-json", "-v",
        "score",
        "--config", str(demo_dir / "dqfs-demo.yaml"),
        "--dsn", f"sqlite:///{demo_dir / 'quality_demo.db'}",
        "--table", "customers",
        "--out", str(tmp_path / "out"),
    ]) in (0, 7)
    lines = [ln for ln in capsys.readouterr().err.splitlines() if ln.strip()]
    assert lines, "expected at least one JSON log line"
    for line in lines:
        payload = json.loads(line)  # raises if any line is not valid JSON
        assert "levelname" in payload and "message" in payload


