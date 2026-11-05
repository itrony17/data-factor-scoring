"""Tests for boundary validation, structured logging and doctor --json."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dqfs.cli import main as cli_main


# --------------------------------------------------------------- doctor --json
def test_doctor_json_output(tmp_path: Path) -> None:
    demo_dir = tmp_path / "demo"
    assert cli_main(["demo", "--out", str(demo_dir)]) == 0
    code = cli_main([
        "doctor",
        "--config", str(demo_dir / "dqfs-demo.yaml"),
        "--dsn", f"sqlite:///{demo_dir / 'quality_demo.db'}",
        "--table", "customers",
        "--json",
    ])
    assert code == 0


def test_doctor_json_is_machine_readable(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    demo_dir = tmp_path / "demo"
    assert cli_main(["demo", "--out", str(demo_dir)]) == 0
    capsys.readouterr()  # discard demo output
    cli_main(["doctor", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["tool"] == "dqfs"
    assert payload["healthy"] is True
    assert payload["critical_passed"] == payload["critical_total"]
    names = {c["name"] for c in payload["checks"]}
    assert any("python" in n for n in names)
    assert any("GITHUB_TOKEN" in n for n in names)
