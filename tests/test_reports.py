"""Tests for dqfs.report - JSON, HTML dashboard, Excel and CSV artifacts."""

from __future__ import annotations

import json

import openpyxl
import pandas as pd
import pytest

from dqfs.config import load_config
from dqfs.errors import ReportError
from dqfs.report import write_csv, write_excel, write_html, write_json
from dqfs.scoring import run_scoring


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    """A real scoring run result shared by all report tests."""
    p = tmp_path_factory.mktemp("cfg") / "c.yaml"
    p.write_text("ml:\n  enabled: false\n", encoding="utf-8")
    cfg = load_config(p)
    df = pd.DataFrame(
        {
            "id": list(range(40)),
            "age": [25.0, 34.0, 19.0, 61.0] * 10,
            "income": [50_000.0, 62_500.0, 41_200.0, 55_000.0] * 10,
            "country": ["US", "GB", "DE", "FR"] * 10,
        }
    )
    return run_scoring(df, cfg, "csv:test").result


def test_json_reports_written(result, tmp_path) -> None:
    paths = write_json(result, tmp_path)
    assert (tmp_path / "latest.json").exists()
    assert len(paths) == 2
    loaded = json.loads(paths[0].read_text(encoding="utf-8"))
    assert loaded["dqs"] == result["dqs"]
    assert loaded["verdict"] == result["verdict"]


def test_html_report_content(result, tmp_path) -> None:
    path = write_html(result, tmp_path)
    text = path.read_text(encoding="utf-8")
    assert path.stat().st_size > 3000
    assert "Data Quality Factor Scoring" in text
    assert result["verdict"] in text
    assert "completeness" in text
    assert "Model Card" in text
    # dynamic values are escaped: no raw angle brackets from data
    assert "<script" not in text


def test_html_escapes_malicious_values(tmp_path) -> None:
    result = {
        "run_id": "r1", "verdict": "PASS", "dqs": 90.0,
        "generated_at": "t", "duration_seconds": 1,
        "tool": {"name": "dqfs", "version": "1"},
        "dataset": {"name": "<script>alert(1)</script>", "source": "s",
                    "rows": 1, "columns": 1, "target": None},
        "factors": {}, "gates": [], "columns": [], "worst_rows": [],
        "model_card": {"enabled": False}, "thresholds": {"critical_factors": []},
    }
    text = write_html(result, tmp_path).read_text(encoding="utf-8")
    assert "<script>alert(1)</script>" not in text
    assert "&lt;script&gt;" in text


def test_excel_report_sheets(result, tmp_path) -> None:
    path = write_excel(result, tmp_path)
    assert path.name.startswith("scores_") and path.suffix == ".xlsx"
    book = openpyxl.load_workbook(path)
    assert book.sheetnames == ["Summary", "Factors", "Gates", "Columns", "Worst Rows"]
    ws = book["Factors"]
    header = [c.value for c in ws[1]]
    assert header[:3] == ["factor", "score", "status"]
    assert ws.max_row >= 7  # six factors + header


def test_csv_report(result, tmp_path) -> None:
    paths = write_csv(result, tmp_path)
    assert len(paths) == 2
    frame = pd.read_csv(tmp_path / "latest.csv")
    assert list(frame.columns)[:3] == ["dqs", "verdict", "factor"]
    assert len(frame) == 6


def test_write_errors_are_typed(result, tmp_path) -> None:
    blocker = tmp_path / "latest.json"
    blocker.write_text("x", encoding="utf-8")
    (tmp_path / "latest.json").chmod(0o000)
    try:
        with pytest.raises(ReportError):
            write_json(result, tmp_path)
    finally:
        (tmp_path / "latest.json").chmod(0o644)
