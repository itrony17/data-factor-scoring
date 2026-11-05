"""End-to-end CLI tests - exercises commands the way a fresh clone would."""

from __future__ import annotations

import json
import sqlite3

import pandas as pd
import pytest

from dqfs.cli import build_parser, main


@pytest.fixture()
def tiny_db(tmp_path):
    """A small SQLite database - keeps the deep ML path fast."""
    df = pd.DataFrame(
        {
            "id": list(range(1, 61)),
            "age": [25.0, 34.0, 19.0, 61.0] * 15,
            "income": [50_000.0, 62_500.0, 41_200.0, 55_000.0] * 15,
            "country": ["US", "GB", "DE", "FR"] * 15,
            "churn": [0, 1, 0, 1] * 15,
        }
    )
    df.loc[0:10, "age"] = None  # visible completeness issue
    db = tmp_path / "tiny.db"
    with sqlite3.connect(db) as conn:
        df.to_sql("items", conn, index=False)
    return f"sqlite:///{db}"


FAST_CONFIG_FILE = None  # built inline via tmp_path in tests


def _fast_config(tmp_path):
    cfg = tmp_path / "fast.yaml"
    cfg.write_text(
        "ml:\n  enabled: true\n  seed: 5\n  epochs: 12\n  batch_size: 32\n"
        "  hidden_dim: 16\n  embedding_dim: 4\n  attention_steps: 2\n"
        "  head_epochs: 8\n  oof_folds: 3\n",
        encoding="utf-8",
    )
    return cfg


def test_version_flag(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert "dqfs" in capsys.readouterr().out


def test_score_end_to_end(tmp_path, tiny_db, capsys) -> None:
    cfg = _fast_config(tmp_path)
    out = tmp_path / "results"
    code = main([
        "score", "--dsn", tiny_db, "--table", "items",
        "--config", str(cfg), "--target", "churn", "--out", str(out),
    ])
    assert code == 0
    captured = capsys.readouterr().out
    assert "verdict" in captured and "DQS" in captured
    assert (out / "latest.json").exists()
    assert (out / "latest.csv").exists()
    assert list(out.glob("report_*.html"))
    assert list(out.glob("scores_*.xlsx"))
    result = json.loads((out / "latest.json").read_text(encoding="utf-8"))
    assert result["model_card"]["enabled"] is True


def test_score_csv_source_and_fail_on_gate(tmp_path, capsys) -> None:
    csv_path = tmp_path / "bad.csv"
    # 35 missing ages + 25 missing countries = 60/150 null cells ->
    # completeness 0.60 < 0.65 warn threshold -> critical FAIL -> exit 7
    rows = []
    for i in range(50):
        age = "" if i < 35 else str(30 + i)
        country = "" if i % 2 == 0 else "US"
        rows.append(f"{i},{age},{country}")
    csv_path.write_text("id,age,country\n" + "\n".join(rows) + "\n", encoding="utf-8")
    cfg = tmp_path / "plain.yaml"
    cfg.write_text("ml:\n  enabled: false\n", encoding="utf-8")
    out = tmp_path / "results"
    code = main([
        "score", "--csv", str(csv_path), "--config", str(cfg),
        "--out", str(out), "--fail-on-gate",
    ])
    assert code == 7  # GateFailureError -> exit code 7
    result = json.loads((out / "latest.json").read_text(encoding="utf-8"))
    assert result["verdict"] == "FAIL"


def test_score_requires_source(tmp_path) -> None:
    cfg = tmp_path / "c.yaml"
    cfg.write_text("ml:\n  enabled: false\n", encoding="utf-8")
    code = main(["score", "--config", str(cfg)])
    assert code == 3  # DataLoadError


def test_report_command_rerenders(tmp_path, tiny_db) -> None:
    cfg = tmp_path / "c.yaml"
    cfg.write_text("ml:\n  enabled: false\n", encoding="utf-8")
    out = tmp_path / "results"
    assert main(["score", "--dsn", tiny_db, "--table", "items",
                 "--config", str(cfg), "--out", str(out)]) == 0
    for path in out.iterdir():
        if path.name not in ("latest.json",):
            path.unlink()
    code = main(["report", "--results", str(out)])
    assert code == 0
    assert list(out.glob("report_*.html"))
    assert list(out.glob("scores_*.xlsx"))
    assert (out / "latest.csv").exists()


def test_demo_command(tmp_path, capsys) -> None:
    code = main(["demo", "--out", str(tmp_path / "demo")])
    assert code == 0
    demo_dir = tmp_path / "demo"
    assert (demo_dir / "quality_demo.db").exists()
    assert (demo_dir / "dqfs-demo.yaml").exists()
    out = capsys.readouterr().out
    assert "quality_demo.db" in out


def test_score_on_generated_demo(tmp_path, capsys) -> None:
    """Full loop: demo -> score with the generated config (ML fast)."""
    demo_dir = tmp_path / "demo"
    assert main(["demo", "--out", str(demo_dir)]) == 0
    cfg = demo_dir / "dqfs-demo.yaml"
    # speed up the ML for CI while keeping the demo data realistic
    text = cfg.read_text(encoding="utf-8")
    text = text.replace("dataset:", "ml:\n  epochs: 12\n  head_epochs: 8\ndataset:")
    fast_cfg = tmp_path / "fast-demo.yaml"
    fast_cfg.write_text(text, encoding="utf-8")
    out = tmp_path / "results"
    code = main([
        "score", "--config", str(fast_cfg),
        "--dsn", f"sqlite:///{demo_dir / 'quality_demo.db'}",
        "--table", "customers", "--out", str(out), "--target", "churn",
    ])
    assert code == 0
    result = json.loads((out / "latest.json").read_text(encoding="utf-8"))
    assert result["dataset"]["rows"] == 2575
    assert result["factors"]["timeliness"]["score"] is not None


def test_publish_dry_run(tmp_path, tiny_db, monkeypatch, capsys) -> None:
    cfg = tmp_path / "c.yaml"
    cfg.write_text("ml:\n  enabled: false\n", encoding="utf-8")
    out = tmp_path / "results"
    main(["score", "--dsn", tiny_db, "--table", "items", "--config", str(cfg),
          "--out", str(out)])
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    code = main(["publish", "--repo", "acme/quality", "--results", str(out),
                 "--dry-run"])
    assert code == 0
    text = capsys.readouterr().out
    assert "dry run" in text
    assert "results/latest/latest.json" in text


def test_publish_missing_token_exit_code(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    code = main(["publish", "--repo", "acme/quality", "--results", str(tmp_path)])
    assert code == 6  # PublishError


def test_init_and_doctor(tmp_path, capsys) -> None:
    cfg = tmp_path / "dqfs.yaml"
    assert main(["init", "--path", str(cfg)]) == 0
    assert cfg.exists()
    assert main(["init", "--path", str(cfg)]) == 1  # exists, no --force
    code = main(["doctor", "--config", str(cfg)])
    assert code == 0
    text = capsys.readouterr().out
    assert "DQFS doctor" in text and "[PASS] config loads" in text


def test_doctor_detects_bad_source(tmp_path, capsys) -> None:
    code = main(["doctor", "--dsn", "notavaliddriver://host/db", "--table", "x"])
    assert code == 1
    assert "[FAIL] data source reachable" in capsys.readouterr().out


def test_parser_requires_command(capsys) -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args([])
    assert "command" in capsys.readouterr().err.lower()
