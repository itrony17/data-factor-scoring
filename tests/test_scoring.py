"""End-to-end tests for the dqfs.scoring pipeline."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from dqfs.config import load_config
from dqfs.errors import ScoringError
from dqfs.scoring import run_scoring

FAST_ML = """
ml:
  enabled: false
"""


def _clean_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": list(range(40)),
            "age": np.repeat([25.0, 34.0, 19.0, 61.0], 10),
            "income": np.repeat([50_000.0, 62_500.0, 41_200.0, 55_000.0], 10),
            "country": ["US", "GB", "DE", "FR"] * 10,
        }
    )


def test_clean_dataset_passes_with_high_dqs(tmp_path) -> None:
    p = tmp_path / "c.yaml"
    p.write_text(FAST_ML, encoding="utf-8")
    cfg = load_config(p)
    run = run_scoring(_clean_df(), cfg, "csv:test")
    assert run.verdict == "PASS"
    assert run.dqs > 85.0
    assert run.result["dqs"] == pytest.approx(run.dqs, abs=0.01)
    assert run.result["dataset"]["rows"] == 40
    assert run.result["dataset"]["columns"] == 4
    # weights normalized
    weights = [f["weight"] for f in run.result["factors"].values() if "weight" in f]
    assert sum(weights) == pytest.approx(1.0, abs=1e-3)
    # gates present for every factor + composite
    assert len(run.result["gates"]) == 7


def test_result_is_json_serializable(tmp_path) -> None:
    p = tmp_path / "c.yaml"
    p.write_text(FAST_ML, encoding="utf-8")
    cfg = load_config(p)
    run = run_scoring(_clean_df(), cfg, "csv:test")
    payload = json.dumps(run.result)  # must not raise
    assert "dqs" in payload


def test_corrupted_dataset_fails_gates(tmp_path) -> None:
    p = tmp_path / "c.yaml"
    p.write_text(FAST_ML, encoding="utf-8")
    cfg = load_config(p)
    df = _clean_df()
    # wipe three columns for half the rows -> completeness ~0.63 (critical FAIL)
    df.loc[0:20, ["age", "income", "country"]] = np.nan
    df = pd.concat([df, df.iloc[[3, 3, 3]]], ignore_index=True)  # duplicates
    run = run_scoring(df, cfg, "csv:bad")
    assert run.verdict == "FAIL"
    assert run.result["factors"]["completeness"]["score"] < 0.65
    assert run.result["factors"]["uniqueness"]["score"] < 1.0
    assert run.result["worst_rows"], "worst rows sample must not be empty"
    assert len(run.result["worst_rows"]) <= 10


def test_ml_enabled_small_run(tmp_path) -> None:
    p = tmp_path / "c.yaml"
    p.write_text(
        "ml:\n  enabled: true\n  seed: 3\n  epochs: 30\n  batch_size: 32\n"
        "  hidden_dim: 16\n  embedding_dim: 4\n  attention_steps: 2\n"
        "  head_epochs: 20\n",
        encoding="utf-8",
    )
    cfg = load_config(p)
    df = _clean_df()
    df["flag"] = (df["age"] > 30).astype(int)
    run = run_scoring(df, cfg, "csv:test", target="flag")
    assert run.result["model_card"]["enabled"] is True
    assert run.result["model_card"]["supervised"] is True
    assert 0.0 <= run.dqs <= 100.0


def test_scoring_is_deterministic(tmp_path) -> None:
    p = tmp_path / "c.yaml"
    p.write_text(
        "ml:\n  enabled: true\n  seed: 11\n  epochs: 25\n  batch_size: 32\n"
        "  hidden_dim: 16\n  embedding_dim: 4\n  attention_steps: 2\n",
        encoding="utf-8",
    )
    cfg = load_config(p)
    r1 = run_scoring(_clean_df(), cfg, "csv:test")
    r2 = run_scoring(_clean_df(), cfg, "csv:test")
    assert r1.dqs == pytest.approx(r2.dqs)


def test_pipeline_failure_raises_typed_error(tmp_path, monkeypatch) -> None:
    import dqfs.scoring as smod

    p = tmp_path / "c.yaml"
    p.write_text(FAST_ML, encoding="utf-8")
    cfg = load_config(p)

    def boom(df, cfg):
        raise ZeroDivisionError("boom")

    monkeypatch.setattr(smod, "evaluate_factors", boom)
    with pytest.raises(ScoringError, match="scoring pipeline failed"):
        run_scoring(_clean_df(), cfg, "csv:test")
