"""Tests for dqfs.ml.denoise / ensemble - the deep stacking pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dqfs.config import load_config
from dqfs.io_loader import infer_column_kinds
from dqfs.ml.denoise import rank_normalize, zscore
from dqfs.ml.ensemble import DeepQualityEnsemble

ML_YAML = """
ml:
  enabled: true
  seed: 7
  epochs: 60
  batch_size: 64
  embedding_dim: 4
  hidden_dim: 16
  attention_steps: 2
  dropout: 0.1
  corruption_rate: 0.3
  oof_folds: 3
  head_epochs: 25
"""


@pytest.fixture()
def cfg(tmp_path):
    p = tmp_path / "ml.yaml"
    p.write_text(ML_YAML, encoding="utf-8")
    return load_config(p)


@pytest.fixture()
def corrupted_data():
    """140 rows, 25% of them visibly corrupted."""
    rng = np.random.default_rng(3)
    n = 140
    a = rng.normal(size=n)
    b = rng.normal(size=n)
    c = rng.choice(["x", "y", "z"], size=n)
    corrupt = np.zeros(n, dtype=bool)
    corrupt[rng.choice(n, size=35, replace=False)] = True
    a[corrupt] = a[corrupt] * 8 + 30
    b[corrupt] = -b[corrupt] * 6 - 5
    c[corrupt] = rng.choice(["x", "y", "z"], size=corrupt.sum())
    df = pd.DataFrame({"a": a, "b": b, "c": c})
    return df, corrupt


def test_rank_normalize_bounds() -> None:
    r = rank_normalize(np.array([3.0, 1.0, 2.0, 5.0]))
    assert r.min() == 0.0 and r.max() == 1.0
    assert np.all(np.diff(r[[1, 2, 0, 3]]) >= 0)  # ordered consistently


def test_zscore_constant_is_zero() -> None:
    assert np.allclose(zscore(np.array([2.0, 2.0, 2.0])), 0.0)
    z = zscore(np.array([1.0, 2.0, 3.0]))
    assert z.mean() == pytest.approx(0.0, abs=1e-9)


def test_ensemble_flags_corrupted_rows(cfg, corrupted_data) -> None:
    df, corrupt = corrupted_data
    specs = infer_column_kinds(df)
    out = DeepQualityEnsemble(cfg).fit(df, specs)
    assert out.enabled
    assert out.row_badness.shape == (len(df),)
    assert out.row_badness.min() >= 0.0 and out.row_badness.max() <= 1.0
    assert out.row_badness[corrupt].mean() > out.row_badness[~corrupt].mean()
    # the worst half must be enriched with corrupted rows well beyond the
    # 25% base rate (a statistically meaningful detection check)
    worst_half = out.row_badness >= np.median(out.row_badness)
    corrupt_share = (worst_half & corrupt).sum() / max(worst_half.sum(), 1)
    assert corrupt_share > 0.25 * 1.2
    assert out.column_importance and max(out.column_importance.values()) <= 1.0
    assert "attention_branch" in out.model_card["architecture"]


def test_ensemble_disabled_by_config(tmp_path, corrupted_data) -> None:
    p = tmp_path / "off.yaml"
    p.write_text(ML_YAML + "\n  enabled: false\n", encoding="utf-8")
    df, _ = corrupted_data
    out = DeepQualityEnsemble(load_config(p)).fit(df, infer_column_kinds(df))
    assert not out.enabled
    assert np.allclose(out.row_badness, 0.0)
    assert out.model_card["reason"] == "disabled by config"


def test_ensemble_too_few_rows(tmp_path) -> None:
    p = tmp_path / "small.yaml"
    p.write_text(ML_YAML, encoding="utf-8")
    df = pd.DataFrame({"a": [1.0, 2.0], "c": ["x", "y"]})
    out = DeepQualityEnsemble(load_config(p)).fit(df, infer_column_kinds(df))
    assert not out.enabled
    assert "too few rows" in out.model_card["reason"]


def test_supervised_target_path(cfg, corrupted_data) -> None:
    df, corrupt = corrupted_data
    rng = np.random.default_rng(9)
    df = df.copy()
    df["flag"] = (df["a"] > 10).astype(int)  # learnable binary target
    del rng
    specs = infer_column_kinds(df.drop(columns=["flag"]))
    out = DeepQualityEnsemble(cfg).fit(df, specs, target="flag")
    assert out.enabled
    assert out.model_card["supervised"] is True
    assert "ridge_base" in out.model_card["signals"]
    assert out.row_badness.shape == (len(df),)


def test_meta_ridge_calibration_recorded(cfg, corrupted_data) -> None:
    df, _ = corrupted_data
    specs = infer_column_kinds(df)
    rng = np.random.default_rng(1)
    rule_badness = np.clip(df["a"].abs() / 10 + rng.random(len(df)) * 0.1, 0, 1)
    out = DeepQualityEnsemble(cfg).fit(df, specs, rule_badness=rule_badness)
    meta = out.model_card.get("meta_ridge")
    assert meta is not None
    assert len(meta["coefficients"]) == len(meta["signals"]) + 1
    assert -1.0 <= meta["r_squared"] <= 1.0


def test_ensemble_is_deterministic(cfg, corrupted_data) -> None:
    df, _ = corrupted_data
    specs = infer_column_kinds(df)
    out1 = DeepQualityEnsemble(cfg).fit(df, specs)
    out2 = DeepQualityEnsemble(cfg).fit(df, specs)
    assert np.allclose(out1.row_badness, out2.row_badness)
    assert out1.column_importance.keys() == out2.column_importance.keys()
