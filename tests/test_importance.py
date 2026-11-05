"""Tests for dqfs.ml.importance - IC statistics and ML-driven weights."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dqfs.config import load_config
from dqfs.factors import completeness, validity
from dqfs.io_loader import infer_column_kinds
from dqfs.ml.importance import blend_factor_weights, spearman


def test_spearman_monotone_relationships() -> None:
    a = np.arange(10, dtype=float)
    assert spearman(a, a * 2 + 1) == pytest.approx(1.0)
    assert spearman(a, -a) == pytest.approx(-1.0)
    assert spearman(a, a**2) > 0.9  # monotone on this range
    assert spearman(a, np.zeros(10)) == 0.0  # constant -> undefined -> 0
    assert spearman(a, a[:5]) == 0.0        # length mismatch guard


def test_spearman_handles_ties() -> None:
    a = np.array([1.0, 1.0, 2.0, 2.0, 3.0])
    b = np.array([10.0, 10.0, 20.0, 20.0, 30.0])
    assert spearman(a, b) == pytest.approx(1.0)


def _setup(tmp_path):
    df = pd.DataFrame(
        {
            "age": [25, 34, 19, 61, 45, 30, 28, 52],
            "income": [50_000.0, 62_500.0, 41_200.0, 88_000.0,
                       55_000.0, 49_900.0, 47_000.0, 71_000.0],
            "country": ["US", "GB", "DE", "FR", "IN", "US", "GB", "DE"],
        }
    )
    cfg = load_config(None)
    specs = infer_column_kinds(df)
    results = {"completeness": completeness(df, cfg, specs)}
    return df, cfg, specs, results


def test_weights_sum_to_one_and_expose_evidence(tmp_path) -> None:
    df, cfg, specs, results = _setup(tmp_path)
    results["validity"] = validity(df, cfg, specs)
    badness = np.linspace(0, 1, len(df))
    info = blend_factor_weights(
        results, badness, {"age": 0.9, "income": 0.5, "country": 0.1}, specs, cfg
    )
    assert set(info) == {"completeness", "validity"}
    total = sum(v["weight"] for v in info.values())
    assert total == pytest.approx(1.0)
    for name, entry in info.items():
        assert -1.0 <= entry["ic"] <= 1.0
        assert 0.0 <= entry["deep_importance"] <= 1.0
        assert 0.0 < entry["weight"] <= 1.0
        assert entry["columns"], f"{name} must map to columns"


def test_min_weight_floor_applied(tmp_path) -> None:
    df, cfg, specs, results = _setup(tmp_path)
    cfg.weights.min_weight = 0.4
    info = blend_factor_weights(results, np.linspace(0, 1, len(df)), {}, specs, cfg)
    assert all(v["weight"] >= 0.4 - 1e-9 for v in info.values())


def test_ic_blend_pure_modes(tmp_path) -> None:
    df, cfg, specs, results = _setup(tmp_path)
    results["validity"] = validity(df, cfg, specs)
    badness = np.linspace(0, 1, len(df))
    cfg.weights.ic_blend = 1.0
    info_ic = blend_factor_weights(results, badness, {}, specs, cfg)
    cfg.weights.ic_blend = 0.0
    info_deep = blend_factor_weights(results, badness, {}, specs, cfg)
    # with no importance data, pure-deep mode falls back to min_weight floor
    assert all(v["weight"] == pytest.approx(0.5) for v in info_deep.values())
    assert all("weight" in v for v in info_ic.values())


def test_factor_column_mapping(tmp_path) -> None:
    df, cfg, specs, results = _setup(tmp_path)
    info = blend_factor_weights(results, np.zeros(len(df)), {}, specs, cfg)
    # completeness draws importance from ALL classified columns
    assert set(info["completeness"]["columns"]) == {"age", "income", "country"}
