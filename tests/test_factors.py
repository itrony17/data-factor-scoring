"""Tests for the six rule-based quality factors in dqfs.factors."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dqfs.config import load_config
from dqfs.factors import FACTOR_ORDER, evaluate_factors
from dqfs.io_loader import infer_column_kinds


def _make_cfg(tmp_path, extra: str = ""):
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text(extra, encoding="utf-8") if extra else cfg_file.touch()
    return load_config(cfg_file if extra else None)


CLEAN = pd.DataFrame(
    {
        "id": list(range(6)),
        "age": [25, 34, 19, 61, 45, 30],
        "income": [50_000.0, 62_500.0, 41_200.0, 88_000.0, 55_000.0, 49_900.0],
        "country": ["US", "GB", "DE", "FR", "IN", "US"],
        "email": [f"u{i}@x.com" for i in range(6)],
    }
)


def test_clean_dataset_scores_high(tmp_path) -> None:
    cfg = _make_cfg(tmp_path)
    specs = infer_column_kinds(CLEAN)
    results = evaluate_factors(CLEAN, cfg, specs)
    assert list(results) == list(FACTOR_ORDER)
    assert results["completeness"].score == pytest.approx(1.0)
    assert results["validity"].score == pytest.approx(1.0)
    assert results["uniqueness"].score == pytest.approx(1.0)
    assert results["accuracy"].score == pytest.approx(1.0)


def test_row_contributions_shape_and_range(tmp_path) -> None:
    cfg = _make_cfg(tmp_path)
    specs = infer_column_kinds(CLEAN)
    for res in evaluate_factors(CLEAN, cfg, specs).values():
        assert res.row_contribution.shape == (len(CLEAN),)
        assert np.all(res.row_contribution >= 0.0)
        assert np.all(res.row_contribution <= 1.0)


def test_completeness_detects_missing(tmp_path) -> None:
    df = CLEAN.copy()
    df.loc[0, "age"] = np.nan
    df.loc[1, "income"] = np.nan
    cfg = _make_cfg(tmp_path)
    res = evaluate_factors(df, cfg, infer_column_kinds(df))["completeness"]
    assert res.score < 1.0
    assert res.details["missing_cells"] == 2
    assert res.row_contribution[0] > 0.0
    # clean rows are perfectly complete
    assert res.row_contribution[2] == pytest.approx(0.0)


def test_validity_constraints(tmp_path) -> None:
    extra = (
        "factors:\n"
        "  validity:\n"
        "    constraints:\n"
        "      age: {min: 18, max: 100}\n"
        "      country: {allowed: [US, GB, DE, FR, IN]}\n"
    )
    cfg = _make_cfg(tmp_path, extra)
    df = CLEAN.copy()
    df.loc[2, "age"] = 7           # below min
    df.loc[3, "country"] = "XX"    # not allowed
    res = evaluate_factors(df, cfg, infer_column_kinds(df))["validity"]
    assert res.score < 1.0
    assert res.details["invalid_cells"] == 2
    assert res.column_scores["age"] < 1.0
    assert res.column_scores["country"] < 1.0


def test_validity_rejects_unparseable_numeric(tmp_path) -> None:
    df = CLEAN.copy()
    df["age"] = df["age"].astype(object)
    df.loc[4, "age"] = "not-a-number"
    cfg = _make_cfg(tmp_path)
    res = evaluate_factors(df, cfg, infer_column_kinds(df))["validity"]
    assert res.score < 1.0
    assert res.details["invalid_cells"] == 1


def test_uniqueness_flags_duplicates(tmp_path) -> None:
    df = pd.concat([CLEAN, CLEAN.iloc[[2]]], ignore_index=True)
    cfg = _make_cfg(tmp_path)
    res = evaluate_factors(df, cfg, infer_column_kinds(df))["uniqueness"]
    assert res.score < 1.0
    assert res.details["duplicate_rows"] == 1
    # last occurrence is the duplicate
    assert res.row_contribution[-1] == pytest.approx(1.0)


def test_uniqueness_uses_key_columns(tmp_path) -> None:
    extra = "factors:\n  uniqueness:\n    key_columns: [id]\n"
    cfg = _make_cfg(tmp_path, extra)
    df = CLEAN.copy()
    df.loc[5, "id"] = df.loc[0, "id"]  # key dup, otherwise different rows
    res = evaluate_factors(df, cfg, infer_column_kinds(df))["uniqueness"]
    assert res.details["duplicate_key_rows"] == 1
    assert res.score < 1.0


def test_consistency_cross_column_check(tmp_path) -> None:
    extra = (
        "factors:\n"
        "  consistency:\n"
        "    checks:\n"
        "      - {left: age, op: '>=', right: '18'}\n"
    )
    cfg = _make_cfg(tmp_path, extra)
    df = CLEAN.copy()
    df.loc[2, "age"] = 12
    res = evaluate_factors(df, cfg, infer_column_kinds(df))["consistency"]
    assert res.score < 1.0
    assert res.details["checks_run"] == 1
    assert res.row_contribution[2] > 0.0
    assert res.column_scores["check:age >= 18"] == pytest.approx(1.0 - 1 / 6)


def test_consistency_format_uniformity(tmp_path) -> None:
    extra = "factors:\n  consistency:\n    format_columns: [phone]\n"
    cfg = _make_cfg(tmp_path, extra)
    df = CLEAN.copy()
    df["phone"] = ["555-0100", "555-0101", "555 0102", "555-0103", "555-0104", "555-0105"]
    res = evaluate_factors(df, cfg, infer_column_kinds(df))["consistency"]
    assert res.score < 1.0
    assert res.details["dominant_patterns"]["phone"] == "999-9999"
    assert res.details["format_mismatch_cells"] == 1


def test_timeliness_na_without_column(tmp_path) -> None:
    cfg = _make_cfg(tmp_path)
    res = evaluate_factors(CLEAN, cfg, infer_column_kinds(CLEAN))["timeliness"]
    assert res.score is None
    assert not res.available


def test_timeliness_fresh_vs_stale(tmp_path) -> None:
    extra = (
        "factors:\n"
        "  timeliness:\n"
        "    column: seen_at\n"
        "    fresh_days: 7\n"
        "    stale_days: 90\n"
    )
    cfg = _make_cfg(tmp_path, extra)
    fresh = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=1)
    stale = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=180)
    df = pd.DataFrame({"seen_at": [str(fresh)] * 3 + [str(stale)] * 3})
    res = evaluate_factors(df, cfg, infer_column_kinds(df))["timeliness"]
    # median age is 90+ days -> score 0
    assert res.score == pytest.approx(0.0)
    assert res.details["median_age_days"] > 90

    df_fresh = pd.DataFrame({"seen_at": [str(fresh)] * 6})
    res2 = evaluate_factors(df_fresh, cfg, infer_column_kinds(df_fresh))["timeliness"]
    assert res2.score == pytest.approx(1.0)


def test_timeliness_unparseable_is_na(tmp_path) -> None:
    extra = "factors:\n  timeliness:\n    column: seen_at\n"
    cfg = _make_cfg(tmp_path, extra)
    df = pd.DataFrame({"seen_at": ["nope", "also-nope"]})
    res = evaluate_factors(df, cfg, infer_column_kinds(df))["timeliness"]
    assert res.score is None
    assert "no parseable timestamps" in res.details["note"]


def test_accuracy_detects_outliers(tmp_path) -> None:
    df = CLEAN.copy()
    df.loc[0, "income"] = 9_500_000.0  # extreme outlier vs tight cluster
    cfg = _make_cfg(tmp_path)
    res = evaluate_factors(df, cfg, infer_column_kinds(df))["accuracy"]
    assert res.score < 1.0
    assert res.details["outlier_cells"] >= 1
    assert res.row_contribution[0] > 0.0


def test_accuracy_na_without_numeric_columns(tmp_path) -> None:
    df = pd.DataFrame({"c": ["a", "b"]})
    cfg = _make_cfg(tmp_path)
    res = evaluate_factors(df, cfg, infer_column_kinds(df))["accuracy"]
    assert res.score is None


def test_empty_dataframe_scores_zero(tmp_path) -> None:
    cfg = _make_cfg(tmp_path)
    empty = CLEAN.iloc[0:0]
    results = evaluate_factors(empty, cfg, infer_column_kinds(empty))
    assert results["completeness"].score == pytest.approx(0.0)
    assert results["validity"].score == pytest.approx(0.0)


def test_broken_factor_is_contained_not_fatal(tmp_path, monkeypatch) -> None:
    import dqfs.factors as fmod

    cfg = _make_cfg(tmp_path)

    def boom(df, cfg, specs):
        raise RuntimeError("boom")

    monkeypatch.setitem(fmod.FACTOR_FUNCS, "accuracy", boom)
    results = evaluate_factors(CLEAN, cfg, infer_column_kinds(CLEAN))
    assert results["accuracy"].score is None
    assert results["accuracy"].details["error_type"] == "RuntimeError"
    # other factors still evaluated
    assert results["completeness"].score == pytest.approx(1.0)
