"""Tests for dqfs.config - defaults, deep merge and validation."""

from __future__ import annotations

import pytest

from dqfs.config import load_config


def test_defaults_load_without_file() -> None:
    cfg = load_config(None)
    assert cfg.thresholds.factor_pass == pytest.approx(0.85)
    assert cfg.thresholds.factor_warn == pytest.approx(0.65)
    assert cfg.thresholds.composite_pass == pytest.approx(80.0)
    assert cfg.ml.seed == 42
    assert cfg.ml.oof_folds == 5
    assert "completeness" in cfg.critical_factors


def test_partial_yaml_is_deep_merged(tmp_path) -> None:
    cfg_file = tmp_path / "dqfs.yaml"
    cfg_file.write_text(
        "ml:\n  epochs: 7\nthresholds:\n  factor_pass: 0.9\n",
        encoding="utf-8",
    )
    cfg = load_config(cfg_file)
    assert cfg.ml.epochs == 7
    # untouched sibling key keeps its default
    assert cfg.ml.batch_size == 256
    assert cfg.thresholds.factor_pass == pytest.approx(0.9)


def test_missing_config_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_config("/nonexistent/dqfs.yaml")


def test_invalid_threshold_order_rejected(tmp_path) -> None:
    cfg_file = tmp_path / "bad.yaml"
    cfg_file.write_text(
        "thresholds:\n  factor_warn: 0.9\n  factor_pass: 0.65\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="factor_warn"):
        load_config(cfg_file)


def test_unknown_critical_factor_rejected(tmp_path) -> None:
    cfg_file = tmp_path / "bad2.yaml"
    cfg_file.write_text("critical_factors: [sparkle]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="sparkle"):
        load_config(cfg_file)


def test_ic_blend_bounds_enforced(tmp_path) -> None:
    cfg_file = tmp_path / "bad3.yaml"
    cfg_file.write_text("weights:\n  ic_blend: 1.5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="ic_blend"):
        load_config(cfg_file)


def test_github_defaults() -> None:
    cfg = load_config(None)
    assert cfg.github["branch"] == "gh-results"
    assert cfg.github["pages"] is False
