"""Configuration loading and validation for DQFS.

The pipeline is driven by a YAML file (see ``dqfs.yaml`` at the repository
root). Every key is optional; values provided by the user are deep-merged on
top of the defaults defined here, so a minimal config file only needs to
override what it wants to change.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

#: Ordered list of quality factors evaluated by the pipeline.
FACTOR_ORDER: tuple[str, ...] = (
    "completeness",
    "validity",
    "uniqueness",
    "consistency",
    "timeliness",
    "accuracy",
)

_DEFAULTS: dict[str, Any] = {
    "dataset": {"name": None, "owner": None},
    "thresholds": {
        "factor_pass": 0.85,
        "factor_warn": 0.65,
        "composite_pass": 80.0,
        "composite_warn": 65.0,
    },
    "critical_factors": ["completeness", "validity"],
    "weights": {"ic_blend": 0.6, "min_weight": 0.02},
    "ml": {
        "enabled": True,
        "seed": 42,
        "epochs": 120,
        "batch_size": 256,
        "embedding_dim": 8,
        "hidden_dim": 64,
        "attention_steps": 3,
        "dropout": 0.15,
        "corruption_rate": 0.18,
        "oof_folds": 5,
        "head_epochs": 60,
    },
    "factors": {
        "timeliness": {"column": None, "fresh_days": 30, "stale_days": 90},
        "uniqueness": {"key_columns": []},
        "consistency": {"checks": [], "format_columns": []},
        "validity": {"constraints": {}},
    },
    "github": {"branch": "gh-results", "pages": False, "pages_branch": "main"},
}


@dataclass
class Thresholds:
    """Score thresholds separating pass / warn / fail states."""

    factor_pass: float
    factor_warn: float
    composite_pass: float
    composite_warn: float


@dataclass
class WeightPolicy:
    """Policy describing how ML evidence is blended into factor weights."""

    ic_blend: float
    min_weight: float


@dataclass
class MlSettings:
    """Hyper-parameters of the deep tabular ensemble."""

    enabled: bool
    seed: int
    epochs: int
    batch_size: int
    embedding_dim: int
    hidden_dim: int
    attention_steps: int
    dropout: float
    corruption_rate: float
    oof_folds: int
    head_epochs: int


@dataclass
class DqfsConfig:
    """Fully resolved configuration for a scoring run."""

    dataset: dict[str, Any] = field(default_factory=dict)
    thresholds: Thresholds = field(
        default_factory=lambda: Thresholds(**_DEFAULTS["thresholds"])
    )
    critical_factors: list[str] = field(
        default_factory=lambda: list(_DEFAULTS["critical_factors"])
    )
    weights: WeightPolicy = field(
        default_factory=lambda: WeightPolicy(**_DEFAULTS["weights"])
    )
    ml: MlSettings = field(default_factory=lambda: MlSettings(**_DEFAULTS["ml"]))
    factors: dict[str, Any] = field(
        default_factory=lambda: copy.deepcopy(_DEFAULTS["factors"])
    )
    github: dict[str, Any] = field(
        default_factory=lambda: copy.deepcopy(_DEFAULTS["github"])
    )
    raw: dict[str, Any] = field(default_factory=dict)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base`` and return a new dict."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _validate(cfg: DqfsConfig) -> None:
    """Raise ``ValueError`` when resolved configuration values are invalid."""
    t = cfg.thresholds
    if not (0.0 <= t.factor_warn <= t.factor_pass <= 1.0):
        raise ValueError(
            "thresholds.factor_warn must be <= thresholds.factor_pass <= 1.0"
        )
    if not (0.0 <= t.composite_warn <= t.composite_pass <= 100.0):
        raise ValueError(
            "thresholds.composite_warn must be <= thresholds.composite_pass <= 100"
        )
    if not 0.0 <= cfg.weights.ic_blend <= 1.0:
        raise ValueError("weights.ic_blend must be within [0, 1]")
    if not 0.0 <= cfg.weights.min_weight <= 1.0:
        raise ValueError("weights.min_weight must be within [0, 1]")
    if cfg.ml.oof_folds < 2:
        raise ValueError("ml.oof_folds must be >= 2")
    unknown = [f for f in cfg.critical_factors if f not in FACTOR_ORDER]
    if unknown:
        raise ValueError(f"unknown critical factors: {unknown}")


def load_config(path: str | Path | None = None) -> DqfsConfig:
    """Load a YAML config file and merge it over the built-in defaults.

    Args:
        path: Path to a ``dqfs.yaml`` style file. ``None`` returns pure
            defaults.

    Returns:
        A validated :class:`DqfsConfig`.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the resolved configuration is invalid.
    """
    user: dict[str, Any] = {}
    if path is not None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"config file not found: {p}")
        with p.open("r", encoding="utf-8") as fh:
            user = yaml.safe_load(fh) or {}
        if not isinstance(user, dict):
            raise ValueError(f"config root must be a mapping, got {type(user)!r}")
        logger.info("loaded config from %s", p)

    merged = _deep_merge(_DEFAULTS, user)
    cfg = DqfsConfig(
        dataset=merged["dataset"],
        thresholds=Thresholds(**merged["thresholds"]),
        critical_factors=list(merged["critical_factors"]),
        weights=WeightPolicy(**merged["weights"]),
        ml=MlSettings(**merged["ml"]),
        factors=merged["factors"],
        github=merged["github"],
        raw=merged,
    )
    _validate(cfg)
    return cfg
