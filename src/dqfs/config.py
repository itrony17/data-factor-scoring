"""Configuration loading and validation for DQFS.

The pipeline is driven by a YAML file (see ``dqfs.yaml`` at the repository
root). Every key is optional; values provided by the user are deep-merged on
top of the defaults defined here, so a minimal config file only needs to
override what it wants to change.
"""

from __future__ import annotations

import copy
import logging
import re
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

#: SQL/CSV column identifiers accepted in factor configuration (table and
#: column names). Conservative on purpose: letters, digits, underscores,
#: optionally dotted ``schema.table`` qualification, up to 128 chars.
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}(?:\.[A-Za-z_][A-Za-z0-9_]{0,127})?$")

#: Numeric literals allowed as the right-hand side of consistency checks.
_NUMERIC_LITERAL_RE = re.compile(r"-?\d+(?:\.\d+)?")


def validate_identifier(name: str, what: str = "identifier") -> str:
    """Validate a column/table identifier coming from user config.

    Args:
        name: The identifier string.
        what: Description used in the error message.

    Returns:
        The validated identifier (unchanged).

    Raises:
        ValueError: If the identifier does not match the allowed pattern.
    """
    if not isinstance(name, str) or not _IDENTIFIER_RE.match(name):
        raise ValueError(
            f"invalid {what}: {name!r} - use letters, digits, underscores "
            "(optionally 'schema.table'), max 128 characters"
        )
    return name

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
    _validate_factor_bindings(cfg.factors)


def _validate_factor_bindings(factors: dict[str, Any]) -> None:
    """Check that all configured column references are safe identifiers.

    Every column/table name that reaches the SQL or pandas layer is
    validated at the config boundary so a hostile YAML cannot smuggle
    arbitrary expressions into queries or frame operations.
    """
    def _check(value: Any, what: str) -> None:
        validate_identifier(str(value), what)

    timeliness = factors.get("timeliness") or {}
    if timeliness.get("column"):
        _check(timeliness["column"], "factors.timeliness.column")
    for key in (factors.get("uniqueness") or {}).get("key_columns", []):
        _check(key, "factors.uniqueness.key_columns")
    for col in (factors.get("consistency") or {}).get("format_columns", []):
        _check(col, "factors.consistency.format_columns")
    for check in (factors.get("consistency") or {}).get("checks", []):
        if isinstance(check, dict):
            _check(check.get("left"), "factors.consistency.checks[].left")
            right = check.get("right")
            # ``right`` may be a literal value (e.g. '18') or a column name;
            # only numeric literals are exempt from identifier validation.
            if isinstance(right, str) and right not in FACTOR_ORDER and (
                not _NUMERIC_LITERAL_RE.fullmatch(right)
            ):
                _check(right, "factors.consistency.checks[].right")
    for column in (factors.get("validity") or {}).get("constraints", {}):
        _check(column, "factors.validity.constraints[]")


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
