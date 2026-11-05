"""Rule-based data quality factor engine.

Six factors are computed per dataset. Each factor returns:

* a dataset-level ``score`` in ``[0, 1]`` (1.0 = perfect, ``None`` = N/A);
* a per-row badness contribution in ``[0, 1]`` (1.0 = worst) used by the ML
  stacking layer to correlate statistical evidence with deep-model signals;
* per-column sub-scores and free-form ``details`` used by the reports.

Factors
-------
completeness  share of non-missing cells
validity      conformance to inferred types and configured constraints
uniqueness    absence of duplicate rows / duplicate business keys
consistency  cross-column logic and value-format uniformity
timeliness    freshness of a configured timestamp column
accuracy      absence of statistical outliers (robust z-score via MAD)
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from dqfs.config import FACTOR_ORDER, DqfsConfig
from dqfs.io_loader import ColumnSpecs

logger = logging.getLogger(__name__)

_ROBUST_Z_LIMIT = 3.5
_TUKEY_FAR_IQR = 3.0
_OPS: dict[str, Callable[[pd.Series, Any], pd.Series]] = {
    ">=": lambda a, b: a >= b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    "<": lambda a, b: a < b,
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
}


@dataclass
class FactorResult:
    """Outcome of evaluating a single quality factor."""

    name: str
    score: float | None
    row_contribution: np.ndarray
    column_scores: dict[str, float] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def available(self) -> bool:
        """False when the factor could not be evaluated (excluded from DQS)."""
        return self.score is not None


def _clip(values: np.ndarray | pd.Series) -> np.ndarray:
    """Clip an array of badness values into ``[0, 1]`` as float64."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return arr
    return np.clip(np.nan_to_num(arr, nan=0.0), 0.0, 1.0)


# --------------------------------------------------------------------------
# completeness
# --------------------------------------------------------------------------
def completeness(
    df: pd.DataFrame, cfg: DqfsConfig, specs: ColumnSpecs
) -> FactorResult:
    """Score the share of non-missing cells across all columns."""
    if df.empty:
        return FactorResult("completeness", 0.0, np.zeros(0))
    null_ratio = df.isna().mean()
    score = float(1.0 - null_ratio.mean())
    row = _clip(df.isna().mean(axis=1)) if len(df.columns) else np.zeros(len(df))
    worst = null_ratio.sort_values(ascending=False).head(5)
    return FactorResult(
        name="completeness",
        score=score,
        row_contribution=row,
        column_scores={c: float(1.0 - r) for c, r in null_ratio.items()},
        details={
            "missing_cells": int(df.isna().sum().sum()),
            "total_cells": int(df.size),
            "worst_columns": [
                {"column": c, "null_ratio": round(float(r), 4)}
                for c, r in worst.items() if r > 0
            ],
        },
    )


# --------------------------------------------------------------------------
# validity
# --------------------------------------------------------------------------
def _column_invalid_mask(series: pd.Series, rule: dict[str, Any]) -> pd.Series:
    """Boolean mask of constraint violations for one column."""
    non_null = series.notna()
    invalid = pd.Series(False, index=series.index)
    if "allowed" in rule:
        invalid |= non_null & ~series.isin(rule["allowed"])
    if "regex" in rule and not pd.api.types.is_numeric_dtype(series):
        pattern = re.compile(str(rule["regex"]))
        invalid |= non_null & ~series.astype("string").str.fullmatch(pattern).fillna(
            False
        )
    if ("min" in rule or "max" in rule) and pd.api.types.is_numeric_dtype(series):
        nums = pd.to_numeric(series, errors="coerce")
        if "min" in rule:
            invalid |= non_null & (nums < float(rule["min"]))
        if "max" in rule:
            invalid |= non_null & (nums > float(rule["max"]))
    return invalid


def validity(df: pd.DataFrame, cfg: DqfsConfig, specs: ColumnSpecs) -> FactorResult:
    """Score conformance to data types and configured value constraints."""
    if df.empty:
        return FactorResult("validity", 0.0, np.zeros(0))
    constraints: dict[str, dict[str, Any]] = cfg.factors.get("validity", {}).get(
        "constraints", {}
    )
    n = len(df)
    invalid_total = 0
    evaluable_total = 0
    col_scores: dict[str, float] = {}
    row_bad = np.zeros(n, dtype=np.float64)
    checked_cols = 0

    for col in df.columns:
        series = df[col]
        non_null = int(series.notna().sum())
        invalid = pd.Series(False, index=series.index)

        if pd.api.types.is_numeric_dtype(series):
            coerced = pd.to_numeric(series, errors="coerce")
            invalid |= series.notna() & coerced.isna() | np.isinf(coerced.fillna(0.0))
        elif series.notna().any():
            # Object columns: if most values parse as numbers, unparseable
            # leftovers count as invalid (catches "12", 13, "N/A" mixes).
            coerced = pd.to_numeric(series.dropna(), errors="coerce")
            if coerced.notna().mean() >= 0.8:
                full = pd.to_numeric(series, errors="coerce")
                invalid |= series.notna() & full.isna()
        if col in constraints:
            invalid |= _column_invalid_mask(series, constraints[col])

        n_invalid = int(invalid.sum())
        denom = max(non_null, 1)
        col_scores[col] = float(1.0 - n_invalid / denom)
        invalid_total += n_invalid
        evaluable_total += non_null
        if non_null:
            row_bad += invalid.to_numpy(dtype=np.float64)
            checked_cols += 1

    score = 1.0 - invalid_total / max(evaluable_total, 1)
    worst = sorted(
        ((c, 1.0 - s) for c, s in col_scores.items()), key=lambda x: -x[1]
    )[:5]
    return FactorResult(
        name="validity",
        score=float(score),
        row_contribution=_clip(row_bad / max(checked_cols, 1)),
        column_scores=col_scores,
        details={
            "invalid_cells": invalid_total,
            "evaluable_cells": evaluable_total,
            "worst_columns": [
                {"column": c, "invalid_ratio": round(float(r), 4)} for c, r in worst if r > 0
            ],
        },
    )


# --------------------------------------------------------------------------
# uniqueness
# --------------------------------------------------------------------------
def uniqueness(df: pd.DataFrame, cfg: DqfsConfig, specs: ColumnSpecs) -> FactorResult:
    """Score absence of duplicate rows and duplicate configured key columns."""
    if df.empty:
        return FactorResult("uniqueness", 0.0, np.zeros(0))
    keys: list[str] = [
        k for k in cfg.factors.get("uniqueness", {}).get("key_columns", [])
        if k in df.columns
    ]
    full_dup = df.duplicated(keep="first")
    key_dup: pd.Series | None = None
    if keys:
        key_dup = df.duplicated(subset=keys, keep="first")
        blend = (2.0 * key_dup.astype(float) + full_dup.astype(float)) / 3.0
    else:
        blend = full_dup.astype(float)
    score = float(1.0 - blend.mean())
    col_scores = {
        k: float(1.0 - df.duplicated(subset=[k], keep="first").mean()) for k in keys
    }
    return FactorResult(
        name="uniqueness",
        score=score,
        row_contribution=_clip(blend.to_numpy()),
        column_scores=col_scores,
        details={
            "duplicate_rows": int(full_dup.sum()),
            "duplicate_key_rows": int(key_dup.sum()) if key_dup is not None else None,
            "key_columns": keys or None,
        },
    )


# --------------------------------------------------------------------------
# consistency
# --------------------------------------------------------------------------
def _pattern(value: str) -> str:
    """Reduce a string to a coarse pattern signature (digits -> 9, a-z -> a)."""
    sig = re.sub(r"\d", "9", str(value))
    return re.sub(r"[A-Za-z]", "a", sig)


def _resolve_side(
    df: pd.DataFrame, token: str
) -> tuple[pd.Series | float | str, str]:
    """Resolve a check operand to a typed Series or constant.

    Returns:
        Tuple of ``(value, kind)`` where kind is ``num``, ``dt`` or ``str``.
    """
    if token in df.columns:
        series = df[token]
        if pd.api.types.is_numeric_dtype(series):
            return pd.to_numeric(series, errors="coerce"), "num"
        parsed = pd.to_datetime(series, errors="coerce", utc=True, format="mixed")
        if parsed.notna().any() and parsed.notna().mean() >= 0.8:
            return parsed, "dt"
        return series, "str"
    try:
        return float(token), "num"
    except ValueError:
        pass
    parsed = pd.to_datetime(token, errors="coerce", utc=True)
    if pd.notna(parsed):
        return parsed, "dt"
    return token, "str"


def consistency(df: pd.DataFrame, cfg: DqfsConfig, specs: ColumnSpecs) -> FactorResult:
    """Score cross-column logical checks and value-format uniformity."""
    if df.empty:
        return FactorResult("consistency", 0.0, np.zeros(0))
    fcfg = cfg.factors.get("consistency", {})
    checks: list[dict[str, str]] = fcfg.get("checks", []) or []
    format_cols: list[str] = [
        c for c in (fcfg.get("format_columns", []) or []) if c in df.columns
    ]

    n = len(df)
    check_scores: list[float] = []
    check_bad = np.zeros(n, dtype=np.float64)
    col_scores: dict[str, float] = {}
    for chk in checks:
        op = str(chk.get("op", ">="))
        if op not in _OPS:
            logger.warning("unknown consistency operator %r - check skipped", op)
            continue
        left, left_kind = _resolve_side(df, str(chk.get("left", "")))
        right, right_kind = _resolve_side(df, str(chk.get("right", "")))
        label = f"{chk.get('left')} {op} {chk.get('right')}"
        if not isinstance(left, pd.Series):
            logger.warning("consistency check left side must be a column: %s", chk)
            continue
        if left_kind != right_kind:
            logger.warning("type mismatch in consistency check %r - skipped", label)
            continue
        if left_kind == "str" and op not in ("==", "!="):
            logger.warning("operator %r unsupported for strings in %r", op, label)
            continue
        ok = _OPS[op](left, right)
        bad = left.notna() & ~ok.fillna(False).astype(bool)
        bad_arr = bad.to_numpy(dtype=np.float64)
        check_bad += bad_arr
        pass_rate = float(1.0 - bad_arr.mean())
        check_scores.append(pass_rate)
        col_scores[f"check:{label}"] = pass_rate
    checks_score = float(np.mean(check_scores)) if check_scores else None

    fmt_scores: list[float] = []
    fmt_bad = np.zeros(n, dtype=np.float64)
    dominant_patterns: dict[str, str] = {}
    for col in format_cols:
        non_null = df[col].dropna().astype(str)
        if non_null.empty:
            continue
        patterns = non_null.map(_pattern)
        dominant = patterns.value_counts().idxmax()
        dominant_patterns[col] = str(dominant)
        share = float((patterns == dominant).mean())
        fmt_scores.append(share)
        col_scores[f"format:{col}"] = share
        mismatch = df[col].notna() & (df[col].astype(str).map(_pattern) != dominant)
        fmt_bad += mismatch.to_numpy(dtype=np.float64)
    format_score = float(np.mean(fmt_scores)) if fmt_scores else None

    parts = [s for s in (checks_score, format_score) if s is not None]
    score = float(np.mean(parts)) if parts else 1.0
    denom = max(len(checks) + len(format_cols), 1)
    return FactorResult(
        name="consistency",
        score=score,
        row_contribution=_clip((check_bad + fmt_bad) / denom),
        column_scores=col_scores,
        details={
            "checks_run": len(check_scores),
            "format_columns": format_cols,
            "dominant_patterns": dominant_patterns,
            "format_mismatch_cells": int(fmt_bad.sum()),
        },
    )


# --------------------------------------------------------------------------
# timeliness
# --------------------------------------------------------------------------
def timeliness(df: pd.DataFrame, cfg: DqfsConfig, specs: ColumnSpecs) -> FactorResult:
    """Score freshness of the configured timestamp column.

    Returns ``score=None`` (N/A) when no timeliness column is configured or
    no value parses as a timestamp; the factor is then excluded from the
    composite score.
    """
    tcfg = cfg.factors.get("timeliness", {})
    col = tcfg.get("column")
    if not col or col not in df.columns or df.empty:
        return FactorResult("timeliness", None, np.zeros(len(df)))
    fresh, stale = float(tcfg.get("fresh_days", 30)), float(tcfg.get("stale_days", 90))
    if stale <= fresh:
        raise ValueError("timeliness.stale_days must be greater than fresh_days")

    parsed = pd.to_datetime(df[col], errors="coerce", utc=True, format="mixed")
    valid = parsed.dropna()
    if valid.empty:
        return FactorResult(
            "timeliness", None, np.zeros(len(df)),
            details={"column": col, "note": "no parseable timestamps"},
        )
    now = pd.Timestamp.now(tz="UTC")
    age_days = (now - valid).dt.total_seconds() / 86400.0
    age_days = age_days.clip(lower=0.0)
    median_age = float(age_days.median())

    def freshness(age: float) -> float:
        if age <= fresh:
            return 1.0
        if age >= stale:
            return 0.0
        return 1.0 - (age - fresh) / (stale - fresh)

    score = freshness(median_age)
    row_age = (now - parsed).dt.total_seconds() / 86400.0 / (stale)
    row_bad = (1.0 - (1.0 - row_age.clip(lower=0.0, upper=1.0))).fillna(0.0)
    return FactorResult(
        name="timeliness",
        score=float(score),
        row_contribution=_clip(row_bad.to_numpy()),
        column_scores={col: float(score)},
        details={
            "column": col,
            "median_age_days": round(median_age, 1),
            "oldest_days": round(float(age_days.max()), 1),
            "fresh_days": fresh,
            "stale_days": stale,
        },
    )


# --------------------------------------------------------------------------
# accuracy (statistical outlier proxy)
# --------------------------------------------------------------------------
def _outlier_mask(values: pd.Series) -> pd.Series:
    """Boolean outlier mask for one numeric column.

    ``n >= 30``: robust z-score via MAD with |z| > 3.5.
    ``n < 30``: Tukey far-out fences (3 x IQR) - both MAD and 1.5 x IQR are
    too unstable on small samples and would flag clean values.
    """
    valid = values.notna()
    n_valid = int(valid.sum())
    if n_valid == 0:
        return pd.Series(False, index=values.index)
    if n_valid >= 30:
        med = float(values[valid].median())
        mad = float((values[valid] - med).abs().median())
        std = float(values[valid].std())
        scale = 0.6745 * mad if mad > 0 else std
        if not scale or np.isnan(scale):
            return pd.Series(False, index=values.index)
        z = (values - med).abs() / scale
        return valid & (z > _ROBUST_Z_LIMIT)
    q1, q3 = values[valid].quantile([0.25, 0.75])
    iqr = float(q3 - q1)
    if iqr <= 0:
        return pd.Series(False, index=values.index)
    lo, hi = float(q1) - _TUKEY_FAR_IQR * iqr, float(q3) + _TUKEY_FAR_IQR * iqr
    return valid & ((values < lo) | (values > hi))


def accuracy(df: pd.DataFrame, cfg: DqfsConfig, specs: ColumnSpecs) -> FactorResult:
    """Score absence of outliers (robust MAD z-score / Tukey fences)."""
    if df.empty:
        return FactorResult("accuracy", 0.0, np.zeros(0))
    numeric = [c for c in specs.numeric if c in df.columns]
    if not numeric:
        return FactorResult(
            "accuracy", None, np.zeros(len(df)),
            details={"note": "no numeric columns to evaluate"},
        )
    n = len(df)
    col_scores: dict[str, float] = {}
    row_bad = np.zeros(n, dtype=np.float64)
    outlier_cells = 0
    evaluable = 0
    for col in numeric:
        values = pd.to_numeric(df[col], errors="coerce").astype(float)
        outliers = _outlier_mask(values)
        n_valid = int(values.notna().sum())
        col_scores[col] = float(1.0 - outliers.sum() / max(n_valid, 1))
        row_bad += outliers.to_numpy(dtype=np.float64)
        outlier_cells += int(outliers.sum())
        evaluable += n_valid
    score = 1.0 - outlier_cells / max(evaluable, 1)
    return FactorResult(
        name="accuracy",
        score=float(score),
        row_contribution=_clip(row_bad / len(numeric)),
        column_scores=col_scores,
        details={
            "outlier_cells": outlier_cells,
            "evaluable_cells": evaluable,
            "method": "MAD robust z (|z|>3.5, n>=30) / Tukey 3xIQR (n<30)",
        },
    )


FACTOR_FUNCS: dict[str, Callable[[pd.DataFrame, DqfsConfig, ColumnSpecs], FactorResult]] = {
    "completeness": completeness,
    "validity": validity,
    "uniqueness": uniqueness,
    "consistency": consistency,
    "timeliness": timeliness,
    "accuracy": accuracy,
}


def evaluate_factors(
    df: pd.DataFrame, cfg: DqfsConfig, specs: ColumnSpecs
) -> dict[str, FactorResult]:
    """Evaluate every factor in :data:`FACTOR_ORDER`.

    Args:
        df: Loaded dataset.
        cfg: Resolved pipeline configuration.
        specs: Column classification.

    Returns:
        Mapping of factor name to :class:`FactorResult`, in canonical order.
    """
    results: dict[str, FactorResult] = {}
    for name in cfg.raw.get("factor_order", FACTOR_ORDER):
        func = FACTOR_FUNCS[name]
        try:
            results[name] = func(df, cfg, specs)
        except Exception as exc:
            # Deliberate containment policy: a single broken factor must not
            # kill the run. The factor is marked N/A, the traceback is logged
            # at WARNING with full context, and the report shows the error.
            logger.warning("factor %s failed: %s", name, exc, exc_info=True)
            results[name] = FactorResult(
                name,
                None,
                np.zeros(len(df)),
                details={"error_type": type(exc).__name__, "error": str(exc)},
            )
    return results
