"""ML-driven factor weighting: blend statistical IC with deep importance.

For each available factor we compute:

* **IC** - the Spearman rank correlation between the factor's per-row
  badness contributions and the fused row-badness signal (which carries
  deep-ensemble and, when available, supervised-target evidence). A factor
  that fires exactly where the deep ensemble sees anomalies earns a high
  |IC|;
* **deep importance** - the mean per-column permutation importance of the
  columns that feed the factor.

The final weight is ``ic_blend * |IC| + (1 - ic_blend) * deep_importance``,
floored at ``min_weight`` and normalized to sum to 1.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from dqfs.config import DqfsConfig
from dqfs.factors import FactorResult
from dqfs.io_loader import ColumnSpecs

logger = logging.getLogger(__name__)


def _rank(values: np.ndarray) -> np.ndarray:
    """Average ranks (ties get the mean rank)."""
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    # average tie ranks
    sorted_vals = values[order]
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return ranks


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation between two equal-length vectors."""
    if len(a) != len(b) or len(a) < 3:
        return 0.0
    ra, rb = _rank(np.asarray(a, float)), _rank(np.asarray(b, float))
    sa, sb = ra.std(), rb.std()
    if sa < 1e-12 or sb < 1e-12:
        return 0.0
    return float(((ra - ra.mean()) * (rb - rb.mean())).mean() / (sa * sb))


def _factor_columns(name: str, res: FactorResult, specs: ColumnSpecs, cfg: DqfsConfig) -> list[str]:
    """Columns whose deep importance feeds a factor's weight."""
    all_cols = specs.all_columns
    fcfg = cfg.factors.get(name, {})
    if name == "completeness":
        return all_cols
    if name == "validity":
        cols = list((fcfg.get("constraints") or {}).keys())
        cols += specs.numeric
        return cols or all_cols
    if name == "uniqueness":
        keys = [k for k in (fcfg.get("key_columns") or []) if k in all_cols]
        return keys or all_cols
    if name == "consistency":
        cols = [c for c in (fcfg.get("format_columns") or []) if c in all_cols]
        for chk in (fcfg.get("checks") or []):
            for side in ("left", "right"):
                if str(chk.get(side)) in all_cols:
                    cols.append(str(chk[side]))
        return cols or all_cols
    if name == "timeliness":
        col = fcfg.get("column")
        return [col] if col in all_cols else (specs.datetime or all_cols)
    if name == "accuracy":
        return specs.numeric or all_cols
    return all_cols


def blend_factor_weights(
    factor_results: dict[str, FactorResult],
    row_badness: np.ndarray,
    column_importance: dict[str, float],
    specs: ColumnSpecs,
    cfg: DqfsConfig,
) -> dict[str, dict[str, Any]]:
    """Compute normalized ML-driven factor weights.

    Returns:
        Mapping factor -> ``{"weight", "ic", "deep_importance", "score"}``
        for every available factor; weights sum to 1.
    """
    available = {n: r for n, r in factor_results.items() if r.available}
    if not available:
        return {}
    ic_blend = cfg.weights.ic_blend
    min_weight = cfg.weights.min_weight

    raw: dict[str, float] = {}
    info: dict[str, dict[str, Any]] = {}
    for name, res in available.items():
        same_len = len(row_badness) == len(res.row_contribution)
        ic = spearman(res.row_contribution, row_badness) if same_len else 0.0
        cols = _factor_columns(name, res, specs, cfg)
        imps = [column_importance[c] for c in cols if c in column_importance]
        deep = float(np.mean(imps)) if imps else 0.0
        w = ic_blend * abs(ic) + (1.0 - ic_blend) * deep
        raw[name] = max(w, min_weight)
        info[name] = {
            "ic": round(float(ic), 4),
            "deep_importance": round(deep, 4),
            "columns": cols,
        }

    total = sum(raw.values())
    weights = {n: w / total for n, w in raw.items()}
    for name, extra in info.items():
        extra["weight"] = round(float(weights[name]), 4)
        extra["score"] = round(float(available[name].score), 4)  # type: ignore[arg-type]
    return info
