"""Scoring pipeline: fuse factors, deep ensemble, weights, gates and DQS.

:func:`run_scoring` is the heart of DQFS. It evaluates the six rule-based
factors, trains the deep quality ensemble, derives ML-driven factor weights
(IC blended with deep permutation importance), computes the composite
Dataset Quality Score (DQS, 0-100), evaluates gates and returns a fully
JSON-serializable result dictionary ready for the report generators.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from dqfs import __version__
from dqfs.config import DqfsConfig
from dqfs.errors import ScoringError
from dqfs.factors import evaluate_factors
from dqfs.gates import evaluate_gates, overall_verdict
from dqfs.io_loader import ColumnSpecs, infer_column_kinds
from dqfs.ml.ensemble import DeepQualityEnsemble
from dqfs.ml.importance import blend_factor_weights

logger = logging.getLogger(__name__)

_WORST_ROWS_SAMPLE = 10


@dataclass
class ScoreRun:
    """Outcome of one full scoring run."""

    result: dict[str, Any]
    verdict: str
    dqs: float


def _column_report(
    df: pd.DataFrame,
    specs: ColumnSpecs,
    factor_results: dict[str, Any],
    column_importance: dict[str, float],
) -> list[dict[str, Any]]:
    """Per-column summary combining every factor's evidence."""
    kind_of = {}
    for kind in ("numeric", "categorical", "datetime", "text"):
        for col in getattr(specs, kind):
            kind_of[col] = kind
    rows: list[dict[str, Any]] = []
    for col in df.columns:
        completeness = factor_results["completeness"].column_scores.get(col)
        validity = factor_results["validity"].column_scores.get(col)
        accuracy = factor_results["accuracy"].column_scores.get(col)
        missing = round(float(1.0 - completeness), 4) if completeness is not None else None
        invalid = round(float(1.0 - validity), 4) if validity is not None else None
        outlier = round(float(1.0 - accuracy), 4) if accuracy is not None else None
        rows.append({
            "column": str(col),
            "kind": kind_of.get(col, "other"),
            "missing_ratio": missing,
            "invalid_ratio": invalid,
            "outlier_ratio": outlier,
            "deep_importance": round(float(column_importance.get(col, 0.0)), 4),
        })
    return rows


def _worst_rows(
    fused_badness: np.ndarray,
    factor_results: dict[str, Any],
    limit: int = _WORST_ROWS_SAMPLE,
) -> list[dict[str, Any]]:
    """Top-N suspicious rows with their per-factor contributions."""
    if len(fused_badness) == 0:
        return []
    order = np.argsort(fused_badness)[::-1][:limit]
    available = [n for n, r in factor_results.items() if r.available]
    out: list[dict[str, Any]] = []
    for idx in order:
        out.append({
            "row_index": int(idx),
            "badness": round(float(fused_badness[idx]), 4),
            "contributions": {
                n: round(float(factor_results[n].row_contribution[idx]), 3)
                for n in available
            },
        })
    return out


def run_scoring(
    df: pd.DataFrame,
    cfg: DqfsConfig,
    source_label: str,
    target: str | None = None,
) -> ScoreRun:
    """Run the complete scoring pipeline on ``df``.

    Args:
        df: Loaded dataset.
        cfg: Resolved configuration.
        source_label: Human-readable data source description.
        target: Optional supervised target column for the deep ensemble.

    Returns:
        A :class:`ScoreRun` whose ``result`` dict is JSON-serializable.

    Raises:
        ScoringError: If the pipeline fails in a non-recoverable way.
    """
    started = time.perf_counter()
    logger.info("scoring %s (%d rows x %d cols)", source_label, len(df), df.shape[1])
    try:
        specs = infer_column_kinds(df)
        factor_results = evaluate_factors(df, cfg, specs)

        # rule-based aggregate badness feeds the meta-learner calibration
        available = [r for r in factor_results.values() if r.available]
        if available and len(df):
            rule_badness = np.clip(
                np.mean([r.row_contribution for r in available], axis=0), 0.0, 1.0
            )
        else:
            rule_badness = None

        ensemble = DeepQualityEnsemble(cfg).fit(
            df, specs, target=target, rule_badness=rule_badness
        )
        fused_badness = ensemble.row_badness
        weights_info = blend_factor_weights(
            factor_results, fused_badness, ensemble.column_importance, specs, cfg
        )

        # composite DQS: weighted mean of available factor scores, 0-100
        if weights_info:
            dqs = 100.0 * sum(
                info["weight"] * factor_results[name].score  # type: ignore[index]
                for name, info in weights_info.items()
            )
        else:
            dqs = 0.0
        dqs = float(np.clip(dqs, 0.0, 100.0))

        gates = evaluate_gates(factor_results, weights_info, dqs, cfg)
        verdict = overall_verdict(gates, cfg)

        factors_payload: dict[str, Any] = {}
        for name, res in factor_results.items():
            entry: dict[str, Any] = {
                "score": round(float(res.score), 4) if res.score is not None else None,
                "status": next(
                    (g.status for g in gates if g.name == f"factor:{name}"), "n/a"
                ),
                "details": res.details,
            }
            if name in weights_info:
                entry.update({
                    "weight": weights_info[name]["weight"],
                    "ic": weights_info[name]["ic"],
                    "deep_importance": weights_info[name]["deep_importance"],
                })
            factors_payload[name] = entry

        run_id = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
        result: dict[str, Any] = {
            "schema_version": 1,
            "run_id": run_id,
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "tool": {"name": "dqfs", "version": __version__},
            "dataset": {
                "name": cfg.dataset.get("name") or source_label,
                "source": source_label,
                "rows": int(len(df)),
                "columns": int(df.shape[1]),
                "target": target,
            },
            "dqs": round(dqs, 2),
            "verdict": verdict,
            "factors": factors_payload,
            "gates": [
                {"name": g.name, "status": g.status, "value": g.value,
                 "threshold": g.threshold}
                for g in gates
            ],
            "columns": _column_report(df, specs, factor_results, ensemble.column_importance),
            "worst_rows": _worst_rows(fused_badness, factor_results),
            "model_card": ensemble.model_card,
            "thresholds": {
                "factor_pass": cfg.thresholds.factor_pass,
                "factor_warn": cfg.thresholds.factor_warn,
                "composite_pass": cfg.thresholds.composite_pass,
                "composite_warn": cfg.thresholds.composite_warn,
                "critical_factors": cfg.critical_factors,
            },
            "duration_seconds": round(time.perf_counter() - started, 2),
        }
        logger.info(
            "scoring done: DQS %.1f, verdict %s (%.1fs)",
            dqs, verdict, result["duration_seconds"],
        )
        return ScoreRun(result=result, verdict=verdict, dqs=dqs)
    except ScoringError:
        raise
    except Exception as exc:
        raise ScoringError(f"scoring pipeline failed: {exc}") from exc


def gate_summary(verdict: str) -> str:
    """One-line summary used by the CLI."""
    return f"verdict={verdict}"
