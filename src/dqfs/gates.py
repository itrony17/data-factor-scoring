"""Quality gates: pass/warn/fail thresholds and the overall verdict.

Gates are evaluated per factor and for the composite DQS. Critical factors
(configurable, default: completeness and validity) escalate any factor-level
failure to a dataset-level FAIL even when the composite score would pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from dqfs.config import DqfsConfig
from dqfs.factors import FactorResult

STATUS_PASS = "pass"
STATUS_WARN = "warn"
STATUS_FAIL = "fail"
STATUS_NA = "n/a"


@dataclass
class GateResult:
    """Outcome of one gate evaluation."""

    name: str
    status: str
    value: float | None
    threshold: str


def factor_status(score: float, cfg: DqfsConfig) -> str:
    """Map a factor score in [0, 1] to pass/warn/fail."""
    if score >= cfg.thresholds.factor_pass:
        return STATUS_PASS
    if score >= cfg.thresholds.factor_warn:
        return STATUS_WARN
    return STATUS_FAIL


def composite_status(dqs: float, cfg: DqfsConfig) -> str:
    """Map the composite DQS (0-100) to pass/warn/fail."""
    if dqs >= cfg.thresholds.composite_pass:
        return STATUS_PASS
    if dqs >= cfg.thresholds.composite_warn:
        return STATUS_WARN
    return STATUS_FAIL


def evaluate_gates(
    factor_results: dict[str, FactorResult],
    weights: dict[str, dict[str, Any]],
    dqs: float,
    cfg: DqfsConfig,
) -> list[GateResult]:
    """Evaluate all factor gates plus the composite DQS gate."""
    gates: list[GateResult] = []
    for name, res in factor_results.items():
        if not res.available:
            gates.append(GateResult(f"factor:{name}", STATUS_NA, None, "n/a"))
            continue
        status = factor_status(float(res.score), cfg)  # type: ignore[arg-type]
        gates.append(
            GateResult(
                name=f"factor:{name}",
                status=status,
                value=round(float(res.score), 4),  # type: ignore[arg-type]
                threshold=f">= {cfg.thresholds.factor_warn} (warn), "
                f">= {cfg.thresholds.factor_pass} (pass)",
            )
        )
    gates.append(
        GateResult(
            name="composite_dqs",
            status=composite_status(dqs, cfg),
            value=round(dqs, 2),
            threshold=f">= {cfg.thresholds.composite_warn} (warn), "
            f">= {cfg.thresholds.composite_pass} (pass)",
        )
    )
    return gates


def overall_verdict(gates: list[GateResult], cfg: DqfsConfig) -> str:
    """Derive the dataset verdict from gate results.

    FAIL if the composite gate fails or any critical factor fails;
    WARN if any other gate warns/fails or a critical factor is N/A
    (an unevaluated critical factor is a blind spot, not a pass);
    PASS otherwise.
    """
    statuses = {g.name: g.status for g in gates}
    if statuses.get("composite_dqs") == STATUS_FAIL:
        return "FAIL"
    for factor in cfg.critical_factors:
        if statuses.get(f"factor:{factor}") == STATUS_FAIL:
            return "FAIL"
    for factor in cfg.critical_factors:
        if statuses.get(f"factor:{factor}") == STATUS_NA:
            return "WARN"
    if any(s in (STATUS_FAIL, STATUS_WARN) for s in statuses.values()):
        return "WARN"
    return "PASS"
