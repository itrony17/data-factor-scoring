"""Tests for dqfs.gates - threshold mapping, verdict escalation."""

from __future__ import annotations

import numpy as np
import pytest

from dqfs.config import load_config
from dqfs.factors import FactorResult
from dqfs.gates import (
    STATUS_FAIL,
    STATUS_NA,
    STATUS_PASS,
    STATUS_WARN,
    composite_status,
    evaluate_gates,
    factor_status,
    overall_verdict,
)


def _res(name: str, score: float | None) -> FactorResult:
    return FactorResult(name, score, np.zeros(3))


@pytest.fixture()
def cfg():
    return load_config(None)


def test_factor_status_bands(cfg) -> None:
    assert factor_status(0.99, cfg) == STATUS_PASS
    assert factor_status(0.85, cfg) == STATUS_PASS
    assert factor_status(0.84, cfg) == STATUS_WARN
    assert factor_status(0.65, cfg) == STATUS_WARN
    assert factor_status(0.64, cfg) == STATUS_FAIL


def test_composite_status_bands(cfg) -> None:
    assert composite_status(90.0, cfg) == STATUS_PASS
    assert composite_status(80.0, cfg) == STATUS_PASS
    assert composite_status(70.0, cfg) == STATUS_WARN
    assert composite_status(64.9, cfg) == STATUS_FAIL


def test_gate_list_covers_factors_and_composite(cfg) -> None:
    results = {
        "completeness": _res("completeness", 0.95),
        "validity": _res("validity", 0.70),
        "timeliness": _res("timeliness", None),
    }
    gates = evaluate_gates(results, {}, 88.0, cfg)
    by_name = {g.name: g for g in gates}
    assert by_name["factor:completeness"].status == STATUS_PASS
    assert by_name["factor:validity"].status == STATUS_WARN
    assert by_name["factor:timeliness"].status == STATUS_NA
    assert by_name["composite_dqs"].status == STATUS_PASS
    assert by_name["factor:timeliness"].value is None


def test_verdict_pass(cfg) -> None:
    results = {n: _res(n, 0.95) for n in ("completeness", "validity", "uniqueness")}
    gates = evaluate_gates(results, {}, 90.0, cfg)
    assert overall_verdict(gates, cfg) == "PASS"


def test_verdict_warn_on_noncritical_failure(cfg) -> None:
    results = {
        "completeness": _res("completeness", 0.95),
        "validity": _res("validity", 0.95),
        "uniqueness": _res("uniqueness", 0.40),  # non-critical, failing
    }
    gates = evaluate_gates(results, {}, 90.0, cfg)
    assert overall_verdict(gates, cfg) == "WARN"


def test_verdict_fail_on_critical_factor(cfg) -> None:
    results = {
        "completeness": _res("completeness", 0.95),
        "validity": _res("validity", 0.30),  # critical + failing
        "uniqueness": _res("uniqueness", 0.95),
    }
    gates = evaluate_gates(results, {}, 92.0, cfg)  # composite would pass
    assert overall_verdict(gates, cfg) == "FAIL"


def test_verdict_fail_on_composite(cfg) -> None:
    results = {n: _res(n, 0.70) for n in ("completeness", "validity", "uniqueness")}
    gates = evaluate_gates(results, {}, 50.0, cfg)
    assert overall_verdict(gates, cfg) == "FAIL"


def test_na_critical_factor_does_not_fail(cfg) -> None:
    results = {
        "completeness": _res("completeness", None),  # N/A, not a failure
        "validity": _res("validity", 0.95),
    }
    gates = evaluate_gates(results, {}, 90.0, cfg)
    assert overall_verdict(gates, cfg) == "WARN"  # N/A counts as non-pass
