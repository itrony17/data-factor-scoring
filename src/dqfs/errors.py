"""Typed exception hierarchy for DQFS.

Every failure mode in the pipeline maps to a specific exception type so that
callers (CLI, tests, CI) can react precisely instead of catching bare
``Exception``. The CLI translates these into documented exit codes.
"""

from __future__ import annotations


class DqfsError(Exception):
    """Base class for all DQFS errors (exit code 1)."""


class ConfigError(DqfsError):
    """Invalid or unreadable configuration (exit code 2)."""


class DataLoadError(DqfsError):
    """Dataset could not be loaded from DSN/query/CSV (exit code 3)."""


class FactorError(DqfsError):
    """A quality factor crashed during evaluation.

    The scoring pipeline catches this per factor, marks the factor as N/A,
    and keeps the run alive; the incident is logged with a full traceback.
    """


class ScoringError(DqfsError):
    """Composite scoring or ML ensemble failure (exit code 4)."""


class ReportError(DqfsError):
    """Report rendering (HTML/Excel/JSON) failure (exit code 5)."""


class PublishError(DqfsError):
    """GitHub publishing failure - auth, network, API errors (exit code 6)."""


class GateFailureError(DqfsError):
    """Quality gates failed and ``--fail-on-gate`` was set (exit code 7).

    Attributes:
        verdict: The gate verdict that triggered the failure.
    """

    def __init__(self, verdict: str, message: str) -> None:
        super().__init__(message)
        self.verdict = verdict
