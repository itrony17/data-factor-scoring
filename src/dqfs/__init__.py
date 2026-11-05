"""DQFS - Data Quality Factor Scoring.

A production-grade CLI pipeline that scores dataset quality factors
(completeness, validity, uniqueness, consistency, timeliness, accuracy),
derives factor weights from a deep tabular ML ensemble, applies pass/fail
quality gates and publishes results to a GitHub repository.
"""

__version__ = "1.0.0"

__all__ = ["__version__"]
