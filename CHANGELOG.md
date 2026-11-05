# Changelog

All notable changes to DQFS are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-11-06

### Added
- **Structured logging** - `--log-json` now emits records through
  `python-json-logger`; scoring and publishing attach structured fields
  (`run_id`, `dataset`, `verdict`, `dqs`, `duration_seconds`, `repo`,
  `branch`) so CI scoring runs can be correlated and audited.
- **Machine-readable health report** - `dqfs doctor --json` prints the full
  check list as JSON for automation and uptime monitoring.
- **Type checking** - the codebase is fully `mypy`-clean and CI runs mypy on
  every push and pull request (`static` job).
- **Dependency auditing** - a dedicated `audit` CI job runs `pip-audit
  --strict` against the pinned requirements on every push.
- **Coverage gating** - pytest runs with branch coverage and fails below an
  85% floor (current: 91.6%); the coverage report is uploaded per run.
- **Boundary validation** - configured column/table names and SQLAlchemy
  DSNs are validated with explicit patterns before they reach the SQL or
  pandas layer; unsafe identifiers (SQL injection payloads, overlong
  strings) are rejected with clear messages.
- `CHANGELOG.md` and `CONTRIBUTING.md` (this file and the contributor guide).

### Changed
- The pinned lockfile is now `requirements.txt` (standard name, fully
  pinned), regenerated from the **oldest supported Python (3.10)** so every
  CI matrix leg installs cleanly; verified on 3.10/3.11/3.12.
- `pyproject.toml` project URLs point at the real repository, changelog and
  contributing guide.

### Fixed
- `uniqueness` factor returned N/A on runs without configured key columns
  (an unbound local introduced by an internal refactor is now impossible).
- Consistency-check right-hand sides that are numeric literals (e.g. `'18'`)
  are no longer mistaken for column identifiers.

## [1.0.0] - 2026-11-05

### Added
- Six rule-based quality factors (completeness, validity, uniqueness,
  consistency, timeliness, accuracy) with per-row evidence vectors.
- Pure-NumPy deep ensemble: TabNet-style attention branch + residual MLP
  branch with categorical embeddings, masked-cell denoising pretraining,
  optional supervised heads, and a stacked meta-learner.
- ML-driven factor weights blending Spearman IC evidence with deep
  permutation importance (`ic_blend`, `min_weight`).
- Quality gates with critical-factor escalation and `--fail-on-gate` exit
  codes for CI pipelines.
- Reporters: standalone HTML dashboard, five-sheet Excel workbook, flat CSV
  and JSON with a complete model card.
- GitHub publisher: atomic commits to a reporting branch via the Git data
  API, commit statuses and a Pages dashboard mode.
- CLI (`dqfs demo|score|report|publish|doctor|init`), typed error hierarchy,
  structured logging, Docker image, Makefile, GitHub Actions workflows,
  Dependabot config and a 108-test offline suite with finite-difference
  gradient checks.
