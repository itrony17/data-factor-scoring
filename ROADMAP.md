# Roadmap

DQFS is under active maintenance. This document records where the project is
headed so contributors and users know what to expect and when.

## Released

| Version | Date       | Highlights                                                        |
|---------|------------|-------------------------------------------------------------------|
| 1.0.0   | 2026-11-05 | Six quality factors, NumPy deep ensemble, gates, reporters, CLI   |
| 1.1.0   | 2026-11-06 | Structured logging, mypy + pip-audit + coverage gates in CI, boundary validation, `doctor --json` |

## 1.1.x — patch line (bugfixes only)

- Regression fixes reported against the 1.1.0 release.
- Documentation corrections and example fixes.
- Dependency bumps via Dependabot (Actions and Python packages).

## 1.2.0 — next feature release

- **Parquet and JSON Lines input** — extend `dqfs score` beyond CSV with
  `--parquet PATH` and `--jsonl PATH`, sharing the existing factor pipeline.
- **DQS badge endpoint** — publish an SVG badge per repository from the
  `gh-results` branch so READMEs can display the live quality score.
- **Configurable timeliness reference** — let `dqfs.yaml` pin the "now"
  reference date so timeliness scoring is reproducible in tests and CI.
- **Sample dataset gallery** — curated broken/clean datasets under
  `examples/` for smoke-testing and docs.

## 2.0.0 — exploration

- **Plugin factor API** — register custom quality factors in `dqfs.yaml`
  without touching core code.
- **Polars backend** — optional backend for datasets beyond ~10M rows.
- **Server mode** — a small REST surface around `dqfs score` for teams that
  want scheduled scoring as a service.

## Maintenance policy

- Bugfix releases (1.1.x) are cut as needed; feature releases (1.2.0, 2.0.0)
  ship when the items above are complete — dates are targets, not promises.
- Only the latest minor line receives fixes; breaking changes land only in
  major versions and are announced in the CHANGELOG.
- Every release gets a tag, a GitHub Release with notes, and a CHANGELOG
  entry before it is announced.
