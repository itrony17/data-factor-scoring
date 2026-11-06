# Security Policy

## Supported versions

DQFS is a CLI tool; users are expected to run the latest release.

| Version | Supported          |
|---------|--------------------|
| 1.1.x   | :white_check_mark: |
| 1.0.x   | :x: (upgrade to 1.1.0) |
| < 1.0   | :x:                |

## Reporting a vulnerability

Please do **not** open a public issue for anything you believe is a security
problem.

Use GitHub's **private vulnerability reporting** for this repository:
*Security* tab → *Report a vulnerability*. This reaches the maintainer
directly and keeps details private until a fix ships.

You can expect:

1. **Acknowledgement** within 72 hours.
2. An assessment and (if confirmed) a fix target — critical issues are
   patched in a 1.1.x release as soon as possible.
3. Credit in the release notes if you would like it.

## Scope and hardening notes

DQFS reads local CSV/Excel/SQL sources and optional YAML configuration. Areas
that have already received hardening and remain in scope for reports:

- **Identifier and DSN validation** — column/table names and SQLAlchemy DSNs
  are pattern-validated before reaching SQL or pandas (added in 1.1.0).
  Bypasses or new injection vectors are in scope.
- **Publisher credentials** — `dqfs publish` uses Git credentials; any path
  that leaks tokens into logs, reports, or error messages is a security bug.
- **Report generation** — HTML/Excel reporters escape dataset-derived values;
  injection through crafted column names or cell values is in scope.

## Operational guidance

- Never commit credentials into `dqfs.yaml`; use environment variables.
- `requirements.txt` is fully pinned; CI runs `pip-audit --strict` on every
  push to catch known CVEs early.
