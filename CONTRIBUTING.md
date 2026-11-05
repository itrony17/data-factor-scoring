# Contributing to DQFS

Thanks for your interest in improving DQFS! This guide covers everything
needed to go from a fresh clone to a merged pull request.

## Development setup

```bash
git clone https://github.com/itrony17/data-factor-scoring.git
cd data-factor-scoring

python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt                     # pinned, tested dependency set
pip install -e ".[dev]"                             # editable install + pytest/ruff/mypy/pip-audit

# sanity check: full pipeline on generated demo data
dqfs demo
dqfs score --config demo/dqfs-demo.yaml --dsn sqlite:///demo/quality_demo.db --table customers
```

## The quality bar

Every pull request must pass the same gates CI enforces:

| Gate | Command | Floor |
| --- | --- | --- |
| Lint | `ruff check src tests` | zero findings |
| Types | `mypy` | zero errors |
| Tests | `pytest` | 100% pass |
| Coverage | `pytest` (configured via `addopts`) | **>= 85%** branch coverage |
| Dependency audit | `pip-audit -r requirements.txt --strict` | zero known CVEs |

Run them all locally before pushing:

```bash
make lint test
```

## Conventions

- **Small, focused commits.** Each commit is one logical change and ships
  together with the tests that prove it. Avoid mixing formatting, refactors
  and features in one commit.
- **Conventional commit messages** (`feat:`, `fix:`, `docs:`, `test:`,
  `chore:`, `ci:`) with a short imperative summary and a body explaining
  *why* when it is not obvious.
- **Typed public functions** with docstrings in the existing style (Google-ish
  `Args:` / `Returns:` / `Raises:` sections).
- **Offline tests only.** The suite must never require network access, API
  tokens or a database service. Use temporary SQLite files and mock HTTP
  (see `tests/test_publisher.py`).
- **New dependency?** Justify it in the PR description, add it to
  `pyproject.toml` with a lower bound, and regenerate the lockfile from
  Python 3.10 (see below).

## Regenerating the lockfile

The pinned lockfile is generated on the **oldest supported interpreter** so
all CI matrix legs (3.10/3.11/3.12) can install it:

```bash
python3.10 -m venv .venv-lock && source .venv-lock/bin/activate
pip install .
pip freeze | grep -v '^dqfs' > requirements.txt
```

Verify on all supported versions before committing:

```bash
for v in 3.10 3.11 3.12; do
  python$v -m venv /tmp/check-$v && /tmp/check-$v/bin/pip install -q -r requirements.txt
done
```

## Reporting issues

Open a GitHub issue with:

1. the command you ran and the full output (redact credentials in DSNs);
2. your Python version and OS;
3. for scoring bugs, the smallest dataset that reproduces the problem.

## Security

See the [Security section of the README](README.md#security) and
`.env.example` for how tokens are handled. Please report vulnerabilities
privately via GitHub's *Security -> Report a vulnerability* rather than a
public issue.
