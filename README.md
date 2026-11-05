# DQFS - Data Quality Factor Scoring

[![CI](https://github.com/itrony17/data-factor-scoring/actions/workflows/ci.yml/badge.svg)](https://github.com/itrony17/data-factor-scoring/actions/workflows/ci.yml)
[![Scoring pipeline](https://github.com/itrony17/data-factor-scoring/actions/workflows/score.yml/badge.svg)](https://github.com/itrony17/data-factor-scoring/actions/workflows/score.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](pyproject.toml)
[![Code style: ruff](https://img.shields.io/badge/lint-ruff-261230)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Tests: offline](https://img.shields.io/badge/tests-offline%20%2B%20mocked-success)](#running-tests)

**DQFS** is a production-grade Python CLI that scores dataset quality across
six statistical factors, derives the factor weights from a **deep tabular ML
ensemble** (TabNet-style attention + residual MLP + stacked meta-learner),
applies **pass/warn/fail quality gates**, and publishes the resulting
dashboard and machine-readable artifacts **to a GitHub repository** - via a
one-command publisher, a GitHub Actions scoring workflow and GitHub Pages.

A full run produces: a composite **DQS 0-100**, a per-factor verdict table
with ML evidence (weight / IC / deep importance), a standalone **HTML
dashboard**, a five-sheet **Excel** workbook, flat **CSV** tables and a
**JSON** result with a complete model card. See
[`examples/sample_report.html`](examples/sample_report.html) for a real
output (DQS 92.0 / WARN on the demo database).

---

## Why DQFS

Typical data-quality scripts print a wall of null counts and stop there. DQFS
is built like a product instead:

- **Six factors with row-level evidence** - every factor contributes a
  per-row badness vector, not just a headline number.
- **ML-driven weights, not arbitrary ones** - factor weights come from the
  rank correlation (IC) between each factor's row evidence and the deep
  ensemble's anomaly signal, blended with per-column permutation importance
  from the trained networks. Configurable, reproducible, and reported in
  every artifact.
- **Real deep architecture, zero GPU, zero torch** - two hand-built NumPy
  networks (prior-relaxed sparse attention; pre-norm residual MLP with
  categorical embeddings) are pretrained with masked-cell denoising and
  verified by finite-difference gradient checks in the test suite.
- **Gates that behave like gates** - critical factors (configurable) force a
  dataset FAIL even when the composite passes, and `--fail-on-gate` turns the
  verdict into a CI exit code.
- **GitHub-native reporting** - results are committed atomically to a
  reporting branch through the Git data API, posted as a commit status, and
  can be served as a live Pages dashboard.

## Quick start (fresh clone)

```bash
git clone https://github.com/itrony17/data-factor-scoring.git
cd data-factor-scoring

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e ".[dev]"

# 1. generate a demo SQLite database (2.6k rows, realistic corruption)
dqfs demo

# 2. run the full scoring pipeline (deep ML enabled)
dqfs score \
  --config demo/dqfs-demo.yaml \
  --dsn sqlite:///demo/quality_demo.db \
  --table customers \
  --target churn \
  --dataset-name customers_demo

# 3. preview what publishing would do (no token needed)
dqfs publish --repo your-org/your-repo --branch gh-results --dry-run
```

`make install demo score` does the same via the Makefile.

## CLI reference

| Command | Purpose |
| --- | --- |
| `dqfs demo [--out demo/]` | Generate `demo/quality_demo.db` + a wired `demo/dqfs-demo.yaml` |
| `dqfs score --dsn URL --table T [--query SQL] [--csv F]` | Score a dataset (database or CSV source) |
| `dqfs score --target COL` | Also train supervised target heads out-of-fold |
| `dqfs score --out DIR --fail-on-gate` | Artifact directory; exit 7 when verdict is FAIL (CI gate) |
| `dqfs report [--results DIR]` | Re-render HTML/XLSX/CSV from `results/latest.json` |
| `dqfs publish --repo owner/name [--branch B]` | Commit result artifacts to the reporting branch |
| `dqfs publish --pages --pages-branch main` | Also push the dashboard to `docs/index.html` for GitHub Pages |
| `dqfs publish --status` | Post a `dqfs/quality-gate` commit status (PASS->success, WARN/FAIL->failure) |
| `dqfs doctor [--config F --dsn URL --table T]` | Self-check environment, config and data source |
| `dqfs init [--path dqfs.yaml]` | Write a starter config |

Global flags: `--version`, `-v/-vv` (info/debug logs), `--log-json`
(one JSON object per log record).

**Exit codes:** `0` success - `2` config - `3` data load - `4` scoring -
`5` report - `6` publish - `7` gate failure with `--fail-on-gate` -
`1` anything unexpected (full traceback in the log).

## How scoring works

```
                      +------------------------------+
  SQLite / Postgres   |  io_loader                   |   CSV fallback
  DSN/table/query --->|  column kind inference       |<-- --csv file
                      +---------------+--------------+
                                      |
        +-----------------------------+-----------------------------+
        v                                                           v
 +------+--------------+  row contributions   +--------------------+------+
 | rule-based factors  | ---------------------> |      deep ML ensemble     |
 | completeness        |        C[n x 6]        | tokenizer -> 2 branches   |
 | validity uniqueness |                        | (attention + residual MLP)|
 | consistency         |                        | masked-cell denoising     |
 | timeliness accuracy |                        | optional supervised heads |
 +------+--------------+                        +------------+--------------+
        |        fused badness = 0.5*rank(rules) + 0.5*rank(deep)
        |               meta-learner: ridge(deep signals -> rules)
        v
 +------+---------------+   weights ~ 0.6*|IC(factor, badness)|
 | ML-driven weighting  |            + 0.4*deep column importance
 +------+---------------+
        v
   DQS = 100 * sum(weight_f * score_f)  ->  gates  ->  PASS / WARN / FAIL
        v
   JSON + HTML dashboard + Excel + CSV  ->  `dqfs publish` -> GitHub
```

### The six factors

| Factor | Measures | Method |
| --- | --- | --- |
| `completeness` | missing cells | per-column null ratios, row = share of null cells |
| `validity` | type conformance + constraints | parseability, `min`/`max`/`regex`/`allowed` rules |
| `uniqueness` | duplicate rows and business keys | full-row + key duplication (keys weighted 2x) |
| `consistency` | cross-column logic, format uniformity | configured checks (`>=`, `==`, ...) + dominant pattern share |
| `timeliness` | freshness of a timestamp column | linear decay between `fresh_days` and `stale_days` (median age) |
| `accuracy` | statistical outliers | MAD robust z (`|z|>3.5`, n>=30), Tukey 3xIQR below 30 |

Every factor also reports per-column sub-scores and free-form details that
surface in the dashboard and workbook.

### Deep ML architecture

The engine is pure NumPy - deterministic, ~50ms/epoch on the demo data, and
small enough for any CI runner:

1. **Feature tokenizer** - numeric columns robustly scaled (median/IQR,
   clipped, missing -> 0 + observed mask); categorical columns frequency-
   ranked into bounded vocabularies (code 0 reserved for missing/unseen).
2. **Branch A - `AttentiveTabularNet`** (TabNet-style): a shared trunk feeds
   per-step attentive transformers; prior-relaxed sparse attention selects
   features multiplicatively (`z = mask * x`, prior relaxation `P <- P *
   (gamma - mask)`) and step outputs accumulate into the representation.
3. **Branch B - `ResidualMlpNet`**: categorical embeddings + numeric input,
   pre-norm residual blocks with inverted dropout.
4. **Masked-cell denoising pretraining** (TabMAE-style): ~18% of observed
   cells are corrupted and *hidden from the input*; both branches learn to
   reconstruct them. Rows the ensemble reconstructs badly are anomalies.
5. **Row-badness signals** - per-row mean z-scored reconstruction error,
   rank-normalized, one signal per branch.
6. **Supervised path (optional `--target`)** - each branch gets a target
   head trained out-of-fold with the trunk frozen (BCE for binary, MSE for
   regression); a ridge base learner over tokenized features adds a fourth
   OOF residual signal.
7. **Stacked meta-learner** - ridge maps the deep signals onto the
   rule-based badness (calibration/alignment diagnostic, R^2 and
   coefficients reported in the model card); the fused row badness blends
   rule evidence and deep evidence 50/50.
8. **ML-driven factor weights** - `w_f ~ ic_blend * |Spearman(C_f, badness)|
   + (1 - ic_blend) * deep_importance_f`, with a `min_weight` floor.
   Deep column importance comes from permuting each source column and
   measuring the reconstruction-loss increase.

Backpropagation for both branches is hand-derived and **verified by
finite-difference gradient checks** on every parameter tensor
(`tests/test_network.py`) - the checks caught and fixed three real bugs
during development, which is exactly why they exist.

## Configuration (`dqfs.yaml`)

Every key is optional; user values deep-merge over defaults. The committed
[`dqfs.yaml`](dqfs.yaml) documents the full surface. Highlights:

```yaml
thresholds:
  factor_pass: 0.85        # factor scores >= this pass
  factor_warn: 0.65
  composite_pass: 80.0     # DQS gates
  composite_warn: 65.0
critical_factors: [completeness, validity]   # failures force FAIL
weights:
  ic_blend: 0.6            # 1.0 = pure IC, 0.0 = pure deep importance
ml:
  epochs: 120              # denoising epochs per branch
  corruption_rate: 0.18    # masked-cell rate
  attention_steps: 3       # TabNet-style decision steps
  hidden_dim: 64
  oof_folds: 5             # supervised OOF splits
factors:
  timeliness: {column: last_purchase_at, fresh_days: 30, stale_days: 90}
  uniqueness: {key_columns: [id]}
  consistency:
    checks: [{left: age, op: '>=', right: '18'}]
    format_columns: [phone]
  validity:
    constraints:
      email: {regex: '^[^@\s]+@[^@\s]+\.[A-Za-z]{2,}$'}
      age: {min: 0, max: 120}
```

## GitHub publishing

### 1. Push-ready repository

The repo commits atomically to a **reporting branch** (default
`gh-results`), keeping `main` clean:

```bash
export GITHUB_TOKEN=...        # fine-grained PAT, Contents: read/write
dqfs publish --repo itrony17/data-factor-scoring --branch gh-results --status
```

Each publish creates `results/<run_id>/*` plus `results/latest/*` in one
commit, and (with `--status`) posts a `dqfs/quality-gate` commit status on
the run commit.

### 2. GitHub Pages dashboard

```bash
dqfs publish --repo itrony17/data-factor-scoring --pages --pages-branch main
```

pushes the newest dashboard to `docs/index.html`; enable *Settings -> Pages
-> Deploy from branch -> main /docs* once and every scoring run updates your
live quality dashboard.

### 3. Automated scoring pipeline (CI/CD)

[`.github/workflows/score.yml`](.github/workflows/score.yml) runs on every
push to `main`, weekly, or on demand: generates the demo database, scores it
with the full deep ensemble, uploads the artifacts and publishes them to
`gh-results` with a commit status - using the built-in `GITHUB_TOKEN`, no
secrets setup required. For a private database, add your DSN as a secret and
swap the `--dsn` argument.

### 4. Environment

`.env.example` documents the single environment variable the source reads:

```bash
cp .env.example .env    # then fill in GITHUB_TOKEN
```

## Running tests

The suite is fully **offline** - no database service, no network, no API
keys. Database tests use temporary SQLite files; GitHub publishing is tested
with mocked HTTP.

```bash
pytest -v            # or: make test - coverage-gated at 85% (current: 91.6%)
pytest tests/test_network.py -v   # gradient checks only
```

CI (`.github/workflows/ci.yml`) runs three jobs on every push and pull
request, installing from the pinned `requirements.txt`:

- **static** - `ruff check` + `mypy` type checking;
- **test** - the full suite with branch coverage on Python 3.10/3.11/3.12,
  failing below the 85% coverage floor;
- **audit** - `pip-audit --strict` against the pinned requirements, failing
  on any known CVE.

## Security

- **Secrets**: the only credential DQFS reads is `GITHUB_TOKEN`, taken from
  the environment (see `.env.example`); it is never logged, written to
  artifacts, or embedded in URLs.
- **Dependency audit**: every CI run executes `pip-audit --strict`; a known
  CVE in the pinned set fails the build. Exceptions require a documented
  rationale in the PR plus a `# nosec`-style comment in the lockfile commit
  and a linked tracking issue.
- **Input validation**: identifiers (table/column names) and SQLAlchemy
  DSNs are validated at the CLI/config boundary with explicit patterns
  before reaching the SQL or pandas layer.
- **Reporting**: please disclose vulnerabilities privately via GitHub's
  *Security -> Report a vulnerability*.

## Docker

```bash
docker build -t dqfs .
docker run --rm -v "$PWD:/data" dqfs score \
  --dsn sqlite:////data/app.db --table customers --out /data/results
```

## Project layout

```
data-factor-scoring/
├── .github/workflows/        # ci.yml (lint+test matrix), score.yml (pipeline)
├── examples/                 # committed sample report + JSON result
├── src/dqfs/
│   ├── cli.py                # argparse CLI, exit-code mapping
│   ├── config.py             # YAML config, deep-merged defaults, validation
│   ├── io_loader.py          # SQLAlchemy/CSV loading, column inference
│   ├── factors.py            # six rule-based factors + row evidence
│   ├── gates.py              # pass/warn/fail gates, critical escalation
│   ├── scoring.py            # pipeline fusion -> JSON result
│   ├── demo.py               # demo SQLite generator
│   ├── logging_setup.py      # console + JSON-lines logging
│   ├── errors.py             # typed exception hierarchy
│   ├── ml/
│   │   ├── tokenizer.py      # robust scaling + categorical vocabularies
│   │   ├── network.py        # attention + residual MLP branches, Adam
│   │   ├── denoise.py        # masked-cell denoising, signals, importance
│   │   ├── ensemble.py       # stacking orchestrator + meta-learner
│   │   └── importance.py     # IC x deep importance weight blending
│   ├── report/               # JSON / HTML dashboard / Excel / CSV writers
│   └── github/publisher.py   # Git-data-API publisher, status, Pages
├── tests/                    # 132 tests, offline, incl. gradient checks
├── dqfs.yaml                 # documented example config
└── requirements.txt          # pinned, tested dependency set
```

## Development

```bash
make install     # editable install with pytest + ruff
make lint        # ruff check src tests
make test        # pytest -v
make demo score  # full pipeline on the demo database
```

Conventions: small focused commits (feature + its tests together), typed
public functions, module docstrings, `ruff` clean. Releases are tagged
(`v1.0.0`).

## Roadmap

- [ ] Great Expectations / Soda-compatible check import
- [ ] Drift scoring between two snapshots (PSI / KS per column)
- [ ] PostgreSQL full integration tests in CI services
- [ ] Optional torch backend for GPU-scale datasets
- [ ] SARIF output for GitHub code scanning

## License

MIT - see [LICENSE](LICENSE).
