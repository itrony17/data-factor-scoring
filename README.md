# DQFS - Data Quality Factor Scoring

Deep-ML data quality factor scoring pipeline with GitHub-native reporting.

> Full documentation is under active development - see `dqfs.yaml` for the
> configuration surface and `python -m dqfs --help` once installed.

## Quick start

```bash
pip install -e ".[dev]"
dqfs demo && dqfs score --dsn sqlite:///demo/quality_demo.db --table customers
```

## Run tests

```bash
pytest -v
```

Tests run fully offline - no database service or API keys required.

## License

MIT - see [LICENSE](LICENSE).
