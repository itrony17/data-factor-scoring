"""Tests for boundary validation, structured logging and doctor --json."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from dqfs.config import load_config, validate_identifier
from dqfs.errors import DataLoadError
from dqfs.io_loader import load_table, validate_dsn


# ------------------------------------------------------------- identifiers
@pytest.mark.parametrize("name", ["age", "_x", "Table1", "schema.column", "a" * 128])
def test_valid_identifiers(name: str) -> None:
    assert validate_identifier(name) == name


@pytest.mark.parametrize(
    "name", ["", "1col", "has space", "semi;colon", "quote'", "a" * 129, "-lead"]
)
def test_invalid_identifiers(name: str) -> None:
    with pytest.raises(ValueError, match="invalid identifier"):
        validate_identifier(name)


def test_config_rejects_unsafe_column_names(tmp_path: Path) -> None:
    cfg_file = tmp_path / "dqfs.yaml"
    cfg_file.write_text(
        "factors:\n  timeliness:\n    column: 'drop; table users'\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="invalid factors"):
        load_config(cfg_file)


def test_config_accepts_safe_factor_bindings(tmp_path: Path) -> None:
    cfg_file = tmp_path / "dqfs.yaml"
    cfg_file.write_text(
        "factors:\n"
        "  timeliness: {column: last_purchase_at, fresh_days: 30}\n"
        "  uniqueness: {key_columns: [id]}\n"
        "  consistency:\n"
        "    checks: [{left: age, op: '>=', right: '18'}]\n"
        "    format_columns: [phone]\n"
        "  validity:\n"
        "    constraints:\n"
        "      age: {min: 0}\n",
        encoding="utf-8",
    )
    cfg = load_config(cfg_file)
    assert cfg.factors["timeliness"]["column"] == "last_purchase_at"


# ----------------------------------------------------------------- DSNs
def test_validate_dsn_accepts_known_backends() -> None:
    for dsn in (
        "sqlite:///x.db",
        "sqlite:///:memory:",
        "postgresql://u:p@h:5432/db",
        "postgresql+psycopg2://u:p@h/db",
        "mysql+pymysql://u:p@h/db",
    ):
        assert validate_dsn(dsn) == dsn


@pytest.mark.parametrize(
    "dsn", ["not-a-url", "oracle://h/sid", "redis://localhost", ""]
)
def test_validate_dsn_rejects_unknown_backends(dsn: str) -> None:
    with pytest.raises(DataLoadError):
        validate_dsn(dsn)


def test_load_table_rejects_bad_table_name(tmp_path: Path) -> None:
    db = tmp_path / "t.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE t (x INTEGER)")
    conn.commit()
    conn.close()
    with pytest.raises(DataLoadError, match="invalid table name"):
        load_table(dsn=f"sqlite:///{db}", table="t; DROP TABLE t")


