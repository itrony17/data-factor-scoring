"""Tests for dqfs.io_loader - sources, DSN masking and column inference."""

from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
import pytest

from dqfs.errors import DataLoadError
from dqfs.io_loader import _short_dsn, infer_column_kinds, load_table


@pytest.fixture()
def demo_db(tmp_path):
    db_path = tmp_path / "t.db"
    df = pd.DataFrame(
        {
            "id": range(10),
            "score": np.linspace(0.0, 1.0, 10),
            "country": ["US", "GB"] * 5,
            "when": pd.date_range("2026-01-01", periods=10).astype(str),
        }
    )
    with sqlite3.connect(db_path) as conn:
        df.to_sql("items", conn, index=False)
    return f"sqlite:///{db_path}"


def test_load_from_sqlite(demo_db) -> None:
    df, label = load_table(dsn=demo_db, table="items")
    assert len(df) == 10
    assert label.startswith("db:sqlite:///")
    assert label.endswith("[items]")


def test_load_query_from_sqlite(demo_db) -> None:
    df, label = load_table(dsn=demo_db, query="SELECT id FROM items")
    assert list(df.columns) == ["id"]
    assert label.startswith("sql:")


def test_load_csv(tmp_path) -> None:
    p = tmp_path / "x.csv"
    pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}).to_csv(p, index=False)
    df, label = load_table(csv_path=str(p))
    assert len(df) == 2
    assert label == f"csv:{p}"


def test_no_source_raises() -> None:
    with pytest.raises(DataLoadError, match="provide --dsn"):
        load_table()


def test_mutually_exclusive_sources(tmp_path) -> None:
    p = tmp_path / "x.csv"
    p.write_text("a\n1\n", encoding="utf-8")
    with pytest.raises(DataLoadError, match="mutually exclusive"):
        load_table(dsn="sqlite:///:memory:", table="t", csv_path=str(p))


def test_missing_csv_raises(tmp_path) -> None:
    with pytest.raises(DataLoadError, match="not found"):
        load_table(csv_path=str(tmp_path / "nope.csv"))


def test_bad_dsn_raises_clean_error() -> None:
    # an unsupported backend is rejected at the validation boundary before
    # any driver is loaded (a malformed sqlite path would create a stray file)
    with pytest.raises(DataLoadError, match="unsupported database backend"):
        load_table(dsn="notavaliddriver://host/db", table="t")


def test_short_dsn_masks_credentials() -> None:
    masked = _short_dsn("postgresql://user:s3cret@db.host:5432/wh")
    assert "s3cret" not in masked
    assert masked == "postgresql://db.host:5432/wh"


def test_infer_column_kinds(demo_db) -> None:
    df, _ = load_table(dsn=demo_db, table="items")
    df["essay"] = [f"essay {i} " + "word " * 20 for i in range(len(df))]
    specs = infer_column_kinds(df)
    assert set(specs.numeric) == {"id", "score"}
    assert specs.categorical == ["country"]
    assert specs.datetime == ["when"]
    assert specs.text == ["essay"]
    assert specs.all_columns == ["id", "score", "country", "when", "essay"]


def test_infer_kinds_empty_column_is_categorical() -> None:
    df = pd.DataFrame({"blank": [None, None], "num": [1.0, 2.0]})
    specs = infer_column_kinds(df)
    assert specs.categorical == ["blank"]
    assert specs.numeric == ["num"]
