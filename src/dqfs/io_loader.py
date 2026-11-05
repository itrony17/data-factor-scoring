"""Data access layer: load scoring inputs from databases or CSV files.

Databases are addressed through SQLAlchemy DSNs, e.g.::

    sqlite:///demo/quality_demo.db
    postgresql+psycopg2://user:pass@host:5432/warehouse

The loader also classifies columns into kinds (numeric / categorical /
datetime / text) that drive both the factor engine and the ML tokenizer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

from dqfs.errors import DataLoadError

logger = logging.getLogger(__name__)

#: Minimum share of parseable values for an object column to be treated as
#: containing datetimes.
_DATETIME_PARSE_THRESHOLD = 0.8

#: Sample size used for datetime sniffing on object columns.
_SNIFF_SAMPLE = 200


@dataclass
class ColumnSpecs:
    """Column classification used by the factor engine and ML tokenizer."""

    numeric: list[str] = field(default_factory=list)
    categorical: list[str] = field(default_factory=list)
    datetime: list[str] = field(default_factory=list)
    text: list[str] = field(default_factory=list)

    @property
    def all_columns(self) -> list[str]:
        """All classified columns in a stable order."""
        return self.numeric + self.categorical + self.datetime + self.text


def load_table(
    dsn: str | None = None,
    table: str | None = None,
    query: str | None = None,
    csv_path: str | None = None,
) -> tuple[pd.DataFrame, str]:
    """Load a DataFrame from a database table, SQL query or CSV file.

    Exactly one source must be provided.

    Args:
        dsn: SQLAlchemy connection string (required for ``table``/``query``).
        table: Database table name to read.
        query: Raw SQL ``SELECT`` query to execute.
        csv_path: Path to a CSV file (fallback source for quick tests).

    Returns:
        Tuple of ``(dataframe, source_label)`` where ``source_label`` is a
        human-readable description used in reports.

    Raises:
        DataLoadError: If the source arguments are inconsistent or the read
            fails.
    """
    provided = [bool(dsn and (table or query)), bool(csv_path)]
    if sum(provided) == 0:
        raise DataLoadError("provide --dsn with --table/--query, or --csv")
    if sum(provided) > 1:
        raise DataLoadError("database and CSV sources are mutually exclusive")

    if csv_path:
        p = Path(csv_path)
        if not p.exists():
            raise DataLoadError(f"CSV file not found: {p}")
        try:
            df = pd.read_csv(p)
        except Exception as exc:
            raise DataLoadError(f"failed to parse CSV {p}: {exc}") from exc
        logger.info("loaded %d rows x %d cols from CSV %s", len(df), df.shape[1], p)
        return df, f"csv:{p}"

    engine = None
    try:
        engine = create_engine(dsn, pool_pre_ping=True)
        if query:
            df = pd.read_sql(text(query), engine)
            label = f"sql:{_short_dsn(dsn)}"
        else:
            df = pd.read_sql_table(table, engine)  # type: ignore[arg-type]
            label = f"db:{_short_dsn(dsn)}[{table}]"
        logger.info("loaded %d rows x %d cols from %s", len(df), df.shape[1], label)
        return df, label
    except DataLoadError:
        raise
    except Exception as exc:
        raise DataLoadError(f"failed to load data from {dsn}: {exc}") from exc
    finally:
        if engine is not None:
            engine.dispose()


def _short_dsn(dsn: str) -> str:
    """Return a DSN with credentials stripped, safe for logs and reports."""
    if "://" not in dsn:
        return dsn
    scheme, rest = dsn.split("://", 1)
    if "@" in rest:
        rest = rest.split("@", 1)[1]
    return f"{scheme}://{rest}"


def _sniff_datetime(series: pd.Series) -> bool:
    """Return True when an object column looks like a datetime column."""
    sample = series.dropna().astype(str).head(_SNIFF_SAMPLE)
    if sample.empty:
        return False
    parsed = pd.to_datetime(sample, errors="coerce", utc=True, format="mixed")
    return parsed.notna().mean() >= _DATETIME_PARSE_THRESHOLD


def infer_column_kinds(df: pd.DataFrame) -> ColumnSpecs:
    """Classify DataFrame columns into numeric / categorical / datetime / text.

    The classification heuristics are intentionally conservative:

    * real numeric dtypes (int/float) are numeric;
    * booleans and low-cardinality object columns are categorical;
    * object columns whose sample mostly parses as timestamps are datetime;
    * high-cardinality long free-text columns are marked ``text`` and are
      factor-scored but excluded from the ML tokenizer.

    Args:
        df: Input DataFrame.

    Returns:
        :class:`ColumnSpecs` with the column name lists.
    """
    specs = ColumnSpecs()
    n = max(len(df), 1)
    for col in df.columns:
        series = df[col]
        kind = series.dtype
        if pd.api.types.is_bool_dtype(kind):
            specs.categorical.append(col)
        elif pd.api.types.is_numeric_dtype(kind):
            specs.numeric.append(col)
        elif pd.api.types.is_datetime64_any_dtype(kind):
            specs.datetime.append(col)
        else:
            non_null = series.dropna()
            if non_null.empty:
                specs.categorical.append(col)
                continue
            if _sniff_datetime(series):
                specs.datetime.append(col)
                continue
            n_unique = non_null.nunique()
            avg_len = float(np.mean([len(str(v)) for v in non_null.head(500)]))
            if n_unique / n > 0.5 and avg_len > 32:
                specs.text.append(col)
            else:
                specs.categorical.append(col)
    logger.debug(
        "column kinds: numeric=%s categorical=%s datetime=%s text=%s",
        specs.numeric, specs.categorical, specs.datetime, specs.text,
    )
    return specs
