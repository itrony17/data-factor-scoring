"""Feature tokenizer: turn raw DataFrames into ML-ready tensors.

Numeric columns are robustly scaled (median / IQR, clipped) with an explicit
missing indicator bit; categorical columns are index-encoded into bounded
vocabularies (code 0 is reserved for missing/unseen values) so each column
can own a trainable embedding table.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TokenizedBatch:
    """Tokenized view of a DataFrame consumed by the deep branches."""

    X_num: np.ndarray                 # (n, p) float64, robust-scaled, 0-filled
    M_num: np.ndarray                 # (n, p) bool, True where cell observed
    X_cat: np.ndarray                 # (n, q) int32 codes, 0 = missing/unseen
    M_cat: np.ndarray                 # (n, q) bool, True where cell observed
    numeric_columns: list[str] = field(default_factory=list)
    categorical_columns: list[str] = field(default_factory=list)
    vocab_sizes: list[int] = field(default_factory=list)

    @property
    def n_rows(self) -> int:
        """Number of rows."""
        return int(self.X_num.shape[0])

    @property
    def n_num(self) -> int:
        """Number of numeric features."""
        return int(self.X_num.shape[1])

    @property
    def n_cat(self) -> int:
        """Number of categorical features."""
        return int(self.X_cat.shape[1])


class FeatureTokenizer:
    """Fit robust numeric scaling and categorical vocabularies.

    Args:
        numeric: Numeric column names.
        categorical: Categorical column names.
        vocab_size: Maximum distinct codes per categorical column
            (including the reserved 0). Frequent categories win.
    """

    def __init__(
        self,
        numeric: list[str],
        categorical: list[str],
        vocab_size: int = 512,
        scale_clip: float = 8.0,
    ) -> None:
        if vocab_size < 4:
            raise ValueError("vocab_size must be >= 4")
        self.numeric = list(numeric)
        self.categorical = list(categorical)
        self.vocab_size = int(vocab_size)
        self.scale_clip = float(scale_clip)
        self._medians: dict[str, float] = {}
        self._scales: dict[str, float] = {}
        self._vocabs: dict[str, dict[str, int]] = {}

    # ------------------------------------------------------------------ fit
    def fit(self, df: pd.DataFrame) -> FeatureTokenizer:
        """Learn medians/scales and category vocabularies from ``df``."""
        for col in self.numeric:
            values = pd.to_numeric(df[col], errors="coerce").astype(float)
            valid = values.dropna()
            med = float(valid.median()) if len(valid) else 0.0
            iqr = float(valid.quantile(0.75) - valid.quantile(0.25)) if len(valid) else 0.0
            scale = iqr / 1.349
            if not scale or np.isnan(scale):
                scale = float(valid.std()) if len(valid) > 1 else 1.0
            if not scale or np.isnan(scale) or scale <= 0:
                scale = 1.0
            self._medians[col] = med
            self._scales[col] = scale
        for col in self.categorical:
            counts = df[col].dropna().astype(str).value_counts()
            # 1 code slot reserved for missing/unseen (0)
            kept = counts.head(self.vocab_size - 1)
            self._vocabs[col] = {v: i + 1 for i, v in enumerate(kept.index)}
        logger.debug(
            "tokenizer fitted: %d numeric, %d categorical", len(self.numeric), len(self.categorical)
        )
        return self

    # ------------------------------------------------------------- transform
    def transform(self, df: pd.DataFrame) -> TokenizedBatch:
        """Tokenize ``df`` into tensors (deterministic, no fitting)."""
        n = len(df)
        p, q = len(self.numeric), len(self.categorical)
        X_num = np.zeros((n, p), dtype=np.float64)
        M_num = np.zeros((n, p), dtype=bool)
        X_cat = np.zeros((n, q), dtype=np.int32)
        M_cat = np.zeros((n, q), dtype=bool)

        for j, col in enumerate(self.numeric):
            values = pd.to_numeric(df[col], errors="coerce").astype(float)
            med, scale = self._medians[col], self._scales[col]
            scaled = (values - med) / scale
            X_num[:, j] = np.clip(scaled.fillna(0.0).to_numpy(), -self.scale_clip, self.scale_clip)
            M_num[:, j] = values.notna().to_numpy()

        for j, col in enumerate(self.categorical):
            vocab = self._vocabs[col]
            codes = df[col].astype("string").map(vocab)  # NaN for missing/unseen
            X_cat[:, j] = codes.fillna(0).astype(np.int32).to_numpy()
            M_cat[:, j] = codes.notna().to_numpy()

        vocab_sizes = [
            (max(self._vocabs[c].values()) + 1 if self._vocabs[c] else 2)
            for c in self.categorical
        ]
        return TokenizedBatch(
            X_num=X_num, M_num=M_num, X_cat=X_cat, M_cat=M_cat,
            numeric_columns=list(self.numeric),
            categorical_columns=list(self.categorical),
            vocab_sizes=vocab_sizes,
        )

    def fit_transform(self, df: pd.DataFrame) -> TokenizedBatch:
        """Fit on ``df`` then transform it."""
        return self.fit(df).transform(df)

    @property
    def vocabularies(self) -> dict[str, dict[str, int]]:
        """Read-only view of learned vocabularies (for model cards)."""
        return {c: dict(v) for c, v in self._vocabs.items()}
