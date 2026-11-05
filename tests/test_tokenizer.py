"""Tests for dqfs.ml.tokenizer - encoding, vocabularies, robust scaling."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dqfs.ml.tokenizer import FeatureTokenizer


@pytest.fixture()
def df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "amount": [10.0, 20.0, np.nan, 40.0, 30.0],
            "country": ["US", "GB", "US", None, "DE"],
            "rare": ["a", "b", "c", "d", "e"],
        }
    )


def test_shapes_and_observed_masks(df) -> None:
    tok = FeatureTokenizer(numeric=["amount"], categorical=["country"])
    batch = tok.fit_transform(df)
    assert batch.X_num.shape == (5, 1)
    assert batch.X_cat.shape == (5, 1)
    assert batch.M_num[:, 0].tolist() == [True, True, False, True, True]
    assert batch.M_cat[:, 0].tolist() == [True, True, True, False, True]


def test_missing_numeric_maps_to_zero(df) -> None:
    tok = FeatureTokenizer(numeric=["amount"], categorical=[])
    batch = tok.fit_transform(df)
    assert batch.X_num[2, 0] == 0.0


def test_robust_scaling_median_centered(df) -> None:
    tok = FeatureTokenizer(numeric=["amount"], categorical=[])
    batch = tok.fit_transform(df)
    observed = batch.X_num[batch.M_num[:, 0], 0]
    # median (25) maps to ~0
    assert abs(np.median(observed)) < 0.5
    assert np.all(np.abs(observed) <= 8.0)  # scale clip


def test_constant_numeric_column_is_safe() -> None:
    tok = FeatureTokenizer(numeric=["c"], categorical=[])
    batch = tok.fit_transform(pd.DataFrame({"c": [7.0, 7.0, 7.0]}))
    assert np.allclose(batch.X_num, 0.0)


def test_vocab_codes_and_reserved_zero(df) -> None:
    tok = FeatureTokenizer(numeric=[], categorical=["country"])
    batch = tok.fit_transform(df)
    # US is most frequent -> code 1; missing -> 0
    assert batch.X_cat[0, 0] == 1
    assert batch.X_cat[2, 0] == 1
    assert batch.X_cat[3, 0] == 0
    assert set(np.unique(batch.X_cat)) <= set(range(0, 4))


def test_vocab_size_cap() -> None:
    big = pd.DataFrame({"c": [f"cat{i}" for i in range(500)]})
    tok = FeatureTokenizer(numeric=[], categorical=["c"], vocab_size=32)
    batch = tok.fit_transform(big)
    assert batch.vocab_sizes == [32]
    assert batch.X_cat.max() <= 31


def test_unseen_category_maps_to_reserved_zero() -> None:
    train = pd.DataFrame({"c": ["x", "y", "x", "y"]})
    test = pd.DataFrame({"c": ["x", "zzz"]})
    tok = FeatureTokenizer(numeric=[], categorical=["c"])
    tok.fit(train)
    batch = tok.transform(test)
    assert batch.X_cat[0, 0] >= 1   # seen
    assert batch.X_cat[1, 0] == 0   # unseen -> reserved code


def test_transform_is_deterministic(df) -> None:
    tok = FeatureTokenizer(numeric=["amount"], categorical=["country"]).fit(df)
    b1, b2 = tok.transform(df), tok.transform(df)
    assert np.array_equal(b1.X_num, b2.X_num)
    assert np.array_equal(b1.X_cat, b2.X_cat)
