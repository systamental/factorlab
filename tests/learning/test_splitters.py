import numpy as np
import pandas as pd
import pytest

from factorlab.learning import (
    ExpandingFrequencyPanelSplit,
    ExpandingIncrementPanelSplit,
    ExpandingKFoldPanelSplit,
)


def _make_xy(n_days: int = 180, n_assets: int = 4, seed: int = 123):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n_days, freq="D")
    tickers = [f"A{i}" for i in range(n_assets)]
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])

    X = pd.DataFrame(
        {
            "f1": rng.normal(size=len(idx)),
            "f2": rng.normal(size=len(idx)),
        },
        index=idx,
    )
    y = pd.Series(rng.normal(size=len(idx)), index=idx, name="y")
    return X, y


def test_expanding_kfold_time_order_and_purge():
    X, y = _make_xy()
    splitter = ExpandingKFoldPanelSplit(n_splits=4, lookahead=1, embargo=1)
    splits = list(splitter.split(X, y))

    assert len(splits) > 0
    for train_idx, test_idx in splits:
        train_dates = X.iloc[train_idx].index.get_level_values("date")
        test_dates = X.iloc[test_idx].index.get_level_values("date")
        assert train_dates.max() < test_dates.min()


def test_expanding_increment_respects_lookahead_embargo_boundary():
    X, y = _make_xy()
    splitter = ExpandingIncrementPanelSplit(
        train_intervals=10,
        test_size=5,
        min_train_periods=50,
        lookahead=2,
        embargo=3,
    )
    splits = list(splitter.split(X, y))

    assert len(splits) > 0
    for train_idx, test_idx in splits:
        train_dates = pd.DatetimeIndex(X.iloc[train_idx].index.get_level_values("date"))
        test_dates = pd.DatetimeIndex(X.iloc[test_idx].index.get_level_values("date"))
        gap_days = (test_dates.min() - train_dates.max()).days
        assert gap_days >= 5


def test_expanding_frequency_produces_non_empty_test_folds():
    X, y = _make_xy(n_days=365)
    splitter = ExpandingFrequencyPanelSplit(
        expansion_freq="M",
        test_freq="M",
        min_train_periods=60,
        lookahead=1,
        embargo=1,
    )
    splits = list(splitter.split(X, y))
    assert len(splits) > 0
    assert all(len(test_idx) > 0 for _, test_idx in splits)


def test_splitter_rejects_unsorted_index():
    X, y = _make_xy()
    shuffled = np.random.default_rng(1).permutation(len(X))
    X_unsorted = X.iloc[shuffled]
    y_unsorted = y.iloc[shuffled]

    splitter = ExpandingIncrementPanelSplit(
        train_intervals=5,
        test_size=3,
        min_train_periods=30,
        lookahead=1,
        embargo=0,
    )
    with pytest.raises(ValueError, match="index must be sorted"):
        _ = list(splitter.split(X_unsorted, y_unsorted))


def test_splitter_indices_align_to_original_rows_after_nan_drop():
    X, y = _make_xy(n_days=140, n_assets=3)
    y = y.copy()
    drop_dates = X.index.get_level_values("date").unique()[:8]
    drop_mask = X.index.get_level_values("date").isin(drop_dates)
    y.loc[drop_mask] = np.nan

    splitter = ExpandingIncrementPanelSplit(
        train_intervals=10,
        test_size=5,
        min_train_periods=40,
        lookahead=1,
        embargo=0,
        drop_nas=True,
    )

    valid_rows = y.notna().to_numpy()
    splits = list(splitter.split(X, y))
    assert len(splits) > 0
    for train_idx, test_idx in splits:
        selected = np.concatenate([train_idx, test_idx])
        assert valid_rows[selected].all()
