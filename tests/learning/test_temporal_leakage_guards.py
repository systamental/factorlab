import pandas as pd
import numpy as np

from factorlab.learning import ExpandingIncrementPanelSplit


def _make_xy(n_days: int = 200, n_assets: int = 3, seed: int = 42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n_days, freq="D")
    tickers = [f"A{i}" for i in range(n_assets)]
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    X = pd.DataFrame({"f1": rng.normal(size=len(idx))}, index=idx)
    y = pd.Series(rng.normal(size=len(idx)), index=idx, name="y")
    return X, y


def test_fold_boundary_respects_lookahead_and_embargo():
    X, y = _make_xy()
    lookahead = 2
    embargo = 3
    splitter = ExpandingIncrementPanelSplit(
        train_intervals=5,
        test_size=3,
        min_train_periods=60,
        lookahead=lookahead,
        embargo=embargo,
    )
    for train_idx, test_idx in splitter.split(X, y):
        train_dates = pd.DatetimeIndex(X.iloc[train_idx].index.get_level_values("date"))
        test_dates = pd.DatetimeIndex(X.iloc[test_idx].index.get_level_values("date"))
        assert train_dates.max() < test_dates.min()
        assert (test_dates.min() - train_dates.max()).days >= lookahead + embargo


def test_horizon_purge_no_train_leakage_into_test_start():
    X, y = _make_xy()
    splitter = ExpandingIncrementPanelSplit(
        train_intervals=1,
        test_size=1,
        min_train_periods=40,
        lookahead=1,
        embargo=0,
    )
    for train_idx, test_idx in splitter.split(X, y):
        train_dates = pd.DatetimeIndex(X.iloc[train_idx].index.get_level_values("date"))
        test_dates = pd.DatetimeIndex(X.iloc[test_idx].index.get_level_values("date"))
        assert train_dates.max() <= test_dates.min() - pd.Timedelta(days=1)
