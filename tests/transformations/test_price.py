import pytest
import pandas as pd
import numpy as np

from factorlab.transformations.price import (
    VWAP,
    NotionalValue,
)


@pytest.fixture
def binance_spot():
    """
    Fixture for crypto OHLCV prices.
    """
    # read csv from datasets/data
    df = pd.read_csv("../datasets/data/binance_spot_prices.csv",
                     index_col=['date', 'ticker'],
                     parse_dates=['date'])

    # drop tickers with nobs < ts_obs
    obs = df.groupby(level=1).count().min(axis=1)
    drop_tickers_list = obs[obs < 365].index.to_list()
    df = df.drop(drop_tickers_list, level=1, axis=0)

    # drop tickers with nobs < cs_obs
    obs = df.groupby(level=0).count().min(axis=1)
    idx_start = obs[obs > 3].index[0]
    df = df.unstack()[df.unstack().index > idx_start].stack()

    return df


@pytest.fixture
def btc_spot_prices(binance_spot):
    """
    Fixture for BTC OHLCV prices.
    """
    # read csv from datasets/data
    return binance_spot.loc[:, 'BTC', :]


def test_vwap(binance_spot, btc_spot_prices) -> None:
    """
    Test VWAP transformation.
    """
    # get actual and expected
    actual = VWAP().compute(binance_spot)
    actual_btc = VWAP().compute(btc_spot_prices)

    # Calculate expected VWAP
    typical_price = (binance_spot['open'] + binance_spot['high'] + binance_spot['low']) / 3
    expected = (binance_spot['close'] + typical_price) / 2

    typical_price_btc = (btc_spot_prices['open'] + btc_spot_prices['high'] + btc_spot_prices['low']) / 3
    expected_btc = (btc_spot_prices['close'] + typical_price_btc) / 2

    # shape
    assert actual.shape[0] == binance_spot.shape[0]
    assert actual_btc.shape[0] == btc_spot_prices.shape[0]
    assert actual.shape[1] == 1
    assert actual_btc.shape[1] == 1

    # values
    assert np.allclose(actual.vwap, expected, equal_nan=True)
    assert np.allclose(actual_btc.vwap, expected_btc, equal_nan=True)
    assert np.allclose(actual.loc[pd.IndexSlice[:, 'BTC'], 'vwap'].droplevel(1),
                       actual_btc.loc[:, 'vwap'], equal_nan=True)

    # dtypes
    assert isinstance(actual, pd.DataFrame)
    assert isinstance(actual_btc, pd.DataFrame)
    assert all(actual.dtypes == np.float64)
    assert all(actual_btc.dtypes == np.float64)

    # index
    assert all(actual.index == binance_spot.index)
    assert all(actual_btc.index == btc_spot_prices.index)

    # cols
    assert (actual.columns == ['vwap']).all()
    assert (actual_btc.columns == ['vwap']).all()


def test_notional_value(binance_spot, btc_spot_prices) -> None:
    """
    Test NotionalValue transformation for both 'mean' and 'sum' aggregation
    and verifies correct grouping/windowing based on the implementation logic.
    """
    # Use the window size (30) from the failing log
    window = 30
    col_name = 'notional_value'

    # --- STEP 1 (Shared): Calculate Daily Notional Value (P * V) ---
    daily_nv = binance_spot['close'] * binance_spot['volume']
    daily_nv_btc = btc_spot_prices['close'] * btc_spot_prices['volume']

    # Group by ticker for multi-index rolling
    g = daily_nv.groupby(level=1)

    # --- Setup for Mean Aggregation ---
    nv_mean = NotionalValue(window_size=window, agg_method='mean')
    actual_mean = nv_mean.compute(binance_spot)
    actual_mean_btc = nv_mean.compute(btc_spot_prices)

    # Calculate Expected Mean Notional Value: Rolling MEAN of Daily NV
    expected_mean = g.rolling(window=window, min_periods=1).mean().to_frame(col_name).droplevel(0).sort_index()
    expected_mean_btc = daily_nv_btc.rolling(window=window, min_periods=1).mean().to_frame(col_name).sort_index()

    # --- Setup for Sum Aggregation ---
    nv_sum = NotionalValue(window_size=window, agg_method='sum')
    actual_sum = nv_sum.compute(binance_spot)
    actual_sum_btc = nv_sum.compute(btc_spot_prices)

    # Calculate Expected Sum Notional Value: Rolling SUM of Daily NV
    expected_sum = g.rolling(window=window, min_periods=1).sum().to_frame(col_name).droplevel(0).sort_index()
    expected_sum_btc = daily_nv_btc.rolling(window=window, min_periods=1).sum().to_frame(col_name).sort_index()

    # --- Assertions for both Mean and Sum ---

    for actual, expected, actual_btc, expected_btc in [
        (actual_mean, expected_mean, actual_mean_btc, expected_mean_btc),
        (actual_sum, expected_sum, actual_sum_btc, expected_sum_btc)
    ]:
        # Shape
        assert actual.shape[0] == binance_spot.shape[0]
        assert actual.shape[1] == 1

        # Values - Compare the actual output against the correctly calculated expected output
        assert np.allclose(actual[col_name].sort_index(), expected[col_name], equal_nan=True)
        assert np.allclose(actual_btc[col_name].sort_index(), expected_btc[col_name], equal_nan=True)

        # Group Consistency
        assert np.allclose(actual.loc[pd.IndexSlice[:, 'BTC'], col_name].droplevel(1).sort_index(),
                           actual_btc.loc[:, col_name].sort_index(), equal_nan=True)

        # dtypes, index, cols checks
        assert isinstance(actual, pd.DataFrame)
        assert all(actual.dtypes == np.float64)
        assert all(actual.index == binance_spot.index)
        assert (actual.columns == [col_name]).all()
