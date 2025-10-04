import pytest
import pandas as pd
import numpy as np
from typing import List

from factorlab.transformations.price import (
    VWAP,
    NotionalValue,
)


@pytest.fixture
def binance_spot():
    """
    Fixture for crypto OHLCV prices (MultiIndex).
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
    Fixture for BTC OHLCV prices (Single-ticker DataFrame).
    """
    # extract BTC data
    return binance_spot.loc[:, 'BTC', :]


def check_accumulation_contract(
        actual: pd.DataFrame,
        original_data: pd.DataFrame,
        expected_feature_data: pd.DataFrame,
        expected_feature_cols: List[str]
) -> None:
    """
    Helper to check the Phase 1: Accumulation Contract assertions:
    1. All original columns are retained and unchanged.
    2. New feature columns are correctly calculated and named.
    """
    # 1. Check final shape (Accumulation: Original Cols + New Feature Cols)
    expected_num_cols = original_data.shape[1] + len(expected_feature_cols)
    assert actual.shape == (original_data.shape[0], expected_num_cols), "Final DataFrame shape mismatch."

    # 2. Check original columns (Context Retention)
    # Ensure all original columns are still present and equal to the input data
    original_cols = original_data.columns.tolist()
    assert original_cols == actual[original_cols].columns.tolist(), "Original columns were modified or lost."
    assert original_data.equals(actual[original_cols]), "Values in original columns were changed."

    # 3. Check new columns (Feature Correctness)
    actual_features = actual[expected_feature_cols]

    # The actual feature values must be close to the expected values
    assert np.allclose(actual_features.values, expected_feature_data.values,
                       equal_nan=True), "Calculated feature values do not match expected values."

    # No infinities allowed
    assert np.isinf(actual_features.values).sum() == 0, "Feature contains infinite values."

    # Check indices and types
    assert actual.index.equals(original_data.index), "Index was modified."
    assert isinstance(actual, pd.DataFrame), "Output is not a DataFrame."
    assert actual_features.columns.tolist() == expected_feature_cols, "Feature column names mismatch."


def test_vwap(binance_spot: pd.DataFrame, btc_spot_prices: pd.DataFrame) -> None:
    """
    Test VWAP transformation, ensuring context accumulation.
    """
    EXPECTED_COL = ['vwap']

    def calculate_expected_vwap(df: pd.DataFrame) -> pd.DataFrame:
        """Helper to compute expected VWAP feature only."""
        # Typical Price = (O + H + L) / 3
        typical_price = (df['open'] + df['high'] + df['low']) / 3
        # VWAP = (Close + Typical Price) / 2
        vwap = (df['close'] + typical_price) / 2
        return vwap.to_frame(EXPECTED_COL[0])

    # --- Multi-ticker (binance_spot) ---
    actual = VWAP().compute(binance_spot)
    expected_features = calculate_expected_vwap(binance_spot)
    check_accumulation_contract(actual, binance_spot, expected_features, EXPECTED_COL)

    # --- Single-ticker (btc_spot_prices) ---
    actual_btc = VWAP().compute(btc_spot_prices)
    expected_features_btc = calculate_expected_vwap(btc_spot_prices)
    check_accumulation_contract(actual_btc, btc_spot_prices, expected_features_btc, EXPECTED_COL)


def test_notional_value(binance_spot: pd.DataFrame, btc_spot_prices: pd.DataFrame) -> None:
    """
    Test NotionalValue transformation for both 'mean' and 'sum' aggregation,
    ensuring context accumulation.
    """
    window = 30
    COL_NAME = ['notional_value']

    # --- STEP 1 (Shared): Calculate Daily Notional Value (P * V) ---
    daily_nv = binance_spot['close'] * binance_spot['volume']
    daily_nv_btc = btc_spot_prices['close'] * btc_spot_prices['volume']

    # Group by ticker for multi-index rolling
    g = daily_nv.groupby(level=1)

    # --- Setup for Mean Aggregation ---
    nv_mean = NotionalValue(window_size=window, agg_method='mean')
    actual_mean = nv_mean.compute(binance_spot)

    # Calculate Expected Mean Notional Value (Feature only)
    expected_mean = g.rolling(window=window, min_periods=1).mean().to_frame(COL_NAME[0]).droplevel(0).sort_index()

    # --- Assertion for Multi-ticker Mean ---
    check_accumulation_contract(actual_mean, binance_spot, expected_mean, COL_NAME)

    # --- Setup for Sum Aggregation ---
    nv_sum = NotionalValue(window_size=window, agg_method='sum')
    actual_sum = nv_sum.compute(binance_spot)

    # Calculate Expected Sum Notional Value (Feature only)
    expected_sum = g.rolling(window=window, min_periods=1).sum().to_frame(COL_NAME[0]).droplevel(0).sort_index()

    # --- Assertion for Multi-ticker Sum ---
    check_accumulation_contract(actual_sum, binance_spot, expected_sum, COL_NAME)

    # --- Single-ticker (btc_spot_prices) Assertions (Mean and Sum) ---
    actual_mean_btc = nv_mean.compute(btc_spot_prices)
    actual_sum_btc = nv_sum.compute(btc_spot_prices)

    expected_mean_btc = daily_nv_btc.rolling(window=window, min_periods=1).mean().to_frame(COL_NAME[0]).sort_index()
    expected_sum_btc = daily_nv_btc.rolling(window=window, min_periods=1).sum().to_frame(COL_NAME[0]).sort_index()

    # --- Assertion for Single-ticker Mean ---
    check_accumulation_contract(actual_mean_btc, btc_spot_prices, expected_mean_btc, COL_NAME)

    # --- Assertion for Single-ticker Sum ---
    check_accumulation_contract(actual_sum_btc, btc_spot_prices, expected_sum_btc, COL_NAME)
