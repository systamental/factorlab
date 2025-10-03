import pytest
import pandas as pd
import numpy as np
from typing import List, Union

from factorlab.transformations.returns import (
    Returns,
    Difference,
    LogReturn,
    PctChange,
    CumulativeReturn,
    TotalReturn
)


# --- FIXTURES (Reused from test_price.py for context) ---

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

    # Add dummy funding rate and dividend for TotalReturn testing
    # Funding rate is slightly dynamic
    np.random.seed(42)
    df['funding_rate'] = np.random.uniform(-0.0001, 0.0001, size=len(df))
    # Dividend is constant small value for simplicity
    df['dividend'] = 0.000001

    return df


@pytest.fixture
def btc_spot_prices(binance_spot):
    """
    Fixture for BTC OHLCV prices (Single-ticker DataFrame).
    """
    # extract BTC data
    return binance_spot.loc[pd.IndexSlice[:, 'BTC'], :].droplevel(1)


# --- HELPER FUNCTION ---

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
    assert original_data.index.equals(actual.index), "Index was modified."
    assert original_cols == actual[original_cols].columns.tolist(), "Original columns were modified or lost."
    assert original_data.equals(actual[original_cols]), "Values in original columns were changed."

    # 3. Check new columns (Feature Correctness)
    actual_features = actual[expected_feature_cols]

    # Crucial: Ensure the indexes match before comparing values
    assert expected_feature_data.index.equals(actual_features.index), "Index of calculated feature does not match index of actual result."

    # The actual feature values must be close to the expected values
    assert np.allclose(actual_features.values, expected_feature_data.values, equal_nan=True), "Calculated feature values do not match expected values."

    # No infinities allowed
    assert np.isinf(actual_features.values).sum() == 0, "Feature contains infinite values."

    # Check dtypes and names
    assert isinstance(actual, pd.DataFrame), "Output is not a DataFrame."
    assert actual_features.columns.tolist() == expected_feature_cols, "Feature column names mismatch."


# --- TESTS ---

def test_returns_factory(binance_spot: pd.DataFrame) -> None:
    """
    Test the Returns factory class delegates to PctChange and maintains the contract.
    """
    EXPECTED_COL = ['ret']

    # Factory uses PctChange by default
    actual = Returns().compute(binance_spot)

    # Expected PctChange calculation (MultiIndex should be preserved)
    expected_feature = binance_spot['close'].groupby(level=1).pct_change(periods=1).to_frame(EXPECTED_COL[0]).sort_index()
    check_accumulation_contract(actual, binance_spot, expected_feature, EXPECTED_COL)

    # Test 'log' method delegation
    actual_log = Returns(method='log').compute(binance_spot)
    expected_log_series = np.log(binance_spot['close'].where(binance_spot['close'] > 0, np.nan))
    # MultiIndex should be preserved
    expected_log_feature = expected_log_series.groupby(level=1).diff(1).to_frame(EXPECTED_COL[0]).sort_index()
    check_accumulation_contract(actual_log, binance_spot, expected_log_feature, EXPECTED_COL)

    # Test error handling
    with pytest.raises(ValueError):
        Returns(method='invalid_method').compute(binance_spot)


def test_difference(binance_spot: pd.DataFrame, btc_spot_prices: pd.DataFrame) -> None:
    """
    Test Difference transformation, ensuring context accumulation.
    """
    EXPECTED_COL = ['diff']
    LAG = 2

    # --- Multi-ticker (binance_spot) ---
    actual = Difference(lags=LAG).compute(binance_spot)

    # Expected feature calculation: p_t - p_{t-lag}. MultiIndex should be preserved.
    expected_feature = binance_spot['close'].groupby(level=1).diff(LAG).to_frame(EXPECTED_COL[0]).sort_index()
    check_accumulation_contract(actual, binance_spot, expected_feature, EXPECTED_COL)

    # --- Single-ticker (btc_spot_prices) ---
    actual_btc = Difference(lags=LAG).compute(btc_spot_prices)

    # Expected feature calculation for single-index
    expected_feature_btc = btc_spot_prices['close'].diff(LAG).to_frame(EXPECTED_COL[0]).sort_index()
    check_accumulation_contract(actual_btc, btc_spot_prices, expected_feature_btc, EXPECTED_COL)


def test_log_return(binance_spot: pd.DataFrame, btc_spot_prices: pd.DataFrame) -> None:
    """
    Test LogReturn transformation, ensuring context accumulation.
    """
    EXPECTED_COL = ['ret']
    LAG = 1

    # Helper to calculate expected log return feature
    def calculate_expected_log_ret(df: pd.DataFrame) -> pd.DataFrame:
        series = df['close']
        series = series.where(series > 0, np.nan)
        log_series = np.log(series)
        if isinstance(df.index, pd.MultiIndex):
            # FIXED: Removed .droplevel(0) to preserve the MultiIndex for comparison
            log_diff = log_series.groupby(level=1).diff(LAG).sort_index()
        else:
            log_diff = log_series.diff(LAG).sort_index()
        return log_diff.to_frame(EXPECTED_COL[0])

    # --- Multi-ticker (binance_spot) ---
    actual = LogReturn(lags=LAG).compute(binance_spot)
    expected_feature = calculate_expected_log_ret(binance_spot)
    check_accumulation_contract(actual, binance_spot, expected_feature, EXPECTED_COL)

    # --- Single-ticker (btc_spot_prices) ---
    actual_btc = LogReturn(lags=LAG).compute(btc_spot_prices)
    expected_feature_btc = calculate_expected_log_ret(btc_spot_prices)
    check_accumulation_contract(actual_btc, btc_spot_prices, expected_feature_btc, EXPECTED_COL)


def test_pct_change(binance_spot: pd.DataFrame, btc_spot_prices: pd.DataFrame) -> None:
    """
    Test PctChange transformation, ensuring context accumulation.
    """
    EXPECTED_COL = ['ret']
    LAG = 3

    # --- Multi-ticker (binance_spot) ---
    actual = PctChange(lags=LAG).compute(binance_spot)

    # Expected feature calculation: (p_t / p_{t-lag}) - 1. MultiIndex should be preserved.
    expected_feature = binance_spot['close'].groupby(level=1).pct_change(periods=LAG).to_frame(EXPECTED_COL[0]).sort_index()
    check_accumulation_contract(actual, binance_spot, expected_feature, EXPECTED_COL)

    # --- Single-ticker (btc_spot_prices) ---
    actual_btc = PctChange(lags=LAG).compute(btc_spot_prices)

    # Expected feature calculation for single-index
    expected_feature_btc = btc_spot_prices['close'].pct_change(periods=LAG).to_frame(EXPECTED_COL[0]).sort_index()
    check_accumulation_contract(actual_btc, btc_spot_prices, expected_feature_btc, EXPECTED_COL)


def test_cumulative_return(binance_spot: pd.DataFrame, btc_spot_prices: pd.DataFrame) -> None:
    """
    Test CumulativeReturn transformation, ensuring context accumulation.
    """
    EXPECTED_COL = ['cum_ret']
    BASE_INDEX = 5

    # Helper to calculate expected cumulative return feature
    def calculate_expected_cum_ret(df: pd.DataFrame) -> pd.DataFrame:
        price_series = df['close']

        if isinstance(df.index, pd.MultiIndex):
            def _get_base(g):
                return g.iloc[BASE_INDEX]
            # Calculate base price for each group and align
            # The .to_frame().groupby().transform() ensures the MultiIndex is retained.
            base = price_series.to_frame().groupby(level=1).transform(_get_base).squeeze()
        else:
            # Single index: simple base price
            base = price_series.iloc[BASE_INDEX]

        cum_ret = (price_series / base) - 1
        return cum_ret.to_frame(EXPECTED_COL[0]).sort_index()

    # --- Multi-ticker (binance_spot) ---
    actual = CumulativeReturn(base_index=BASE_INDEX).compute(binance_spot)
    expected_feature = calculate_expected_cum_ret(binance_spot)
    check_accumulation_contract(actual, binance_spot, expected_feature, EXPECTED_COL)

    # --- Single-ticker (btc_spot_prices) ---
    actual_btc = CumulativeReturn(base_index=BASE_INDEX).compute(btc_spot_prices)
    expected_feature_btc = calculate_expected_cum_ret(btc_spot_prices)
    check_accumulation_contract(actual_btc, btc_spot_prices, expected_feature_btc, EXPECTED_COL)


def test_total_return(binance_spot: pd.DataFrame) -> None:
    """
    Test TotalReturn transformation, ensuring context accumulation.
    TotalReturn = PriceReturn - FinancingCost + DividendYield
    """
    EXPECTED_COL = ['total_ret']

    # 1. Pre-calculate the required Price Return column ('ret') using PctChange
    pct_change = PctChange(output_col='ret')
    df_with_ret = pct_change.compute(binance_spot)

    # 2. Test TotalReturn with all components
    tr_all = TotalReturn(dividend_col='dividend')
    actual_all = tr_all.compute(df_with_ret)

    # Expected Feature calculation for all components: ret - funding_rate + dividend
    expected_feature_all = (
        df_with_ret['ret'] - df_with_ret['funding_rate'] + df_with_ret['dividend']
    ).to_frame(EXPECTED_COL[0]).sort_index() # Ensure index is sorted
    check_accumulation_contract(actual_all, df_with_ret, expected_feature_all, EXPECTED_COL)

    # 3. Test TotalReturn without dividend component
    tr_no_div = TotalReturn(dividend_col=None)
    actual_no_div = tr_no_div.compute(df_with_ret)

    # Expected Feature calculation without dividend: ret - funding_rate
    expected_feature_no_div = (
        df_with_ret['ret'] - df_with_ret['funding_rate']
    ).to_frame(EXPECTED_COL[0]).sort_index() # Ensure index is sorted
    check_accumulation_contract(actual_no_div, df_with_ret, expected_feature_no_div, EXPECTED_COL)
