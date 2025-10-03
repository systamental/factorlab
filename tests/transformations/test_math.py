import pytest
import pandas as pd
import numpy as np
from typing import List

from factorlab.transformations.math import (
    Log,
    SquareRoot,
    Square,
    Power
)


@pytest.fixture
def binance_spot():
    """
    Fixture for crypto OHLCV prices (MultiIndex).
    """
    # read csv from datasets/data
    df = pd.read_csv("../datasets/data/binance_spot_prices.csv",
                     index_col=['date', 'ticker'],
                     parse_dates=['date']).loc[:, : 'close']

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


def test_log(binance_spot: pd.DataFrame, btc_spot_prices: pd.DataFrame) -> None:
    """
    Test Log transformation across multiple columns, ensuring accumulation.
    """
    INPUT_COLS = ['open', 'close']
    EXPECTED_COLS = [f'log_{col}' for col in INPUT_COLS]

    # Define a helper to calculate expected features only
    def calculate_expected(df: pd.DataFrame) -> pd.DataFrame:
        expected_data = np.log(df[INPUT_COLS].mask(df[INPUT_COLS] <= 0)).replace([np.inf, -np.inf], np.nan)
        expected_data.columns = EXPECTED_COLS
        return expected_data

    # --- Multi-ticker (binance_spot) ---
    actual = Log(input_cols=INPUT_COLS).compute(binance_spot)
    expected_features = calculate_expected(binance_spot)
    check_accumulation_contract(actual, binance_spot, expected_features, EXPECTED_COLS)

    # --- Single-ticker (btc_spot_prices) ---
    actual_btc = Log(input_cols=INPUT_COLS).compute(btc_spot_prices)
    expected_features_btc = calculate_expected(btc_spot_prices)
    check_accumulation_contract(actual_btc, btc_spot_prices, expected_features_btc, EXPECTED_COLS)


def test_square_root(binance_spot: pd.DataFrame, btc_spot_prices: pd.DataFrame) -> None:
    """
    Test square root transformation on a single column, ensuring accumulation.
    """
    INPUT_COL = ['close']
    EXPECTED_COL = [f'sqrt_{INPUT_COL[0]}']

    # Define a helper to calculate expected features only
    def calculate_expected(df: pd.DataFrame) -> pd.DataFrame:
        # Expected: clip to handle non-negative values, then sqrt
        expected_data = np.sqrt(df[INPUT_COL].mask(df[INPUT_COL] < 0)).replace([np.inf, -np.inf], np.nan)
        expected_data.columns = EXPECTED_COL
        return expected_data

    # --- Multi-ticker (binance_spot) ---
    actual = SquareRoot(input_cols=INPUT_COL).compute(binance_spot)
    expected_features = calculate_expected(binance_spot)
    check_accumulation_contract(actual, binance_spot, expected_features, EXPECTED_COL)

    # --- Single-ticker (btc_spot_prices) ---
    actual_btc = SquareRoot(input_cols=INPUT_COL).compute(btc_spot_prices)
    expected_features_btc = calculate_expected(btc_spot_prices)
    check_accumulation_contract(actual_btc, btc_spot_prices, expected_features_btc, EXPECTED_COL)


def test_square(binance_spot: pd.DataFrame, btc_spot_prices: pd.DataFrame) -> None:
    """
    Test square transformation across multiple columns, ensuring accumulation.
    """
    INPUT_COLS = ['high', 'low']
    EXPECTED_COLS = [f'sq_{col}' for col in INPUT_COLS]

    # Define a helper to calculate expected features only
    def calculate_expected(df: pd.DataFrame) -> pd.DataFrame:
        expected_data = df[INPUT_COLS] ** 2
        expected_data.columns = EXPECTED_COLS
        return expected_data

    # --- Multi-ticker (binance_spot) ---
    actual = Square(input_cols=INPUT_COLS).compute(binance_spot)
    expected_features = calculate_expected(binance_spot)
    check_accumulation_contract(actual, binance_spot, expected_features, EXPECTED_COLS)

    # --- Single-ticker (btc_spot_prices) ---
    actual_btc = Square(input_cols=INPUT_COLS).compute(btc_spot_prices)
    expected_features_btc = calculate_expected(btc_spot_prices)
    check_accumulation_contract(actual_btc, btc_spot_prices, expected_features_btc, EXPECTED_COLS)


def test_power(binance_spot: pd.DataFrame, btc_spot_prices: pd.DataFrame) -> None:
    """
    Test power transformation, ensuring accumulation.
    """
    EXPONENT = 3
    INPUT_COL = ['close']
    # The refactored Power class creates a name like 'power3_close'
    EXPECTED_COL = [f'power{EXPONENT}_{INPUT_COL[0]}']

    # Define a helper to calculate expected features only
    def calculate_expected(df: pd.DataFrame) -> pd.DataFrame:
        expected_data = df[INPUT_COL] ** EXPONENT
        expected_data.columns = EXPECTED_COL
        return expected_data

    # --- Multi-ticker (binance_spot) ---
    actual = Power(exponent=EXPONENT, input_cols=INPUT_COL).compute(binance_spot)
    expected_features = calculate_expected(binance_spot)
    check_accumulation_contract(actual, binance_spot, expected_features, EXPECTED_COL)

    # --- Single-ticker (btc_spot_prices) ---
    actual_btc = Power(exponent=EXPONENT, input_cols=INPUT_COL).compute(btc_spot_prices)
    expected_features_btc = calculate_expected(btc_spot_prices)
    check_accumulation_contract(actual_btc, btc_spot_prices, expected_features_btc, EXPECTED_COL)


def test_power_with_float_exponent(binance_spot: pd.DataFrame) -> None:
    """
    Test Power transformation with a float exponent, ensuring proper column naming and accumulation.
    """
    EXPONENT = 1.5
    INPUT_COL = ['open']
    # Exponent '1.5' is sanitized to '1_5' in the column name
    EXPECTED_COL = [f'power1_5_{INPUT_COL[0]}']

    # Define a helper to calculate expected features only
    def calculate_expected(df: pd.DataFrame) -> pd.DataFrame:
        expected_data = df[INPUT_COL] ** EXPONENT
        expected_data.columns = EXPECTED_COL
        return expected_data

    actual = Power(exponent=EXPONENT, input_cols=INPUT_COL).compute(binance_spot)
    expected_features = calculate_expected(binance_spot)
    check_accumulation_contract(actual, binance_spot, expected_features, EXPECTED_COL)


def test_power_invalid_exponent_type() -> None:
    """
    Test Power transformation with a non-numeric exponent (testing the validation).
    """
    # This checks for non-numeric type, as defined in the refactored class
    with pytest.raises(ValueError, match="Exponent must be an integer or float."):
        Power(exponent="two").compute(pd.DataFrame([1, 2, 3]))