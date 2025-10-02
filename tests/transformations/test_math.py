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
    Fixture for crypto OHLCV prices.
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
    Fixture for BTC OHLCV prices.
    """
    # read csv from datasets/data
    return binance_spot.loc[:, 'BTC', :]


def check_assertions(actual: pd.DataFrame, expected_data: pd.DataFrame, expected_cols: List[str]):
    """Helper to check common assertions."""
    # The actual shape should match the expected number of rows and columns
    assert actual.shape == (expected_data.shape[0], len(expected_cols))

    # The actual values must be close (due to floating point math)
    assert np.allclose(actual.values, expected_data.values, equal_nan=True)
    assert np.isinf(actual.values).sum() == 0

    # dtypes
    assert isinstance(actual, pd.DataFrame)
    assert all(actual.dtypes == np.float64)

    # index
    assert actual.index.equals(expected_data.index)

    # cols (must match the new, prefixed names)
    assert actual.columns.tolist() == expected_cols


def test_log(binance_spot, btc_spot_prices) -> None:
    """
    Test Log transformation across multiple columns.
    """
    INPUT_COLS = ['open', 'close']
    EXPECTED_COLS = [f'log_{col}' for col in INPUT_COLS]

    # --- Multi-ticker (binance_spot) ---
    actual = Log(input_cols=INPUT_COLS).compute(binance_spot)
    expected_data = np.log(binance_spot[INPUT_COLS]).replace([np.inf, -np.inf], np.nan)
    expected_data.columns = EXPECTED_COLS  # Match the output column names
    check_assertions(actual, expected_data, EXPECTED_COLS)

    # --- Single-ticker (btc_spot_prices) ---
    actual_btc = Log(input_cols=INPUT_COLS).compute(btc_spot_prices)
    expected_data_btc = np.log(btc_spot_prices[INPUT_COLS]).replace([np.inf, -np.inf], np.nan)
    expected_data_btc.columns = EXPECTED_COLS  # Match the output column names
    check_assertions(actual_btc, expected_data_btc, EXPECTED_COLS)


def test_square_root(binance_spot, btc_spot_prices) -> None:
    """
    Test square root transformation on a single column.
    """
    INPUT_COL = ['close']
    EXPECTED_COL = [f'sqrt_{INPUT_COL[0]}']

    # --- Multi-ticker (binance_spot) ---
    actual = SquareRoot(input_cols=INPUT_COL).compute(binance_spot)
    # Expected: clip to handle non-negative values, then sqrt
    expected_data = np.sqrt(binance_spot[INPUT_COL].clip(lower=0)).replace([np.inf, -np.inf], np.nan)
    expected_data.columns = EXPECTED_COL
    check_assertions(actual, expected_data, EXPECTED_COL)

    # --- Single-ticker (btc_spot_prices) ---
    actual_btc = SquareRoot(input_cols=INPUT_COL).compute(btc_spot_prices)
    expected_data_btc = np.sqrt(btc_spot_prices[INPUT_COL].clip(lower=0)).replace([np.inf, -np.inf], np.nan)
    expected_data_btc.columns = EXPECTED_COL
    check_assertions(actual_btc, expected_data_btc, EXPECTED_COL)


def test_square(binance_spot, btc_spot_prices) -> None:
    """
    Test square transformation across multiple columns.
    """
    INPUT_COLS = ['high', 'low']
    EXPECTED_COLS = [f'sq_{col}' for col in INPUT_COLS]

    # --- Multi-ticker (binance_spot) ---
    actual = Square(input_cols=INPUT_COLS).compute(binance_spot)
    expected_data = binance_spot[INPUT_COLS] ** 2
    expected_data.columns = EXPECTED_COLS
    check_assertions(actual, expected_data, EXPECTED_COLS)

    # --- Single-ticker (btc_spot_prices) ---
    actual_btc = Square(input_cols=INPUT_COLS).compute(btc_spot_prices)
    expected_data_btc = btc_spot_prices[INPUT_COLS] ** 2
    expected_data_btc.columns = EXPECTED_COLS
    check_assertions(actual_btc, expected_data_btc, EXPECTED_COLS)


def test_power(binance_spot, btc_spot_prices) -> None:
    """
    Test power transformation.
    """
    EXPONENT = 3
    INPUT_COL = ['close']
    # The refactored Power class creates a name like 'power3_close'
    EXPECTED_COL = [f'power{EXPONENT}_{INPUT_COL[0]}']

    # --- Multi-ticker (binance_spot) ---
    actual = Power(exponent=EXPONENT, input_cols=INPUT_COL).compute(binance_spot)
    expected_data = binance_spot[INPUT_COL] ** EXPONENT
    expected_data.columns = EXPECTED_COL
    check_assertions(actual, expected_data, EXPECTED_COL)

    # --- Single-ticker (btc_spot_prices) ---
    actual_btc = Power(exponent=EXPONENT, input_cols=INPUT_COL).compute(btc_spot_prices)
    expected_data_btc = btc_spot_prices[INPUT_COL] ** EXPONENT
    expected_data_btc.columns = EXPECTED_COL
    check_assertions(actual_btc, expected_data_btc, EXPECTED_COL)


def test_power_with_float_exponent(binance_spot) -> None:
    """
    Test Power transformation with a float exponent, ensuring proper column naming.
    """
    EXPONENT = 1.5
    INPUT_COL = ['open']
    # Exponent '1.5' is sanitized to '1_5' in the column name
    EXPECTED_COL = [f'power1_5_{INPUT_COL[0]}']

    actual = Power(exponent=EXPONENT, input_cols=INPUT_COL).compute(binance_spot)
    expected_data = binance_spot[INPUT_COL] ** EXPONENT
    expected_data.columns = EXPECTED_COL
    check_assertions(actual, expected_data, EXPECTED_COL)


def test_power_invalid_exponent_type() -> None:
    """
    Test Power transformation with a non-numeric exponent.
    (Testing the validation in the refactored class)
    """
    # This now checks for non-numeric type, not positive integer, per the refactored logic
    with pytest.raises(ValueError, match="Exponent must be an integer or float."):
        Power(exponent="two").compute(pd.DataFrame([1, 2, 3]))
