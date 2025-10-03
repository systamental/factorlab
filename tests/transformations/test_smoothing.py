import pytest
import pandas as pd
import numpy as np

# Assuming WindowSmoother is correctly importable from this path
from factorlab.transformations.smoothing import WindowSmoother

# --- Test Parameterization ---

# Define all valid parameter combinations for the smoother
SMOOTHER_PARAMS = [
    # EWM
    (30, 'ewm', 'mean'),
    # Rolling
    (30, 'rolling', 'mean'),
    (30, 'rolling', 'median'),
    # Expanding (window_size is ignored, setting to 0 or 30 is fine)
    (0, 'expanding', 'mean'),
    (30, 'expanding', 'median')
]


@pytest.fixture(params=SMOOTHER_PARAMS)
def smoother_params(request):
    """Fixture returning (window_size, window_type, central_tendency) for parametrization."""
    return request.param


@pytest.fixture
def smoother_instance(smoother_params):
    """Creates a WindowSmoother instance based on parameterized inputs."""
    window_size, window_type, central_tendency = smoother_params
    return WindowSmoother(
        window_size=window_size,
        window_type=window_type,
        central_tendency=central_tendency
    )


# --- Data Fixtures (Copied from User Input) ---

@pytest.fixture
def binance_spot():
    """
    Fixture for crypto OHLCV prices (MultiIndex).
    Note: Requires 'datasets/data/binance_spot_prices.csv' to be available.
    """
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
    Fixture for BTC OHLCV prices (Single Index).
    """
    # Selects 'BTC' data from the MultiIndex fixture
    return binance_spot.loc[pd.IndexSlice[:, 'BTC'], :]


# --- Test Class ---

class TestWindowSmoother:
    """Grouped tests for the WindowSmoother transformation."""

    # --- MultiIndex (binance_spot) Tests ---

    def test_multiindex_shape(self, smoother_instance, binance_spot):
        """Tests that the output MultiIndex DataFrame has the same shape as the input."""
        actual = smoother_instance.fit(binance_spot).transform(binance_spot)
        assert actual.shape[0] == binance_spot.shape[0]
        assert actual.shape[1] == binance_spot.shape[1] + 1  # one extra column for smoothed output

    def test_multiindex_index(self, smoother_instance, binance_spot):
        """Tests that the output MultiIndex DataFrame has the same index as the input."""
        actual = smoother_instance.fit(binance_spot).transform(binance_spot)
        assert actual.index.equals(binance_spot.index)

    def test_multiindex_dtypes(self, smoother_instance, binance_spot):
        """Tests that all dtypes remain float64 in the MultiIndex output."""
        actual = smoother_instance.fit(binance_spot).transform(binance_spot)
        assert isinstance(actual, pd.DataFrame)
        assert all(actual.dtypes == np.float64)

    # --- Single Index (btc_spot_prices) Tests ---

    def test_singleindex_shape(self, smoother_instance, btc_spot_prices):
        """Tests that the output Single Index DataFrame has the same shape as the input."""
        actual = smoother_instance.fit(btc_spot_prices).transform(btc_spot_prices)
        assert actual.shape[0] == btc_spot_prices.shape[0]
        assert actual.shape[1] == btc_spot_prices.shape[1] + 1  # one extra column for smoothed output

    def test_singleindex_index(self, smoother_instance, btc_spot_prices):
        """Tests that the output Single Index DataFrame has the same index as the input."""
        actual = smoother_instance.fit(btc_spot_prices).transform(btc_spot_prices)
        assert actual.index.equals(btc_spot_prices.index)

    def test_singleindex_dtypes(self, smoother_instance, btc_spot_prices):
        """Tests that all dtypes remain float64 in the Single Index output."""
        actual = smoother_instance.fit(btc_spot_prices).transform(btc_spot_prices)
        assert isinstance(actual, pd.DataFrame)
        assert all(actual.dtypes == np.float64)

    # --- Values Consistency Test ---

    def test_values_match_subset(self, smoother_instance, binance_spot, btc_spot_prices):
        """
        Tests that the calculation for the 'BTC' subset is identical whether performed
        on the MultiIndex frame or the Single Index frame (ensures group-wise logic is correct).
        This is the most critical test for correctness.
        """
        # MultiIndex transformation
        actual_multi = smoother_instance.fit(binance_spot).transform(binance_spot)

        # Single Index transformation
        actual_single = smoother_instance.fit(btc_spot_prices).transform(btc_spot_prices)

        # Extract the 'BTC' subset from the MultiIndex result
        multi_subset = actual_multi.loc[pd.IndexSlice[:, 'BTC'], :]

        # Check that the two results are almost identical (handling NaNs)
        np.allclose(multi_subset, actual_single, equal_nan=True)

    # --- Error Handling Test ---

    def test_ewm_median_error(self, binance_spot):
        """Tests that EWM with median raises the expected ValueError."""
        smoother = WindowSmoother(window_type='ewm', central_tendency='median')
        with pytest.raises(ValueError) as excinfo:
            smoother.compute(binance_spot)  # Using compute() for brevity
        assert "Median is not supported for ewm smoothing" in str(excinfo.value)
