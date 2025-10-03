import pytest
import pandas as pd
import numpy as np

# Assuming the moments module is in the correct path
from factorlab.transformations.moments import Skewness, Kurtosis

# --- Test Parameterization Setup ---

# Define all valid parameter combinations for Skewness and Kurtosis
# We use 'close' as the target column since the fixture provides OHLCV data.
MOMENT_PARAMS = [
    # Time Series (ts) - Rolling
    ('close', 'ts', 'rolling', 30, 5),
    ('close', 'ts', 'rolling', 10, 2),
    # Time Series (ts) - Expanding
    ('close', 'ts', 'expanding', 0, 5),
    ('close', 'ts', 'expanding', 0, 2),
    # Time Series (ts) - Fixed (uses min_periods only for consistency, ignored in calculation)
    ('close', 'ts', 'fixed', 0, 2),
    # Cross Sectional (cs) - window_type/size/min_periods are ignored for 'cs'
    ('close', 'cs', 'rolling', 30, 2),
]


@pytest.fixture(params=MOMENT_PARAMS)
def moment_params(request):
    """Fixture returning (ret_col, axis, window_type, window_size, min_periods) for parametrization."""
    return request.param


# --- Data Fixtures (Using the same fixtures as test_smoothing.py) ---

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
    Fixture for BTC OHLCV prices (Single Index, subset of MultiIndex).
    """
    # Selects 'BTC' data from the MultiIndex fixture
    return binance_spot.loc[pd.IndexSlice[:, 'BTC'], :]


# --- Test Class for Skewness ---

@pytest.fixture
def skewness_instance(moment_params):
    """Creates a Skewness instance based on parameterized inputs."""
    ret_col, axis, window_type, window_size, min_periods = moment_params
    return Skewness(
        ret_col=ret_col,
        axis=axis,
        window_type=window_type,
        window_size=window_size,
        min_periods=min_periods
    )


class TestSkewness:
    """Grouped tests for the Skewness transformation."""

    def test_multiindex_shape(self, skewness_instance, binance_spot):
        """Tests that the output MultiIndex DataFrame has the same shape as the input + 1 column."""
        actual = skewness_instance.fit(binance_spot).transform(binance_spot)
        # Should have the original 4 columns + 1 new skewness column
        assert actual.shape[0] == binance_spot.shape[0]
        assert actual.shape[1] == binance_spot.shape[1] + 1

    def test_singleindex_shape(self, skewness_instance, btc_spot_prices):
        """Tests that the output Single Index DataFrame has the same shape as the input + 1 column."""
        # CS axis will raise an error on single-index, so we skip it.
        if skewness_instance.axis == 'cs':
            pytest.skip("CS axis not valid for single-index DataFrame. Tested in error cases.")

        actual = skewness_instance.fit(btc_spot_prices).transform(btc_spot_prices)
        assert actual.shape[0] == btc_spot_prices.shape[0]
        assert actual.shape[1] == btc_spot_prices.shape[1] + 1

    def test_index_and_columns(self, skewness_instance, binance_spot):
        """Tests output index matches input, and the new column is present."""
        actual = skewness_instance.fit(binance_spot).transform(binance_spot)

        # Check index equality
        assert actual.index.equals(binance_spot.index)

        # Check new column is correctly named and present
        expected_col = f'{skewness_instance.ret_col}_skew'
        assert expected_col in actual.columns
        # Check original columns are preserved
        assert all(c in actual.columns for c in binance_spot.columns)

    def test_dtypes(self, skewness_instance, binance_spot):
        """Tests that the new skewness column is float64."""
        actual = skewness_instance.fit(binance_spot).transform(binance_spot)
        expected_col = f'{skewness_instance.ret_col}_skew'
        assert isinstance(actual, pd.DataFrame)
        assert actual[expected_col].dtype == np.float64

    def test_values_match_subset(self, skewness_instance, binance_spot, btc_spot_prices):
        """
        Tests that the skewness calculation for the 'BTC' subset is identical whether performed
        on the MultiIndex frame or the Single Index frame (ensures group-wise logic is correct).
        Skip cross-sectional, as it cannot be compared this way.
        """
        if skewness_instance.axis == 'cs':
            pytest.skip("Cross-sectional logic is fundamentally different and requires separate comparison.")

        # MultiIndex transformation
        actual_multi = skewness_instance.fit(binance_spot).transform(binance_spot)

        # Single Index transformation
        actual_single = skewness_instance.fit(btc_spot_prices).transform(btc_spot_prices)

        # Extract the 'BTC' subset from the MultiIndex result
        multi_subset = actual_multi.loc[pd.IndexSlice[:, 'BTC'], :]

        expected_col = f'{skewness_instance.ret_col}_skew'

        # Check that the two derived series are almost identical (handling NaNs)
        np.allclose(multi_subset[expected_col], actual_single[expected_col], equal_nan=True)

    def test_ts_fixed_value_consistency(self, binance_spot):
        """
        Specifically test that when window_type='fixed' (ts axis), all rows for a
        given ticker share the same calculated skewness value.
        """
        # Create a specific instance for fixed TS calculation
        skew_fixed = Skewness(ret_col='close', axis='ts', window_type='fixed')
        actual = skew_fixed.fit(binance_spot).transform(binance_spot)
        output_col = skew_fixed.output_col

        for ticker, group in actual.groupby(level=1):
            # Check if all values in the new column for this ticker are equal
            first_value = group[output_col].iloc[0]
            assert (group[output_col] == first_value).all(), f"Fixed TS skewness for {ticker} is not constant."
            # Also check against the raw pandas calculation
            expected_value = binance_spot.loc[pd.IndexSlice[:, ticker], 'close'].skew()
            assert np.isclose(first_value, expected_value,
                              equal_nan=True), f"Fixed TS skewness for {ticker} is incorrect."

    def test_cs_error_single_index(self):
        """Tests that cross-sectional (cs) axis raises an error on a Single Index DataFrame."""
        # Use a dummy data frame for the error test
        dummy_df = pd.DataFrame({'close': [1, 2, 3]})
        skew_cs = Skewness(ret_col='close', axis='cs', window_type='fixed')  # window type doesn't matter here

        with pytest.raises(ValueError) as excinfo:
            skew_cs.fit(dummy_df).transform(dummy_df)
        assert "Cross-sectional skewness ('cs') requires a MultiIndex DataFrame" in str(excinfo.value)

    def test_unsupported_window_type_error(self, binance_spot):
        """Tests that an unsupported window type raises an error."""
        skew_bad = Skewness(ret_col='close', axis='ts', window_type='bad_window')
        with pytest.raises(ValueError) as excinfo:
            skew_bad.fit(binance_spot).transform(binance_spot)
        assert "Unsupported window type: bad_window for axis 'ts'." in str(excinfo.value)


# --- Test Class for Kurtosis ---

@pytest.fixture
def kurtosis_instance(moment_params):
    """Creates a Kurtosis instance based on parameterized inputs."""
    ret_col, axis, window_type, window_size, min_periods = moment_params
    return Kurtosis(
        ret_col=ret_col,
        axis=axis,
        window_type=window_type,
        window_size=window_size,
        min_periods=min_periods
    )


class TestKurtosis:
    """Grouped tests for the Kurtosis transformation."""

    # Note: Shape, index, columns, and dtypes tests for Kurtosis are structurally
    # identical to Skewness, ensuring consistency in the transformation output contract.

    def test_multiindex_shape(self, kurtosis_instance, binance_spot):
        """Tests that the output MultiIndex DataFrame has the same shape as the input + 1 column."""
        actual = kurtosis_instance.fit(binance_spot).transform(binance_spot)
        assert actual.shape[0] == binance_spot.shape[0]
        assert actual.shape[1] == binance_spot.shape[1] + 1

    def test_singleindex_shape(self, kurtosis_instance, btc_spot_prices):
        """Tests that the output Single Index DataFrame has the same shape as the input + 1 column."""
        if kurtosis_instance.axis == 'cs':
            pytest.skip("CS axis not valid for single-index DataFrame. Tested in error cases.")

        actual = kurtosis_instance.fit(btc_spot_prices).transform(btc_spot_prices)
        assert actual.shape[0] == btc_spot_prices.shape[0]
        assert actual.shape[1] == btc_spot_prices.shape[1] + 1

    def test_index_and_columns(self, kurtosis_instance, binance_spot):
        """Tests output index matches input, and the new column is present."""
        actual = kurtosis_instance.fit(binance_spot).transform(binance_spot)

        assert actual.index.equals(binance_spot.index)

        expected_col = f'{kurtosis_instance.ret_col}_kurt'
        assert expected_col in actual.columns
        assert all(c in actual.columns for c in binance_spot.columns)

    def test_dtypes(self, kurtosis_instance, binance_spot):
        """Tests that the new kurtosis column is float64."""
        actual = kurtosis_instance.fit(binance_spot).transform(binance_spot)
        expected_col = f'{kurtosis_instance.ret_col}_kurt'
        assert isinstance(actual, pd.DataFrame)
        assert actual[expected_col].dtype == np.float64

    def test_values_match_subset(self, kurtosis_instance, binance_spot, btc_spot_prices):
        """
        Tests that the kurtosis calculation for the 'BTC' subset is identical
        between MultiIndex and Single Index inputs.
        """
        if kurtosis_instance.axis == 'cs':
            pytest.skip("Cross-sectional logic is fundamentally different and requires separate comparison.")

        # MultiIndex transformation
        actual_multi = kurtosis_instance.fit(binance_spot).transform(binance_spot)
        # Single Index transformation
        actual_single = kurtosis_instance.fit(btc_spot_prices).transform(btc_spot_prices)

        multi_subset = actual_multi.loc[pd.IndexSlice[:, 'BTC'], :]

        expected_col = f'{kurtosis_instance.ret_col}_kurt'

        np.allclose(multi_subset[expected_col], actual_single[expected_col], equal_nan=True)

    def test_ts_fixed_value_consistency(self, binance_spot):
        """
        Specifically test that when window_type='fixed' (ts axis), all rows for a
        given ticker share the same calculated kurtosis value.
        """
        # Create a specific instance for fixed TS calculation
        kurt_fixed = Kurtosis(ret_col='close', axis='ts', window_type='fixed')
        actual = kurt_fixed.fit(binance_spot).transform(binance_spot)
        output_col = kurt_fixed.output_col

        for ticker, group in actual.groupby(level=1):
            # Check if all values in the new column for this ticker are equal
            first_value = group[output_col].iloc[0]
            assert (group[output_col] == first_value).all(), f"Fixed TS kurtosis for {ticker} is not constant."
            # Also check against the raw pandas calculation
            expected_value = binance_spot.loc[pd.IndexSlice[:, ticker], 'close'].kurt()
            assert np.isclose(first_value, expected_value,
                              equal_nan=True), f"Fixed TS kurtosis for {ticker} is incorrect."

    def test_cs_error_single_index(self):
        """Tests that cross-sectional (cs) axis raises an error on a Single Index DataFrame."""
        # Use a dummy data frame for the error test
        dummy_df = pd.DataFrame({'close': [1, 2, 3]})
        kurt_cs = Kurtosis(ret_col='close', axis='cs', window_type='fixed')

        with pytest.raises(ValueError) as excinfo:
            kurt_cs.fit(dummy_df).transform(dummy_df)
        assert "Cross-sectional kurtosis ('cs') requires a MultiIndex DataFrame" in str(excinfo.value)

    def test_unsupported_window_type_error(self, binance_spot):
        """Tests that an unsupported window type raises an error."""
        kurt_bad = Kurtosis(ret_col='close', axis='ts', window_type='bad_window')
        with pytest.raises(ValueError) as excinfo:
            kurt_bad.fit(binance_spot).transform(binance_spot)
        assert "Unsupported window type: bad_window for axis 'ts'." in str(excinfo.value)
