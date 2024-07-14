import pytest
import pandas as pd
import numpy as np

from factorlab.feature_engineering.transformations import Transform


@pytest.fixture
def binance_spot():
    """
    Fixture for crypto OHLCV prices.
    """
    # read csv from datasets/data
    return pd.read_csv("../src/factorlab/datasets/data/binance_spot_prices.csv", index_col=['date', 'ticker'],
                       parse_dates=True).loc[:, : 'close']


@pytest.fixture
def btc_spot_prices(binance_spot):
    """
    Fixture for BTC OHLCV prices.
    """
    # read csv from datasets/data
    return binance_spot.loc[pd.IndexSlice[:, 'BTC'], : 'close'].droplevel(1)


@pytest.fixture
def us_real_rates():
    """
    Fixture for US real rates data.
    """
    # read csv from datasets/data
    return pd.read_csv("../src/factorlab/datasets/data/us_real_rates_10y_monthly.csv", index_col=['date'],
                       parse_dates=True)


class TestTransform:
    """
    Test class for Transform.
    """

    @pytest.fixture(autouse=True)
    def transform_setup_default(self, binance_spot):
        self.default_transform_instance = Transform(binance_spot)

    @pytest.fixture(autouse=True)
    def transform_setup_no_missing(self, binance_spot):
        df = binance_spot.loc[pd.IndexSlice[:, ['BTC', 'ETH', 'LTC', 'DOGE']], :].unstack().dropna().stack()
        self.no_missing_transform_instance = Transform(df)

    @pytest.fixture(autouse=True)
    def transform_setup_btc(self, btc_spot_prices):
        self.btc_transform_instance = Transform(btc_spot_prices)

    @pytest.fixture(autouse=True)
    def transform_setup_btc_no_missing(self, binance_spot):
        df = binance_spot.loc[pd.IndexSlice[:, ['BTC', 'ETH', 'LTC', 'DOGE']], :].unstack().dropna().stack()
        df = df.loc[pd.IndexSlice[:, 'BTC'], :].droplevel(1)
        self.btc_no_missing_transform_instance = Transform(df)

    def test_initialization(self) -> None:
        """
        Test initialization.
        """
        # data type
        assert isinstance(self.default_transform_instance, Transform)
        assert isinstance(self.btc_transform_instance, Transform)
        assert isinstance(self.default_transform_instance.df, pd.DataFrame)
        assert isinstance(self.btc_transform_instance.df, pd.DataFrame)
        assert isinstance(self.default_transform_instance.arr, np.ndarray)
        assert isinstance(self.btc_transform_instance.arr, np.ndarray)
        assert isinstance(self.default_transform_instance.trans_df, pd.DataFrame)
        assert isinstance(self.btc_transform_instance.trans_df, pd.DataFrame)
        if isinstance(self.default_transform_instance.index, pd.MultiIndex):
            assert isinstance(self.default_transform_instance.index.droplevel(1), pd.DatetimeIndex)
        assert isinstance(self.btc_transform_instance.index, pd.DatetimeIndex)
        # dtypes
        assert (self.default_transform_instance.df.dtypes == np.float64).all()
        assert (self.btc_transform_instance.df.dtypes == np.float64).all()
        # shape
        assert self.default_transform_instance.raw_data.shape[0] == self.default_transform_instance.df.shape[0]
        assert self.btc_transform_instance.raw_data.shape[0] == self.btc_transform_instance.df.shape[0]
        assert self.default_transform_instance.raw_data.shape[0] == self.default_transform_instance.arr.shape[0]
        assert self.btc_transform_instance.raw_data.shape[0] == self.btc_transform_instance.arr.shape[0]
        assert self.default_transform_instance.raw_data.shape[1] == self.default_transform_instance.df.shape[1]
        assert self.btc_transform_instance.raw_data.shape[1] == self.btc_transform_instance.df.shape[1]
        assert self.default_transform_instance.raw_data.shape[1] == self.default_transform_instance.arr.shape[1]
        assert self.btc_transform_instance.raw_data.shape[1] == self.btc_transform_instance.arr.shape[1]
        # values
        assert np.array_equal(self.default_transform_instance.raw_data.values,
                              self.default_transform_instance.arr, equal_nan=True)
        assert np.array_equal(self.btc_transform_instance.raw_data.values,
                              self.btc_transform_instance.arr, equal_nan=True)
        # index
        assert (self.default_transform_instance.index == self.default_transform_instance.df.index).all()
        assert (self.btc_transform_instance.index == self.btc_transform_instance.df.index).all()
        # cols
        assert (self.default_transform_instance.raw_data.columns == self.default_transform_instance.df.columns).all()
        assert (self.btc_transform_instance.raw_data.columns == self.btc_transform_instance.df.columns).all()

    def test_vwap(self) -> None:
        """
        Test VWAP method.
        """
        # get actual and expected
        actual = self.default_transform_instance.vwap()
        actual_btc = self.btc_transform_instance.vwap()
        expected = (self.default_transform_instance.df.close +
                    (self.default_transform_instance.df.open + self.default_transform_instance.df.high +
                     self.default_transform_instance.df.low) / 3) / 2
        expected_btc = (self.btc_transform_instance.df.close +
                        (self.btc_transform_instance.df.open + self.btc_transform_instance.df.high +
                         self.btc_transform_instance.df.low) / 3) / 2

        # shape
        assert actual.shape[0] == self.default_transform_instance.df.shape[0]
        assert actual_btc.shape[0] == self.btc_transform_instance.df.shape[0]
        assert actual.shape[1] == self.default_transform_instance.df.shape[1] + 1
        assert actual_btc.shape[1] == self.btc_transform_instance.df.shape[1] + 1
        # values
        assert np.allclose(actual.vwap, expected, equal_nan=True)
        assert np.allclose(actual_btc.vwap, expected_btc, equal_nan=True)
        assert np.allclose(actual.loc[pd.IndexSlice[:, 'BTC'], 'vwap'].droplevel(1),
                           actual_btc.loc[:, 'vwap'], equal_nan=True)
        assert (actual.high >= actual.vwap).sum() / actual.shape[0] > 0.99
        assert (actual_btc.high >= actual_btc.vwap).sum() / actual_btc.shape[0] > 0.99
        assert (actual.low <= actual.vwap).sum() / actual.shape[0] > 0.99
        assert (actual_btc.low <= actual_btc.vwap).sum() / actual_btc.shape[0] > 0.99
        # dtypes
        assert isinstance(actual, pd.DataFrame)
        assert isinstance(actual_btc, pd.DataFrame)
        assert (actual.dtypes == np.float64).all()
        assert (actual_btc.dtypes == np.float64).all()
        # index
        assert (actual.index == self.default_transform_instance.index).all()
        assert (actual_btc.index == self.btc_transform_instance.index).all()
        # cols
        assert (actual.columns == self.default_transform_instance.df.columns.tolist() + ['vwap']).all()
        assert (actual_btc.columns == self.btc_transform_instance.df.columns.tolist() + ['vwap']).all()

    def test_vwap_param_errors(self) -> None:
        """
        Test VWAP method parameter errors.
        """
        # test if data is does not have OHLCV columns
        with pytest.raises(ValueError):
            Transform(self.btc_transform_instance.df.drop(columns=['close'])).vwap()

    def test_log(self) -> None:
        """
        Test log method.
        """
        # get actual and expected
        actual = self.default_transform_instance.log()
        actual_btc = self.btc_transform_instance.log()
        expected = np.log(self.default_transform_instance.df)
        expected_btc = np.log(self.btc_transform_instance.df)

        # shape
        assert actual.shape[0] == expected.shape[0]
        assert actual_btc.shape[0] == expected_btc.shape[0]
        assert actual.shape[1] == expected.shape[1]
        assert actual_btc.shape[1] == expected_btc.shape[1]
        # values
        assert np.array_equal(actual, expected, equal_nan=True)
        assert np.array_equal(actual_btc, expected_btc, equal_nan=True)
        assert np.isinf(actual).sum().sum() == 0
        assert np.isinf(actual_btc).sum().sum() == 0
        # dtypes
        assert isinstance(actual, pd.DataFrame)
        assert isinstance(actual_btc, pd.DataFrame)
        assert (actual.dtypes == np.float64).all()
        assert (actual_btc.dtypes == np.float64).all()
        # index
        assert (actual.index == self.default_transform_instance.index).all()
        assert (actual_btc.index == self.btc_transform_instance.index).all()
        # cols
        assert (actual.columns == self.default_transform_instance.df.columns).all()
        assert (actual_btc.columns == self.btc_transform_instance.df.columns).all()

    def test_diff(self) -> None:
        """
        Test diff method.
        """
        # get actual and expected
        actual = self.default_transform_instance.diff()
        actual_btc = self.btc_transform_instance.diff()

        # shape
        assert actual.shape == self.default_transform_instance.df.shape
        assert actual_btc.shape == self.btc_transform_instance.df.shape
        # values
        assert np.allclose(self.btc_transform_instance.df.iloc[1:], (actual_btc.cumsum() +
                           self.btc_transform_instance.df.iloc[0]).dropna())
        assert np.allclose(actual.loc[pd.IndexSlice[:, 'BTC'], :], actual_btc, equal_nan=True)
        # dtypes
        assert isinstance(actual, pd.DataFrame)
        assert isinstance(actual_btc, pd.DataFrame)
        assert (actual.dtypes == np.float64).all()
        assert (actual_btc.dtypes == np.float64).all()
        # index
        assert (actual.index == self.default_transform_instance.index).all()
        assert (actual_btc.index == self.btc_transform_instance.index).all()
        # cols
        assert (actual.columns == self.default_transform_instance.df.columns).all()
        assert (actual_btc.columns == self.btc_transform_instance.df.columns).all()

    @pytest.mark.parametrize("method", ['simple', 'log'])
    def test_returns(self, method) -> None:
        """
        Test returns method.
        """
        # get actual and expected
        actual = self.default_transform_instance.returns(method=method)
        actual_btc = self.btc_transform_instance.returns(method=method)
        mkt = self.default_transform_instance.returns(market=True)

        if method == 'simple':
            expected = self.default_transform_instance.df.groupby(level=1).pct_change(fill_method=None)
            expected_btc = self.btc_transform_instance.df.pct_change(fill_method=None)
        else:
            expected = np.log(self.default_transform_instance.df).groupby(level=1).diff()
            expected_btc = np.log(self.btc_transform_instance.df).diff()

        # shape
        assert actual.shape[0] == self.default_transform_instance.df.shape[0]
        assert actual_btc.shape[0] == self.btc_transform_instance.df.shape[0]
        assert actual.shape[1] == self.default_transform_instance.df.shape[1]
        assert actual_btc.shape[1] == self.btc_transform_instance.df.shape[1]
        assert mkt.shape[0] == self.btc_transform_instance.df.shape[0]
        # values
        assert np.allclose(actual, expected, equal_nan=True)
        assert np.allclose(actual_btc, expected_btc, equal_nan=True)
        assert np.allclose(actual.loc[pd.IndexSlice[:, 'BTC'], :].droplevel(1), actual_btc, equal_nan=True)
        # dtypes
        assert isinstance(actual, pd.DataFrame)
        assert isinstance(actual_btc, pd.DataFrame)
        assert isinstance(mkt, pd.DataFrame)
        assert (actual.dtypes == np.float64).all()
        assert (actual_btc.dtypes == np.float64).all()
        assert (mkt.dtypes == np.float64).all()
        # index
        assert (actual.index == self.default_transform_instance.index).all()
        assert (actual_btc.index == self.btc_transform_instance.index).all()
        assert (mkt.index == self.btc_transform_instance.index).all()
        # cols
        assert all(actual.columns == self.default_transform_instance.df.columns)
        assert all(actual_btc.columns == self.btc_transform_instance.df.columns)
        assert mkt.columns == ['mkt_ret']

    @pytest.mark.parametrize("ret_type, start_val",
                             [('simple', 1), ('log', 1), ('simple', 100), ('log', 100), ('simple', 16616.75),
                              ('log', 16616.75)]
                             )
    def test_returns_to_price(self, ret_type, start_val):
        """
        Test returns_to_price method.
        """
        # get returns
        ret = self.default_transform_instance.returns(method=ret_type)
        ret_btc = self.btc_transform_instance.returns(method=ret_type)

        # actual
        actual = self.default_transform_instance.returns_to_price(ret, ret_type, start_val)
        actual_btc = self.btc_transform_instance.returns_to_price(ret_btc, ret_type, start_val)

        # shape
        assert actual.shape[0] == self.default_transform_instance.df.shape[0]
        assert actual_btc.shape[0] == self.btc_transform_instance.df.shape[0]
        assert actual.shape[1] == self.default_transform_instance.df.shape[1]
        assert actual_btc.shape[1] == self.btc_transform_instance.df.shape[1]
        # values
        if start_val ==  16616.75:
            assert np.allclose(actual.unstack().close.BTC.iloc[1:],
                           self.default_transform_instance.df.unstack().close.BTC.iloc[1:])
        # dtypes
        assert isinstance(actual, pd.DataFrame)
        assert isinstance(actual_btc, pd.DataFrame)
        assert (actual.dtypes == np.float64).all()
        assert (actual_btc.dtypes == np.float64).all()
        # index
        assert (actual.index == self.default_transform_instance.index).all()
        assert (actual_btc.index == self.btc_transform_instance.index).all()
        # cols
        assert (actual.columns == self.default_transform_instance.df.columns).all()
        assert (actual_btc.columns == self.btc_transform_instance.df.columns).all()

    def test_target_vol(self) -> None:
        """
        Test target_vol method.
        """
        # get actual
        actual = self.default_transform_instance.target_vol()
        actual_btc = self.btc_transform_instance.target_vol()

        # shape
        assert actual.shape[0] == self.default_transform_instance.df.shape[0]
        assert actual_btc.shape[0] == self.btc_transform_instance.df.shape[0]
        assert actual.shape[1] == self.default_transform_instance.df.shape[1]
        assert actual_btc.shape[1] == self.btc_transform_instance.df.shape[1]
        # values
        assert np.allclose(actual.loc[pd.IndexSlice[:, 'BTC'], :], actual_btc, equal_nan=True)
        assert np.allclose(actual.unstack().describe().loc['std'] * np.sqrt(365), 0.15)
        assert np.allclose((actual_btc.describe().loc['std'] * np.sqrt(365)), 0.15)
        # dtypes
        assert isinstance(actual, pd.DataFrame)
        assert isinstance(actual_btc, pd.DataFrame)
        assert (actual.dtypes == np.float64).all()
        assert (actual_btc.dtypes == np.float64).all()
        # index
        assert (actual.index == self.default_transform_instance.index).all()
        assert (actual_btc.index == self.btc_transform_instance.index).all()
        # cols
        assert (actual.columns == self.default_transform_instance.df.columns).all()
        assert (actual_btc.columns == self.btc_transform_instance.df.columns).all()

    @pytest.mark.parametrize("window_size, window_type, central_tendency",
                             [(30, 'ewm', 'mean'), (30, 'rolling', 'mean'), (30, 'rolling', 'median'),
                              (0, 'expanding', 'mean'), (30, 'expanding', 'median')]
                             )
    def test_smooth(self, window_size, window_type, central_tendency) -> None:
        """
        Test smooth method.
        """
        # get actual
        actual = self.default_transform_instance.smooth(window_size=window_size, window_type=window_type,
                                                        central_tendency=central_tendency)
        actual_btc = self.btc_transform_instance.smooth(window_size=window_size, window_type=window_type,
                                                        central_tendency=central_tendency)

        # shape
        assert actual.shape[0] == self.default_transform_instance.df.shape[0]
        assert actual_btc.shape[0] == self.btc_transform_instance.df.shape[0]
        assert actual.shape[1] == self.default_transform_instance.df.shape[1]
        assert actual_btc.shape[1] == self.btc_transform_instance.df.shape[1]
        # values
        assert np.allclose(actual.loc[pd.IndexSlice[:, 'BTC'], :], actual_btc, equal_nan=True)
        # dtypes
        assert isinstance(actual, pd.DataFrame)
        assert isinstance(actual_btc, pd.DataFrame)
        assert (actual.dtypes == np.float64).all()
        assert (actual_btc.dtypes == np.float64).all()
        # index
        assert (actual.index == self.default_transform_instance.index).all()
        assert (actual_btc.index == self.btc_transform_instance.index).all()
        # cols
        assert (actual.columns == self.default_transform_instance.df.columns).all()
        assert (actual_btc.columns == self.btc_transform_instance.df.columns).all()

    @pytest.mark.parametrize("axis, central_tendency, window_type",
                             [('ts', 'mean', 'fixed'), ('ts', 'mean', 'expanding'), ('ts', 'mean', 'rolling'),
                              ('cs', 'mean', 'fixed'),
                              ('ts', 'median', 'fixed'), ('ts', 'median', 'expanding'),
                              ('ts', 'median', 'rolling'), ('cs', 'median', 'fixed')
                              ]
                             )
    def test_center(self, axis, central_tendency, window_type) -> None:
        """
        Test center method.
        """
        # get actual and expected
        actual = self.default_transform_instance.center(axis=axis, central_tendency=central_tendency,
                                                        window_type=window_type)
        actual_btc = self.btc_transform_instance.center(axis=axis, central_tendency=central_tendency,
                                                        window_type=window_type)

        # window type
        if axis == 'ts' and window_type == 'fixed':
            assert actual.shape[0] == self.default_transform_instance.df.shape[0]  # shape
            assert actual_btc.shape[0] == self.btc_transform_instance.df.shape[0]
            assert all(getattr(actual.groupby(level=1), central_tendency)() < 0.0001)  # values
            assert all(getattr(actual_btc, central_tendency)() < 0.0001)
            assert np.allclose(actual.loc[pd.IndexSlice[:, 'BTC'], :], actual_btc, equal_nan=True)
            assert isinstance(actual, pd.DataFrame)  # dtypes
            assert isinstance(actual_btc, pd.DataFrame)
            assert (actual.dtypes == np.float64).all()
            assert (actual_btc.dtypes == np.float64).all()
            assert (actual.index == self.default_transform_instance.df.index).all()  # index
            assert (actual_btc.index == self.btc_transform_instance.df.index).all()
            assert (actual.columns == self.default_transform_instance.df.columns).all()  # cols
            assert (actual_btc.columns == self.btc_transform_instance.df.columns).all()

        elif axis == 'ts' and window_type == 'expanding':
            fixed_expected_close = self.default_transform_instance.center('ts', central_tendency, 'fixed').close
            fixed_expected_close_btc = self.btc_transform_instance.center('ts', central_tendency, 'fixed').close
            assert actual.shape[0] == self.default_transform_instance.df.shape[0]  # shape
            assert actual_btc.shape[0] == self.btc_transform_instance.df.shape[0]
            assert actual.shape[1] == self.default_transform_instance.df.shape[1]
            assert actual_btc.shape[1] == self.btc_transform_instance.df.shape[1]
            assert np.allclose(actual.loc[pd.IndexSlice[:, 'BTC'], :], actual_btc, equal_nan=True)  # values
            assert np.allclose(actual_btc, self.btc_transform_instance.df -
                               getattr(self.btc_transform_instance.df.expanding(), central_tendency)().iloc[1:],
                               equal_nan=True)
            assert np.allclose(actual.unstack().close.iloc[-1], fixed_expected_close.unstack().iloc[-1], equal_nan=True)
            assert np.allclose(actual_btc.close.iloc[-1], fixed_expected_close_btc.iloc[-1], equal_nan=True)
            assert isinstance(actual, pd.DataFrame)  # dtypes
            assert isinstance(actual_btc, pd.DataFrame)
            assert (actual.dtypes == np.float64).all()
            assert (actual_btc.dtypes == np.float64).all()
            assert (actual.index == self.default_transform_instance.df.index).all()  # index
            assert (actual_btc.index == self.btc_transform_instance.df.index).all()
            assert (actual.columns == self.default_transform_instance.df.columns).all()  # cols
            assert (actual_btc.columns == self.btc_transform_instance.df.columns).all()

        elif axis == 'ts' and window_type == 'rolling':
            assert actual.shape[0] == self.default_transform_instance.df.shape[0]  # shape
            assert actual_btc.shape[0] == self.btc_transform_instance.df.shape[0]
            assert actual.shape[1] == self.default_transform_instance.df.shape[1]
            assert actual_btc.shape[1] == self.btc_transform_instance.df.shape[1]
            assert np.allclose(actual.loc[pd.IndexSlice[:, 'BTC'], :], actual_btc, equal_nan=True)  # values
            assert np.allclose(actual_btc, self.btc_transform_instance.df -
                               getattr(self.btc_transform_instance.df.rolling(30, min_periods=1),
                                       central_tendency)().iloc[1:], equal_nan=True)
            assert isinstance(actual, pd.DataFrame)  # dtypes
            assert isinstance(actual_btc, pd.DataFrame)
            assert (actual.dtypes == np.float64).all()
            assert (actual_btc.dtypes == np.float64).all()
            assert (actual.index == self.default_transform_instance.df.index).all()  # index
            assert (actual_btc.index == self.btc_transform_instance.df.index).all()
            assert (actual.columns == self.default_transform_instance.df.columns).all()  # cols
            assert (actual_btc.columns == self.btc_transform_instance.df.columns).all()

        if axis == 'cs':
            assert actual.shape[0] == self.default_transform_instance.df.shape[0]  # shape
            assert actual_btc.shape[0] == self.btc_transform_instance.df.shape[0]
            assert actual.shape[1] == self.default_transform_instance.df.shape[1]
            assert actual_btc.shape[1] == self.btc_transform_instance.df.shape[1]
            assert np.allclose(actual_btc, self.btc_transform_instance.df.subtract(
                getattr(self.btc_transform_instance.df, central_tendency)(axis=1), axis=0), equal_nan=True)
            assert isinstance(actual, pd.DataFrame)  # dtypes
            assert isinstance(actual_btc, pd.DataFrame)
            assert (actual.dtypes == np.float64).all()
            assert (actual_btc.dtypes == np.float64).all()
            assert (actual.index == self.default_transform_instance.df.index).all()  # index
            assert (actual_btc.index == self.btc_transform_instance.df.index).all()
            assert (actual.columns == self.default_transform_instance.df.columns).all()  # cols
            assert (actual_btc.columns == self.btc_transform_instance.df.columns).all()

    @pytest.mark.parametrize("axis, window_type",
                             [('ts', 'fixed'), ('ts', 'expanding'), ('ts', 'rolling'), ('cs', 'fixed')]
                             )
    def test_compute_std(self, axis, window_type) -> None:
        """
        Test compute_std method.
        """
        # get actual and expected
        actual = self.default_transform_instance.compute_std(axis, window_type)
        actual_btc = self.btc_transform_instance.compute_std(axis, window_type)

        # window type
        if axis == 'ts' and window_type == 'fixed':
            assert actual.shape[0] == len(self.default_transform_instance.df.index.droplevel(0).unique())  # shape
            assert actual_btc.shape[0] == self.btc_transform_instance.df.shape[1]
            assert (actual / self.default_transform_instance.df.groupby(level=1).std() == 1.0).all().all()  # values
            assert (actual_btc / self.btc_transform_instance.df.std().to_frame() == 1.0).all().all()
            assert np.allclose(actual.loc['BTC'].to_frame(), actual_btc, equal_nan=True)
            assert isinstance(actual, pd.DataFrame)  # dtypes
            assert isinstance(actual_btc, pd.DataFrame)
            assert (actual.dtypes == np.float64).all()
            assert (actual_btc.dtypes == np.float64).all()
            assert set(actual.index.to_list()) == set(self.default_transform_instance.df.index.droplevel(0).unique())
            assert (actual_btc.index == self.btc_transform_instance.df.columns).all()
            assert (actual.columns == self.default_transform_instance.df.columns).all()  # cols

        elif axis == 'ts' and window_type == 'expanding':
            fixed_expected = self.default_transform_instance.compute_std('ts', 'fixed')
            fixed_expected_btc = self.btc_transform_instance.compute_std('ts', 'fixed')
            assert actual.shape[0] == self.default_transform_instance.df.shape[0]  # shape
            assert actual_btc.shape[0] == self.btc_transform_instance.df.shape[0]
            assert actual.shape[1] == self.default_transform_instance.df.shape[1]
            assert actual_btc.shape[1] == self.btc_transform_instance.df.shape[1]
            assert np.allclose(actual.loc[pd.IndexSlice[:, 'BTC'], :], actual_btc, equal_nan=True)  # values
            assert np.allclose(actual.groupby('ticker').last(), fixed_expected, equal_nan=True)
            assert np.allclose(actual_btc.iloc[-1].to_frame(), fixed_expected_btc)
            assert isinstance(actual, pd.DataFrame)
            assert isinstance(actual_btc, pd.DataFrame)
            assert (actual.dtypes == np.float64).all()
            assert (actual_btc.dtypes == np.float64).all()
            assert (actual.index == self.default_transform_instance.df.index).all()  # index
            assert (actual_btc.index == self.btc_transform_instance.df.index).all()
            assert (actual.columns == self.default_transform_instance.df.columns).all()  # cols
            assert (actual_btc.columns == self.btc_transform_instance.df.columns).all()

        elif axis == 'ts' and window_type == 'rolling':
            assert actual.shape[0] == self.default_transform_instance.df.shape[0]  # shape
            assert actual_btc.shape[0] == self.btc_transform_instance.df.shape[0]
            assert actual.shape[1] == self.default_transform_instance.df.shape[1]
            assert actual_btc.shape[1] == self.btc_transform_instance.df.shape[1]
            assert np.allclose(actual.loc[pd.IndexSlice[:, 'BTC'], :], actual_btc, equal_nan=True)  # values
            assert isinstance(actual, pd.DataFrame)  # dtypes
            assert isinstance(actual_btc, pd.DataFrame)
            assert (actual.dtypes == np.float64).all()
            assert (actual_btc.dtypes == np.float64).all()
            assert (actual.index == self.default_transform_instance.df.index).all()  # index
            assert (actual_btc.index == self.btc_transform_instance.df.index).all()
            assert (actual.columns == self.default_transform_instance.df.columns).all()  # cols
            assert (actual_btc.columns == self.btc_transform_instance.df.columns).all()

        if axis == 'cs':
            assert actual.shape[0] == self.btc_transform_instance.df.shape[0]  # shape
            assert actual_btc.shape[0] == self.btc_transform_instance.df.shape[0]
            assert actual.shape[1] == self.btc_transform_instance.df.shape[1]
            assert isinstance(actual, pd.DataFrame)  # dtypes
            assert isinstance(actual_btc, pd.DataFrame)
            assert (actual.dtypes == np.float64).all()
            assert (actual_btc.dtypes == np.float64).all()
            assert (actual.index == self.btc_transform_instance.df.index).all()  # index
            assert (actual_btc.index == self.btc_transform_instance.df.index).all()
            assert (actual.columns == self.btc_transform_instance.df.columns).all()  # cols

    @pytest.mark.parametrize("q, axis, window_type",
                             [(0.05, 'ts', 'fixed'), (0.05, 'ts', 'expanding'), (0.05, 'ts', 'rolling'),
                              (0.05, 'cs', 'fixed')]
                             )
    def test_compute_quantile(self, q, axis, window_type) -> None:
        """
        Test compute_quantile method.
        """
        # get actual and expected
        actual = self.default_transform_instance.compute_quantile(q, axis, window_type)
        actual_btc = self.btc_transform_instance.compute_quantile(q, axis, window_type)

        # window type
        if axis == 'ts' and window_type == 'fixed':
            assert actual.shape[0] == len(self.default_transform_instance.df.index.droplevel(0).unique())  # shape
            assert actual_btc.shape[0] == self.btc_transform_instance.df.shape[1]
            assert np.allclose(actual.loc['BTC'].to_frame(), actual_btc, equal_nan=True)  # values
            assert isinstance(actual, pd.DataFrame)  # dtypes
            assert isinstance(actual_btc, pd.DataFrame)
            assert (actual.dtypes == np.float64).all()
            assert (actual_btc.dtypes == np.float64).all()
            assert set(actual.index.to_list()) == set(self.default_transform_instance.df.index.droplevel(0).unique())
            assert (actual_btc.index == self.btc_transform_instance.df.columns).all()
            assert (actual.columns == self.default_transform_instance.df.columns).all()  # cols

        elif axis == 'ts' and window_type == 'expanding':
            fixed_expected = self.default_transform_instance.compute_quantile(q, 'ts', 'fixed')
            fixed_expected_btc = self.btc_transform_instance.compute_quantile(q, 'ts', 'fixed')
            assert actual.shape[0] == self.default_transform_instance.df.shape[0]  # shape
            assert actual_btc.shape[0] == self.btc_transform_instance.df.shape[0]
            assert actual.shape[1] == self.default_transform_instance.df.shape[1]
            assert actual_btc.shape[1] == self.btc_transform_instance.df.shape[1]
            assert np.allclose(actual.loc[pd.IndexSlice[:, 'BTC'], :], actual_btc, equal_nan=True)  # values
            assert np.allclose(actual.groupby('ticker').last(), fixed_expected, equal_nan=True)
            assert np.allclose(actual_btc.iloc[-1].to_frame(), fixed_expected_btc)
            assert isinstance(actual, pd.DataFrame)
            assert isinstance(actual_btc, pd.DataFrame)
            assert (actual.dtypes == np.float64).all()
            assert (actual_btc.dtypes == np.float64).all()
            assert (actual.index == self.default_transform_instance.df.index).all()  # index
            assert (actual_btc.index == self.btc_transform_instance.df.index).all()
            assert (actual.columns == self.default_transform_instance.df.columns).all()  # cols
            assert (actual_btc.columns == self.btc_transform_instance.df.columns).all()

        elif axis == 'ts' and window_type == 'rolling':
            assert actual.shape[0] == self.default_transform_instance.df.shape[0]  # shape
            assert actual_btc.shape[0] == self.btc_transform_instance.df.shape[0]
            assert actual.shape[1] == self.default_transform_instance.df.shape[1]
            assert actual_btc.shape[1] == self.btc_transform_instance.df.shape[1]
            assert np.allclose(actual.loc[pd.IndexSlice[:, 'BTC'], :], actual_btc, equal_nan=True)  # values
            assert isinstance(actual, pd.DataFrame)  # dtypes
            assert isinstance(actual_btc, pd.DataFrame)
            assert (actual.dtypes == np.float64).all()
            assert (actual_btc.dtypes == np.float64).all()
            assert (actual.index == self.default_transform_instance.df.index).all()  # index
            assert (actual_btc.index == self.btc_transform_instance.df.index).all()
            assert (actual.columns == self.default_transform_instance.df.columns).all()  # cols
            assert (actual_btc.columns == self.btc_transform_instance.df.columns).all()

        if axis == 'cs':
            assert actual.shape[0] == self.btc_transform_instance.df.shape[0]  # shape
            assert actual_btc.shape[0] == self.btc_transform_instance.df.shape[0]
            assert actual.shape[1] == self.btc_transform_instance.df.shape[1]
            assert isinstance(actual, pd.DataFrame)  # dtypes
            assert isinstance(actual_btc, pd.DataFrame)
            assert (actual.dtypes == np.float64).all()
            assert (actual_btc.dtypes == np.float64).all()
            assert (actual.index == self.btc_transform_instance.df.index).all()  # index
            assert (actual_btc.index == self.btc_transform_instance.df.index).all()
            assert (actual.columns == self.btc_transform_instance.df.columns).all()  # cols

    @pytest.mark.parametrize("axis, window_type",
                             [('ts', 'fixed'), ('ts', 'expanding'), ('ts', 'rolling'), ('cs', 'fixed')]
                             )
    def test_compute_iqr(self, axis, window_type) -> None:
        """
        Test compute_iqr method.
        """
        # get actual and expected
        actual = self.default_transform_instance.compute_iqr(axis, window_type)
        actual_btc = self.btc_transform_instance.compute_iqr(axis, window_type)

        # window type
        if axis == 'ts' and window_type == 'fixed':
            assert actual.shape[0] == len(self.default_transform_instance.df.index.droplevel(0).unique())  # shape
            assert actual_btc.shape[0] == self.btc_transform_instance.df.shape[1]
            assert np.allclose(actual.loc['BTC'].to_frame(), actual_btc, equal_nan=True)  # values
            assert isinstance(actual, pd.DataFrame)  # dtypes
            assert isinstance(actual_btc, pd.DataFrame)
            assert (actual.dtypes == np.float64).all()
            assert (actual_btc.dtypes == np.float64).all()
            assert set(actual.index.to_list()) == set(self.default_transform_instance.df.index.droplevel(0).unique())
            assert (actual_btc.index == self.btc_transform_instance.df.columns).all()
            assert (actual.columns == self.default_transform_instance.df.columns).all()  # cols

        elif axis == 'ts' and window_type == 'expanding':
            fixed_expected = self.default_transform_instance.compute_iqr('ts', 'fixed')
            fixed_expected_btc = self.btc_transform_instance.compute_iqr('ts', 'fixed')
            assert actual.shape[0] == self.default_transform_instance.df.shape[0]  # shape
            assert actual_btc.shape[0] == self.btc_transform_instance.df.shape[0]
            assert actual.shape[1] == self.default_transform_instance.df.shape[1]
            assert actual_btc.shape[1] == self.btc_transform_instance.df.shape[1]
            assert np.allclose(actual.loc[pd.IndexSlice[:, 'BTC'], :], actual_btc, equal_nan=True)  # values
            assert np.allclose(actual.groupby('ticker').last(), fixed_expected, equal_nan=True)
            assert np.allclose(actual_btc.iloc[-1].to_frame(), fixed_expected_btc)
            assert isinstance(actual, pd.DataFrame)
            assert isinstance(actual_btc, pd.DataFrame)
            assert (actual.dtypes == np.float64).all()
            assert (actual_btc.dtypes == np.float64).all()
            assert (actual.index == self.default_transform_instance.df.index).all()  # index
            assert (actual_btc.index == self.btc_transform_instance.df.index).all()
            assert (actual.columns == self.default_transform_instance.df.columns).all()  # cols
            assert (actual_btc.columns == self.btc_transform_instance.df.columns).all()

        elif axis == 'ts' and window_type == 'rolling':
            assert actual.shape[0] == self.default_transform_instance.df.shape[0]  # shape
            assert actual_btc.shape[0] == self.btc_transform_instance.df.shape[0]
            assert actual.shape[1] == self.default_transform_instance.df.shape[1]
            assert actual_btc.shape[1] == self.btc_transform_instance.df.shape[1]
            assert np.allclose(actual.loc[pd.IndexSlice[:, 'BTC'], :], actual_btc, equal_nan=True)  # values
            assert isinstance(actual, pd.DataFrame)  # dtypes
            assert isinstance(actual_btc, pd.DataFrame)
            assert (actual.dtypes == np.float64).all()
            assert (actual_btc.dtypes == np.float64).all()
            assert (actual.index == self.default_transform_instance.df.index).all()  # index
            assert (actual_btc.index == self.btc_transform_instance.df.index).all()
            assert (actual.columns == self.default_transform_instance.df.columns).all()  # cols
            assert (actual_btc.columns == self.btc_transform_instance.df.columns).all()

        if axis == 'cs':
            assert actual.shape[0] == self.btc_transform_instance.df.shape[0]  # shape
            assert actual_btc.shape[0] == self.btc_transform_instance.df.shape[0]
            assert actual.shape[1] == self.btc_transform_instance.df.shape[1]
            assert isinstance(actual, pd.DataFrame)  # dtypes
            assert isinstance(actual_btc, pd.DataFrame)
            assert (actual.dtypes == np.float64).all()
            assert (actual_btc.dtypes == np.float64).all()
            assert (actual.index == self.btc_transform_instance.df.index).all()  # index
            assert (actual_btc.index == self.btc_transform_instance.df.index).all()
            assert (actual.columns == self.btc_transform_instance.df.columns).all()  # cols

    @pytest.mark.parametrize("axis, window_type",
                             [('ts', 'fixed'), ('ts', 'expanding'), ('ts', 'rolling'), ('cs', 'fixed')]
                             )
    def test_compute_mad(self, axis, window_type) -> None:
        """
        Test compute_mad method.
        """
        # get actual and expected
        actual = self.default_transform_instance.compute_mad(axis, window_type)
        actual_btc = self.btc_transform_instance.compute_mad(axis, window_type)

        # window type
        if axis == 'ts' and window_type == 'fixed':
            assert actual.shape[0] == len(self.default_transform_instance.df.index.droplevel(0).unique())  # shape
            assert actual_btc.shape[0] == self.btc_transform_instance.df.shape[1]
            assert np.allclose(actual.loc['BTC'].to_frame(), actual_btc, equal_nan=True)  # values
            assert isinstance(actual, pd.DataFrame)  # dtypes
            assert isinstance(actual_btc, pd.DataFrame)
            assert (actual.dtypes == np.float64).all()
            assert (actual_btc.dtypes == np.float64).all()
            assert set(actual.index.to_list()) == set(self.default_transform_instance.df.index.droplevel(0).unique())
            assert (actual_btc.index == self.btc_transform_instance.df.columns).all()
            assert (actual.columns == self.default_transform_instance.df.columns).all()  # cols

        elif axis == 'ts' and window_type == 'expanding':
            assert actual.shape[0] == self.default_transform_instance.df.shape[0]  # shape
            assert actual_btc.shape[0] == self.btc_transform_instance.df.shape[0]
            assert actual.shape[1] == self.default_transform_instance.df.shape[1]
            assert actual_btc.shape[1] == self.btc_transform_instance.df.shape[1]
            assert np.allclose(actual.loc[pd.IndexSlice[:, 'BTC'], :], actual_btc, equal_nan=True)  # values
            assert isinstance(actual, pd.DataFrame)
            assert isinstance(actual_btc, pd.DataFrame)
            assert (actual.dtypes == np.float64).all()
            assert (actual_btc.dtypes == np.float64).all()
            assert (actual.index == self.default_transform_instance.df.index).all()  # index
            assert (actual_btc.index == self.btc_transform_instance.df.index).all()
            assert (actual.columns == self.default_transform_instance.df.columns).all()  # cols
            assert (actual_btc.columns == self.btc_transform_instance.df.columns).all()

        elif axis == 'ts' and window_type == 'rolling':
            assert actual.shape[0] == self.default_transform_instance.df.shape[0]  # shape
            assert actual_btc.shape[0] == self.btc_transform_instance.df.shape[0]
            assert actual.shape[1] == self.default_transform_instance.df.shape[1]
            assert actual_btc.shape[1] == self.btc_transform_instance.df.shape[1]
            assert np.allclose(actual.loc[pd.IndexSlice[:, 'BTC'], :], actual_btc, equal_nan=True)  # values
            assert isinstance(actual, pd.DataFrame)  # dtypes
            assert isinstance(actual_btc, pd.DataFrame)
            assert (actual.dtypes == np.float64).all()
            assert (actual_btc.dtypes == np.float64).all()
            assert (actual.index == self.default_transform_instance.df.index).all()  # index
            assert (actual_btc.index == self.btc_transform_instance.df.index).all()
            assert (actual.columns == self.default_transform_instance.df.columns).all()  # cols
            assert (actual_btc.columns == self.btc_transform_instance.df.columns).all()

        if axis == 'cs':
            assert actual.shape[0] == self.btc_transform_instance.df.shape[0]  # shape
            assert actual_btc.shape[0] == self.btc_transform_instance.df.shape[0]
            assert actual.shape[1] == self.btc_transform_instance.df.shape[1]
            assert isinstance(actual, pd.DataFrame)  # dtypes
            assert isinstance(actual_btc, pd.DataFrame)
            assert (actual.dtypes == np.float64).all()
            assert (actual_btc.dtypes == np.float64).all()
            assert (actual.index == self.btc_transform_instance.df.index).all()  # index
            assert (actual_btc.index == self.btc_transform_instance.df.index).all()
            assert (actual.columns == self.btc_transform_instance.df.columns).all()  # cols

    @pytest.mark.parametrize("axis, window_type",
                             [('ts', 'fixed'), ('ts', 'expanding'), ('ts', 'rolling'), ('cs', 'fixed')]
                             )
    def test_compute_range(self, axis, window_type) -> None:
        """
        Test compute_range method.
        """
        # get actual and expected
        actual = self.default_transform_instance.compute_range(axis, window_type)
        actual_btc = self.btc_transform_instance.compute_range(axis, window_type)

        # window type
        if axis == 'ts' and window_type == 'fixed':
            assert actual.shape[0] == len(self.default_transform_instance.df.index.droplevel(0).unique())  # shape
            assert actual_btc.shape[0] == self.btc_transform_instance.df.shape[1]
            assert np.allclose(actual.loc['BTC'].to_frame(), actual_btc, equal_nan=True)  # values
            assert isinstance(actual, pd.DataFrame)  # dtypes
            assert isinstance(actual_btc, pd.DataFrame)
            assert (actual.dtypes == np.float64).all()
            assert (actual_btc.dtypes == np.float64).all()
            assert set(actual.index.to_list()) == set(self.default_transform_instance.df.index.droplevel(0).unique())
            assert (actual_btc.index == self.btc_transform_instance.df.columns).all()
            assert (actual.columns == self.default_transform_instance.df.columns).all()  # cols

        elif axis == 'ts' and window_type == 'expanding':
            fixed_expected = self.default_transform_instance.compute_range('ts', 'fixed')
            fixed_expected_btc = self.btc_transform_instance.compute_range('ts', 'fixed')
            assert actual.shape[0] == self.default_transform_instance.df.shape[0]  # shape
            assert actual_btc.shape[0] == self.btc_transform_instance.df.shape[0]
            assert actual.shape[1] == self.default_transform_instance.df.shape[1]
            assert actual_btc.shape[1] == self.btc_transform_instance.df.shape[1]
            assert np.allclose(actual.loc[pd.IndexSlice[:, 'BTC'], :], actual_btc, equal_nan=True)  # values
            assert np.allclose(actual.groupby('ticker').last(), fixed_expected, equal_nan=True)
            assert np.allclose(actual_btc.iloc[-1].to_frame(), fixed_expected_btc)
            assert isinstance(actual, pd.DataFrame)
            assert isinstance(actual_btc, pd.DataFrame)
            assert (actual.dtypes == np.float64).all()
            assert (actual_btc.dtypes == np.float64).all()
            assert (actual.index == self.default_transform_instance.df.index).all()  # index
            assert (actual_btc.index == self.btc_transform_instance.df.index).all()
            assert (actual.columns == self.default_transform_instance.df.columns).all()  # cols
            assert (actual_btc.columns == self.btc_transform_instance.df.columns).all()

        elif axis == 'ts' and window_type == 'rolling':
            assert actual.shape[0] == self.default_transform_instance.df.shape[0]  # shape
            assert actual_btc.shape[0] == self.btc_transform_instance.df.shape[0]
            assert actual.shape[1] == self.default_transform_instance.df.shape[1]
            assert actual_btc.shape[1] == self.btc_transform_instance.df.shape[1]
            assert np.allclose(actual.loc[pd.IndexSlice[:, 'BTC'], :], actual_btc, equal_nan=True)  # values
            assert isinstance(actual, pd.DataFrame)  # dtypes
            assert isinstance(actual_btc, pd.DataFrame)
            assert (actual.dtypes == np.float64).all()
            assert (actual_btc.dtypes == np.float64).all()
            assert (actual.index == self.default_transform_instance.df.index).all()  # index
            assert (actual_btc.index == self.btc_transform_instance.df.index).all()
            assert (actual.columns == self.default_transform_instance.df.columns).all()  # cols
            assert (actual_btc.columns == self.btc_transform_instance.df.columns).all()

        if axis == 'cs':
            assert actual.shape[0] == self.btc_transform_instance.df.shape[0]  # shape
            assert actual_btc.shape[0] == self.btc_transform_instance.df.shape[0]
            assert actual.shape[1] == self.btc_transform_instance.df.shape[1]
            assert isinstance(actual, pd.DataFrame)  # dtypes
            assert isinstance(actual_btc, pd.DataFrame)
            assert (actual.dtypes == np.float64).all()
            assert (actual_btc.dtypes == np.float64).all()
            assert (actual.index == self.btc_transform_instance.df.index).all()  # index
            assert (actual_btc.index == self.btc_transform_instance.df.index).all()
            assert (actual.columns == self.btc_transform_instance.df.columns).all()  # cols

    @pytest.mark.parametrize("axis, window_type",
                             [('ts', 'fixed'), ('ts', 'expanding'), ('ts', 'rolling'), ('cs', 'fixed')]
                             )
    def test_compute_var(self, axis, window_type) -> None:
        """
        Test compute_var method.
        """
        # actual
        actual = self.default_transform_instance.compute_var(axis, window_type)
        actual_btc = self.btc_transform_instance.compute_var(axis, window_type)

        # window type
        if axis == 'ts' and window_type == 'fixed':
            assert actual.shape[0] == len(self.default_transform_instance.df.index.droplevel(0).unique())  # shape
            assert actual_btc.shape[0] == self.btc_transform_instance.df.shape[1]
            assert np.allclose(actual.loc['BTC'].to_frame(), actual_btc, equal_nan=True)  # values
            assert isinstance(actual, pd.DataFrame)  # dtypes
            assert isinstance(actual_btc, pd.DataFrame)
            assert (actual.dtypes == np.float64).all()
            assert (actual_btc.dtypes == np.float64).all()
            assert set(actual.index.to_list()) == set(self.default_transform_instance.df.index.droplevel(0).unique())
            assert (actual_btc.index == self.btc_transform_instance.df.columns).all()
            assert (actual.columns == self.default_transform_instance.df.columns).all()  # cols

        elif axis == 'ts' and window_type == 'expanding':
            fixed_expected = self.default_transform_instance.compute_var('ts', 'fixed')
            fixed_expected_btc = self.btc_transform_instance.compute_var('ts', 'fixed')
            assert actual.shape[0] == self.default_transform_instance.df.shape[0]  # shape
            assert actual_btc.shape[0] == self.btc_transform_instance.df.shape[0]
            assert actual.shape[1] == self.default_transform_instance.df.shape[1]
            assert actual_btc.shape[1] == self.btc_transform_instance.df.shape[1]
            assert np.allclose(actual.loc[pd.IndexSlice[:, 'BTC'], :], actual_btc, equal_nan=True)  # values
            assert np.allclose(actual.groupby('ticker').last(), fixed_expected, equal_nan=True)
            assert np.allclose(actual_btc.iloc[-1].to_frame(), fixed_expected_btc)
            assert isinstance(actual, pd.DataFrame)
            assert isinstance(actual_btc, pd.DataFrame)
            assert (actual.dtypes == np.float64).all()
            assert (actual_btc.dtypes == np.float64).all()
            assert (actual.index == self.default_transform_instance.df.index).all()  # index
            assert (actual_btc.index == self.btc_transform_instance.df.index).all()
            assert (actual.columns == self.default_transform_instance.df.columns).all()  # cols
            assert (actual_btc.columns == self.btc_transform_instance.df.columns).all()

        elif axis == 'ts' and window_type == 'rolling':
            assert actual.shape[0] == self.default_transform_instance.df.shape[0]  # shape
            assert actual_btc.shape[0] == self.btc_transform_instance.df.shape[0]
            assert actual.shape[1] == self.default_transform_instance.df.shape[1]
            assert actual_btc.shape[1] == self.btc_transform_instance.df.shape[1]
            assert np.allclose(actual.loc[pd.IndexSlice[:, 'BTC'], :], actual_btc, equal_nan=True)  # values
            assert isinstance(actual, pd.DataFrame)  # dtypes
            assert isinstance(actual_btc, pd.DataFrame)
            assert (actual.dtypes == np.float64).all()
            assert (actual_btc.dtypes == np.float64).all()
            assert (actual.index == self.default_transform_instance.df.index).all()  # index
            assert (actual_btc.index == self.btc_transform_instance.df.index).all()
            assert (actual.columns == self.default_transform_instance.df.columns).all()  # cols
            assert (actual_btc.columns == self.btc_transform_instance.df.columns).all()

        if axis == 'cs':
            assert actual.shape[0] == self.btc_transform_instance.df.shape[0]  # shape
            assert actual_btc.shape[0] == self.btc_transform_instance.df.shape[0]
            assert actual.shape[1] == self.btc_transform_instance.df.shape[1]
            assert isinstance(actual, pd.DataFrame)  # dtypes
            assert isinstance(actual_btc, pd.DataFrame)
            assert (actual.dtypes == np.float64).all()
            assert (actual_btc.dtypes == np.float64).all()
            assert (actual.index == self.btc_transform_instance.df.index).all()  # index
            assert (actual_btc.index == self.btc_transform_instance.df.index).all()
            assert (actual.columns == self.btc_transform_instance.df.columns).all()  # cols

    @pytest.mark.parametrize("axis, window_type",
                             [('ts', 'fixed'), ('ts', 'expanding'), ('ts', 'rolling')]
                             )
    def test_compute_atr(self, axis, window_type) -> None:
        """
        Test compute_atr method.
        """
        # actual
        actual = self.default_transform_instance.compute_atr(axis, window_type)
        actual_btc = self.btc_transform_instance.compute_atr(axis, window_type)

        # window type
        if axis == 'ts' and window_type == 'fixed':
            assert actual.shape[0] == len(self.default_transform_instance.df.index.droplevel(0).unique())  # shape
            assert np.allclose(actual.loc['BTC'], actual_btc, equal_nan=True)  # values
            assert isinstance(actual, pd.DataFrame)  # dtypes
            assert isinstance(actual_btc, np.float64)
            assert (actual.dtypes == np.float64).all()
            assert type(actual_btc) == np.float64
            assert set(actual.index.to_list()) == set(self.default_transform_instance.df.index.droplevel(0).unique())
            assert actual.columns == ['atr']  # cols

        elif axis == 'ts' and window_type == 'expanding':
            fixed_expected_atr = self.default_transform_instance.compute_atr('ts', 'fixed')
            fixed_expected_atr_btc = self.btc_transform_instance.compute_atr('ts', 'fixed')
            assert actual.shape[0] == self.default_transform_instance.df.shape[0]  # shape
            assert actual_btc.shape[0] == self.btc_transform_instance.df.shape[0]
            assert actual.shape[1] == 1
            assert actual_btc.shape[1] == 1
            assert np.allclose(fixed_expected_atr.loc['BTC'], fixed_expected_atr_btc, equal_nan=True)  # values
            assert ((actual.unstack().atr.iloc[-1].dropna().to_frame('atr') - fixed_expected_atr).sum() == 0).all()
            assert np.allclose(actual_btc.atr.iloc[-1], fixed_expected_atr_btc, equal_nan=True)
            assert isinstance(actual, pd.DataFrame)  # dtypes
            assert isinstance(actual_btc, pd.DataFrame)
            assert (actual.dtypes == np.float64).all()
            assert (actual_btc.dtypes == np.float64).all()
            assert (actual.index == self.default_transform_instance.df.index).all()  # index
            assert (actual_btc.index == self.btc_transform_instance.df.index).all()
            assert actual.columns == ['atr']  # cols
            assert actual_btc.columns == ['atr']

        elif axis == 'ts' and window_type == 'rolling':
            assert actual.shape[0] == self.default_transform_instance.df.shape[0]  # shape
            assert actual_btc.shape[0] == self.btc_transform_instance.df.shape[0]
            assert actual.shape[1] == 1
            assert actual_btc.shape[1] == 1
            assert np.allclose(actual.loc[pd.IndexSlice[:, 'BTC'], :], actual_btc, equal_nan=True)  # values
            assert isinstance(actual, pd.DataFrame)
            assert isinstance(actual_btc, pd.DataFrame)
            assert (actual.dtypes == np.float64).all()
            assert (actual_btc.dtypes == np.float64).all()
            assert (actual.index == self.default_transform_instance.df.index).all()  # index
            assert (actual_btc.index == self.btc_transform_instance.df.index).all()
            assert actual.columns == ['atr']  # cols
            assert actual_btc.columns == ['atr']

    def test_compute_atr_param_errors(self, btc_spot_prices) -> None:
        """
        Test compute_atr method parameter errors.
        """
        # test if data is does not have OHLCV columns
        with pytest.raises(ValueError):
            Transform(btc_spot_prices.drop(columns=['close'])).compute_atr()
        with pytest.raises(ValueError):
            self.default_transform_instance.compute_atr(axis='cs')

    @pytest.mark.parametrize("axis, window_type",
                             [('ts', 'fixed'), ('ts', 'expanding'), ('ts', 'rolling'), ('cs', 'fixed')]
                             )
    def test_compute_percentile(self, axis, window_type) -> None:
        """
        Test compute_percentile method.
        """
        # get actual and expected
        actual = self.default_transform_instance.compute_percentile(axis, window_type)
        actual_btc = self.btc_transform_instance.compute_percentile(axis, window_type)

        # window type
        if axis == 'ts' and window_type == 'fixed':
            assert actual.shape[0] == self.default_transform_instance.df.shape[0]  # shape
            assert actual_btc.shape[0] == self.btc_transform_instance.df.shape[0]
            assert actual.shape[1] == self.default_transform_instance.df.shape[1]
            assert actual_btc.shape[1] == self.btc_transform_instance.df.shape[1]
            assert np.allclose(actual.loc[pd.IndexSlice[:, 'BTC'], :], actual_btc, equal_nan=True)  # values
            assert ((actual.dropna() <= 1.0) & (actual.dropna() >= 0.0)).all().all()
            assert ((actual_btc.dropna() <= 1.0) & (actual_btc.dropna() >= 0.0)).all().all()
            assert isinstance(actual, pd.DataFrame)  # dtypes
            assert isinstance(actual_btc, pd.DataFrame)
            assert (actual.dtypes == np.float64).all()  # dtypes
            assert (actual_btc.dtypes == np.float64).all()
            assert (actual.index == self.default_transform_instance.df.index).all()  # index
            assert (actual_btc.index == self.btc_transform_instance.df.index).all()
            assert (actual.columns == self.default_transform_instance.df.columns).all()  # cols
            assert (actual_btc.columns == self.btc_transform_instance.df.columns).all()

        elif axis == 'ts' and window_type == 'expanding':
            fixed_expected_close = self.default_transform_instance.compute_percentile('ts', 'fixed')
            fixed_expected_close_btc = self.btc_transform_instance.compute_percentile('ts', 'fixed')
            assert actual.shape[0] == self.default_transform_instance.df.shape[0]  # shape
            assert actual_btc.shape[0] == self.btc_transform_instance.df.shape[0]
            assert actual.shape[1] == self.default_transform_instance.df.shape[1]
            assert actual_btc.shape[1] == self.btc_transform_instance.df.shape[1]
            assert np.allclose(actual.loc[pd.IndexSlice[:, 'BTC'], :], actual_btc, equal_nan=True)  # values
            assert np.allclose(actual.unstack().iloc[-1], fixed_expected_close.unstack().iloc[-1], equal_nan=True)
            assert np.allclose(actual_btc.iloc[-1], fixed_expected_close_btc.iloc[-1], equal_nan=True)
            assert ((actual.dropna() <= 1.0) & (actual.dropna() >= 0.0)).all().all()
            assert ((actual_btc.dropna() <= 1.0) & (actual_btc.dropna() >= 0.0)).all().all()
            assert isinstance(actual, pd.DataFrame)  # dtypes
            assert isinstance(actual_btc, pd.DataFrame)
            assert (actual.dtypes == np.float64).all()
            assert (actual_btc.dtypes == np.float64).all()
            assert (actual.index == self.default_transform_instance.df.index).all()  # index
            assert (actual_btc.index == self.btc_transform_instance.df.index).all()
            assert (actual.columns == self.default_transform_instance.df.columns).all()  # cols
            assert (actual_btc.columns == self.btc_transform_instance.df.columns).all()

        elif axis == 'ts' and window_type == 'rolling':
            assert actual.shape[0] == self.default_transform_instance.df.shape[0]  # shape
            assert actual_btc.shape[0] == self.btc_transform_instance.df.shape[0]
            assert actual.shape[1] == self.default_transform_instance.df.shape[1]
            assert actual_btc.shape[1] == self.btc_transform_instance.df.shape[1]
            assert np.allclose(actual.loc[pd.IndexSlice[:, 'BTC'], :], actual_btc, equal_nan=True)  # values
            assert ((actual.dropna() <= 1.0) & (actual.dropna() >= 0.0)).all().all()
            assert ((actual_btc.dropna() <= 1.0) & (actual_btc.dropna() >= 0.0)).all().all()
            assert isinstance(actual, pd.DataFrame)  # dtypes
            assert isinstance(actual_btc, pd.DataFrame)
            assert (actual.dtypes == np.float64).all()
            assert (actual_btc.dtypes == np.float64).all()
            assert (actual.index == self.default_transform_instance.df.index).all()  # index
            assert (actual_btc.index == self.btc_transform_instance.df.index).all()
            assert (actual.columns == self.default_transform_instance.df.columns).all()  # cols
            assert (actual_btc.columns == self.btc_transform_instance.df.columns).all()

        if axis == 'cs':
            assert actual.shape[0] == self.default_transform_instance.df.shape[0]  # shape
            assert actual_btc.shape[0] == self.btc_transform_instance.df.shape[0]
            assert actual.shape[1] == self.default_transform_instance.df.shape[1]
            assert actual_btc.shape[1] == self.btc_transform_instance.df.shape[1]
            assert ((actual.dropna() <= 1.0) & (actual.dropna() >= 0.0)).all().all()  # values
            assert ((actual_btc.dropna() <= 1.0) & (actual_btc.dropna() >= 0.0)).all().all()
            assert isinstance(actual, pd.DataFrame)  # dtypes
            assert isinstance(actual_btc, pd.DataFrame)
            assert (actual.dtypes == np.float64).all()
            assert (actual_btc.dtypes == np.float64).all()
            assert (actual.index == self.default_transform_instance.df.index).all()  # index
            assert (actual_btc.index == self.btc_transform_instance.df.index).all()
            assert (actual.columns == self.btc_transform_instance.df.columns).all()  # cols
            assert (actual_btc.columns == self.btc_transform_instance.df.columns).all()

    @pytest.mark.parametrize("method, axis, window_type", [('std', 'ts', 'fixed'), ('std', 'ts', 'expanding'),
                                                           ('mad', 'ts', 'rolling'), ('range', 'cs', 'fixed')]
                             )
    def test_dispersion(self, method, axis, window_type) -> None:
        """
        Test dispersion method.
        """
        # get actual and expected
        actual = self.default_transform_instance.dispersion(method, axis, window_type)
        actual_btc = self.btc_transform_instance.dispersion(method, axis, window_type)

        # window type
        if axis == 'ts' and window_type == 'fixed':
            assert actual.shape[0] == len(self.default_transform_instance.df.index.droplevel(0).unique())  # shape
            assert actual_btc.shape[0] == self.btc_transform_instance.df.shape[1]
            assert np.allclose(actual.loc['BTC'].to_frame(), actual_btc, equal_nan=True)  # values
            assert isinstance(actual, pd.DataFrame)  # dtypes
            assert isinstance(actual_btc, pd.DataFrame)
            assert (actual.dtypes == np.float64).all()
            assert (actual_btc.dtypes == np.float64).all()
            assert set(actual.index.to_list()) == set(self.default_transform_instance.df.index.droplevel(0).unique())
            assert (actual_btc.index == self.btc_transform_instance.df.columns).all()
            assert (actual.columns == self.default_transform_instance.df.columns).all()  # cols

        elif axis == 'ts' and window_type == 'expanding':
            fixed_expected = self.default_transform_instance.dispersion(method, 'ts', 'fixed')
            fixed_expected_btc = self.btc_transform_instance.dispersion(method, 'ts', 'fixed')
            assert actual.shape[0] == self.default_transform_instance.df.shape[0]  # shape
            assert actual_btc.shape[0] == self.btc_transform_instance.df.shape[0]
            assert actual.shape[1] == self.default_transform_instance.df.shape[1]
            assert actual_btc.shape[1] == self.btc_transform_instance.df.shape[1]
            assert np.allclose(actual.loc[pd.IndexSlice[:, 'BTC'], :], actual_btc, equal_nan=True)  # values
            assert np.allclose(actual.groupby('ticker').last(), fixed_expected, equal_nan=True)
            assert np.allclose(actual_btc.iloc[-1].to_frame(), fixed_expected_btc)
            assert isinstance(actual, pd.DataFrame)
            assert isinstance(actual_btc, pd.DataFrame)
            assert (actual.dtypes == np.float64).all()
            assert (actual_btc.dtypes == np.float64).all()
            assert (actual.index == self.default_transform_instance.df.index).all()  # index
            assert (actual_btc.index == self.btc_transform_instance.df.index).all()
            assert (actual.columns == self.default_transform_instance.df.columns).all()  # cols
            assert (actual_btc.columns == self.btc_transform_instance.df.columns).all()

        elif axis == 'ts' and window_type == 'rolling':
            assert actual.shape[0] == self.default_transform_instance.df.shape[0]  # shape
            assert actual_btc.shape[0] == self.btc_transform_instance.df.shape[0]
            assert actual.shape[1] == self.default_transform_instance.df.shape[1]
            assert actual_btc.shape[1] == self.btc_transform_instance.df.shape[1]
            assert np.allclose(actual.loc[pd.IndexSlice[:, 'BTC'], :], actual_btc, equal_nan=True)  # values
            assert isinstance(actual, pd.DataFrame)  # dtypes
            assert isinstance(actual_btc, pd.DataFrame)
            assert (actual.dtypes == np.float64).all()
            assert (actual_btc.dtypes == np.float64).all()
            assert (actual.index == self.default_transform_instance.df.index).all()  # index
            assert (actual_btc.index == self.btc_transform_instance.df.index).all()
            assert (actual.columns == self.default_transform_instance.df.columns).all()  # cols
            assert (actual_btc.columns == self.btc_transform_instance.df.columns).all()

        if axis == 'cs':
            assert actual.shape[0] == self.btc_transform_instance.df.shape[0]  # shape
            assert actual_btc.shape[0] == self.btc_transform_instance.df.shape[0]
            assert actual.shape[1] == self.btc_transform_instance.df.shape[1]
            assert isinstance(actual, pd.DataFrame)  # dtypes
            assert isinstance(actual_btc, pd.DataFrame)
            assert (actual.dtypes == np.float64).all()
            assert (actual_btc.dtypes == np.float64).all()
            assert (actual.index == self.btc_transform_instance.df.index).all()  # index
            assert (actual_btc.index == self.btc_transform_instance.df.index).all()
            assert (actual.columns == self.btc_transform_instance.df.columns).all()  # cols

    @pytest.mark.parametrize("method, axis, centering, window_type",
                             [
                                 ('z-score', 'ts', True, 'fixed'),
                                 ('z-score', 'ts', True, 'expanding'),
                                 ('z-score', 'ts', True, 'rolling'),
                                 ('z-score', 'cs', True, 'fixed'),
                                 ('iqr', 'ts', True, 'fixed'),
                                 ('iqr', 'ts', True, 'expanding'),
                                 ('iqr', 'ts', True, 'rolling'),
                                 ('iqr', 'cs', True, 'fixed'),
                                 ('mod_z', 'ts', True, 'fixed'),
                                 ('mod_z', 'ts', True, 'expanding'),
                                 ('mod_z', 'ts', True, 'rolling'),
                                 ('mod_z', 'cs', True, 'fixed'),
                                 ('min-max', 'ts', True, 'fixed'),
                                 ('min-max', 'ts', True, 'expanding'),
                                 ('min-max', 'ts', True, 'rolling'),
                                 ('min-max', 'cs', True, 'fixed'),
                                 ('percentile', 'ts', True, 'fixed'),
                                 ('percentile', 'ts', True, 'expanding'),
                                 ('percentile', 'ts', True, 'rolling'),
                                 ('percentile', 'cs', True, 'fixed')
                             ]
                             )
    def test_normalize(self, method, axis, centering, window_type) -> None:
        """
        Test normalize method.
        """
        # get actual and expected
        actual = self.default_transform_instance.normalize(method=method, axis=axis, centering=centering,
                                                           window_type=window_type)
        actual_btc = self.btc_transform_instance.normalize(method=method, axis=axis, centering=centering,
                                                           window_type=window_type)
        # shape
        assert actual.shape[0] == self.default_transform_instance.df.shape[0]
        assert actual_btc.shape[0] == self.btc_transform_instance.df.shape[0]
        assert actual.shape[1] == self.default_transform_instance.df.shape[1]
        assert actual_btc.shape[1] == self.btc_transform_instance.df.shape[1]
        # values
        if method == 'z-score' and axis == 'ts' and window_type == 'fixed':
            assert np.allclose(actual.groupby(level=1).std(), 1.0)
            assert np.allclose(actual_btc.std(), 1.0)
        if axis == 'ts':
            assert np.allclose(actual.loc[pd.IndexSlice[:, 'BTC'], :], actual_btc, equal_nan=True)
        # dtypes
        assert isinstance(actual, pd.DataFrame)
        assert isinstance(actual_btc, pd.DataFrame)
        assert (actual.dtypes == np.float64).all()
        assert (actual_btc.dtypes == np.float64).all()
        # index
        assert (actual.index == self.default_transform_instance.df.index).all()
        assert (actual_btc.index == self.btc_transform_instance.df.index).all()
        # cols
        assert (actual.columns == self.default_transform_instance.df.columns).all()
        assert (actual_btc.columns == self.btc_transform_instance.df.columns).all()

    @pytest.mark.parametrize("bins, axis, window_type",
                             [
                                 (5, 'ts', 'fixed'),
                                 (5, 'ts', 'expanding'),
                                 (5, 'ts', 'rolling'),
                                 (5, 'cs', 'fixed'),
                                 (10, 'ts', 'fixed'),
                                 (10, 'ts', 'expanding'),
                                 (10, 'ts', 'rolling'),
                                 (10, 'cs', 'fixed'),
                             ]
                             )
    def test_quantize(self, bins, axis, window_type) -> None:
        """
        Test quantize method.
        """
        # get actual and expected
        actual = self.default_transform_instance.quantize(bins=bins, axis=axis, window_type=window_type)
        actual_btc = self.btc_transform_instance.quantize(bins=bins, axis=axis, window_type=window_type)

        # shape
        assert actual.shape[0] == self.default_transform_instance.df.shape[0]
        assert actual_btc.shape[0] == self.btc_transform_instance.df.shape[0]
        assert actual.shape[1] == self.default_transform_instance.df.shape[1]
        assert actual_btc.shape[1] == self.btc_transform_instance.df.shape[1]
        # values
        if axis == 'ts':
            fixed_expected_close = self.default_transform_instance.quantize(bins, 'ts', 'fixed')
            fixed_expected_close_btc = self.btc_transform_instance.quantize(bins, 'ts', 'fixed')
            assert (actual.unstack().nunique() == bins).sum() / \
                   (len(actual.index.get_level_values(1).unique()) * actual.shape[1]) >= 0.8  # number of unique values
            assert actual_btc.nunique().sum() / (actual_btc.shape[1] * bins) >= 0.8
            assert np.allclose(actual.loc[pd.IndexSlice[:, 'BTC'], :], actual_btc, equal_nan=True)  # multi vs single

            if window_type == 'fixed':
                counts_per_bin = actual.groupby(level=1).close.value_counts()  # number of values per bin
                counts_per_ticker = actual.groupby(level=1).close.value_counts().groupby('ticker').sum()
                bin_perc = counts_per_bin / counts_per_ticker
                assert ((bin_perc - (1/bins)).abs() < 0.01).sum() / \
                       (len(actual.index.get_level_values(1).unique()) * 5) >= 0.8
                assert (((actual_btc.close.dropna().value_counts() / actual_btc.dropna().shape[0]) - (1/bins)).abs()
                        < 0.01).all()
            if window_type == 'expanding':
                assert np.allclose(actual.unstack().iloc[-1], fixed_expected_close.unstack().iloc[-1], equal_nan=True)
                assert np.allclose(actual_btc.iloc[-1], fixed_expected_close_btc.iloc[-1], equal_nan=True)
        else:
            assert (actual.groupby(level=0).nunique().iloc[-2000:].mean() == bins).all()  # number of unique values
            assert np.allclose(actual.groupby(level=0).close.value_counts(bins=bins).groupby(level=0).mean() /
                   actual.groupby(level=0).close.count(), 1/bins)
        # dtypes
        assert isinstance(actual, pd.DataFrame)
        assert isinstance(actual_btc, pd.DataFrame)
        assert (actual.dtypes == np.float64).all()
        assert (actual_btc.dtypes == np.float64).all()
        # index
        assert (actual.index == self.default_transform_instance.df.index).all()
        assert (actual_btc.index == self.btc_transform_instance.df.index).all()
        # cols
        assert (actual.columns == self.default_transform_instance.df.columns).all()
        assert (actual_btc.columns == self.btc_transform_instance.df.columns).all()

    @pytest.mark.parametrize("bins, axis, method, window_type",
                             [
                                 (4, 'ts', 'quantile', 'fixed'),
                                 (4, 'ts', 'quantile',  'expanding'),
                                 (4, 'ts', 'quantile', 'rolling'),
                                 (4, 'cs', 'quantile', 'fixed'),
                                 (2, 'ts', 'uniform', 'fixed'),
                                 (2, 'ts', 'uniform',  'expanding'),
                                 (2, 'ts', 'uniform', 'rolling'),
                                 (2, 'cs', 'uniform', 'fixed'),
                             ]
                             )
    def test_discretize(self, bins, axis, method, window_type) -> None:
        """
        Test quantize method.
        """
        # get actual and expected
        actual = self.no_missing_transform_instance.discretize(bins=bins, axis=axis, method=method,
                                                               window_type=window_type)
        actual_btc = self.btc_no_missing_transform_instance.discretize(bins=bins, axis=axis, method=method,
                                                                 window_type=window_type)

        # shape
        if window_type == 'fixed':
            assert actual.shape[0] == self.no_missing_transform_instance.df.shape[0]
            assert actual_btc.shape[0] == self.btc_no_missing_transform_instance.df.shape[0]
        assert actual.shape[1] == self.no_missing_transform_instance.df.shape[1]
        assert actual_btc.shape[1] == self.btc_no_missing_transform_instance.df.shape[1]
        # values
        if axis == 'ts':
            fixed_expected_close = self.no_missing_transform_instance.discretize(bins, 'ts', method, 'fixed')
            fixed_expected_close_btc = self.btc_no_missing_transform_instance.discretize(bins, 'ts', method, 'fixed')
            assert (actual.unstack().nunique() == bins).sum() / \
                   (len(actual.index.get_level_values(1).unique()) * actual.shape[1]) >= 0.8  # number of unique values
            assert actual_btc.nunique().sum() / (actual_btc.shape[1] * bins) >= 0.8
            assert np.allclose(actual.loc[pd.IndexSlice[:, 'BTC'], :], actual_btc, equal_nan=True)  # multi vs single

            if window_type == 'fixed' and method == 'quantile':
                counts_per_bin = actual.groupby(level=1).close.value_counts()  # number of values per bin
                counts_per_ticker = actual.groupby(level=1).close.value_counts().groupby('ticker').sum()
                bin_perc = counts_per_bin / counts_per_ticker
                assert ((bin_perc - (1/bins)).abs() < 0.01).sum() / \
                       (len(actual.index.get_level_values(1).unique()) * 5) >= 0.8
                assert (((actual_btc.close.dropna().value_counts() / actual_btc.dropna().shape[0]) - (1/bins)).abs()
                        < 0.01).all()
            if window_type == 'expanding':
                assert np.allclose(actual.unstack().iloc[-1], fixed_expected_close.unstack().iloc[-1], equal_nan=True)
                assert np.allclose(actual_btc.iloc[-1], fixed_expected_close_btc.iloc[-1], equal_nan=True)
        else:
            assert (actual.groupby(level=0).nunique().iloc[-2000:].mean() == bins).all()  # number of unique values
            assert np.allclose(actual.groupby(level=0).close.value_counts(bins=bins).groupby(level=0).mean() /
                   actual.groupby(level=0).close.count(), 1/bins)
        # dtypes
        assert isinstance(actual, pd.DataFrame)
        assert isinstance(actual_btc, pd.DataFrame)
        assert (actual.dtypes == np.float64).all()
        assert (actual_btc.dtypes == np.float64).all()
        # index
        if window_type == 'fixed':
            assert (actual.index == self.no_missing_transform_instance.df.index).all()
            assert (actual_btc.index == self.btc_no_missing_transform_instance.df.index).all()
        # cols
        assert (actual.columns == self.no_missing_transform_instance.df.columns).all()
        assert (actual_btc.columns == self.btc_no_missing_transform_instance.df.columns).all()
