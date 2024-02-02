import pytest
import pandas as pd
import numpy as np

from factorlab.feature_engineering.transformations import Transform
from factorlab.feature_engineering.factors.trend import Trend


@pytest.fixture
def ohlcv():
    """
    Fixture for daily OHLCV prices.
    """
    # read csv from datasets/data
    df = pd.read_csv('../src/factorlab/datasets/data/cc_spot_prices.csv', index_col=['date', 'ticker'],
                     parse_dates=True)

    # drop tickers with nobs < ts_obs
    obs = df.groupby(level=1).count().min(axis=1)
    drop_tickers_list = obs[obs < 365].index.to_list()
    df = df.drop(drop_tickers_list, level=1, axis=0)

    # drop tickers with nobs < cs_obs
    obs = df.groupby(level=0).count().min(axis=1)
    idx_start = obs[obs > 3].index[0]
    df = df.unstack()[df.unstack().index > idx_start].stack()

    return df


@pytest.fixture()
def trend_instance_default(ohlcv):
    return Trend(ohlcv)


@pytest.fixture()
def trend_instance_close_only(ohlcv):
    return Trend(ohlcv[['close']], vwap=False)


@pytest.fixture()
def trend_instance_log_false(ohlcv):
    return Trend(ohlcv, log=False)


class TestTrend:
    """
    Test Trend class.
    """
    @pytest.fixture(autouse=True)
    def trend_setup_default(self, trend_instance_default):
        self.default_trend_instance = trend_instance_default

    @pytest.fixture(autouse=True)
    def trend_setup_close_only(self, trend_instance_close_only):
        self.close_only_trend_instance = trend_instance_close_only

    @pytest.fixture(autouse=True)
    def trend_setup_log_false(self, trend_instance_log_false):
        self.log_false_trend_instance = trend_instance_log_false

    def test_initialization(self):
        """
        Test initialization.
        """
        assert isinstance(self.default_trend_instance, Trend)
        assert isinstance(self.close_only_trend_instance, Trend)
        assert isinstance(self.log_false_trend_instance, Trend)
        assert isinstance(self.default_trend_instance.df, pd.DataFrame)
        assert isinstance(self.close_only_trend_instance.df, pd.DataFrame)
        assert isinstance(self.log_false_trend_instance.df, pd.DataFrame)
        assert isinstance(self.default_trend_instance.df.index, pd.MultiIndex)
        assert isinstance(self.close_only_trend_instance.df.index, pd.MultiIndex)
        assert isinstance(self.log_false_trend_instance.df.index, pd.MultiIndex)
        assert 'close' in self.default_trend_instance.df.columns
        assert 'close' in self.close_only_trend_instance.df.columns
        assert 'close' in self.log_false_trend_instance.df.columns

    def test_compute_price(self, ohlcv) -> None:
        """
        Test compute_price method.
        """
        # get actual
        actual = self.default_trend_instance.compute_price()
        expected = Transform(Transform(ohlcv).vwap()[['vwap']]).log()
        actual_vwap_false = self.close_only_trend_instance.compute_price()
        expected_vwap_false = Transform(ohlcv[['close']]).log()
        actual_log_false = self.log_false_trend_instance.compute_price()
        expected_log_false = Transform(Transform(ohlcv).vwap()[['vwap']])

        # test values
        assert all(actual == expected)
        assert all(actual_vwap_false == expected_vwap_false)
        assert all(actual_log_false == expected_log_false)

        # test shape
        assert actual.shape[1] == 1
        assert actual.unstack().shape[1] == expected.unstack().shape[1]
        assert actual.unstack().shape[0] == expected.unstack().shape[0]

        # test data type
        assert isinstance(actual, pd.DataFrame)
        assert all(actual.dtypes == np.float64)

        # test name
        assert actual.columns[0] == 'vwap'

    def test_breakout(self, ohlcv) -> None:
        """
        Test breakout method.
        """
        # get actual
        actual = self.default_trend_instance.breakout()
        actual_close_only = self.close_only_trend_instance.breakout()
        actual_log_false = self.log_false_trend_instance.breakout()

        # check inf
        assert not actual.isin([np.inf, -np.inf]).any().any()
        assert not actual_close_only.isin([np.inf, -np.inf]).any().any()
        assert not actual_log_false.isin([np.inf, -np.inf]).any().any()

        # test values
        assert (actual.dropna().abs() <= 1).all().all()
        assert (actual_close_only.dropna().abs() <= 1).all().all()
        assert (actual_log_false.dropna().abs() <= 1).all().all()

        # test shape
        assert actual.shape[1] == 1
        assert actual.unstack().shape[1] == ohlcv.close.unstack().shape[1]
        assert actual.unstack().shape[0] >= ohlcv.close.unstack().shape[0] - self.default_trend_instance.lookback

        # test data types
        assert isinstance(actual, pd.DataFrame)
        assert all(actual.dtypes == np.float64)

        # test name
        assert actual.columns[0].split('_')[0] == 'breakout'
        assert actual.columns[0].split('_')[-1] == str(self.default_trend_instance.lookback)

    def test_breakout_param_errors(self) -> None:
        """
        Test breakout method parameter errors.
        """
        # test if method is not valid
        with pytest.raises(ValueError):
            self.default_trend_instance.breakout(method='invalid')

    def test_price_mom(self, ohlcv) -> None:
        """
        Test price_mom method.
        """
        # get actual
        actual = self.default_trend_instance.price_mom()
        actual_close_only = self.close_only_trend_instance.price_mom()

        # check inf
        assert not actual.isin([np.inf, -np.inf]).any().any()
        assert not actual_close_only.isin([np.inf, -np.inf]).any().any()

        # test shape
        assert actual.shape[1] == 1
        assert actual.unstack().shape[1] == ohlcv.close.unstack().shape[1]
        assert actual.unstack().shape[0] >= ohlcv.close.unstack().shape[0] - self.default_trend_instance.lookback

        # test data types
        assert isinstance(actual, pd.DataFrame)
        assert all(actual.dtypes == np.float64)

        # test name
        assert actual.columns[0].split('_')[0] == 'price'
        assert actual.columns[0].split('_')[1] == 'mom'
        assert actual.columns[0].split('_')[-1] == str(self.default_trend_instance.lookback)

    def test_divergence(self, ohlcv) -> None:
        """
        Test divergence method.
        """
        # get actual
        actual = self.default_trend_instance.divergence()
        actual_close_only = self.close_only_trend_instance.divergence()
        actual_log_false = self.log_false_trend_instance.divergence()

        # check inf
        assert not actual.isin([np.inf, -np.inf]).any().any()
        assert not actual_close_only.isin([np.inf, -np.inf]).any().any()
        assert not actual_log_false.isin([np.inf, -np.inf]).any().any()

        # test values
        assert (actual.dropna().abs() <= 1).all().all()
        assert (actual_close_only.dropna().abs() <= 1).all().all()
        assert (actual_log_false.dropna().abs() <= 1).all().all()

        # test shape
        assert actual.shape[1] == 1
        assert actual.unstack().shape[1] == ohlcv.close.unstack().shape[1]
        assert actual.unstack().shape[0] >= ohlcv.close.unstack().shape[0] - self.default_trend_instance.lookback

        # test data types
        assert isinstance(actual, pd.DataFrame)
        assert all(actual.dtypes == np.float64)

        # test name
        assert actual.columns[0].split('_')[0] == 'divergence'
        assert actual.columns[0].split('_')[-1] == str(self.default_trend_instance.lookback)

    def test_time_trend(self, ohlcv) -> None:
        """
        Test time_trend method.
        """
        # get actual
        actual = self.default_trend_instance.time_trend()
        actual_close_only = self.close_only_trend_instance.time_trend()
        actual_log_false = self.log_false_trend_instance.time_trend()

        # check inf
        assert not actual.isin([np.inf, -np.inf]).any().any()
        assert not actual_close_only.isin([np.inf, -np.inf]).any().any()
        assert not actual_log_false.isin([np.inf, -np.inf]).any().any()

        # test values
        assert ((actual.dropna().unstack().fillna(0) < 1).sum().sum() /
                (actual.unstack().shape[0] * actual.unstack().shape[1])) > 0.9
        assert ((actual_close_only.dropna().unstack().fillna(0) < 1).sum().sum() /
                (actual_close_only.unstack().shape[0] * actual_close_only.unstack().shape[1])) > 0.9
        assert ((actual_log_false.dropna().unstack().fillna(0) < 1).sum().sum() /
                (actual_log_false.unstack().shape[0] * actual_log_false.unstack().shape[1])) > 0.9

        # test shape
        assert actual.shape[1] == 1
        assert actual.unstack().shape[1] == ohlcv.close.unstack().shape[1]
        assert actual.unstack().shape[0] >= ohlcv.close.unstack().shape[0] - self.default_trend_instance.lookback

        # test data types
        assert isinstance(actual, pd.DataFrame)
        assert all(actual.dtypes == np.float64)

        # test name
        assert actual.columns[0].split('_')[0] == 'time'
        assert actual.columns[0].split('_')[1] == 'trend'
        assert actual.columns[0].split('_')[-1] == str(self.default_trend_instance.lookback)

    def test_price_acc(self, ohlcv) -> None:
        """
        Test price_acc method.
        """
        # get actual
        actual = self.default_trend_instance.price_acc()
        actual_log_false = self.log_false_trend_instance.price_acc()
        actual_close_only = self.close_only_trend_instance.price_acc()

        # check inf
        assert not actual.isin([np.inf, -np.inf]).any().any()
        assert not actual_close_only.isin([np.inf, -np.inf]).any().any()
        assert not actual_log_false.isin([np.inf, -np.inf]).any().any()

        # test values
        assert ((actual.dropna().unstack().fillna(0) < 1).sum().sum() /
                (actual.unstack().shape[0] * actual.unstack().shape[1])) > 0.9
        assert ((actual_close_only.dropna().unstack().fillna(0) < 1).sum().sum() /
                (actual_close_only.unstack().shape[0] * actual_close_only.unstack().shape[1])) > 0.9
        assert ((actual_log_false.dropna().unstack().fillna(0) < 1).sum().sum() /
                (actual_log_false.unstack().shape[0] * actual_log_false.unstack().shape[1])) > 0.9

        # test shape
        assert actual.shape[1] == 1
        assert actual.unstack().shape[1] == ohlcv.close.unstack().shape[1]
        assert actual.unstack().shape[0] >= ohlcv.close.unstack().shape[0] - self.default_trend_instance.lookback

        # test data types
        assert isinstance(actual, pd.DataFrame)
        assert all(actual.dtypes == np.float64)

        # test name
        assert actual.columns[0].split('_')[0] == 'price'
        assert actual.columns[0].split('_')[1] == 'acc'
        assert actual.columns[0].split('_')[-1] == str(self.default_trend_instance.lookback)

    def test_alpha_mom(self, ohlcv) -> None:
        """
        Test alpha_mom method.
        """
        # get actual
        actual = self.default_trend_instance.alpha_mom()
        actual_log_false = self.log_false_trend_instance.alpha_mom()
        actual_close_only = self.close_only_trend_instance.alpha_mom()

        # check inf
        assert not actual.isin([np.inf, -np.inf]).any().any()
        assert not actual_close_only.isin([np.inf, -np.inf]).any().any()
        assert not actual_log_false.isin([np.inf, -np.inf]).any().any()

        # test values
        assert ((actual.dropna().unstack().fillna(0) < 1).sum().sum() /
                (actual.unstack().shape[0] * actual.unstack().shape[1])) > 0.9
        assert ((actual_close_only.dropna().unstack().fillna(0) < 1).sum().sum() /
                (actual_close_only.unstack().shape[0] * actual_close_only.unstack().shape[1])) > 0.9
        assert ((actual_log_false.dropna().unstack().fillna(0) < 1).sum().sum() /
                (actual_log_false.unstack().shape[0] * actual_log_false.unstack().shape[1])) > 0.9

        # test shape
        assert actual.shape[1] == 1
        assert actual.unstack().shape[1] == ohlcv.close.unstack().shape[1]
        assert actual.unstack().shape[0] >= ohlcv.close.unstack().shape[0] - self.default_trend_instance.lookback

        # test data types
        assert isinstance(actual, pd.DataFrame)
        assert all(actual.dtypes == np.float64)

        # test name
        assert actual.columns[0].split('_')[0] == 'alpha'
        assert actual.columns[0].split('_')[1] == 'mom'
        assert actual.columns[0].split('_')[-1] == str(self.default_trend_instance.lookback)

    def test_rsi(self, ohlcv) -> None:
        """
        Test rsi method.
        """
        # get actual
        actual = self.default_trend_instance.rsi()
        actual_log_false = self.log_false_trend_instance.rsi()
        actual_close_only = self.close_only_trend_instance.rsi()

        # check inf
        assert not actual.isin([np.inf, -np.inf]).any().any()
        assert not actual_close_only.isin([np.inf, -np.inf]).any().any()
        assert not actual_log_false.isin([np.inf, -np.inf]).any().any()

        # test values
        assert (actual.dropna().abs() <= 1).all().all()
        assert (actual_close_only.dropna().abs() <= 1).all().all()
        assert (actual_log_false.dropna().abs() <= 1).all().all()

        # test shape
        assert actual.shape[1] == 1
        assert actual.unstack().shape[1] == ohlcv.close.unstack().shape[1]
        assert actual.unstack().shape[0] >= ohlcv.close.unstack().shape[0] - self.default_trend_instance.lookback

        # test data types
        assert isinstance(actual, pd.DataFrame)
        assert all(actual.dtypes == np.float64)

        # test name
        assert actual.columns[0].split('_')[0] == 'rsi'
        assert actual.columns[0].split('_')[-1] == str(self.default_trend_instance.lookback)

    def test_stochastic(self, ohlcv) -> None:
        """
        Test stochastic method.
        """
        # get actual
        actual = self.default_trend_instance.stochastic()
        actual_log_false = self.log_false_trend_instance.stochastic()

        # check inf
        assert not actual.isin([np.inf, -np.inf]).any().any()
        assert not actual_log_false.isin([np.inf, -np.inf]).any().any()

        # test values
        assert (actual.dropna().abs() <= 1).all().all()
        assert (actual_log_false.dropna().abs() <= 1).all().all()

        # test shape
        assert actual.shape[1] == 1
        assert actual.unstack().shape[1] == ohlcv.close.unstack().shape[1]
        assert actual.unstack().shape[0] >= ohlcv.close.unstack().shape[0] - self.default_trend_instance.lookback

        # test data types
        assert isinstance(actual, pd.DataFrame)
        assert all(actual.dtypes == np.float64)

        # test name
        assert actual.columns[0].split('_')[0] == 'stochastic'
        assert actual.columns[0].split('_')[-1] == str(self.default_trend_instance.lookback)

    def test_stochastic_param_errors(self) -> None:
        """
        Test stochastic method parameter errors.
        """
        # test if method is not valid
        with pytest.raises(ValueError):
            self.close_only_trend_instance.stochastic()

    def test_intensity(self, ohlcv) -> None:
        """
        Test intensity method.
        """
        # get actual
        actual = self.default_trend_instance.intensity()
        actual_log_false = self.log_false_trend_instance.intensity()

        # check inf
        assert not actual.isin([np.inf, -np.inf]).any().any()
        assert not actual_log_false.isin([np.inf, -np.inf]).any().any()

        # test values
        assert ((actual.dropna().unstack().fillna(0) < 1).sum().sum() /
                (actual.unstack().shape[0] * actual.unstack().shape[1])) > 0.9
        assert ((actual_log_false.dropna().unstack().fillna(0) < 1).sum().sum() /
                (actual_log_false.unstack().shape[0] * actual_log_false.unstack().shape[1])) > 0.9

        # test shape
        assert actual.shape[1] == 1
        assert actual.unstack().shape[1] == ohlcv.close.unstack().shape[1]
        assert actual.unstack().shape[0] >= ohlcv.close.unstack().shape[0] - self.default_trend_instance.lookback

        # test data types
        assert isinstance(actual, pd.DataFrame)
        assert all(actual.dtypes == np.float64)

        # test name
        assert actual.columns[0].split('_')[0] == 'intensity'
        assert actual.columns[0].split('_')[-1] == str(self.default_trend_instance.lookback)

    def test_intensity_param_errors(self):
        """
        Test intensity method parameter errors.
        """
        # test value error
        with pytest.raises(ValueError):
            self.close_only_trend_instance.intensity()

    def test_mw_diff(self, ohlcv) -> None:
        """
        Test moving window difference method.
        """
        # get actual
        actual = self.default_trend_instance.mw_diff()
        actual_close_only = self.close_only_trend_instance.mw_diff()

        # check inf
        assert not actual.isin([np.inf, -np.inf]).any().any()
        assert not actual_close_only.isin([np.inf, -np.inf]).any().any()

        # test values
        assert ((actual.dropna().unstack().fillna(0) < 1).sum().sum() /
                (actual.unstack().shape[0] * actual.unstack().shape[1])) > 0.9
        assert ((actual_close_only.dropna().unstack().fillna(0) < 1).sum().sum() /
                (actual_close_only.unstack().shape[0] * actual_close_only.unstack().shape[1])) > 0.9

        # test shape
        assert actual.shape[1] == 1
        assert actual.unstack().shape[1] == ohlcv.close.unstack().shape[1]
        assert actual.unstack().shape[0] >= ohlcv.close.unstack().shape[0] - self.default_trend_instance.lookback

        # test data types
        assert isinstance(actual, pd.DataFrame)
        assert all(actual.dtypes == np.float64)

        # test name
        assert actual.columns[0].split('_')[0] == 'mw'
        assert actual.columns[0].split('_')[1] == 'diff'
        assert actual.columns[0].split('_')[-1] == str(self.default_trend_instance.lookback)

    def test_ewma_wxover(self, ohlcv) -> None:
        """
        Test intensity method.
        """
        # get actual
        actual = self.default_trend_instance.ewma_wxover(signal=True)
        actual_close_only = self.close_only_trend_instance.ewma_wxover(signal=True)
        actual_log_false = self.log_false_trend_instance.ewma_wxover(signal=True)

        # check inf
        assert not actual.isin([np.inf, -np.inf]).any().any()
        assert not actual_close_only.isin([np.inf, -np.inf]).any().any()
        assert not actual_log_false.isin([np.inf, -np.inf]).any().any()

        # test values
        assert (actual.dropna().abs() <= 1).all().all()
        assert (actual_close_only.dropna().abs() <= 1).all().all()
        assert (actual_log_false.dropna().abs() <= 1).all().all()

        # test shape
        assert actual.shape[1] == 1
        assert actual.unstack().shape[1] == ohlcv.close.unstack().shape[1]
        assert actual.unstack().shape[0] >= ohlcv.close.unstack().shape[0] - 365

        # test data types
        assert isinstance(actual, pd.DataFrame)
        assert all(actual.dtypes == np.float64)

        # test name
        assert actual.columns[0].split('_')[0] == 'ewma'
        assert actual.columns[0].split('_')[1] == 'wxover'

    def test_energy(self, ohlcv) -> None:
        """
        Test intensity method.
        """
        # get actual
        actual = self.default_trend_instance.energy()
        actual_close_only = self.close_only_trend_instance.energy()
        actual_log_false = self.log_false_trend_instance.energy()

        # check inf
        assert not actual.isin([np.inf, -np.inf]).any().any()
        assert not actual_close_only.isin([np.inf, -np.inf]).any().any()
        assert not actual_log_false.isin([np.inf, -np.inf]).any().any()

        # test shape
        assert actual.shape[1] == 1
        assert actual.unstack().shape[1] == ohlcv.close.unstack().shape[1]
        assert actual.unstack().shape[0] >= ohlcv.close.unstack().shape[0] - self.default_trend_instance.lookback

        # test data types
        assert isinstance(actual, pd.DataFrame)
        assert all(actual.dtypes == np.float64)

        # test name
        assert actual.columns[0].split('_')[0] == 'energy'
        assert actual.columns[0].split('_')[1] == str(self.default_trend_instance.lookback)
