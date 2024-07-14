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
    df = pd.read_csv('../src/factorlab/datasets/data/binance_spot_prices.csv', index_col=['date', 'ticker'],
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


@pytest.fixture
def btc_ohlcv(ohlcv):
    """
    Fixture for daily BTC OHLCV prices.
    """
    # get BTC prices
    df = ohlcv.loc[pd.IndexSlice[:, 'BTC'], :].droplevel(1)

    return df


class TestTrend:
    """
    Test Trend class.
    """
    @pytest.fixture(autouse=True)
    def trend_setup_default(self, ohlcv):
        self.trend_instance = Trend(ohlcv)

    @pytest.fixture(autouse=True)
    def trend_setup_btc_default(self, btc_ohlcv):
        self.btc_trend_instance = Trend(btc_ohlcv)

    def test_initialization(self):
        """
        Test initialization.
        """
        assert isinstance(self.trend_instance, Trend)
        assert isinstance(self.btc_trend_instance, Trend)
        assert isinstance(self.trend_instance.df, pd.DataFrame)
        assert isinstance(self.btc_trend_instance.df, pd.DataFrame)

    @pytest.mark.parametrize("vwap, log",
                             [(True, True), (True, False), (False, False), (False, True)]
                             )
    def test_compute_price(self, ohlcv, btc_ohlcv, vwap, log) -> None:
        """
        Test compute_price method.
        """
        # get actual
        actual = Trend(ohlcv, vwap=vwap, log=log).compute_price()
        actual_btc = Trend(btc_ohlcv, vwap=vwap, log=log).compute_price()

        # shape
        if vwap:
            assert actual.shape[1] == 1
            assert actual_btc.shape[1] == 1
        else:
            assert actual.shape[1] == ohlcv.shape[1]
            assert actual_btc.shape[1] == btc_ohlcv.shape[1]
        assert actual.shape[0] == ohlcv.shape[0]
        assert actual_btc.shape[0] == btc_ohlcv.shape[0]

        # values
        assert np.allclose(actual.loc[pd.IndexSlice[:, 'BTC'], :], actual_btc, equal_nan=True)

        # dtypes
        assert isinstance(actual, pd.DataFrame)
        assert isinstance(actual_btc, pd.DataFrame)
        assert all(actual.dtypes == np.float64)
        assert all(actual_btc.dtypes == np.float64)

        # index
        assert (actual.index == ohlcv.index).all()
        assert (actual_btc.index == btc_ohlcv.index).all()

        # cols
        if vwap:
            assert actual.columns[0] == 'vwap'
            assert actual_btc.columns[0] == 'vwap'
        else:
            assert (actual.columns == ohlcv.columns).all()
            assert (actual_btc.columns == btc_ohlcv.columns).all()

    @pytest.mark.parametrize("vwap, log",
                             [(True, True), (True, False), (False, False), (False, True)]
                             )
    def test_breakout(self, ohlcv, btc_ohlcv, vwap, log) -> None:
        """
        Test breakout method.
        """
        # get actual
        actual = Trend(ohlcv, vwap=vwap, log=log).breakout()
        actual_btc = Trend(btc_ohlcv, vwap=vwap, log=log).breakout()

        # values
        assert not actual.isin([np.inf, -np.inf]).any().any()  # inf
        assert not actual_btc.isin([np.inf, -np.inf]).any().any()
        assert (actual.dropna().abs() <= 1).all().all()
        assert (actual_btc.dropna().abs() <= 1).all().all()
        assert np.allclose(actual.loc[pd.IndexSlice[:, 'BTC'], :], actual_btc, equal_nan=True)

        # shape
        if vwap:
            assert actual.shape[1] == 1
            assert actual_btc.shape[1] == 1
        else:
            assert actual.shape[1] == ohlcv.shape[1]
            assert actual_btc.shape[1] == btc_ohlcv.shape[1]
        assert actual.shape[0] == ohlcv.shape[0]
        assert actual_btc.shape[0] == btc_ohlcv.shape[0]

        # dtypes
        assert isinstance(actual, pd.DataFrame)
        assert isinstance(actual_btc, pd.DataFrame)
        assert all(actual.dtypes == np.float64)
        assert all(actual_btc.dtypes == np.float64)

        # index
        assert (actual.index == ohlcv.index).all()
        assert (actual_btc.index == btc_ohlcv.index).all()

        # cols
        if vwap:
            assert actual.columns[0].split('_')[0] == 'breakout'
            assert actual.columns[0].split('_')[1] == str(self.trend_instance.lookback)
            assert actual_btc.columns[0].split('_')[0] == 'breakout'
            assert actual_btc.columns[0].split('_')[1] == str(self.trend_instance.lookback)
        else:
            assert (actual.columns == ohlcv.columns).all()
            assert (actual_btc.columns == btc_ohlcv.columns).all()

    def test_breakout_param_errors(self) -> None:
        """
        Test breakout method parameter errors.
        """
        # test if method is not valid
        with pytest.raises(ValueError):
            self.trend_instance.breakout(method='invalid')

    def test_price_mom(self) -> None:
        """
        Test price_mom method.
        """
        # get actual
        actual = self.trend_instance.price_mom()
        actual_btc = self.btc_trend_instance.price_mom()
        # values
        assert not actual.isin([np.inf, -np.inf]).any().any()  # inf
        assert not actual_btc.isin([np.inf, -np.inf]).any().any()
        assert np.allclose(actual.loc[pd.IndexSlice[:, 'BTC'], :], actual_btc, equal_nan=True)

        # shape
        assert actual.shape[1] == 1
        assert actual_btc.shape[1] == 1
        assert actual.shape[0] == self.trend_instance.df.shape[0]
        assert actual_btc.shape[0] == self.btc_trend_instance.df.shape[0]

        # dtypes
        assert isinstance(actual, pd.DataFrame)
        assert isinstance(actual_btc, pd.DataFrame)
        assert all(actual.dtypes == np.float64)
        assert all(actual_btc.dtypes == np.float64)

        # index
        assert (actual.index == self.trend_instance.df.index).all()
        assert (actual_btc.index == self.btc_trend_instance.df.index).all()

        # cols
        assert actual.columns[0].split('_')[0] == 'price'
        assert actual.columns[0].split('_')[1] == 'mom'
        assert actual.columns[0].split('_')[-1] == str(self.trend_instance.lookback)

    def test_divergence(self) -> None:
        """
        Test divergence method.
        """
        # get actual
        actual = self.trend_instance.divergence()
        actual_btc = self.btc_trend_instance.divergence()

        # values
        assert not actual.isin([np.inf, -np.inf]).any().any()
        assert not actual_btc.isin([np.inf, -np.inf]).any().any()
        assert (actual.dropna().abs() <= 1).all().all()
        assert (actual_btc.dropna().abs() <= 1).all().all()
        assert np.allclose(actual.loc[pd.IndexSlice[:, 'BTC'], :], actual_btc, equal_nan=True)

        # shape
        assert actual.shape[1] == 1
        assert actual_btc.shape[1] == 1
        assert actual.shape[0] == self.trend_instance.df.shape[0]
        assert actual_btc.shape[0] == self.btc_trend_instance.df.shape[0]

        # dtypes
        assert isinstance(actual, pd.DataFrame)
        assert isinstance(actual_btc, pd.DataFrame)
        assert all(actual.dtypes == np.float64)
        assert all(actual_btc.dtypes == np.float64)

        # index
        assert (actual.index == self.trend_instance.df.index).all()
        assert (actual_btc.index == self.btc_trend_instance.df.index).all()

        # cols
        assert actual.columns[0].split('_')[0] == 'divergence'
        assert actual.columns[0].split('_')[1] == str(self.trend_instance.lookback)

    def test_time_trend(self) -> None:
        """
        Test time_trend method.
        """
        # get actual
        actual = self.trend_instance.time_trend()
        actual_btc = self.btc_trend_instance.time_trend()

        # values
        assert not actual.isin([np.inf, -np.inf]).any().any()
        assert not actual_btc.isin([np.inf, -np.inf]).any().any()
        assert ((actual.dropna().unstack().fillna(0) < 1).sum().sum() /
                (actual.unstack().shape[0] * actual.unstack().shape[1])) > 0.9
        assert ((actual_btc.dropna().fillna(0) < 1).sum().sum() /
                (actual_btc.shape[0] * actual_btc.shape[1])) > 0.9
        assert np.allclose(actual.loc[pd.IndexSlice[:, 'BTC'], :], actual_btc, equal_nan=True)

        # shape
        assert actual.shape[1] == 1
        assert actual_btc.shape[1] == 1
        assert actual.shape[0] == self.trend_instance.df.shape[0]
        assert actual_btc.shape[0] == self.btc_trend_instance.df.shape[0]

        # dtypes
        assert isinstance(actual, pd.DataFrame)
        assert isinstance(actual_btc, pd.DataFrame)
        assert all(actual.dtypes == np.float64)
        assert all(actual_btc.dtypes == np.float64)

        # index
        assert actual.index.difference(self.trend_instance.df.index).empty
        assert (actual_btc.index == self.btc_trend_instance.df.index).all()

        # cols
        assert actual.columns[0].split('_')[0] == 'time'
        assert actual.columns[0].split('_')[1] == 'trend'
        assert actual.columns[0].split('_')[-1] == str(self.trend_instance.lookback)

    def test_price_acc(self) -> None:
        """
        Test price_acc method.
        """
        # get actual
        actual = self.trend_instance.price_acc()
        actual_btc = self.btc_trend_instance.price_acc()

        # values
        assert not actual.isin([np.inf, -np.inf]).any().any()
        assert not actual_btc.isin([np.inf, -np.inf]).any().any()
        assert ((actual.dropna().unstack().fillna(0) < 1).sum().sum() /
                (actual.unstack().shape[0] * actual.unstack().shape[1])) > 0.9
        assert ((actual_btc.dropna().fillna(0) < 1).sum().sum() /
                (actual_btc.shape[0] * actual_btc.shape[1])) > 0.9
        assert np.allclose(actual.loc[pd.IndexSlice[:, 'BTC'], :], actual_btc, equal_nan=True)

        # shape
        assert actual.shape[1] == 1
        assert actual_btc.shape[1] == 1
        assert actual.shape[0] == self.trend_instance.df.shape[0]
        assert actual_btc.shape[0] == self.btc_trend_instance.df.shape[0]

        # dtypes
        assert isinstance(actual, pd.DataFrame)
        assert isinstance(actual_btc, pd.DataFrame)
        assert all(actual.dtypes == np.float64)
        assert all(actual_btc.dtypes == np.float64)

        # index
        assert actual.index.difference(self.trend_instance.df.index).empty
        assert (actual_btc.index == self.btc_trend_instance.df.index).all()

        # cols
        assert actual.columns[0].split('_')[0] == 'price'
        assert actual.columns[0].split('_')[1] == 'acc'
        assert actual.columns[0].split('_')[-1] == str(self.trend_instance.lookback)

    def test_alpha_mom(self) -> None:
        """
        Test alpha_mom method.
        """
        # get actual
        actual = self.trend_instance.alpha_mom()

        # values
        assert not actual.isin([np.inf, -np.inf]).any().any()
        assert ((actual.dropna().unstack().fillna(0) < 1).sum().sum() /
                (actual.unstack().shape[0] * actual.unstack().shape[1])) > 0.9

        # shape
        assert actual.shape[1] == 1

        # dtypes
        assert isinstance(actual, pd.DataFrame)
        assert all(actual.dtypes == np.float64)

        # cols
        assert actual.columns[0].split('_')[0] == 'alpha'
        assert actual.columns[0].split('_')[1] == 'mom'
        assert actual.columns[0].split('_')[-1] == str(self.trend_instance.lookback)

    def test_alpha_mom_param_errors(self) -> None:
        """
        Test alpha_mom method parameter errors.
        """
        # test value error
        with pytest.raises(TypeError):
            self.btc_trend_instance.alpha_mom()

    def test_rsi(self) -> None:
        """
        Test rsi method.
        """
        # get actual
        actual = self.trend_instance.rsi()
        actual_btc = self.btc_trend_instance.rsi()

        # values
        assert not actual.isin([np.inf, -np.inf]).any().any()
        assert not actual_btc.isin([np.inf, -np.inf]).any().any()
        assert (actual.dropna().abs() <= 1).all().all()
        assert (actual_btc.dropna().abs() <= 1).all().all()
        assert np.allclose(actual.loc[pd.IndexSlice[:, 'BTC'], :], actual_btc, equal_nan=True)

        # shape
        assert actual.shape[1] == 1
        assert actual_btc.shape[1] == 1
        assert actual.shape[0] == self.trend_instance.df.shape[0]
        assert actual_btc.shape[0] == self.btc_trend_instance.df.shape[0]

        # dtypes
        assert isinstance(actual, pd.DataFrame)
        assert isinstance(actual_btc, pd.DataFrame)
        assert all(actual.dtypes == np.float64)
        assert all(actual_btc.dtypes == np.float64)

        # index
        assert (actual.index == self.trend_instance.df.index).all()
        assert (actual_btc.index == self.btc_trend_instance.df.index).all()

        # cols
        assert actual.columns[0].split('_')[0] == 'rsi'
        assert actual.columns[0].split('_')[-1] == str(self.trend_instance.lookback)

    def test_stochastic(self) -> None:
        """
        Test stochastic method.
        """
        # get actual
        actual = self.trend_instance.stochastic()
        actual_btc = self.btc_trend_instance.stochastic()

        # values
        assert not actual.isin([np.inf, -np.inf]).any().any()
        assert not actual_btc.isin([np.inf, -np.inf]).any().any()
        assert (actual.dropna().abs() <= 1).all().all()
        assert (actual_btc.dropna().abs() <= 1).all().all()
        assert np.allclose(actual.loc[pd.IndexSlice[:, 'BTC'], :], actual_btc, equal_nan=True)

        # shape
        assert actual.shape[1] == 1
        assert actual_btc.shape[1] == 1
        assert actual.shape[0] == self.trend_instance.df.shape[0]
        assert actual_btc.shape[0] == self.btc_trend_instance.df.shape[0]

        # dtypes
        assert isinstance(actual, pd.DataFrame)
        assert isinstance(actual_btc, pd.DataFrame)
        assert all(actual.dtypes == np.float64)
        assert all(actual_btc.dtypes == np.float64)

        # index
        assert (actual.index == self.trend_instance.df.index).all()
        assert (actual_btc.index == self.btc_trend_instance.df.index).all()

        # cols
        assert actual.columns[0].split('_')[0] == 'stochastic'
        assert actual.columns[0].split('_')[-1] == str(self.trend_instance.lookback)

    def test_stochastic_param_errors(self, ohlcv) -> None:
        """
        Test stochastic method parameter errors.
        """
        # test if method is not valid
        with pytest.raises(ValueError):
            Trend(ohlcv.drop(columns='high')).stochastic()

    def test_intensity(self) -> None:
        """
        Test intensity method.
        """
        # get actual
        actual = self.trend_instance.intensity()
        actual_btc = self.btc_trend_instance.intensity()

        # values
        assert not actual.isin([np.inf, -np.inf]).any().any()
        assert not actual_btc.isin([np.inf, -np.inf]).any().any()
        assert ((actual.abs() < 1).sum() / (actual.shape[0] * actual.shape[1]) > 0.95).all()
        assert ((actual_btc.abs() < 1).sum() / (actual_btc.shape[0] * actual_btc.shape[1]) > 0.95).all()
        assert np.allclose(actual.loc[pd.IndexSlice[:, 'BTC'], :], actual_btc, equal_nan=True)

        # shape
        assert actual.shape[1] == 1
        assert actual_btc.shape[1] == 1
        assert actual.shape[0] == self.trend_instance.df.shape[0]
        assert actual_btc.shape[0] == self.btc_trend_instance.df.shape[0]

        # dtypes
        assert isinstance(actual, pd.DataFrame)
        assert isinstance(actual_btc, pd.DataFrame)
        assert all(actual.dtypes == np.float64)
        assert all(actual_btc.dtypes == np.float64)

        # index
        assert (actual.index == self.trend_instance.df.index).all()
        assert (actual_btc.index == self.btc_trend_instance.df.index).all()

        # cols
        assert actual.columns[0].split('_')[0] == 'intensity'
        assert actual.columns[0].split('_')[-1] == str(self.trend_instance.lookback)

    def test_intensity_param_errors(self, ohlcv) -> None:
        """
        Test intensity method parameter errors.
        """
        # test value error
        with pytest.raises(ValueError):
            Trend(ohlcv.drop(columns='high')).intensity()

    def test_mw_diff(self) -> None:
        """
        Test moving window difference method.
        """
        # get actual
        actual = self.trend_instance.mw_diff()
        actual_btc = self.btc_trend_instance.mw_diff()

        # values
        assert not actual.isin([np.inf, -np.inf]).any().any()
        assert not actual_btc.isin([np.inf, -np.inf]).any().any()
        assert np.allclose(actual.loc[pd.IndexSlice[:, 'BTC'], :], actual_btc, equal_nan=True)

        # shape
        assert actual.shape[1] == 1
        assert actual_btc.shape[1] == 1
        assert actual.shape[0] == self.trend_instance.df.shape[0]
        assert actual_btc.shape[0] == self.btc_trend_instance.df.shape[0]

        # dtypes
        assert isinstance(actual, pd.DataFrame)
        assert isinstance(actual_btc, pd.DataFrame)
        assert all(actual.dtypes == np.float64)
        assert all(actual_btc.dtypes == np.float64)

        # index
        assert (actual.index == self.trend_instance.df.index).all()
        assert (actual_btc.index == self.btc_trend_instance.df.index).all()

        # cols
        assert actual.columns[0].split('_')[0] == 'mw'
        assert actual.columns[0].split('_')[1] == 'diff'
        assert actual.columns[0].split('_')[-1] == str(self.trend_instance.lookback)

    def test_ewma_wxover(self) -> None:
        """
        Test intensity method.
        """
        # get actual
        actual = self.trend_instance.ewma_wxover(signal=True)
        actual_btc = self.btc_trend_instance.ewma_wxover(signal=True)

        # values
        assert not actual.isin([np.inf, -np.inf]).any().any()
        assert not actual_btc.isin([np.inf, -np.inf]).any().any()
        assert (actual.dropna().abs() <= 1).all().all()
        assert (actual_btc.dropna().abs() <= 1).all().all()
        assert np.allclose(actual.loc[pd.IndexSlice[:, 'BTC'], :], actual_btc, equal_nan=True)

        # shape
        assert actual.shape[1] == 1
        assert actual_btc.shape[1] == 1

        # dtypes
        assert isinstance(actual, pd.DataFrame)
        assert isinstance(actual_btc, pd.DataFrame)
        assert all(actual.dtypes == np.float64)
        assert all(actual_btc.dtypes == np.float64)

        # cols
        assert actual.columns[0].split('_')[0] == 'ewma'
        assert actual.columns[0].split('_')[1] == 'wxover'

    def test_energy(self) -> None:
        """
        Test intensity method.
        """
        # get actual
        actual = self.trend_instance.energy()
        actual_btc = self.btc_trend_instance.energy()

        # values
        assert not actual.isin([np.inf, -np.inf]).any().any()
        assert not actual_btc.isin([np.inf, -np.inf]).any().any()
        assert np.allclose(actual.loc[pd.IndexSlice[:, 'BTC'], :], actual_btc, equal_nan=True)

        # shape
        assert actual.shape[1] == 1
        assert actual_btc.shape[1] == 1
        assert actual.shape[0] == self.trend_instance.df.shape[0]
        assert actual_btc.shape[0] == self.btc_trend_instance.df.shape[0]

        # dtypes
        assert isinstance(actual, pd.DataFrame)
        assert isinstance(actual_btc, pd.DataFrame)
        assert all(actual.dtypes == np.float64)
        assert all(actual_btc.dtypes == np.float64)

        # index
        assert (actual.index == self.trend_instance.df.index).all()
        assert (actual_btc.index == self.btc_trend_instance.df.index).all()

        # cols
        assert actual.columns[0].split('_')[0] == 'energy'
        assert actual.columns[0].split('_')[-1] == str(self.trend_instance.lookback)
