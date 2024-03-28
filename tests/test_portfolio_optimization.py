import pytest
import pandas as pd
import numpy as np

from factorlab.feature_engineering.transformations import Transform
from factorlab.feature_engineering.factors.trend import Trend
from factorlab.signal_generation.signal import Signal
from factorlab.strategy_backtesting.portfolio_optimization import PortfolioOptimization


@pytest.fixture
def spot_prices():
    """
    Fixture for crypto spot prices.
    """
    # read csv from datasets/data
    df = pd.read_csv("../src/factorlab/datasets/data/cc_spot_prices.csv", index_col=['date', 'ticker'],
                     parse_dates=True).loc[:, : 'close']

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
def spot_ret(spot_prices):
    """
    Fixture for spot returns.
    """
    # compute returns
    spot_ret = Transform(spot_prices).returns()

    return spot_ret


@pytest.fixture
def btc_spot_ret(spot_ret):
    """
    Fixture for spot returns.
    """
    # get btc returns
    btc_spot_ret = spot_ret.loc[pd.IndexSlice[:, 'BTC'], :].droplevel(1)

    return btc_spot_ret


@pytest.fixture
def price_mom(spot_prices):
    """
    Fixture for crypto price momentum.
    """
    # compute price mom
    price_mom = Trend(spot_prices).price_mom()

    return price_mom


@pytest.fixture
def btc_price_mom(price_mom):
    """
    Fixture for BTC price momentum.
    """
    # compute btc price mom
    btc_price_mom = price_mom.loc[pd.IndexSlice[:, 'BTC'], :].droplevel(1)

    return btc_price_mom


@pytest.fixture
def signals_ts(price_mom, spot_ret):
    """
    Fixture for time series signals.
    """
    # get signals
    signals = Signal(price_mom, spot_ret.close, strategy='ts_ls', factor_bins=5).\
        compute_signals(signal_type='signal', lags=1)

    return signals


@pytest.fixture
def signals_cs(price_mom, spot_ret):
    """
    Fixture for cross-sectional signals.
    """
    # get signals
    signals = Signal(price_mom, spot_ret.close, strategy='cs_ls', factor_bins=5).\
        compute_signals(signal_type='signal', lags=1)

    return signals


@pytest.fixture
def signals_btc_ts(signals_ts):
    """
    Fixture for BTC time series signals.
    """
    # get signals
    signals = signals_ts.loc[pd.IndexSlice[:, 'BTC'], :].droplevel(1)

    return signals


class TestPortfolioOptimization:
    """
    Test class for PortfolioOptimization.
    """
    @pytest.fixture(autouse=True)
    def portfolio_opt_ts(self, signals_ts, spot_ret):
        self.port_opt_ts_instance = PortfolioOptimization(spot_ret.close, signals_ts, method='ew',
                                                          target_vol=0.1, t_cost=0.001)

    @pytest.fixture(autouse=True)
    def portfolio_opt_cs(self, signals_cs, spot_ret):
        self.port_opt_cs_instance = PortfolioOptimization(spot_ret.close, signals_cs, method='ew',
                                                          target_vol=0.1, t_cost=0.001)

    @pytest.fixture(autouse=True)
    def btc_portfolio_opt_ts(self, signals_btc_ts, btc_spot_ret):
        self.btc_port_opt_ts_instance = PortfolioOptimization(btc_spot_ret.close, signals_btc_ts, method='ew',
                                                          target_vol=0.1, t_cost=0.001)

    def test_initialization(self) -> None:
        """
        Test initialization.
        """
        # data type
        assert isinstance(self.port_opt_ts_instance, PortfolioOptimization)
        assert isinstance(self.port_opt_cs_instance, PortfolioOptimization)
        assert isinstance(self.btc_port_opt_ts_instance, PortfolioOptimization)
        assert isinstance(self.port_opt_ts_instance.signals, pd.DataFrame)
        assert isinstance(self.port_opt_cs_instance.signals, pd.DataFrame)
        assert isinstance(self.btc_port_opt_ts_instance.signals, pd.DataFrame)
        assert isinstance(self.port_opt_ts_instance.ret, pd.DataFrame)
        assert isinstance(self.port_opt_cs_instance.ret, pd.DataFrame)
        assert isinstance(self.btc_port_opt_ts_instance.ret, pd.DataFrame)
        assert (self.port_opt_ts_instance.signals.dtypes == np.float64).all()
        assert (self.port_opt_cs_instance.signals.dtypes == np.float64).all()
        assert (self.btc_port_opt_ts_instance.signals.dtypes == np.float64).all()
        assert (self.port_opt_ts_instance.ret.dtypes == np.float64).all()
        assert (self.port_opt_cs_instance.ret.dtypes == np.float64).all()
        assert (self.btc_port_opt_ts_instance.ret.dtypes == np.float64).all()

    def test_compute_signal_returns(self):
        """
        Test compute_signal_returns method.
        """
        # get actual
        actual_ts = self.port_opt_ts_instance.compute_signal_returns()
        actual_cs = self.port_opt_cs_instance.compute_signal_returns()
        actual_btc = self.btc_port_opt_ts_instance.compute_signal_returns()

        # shape
        assert self.port_opt_ts_instance.signals.dropna().shape == actual_ts.shape
        assert self.port_opt_cs_instance.signals.dropna().shape == actual_cs.shape
        assert self.btc_port_opt_ts_instance.signals.dropna().shape == actual_btc.shape
        # dtypes
        assert isinstance(actual_ts, pd.DataFrame)
        assert isinstance(actual_cs, pd.DataFrame)
        assert isinstance(actual_btc, pd.DataFrame)
        assert (actual_ts.dtypes == np.float64).all()
        assert (actual_cs.dtypes == np.float64).all()
        assert (actual_btc.dtypes == np.float64).all()
        # values
        assert np.allclose(actual_ts.loc[pd.IndexSlice[:, 'BTC'], :].dropna(), actual_btc.dropna())
        # index
        assert (actual_ts.dropna().index == self.port_opt_ts_instance.signals.index).all()
        assert (actual_cs.dropna().index == self.port_opt_cs_instance.signals.index).all()
        assert (actual_btc.dropna().index == self.btc_port_opt_ts_instance.signals.index).all()
        # cols
        assert (actual_ts.columns == self.port_opt_ts_instance.signals.columns).all()
        assert (actual_cs.columns == self.port_opt_cs_instance.signals.columns).all()
        assert (actual_btc.columns == self.btc_port_opt_ts_instance.signals.columns).all()

    def test_compute_equal_weights(self):
        """
        Test compute_equal_weights method.
        """
        # get actual
        actual_ts = self.port_opt_ts_instance.compute_ew_weights()
        actual_cs = self.port_opt_cs_instance.compute_ew_weights()

        # shape
        assert self.port_opt_ts_instance.signals.shape == actual_ts.shape
        assert self.port_opt_cs_instance.signals.shape == actual_cs.shape
        # values
        assert ((actual_ts.dropna() >= 0) & (actual_ts.dropna() <= 1)).all().all()
        assert ((actual_cs.dropna() >= 0) & (actual_cs.dropna() <= 1)).all().all()
        assert np.allclose(actual_ts.groupby('date').sum(), 1.0)
        assert np.allclose(actual_cs.groupby('date').sum(), 1.0)
        # dtypes
        assert isinstance(actual_ts, pd.DataFrame)
        assert isinstance(actual_cs, pd.DataFrame)
        assert (actual_ts.dtypes == np.float64).all()
        assert (actual_cs.dtypes == np.float64).all()
        # index
        assert (actual_ts.index == self.port_opt_ts_instance.signals.index).all()
        assert (actual_cs.index == self.port_opt_cs_instance.signals.index).all()
        # cols
        assert (actual_ts.columns == self.port_opt_ts_instance.signals.columns).all()
        assert (actual_cs.columns == self.port_opt_cs_instance.signals.columns).all()

    def test_compute_iv_weights(self):
        """
        Test compute_iv_weights method.
        """
        # get actual
        # get actual
        actual_ts = self.port_opt_ts_instance.compute_iv_weights()
        actual_cs = self.port_opt_cs_instance.compute_iv_weights()
        # shape
        assert self.port_opt_ts_instance.signals.shape == actual_ts.shape
        assert self.port_opt_cs_instance.signals.shape == actual_cs.shape
        # values
        assert ((actual_ts.dropna() >= 0) & (actual_ts.dropna() <= 1)).all().all()
        assert ((actual_cs.dropna() >= 0) & (actual_cs.dropna() <= 1)).all().all()
        assert np.allclose(actual_ts.dropna().groupby('date').sum(), 1.0)
        assert np.allclose(actual_cs.dropna().groupby('date').sum(), 1.0)
        # dtypes
        assert isinstance(actual_ts, pd.DataFrame)
        assert isinstance(actual_cs, pd.DataFrame)
        assert (actual_ts.dtypes == np.float64).all()
        assert (actual_cs.dtypes == np.float64).all()
        # index
        assert (actual_ts.index == self.port_opt_ts_instance.signals.index).all()
        assert (actual_cs.index == self.port_opt_cs_instance.signals.index).all()
        # cols
        assert (actual_ts.columns == self.port_opt_ts_instance.signals.columns).all()
        assert (actual_cs.columns == self.port_opt_cs_instance.signals.columns).all()

    def test_compute_weights(self):
        """
        Test compute_weights method.
        """
        # get actual
        self.port_opt_ts_instance.compute_weights()
        self.port_opt_cs_instance.compute_weights()

        # shape
        assert self.port_opt_ts_instance.weights.shape == self.port_opt_ts_instance.signals.shape
        assert self.port_opt_cs_instance.weights.shape == self.port_opt_cs_instance.signals.shape
        # values
        assert ((self.port_opt_ts_instance.weights.dropna() >= 0) &
                (self.port_opt_ts_instance.weights.dropna() <= 1)).all().all()
        assert ((self.port_opt_cs_instance.weights.dropna() >= 0) &
                (self.port_opt_cs_instance.weights.dropna() <= 1)).all().all()
        assert np.allclose(self.port_opt_ts_instance.weights.dropna().groupby('date').sum(), 1.0)
        assert np.allclose(self.port_opt_cs_instance.weights.dropna().groupby('date').sum(), 1.0)
        # dtypes
        assert isinstance(self.port_opt_ts_instance.weights, pd.DataFrame)
        assert isinstance(self.port_opt_cs_instance.weights, pd.DataFrame)
        assert (self.port_opt_ts_instance.weights.dtypes == np.float64).all()
        assert (self.port_opt_cs_instance.weights.dtypes == np.float64).all()
        # index
        assert (self.port_opt_ts_instance.weights.index == self.port_opt_ts_instance.signals.index).all()
        assert (self.port_opt_cs_instance.weights.index == self.port_opt_cs_instance.signals.index).all()
        # cols
        assert (self.port_opt_ts_instance.weights.columns == self.port_opt_ts_instance.signals.columns).all()
        assert (self.port_opt_cs_instance.weights.columns == self.port_opt_cs_instance.signals.columns).all()

    def test_weighted_signals(self):
        """
        Test weighted_signals method.
        """
        # get actual
        actual_ts = self.port_opt_ts_instance.compute_weighted_signals()
        actual_cs = self.port_opt_cs_instance.compute_weighted_signals()
        actual_btc = self.btc_port_opt_ts_instance.compute_weighted_signals()

        # shape
        assert self.port_opt_ts_instance.signals.shape == actual_ts.shape
        assert self.port_opt_cs_instance.signals.shape == actual_cs.shape
        assert self.btc_port_opt_ts_instance.signals.shape == actual_btc.shape
        # values
        assert ((actual_ts.dropna().abs()).le(
            (self.port_opt_ts_instance.signals.reindex(actual_ts.dropna().index).abs()))).all().all()
        assert ((actual_cs.dropna().abs()).le(
            (self.port_opt_cs_instance.signals.reindex(actual_cs.dropna().index).abs()))).all().all()
        assert (actual_btc == self.btc_port_opt_ts_instance.signals).all().all()
        # dtypes
        assert isinstance(actual_ts, pd.DataFrame)
        assert isinstance(actual_cs, pd.DataFrame)
        assert isinstance(actual_btc, pd.DataFrame)
        assert (actual_ts.dtypes == np.float64).all()
        assert (actual_cs.dtypes == np.float64).all()
        assert (actual_btc.dtypes == np.float64).all()
        # index
        assert (actual_ts.index[-365:] == self.port_opt_ts_instance.signals.index[-365:]).all()
        assert (actual_cs.index[-365:] == self.port_opt_cs_instance.signals.index[-365:]).all()
        assert (actual_btc.index == self.btc_port_opt_ts_instance.signals.index).all()
        # cols
        assert (actual_ts.columns == self.port_opt_ts_instance.signals.columns).all()
        assert (actual_cs.columns == self.port_opt_cs_instance.signals.columns).all()
        assert (actual_btc.columns == self.btc_port_opt_ts_instance.signals.columns).all()

    @pytest.mark.parametrize("rebal_freq", [None, 'monday', '15th', 'month_end', 5, 7])
    def test_rebalance_portfolio(self, rebal_freq):
        """
        Test rebalancing method.
        """
        # get actual
        self.port_opt_ts_instance.rebal_freq = rebal_freq
        actual_ts = self.port_opt_ts_instance.rebalance_portfolio()
        self.port_opt_cs_instance.rebal_freq = rebal_freq
        actual_cs = self.port_opt_cs_instance.rebalance_portfolio()
        self.btc_port_opt_ts_instance.rebal_freq = rebal_freq
        actual_btc = self.btc_port_opt_ts_instance.rebalance_portfolio()

        # shape
        assert actual_ts.unstack().shape == self.port_opt_ts_instance.weighted_signals.unstack().shape
        assert actual_cs.unstack().shape == self.port_opt_cs_instance.weighted_signals.unstack().shape
        assert actual_btc.shape == self.btc_port_opt_ts_instance.weighted_signals.shape
        # values
        if rebal_freq is None:
            assert (actual_ts == self.port_opt_ts_instance.weighted_signals).all().all()
            assert (actual_cs == self.port_opt_cs_instance.weighted_signals).all().all()
            assert (actual_btc == self.btc_port_opt_ts_instance.weighted_signals).all().all()
        elif rebal_freq == 5 or rebal_freq == 7:
            (actual_ts.corrwith(self.port_opt_ts_instance.weighted_signals) > 0.2).all()
            (actual_cs.corrwith(self.port_opt_cs_instance.weighted_signals) > 0.2).all()
            (actual_btc.corrwith(self.btc_port_opt_ts_instance.weighted_signals) > 0.2).all()
        # dtypes
        assert isinstance(actual_ts, pd.DataFrame)
        assert isinstance(actual_cs, pd.DataFrame)
        assert isinstance(actual_btc, pd.DataFrame)
        assert (actual_ts.dtypes == np.float64).all()
        assert (actual_cs.dtypes == np.float64).all()
        assert (actual_btc.dtypes == np.float64).all()
        # index
        idx_ts = actual_ts.unstack()[actual_ts.groupby(level=1).diff().unstack() != 0].dropna(how='all').index
        idx_cs = actual_cs.unstack()[actual_cs.groupby(level=1).diff().unstack() != 0].dropna(how='all').index
        idx_btc = actual_btc[actual_btc.diff() != 0].dropna(how='all').index
        if rebal_freq == 'monday':
            assert (idx_ts.dayofweek == 0).all()
            assert (idx_cs.dayofweek == 0).all()
            assert (idx_btc.dayofweek == 0).all()
        elif rebal_freq == '15th':
            assert (idx_ts.day == 15).all()
            assert (idx_cs.day == 15).all()
            assert (idx_btc.day == 15).all()
        elif rebal_freq == 'month_end':
            assert idx_ts.is_month_end.all()
            assert idx_cs.is_month_end.all()
            assert idx_btc.is_month_end.all()
        elif isinstance(rebal_freq, int):
            idx_ts_chg = idx_ts.to_series().diff()
            idx_cs_chg = idx_cs.to_series().diff()
            idx_btc_chg = idx_btc.to_series().diff()
            assert ((idx_ts_chg == f"{rebal_freq} days").sum() / idx_ts_chg.shape[0]) > 0.95
            assert ((idx_cs_chg == f"{rebal_freq} days").sum() / idx_cs_chg.shape[0]) > 0.95
            assert ((idx_btc_chg == f"{rebal_freq} days").sum() / idx_btc_chg.shape[0]) > 0.95
        # cols
        assert (actual_ts.columns == self.port_opt_ts_instance.weighted_signals.columns).all()
        assert (actual_cs.columns == self.port_opt_cs_instance.weighted_signals.columns).all()
        assert (actual_btc.columns == self.btc_port_opt_ts_instance.weighted_signals.columns).all()

    def test_compute_tcosts(self):
        """
        Test tcosts method.
        """
        # get actual
        actual_ts = self.port_opt_ts_instance.compute_tcosts()
        actual_cs = self.port_opt_cs_instance.compute_tcosts()
        actual_btc = self.btc_port_opt_ts_instance.compute_tcosts()

        # shape
        assert actual_ts.shape == self.port_opt_ts_instance.weighted_signals.shape
        assert actual_cs.shape == self.port_opt_cs_instance.weighted_signals.shape
        assert actual_btc.shape == self.btc_port_opt_ts_instance.weighted_signals.shape
        # values
        assert (actual_ts.dropna() >= 0).all().all()
        assert (actual_cs.dropna() >= 0).all().all()
        assert (actual_btc.dropna() >= 0).all().all()
        # dtypes
        assert isinstance(actual_ts, pd.DataFrame)
        assert isinstance(actual_cs, pd.DataFrame)
        assert isinstance(actual_btc, pd.DataFrame)
        assert (actual_ts.dtypes == np.float64).all()
        assert (actual_cs.dtypes == np.float64).all()
        assert (actual_btc.dtypes == np.float64).all()
        # index
        assert (actual_ts.index == self.port_opt_ts_instance.weighted_signals.index).all()
        assert (actual_cs.index == self.port_opt_cs_instance.weighted_signals.index).all()
        assert (actual_btc.index == self.btc_port_opt_ts_instance.weighted_signals.index).all()
        # cols
        assert (actual_ts.columns == self.port_opt_ts_instance.weighted_signals.columns).all()
        assert (actual_cs.columns == self.port_opt_cs_instance.weighted_signals.columns).all()
        assert (actual_btc.columns == self.btc_port_opt_ts_instance.weighted_signals.columns).all()

    def test_compute_portfolio_returns(self):
        """
        Test compute_portfolio_returns method.
        """
        # get actual
        actual_ts = self.port_opt_ts_instance.compute_portfolio_returns()
        actual_cs = self.port_opt_cs_instance.compute_portfolio_returns()
        actual_btc = self.btc_port_opt_ts_instance.compute_portfolio_returns()

        # shape
        assert actual_ts.shape[0] == self.port_opt_ts_instance.weighted_signals.unstack().shape[0] - 1
        assert actual_ts.shape[1] == self.port_opt_ts_instance.weighted_signals.shape[1]
        assert actual_cs.shape[0] == self.port_opt_cs_instance.weighted_signals.unstack().shape[0] - 1
        assert actual_cs.shape[1] == self.port_opt_cs_instance.weighted_signals.shape[1]
        assert actual_btc.shape[0] == self.btc_port_opt_ts_instance.weighted_signals.shape[0] - 1
        assert actual_btc.shape[1] == self.btc_port_opt_ts_instance.weighted_signals.shape[1]
        # dtypes
        assert isinstance(actual_ts, pd.DataFrame)
        assert isinstance(actual_cs, pd.DataFrame)
        assert isinstance(actual_btc, pd.DataFrame)
        assert (actual_ts.dtypes == np.float64).all()
        assert (actual_cs.dtypes == np.float64).all()
        assert (actual_btc.dtypes == np.float64).all()
        # index
        assert (actual_ts.index == self.port_opt_ts_instance.weighted_signals.unstack().iloc[1:].index).all()
        assert (actual_cs.index == self.port_opt_cs_instance.weighted_signals.unstack().iloc[1:].index).all()
        assert (actual_btc.index == self.btc_port_opt_ts_instance.weighted_signals.index[1:]).all()
        # cols
        assert (actual_ts.columns == self.port_opt_ts_instance.weighted_signals.columns).all()
        assert (actual_cs.columns == self.port_opt_cs_instance.weighted_signals.columns).all()
        assert (actual_btc.columns == self.btc_port_opt_ts_instance.weighted_signals.columns).all()
