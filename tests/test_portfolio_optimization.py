import pytest
import pandas as pd
import numpy as np

from factorlab.feature_engineering.factors.trend import Trend
from factorlab.signal_generation.signal import Signal
from factorlab.strategy_backtesting.portfolio_optimization._portfolio_optimization import PortfolioOptimization
from factorlab.strategy_backtesting.portfolio_optimization.naive import NaiveOptimization
from factorlab.strategy_backtesting.portfolio_optimization.mvo import MVO
from factorlab.strategy_backtesting.portfolio_optimization.clustering import HRP, HERC


@pytest.fixture
def binance_ohlc():
    """
    Fixture for OHLCV prices.
    """
    # read csv from datasets/data
    df = pd.read_csv("../src/factorlab/datasets/data/binance_spot_prices.csv", index_col=['date', 'ticker'],
                     parse_dates=['date']).loc[:, : 'close']

    # drop tickers with nobs < ts_obs
    obs = df.groupby(level=1).count().min(axis=1)
    drop_tickers_list = obs[obs < 365].index.to_list()
    df = df.drop(drop_tickers_list, level=1, axis=0)

    # drop tickers with nobs < cs_obs
    obs = df.groupby(level=0).count().min(axis=1)
    idx_start = obs[obs > 3].index[0]
    df = df.unstack()[df.unstack().index > idx_start].stack(future_stack=True)

    ohlc = df.copy()

    return ohlc


@pytest.fixture
def binance_ret(binance_ohlc):
    """
    Fixture for asset returns.
    """
    # asset returns
    ret = binance_ohlc.groupby(level=1).close.pct_change().dropna(how='all')

    return ret


@pytest.fixture
def trend_ret_multi(binance_ohlc, binance_ret):
    """
    Fixture for trend factor.
    """
    # trend factor
    trend_df = Trend(binance_ohlc, vwap=True, log=True, lookback=20).price_mom()
    trend_df['price_mom_30'] = Trend(binance_ohlc, vwap=True, log=True, lookback=30).price_mom()

    # signal
    trend_ts_ret = Signal(binance_ret, trend_df).compute_signal_returns()

    return trend_ts_ret


@pytest.fixture
def trend_ret_single(trend_ret_multi):
    """
    Fixture for single asset trend factor.
    """
    return trend_ret_multi['price_mom_30'].unstack().dropna(how='all')


class TestPortfolioOptimization:
    """
    Test class for Portfolio Optimization.
    """

    @pytest.fixture(autouse=True)
    def multi_opt_instance(self, trend_ret_multi):
        self.multi_opt_instance = PortfolioOptimization(trend_ret_multi, window_size=300, target_return=0.75,
                                                        target_risk=0.50)

    @pytest.fixture(autouse=True)
    def single_opt_instance(self, trend_ret_single):
        self.single_opt_instance = PortfolioOptimization(trend_ret_single, window_size=300, target_return=0.75,
                                                         target_risk=0.50)

    def test_initialization(self):
        """
        Test initialization.
        """
        # types
        assert isinstance(self.multi_opt_instance, PortfolioOptimization)
        assert isinstance(self.single_opt_instance, PortfolioOptimization)
        assert isinstance(self.multi_opt_instance.returns, pd.DataFrame)
        assert isinstance(self.single_opt_instance.returns, pd.DataFrame)
        assert isinstance(self.multi_opt_instance.method, str)
        assert isinstance(self.single_opt_instance.method, str)
        assert isinstance(self.multi_opt_instance.lags, int)
        assert isinstance(self.single_opt_instance.lags, int)
        assert isinstance(self.multi_opt_instance.max_weight, float)
        assert isinstance(self.single_opt_instance.max_weight, float)
        assert isinstance(self.multi_opt_instance.min_weight, float)
        assert isinstance(self.single_opt_instance.min_weight, float)
        assert isinstance(self.multi_opt_instance.risk_free_rate, float)
        assert isinstance(self.single_opt_instance.risk_free_rate, float)
        assert isinstance(self.multi_opt_instance.as_excess_returns, bool)
        assert isinstance(self.single_opt_instance.as_excess_returns, bool)
        assert isinstance(self.multi_opt_instance.n_jobs, int)
        assert isinstance(self.single_opt_instance.n_jobs, int)
        # vals
        assert self.multi_opt_instance.method == 'equal_weight'
        assert self.single_opt_instance.method == 'equal_weight'
        assert self.multi_opt_instance.lags == 1
        assert self.single_opt_instance.lags == 1
        assert self.multi_opt_instance.risk_free_rate == 0.0
        assert self.single_opt_instance.risk_free_rate == 0.0
        assert self.multi_opt_instance.as_excess_returns is False
        assert self.single_opt_instance.as_excess_returns is False
        assert self.multi_opt_instance.max_weight == 1.0
        assert self.single_opt_instance.max_weight == 1.0
        assert self.multi_opt_instance.min_weight == 0.0
        assert self.single_opt_instance.min_weight == 0.0
        assert self.multi_opt_instance.leverage == 1.0
        assert self.single_opt_instance.leverage == 1.0
        assert self.multi_opt_instance.risk_aversion == 1.0
        assert self.single_opt_instance.risk_aversion == 1.0
        assert self.multi_opt_instance.exp_ret_method == 'mean'
        assert self.single_opt_instance.exp_ret_method == 'mean'
        assert self.multi_opt_instance.cov_matrix_method == 'covariance'
        assert self.single_opt_instance.cov_matrix_method == 'covariance'
        assert self.multi_opt_instance.target_return == 0.75
        assert self.single_opt_instance.target_return == 0.75
        assert self.multi_opt_instance.target_risk == 0.50
        assert self.single_opt_instance.target_risk == 0.50
        assert self.multi_opt_instance.risk_measure == 'variance'
        assert self.single_opt_instance.risk_measure == 'variance'
        assert self.multi_opt_instance.alpha == 0.05
        assert self.single_opt_instance.alpha == 0.05
        assert self.multi_opt_instance.window_type == 'rolling'
        assert self.single_opt_instance.window_type == 'rolling'

    def test_get_optimizer(self):
        """
        Test get_optimizer.
        """
        # get optimizer
        opt = self.single_opt_instance.get_optimizer(self.single_opt_instance.returns)

        # types
        if self.single_opt_instance.method in ['equal_weight', 'inverse_variance', 'inverse_vol', 'target_vol',
                                               'random']:
            assert isinstance(opt, NaiveOptimization)
        elif self.single_opt_instance.method in ['min_vol', 'max_return_min_vol', 'max_sharpe',
                                                 'max_diversification', 'efficient_return',
                                                 'efficient_risk', 'risk_parity']:
            assert isinstance(opt, MVO)
        elif self.single_opt_instance.method == 'hrp':
            assert isinstance(opt, HRP)
        elif self.single_opt_instance.method == 'herc':
            assert isinstance(opt, HERC)

    @pytest.mark.parametrize('method', ['equal_weight', 'inverse_variance', 'inverse_vol', 'target_vol', 'random',
                                        'min_vol', 'max_return_min_vol', 'max_sharpe', 'max_diversification',
                                        'efficient_return', 'efficient_risk', 'risk_parity', 'hrp', 'herc'])
    def test_compute_fixed_weights(self, method):
        """
        Test compute_fixed_weights.
        """
        # compute fixed weights
        self.single_opt_instance.method = method
        weights = self.single_opt_instance.compute_fixed_weights()

        # dtypes
        assert isinstance(weights, pd.DataFrame)
        assert (weights.dtypes == 'float64').all()
        # shape
        assert weights.shape[1] == self.single_opt_instance.returns.shape[1]
        # values
        assert (weights >= 0).all().all()
        if self.single_opt_instance.method != 'target_vol':
            assert np.isclose(weights.sum(axis=1), 1)
        # cols
        assert set(weights.columns.to_list()) == set(self.single_opt_instance.returns.columns.to_list())
        # index
        assert (weights.index == self.single_opt_instance.returns.index[-1]).all()

    @pytest.mark.parametrize('method', ['equal_weight', 'inverse_variance', 'inverse_vol', 'target_vol', 'random',
                                        'min_vol', 'max_return_min_vol', 'max_sharpe', 'max_diversification',
                                        'efficient_return', 'efficient_risk', 'risk_parity', 'hrp', 'herc'])
    def test_expanding_window_weights(self, method):
        """
        Test expanding_window_weights.
        """
        # expanding window weights
        self.single_opt_instance.method = method
        weights = self.single_opt_instance.compute_expanding_window_weights()

        # dtypes
        assert isinstance(weights, pd.DataFrame)
        assert (weights.dtypes == 'float64').all()
        # shape
        assert weights.shape[1] == self.single_opt_instance.returns.shape[1]
        # values
        assert (weights.round(4) >= 0).all().all()
        if self.single_opt_instance.method != 'target_vol':
            assert (np.isclose(weights.sum(axis=1), 1)).all()
        # cols
        assert set(weights.columns.to_list()) == set(self.single_opt_instance.returns.columns.to_list())
        # index
        if self.single_opt_instance.method in ['equal_weight', 'inverse_variance', 'inverse_vol', 'target_vol',
                                               'random']:
            assert (weights.index ==
                    self.single_opt_instance.returns.index[self.single_opt_instance.window_size - 1:]).all()
        else:
            assert (weights.index ==
                    self.single_opt_instance.returns.dropna().index[self.single_opt_instance.window_size - 1:]).all()

    @pytest.mark.parametrize('method', ['equal_weight', 'inverse_variance', 'inverse_vol', 'target_vol', 'random',
                                        'min_vol', 'max_return_min_vol', 'max_sharpe', 'max_diversification',
                                        'efficient_return', 'efficient_risk', 'risk_parity', 'hrp', 'herc'])
    def test_rolling_window_weights(self, method):
        """
        Test rolling_window_weights.
        """
        # rolling window weights
        self.single_opt_instance.method = method
        weights = self.single_opt_instance.compute_rolling_window_weights()

        # dtypes
        assert isinstance(weights, pd.DataFrame)
        assert (weights.dtypes == 'float64').all()
        # shape
        assert weights.shape[1] == self.single_opt_instance.returns.shape[1]
        # values
        assert (weights.round(4) >= 0).all().all()
        if self.single_opt_instance.method != 'target_vol':
            assert (np.isclose(weights.sum(axis=1), 1)).all()
        # cols
        assert set(weights.columns.to_list()) == set(self.single_opt_instance.returns.columns.to_list())
        # index
        if self.single_opt_instance.method in ['equal_weight', 'inverse_variance', 'inverse_vol', 'target_vol',
                                               'random']:
            assert (weights.index ==
                    self.single_opt_instance.returns.index[self.single_opt_instance.window_size - 1:]).all()
        else:
            assert (weights.index ==
                    self.single_opt_instance.returns.dropna().index[self.single_opt_instance.window_size - 1:]).all()

    @pytest.mark.parametrize('lags', [0, 1, 2, 7])
    def test_get_weights(self, lags):
        """
        Test get_weights.
        """
        # lags
        self.single_opt_instance.lags = lags
        # get weights
        weights = self.single_opt_instance.get_weights()
        w = self.single_opt_instance.compute_rolling_window_weights()

        # dtypes
        assert isinstance(weights, pd.DataFrame)
        assert (weights.dtypes == 'float64').all()
        # shape
        assert weights.shape[1] == self.single_opt_instance.returns.shape[1]
        # values
        assert (weights.round(4) >= 0).all().all()
        if self.single_opt_instance.method != 'target_vol':
            assert (np.isclose(weights.sum(axis=1), 1)).all()
        assert (weights.iloc[-1] == w.iloc[-lags - 1]).all()  # lagged weights
        # cols
        assert set(weights.columns.to_list()) == set(self.single_opt_instance.returns.columns.to_list())
        # index
        assert (weights.index ==
                self.single_opt_instance.returns.index[self.single_opt_instance.window_size - 1 + lags:]).all()

    @pytest.mark.parametrize('rebal_freq', [None, 5, 7, 'monday', 'friday'])
    def test_rebalance_portfolio(self, rebal_freq):
        """
        Test rebalance_portfolio.
        """
        # method
        self.single_opt_instance.method = 'inverse_variance'
        # rebal freq
        self.single_opt_instance.rebal_freq = rebal_freq
        # get weights
        self.single_opt_instance.get_weights()
        # rebalance portfolio
        weights = self.single_opt_instance.rebalance_portfolio()

        # dtypes
        assert isinstance(weights, pd.DataFrame)
        assert (weights.dtypes == 'float64').all()
        # shape
        assert weights.shape[1] == self.single_opt_instance.returns.shape[1]
        # values
        assert (weights.round(4) >= 0).all().all()
        if self.single_opt_instance.method != 'target_vol':
            assert (np.isclose(weights.sum(axis=1), 1)).all()
        if isinstance(rebal_freq, int):
            assert np.allclose((weights.diff().dropna().abs() == 0).sum() /
                               weights.shape[0], 1 - 1 / rebal_freq, rtol=0.05)
        elif rebal_freq is None:
            assert (weights.diff().abs().dropna() >= 0).all().all()
        else:
            assert np.allclose((weights.diff().dropna() == 0).sum() / weights.shape[0], 1 - 1 / 7, rtol=0.05)
        # cols
        assert set(weights.columns.to_list()) == set(self.single_opt_instance.returns.columns.to_list())

    @pytest.mark.parametrize('t_cost', [None, 0.001])
    def test_compute_tcosts(self, t_cost):
        """
        Test compute_tcosts.
        """
        # method
        self.single_opt_instance.method = 'inverse_variance'
        # rebal freq
        self.single_opt_instance.rebal_freq = 7
        # tcost
        self.single_opt_instance.t_cost = t_cost
        # get weights
        self.single_opt_instance.get_weights()
        # rebalance portfolio
        self.single_opt_instance.rebalance_portfolio()
        # compute transaction costs
        tcosts = self.single_opt_instance.compute_tcosts()

        # dtypes
        assert isinstance(tcosts, pd.DataFrame)
        assert (tcosts.dtypes == 'float64').all()
        # shape
        assert tcosts.shape[1] == self.single_opt_instance.returns.shape[1]
        # values
        if t_cost is None:
            assert (tcosts == 0).all().all()
        else:
            assert (tcosts.sum(axis=1) < t_cost).all()
            assert np.allclose(np.sign(tcosts).sum() / tcosts.shape[0], 1 / self.single_opt_instance.rebal_freq,
                               rtol=0.1)
        # cols
        assert set(tcosts.columns.to_list()) == set(self.single_opt_instance.returns.columns.to_list())
        # index
        assert (tcosts.index == self.single_opt_instance.returns.index[self.single_opt_instance.window_size:]).all()

    @pytest.mark.parametrize('method', ['equal_weight', 'inverse_variance', 'inverse_vol', 'target_vol', 'random',
                                        'min_vol', 'max_return_min_vol', 'max_sharpe', 'max_diversification',
                                        'efficient_return', 'efficient_risk', 'risk_parity', 'hrp', 'herc'])
    def test_compute_portfolio_returns(self, method):
        """
        Test compute_portfolio_returns.
        """
        # method
        self.single_opt_instance.method = method
        self.multi_opt_instance.method = method
        # rebal freq
        self.single_opt_instance.rebal_freq = 7
        self.multi_opt_instance.rebal_freq = 7
        # t-costs
        self.single_opt_instance.t_cost = 0.001
        self.multi_opt_instance.t_cost = 0.001
        # compute portfolio returns
        port_ret = self.single_opt_instance.compute_portfolio_returns()
        port_ret_multi = self.multi_opt_instance.compute_portfolio_returns()

        # dtypes
        assert isinstance(port_ret, pd.DataFrame)
        assert isinstance(port_ret_multi, pd.DataFrame)
        assert (port_ret.dtypes == 'float64').all()
        assert (port_ret_multi.dtypes == 'float64').all()
        # shape
        assert port_ret.shape[1] == 1
        assert port_ret_multi.shape[1] == 2
        # index
        assert isinstance(port_ret.index, pd.DatetimeIndex)
        assert isinstance(port_ret_multi.index, pd.DatetimeIndex)
        # cols
        assert port_ret.columns.to_list() == ['portfolio']
        assert port_ret_multi.columns.to_list() == ['price_mom_20', 'price_mom_30']
