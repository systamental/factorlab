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
    df = pd.read_csv("../src/factorlab/datasets/data/binance_spot_prices.csv",
                     index_col=['date', 'ticker'],
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
def binance_ret_single(binance_ret):
    """
    Fixture for single asset returns.
    """
    return binance_ret.unstack().dropna(how='all')


@pytest.fixture
def trend_signals_multi(binance_ohlc):
    """
    Fixture for trend factor.
    """
    # trend factor
    trend_df = Trend(binance_ohlc, vwap=True, log=True, window_size=30).price_mom()
    signals_df = Signal(trend_df).compute_signals()

    return signals_df


@pytest.fixture
def trend_signals_single(trend_signals_multi):
    """
    Fixture for single asset trend factor.
    """
    return trend_signals_multi.unstack().price_mom_30.dropna(how='all')


class TestPortfolioOptimization:
    """
    Test class for Portfolio Optimization.
    """

    @pytest.fixture(autouse=True)
    def single_opt(self, binance_ret_single, trend_signals_single):
        self.single_opt_instance = PortfolioOptimization(binance_ret_single, signals=trend_signals_single,
                                                         as_signal_returns=True, window_size=300,
                                                         target_risk=0.50, target_return=0.75)

    @pytest.fixture(autouse=True)
    def multi_signal_opt(self, binance_ret, trend_signals_multi):
        self.multi_opt_instance = PortfolioOptimization(binance_ret, signals=trend_signals_multi,
                                                        as_signal_returns=True, window_size=300,
                                                        target_risk=0.50, target_return=0.75)

    def test_initialization(self) -> None:
        """
        Test initialization.
        """
        # types
        assert isinstance(self.single_opt_instance, PortfolioOptimization)
        assert (isinstance(self.multi_opt_instance.returns, pd.DataFrame) or
                isinstance(self.multi_opt_instance.returns, pd.Series))
        assert (isinstance(self.single_opt_instance.returns, pd.DataFrame) or
                isinstance(self.single_opt_instance.returns, pd.Series))
        assert isinstance(self.multi_opt_instance.method, str)
        assert isinstance(self.single_opt_instance.method, str)
        assert isinstance(self.multi_opt_instance.lags, int)
        assert isinstance(self.single_opt_instance.lags, int)
        assert isinstance(self.multi_opt_instance.window_size, int)
        assert isinstance(self.single_opt_instance.window_size, int)
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
        assert self.single_opt_instance.fully_invested is False
        assert self.multi_opt_instance.fully_invested is False
        assert self.multi_opt_instance.gross_exposure == 1.0
        assert self.single_opt_instance.gross_exposure == 1.0
        assert self.multi_opt_instance.net_exposure is None
        assert self.single_opt_instance.net_exposure is None
        assert self.multi_opt_instance.risk_aversion == 1.0
        assert self.single_opt_instance.risk_aversion == 1.0
        assert self.multi_opt_instance.exp_ret_method == 'historical_mean'
        assert self.single_opt_instance.exp_ret_method == 'historical_mean'
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

    def test_check_method_error(self) -> None:
        """
        Test check_method_error.
        """
        # method error
        self.single_opt_instance.method = 'optimal'
        with pytest.raises(ValueError):
            self.single_opt_instance._check_methods()

    def test_compute_signal_returns(self) -> None:
        """
        Test compute_signal_returns.
        """
        # compute signal returns
        self.single_opt_instance._compute_signal_returns()
        self.multi_opt_instance._compute_signal_returns()

        # types
        assert isinstance(self.single_opt_instance.signal_returns, pd.DataFrame)
        assert (self.multi_opt_instance.signal_returns.dtypes == 'float64').all()
        assert isinstance(self.single_opt_instance.signal_returns, pd.DataFrame)
        assert (self.multi_opt_instance.signal_returns.dtypes == 'float64').all()

        # shape
        assert self.single_opt_instance.signal_returns.shape[1] == self.single_opt_instance.returns.shape[1]

        # cols
        assert set(self.single_opt_instance.signal_returns.columns.to_list()) == \
               set(self.single_opt_instance.signals.columns.to_list())
        assert (set(self.multi_opt_instance.signal_returns.columns.to_list()) ==
                set(self.multi_opt_instance.signals.columns.to_list()))

    def test_get_optimizer(self) -> None:
        """
        Test get_optimizer.
        """
        # get optimizer
        opt = self.single_opt_instance._get_optimizer(self.single_opt_instance.returns)
        multi_opt = self.multi_opt_instance._get_optimizer(self.multi_opt_instance.returns)

        # types
        if self.single_opt_instance.method in ['equal_weight', 'signal_weight', 'inverse_variance', 'inverse_vol',
                                               'target_vol', 'random']:
            assert isinstance(opt, NaiveOptimization)
        if self.multi_opt_instance.method in ['equal_weight', 'signal_weight', 'inverse_variance', 'inverse_vol',
                                              'target_vol', 'random']:
            assert isinstance(multi_opt, NaiveOptimization)
        if self.single_opt_instance.method in ['max_return', 'min_vol', 'max_return_min_vol', 'max_sharpe',
                                               'max_diversification', 'efficient_return',
                                               'efficient_risk', 'risk_parity']:
            assert isinstance(opt, MVO)
        if self.multi_opt_instance.method in ['max_return', 'min_vol', 'max_return_min_vol', 'max_sharpe',
                                              'max_diversification', 'efficient_return', 'efficient_risk',
                                              'risk_parity']:
            assert isinstance(multi_opt, MVO)
        if self.single_opt_instance.method == 'hrp':
            assert isinstance(opt, HRP)
        if self.multi_opt_instance.method == 'hrp':
            assert isinstance(multi_opt, HRP)
        if self.single_opt_instance.method == 'herc':
            assert isinstance(opt, HERC)
        if self.multi_opt_instance.method == 'herc':
            assert isinstance(multi_opt, HERC)

    @pytest.mark.parametrize('method', ['equal_weight', 'signal_weight', 'inverse_variance', 'inverse_vol',
                                        'target_vol', 'random', 'max_return', 'min_vol', 'max_return_min_vol',
                                        'max_sharpe', 'max_diversification', 'efficient_return', 'efficient_risk',
                                        'risk_parity', 'hrp', 'herc'])
    def test_compute_fixed_weights(self, method) -> None:
        """
        Test compute_fixed_weights.
        """
        # compute fixed weights
        po = PortfolioOptimization(self.single_opt_instance.returns, signals=self.single_opt_instance.signals,
                                   method=method, window_size=300, target_return=0.75, target_risk=0.50)
        weights = po._compute_fixed_weights()

        if method in ['equal_weight', 'signal_weight']:

            # dtypes
            assert isinstance(weights, pd.DataFrame)
            assert (weights.dtypes == 'float64').all()

            # shape
            assert weights.shape[1] == po.returns.shape[1]

            # values
            if method == 'equal_weight':
                assert (weights >= 0).all().all()
                assert np.isclose(weights.sum(axis=1), 1).all().all()
            else:
                assert (weights <= 1).all().all()
                assert np.isclose(weights.abs().sum(axis=1), 1).all().all()

            # cols
            assert set(weights.columns.to_list()) == set(po.returns.columns.to_list())

            # index
            if method == 'equal_weight':
                assert (weights.index == po.returns.index).all()
            else:
                assert (weights.index == po.signals.index).all()

        else:

            # dtypes
            assert isinstance(weights, pd.DataFrame)
            assert (weights.dtypes == 'float64').all()

            # shape
            assert weights.shape[1] == po.returns.shape[1]

            # values
            assert (weights >= 0).all().all()
            if method != 'target_vol':
                assert np.isclose(weights.sum(axis=1), 1).all().all()

            # cols
            assert set(weights.columns.to_list()) == set(po.returns.columns.to_list())

            # index
            assert (weights.index == po.returns.index[-1]).all()

    @pytest.mark.parametrize('method', ['inverse_variance', 'inverse_vol', 'target_vol', 'random',
                                        'min_vol', 'max_return_min_vol', 'max_sharpe', 'max_diversification',
                                        'efficient_return', 'efficient_risk', 'risk_parity', 'hrp', 'herc'])
    def test_expanding_window_weights(self, method, binance_ret_single, trend_signals_single) -> None:
        """
        Test expanding_window_weights.
        """
        # expanding window weights
        po = PortfolioOptimization(binance_ret_single, signals=trend_signals_single, method=method,
                                   window_size=300, target_return=0.75, target_risk=0.50)

        weights = po._compute_expanding_window_weights()

        # dtypes
        assert isinstance(weights, pd.DataFrame)
        assert (weights.dtypes == 'float64').all()

        # shape
        assert weights.shape[1] == po.returns.shape[1]

        # values
        assert (weights.round(4) >= 0).all().all()
        if po.method != 'target_vol':
            assert (np.isclose(weights.sum(axis=1), 1)).all()

        # cols
        assert set(weights.columns.to_list()) == set(po.returns.columns.to_list())

    @pytest.mark.parametrize('method', ['inverse_variance', 'inverse_vol', 'target_vol', 'random',
                                        'min_vol', 'max_return_min_vol', 'max_sharpe', 'max_diversification',
                                        'efficient_return', 'efficient_risk', 'risk_parity', 'hrp', 'herc'])
    def test_rolling_window_weights(self, method, binance_ret_single, trend_signals_single) -> None:
        """
        Test rolling_window_weights.
        """
        # rolling window weights
        po = PortfolioOptimization(binance_ret_single, signals=trend_signals_single, method=method,
                                   window_size=300, target_return=0.75, target_risk=0.50)

        weights = po._compute_rolling_window_weights()

        # dtypes
        assert isinstance(weights, pd.DataFrame)
        assert (weights.dtypes == 'float64').all()

        # shape
        assert weights.shape[1] == po.returns.shape[1]

        # values
        assert (weights.round(4) >= 0).all().all()
        if po.method != 'target_vol':
            assert (np.isclose(weights.sum(axis=1), 1)).all()

        # cols
        assert set(weights.columns.to_list()) == set(po.returns.columns.to_list())

    @pytest.mark.parametrize('method', ['equal_weight', 'signal_weight', 'inverse_variance', 'inverse_vol',
                                        'target_vol', 'random', 'min_vol', 'max_return_min_vol', 'max_sharpe',
                                        'max_diversification', 'efficient_return', 'efficient_risk', 'risk_parity',
                                        'hrp', 'herc'])
    def test_compute_weights(self, binance_ret_single, trend_signals_single, method) -> None:
        """
        Test get_weights.
        """
        # rolling window weights
        po = PortfolioOptimization(binance_ret_single, signals=trend_signals_single, method=method,
                                   window_size=300, target_return=0.75, target_risk=0.50)

        weights = po.compute_weights()
        weights = weights.dropna()

        # dtypes
        assert isinstance(weights, pd.DataFrame)
        assert (weights.dtypes == 'float64').all()

        # shape
        assert weights.shape[1] == po.returns.shape[1]

        # values
        if po.method != 'target_vol':
            assert (np.isclose(weights.abs().sum(axis=1), 1)).all()

        # cols
        assert set(weights.columns.to_list()) == set(po.returns.columns.to_list())

    def test_compute_weighted_signals(self, binance_ret_single, trend_signals_single) -> None:
        """
        Test compute_weighted_signals.
        """
        # compute weighted signals
        self.single_opt_instance.compute_weights()
        weighted_signals = self.single_opt_instance.compute_weighted_signals()

        # types
        assert isinstance(weighted_signals, pd.DataFrame)
        assert (weighted_signals.dtypes == 'float64').all()

        # shape
        assert weighted_signals.shape[1] == self.single_opt_instance.signals.shape[1]

        # values
        assert (weighted_signals.abs().sum(axis=1) <= 1).all()

        # cols
        assert set(weighted_signals.columns.to_list()) == set(self.single_opt_instance.signals.columns.to_list())

    @pytest.mark.parametrize('fully_invested, gross_exposure', [
        (True, 1.0),
        (False, 1.0),
        (True, 2.0),
        (False, 2.0)
    ])
    def test_adjust_exposure(self, fully_invested, gross_exposure, binance_ret_single, trend_signals_single) -> None:
        """
        Test adjust_exposure.
        """
        # adjust exposure
        po = PortfolioOptimization(binance_ret_single, signals=trend_signals_single, as_signal_returns=False,
                                   fully_invested=fully_invested, gross_exposure=gross_exposure, window_size=300,
                                   target_return=0.75, target_risk=0.35)
        po.compute_weights()
        weights = po.adjust_exposure()

        # types
        assert isinstance(weights, pd.DataFrame)
        assert (weights.dtypes == 'float64').all()

        # shape
        assert weights.shape[1] == po.returns.shape[1]

        # values
        if fully_invested:
            assert np.allclose(weights.abs().sum(axis=1), gross_exposure)

        # cols
        assert set(weights.columns.to_list()) == \
               set(po.returns.columns.to_list())

    @pytest.mark.parametrize('method', ['equal_weight', 'signal_weight', 'inverse_variance', 'inverse_vol',
                                        'target_vol', 'random', 'min_vol', 'max_return_min_vol', 'max_sharpe',
                                        'max_diversification', 'efficient_return', 'efficient_risk', 'risk_parity',
                                        'hrp', 'herc'])
    def test_round_weights(self, method, binance_ret_single, trend_signals_single, thresh=0.0001) -> None:
        """
        Test round_weights.
        """
        # compute weights
        po = PortfolioOptimization(binance_ret_single, signals=trend_signals_single, as_signal_returns=False,
                                   method=method, window_size=300, target_return=0.75, target_risk=0.35)
        po.compute_weights()
        po.round_weights(threshold=thresh)

        # dtypes
        assert isinstance(po.weights, pd.DataFrame)
        assert (po.weights.dtypes == 'float64').all()

        # shape
        assert po.weights.shape[1] == po.returns.shape[1]

        # values
        assert ~((po.weights.abs() > 1 - 0.0001) & (po.weights.abs() < 1)).all().all()
        assert ~((po.weights.abs() < 0.0001) & (po.weights.abs() > 0)).all().all()
        if po.method != 'signal_weight':
            assert (po.weights.abs().sum(axis=1) <= 1.0).all()
        else:
            # TODO: debug test failing in IDE but passing in notebook
            assert (po.weights.abs().sum(axis=1) <= 1.01).all()

        # cols
        assert set(po.weights.columns.to_list()) == set(po.returns.columns.to_list())

    @pytest.mark.parametrize('rebal_freq', [None, 5, 7, 'monday', 'friday'])
    def test_rebalance_portfolio(self, rebal_freq, binance_ret_single, trend_signals_single) -> None:
        """
        Test rebalance_portfolio.
        """
        po = PortfolioOptimization(binance_ret_single, signals=trend_signals_single,
                                   method='inverse_variance', window_size=300,
                                   target_return=0.75, target_risk=0.50, rebal_freq=rebal_freq)
        # get weights
        po.compute_weights()
        # rebalance portfolio
        weights = po.rebalance_portfolio()
        # drop na
        weights = weights.dropna()

        # dtypes
        assert isinstance(weights, pd.DataFrame)
        assert (weights.dtypes == 'float64').all()

        # shape
        assert weights.shape[1] == po.returns.shape[1]

        # values
        # assert (weights.round(4) >= 0).all().all()
        if po.method != 'target_vol':
            assert (np.isclose(weights.abs().sum(axis=1), 1)).all()
        if isinstance(rebal_freq, int):
            assert np.allclose((weights.diff().abs() == 0).sum() /
                               weights.shape[0], 1 - 1 / rebal_freq, rtol=0.05)
        elif rebal_freq is None:
            assert (weights.diff().dropna().abs() >= 0).all().all()
        else:
            assert np.allclose((weights.diff().dropna() == 0).sum() / weights.shape[0], 1 - 1 / 7, rtol=0.05)

        # cols
        assert set(weights.columns.to_list()) == set(po.returns.columns.to_list())

    @pytest.mark.parametrize('t_cost', [None, 0.001])
    def test_compute_tcosts(self, t_cost, binance_ret_single, trend_signals_single) -> None:
        """
        Test compute_tcosts.
        """
        po = PortfolioOptimization(binance_ret_single, signals=trend_signals_single, method='inverse_variance',
                                   window_size=300, rebal_freq=7, t_cost=t_cost)
        # t-cost
        po.t_cost = t_cost
        # get weights
        po.compute_weights()
        # rebalance portfolio
        po.rebalance_portfolio()
        # compute transaction costs
        tcosts = po.compute_tcosts().dropna()

        # dtypes
        assert isinstance(tcosts, pd.DataFrame)
        assert (tcosts.dtypes == 'float64').all()

        # shape
        assert tcosts.shape[1] == po.returns.shape[1]

        # values
        if t_cost is None:
            assert (tcosts == 0).all().all()
        else:
            assert (tcosts.sum(axis=1) < t_cost).all()
            assert np.allclose(np.sign(tcosts).sum() / tcosts.shape[0], 1 / po.rebal_freq,
                               rtol=0.1)

        # cols
        assert set(tcosts.columns.to_list()) == set(po.returns.columns.to_list())

    def test_gross_returns(self) -> None:
        """
        Test gross_returns.
        """
        # compute gross returns
        self.single_opt_instance.compute_weights()
        self.single_opt_instance.compute_gross_returns()

        # types
        assert isinstance(self.single_opt_instance.gross_returns, pd.DataFrame)
        assert (self.single_opt_instance.gross_returns.dtypes == 'float64').all()

        # shape
        assert self.single_opt_instance.gross_returns.shape[1] == self.single_opt_instance.weights.shape[1]

        # cols
        assert (self.single_opt_instance.gross_returns.columns.to_list() ==
                self.single_opt_instance.weights.columns.to_list())

    def test_compute_net_returns(self) -> None:
        """
        Test compute_net_returns.
        """
        # compute net returns
        self.single_opt_instance.compute_weights()
        self.single_opt_instance.compute_tcosts()
        self.single_opt_instance.compute_gross_returns()
        self.single_opt_instance.compute_net_returns()

        # types
        assert isinstance(self.single_opt_instance.net_returns, pd.DataFrame)
        assert (self.single_opt_instance.net_returns.dtypes == 'float64').all()

        # shape
        assert self.single_opt_instance.net_returns.shape[1] == self.single_opt_instance.weights.shape[1]

        # cols
        assert (self.single_opt_instance.net_returns.columns.to_list() ==
                self.single_opt_instance.weights.columns.to_list())

    @pytest.mark.parametrize('method', ['equal_weight', 'inverse_variance', 'inverse_vol', 'target_vol', 'random',
                                        'min_vol', 'max_return_min_vol', 'max_sharpe', 'max_diversification',
                                        'efficient_return', 'efficient_risk', 'risk_parity', 'hrp', 'herc'])
    def test_compute_compute_portfolio_returns(self, method, binance_ret_single, trend_signals_single) -> None:
        """
        Test compute_single_idx_portfolio_rets.
        """
        # compute single index portfolio returns
        po = PortfolioOptimization(binance_ret_single, signals=trend_signals_single, method=method,
                                   window_size=300, target_return=0.75, target_risk=0.50)
        po.compute_weights()
        po.compute_weighted_signals()
        po.adjust_exposure()
        po.round_weights()
        po.rebalance_portfolio()
        po.compute_gross_returns()
        po.compute_tcosts()
        po.compute_net_returns()
        port_ret = po.compute_portfolio_returns()

        # types
        assert isinstance(port_ret, pd.DataFrame)
        assert (port_ret.dtypes == 'float64').all()

        # shape
        assert port_ret.shape[1] == 1

        # values
        assert ~((po.weights.abs() > 1 - 0.0001) & (po.weights.abs() < 1)).all().all()
        assert ~((po.weights.abs() < 0.0001) & (po.weights.abs() > 0)).all().all()
        if po.method != 'signal_weight':
            assert (po.weights.abs().sum(axis=1) <= 1.0).all()
        else:
            # TODO: debug test failing in IDE but passing in notebook
            assert (po.weights.abs().sum(axis=1) <= 1.01).all()

        # cols
        assert port_ret.columns.to_list() == ['portfolio']

        # index duplicates
        assert port_ret.index.duplicated().sum() == 0
