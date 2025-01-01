import pytest
import pandas as pd
import numpy as np

from factorlab.feature_engineering.factors.trend import Trend
from factorlab.signal_generation.signal import Signal
from factorlab.feature_engineering.transformations import Transform
from factorlab.strategy_backtesting.portfolio_optimization.naive import NaiveOptimization


@pytest.fixture
def asset_returns():
    """
    Fixture for crypto spot prices.
    """
    # read csv from datasets/data
    df = pd.read_csv("../src/factorlab/datasets/data/asset_excess_returns_daily.csv", index_col=0)
    df.index = pd.to_datetime(df.index)
    df.index.name = 'date'
    df.columns.name = 'ticker'

    # drop tickers with nobs < ts_obs
    obs = df.count()
    drop_tickers_list = obs[obs < 260].index.to_list()
    df = df.drop(columns=drop_tickers_list)

    # stack
    df = df.stack(future_stack=True).to_frame('ret')
    # replace no chg
    df = df.replace(0, np.nan)
    # start date
    df = df.dropna().unstack().ret

    return df


@pytest.fixture
def spot_prices():
    """
    Fixture for crypto spot prices.
    """
    # read csv from datasets/data
    df = pd.read_csv("../src/factorlab/datasets/data/binance_spot_prices.csv",
                     index_col=['date', 'ticker'],
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
    spot_ret = Transform(spot_prices.close).returns()

    return spot_ret


@pytest.fixture
def trend_signals(spot_prices):
    """
    Fixture for crypto price momentum.
    """
    # compute price mom
    price_mom = Trend(spot_prices, window_size=10).price_mom()
    # signals
    signals = Signal(price_mom).compute_signals().unstack()

    return signals


@pytest.fixture
def trend_signal_rets(spot_prices, spot_ret):
    """
    Fixture for crypto price momentum signal returns.
    """
    # compute price mom
    price_mom = Trend(spot_prices, window_size=10).price_mom()
    # signals
    signal_rets = Signal(price_mom, returns=spot_ret).compute_signal_returns()
    # single index
    signal_rets = signal_rets.unstack()

    return signal_rets


class TestNaiveOptimization:
    """
    Test class for naive portoflio optimization.
    """
    @pytest.fixture(autouse=True)
    def naive_opt(self, asset_returns):
        self.naive_instance = NaiveOptimization(asset_returns)

    @pytest.fixture(autouse=True)
    def signal_opt(self, trend_signal_rets, trend_signals):
        self.signal_instance = NaiveOptimization(trend_signal_rets, signals=trend_signals)

    def test_initialization(self) -> None:
        """
        Test initialization.
        """
        # data types
        assert isinstance(self.naive_instance, NaiveOptimization)
        assert isinstance(self.signal_instance, NaiveOptimization)
        assert isinstance(self.naive_instance.returns, pd.DataFrame)
        assert isinstance(self.signal_instance.returns, pd.DataFrame)
        assert isinstance(self.signal_instance.signals, pd.DataFrame)
        assert (self.naive_instance.returns.dtypes == np.float64).all()
        assert (self.signal_instance.returns.dtypes == np.float64).all()
        assert (self.signal_instance.signals.dtypes == np.float64).all()
        assert isinstance(self.naive_instance.method, str)
        assert isinstance(self.signal_instance.method, str)
        assert isinstance(self.naive_instance.asset_names, list)
        assert isinstance(self.signal_instance.asset_names, list)
        assert isinstance(self.naive_instance.leverage, float)
        assert isinstance(self.signal_instance.leverage, float)
        assert isinstance(self.naive_instance.target_vol, float)
        assert isinstance(self.signal_instance.target_vol, float)

        # data shape
        assert self.naive_instance.returns.shape[1] == self.naive_instance.n_assets
        assert self.signal_instance.returns.shape[1] == self.signal_instance.n_assets
        assert self.naive_instance.returns.shape[0] == self.naive_instance.returns.shape[0]
        assert self.signal_instance.returns.shape[0] == self.signal_instance.returns.shape[0]

        # cols
        assert self.naive_instance.returns.columns.to_list() == self.naive_instance.asset_names
        assert self.signal_instance.returns.columns.to_list() == self.signal_instance.asset_names

    def test_compute_estimators(self) -> None:
        """
        Test compute estimators.
        """
        # compute estimators
        self.naive_instance.compute_estimators()
        self.signal_instance.compute_estimators()

        # dtypes
        assert isinstance(self.naive_instance.exp_ret, np.ndarray)
        assert isinstance(self.signal_instance.exp_ret, np.ndarray)
        assert self.naive_instance.exp_ret.dtype == np.float64
        assert self.signal_instance.exp_ret.dtype == np.float64
        assert isinstance(self.naive_instance.cov_matrix, np.ndarray)
        assert isinstance(self.signal_instance.cov_matrix, np.ndarray)
        assert self.naive_instance.cov_matrix.dtype == np.float64
        assert self.signal_instance.cov_matrix.dtype == np.float64

    def test_compute_equal_weight(self) -> None:
        """
        Test equal weight computation.
        """
        # compute equal weights
        self.naive_instance.compute_equal_weight()

        # shape
        assert self.naive_instance.weights.shape[1] == self.naive_instance.n_assets

        # values
        assert (self.naive_instance.weights.dropna() >= 0).all().all()
        assert np.isclose(self.naive_instance.weights.sum(axis=1), 1).all()

        # dtypes
        assert (self.naive_instance.weights.dtypes == np.float64).all()
        assert isinstance(self.naive_instance.weights, pd.DataFrame)

    def test_compute_signal_weight(self) -> None:
        """
        Test signal weight computation.
        """
        # compute signal weights
        self.signal_instance.compute_signal_weight()

        # shape
        assert self.signal_instance.weights.shape[1] == self.signal_instance.n_assets

        # values
        assert (self.signal_instance.weights.abs().dropna() >= 0).all().all()
        assert (self.signal_instance.weights.abs().dropna() <= 1).all().all()
        assert np.isclose(self.signal_instance.weights.dropna().abs().sum(axis=1), 1).all()

        # dtypes
        assert (self.signal_instance.weights.dtypes == np.float64).all()
        assert isinstance(self.signal_instance.weights, pd.DataFrame)

    def test_compute_inverse_variance(self) -> None:
        """
        Test inverse variance computation.
        """
        # compute inverse variance
        self.naive_instance.compute_inverse_variance()

        # shape
        assert self.naive_instance.weights.shape[0] == self.naive_instance.n_assets

        # values
        assert (self.naive_instance.weights >= 0).all()
        assert np.isclose(np.sum(self.naive_instance.weights), 1)

        # dtypes
        assert self.naive_instance.weights.dtype == np.float64
        assert isinstance(self.naive_instance.weights, np.ndarray)

    def test_compute_inverse_vol(self) -> None:
        """
        Test inverse volatility computation.
        """
        # compute inverse volatility
        self.naive_instance.compute_inverse_vol()

        # shape
        assert self.naive_instance.weights.shape[0] == self.naive_instance.n_assets
        # values
        assert (self.naive_instance.weights >= 0).all()
        assert np.isclose(np.sum(self.naive_instance.weights), 1)
        # dtypes
        assert self.naive_instance.weights.dtype == np.float64
        assert isinstance(self.naive_instance.weights, np.ndarray)

    def test_compute_target_vol(self) -> None:
        """
        Test target volatility computation.
        """
        # compute target volatility
        self.naive_instance.compute_target_vol()

        # shape
        assert self.naive_instance.weights.shape[0] == self.naive_instance.n_assets

        # values
        assert (self.naive_instance.weights >= 0).all()

        # dtypes
        assert self.naive_instance.weights.dtype == np.float64
        assert isinstance(self.naive_instance.weights, np.ndarray)

    def test_compute_random(self) -> None:
        """
        Test random weights.
        """
        # random weights
        self.naive_instance.compute_random()

        # shape
        assert self.naive_instance.weights.shape[0] == self.naive_instance.n_assets

        # values
        assert (self.naive_instance.weights >= 0).all()
        assert np.isclose(np.sum(self.naive_instance.weights), 1)

        # dtypes
        assert self.naive_instance.weights.dtype == np.float64
        assert isinstance(self.naive_instance.weights, np.ndarray)

    @pytest.mark.parametrize("method", ['equal_weight', 'signal_weight', 'inverse_variance',
                                        'inverse_vol', 'target_vol', 'random'])
    def test_compute_weights(self, method) -> None:
        """
        Test compute weights.
        """
        # set method
        if method != 'signal_weight':
            self.naive_instance.method = method
            # compute weights
            self.naive_instance.compute_weights()

            # dtypes
            assert (self.naive_instance.weights.dtypes == np.float64).all()
            assert isinstance(self.naive_instance.weights, pd.DataFrame)

            # shape
            assert self.naive_instance.weights.shape[1] == self.naive_instance.n_assets

            # values
            assert (self.naive_instance.weights.dropna() >= 0).all().all()
            if method == 'equal_weight':
                assert np.isclose(self.naive_instance.weights.sum(axis=1), 1).all()
            elif method in ['inverse_variance', 'inverse_vol','random']:
                assert np.isclose(self.naive_instance.weights.sum(axis=1), 1)
            assert np.allclose(self.naive_instance.portfolio_ret,
                               (self.naive_instance.weights.values * self.naive_instance.returns).sum(axis=1).mean(),
                               atol=1e-3)
            assert np.allclose(np.sqrt(self.naive_instance.portfolio_risk),
                               (self.naive_instance.weights.values * self.naive_instance.returns).sum(axis=1).std(),
                               atol=1e-3)

        else:
            self.signal_instance.method = method
            self.signal_instance.compute_weights()

            # dtypes
            assert (self.signal_instance.weights.dtypes == np.float64).all()
            assert isinstance(self.signal_instance.weights, pd.DataFrame)

            # shape
            assert self.signal_instance.weights.shape[1] == self.signal_instance.n_assets

            # values
            assert (self.signal_instance.weights.abs().dropna() >= 0).all().all()
            assert (self.signal_instance.weights.abs().dropna() <= 1).all().all()
            assert np.isclose(self.signal_instance.weights.abs().dropna().sum(axis=1), 1).all()
            assert np.allclose(self.signal_instance.portfolio_ret,
                               (self.signal_instance.weights.iloc[-1] * self.signal_instance.exp_ret).sum(),
                               atol=1e-3)
            assert np.allclose(np.sqrt(self.signal_instance.portfolio_risk),
                               (self.signal_instance.weights.iloc[-1] *
                                self.signal_instance.returns).sum(axis=1).std(),
                               atol=1e-3)


