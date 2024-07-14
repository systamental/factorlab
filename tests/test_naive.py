import pytest
import pandas as pd
import numpy as np

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


class TestNaiveOptimization:
    """
    Test class for naive portoflio optimization.
    """
    @pytest.fixture(autouse=True)
    def naive_opt(self, asset_returns):
        self.naive_instance = NaiveOptimization(asset_returns)

    def test_initialization(self) -> None:
        """
        Test initialization.
        """
        # data type
        assert isinstance(self.naive_instance, NaiveOptimization)
        assert isinstance(self.naive_instance.returns, pd.DataFrame)
        assert isinstance(self.naive_instance.method, str)
        assert isinstance(self.naive_instance.asset_names, list)
        assert isinstance(self.naive_instance.leverage, float)
        assert isinstance(self.naive_instance.target_vol, float)
        # data shape
        assert self.naive_instance.returns.shape[1] == self.naive_instance.n_assets
        assert self.naive_instance.returns.shape[0] == self.naive_instance.returns.shape[0]
        # cols
        assert self.naive_instance.returns.columns.to_list() == self.naive_instance.asset_names

    def test_compute_estimators(self) -> None:
        """
        Test compute estimators.
        """
        # compute estimators
        self.naive_instance.compute_estimators()

        # dtypes
        assert isinstance(self.naive_instance.exp_ret, np.ndarray)
        assert self.naive_instance.exp_ret.dtype == np.float64
        assert isinstance(self.naive_instance.cov_matrix, np.ndarray)
        assert self.naive_instance.cov_matrix.dtype == np.float64

    def test_compute_equal_weight(self) -> None:
        """
        Test equal weight computation.
        """
        # compute equal weights
        self.naive_instance.compute_equal_weight()

        # shape
        assert self.naive_instance.weights.shape[0] == self.naive_instance.n_assets
        # values
        assert np.allclose(self.naive_instance.weights, 1 / self.naive_instance.n_assets)
        assert (self.naive_instance.weights >= 0).all()
        assert np.isclose(np.sum(self.naive_instance.weights), 1)
        # dtypes
        assert self.naive_instance.weights.dtype == np.float64
        assert isinstance(self.naive_instance.weights, np.ndarray)

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

    @pytest.mark.parametrize("method", ['equal_weight', 'inverse_variance', 'inverse_vol', 'target_vol',
                                        'random'])
    def test_compute_weights(self, method) -> None:
        """
        Test compute weights.
        """
        # set method
        self.naive_instance.method = method
        # drop nans
        self.naive_instance.returns = self.naive_instance.returns.dropna()
        # compute weights
        self.naive_instance.compute_weights()

        # dtypes
        assert (self.naive_instance.weights.dtypes == np.float64).all()
        assert isinstance(self.naive_instance.weights, pd.DataFrame)
        # shape
        assert self.naive_instance.weights.shape[1] == self.naive_instance.n_assets
        # values
        assert (self.naive_instance.weights >= 0).all().all()
        if method in ['equal_weight', 'inverse_variance', 'inverse_vol', 'random']:
            assert np.isclose(self.naive_instance.weights.sum(axis=1), 1)
        assert np.allclose(self.naive_instance.portfolio_ret,
                           (self.naive_instance.weights.values * self.naive_instance.returns).sum(axis=1).mean())
        assert np.allclose(np.sqrt(self.naive_instance.portfolio_risk),
                           (self.naive_instance.weights.values * self.naive_instance.returns).sum(axis=1).std())
