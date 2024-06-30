import pytest
import pandas as pd
import numpy as np

import cvxpy as cp

from factorlab.strategy_backtesting.portfolio_optimization.mvo import MVO


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
    drop_tickers_list = obs[obs < 2500].index.to_list()
    df = df.drop(columns=drop_tickers_list)

    # stack
    df = df.stack(future_stack=True).to_frame('ret')

    # replace no chg
    df = df.replace(0, np.nan)

    # start date
    df = df.loc['1870-01-01':, :].dropna().unstack().ret

    return df


class TestMVO:
    """
    Test class for MVO, mean variance optimization.
    """
    @pytest.fixture(autouse=True)
    def mvo(self, asset_returns):
        self.mvo_instance = MVO(asset_returns, target_ret=0.1, target_risk=0.05)

    def test_initialization(self) -> None:
        """
        Test initialization.
        """
        # data type
        assert isinstance(self.mvo_instance, MVO)
        assert isinstance(self.mvo_instance.returns, pd.DataFrame)
        assert isinstance(self.mvo_instance.method, str)
        assert isinstance(self.mvo_instance.max_weight, float)
        assert isinstance(self.mvo_instance.min_weight, float)
        assert isinstance(self.mvo_instance.budget, float)
        assert isinstance(self.mvo_instance.risk_free_rate, float)
        assert isinstance(self.mvo_instance.risk_aversion, float)
        assert isinstance(self.mvo_instance.asset_names, list)
        assert isinstance(self.mvo_instance.returns.index, pd.DatetimeIndex)
        assert isinstance(self.mvo_instance.n_assets, int)
        assert isinstance(self.mvo_instance.ann_factor, np.int64)
        assert (self.mvo_instance.returns.dtypes == np.float64).all()
        # data shape
        assert self.mvo_instance.returns.shape[1] == self.mvo_instance.n_assets
        assert self.mvo_instance.returns.shape[0] == self.mvo_instance.returns.dropna().shape[0]
        # cols
        assert self.mvo_instance.returns.columns.to_list() == self.mvo_instance.asset_names

    def test_get_start_weights(self) -> None:
        """
        Test get_start_weights.
        """
        # get start weights
        self.mvo_instance.get_start_weights()

        # shape
        assert self.mvo_instance.weights.value.shape[0] == self.mvo_instance.n_assets
        # values
        assert np.allclose(self.mvo_instance.weights.value.sum(), 1)  # sum of weights is 1
        assert (self.mvo_instance.weights.value == 1 / self.mvo_instance.returns.shape[1]).all()  # equal weights
        # dtypes
        assert isinstance(self.mvo_instance.weights, cp.expressions.variable.Variable)
        assert isinstance(self.mvo_instance.weights.value, np.ndarray)
        assert isinstance(self.mvo_instance.weights.value[0], np.float64)

    @pytest.mark.parametrize("exp_ret, window_size", [('mean', 30), ('exponential', 252)])
    def test_compute_estimators(self, exp_ret, window_size) -> None:
        """
        Test compute_estimators.
        """
        # compute estimators
        self.mvo_instance.exp_ret_method = exp_ret
        self.mvo_instance.compute_estimators(window_size=window_size)

        # shape
        assert self.mvo_instance.exp_ret.shape[0] == self.mvo_instance.n_assets
        assert self.mvo_instance.cov_matrix.shape == (self.mvo_instance.n_assets, self.mvo_instance.n_assets)
        assert self.mvo_instance.corr_matrix.shape == (self.mvo_instance.n_assets, self.mvo_instance.n_assets)
        # values
        if exp_ret == 'mean':
            assert np.allclose(self.mvo_instance.exp_ret, self.mvo_instance.returns.mean().values *
                               self.mvo_instance.ann_factor)
        elif exp_ret == 'exponential':
            assert np.allclose(self.mvo_instance.exp_ret,
                               self.mvo_instance.returns.ewm(span=window_size).mean().iloc[-1].values *
                               self.mvo_instance.ann_factor)
        # dtypes
        assert isinstance(self.mvo_instance.exp_ret, np.ndarray)
        assert self.mvo_instance.exp_ret.dtype == np.float64
        assert isinstance(self.mvo_instance.cov_matrix, np.ndarray)
        assert self.mvo_instance.cov_matrix.dtype == np.float64
        assert isinstance(self.mvo_instance.corr_matrix, np.ndarray)
        assert self.mvo_instance.corr_matrix.dtype == np.float64

    @pytest.mark.parametrize("method", ['min_vol', 'max_return_min_vol', 'max_sharpe', 'max_diversification',
                                        'return_targeting', 'risk_budgeting'])
    def test_objective_function(self, method) -> None:
        """
        Test objective_function.
        """
        # method
        self.mvo_instance.method = method
        # get start weights
        self.mvo_instance.get_start_weights()
        # compute estimators
        self.mvo_instance.compute_estimators()
        # objective function
        self.mvo_instance.objective_function()

        # dtypes
        if method == 'risk_budgeting':
            assert isinstance(self.mvo_instance.objective, cp.problems.objective.Maximize)
        else:
            assert isinstance(self.mvo_instance.objective, cp.problems.objective.Minimize)
        assert isinstance(self.mvo_instance.portfolio_ret, cp.atoms.affine.binary_operators.MulExpression)

    @pytest.mark.parametrize("method", ['min_vol', 'max_return_min_vol', 'max_sharpe', 'max_diversification',
                                        'return_targeting', 'risk_budgeting'])
    def test_get_constraints(self, method) -> None:
        """
        Test get_constraints.
        """
        # method
        self.mvo_instance.method = method
        # get start weights
        self.mvo_instance.get_start_weights()
        # compute estimators
        self.mvo_instance.compute_estimators()
        # objective function
        self.mvo_instance.objective_function()
        # constraints
        self.mvo_instance.get_constraints()

        if self.mvo_instance.method == 'max_sharpe':
            # shape
            assert len(self.mvo_instance.constraints) == 5
            # dtypes
            assert isinstance(self.mvo_instance.constraints[0], cp.constraints.zero.Equality)
            assert isinstance(self.mvo_instance.constraints[1], cp.constraints.zero.Equality)
            assert isinstance(self.mvo_instance.constraints[2], cp.constraints.nonpos.Inequality)
            assert isinstance(self.mvo_instance.constraints[3], cp.constraints.nonpos.Inequality)
            assert isinstance(self.mvo_instance.constraints[4], cp.constraints.nonpos.Inequality)
        elif self.mvo_instance.method == 'return_targeting':
            # shape
            assert len(self.mvo_instance.constraints) == 4
            # dtypes
            assert isinstance(self.mvo_instance.constraints[0], cp.constraints.nonpos.Inequality)
            assert isinstance(self.mvo_instance.constraints[1], cp.constraints.nonpos.Inequality)
            assert isinstance(self.mvo_instance.constraints[2], cp.constraints.zero.Equality)
            assert isinstance(self.mvo_instance.constraints[3], cp.constraints.nonpos.Inequality)
        elif self.mvo_instance.method == 'risk_budgeting':
            # shape
            assert len(self.mvo_instance.constraints) == 4
            # dtypes
            assert isinstance(self.mvo_instance.constraints[0], cp.constraints.nonpos.Inequality)
            assert isinstance(self.mvo_instance.constraints[1], cp.constraints.nonpos.Inequality)
            assert isinstance(self.mvo_instance.constraints[2], cp.constraints.zero.Equality)
            assert isinstance(self.mvo_instance.constraints[3], cp.constraints.nonpos.Inequality)
        else:
            # shape
            assert len(self.mvo_instance.constraints) == 3
            # dtypes
            assert isinstance(self.mvo_instance.constraints[0], cp.constraints.nonpos.Inequality)
            assert isinstance(self.mvo_instance.constraints[1], cp.constraints.nonpos.Inequality)
            assert isinstance(self.mvo_instance.constraints[2], cp.constraints.zero.Equality)

    @pytest.mark.parametrize("method", ['min_vol', 'max_return_min_vol', 'max_sharpe', 'max_diversification',
                                        'return_targeting', 'risk_budgeting'])
    def test_optimize(self, method) -> None:
        """
        Test optimize.
        """
        # method
        self.mvo_instance.method = method
        # get start weights
        self.mvo_instance.get_start_weights()
        # compute estimators
        self.mvo_instance.compute_estimators()
        # objective function
        self.mvo_instance.objective_function()
        # constraints
        self.mvo_instance.get_constraints()
        # optimize
        self.mvo_instance.optimize()

        # shape
        assert self.mvo_instance.weights.shape[0] == self.mvo_instance.n_assets
        # values
        assert np.allclose(self.mvo_instance.weights.sum(), 1)
        # dtypes
        assert isinstance(self.mvo_instance.weights, np.ndarray)
        assert self.mvo_instance.weights.dtype == np.float64

    def test_check_weights(self) -> None:
        """
        Test check_weights.
        """
        # method
        self.mvo_instance.method = 'risk_budgeting'
        # get start weights
        self.mvo_instance.get_start_weights()
        # compute estimators
        self.mvo_instance.compute_estimators()
        # objective function
        self.mvo_instance.objective_function()
        # constraints
        self.mvo_instance.get_constraints()
        # optimize
        self.mvo_instance.optimize()
        # check weights
        self.mvo_instance.check_weights()

        # shape
        assert self.mvo_instance.weights.shape[0] == self.mvo_instance.n_assets
        # values
        assert np.allclose(self.mvo_instance.weights.sum(), 1)
        # dtypes
        assert isinstance(self.mvo_instance.weights, np.ndarray)
        assert self.mvo_instance.weights.dtype == np.float64
        assert (self.mvo_instance.weights == 1).sum() == 1

    @pytest.mark.parametrize("method", ['min_vol', 'max_return_min_vol', 'max_sharpe', 'max_diversification',
                                        'return_targeting', 'risk_budgeting'])
    def test_get_optimal_weights(self, method) -> None:
        """
        Test get_optimal_weights.
        """
        # method
        self.mvo_instance.method = method
        # get optimal weights
        self.mvo_instance.get_optimal_weights()

        # shape
        assert self.mvo_instance.weights.shape[1] == self.mvo_instance.n_assets
        # values
        assert np.allclose(self.mvo_instance.weights.sum(axis=1), 1)
        assert (self.mvo_instance.weights >= 0).all().all()
        # dtypes
        assert isinstance(self.mvo_instance.weights, pd.DataFrame)
        assert (self.mvo_instance.weights.dtypes == np.float64).all().all()



