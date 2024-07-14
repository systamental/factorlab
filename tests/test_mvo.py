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
    drop_tickers_list = obs[obs < 260].index.to_list()
    df = df.drop(columns=drop_tickers_list)
    
    # stack
    df = df.stack(future_stack=True).to_frame('ret')
    # replace no chg
    df = df.replace(0, np.nan)
    # start date
    df = df.dropna().unstack().ret
    
    return df


class TestMVO:
    """
    Test class for MVO, mean variance optimization.
    """
    
    @pytest.fixture(autouse=True)
    def mvo(self, asset_returns):
        self.mvo_instance = MVO(asset_returns, target_return=0.1, target_risk=0.05)
    
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
    
    def test_get_initial_weights(self) -> None:
        """
        Test get_start_weights.
        """
        # get start weights
        self.mvo_instance.get_initial_weights()
        
        # shape
        assert self.mvo_instance.weights.value.shape[0] == self.mvo_instance.n_assets
        # values
        assert np.allclose(self.mvo_instance.weights.value.sum(), 1)  # sum of weights is 1
        assert (self.mvo_instance.weights.value == 1 / self.mvo_instance.returns.shape[1]).all()  # equal weights
        # dtypes
        assert isinstance(self.mvo_instance.weights, cp.expressions.variable.Variable)
        assert isinstance(self.mvo_instance.weights.value, np.ndarray)
        assert isinstance(self.mvo_instance.weights.value[0], np.float64)
    
    @pytest.mark.parametrize("exp_ret, window_size", [('mean', 30), ('ewma', 252)])
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
            assert np.allclose(self.mvo_instance.exp_ret, self.mvo_instance.returns.mean().values)
        elif exp_ret == 'ewma':
            assert np.allclose(self.mvo_instance.exp_ret,
                               self.mvo_instance.returns.ewm(span=window_size).mean().iloc[-1].values)
        # dtypes
        assert isinstance(self.mvo_instance.exp_ret, np.ndarray)
        assert self.mvo_instance.exp_ret.dtype == np.float64
        assert isinstance(self.mvo_instance.cov_matrix, np.ndarray)
        assert self.mvo_instance.cov_matrix.dtype == np.float64
        assert isinstance(self.mvo_instance.corr_matrix, np.ndarray)
        assert self.mvo_instance.corr_matrix.dtype == np.float64

    # TODO: add max return
    @pytest.mark.parametrize("method", ['max_return', 'min_vol', 'max_return_min_vol', 'max_sharpe',
                                        'max_diversification', 'efficient_return', 'efficient_risk', 'risk_parity'])
    def test_objective_function(self, method) -> None:
        """
        Test objective_function.
        """
        # method
        self.mvo_instance.method = method
        # get start weights
        self.mvo_instance.get_initial_weights()
        # compute estimators
        self.mvo_instance.compute_estimators()
        # objective function
        self.mvo_instance.objective_function()
        
        # dtypes
        if method == 'efficient_risk' or method == 'max_return':
            assert isinstance(self.mvo_instance.objective, cp.problems.objective.Maximize)
            assert isinstance(self.mvo_instance.portfolio_ret, cp.atoms.affine.binary_operators.MulExpression)
        elif method == 'risk_parity':
            assert callable(self.mvo_instance.objective)
            assert isinstance(self.mvo_instance.portfolio_ret, np.float64)
        else:
            assert isinstance(self.mvo_instance.objective, cp.problems.objective.Minimize)
            assert isinstance(self.mvo_instance.portfolio_ret, cp.atoms.affine.binary_operators.MulExpression)
    
    @pytest.mark.parametrize("method", ['max_return', 'min_vol', 'max_return_min_vol', 'max_sharpe',
                                        'max_diversification', 'efficient_return', 'efficient_risk', 'risk_parity'])
    def test_get_constraints(self, method) -> None:
        """
        Test get_constraints.
        """
        # method
        self.mvo_instance.method = method
        # get start weights
        self.mvo_instance.get_initial_weights()
        # compute estimators
        self.mvo_instance.compute_estimators()
        # objective function
        self.mvo_instance.objective_function()
        # constraints
        self.mvo_instance.get_constraints()
        
        if self.mvo_instance.method == 'max_return_min_vol':
            # shape
            assert len(self.mvo_instance.constraints) == 3
            # dtypes
            assert isinstance(self.mvo_instance.constraints[0], cp.constraints.nonpos.Inequality)
            assert isinstance(self.mvo_instance.constraints[1], cp.constraints.nonpos.Inequality)
            assert isinstance(self.mvo_instance.constraints[2], cp.constraints.nonpos.Inequality)
        
        elif self.mvo_instance.method == 'max_sharpe':
            # shape
            assert len(self.mvo_instance.constraints) == 5
            # dtypes
            assert isinstance(self.mvo_instance.constraints[0], cp.constraints.zero.Equality)
            assert isinstance(self.mvo_instance.constraints[1], cp.constraints.zero.Equality)
            assert isinstance(self.mvo_instance.constraints[2], cp.constraints.nonpos.Inequality)
            assert isinstance(self.mvo_instance.constraints[3], cp.constraints.nonpos.Inequality)
            assert isinstance(self.mvo_instance.constraints[4], cp.constraints.nonpos.Inequality)
        
        elif self.mvo_instance.method == 'efficient_return':
            # shape
            assert len(self.mvo_instance.constraints) == 4
            # dtypes
            assert isinstance(self.mvo_instance.constraints[0], cp.constraints.zero.Equality)
            assert isinstance(self.mvo_instance.constraints[1], cp.constraints.nonpos.Inequality)
            assert isinstance(self.mvo_instance.constraints[2], cp.constraints.nonpos.Inequality)
            assert isinstance(self.mvo_instance.constraints[3], cp.constraints.nonpos.Inequality)
        
        elif self.mvo_instance.method == 'efficient_risk':
            # shape
            assert len(self.mvo_instance.constraints) == 4
            # dtypes
            assert isinstance(self.mvo_instance.constraints[0], cp.constraints.zero.Equality)
            assert isinstance(self.mvo_instance.constraints[1], cp.constraints.nonpos.Inequality)
            assert isinstance(self.mvo_instance.constraints[2], cp.constraints.nonpos.Inequality)
            assert isinstance(self.mvo_instance.constraints[3], cp.constraints.nonpos.Inequality)
        
        elif self.mvo_instance.method == 'risk_parity':
            # shape
            assert len(self.mvo_instance.constraints) == 2
            # dtypes
            assert isinstance(self.mvo_instance.constraints, dict)
            # values
            assert self.mvo_instance.constraints['type'] == 'eq'
        
        else:
            # shape
            assert len(self.mvo_instance.constraints) == 3
            # dtypes
            assert isinstance(self.mvo_instance.constraints[0], cp.constraints.zero.Equality)
            assert isinstance(self.mvo_instance.constraints[1], cp.constraints.nonpos.Inequality)
            assert isinstance(self.mvo_instance.constraints[2], cp.constraints.nonpos.Inequality)
    
    @pytest.mark.parametrize("method", ['max_return', 'min_vol', 'max_return_min_vol', 'max_sharpe',
                                        'max_diversification', 'efficient_return', 'efficient_risk', 'risk_parity'])
    def test_optimize(self, method) -> None:
        """
        Test optimize.
        """
        # method
        self.mvo_instance.method = method
        # get start weights
        self.mvo_instance.get_initial_weights()
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
    
    @pytest.mark.parametrize("method", ['max_return', 'min_vol', 'max_return_min_vol', 'max_sharpe',
                                        'max_diversification', 'efficient_return', 'efficient_risk', 'risk_parity'])
    def test_check_weights(self, method) -> None:
        """
        Test check_weights.
        """
        # method
        self.mvo_instance.method = method
        # get start weights
        self.mvo_instance.get_initial_weights()
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
        assert (self.mvo_instance.weights >= 0).all()
        assert np.allclose(self.mvo_instance.weights.sum(), 1)
        # dtypes
        assert isinstance(self.mvo_instance.weights, np.ndarray)
        assert self.mvo_instance.weights.dtype == np.float64
    
    @pytest.mark.parametrize("method, cov_matrix_method",
                             [('max_return', 'covariance'),
                              ('min_vol', 'covariance'),
                              ('max_return_min_vol', 'covariance'),
                              ('max_sharpe', 'covariance'),
                              ('max_diversification', 'covariance'),
                              ('efficient_return', 'covariance'),
                              ('efficient_risk', 'covariance'),
                              ('risk_parity', 'covariance'),
                              ('max_return', 'empirical_covariance'),
                              ('min_vol', 'empirical_covariance'),
                              ('max_return_min_vol', 'empirical_covariance'),
                              ('max_sharpe', 'empirical_covariance'),
                              ('max_diversification', 'empirical_covariance'),
                              ('efficient_return', 'empirical_covariance'),
                              ('efficient_risk', 'empirical_covariance'),
                              ('risk_parity', 'empirical_covariance'),
                              ('max_return', 'shrunk_covariance'),
                              ('min_vol', 'shrunk_covariance'),
                              ('max_return_min_vol', 'shrunk_covariance'),
                              ('max_sharpe', 'shrunk_covariance'),
                              ('max_diversification', 'shrunk_covariance'),
                              ('efficient_return', 'shrunk_covariance'),
                              ('efficient_risk', 'shrunk_covariance'),
                              ('risk_parity', 'shrunk_covariance'),
                              ('max_return', 'ledoit_wolf'),
                              ('min_vol', 'ledoit_wolf'),
                              ('max_return_min_vol', 'ledoit_wolf'),
                              ('max_sharpe', 'ledoit_wolf'),
                              ('max_diversification', 'ledoit_wolf'),
                              ('efficient_return', 'ledoit_wolf'),
                              ('efficient_risk', 'ledoit_wolf'),
                              ('risk_parity', 'ledoit_wolf'),
                              ('max_return', 'oas'),
                              ('min_vol', 'oas'),
                              ('max_return_min_vol', 'oas'),
                              ('max_sharpe', 'oas'),
                              ('max_diversification', 'oas'),
                              ('efficient_return', 'oas'),
                              ('efficient_risk', 'oas'),
                              ('risk_parity', 'oas'),
                              ('max_return', 'graphical_lasso'),
                              ('min_vol', 'graphical_lasso'),
                              ('max_return_min_vol', 'graphical_lasso'),
                              ('max_sharpe', 'graphical_lasso'),
                              ('max_diversification', 'graphical_lasso'),
                              ('efficient_return', 'graphical_lasso'),
                              ('efficient_risk', 'graphical_lasso'),
                              ('risk_parity', 'graphical_lasso'),
                              ('max_return', 'graphical_lasso_cv'),
                              ('min_vol', 'graphical_lasso_cv'),
                              ('max_return_min_vol', 'graphical_lasso_cv'),
                              ('max_sharpe', 'graphical_lasso_cv'),
                              ('max_diversification', 'graphical_lasso_cv'),
                              ('efficient_return', 'graphical_lasso_cv'),
                              ('efficient_risk', 'graphical_lasso_cv'),
                              ('risk_parity', 'graphical_lasso_cv'),
                              ('max_return', 'minimum_covariance_determinant'),
                              ('min_vol', 'minimum_covariance_determinant'),
                              ('max_return_min_vol', 'minimum_covariance_determinant'),
                              ('max_sharpe', 'minimum_covariance_determinant'),
                              ('max_diversification', 'minimum_covariance_determinant'),
                              ('efficient_return', 'minimum_covariance_determinant'),
                              ('efficient_risk', 'minimum_covariance_determinant'),
                              ('risk_parity', 'minimum_covariance_determinant'),
                              ('max_return', 'semi_covariance'),
                              ('min_vol', 'semi_covariance'),
                              ('max_return_min_vol', 'semi_covariance'),
                              ('max_sharpe', 'semi_covariance'),
                              ('max_diversification', 'semi_covariance'),
                              ('efficient_return', 'semi_covariance'),
                              ('efficient_risk', 'semi_covariance'),
                              ('risk_parity', 'semi_covariance'),
                              ('max_return', 'exponential_covariance'),
                              ('min_vol', 'exponential_covariance'),
                              ('max_return_min_vol', 'exponential_covariance'),
                              ('max_sharpe', 'exponential_covariance'),
                              ('max_diversification', 'exponential_covariance'),
                              ('efficient_return', 'exponential_covariance'),
                              ('efficient_risk', 'exponential_covariance'),
                              ('risk_parity', 'exponential_covariance'),
                              ('max_return', 'denoised_covariance'),
                              ('min_vol', 'denoised_covariance'),
                              ('max_return_min_vol', 'denoised_covariance'),
                              ('max_sharpe', 'denoised_covariance'),
                              ('max_diversification', 'denoised_covariance'),
                              ('efficient_return', 'denoised_covariance'),
                              ('efficient_risk', 'denoised_covariance'),
                              ('risk_parity', 'denoised_covariance')
                              ])
    def test_compute_weights(self, method, cov_matrix_method) -> None:
        """
        Test get_optimal_weights.
        """
        # method
        self.mvo_instance.method = method
        self.mvo_instance.cov_matrix_method = cov_matrix_method
        # get optimal weights
        self.mvo_instance.compute_weights()
        
        # shape
        assert self.mvo_instance.weights.shape[1] == self.mvo_instance.n_assets
        # values
        assert np.allclose(self.mvo_instance.weights.sum(axis=1), 1)
        assert (self.mvo_instance.weights >= 0).all().all()
        # dtypes
        assert isinstance(self.mvo_instance.weights, pd.DataFrame)
        assert (self.mvo_instance.weights.dtypes == np.float64).all().all()
        # col
        assert self.mvo_instance.weights.columns.to_list() == self.mvo_instance.asset_names
        # index
        assert self.mvo_instance.weights.index == [self.mvo_instance.returns.index[-1]]
