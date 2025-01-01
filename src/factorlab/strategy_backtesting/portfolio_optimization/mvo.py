import pandas as pd
import numpy as np
from typing import Optional, Union, List, Any
import cvxpy as cp
from scipy.optimize import minimize

from factorlab.strategy_backtesting.portfolio_optimization.return_estimators import ReturnEstimators
from factorlab.strategy_backtesting.portfolio_optimization.risk_estimators import RiskEstimators
from factorlab.data_viz.plot import plot_bar


class MVO:
    """
    Mean variance optimization class.

    This class computes the optimized portfolio weights based on the returns of the assets or strategies
    using mean-variance optimization techniques.

    The mean-variance optimization problem can be solved using the following methods:
    - Maximum return
    - Minimum volatility
    - Maximum return minimum volatility
    - Maximum Sharpe ratio
    - Maximum diversification ratio
    - Efficient return
    - Efficient risk
    - Risk parity
    """
    def __init__(self,
                 returns: Union[pd.DataFrame, pd.Series],
                 method: str = 'max_return',
                 max_weight: float = 1.0,
                 min_weight: float = 0.0,
                 budget: float = 1.0,
                 risk_aversion: float = 1.0,
                 risk_free_rate: Optional[float] = None,
                 as_excess_returns: bool = False,
                 asset_names: Optional[List[str]] = None,
                 exp_ret_method: Optional[str] = None,
                 cov_matrix_method: Optional[str] = None,
                 target_return: Optional[float] = None,
                 target_risk: Optional[float] = None,
                 ann_factor: Optional[int] = None,
                 window_size: Optional[int] = 30,
                 solver: Optional[None] = None,
                 **kwargs: Any
                 ):
        """
        Constructor

        Parameters
        ----------
        returns: pd.DataFrame
            The returns of the assets or strategies. If not provided, the returns are computed from the prices.
        method: str, {'max_return', 'min_vol', 'max_return_min_vol', 'max_sharpe', 'max_diversification',
        'efficient_return', 'efficient_risk', 'risk_parity'}, default 'max_return_min_vol'
            Optimization method to compute weights.
        max_weight: float, default 1
            Maximum weight of the asset.
        min_weight: float, default 0
            Minimum weight of the asset.
        budget: float, default 1
            Budget of the portfolio.
        risk_aversion: float, default 1
            Risk aversion parameter.
        risk_free_rate: float, default None
            Risk-free rate.
        as_excess_returns: bool, default False
            Whether to compute excess returns.
        asset_names: list, default None
            List of asset names.
        exp_ret_method: str, {'historical_mean', 'historical_median', 'rolling_mean', 'rolling_median', 'ewma',
        'rolling_sharpe', 'rolling_sortino'}, default 'historical_mean'
            Method to compute the expected returns.
        cov_matrix_method: str, {'covariance', 'empirical_covariance', 'shrunk_covariance', 'ledoit_wolf', 'oas',
                      'graphical_lasso', 'graphical_lasso_cv', 'minimum_covariance_determinant', 'semi_covariance',
                      'exponential_covariance', 'denoised_covariance'}, default None
            Method to compute covariance matrix.
        target_return: float, default None
            Target return.
        target_risk: float, default None
            Target risk.
        ann_factor: int, default None
            Annualization factor. Default is 252 for daily data, 52 for weekly data, and 12 for monthly data.
        window_size: int, default 30
            Window size for the rolling estimators.
        solver: str, {'ECOS', 'SCS', 'OSQP', 'CLARABEL'}, default None
            Solver to use for optimization.
        **kwargs: dict
            Additional parameters.
        """
        self.returns = returns
        self.method = method
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.budget = budget
        self.risk_aversion = risk_aversion
        self.risk_free_rate = risk_free_rate
        self.as_excess_returns = as_excess_returns
        self.asset_names = asset_names
        self.exp_ret_method = exp_ret_method
        self.cov_matrix_method = cov_matrix_method
        self.target_return = target_return
        self.target_risk = target_risk
        self.ann_factor = ann_factor
        self.window_size = window_size
        self.solver = solver
        self.kwargs = kwargs

        self.freq = None
        self.n_assets = None
        self.weights = None
        self.y = None
        self.k = None
        self.exp_ret = None
        self.cov_matrix = None
        self.corr_matrix = None
        self.objective = None
        self.constraints = None
        self.bounds = None
        self.portfolio_ret = None
        self.portfolio_risk = None
        self.portfolio_corr = None
        self.preprocess_data()

    def preprocess_data(self) -> None:
        """
        Preprocesses data.
        """
        # returns
        if not isinstance(self.returns, pd.DataFrame) and not isinstance(self.returns, pd.Series):  # check data type
            raise ValueError('rets must be a pd.DataFrame or pd.Series')
        # data type conversion to float64
        if isinstance(self.returns, pd.Series):  # convert to df
            self.returns = self.returns.to_frame().astype('float64')
        elif isinstance(self.returns, pd.DataFrame):  # convert to float
            self.returns = self.returns.astype('float64')
        if isinstance(self.returns.index, pd.MultiIndex):  # convert to single index
            self.returns = self.returns.unstack()
        self.returns.index = pd.to_datetime(self.returns.index)  # convert to index to datetime

        # remove missing vals
        self.returns = self.returns.dropna(how='all').dropna(how='any', axis=1)

        # method
        if self.method not in ['max_return', 'min_vol', 'max_return_min_vol', 'max_sharpe', 'max_diversification',
                               'efficient_return', 'efficient_risk', 'risk_parity']:
            raise ValueError("Method is not supported. Valid methods are: 'max_return', 'min_vol', "
                             "'max_return_min_vol', 'max_sharpe', 'max_diversification', 'efficient_return', "
                             "'efficient_risk', 'risk_parity")

        # risk-free rate
        if self.risk_free_rate is None:
            self.risk_free_rate = 0.0

        # asset names
        if self.asset_names is None:
            self.asset_names = self.returns.columns.tolist()

        # exp_ret_method
        if self.exp_ret_method is None:
            self.exp_ret_method = 'historical_mean'

        # cov_matrix_method
        if self.cov_matrix_method is None:
            self.cov_matrix_method = 'covariance'

        # n_assets
        if self.n_assets is None:
            self.n_assets = self.returns.shape[1]

        # ann_factor
        if self.ann_factor is None:
            self.ann_factor = self.returns.groupby(self.returns.index.year).count().max().mode()[0]

        # freq
        self.freq = pd.infer_freq(self.returns.index)
        if self.freq is None:
            if self.ann_factor == 1:
                self.freq = 'Y'
            elif self.ann_factor == 4:
                self.freq = 'Q'
            elif self.ann_factor == 12:
                self.freq = 'M'
            elif self.ann_factor == 52:
                self.freq = 'W'
            else:
                self.freq = 'D'

        # solver
        if self.solver is None:
            self.solver = 'CLARABEL'

    def get_initial_weights(self) -> np.ndarray:
        """
        Get initial weights.
        """
        # initial weights
        if self.weights is None:
            self.weights = cp.Variable(self.n_assets)
            self.weights.value = np.ones(self.n_assets) * 1 / self.n_assets

        # max sharpe
        if self.method == 'max_sharpe':
            self.y = cp.Variable(self.n_assets)
            self.y.value = np.ones(self.n_assets) * 1 / self.n_assets

        # risk parity
        if self.method == 'risk_parity':
            self.weights = np.ones(self.n_assets) * 1 / self.n_assets

    def compute_estimators(self) -> None:
        """
        Compute estimators.
        """
        # expected returns
        self.exp_ret = ReturnEstimators(
            self.returns, method=self.exp_ret_method, as_excess_returns=self.as_excess_returns,
            risk_free_rate=self.risk_free_rate, ann_factor=self.ann_factor, window_size=self.window_size
        ).compute_expected_returns().to_numpy('float64')

        # covariance matrix
        self.cov_matrix = RiskEstimators(
            self.returns,
            window_size=self.window_size
        ).compute_covariance_matrix(method=self.cov_matrix_method)

        # correlation matrix
        if self.corr_matrix is None:
            self.corr_matrix = self.returns.corr().to_numpy('float64')

    def objective_function(self):
        """
        Optimize portfolio weights.
        """
        if self.method == 'max_return':
            self.portfolio_ret = cp.matmul(self.weights, self.exp_ret)
            self.portfolio_risk = cp.quad_form(self.weights, self.cov_matrix)
            self.objective = cp.Maximize(self.portfolio_ret)

        elif self.method == 'min_vol':
            self.portfolio_ret = cp.matmul(self.weights, self.exp_ret)
            self.portfolio_risk = cp.quad_form(self.weights, self.cov_matrix)
            self.objective = cp.Minimize(self.portfolio_risk)

        elif self.method == 'max_return_min_vol':
            self.portfolio_ret = cp.matmul(self.weights, self.exp_ret)
            self.portfolio_risk = cp.quad_form(self.weights, self.cov_matrix)
            self.objective = cp.Minimize(self.risk_aversion * self.portfolio_risk - self.portfolio_ret)

        elif self.method == 'max_sharpe':
            self.k = cp.Variable()
            self.portfolio_risk = cp.quad_form(self.y, self.cov_matrix)
            self.weights = self.y / self.k
            self.portfolio_ret = cp.matmul(self.weights, self.exp_ret)
            self.objective = cp.Minimize(self.portfolio_risk)

        elif self.method == 'max_diversification':
            self.portfolio_ret = cp.matmul(self.weights, self.exp_ret)
            self.portfolio_risk = cp.quad_form(self.weights, self.cov_matrix)
            self.portfolio_corr = cp.quad_form(self.weights, self.corr_matrix)
            self.objective = cp.Minimize(self.portfolio_corr)

        elif self.method == 'efficient_return':
            self.portfolio_ret = cp.matmul(self.weights, self.exp_ret)
            self.portfolio_risk = cp.quad_form(self.weights, self.cov_matrix)
            self.objective = cp.Minimize(self.portfolio_risk)

        elif self.method == 'efficient_risk':
            self.portfolio_ret = cp.matmul(self.weights, self.exp_ret)
            self.portfolio_risk = cp.quad_form(self.weights, self.cov_matrix)
            self.objective = cp.Maximize(self.portfolio_ret)

        elif self.method == 'risk_parity':
            def risk_parity_obj(weights, cov_matrix):
                portfolio_var = np.dot(weights.T, np.dot(cov_matrix, weights))
                mrc = np.dot(cov_matrix, weights)
                rc = weights * mrc / portfolio_var
                return np.sum((rc - self.weights) ** 2)
            self.objective = risk_parity_obj
            self.portfolio_ret = np.dot(self.weights, self.exp_ret)
            self.portfolio_risk = np.dot(self.weights.T, np.dot(self.cov_matrix, self.weights))

        else:
            raise ValueError("Method is not supported. Valid methods are: 'max_return',  'min_vol', "
                             "'max_return_min_vol', 'max_sharpe', 'max_diversification', 'efficient_return', "
                             "'efficient_risk', 'risk_parity'")

        return self.objective

    def get_constraints(self):
        """
        Get constraints.
        """
        if self.method == 'max_return':
            self.constraints = [
                cp.sum(self.weights) == self.budget,
                self.weights <= self.max_weight,
                self.weights >= self.min_weight
            ]

        elif self.method == 'min_vol':
            self.constraints = [
                cp.sum(self.weights) == self.budget,
                self.weights <= self.max_weight,
                self.weights >= self.min_weight
            ]

        elif self.method == 'max_return_min_vol':
            self.constraints = [
                cp.sum(self.weights) <= self.budget,
                self.weights <= self.max_weight,
                self.weights >= self.min_weight
            ]

        # max sharpe
        elif self.method == 'max_sharpe':
            self.constraints = [
                cp.sum((self.exp_ret - self.risk_free_rate).T @ self.y) == self.budget,
                cp.sum(self.y) == self.k,
                self.k >= 0,
                self.y <= self.k * self.max_weight,
                self.y >= self.k * self.min_weight
            ]

        elif self.method == 'max_diversification':
            self.constraints = [
                cp.sum(self.weights) == self.budget,
                self.weights <= self.max_weight,
                self.weights >= self.min_weight
            ]

        elif self.method == 'efficient_return':
            if self.target_return is None:
                raise ValueError('Target return is required for this method.')
            else:
                self.constraints = [
                    cp.sum(self.weights) == self.budget,
                    self.weights <= self.max_weight,
                    self.weights >= self.min_weight,
                    self.portfolio_ret >= self.target_return/self.ann_factor
                ]

        elif self.method == 'efficient_risk':
            if self.target_risk is None:
                raise ValueError('Target risk is required for this method.')
            else:
                self.constraints = [
                    cp.sum(self.weights) == self.budget,
                    self.weights <= self.max_weight,
                    self.weights >= self.min_weight,
                    self.portfolio_risk <= (self.target_risk / np.sqrt(self.ann_factor))**2
                ]

        elif self.method == 'risk_parity':
            self.constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - self.budget})
            self.bounds = tuple((self.min_weight,  self.max_weight) for _ in range(self.n_assets))

        else:
            self.constraints = [
                cp.sum(self.weights) == self.budget,
                self.weights <= self.max_weight,
                self.weights >= self.min_weight
            ]

        return self.constraints

    def optimize(self, **kwargs):
        """
        Optimize portfolio weights.

        Returns
        -------
        np.ndarray
            Optimized portfolio weights.
        """
        # optimization problem
        if self.method != 'risk_parity':
            prob = cp.Problem(objective=self.objective, constraints=self.constraints)
            prob.solve(solver=self.solver, **kwargs)

            if self.weights.value is None:
                raise ValueError('Could not find optimal weights.')
            else:
                self.weights = self.weights.value
                self.portfolio_risk = self.portfolio_risk.value
                self.portfolio_ret = self.portfolio_ret.value

            # max sharpe
            if self.method == 'max_sharpe':
                self.weights *= self.budget
                self.portfolio_risk = self.weights @ self.cov_matrix @ self.weights.T
                self.portfolio_ret = self.exp_ret.T @ self.weights

            # max diversification ratio
            if self.method == 'max_diversification':
                self.weights /= np.diag(self.cov_matrix)
                self.weights /= np.sum(self.weights)
                self.weights *= self.budget
                self.portfolio_risk = self.weights @ self.cov_matrix @ self.weights.T
                self.portfolio_ret = self.exp_ret.T @ self.weights

        # risk parity
        else:
            result = minimize(self.objective, self.weights, args=self.cov_matrix, constraints=self.constraints,
                           bounds=self.bounds, method='SLSQP')
            self.weights = result.x
            self.portfolio_risk = self.weights @ self.cov_matrix @ self.weights.T
            self.portfolio_ret = self.exp_ret.T @ self.weights

    def check_weights(self, threshold: float = 1e-4):
        """
        Check if weights array has one element close to 1 and the rest close to 0,
        and return the rounded weights.

        Parameters
        ----------
        threshold: float, default 1e-4
            Threshold value to round the weights.

        Returns
        -------
        np.ndarray: numpy array of rounded weights to 1 and 0 if the condition is met.
                If the condition is not met, returns the original weights.
        """
        # Find the index of the weight closest to 1
        close_to_one_index = np.argmax(self.weights)

        # Check if the weight at this index is within the threshold of 1
        if np.abs(self.weights[close_to_one_index] - 1) <= threshold:
            # Check if all other weights are within the threshold of 0
            all_close_to_zero = np.all(np.abs(np.delete(self.weights, close_to_one_index)) <= threshold)

            if all_close_to_zero:
                # Create a rounded weights array
                rounded_weights = np.zeros_like(self.weights)
                rounded_weights[close_to_one_index] = 1
                self.weights = rounded_weights

        # If conditions are not met, return the original weights
        return self.weights

    def compute_weights(self, **kwargs):
        """
        Computes optimal portfolio weights.
        """
        # get start weights
        self.get_initial_weights()

        # compute estimators
        self.compute_estimators()

        # objective function
        self.objective_function()

        # constraints
        self.get_constraints()

        # optimize
        self.optimize(**kwargs)

        # check weights
        self.weights = self.check_weights()

        self.weights = pd.DataFrame(self.weights, index=self.asset_names, columns=[self.returns.index[-1]]).T

        return self.weights

    def plot_weights(self):
        """
        Plot the optimized portfolio weights.
        """
        plot_bar(self.weights.T.sort_values(by=[self.returns.index[-1]]), axis='horizontal', x_label='weights')
