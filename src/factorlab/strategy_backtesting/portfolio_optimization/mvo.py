import pandas as pd
import numpy as np
from typing import Optional, Union, List
import cvxpy as cp


class MVO:
    """
    Mean variance optimization class.

    This class computes the optimized portfolio weights based on the returns of the assets or strategies
    using mean-variance optimization techniques.

    The mean-variance optimization problem is formulated as follows:

    min w'Σw
    s.t. w'μ >= r_min
            w'μ <= r_max
            sum(w) = 1
            w >= 0

    where:
    w is the vector of weights
    Σ is the covariance matrix
    μ is the vector of returns
    r_min is the minimum return
    r_max is the maximum return

    The mean-variance optimization problem can be solved using the following methods:
    - Minimum volatility
    - Maximum return minimum volatility
    - Maximum Sharpe ratio
    - Maximum diversification ratio
    - Efficient return
    - Efficient risk
    """
    def __init__(self,
                 returns: Union[pd.DataFrame, pd.Series],
                 method: str = 'min_volatility',
                 max_weight: float = 1,
                 min_weight: float = 0,
                 budget: float = 1,
                 risk_aversion: float = 1,
                 risk_free_rate: Optional[float] = None,
                 asset_names: Optional[List[str]] = None,
                 exp_ret_method: Optional[str] = None,
                 cov_matrix_method: Optional[str] = None,
                 target_ret: Optional[float] = None,
                 target_risk: Optional[float] = None,
                 ann_factor: Optional[int] = None
                 ):
        """
        Constructor

        Parameters
        ----------
        returns: pd.DataFrame
            The returns of the assets or strategies. If not provided, the returns are computed from the prices.
        method: str, {'min_volatility', 'target_return_min_volatility', 'max_sharpe_ratio', 'max_return_min_vol',
        'max_diversification_ratio', 'efficient_return', 'efficient_risk'}, default 'min_volatility'
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
        asset_names: list, default None
            List of asset names.
        exp_ret_method: str, {'mean', 'median', 'ewm'}, default None
            Method to compute expected returns.
        cov_matrix_method: str, {'sample', 'ewm', 'shrinkage'}, default None
            Method to compute covariance matrix.
        target_ret: float, default None
            Target return.
        target_risk: float, default None
            Target risk.
        ann_factor: int, default None
            Annualization factor. Default is 252 for daily data, 52 for weekly data, and 12 for monthly data.
        """
        self.returns = returns
        self.method = method
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.budget = budget
        self.risk_aversion = risk_aversion
        self.risk_free_rate = risk_free_rate
        self.asset_names = asset_names
        self.exp_ret_method = exp_ret_method
        self.cov_matrix_method = cov_matrix_method
        self.target_ret = target_ret
        self.target_risk = target_risk
        self.ann_factor = ann_factor
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
        self.portfolio_ret = None
        self.portfolio_risk = None
        self.portfolio_corr = None
        self.preprocess_data()

    def preprocess_data(self) -> None:
        """
        Preprocesses data.
        """
        # ret
        if not isinstance(self.returns, pd.DataFrame) and not isinstance(self.returns, pd.Series):  # check data type
            raise ValueError('rets must be a pd.DataFrame or pd.Series')
        if isinstance(self.returns, pd.Series):  # convert to df
            self.returns = self.returns.to_frame()
        if isinstance(self.returns.index, pd.MultiIndex):  # convert to single index
            self.returns = self.ret.unstack()
        self.returns.index = pd.to_datetime(self.returns.index)  # convert to index to datetime

        # method
        if self.method not in ['min_vol', 'max_return_min_vol', 'max_sharpe', 'max_diversification',
                               'return_targeting', 'risk_budgeting']:
            raise ValueError("Method is not supported. Valid methods are: 'min_vol', 'max_return_min_vol', "
                             "'max_sharpe', 'max_diversification', 'return_targeting', 'risk_budgeting'")

        # risk-free rate
        if self.risk_free_rate is None:
            self.risk_free_rate = 0.0

        # asset names
        if self.asset_names is None:
            self.asset_names = self.returns.columns.tolist()

        # n_assets
        if self.n_assets is None:
            self.n_assets = self.returns.shape[1]

        # freq
        self.freq = pd.infer_freq(self.returns.index)

        # ann_factor
        if self.ann_factor is None:
            if self.freq == 'D':
                self.ann_factor = 252
            elif self.freq == 'W':
                self.ann_factor = 52
            elif self.freq == 'M':
                self.ann_factor = 12
            else:
                self.ann_factor = 252

    def get_start_weights(self) -> np.ndarray:
        """
        Get start weights.
        """
        # start weights
        if self.weights is None:
            self.weights = cp.Variable(self.n_assets)
            self.weights.value = np.ones(self.n_assets) * 1 / self.n_assets

        # max sharpe
        if self.method == 'max_sharpe':
            self.y = cp.Variable(self.n_assets)
            self.y.value = np.ones(self.n_assets) * 1 / self.n_assets

    def compute_estimators(self, window_size: int = 30) -> None:
        """
        Compute estimators.
        """
        # expected returns
        if self.exp_ret_method == 'exponential':
            self.exp_ret = self.returns.ewm(span=window_size).mean().iloc[-1].values * self.ann_factor
        else:
            self.exp_ret = self.returns.mean().values * self.ann_factor

        # covariance matrix
        if self.cov_matrix is None:
            self.cov_matrix = self.returns.cov().values

        # correlation matrix
        if self.corr_matrix is None:
            self.corr_matrix = self.returns.corr().values

    def objective_function(self):
        """
        Optimize portfolio weights.
        """
        if self.method == 'min_vol':
            self.portfolio_ret = cp.matmul(self.weights, self.exp_ret)
            self.portfolio_risk = cp.quad_form(self.weights, self.cov_matrix)
            self.objective = cp.Minimize(self.portfolio_risk)

        elif self.method == 'max_return_min_vol':
            self.portfolio_ret = cp.matmul(self.weights, self.exp_ret)
            self.portfolio_risk = cp.quad_form(self.weights, self.cov_matrix)
            self.objective = cp.Minimize(self.risk_aversion * self.portfolio_risk - self.portfolio_ret)

        elif self.method == 'max_sharpe':
            self.k = cp.Variable(1)
            self.portfolio_risk = cp.quad_form(self.y, self.cov_matrix)
            self.weights = self.y / self.k
            self.portfolio_ret = cp.matmul(self.weights, self.exp_ret)
            self.objective = cp.Minimize(self.portfolio_risk)

        elif self.method == 'max_diversification':
            self.portfolio_ret = cp.matmul(self.weights, self.exp_ret)
            self.portfolio_risk = cp.quad_form(self.weights, self.cov_matrix)
            self.portfolio_corr = cp.quad_form(self.weights, self.corr_matrix)
            self.objective = cp.Minimize(self.portfolio_corr)

        elif self.method == 'return_targeting':
            self.portfolio_ret = cp.matmul(self.weights, self.exp_ret)
            self.portfolio_risk = cp.quad_form(self.weights, self.cov_matrix)
            self.objective = cp.Minimize(self.portfolio_risk)

        elif self.method == 'risk_budgeting':
            self.portfolio_ret = cp.matmul(self.weights, self.exp_ret)
            self.portfolio_risk = cp.quad_form(self.weights, self.cov_matrix)
            self.objective = cp.Maximize(self.portfolio_ret)

        else:
            raise ValueError("Method is not supported. Valid methods are: 'min_vol', 'max_return_min_vol', "
                             "'max_sharpe', 'max_diversification', 'return_targeting', 'risk_budgeting'")

        return self.objective

    def get_constraints(self):
        """
        Get constraints.
        """
        if self.method == 'return_targeting':
            if self.target_ret is None:
                raise ValueError('Target return is required for this method.')
            else:
                self.constraints = [
                    self.weights <= self.max_weight,
                    self.weights >= self.min_weight,
                    cp.sum(self.weights) == self.budget,
                    self.portfolio_ret >= self.target_ret
                ]

        elif self.method == 'risk_budgeting':
            if self.target_risk is None:
                raise ValueError('Target risk is required for this method.')
            else:
                self.constraints = [
                    self.weights <= self.max_weight,
                    self.weights >= self.min_weight,
                    cp.sum(self.weights) == self.budget,
                    self.portfolio_risk <= self.target_risk
                ]

        elif self.method == 'max_sharpe':
            self.constraints = [
                cp.sum((self.exp_ret - self.risk_free_rate).T @ self.y) == self.budget,
                cp.sum(self.y) == self.k,
                self.k >= 0,
                self.y <= self.k * self.max_weight,
                self.y >= self.k * self.min_weight
            ]

        else:
            self.constraints = [
                self.weights <= self.max_weight,
                self.weights >= self.min_weight,
                cp.sum(self.weights) == self.budget
            ]

        return self.constraints

    def optimize(self, solver: Optional[str] = 'ECOS', **kwargs):
        """
        Optimize portfolio weights.

        Parameters
        ----------
        solver: str, {''ECOS', 'SCS', OSQP'}, default 'ECOS'
            Solver to use for optimization.

        Returns
        -------
        np.ndarray
            Optimized portfolio weights.
        """
        # optimization problem
        prob = cp.Problem(objective=self.objective, constraints=self.constraints)
        prob.solve(solver=solver, **kwargs)

        if self.weights.value is None:
            raise ValueError('Could not find optimal weights.')
        else:
            self.weights = self.weights.value
            self.portfolio_risk = self.portfolio_risk.value
            self.portfolio_ret = self.portfolio_ret.value

        # max diversification ratio
        if self.method == 'max_diversification':
            self.weights /= np.diag(self.cov_matrix)
            self.weights /= np.sum(self.weights)
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
                return rounded_weights

        # If conditions are not met, return the original weights
        return self.weights

    def get_optimal_weights(self, solver: Optional[str] = 'ECOS', **kwargs):
        """
        Computes optimal portfolio weights.
        """
        # get start weights
        self.get_start_weights()

        # compute estimators
        self.compute_estimators()

        # objective function
        self.objective_function()

        # constraints
        self.get_constraints()

        # optimize
        self.optimize(solver=solver, **kwargs)

        # check weights
        self.weights = self.check_weights()

        return self.weights

    def compute_portfolio_metrics(self):
        """
        Get portfolio metrics.
        """
        pass
        # # portfolio return
        # self.portfolio_rets = self.weights.value.T @ self.exp_rets
        # # portfolio risk
        # self.portfolio_risk = np.sqrt(self.weights.value.T @ self.cov_matrix @ self.weights.value)


# def mean_variance_optimize(μ, Σ, vol_tgt=0.05, Tht=None):
#     """
#
#     :param μ: Return vector
#     :param Σ: Covariance matrix
#     :param vol_tgt: Volatility target
#     :param Tht: (Optional) constraint matrix. The optimization enforces w.T @ Θ = 0
#     :return: Optimal portfolio weights
#     """
#
#     # reshape μ if necessary and check dimensions are consistent
#     μ = np.atleast_2d(μ).T
#     assert (Σ.shape == (μ.shape[0], μ.shape[0]))
#
#     if Tht is None:
#         # default mean-variance optimizer
#         Σi = np.linalg.inv(Σ)
#         α = np.sqrt(μ.T @ Σi @ μ) / vol_tgt
#         return (Σi @ μ) / α
#     else:
#         Tht = np.atleast_2d(Tht)
#         assert (μ.shape[0] == Tht.shape[0])
#         if Tht.shape[1] == 1:
#             # single constraint on the portfolio weights
#             Σi = np.linalg.inv(Σ)
#             d_μ_μ = μ.T @ Σi @ μ
#             d_μ_Θ = μ.T @ Σi @ Tht
#             d_Θ_Θ = Tht.T @ Σi @ Tht
#
#             β = d_μ_Θ / d_Θ_Θ
#             α = np.sqrt((d_μ_μ * d_Θ_Θ - d_μ_Θ * d_μ_Θ) / d_Θ_Θ) / vol_tgt
#             return Σi @ (μ - β * Tht) / α
#         else:
#             # perform an svd of the constraint matrix...
#             u, d, vt = np.linalg.svd(Tht, full_matrices=True)
#             # ...and find the basis for its null space
#             p = u[:, Tht.shape[1]:]
#
#             # The require portfolio weight matrix w is now given by w = p @ w_p
#             # for some vector w_p i.e. it lies in the null space of Θ. Therefore,
#             # we can solve a simple mean-variance optimization in the null space
#             # by using projected versions of the return and covariance matrices.
#             # These are calculated below and the estimated w_p is converted back
#             # into the portfolio space by left multiplying by p
#             μ_p = p.T @ μ
#             Σ_p = p.T @ Σ @ p
#             Σi_p = np.linalg.inv(Σ_p)
#
#             α = np.sqrt(μ_p.T @ Σi_p @ μ_p) / vol_tgt
#             return (p @ Σi_p @ μ_p) / α
#
#
# def robust_regression_cqp(A, b, γ):
#     '''
#     Implementation of robust regression with Huber estiamtors as described in:
#     "Robust Linear and Support Vector Regression" Mangasarian and Musicant
#
#     Given A, b we are trying to solve for x which minimizes
#                 Σ ρ((Ax-b)[i])
#     where ρ is the Huber M-estimator.
#
#     This is solved by solving the convex quadratic program (1):
#                 min z'z/2 + γ e' (r + s)
#                 s.t. Ax-b-z=r-s
#                      r,s >= 0
#
#     Cvxopt solves problems of the form (2):
#                 min x'Px/2 + q'x
#                 s.t. Gx <= h
#                      Cx = d
#
#     We rewrite the constraints above in terms of P, q, G, h, C and d. Note that
#     this will solve for x, z, r and s from (1). The function splits and returns this
#     '''
#     A, b = np.atleast_2d(A, b)  # ensure they are 2d matrices
#     t, l = A.shape  # check shapes match
#     assert (b.shape[0] == t)
#     assert (γ > 0)
#
#     n = l + 3 * t  # size of the solution vector for convex optimization
#
#     P = np.diag(np.hstack([np.zeros(l), np.ones(t), np.zeros(2 * t)]))
#     q = np.zeros((n, 1))
#     q[l + t:] = γ
#     G = -np.diag(np.sign(q.reshape(q.shape[0])))
#     h = np.zeros((n, 1))
#     C = np.hstack([A, -np.eye(t), -np.eye(t), np.eye(t)])
#
#     P = matrix(P)
#     q = matrix(q)
#     G = matrix(G)
#     h = matrix(h)
#     C = matrix(C)
#     d = matrix(b)
#
#     sol = solvers.qp(P, q, G, h, C, d)
#
#     y = np.array(sol['x'])
#
#     x = y[:l]
#     z = y[l:l + t]
#     r = y[l + t:l + 2 * t]
#     s = y[l + 2 * t:]
#
#     return x, z, r, s
#
#
# def robust_mvo_huber(X, r=None, scale=None):
#     '''
#     Performs MVO by treating it as a regression. However, we use
#     the robust SVM function to perform a robust regression.
#     '''
#     T, k = X.shape
#
#     if r is None:
#         r = np.mean(X, axis=0, keepdims=True).T
#     else:
#         r = np.atleast_2d(r)
#         assert (r.shape == (k, 1))
#
#     b = np.ones((T, 1))
#
#     if scale is None:
#         scale = robust.mad(b - X @ np.linalg.inv(X.T @ X) @ X.T @ b, c=1.)
#     else:
#         assert (scale > 0)
#
#     w, z, r, s = robust_regression_cqp(X, b, scale)
#     return w / np.sum(w)
