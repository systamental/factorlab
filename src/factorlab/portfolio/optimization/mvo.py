import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import cvxpy as cp
from warnings import warn

from factorlab.portfolio_opt.base import PortfolioOptimizerBase
from factorlab.portfolio_opt.estimators.risk import RiskMetrics


class MeanVarianceOptimizer(PortfolioOptimizerBase):
    """
    Solves the standard Mean-Variance Optimization (MVO) problem to maximize
    investor utility, defined as Expected Return - (Risk Aversion * Portfolio Variance).

    Objective: Maximize (W^T * mu) - lambda * (W^T * Sigma * W)
    """

    def __init__(self,
                 risk_aversion: float = 1.0,
                 max_weight: float = 1.0,
                 min_weight: float = 0.0,
                 budget: float = 1.0,
                 risk_estimator: str = 'ledoit_wolf',
                 solver: Optional[str] = None,
                 **kwargs):
        """
        Initializes the MVO Utility optimizer.

        Parameters
        ----------
        risk_aversion: float, default 1.0
            The investor's risk tolerance (lambda). Higher values lead to lower risk portfolios.
        max_weight: float, default 1.0
            Maximum weight constraint per asset.
        min_weight: float, default 0.0
            Minimum weight constraint per asset (allows short selling if set < 0).
        budget: float, default 1.0
            The total sum of weights (e.g., 1.0 for a fully invested portfolio).
        risk_estimator: str, default 'covariance'
            The name of the method from RiskMetrics to use for covariance matrix estimation.
        risk_estimator_params: Dict[str, Any], default None
            Keyword arguments passed to the selected RiskMetrics method.
        solver: str, default 'CLARABEL'
            The CVXPY solver to use for the quadratic optimization problem.
        """
        # Pass base parameters to parent class
        super().__init__(**kwargs)

        self.name = "MeanVarianceOptimizer"
        self.description = "Mean variance optimizer (Return - lambda * Risk)."

        # MVO Utility specific parameters
        self.risk_aversion = risk_aversion
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.budget = budget
        self.risk_estimator = risk_estimator
        self.solver = solver if solver else 'CLARABEL'

        # Internal state
        self._n_assets: Optional[int] = None
        self._asset_names: Optional[List[str]] = None

    def _regularize_cov_matrix(self, cov_matrix_np: np.ndarray, min_eigenvalue: float = 1e-6) -> np.ndarray:
        """
        Performs eigenvalue clipping (regularization) to ensure the covariance matrix
        is Positive Semidefinite (PSD), which is required for DCP compliance.
        """
        # Perform Eigendecomposition
        try:
            # We use 'eigh' for symmetric matrices (like covariance)
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix_np)
        except np.linalg.LinAlgError:
            warn("Covariance matrix is numerically singular. Cannot perform Eigendecomposition.")
            # Fallback to adding a small diagonal if decomposition fails
            return cov_matrix_np + np.diag(np.ones(cov_matrix_np.shape[0]) * min_eigenvalue)

        # clip negative or zero eigenvalues to a small positive number to force PSD
        eigenvalues[eigenvalues < min_eigenvalue] = min_eigenvalue

        # reconstruct the regularized covariance matrix (Sigma')
        # Sigma' = V * Lambda' * V_T
        regularized_cov = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

        # ensure numerical symmetry after reconstruction
        regularized_cov = (regularized_cov + regularized_cov.T) / 2

        return regularized_cov

    def _compute_estimators(self, returns: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculates the Covariance Matrix (Sigma) required for the risk component of the utility function.
        """
        returns = returns.dropna(how='all').dropna(how='any', axis=1)

        # validate and clean returns
        returns = RiskMetrics._validate_returns(returns)
        self._n_assets = returns.shape[1]
        self._asset_names = returns.columns.tolist()

        # 1. Dynamically calculate the Covariance Matrix (Sigma)
        try:
            risk_func = getattr(RiskMetrics, self.risk_estimator)
        except AttributeError:
            raise ValueError(f"RiskMetrics has no method named '{self.risk_estimator}'.")

        # Execute the chosen risk estimator function using its params
        cov_matrix = risk_func(returns, **self.risk_estimator_params)
        cov_matrix_np = cov_matrix.values

        cov_matrix_regularized = self._regularize_cov_matrix(cov_matrix_np)

        return {
            'assets': self._asset_names,
            'n_assets': self._n_assets,
            'cov_matrix': cov_matrix_regularized
        }

    def _compute_weights(self,
                         estimators: Dict[str, Any],
                         signals: pd.Series,
                         current_weights: pd.Series
                         ) -> pd.Series:
        """
        The transform step: Solves the quadratic MVO optimization problem to maximize utility.

        The `signals` input is treated as the Expected Returns vector (mu).

        Parameters
        ----------
        estimators : Dict[str, Any]
            Dictionary containing estimators calculated from historical returns.
        signals : pd.Series
            The strategy's directional view for the period, treated as expected returns.
        current_weights : pd.Series
            The weights currently held (not used in this strategy).

        Returns
        -------
        pd.Series
            The target weights for the period.
        """
        # extract estimators
        n_assets = estimators['n_assets']
        asset_names = estimators['assets']
        cov_matrix = estimators['cov_matrix']

        # expected Returns (mu) - sourced from the signals
        exp_ret = signals.reindex(asset_names, fill_value=0.0).values

        # --- 1. Define Optimization Variables ---
        weights = cp.Variable(n_assets)

        # --- 2. Define Objective Function (Maximize Utility) ---
        portfolio_ret = cp.matmul(weights, exp_ret)
        portfolio_risk_sq = cp.quad_form(weights, cov_matrix)

        # Objective: Maximize (Return) - lambda * (Risk)
        objective = cp.Minimize(self.risk_aversion * portfolio_risk_sq - portfolio_ret)

        # --- 3. Define Constraints ---
        constraints = [
            cp.sum(weights) <= self.budget,  # Budget constraint
            weights >= self.min_weight,  # Minimum weight constraint
            weights <= self.max_weight  # Maximum weight constraint
        ]

        # --- 4. Solve Problem ---
        try:
            prob = cp.Problem(objective, constraints)
            prob.solve(solver=self.solver)
        except Exception as e:
            warn(f"{self.name} failed to solve with {self.solver}. Error: {e}")

        # extract weights
        optimal_weights = weights.value

        if optimal_weights is None or 'optimal' not in prob.status:
            warn("Optimization failed or returned None. Falling back to equal weights.")
            # fallback to equal weighting if solver fails
            optimal_weights = np.ones(n_assets) * (self.budget / n_assets)

        # add to pd.Series
        target_weights = pd.Series(optimal_weights, index=asset_names)

        return target_weights
