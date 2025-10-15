import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import cvxpy as cp
from warnings import warn

from factorlab.portfolio_opt.base import PortfolioOptimizerBase
from factorlab.portfolio_opt.estimators.risk import RiskMetrics


class MinVolOptimizer(PortfolioOptimizerBase):
    """
    Solves the Minimum Volatility (MinVol) problem.

    The objective is to find the portfolio weights (W) that minimize the
    portfolio's total variance (W^T * Sigma * W) subject to budget and
    individual weight constraints.
    """

    def __init__(self,
                 max_weight: float = 1.0,
                 min_weight: float = 0.0,
                 budget: float = 1.0,
                 risk_estimator: str = 'ledoit_wolf',
                 solver: Optional[str] = None,
                 **kwargs):
        """
        Initializes the Minimum Volatility optimizer.

        Parameters
        ----------
        max_weight: float, default 1.0
            Maximum weight constraint per asset.
        min_weight: float, default 0.0
            Minimum weight constraint per asset (allows short selling if set < 0).
        budget: float, default 1.0
            The total sum of weights (e.g., 1.0 for a fully invested portfolio).
        risk_estimator: str, default 'ledoit_wolf'
            The name of the method from RiskMetrics to use for covariance matrix estimation.
        solver: str, default 'CLARABEL'
            The CVXPY solver to use for the quadratic optimization problem.
        """
        # Pass base parameters to parent class
        super().__init__(**kwargs)

        self.name = "MinVolOptimizer"
        self.description = "Minimizes portfolio volatility."

        # MinVol specific parameters
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
        is Positive Semidefinite (PSD), which is required for DCP compliance in CVXPY.
        """
        # Perform Eigendecomposition
        try:
            # We use 'eigh' for symmetric matrices (like covariance)
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix_np)
        except np.linalg.LinAlgError:
            warn("Covariance matrix is numerically singular. Cannot perform Eigendecomposition.")
            # Fallback to adding a small diagonal if decomposition fails
            return cov_matrix_np + np.diag(np.ones(cov_matrix_np.shape[0]) * min_eigenvalue)

        # Clip negative or zero eigenvalues to a small positive number to force PSD
        eigenvalues[eigenvalues < min_eigenvalue] = min_eigenvalue

        # Reconstruct the regularized covariance matrix (Sigma')
        regularized_cov = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

        # Ensure numerical symmetry after reconstruction
        regularized_cov = (regularized_cov + regularized_cov.T) / 2

        return regularized_cov

    def _compute_estimators(self, returns: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculates the Covariance Matrix (Sigma) required for the risk component.
        """
        # Drop rows/columns with all missing data
        returns = returns.dropna(how='all').dropna(how='any', axis=1)

        # Validate and clean returns
        returns = RiskMetrics._validate_returns(returns)
        self._n_assets = returns.shape[1]
        self._asset_names = returns.columns.tolist()

        # 1. Dynamically calculate the Covariance Matrix (Sigma)
        try:
            risk_func = getattr(RiskMetrics, self.risk_estimator)
        except AttributeError:
            raise ValueError(f"RiskMetrics has no method named '{self.risk_estimator}'.")

        # Execute the chosen risk estimator function
        cov_matrix = risk_func(returns, **self.risk_estimator_params)
        cov_matrix_np = cov_matrix.values

        # Regularize the matrix
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
        The transform step: Solves the Minimum Volatility quadratic problem.

        Parameters
        ----------
        estimators : Dict[str, Any]
            Contains the calculated covariance matrix and asset list.
        signals : pd.Series
            The strategy's directional view for the period.
        current_weights : pd.Series
            The weights currently held (required by interface but unused here).

        Returns
        -------
        pd.Series
            The target weights for the period.
        """
        # Extract estimators
        n_assets = estimators['n_assets']
        asset_names = estimators['assets']
        cov_matrix = estimators['cov_matrix']

        # --- 1. Define Optimization Variables ---
        weights = cp.Variable(n_assets)

        # --- 2. Define Objective Function (Minimize Portfolio Risk) ---
        portfolio_risk_sq = cp.quad_form(weights, cov_matrix)
        objective = cp.Minimize(portfolio_risk_sq)

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

        # min vol weights
        mv_weights = pd.Series(np.abs(optimal_weights), index=asset_names)

        # Apply signals
        target_weights = signals.multiply(mv_weights, axis=0)

        return target_weights
