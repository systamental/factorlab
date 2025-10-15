import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import cvxpy as cp
from warnings import warn

from factorlab.portfolio_opt.base import PortfolioOptimizerBase
from factorlab.portfolio_opt.estimators.risk import RiskMetrics


class MaxDiversification(PortfolioOptimizerBase):
    """
    Solves the standard Maximum Diversification (MD) problem by finding the portfolio
    that maximizes the Diversification Ratio (DR).

    The optimization is simplified to finding the unnormalized weights (W') that
    minimize portfolio correlation. These weights are then scaled by asset volatility
    and finally by the strategy's directional signal.
    """

    def __init__(self,
                 max_weight: float = 1.0,
                 min_weight: float = 0.0,
                 budget: float = 1.0,
                 risk_estimator: str = 'ledoit_wolf',
                 solver: Optional[str] = None,
                 **kwargs):
        """
        Initializes the Max Diversification optimizer.

        Parameters
        ----------
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

        self.name = "MaxDiversificationOptimizer"
        self.description = "Optimizer maximizing the Diversification Ratio."

        # Max Diversification specific parameters
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

        corr_matrix = returns.corr().to_numpy('float64')

        return {
            'assets': self._asset_names,
            'n_assets': self._n_assets,
            # Daily covariance matrix is required for volatility calculation
            'covariance_matrix': pd.DataFrame(cov_matrix_regularized, index=self._asset_names,
                                              columns=self._asset_names),
            'cov_matrix_np': cov_matrix_regularized,
            'corr_matrix': corr_matrix
        }

    def _compute_weights(self,
                         estimators: Dict[str, Any],
                         signals: pd.Series,
                         current_weights: pd.Series
                         ) -> pd.Series:
        """
        The transform step: Solves the Max Diversification problem and scales the weights.

        Parameters
        ----------
        estimators : Dict[str, Any]
            Contains the calculated covariance matrix and asset info.
        signals : pd.Series
            The strategy's directional view for the period.
        current_weights : pd.Series
            The weights currently held.

        Returns
        -------
        pd.Series
            The target weights for the period.
        """
        # Extract estimators
        n_assets = estimators['n_assets']
        asset_names = estimators['assets']
        cov_matrix_np = estimators['cov_matrix_np']
        corr_matrix = estimators['corr_matrix']

        # --- 1. Define Optimization Variables ---
        weights = cp.Variable(n_assets)

        # --- 2. Define Objective Function ---
        portfolio_corr = cp.quad_form(weights, corr_matrix)
        objective = cp.Minimize(portfolio_corr)

        # --- 3. Define Constraints ---
        constraints = [
            cp.sum(weights) <= self.budget,
            weights <= self.max_weight,
            weights >= self.min_weight
        ]

        # --- 4. Solve Problem ---
        try:
            prob = cp.Problem(objective, constraints)
            prob.solve(solver=self.solver)

            # extract val
            w_prime = weights.value

            if w_prime is None or 'optimal' not in prob.status:
                raise Exception("Solver failed to find an optimal solution.")

            # get asset vol (daily)
            asset_daily_variance = np.diag(cov_matrix_np)
            asset_daily_volatility = np.sqrt(asset_daily_variance)

            # max diversification scaling factor: W_scaled_i = W'_i / sigma_i
            # scale by the inverse of their daily volatility.
            md_weights_scaled_raw = w_prime / asset_daily_volatility

            # normalize the scaled weights
            sum_of_scaled_abs = np.sum(np.abs(md_weights_scaled_raw))

            if sum_of_scaled_abs == 0:
                raise Exception("Sum of scaled MD weights is zero, cannot normalize.")

            # normalize to 1.0 gross leverage and apply final budget
            optimal_weights = (md_weights_scaled_raw / sum_of_scaled_abs) * self.budget

        except Exception as e:
            warn(f"{self.name} failed to solve or scale. Error: {e}")
            # fallback to equal weighting if solver or scaling fails
            optimal_weights = np.ones(n_assets) * (self.budget / n_assets)

        # max diversification weights
        md_weights = pd.Series(optimal_weights, index=asset_names)

        # signal application
        target_weights = signals.multiply(md_weights, axis=0)

        return target_weights
