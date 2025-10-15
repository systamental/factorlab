import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import cvxpy as cp
from warnings import warn

from factorlab.portfolio_opt.base import PortfolioOptimizerBase
from factorlab.portfolio_opt.estimators.risk import RiskMetrics


class RiskParity(PortfolioOptimizerBase):
    """
    Solves the Equal Risk Contribution (ERC) portfolio problem.

    The ERC strategy aims to allocate capital such that each asset contributes
    equally to the total portfolio volatility (risk budget).

    This implementation uses the standard convex Log-Barrier optimization
    (Spinu, 2013) which minimizes a proxy objective known to converge to the
    true ERC solution.
    """

    def __init__(self,
                 max_weight: float = 1.0,
                 min_weight: float = 0.0,
                 budget: float = 1.0,
                 risk_estimator: str = 'ledoit_wolf',
                 solver: Optional[str] = None,
                 **kwargs):
        """
        Initializes the Risk Parity (ERC) optimizer.

        Note: The ERC formulation requires long-only weights (w_i > 0) due to
        the log-barrier term. While min_weight is accepted, setting it below
        a very small positive number (e.g., 1e-6) may lead to solver failure.

        Parameters
        ----------
        max_weight: float, default 1.0
            Maximum weight constraint per asset.
        min_weight: float, default 0.0
            Minimum weight constraint per asset. Must be >= 0 for the log-barrier.
        budget: float, default 1.0
            The total sum of weights.
        risk_estimator: str, default 'ledoit_wolf'
            The name of the method from RiskMetrics to use for covariance matrix estimation.
        solver: str, default 'CLARABEL'
            The CVXPY solver to use for the quadratic optimization problem.
        """
        # Pass base parameters to parent class
        super().__init__(**kwargs)

        self.name = "RiskParityOptimizer"
        self.description = "Equal Risk Contribution (ERC) optimizer."

        # ERC specific parameters
        # We enforce min_weight >= 1e-6 to ensure log(w) is defined and positive
        self.min_weight = max(min_weight, 1e-6)
        self.max_weight = max_weight
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
        This is directly copied from the MinVol implementation as it is good practice.
        """
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix_np)
        except np.linalg.LinAlgError:
            warn("Covariance matrix is numerically singular. Cannot perform Eigendecomposition.")
            return cov_matrix_np + np.diag(np.ones(cov_matrix_np.shape[0]) * min_eigenvalue)

        eigenvalues[eigenvalues < min_eigenvalue] = min_eigenvalue
        regularized_cov = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        regularized_cov = (regularized_cov + regularized_cov.T) / 2

        return regularized_cov

    def _compute_estimators(self, returns: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculates the Covariance Matrix (Sigma) required for the risk component.
        """
        # Drop rows/columns with all missing data
        returns = returns.dropna(how='all').dropna(how='any', axis=1)

        returns = RiskMetrics._validate_returns(returns)
        self._n_assets = returns.shape[1]
        self._asset_names = returns.columns.tolist()

        # Dynamically calculate the Covariance Matrix (Sigma)
        try:
            risk_func = getattr(RiskMetrics, self.risk_estimator)
        except AttributeError:
            raise ValueError(f"RiskMetrics has no method named '{self.risk_estimator}'.")

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
        Solves the Log-Barrier minimization problem to find Equal Risk Contribution (ERC) weights.

        The `signals` input is used afterward to apply the final directional
        (long/short) scaling to the risk-optimized magnitude.

        Parameters
        ----------
        estimators : Dict[str, Any]
            Dictionary containing estimators calculated from historical returns.
        signals : pd.Series
            The strategy's directional view for the period.
        current_weights : pd.Series
            The weights currently held (not used in this optimizer).

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
        # The variables represent the MAGNITUDE of the weights (must be positive)
        weights = cp.Variable(n_assets)

        # --- 2. Define Objective Function (Log-Barrier Minimization) ---
        # Objective: min 0.5 * (W^T * Sigma * W) - sum(log(W))
        portfolio_risk_sq = cp.quad_form(weights, cov_matrix)
        log_barrier = cp.sum(cp.log(weights))

        # The objective is convex and is known to enforce ERC.
        objective = cp.Minimize(0.5 * portfolio_risk_sq - log_barrier)

        # --- 3. Define Constraints ---
        constraints = [
            cp.sum(weights) == self.budget,  # Must be fully invested
            weights >= self.min_weight,  # Minimum weight (must be > 0)
            weights <= self.max_weight  # Maximum weight
        ]

        # --- 4. Solve Problem ---
        try:
            prob = cp.Problem(objective, constraints)
            # Use 'CLARABEL' as it is a common, reliable solver for QCQP problems
            prob.solve(solver=self.solver)
        except Exception as e:
            warn(f"{self.name} failed to solve with {self.solver}. Error: {e}")

        # extract weights
        optimal_weights = weights.value

        if optimal_weights is None or 'optimal' not in prob.status:
            warn("Optimization failed or returned None. Falling back to equal weights.")
            # Fallback to equal weighting if solver fails
            optimal_weights = np.ones(n_assets) * (self.budget / n_assets)

        # risk parity weights as a pd.Series
        rp_weights = pd.Series(optimal_weights, index=asset_names)

        # signals applied
        target_weights = signals.multiply(rp_weights, axis=0)

        return target_weights
