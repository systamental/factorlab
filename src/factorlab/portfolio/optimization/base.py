import pandas as pd
import numpy as np
from abc import abstractmethod
from typing import Dict, Any

from factorlab.core.base_transform import BaseTransform


class PortfolioOptimizerBase(BaseTransform):
    """
    Abstract Base Class for all Portfolio Optimizers.

    This class enforces the fit/transform pattern required for rolling backtests
    to prevent look-ahead bias. The fit() step learns the required market moments
    (e.g., volatility, correlation) from historical data. The transform() step
    then computes the weights based on those learned moments.

    Parameters
    ----------
    window_size : int
        The lookback window size (in days) to use for estimating moments.
    risk_estimator : str
        The risk estimator method to use (e.g., 'covariance', 'ewma', 'dcc').
    risk_estimator_params : Dict[str, Any]
        Additional parameters to pass to the risk estimator function.
    vol_target : float, optional
        If provided, the final target weights will be scaled such that the
        ex-ante annual volatility of the portfolio equals this value.
        This scaling **overrides** the 'leverage' parameter if both are set.
        (e.g., 0.10 for 10% target vol).
    leverage : float
        The gross leverage constraint applied if vol_target is None.
    """

    def __init__(self,
                 window_size: int = 360,
                 risk_estimator: str = 'covariance',
                 risk_estimator_params: Dict[str, Any] = {},
                 vol_target: float = None,
                 ann_factor: int = 365,
                 leverage: float = 1.0):

        super().__init__(name="PortfolioOptimizer",
                         description="Abstract base class for portfolio optimizers.")

        self.window_size = window_size
        self.risk_estimator = risk_estimator
        self.risk_estimator_params = risk_estimator_params
        self.vol_target = vol_target
        self.ann_factor = ann_factor
        self.leverage = leverage
        self.estimators: Dict[str, Any] = {}

    @abstractmethod
    def _compute_estimators(self, returns: pd.DataFrame) -> Dict[str, Any]:
        """
        Abstract method to estimate the statistical moments (e.g., covariance matrix).

        Parameters
        ----------
        returns : pd.DataFrame
            DataFrame of historical returns used to estimate moments.
        """
        pass

    @abstractmethod
    def _compute_weights(self,
                         estimators: Dict[str, Any],
                         signals: pd.Series,
                         current_weights: pd.Series
                         ) -> pd.Series:
        """
        Abstract method to solve the optimization problem using computed estimators.
        """
        pass

    def _compute_portfolio_volatility(self, target_weights: pd.Series) -> float:
        """
        Calculates the ex-ante annualized volatility of the portfolio
        using the fitted daily covariance matrix estimator.

        Parameters
        ----------
        target_weights : pd.Series
            The target portfolio weights W_t (should sum to 1).
        """
        if 'cov_matrix' not in self.estimators:
            raise RuntimeError(
                f"{self.name}: Cannot perform volatility targeting. "
                "The subclass must provide 'cov_matrix' in self.estimators."
            )

        # ensure the weights vector aligns with the covariance matrix columns
        # indexing ensures correct ordering and subsetting
        cov_matrix = self.estimators['cov_matrix'].loc[
            target_weights.index, target_weights.index
        ]

        # portfolio daily variance: W^T * Sigma_daily * W
        daily_variance = target_weights.T @ cov_matrix @ target_weights

        # annualized volatility: sqrt(daily_Variance * annualization Factor)
        ann_var = daily_variance * self.ann_factor

        return np.sqrt(ann_var)

    def fit(self, returns: pd.DataFrame, **kwargs) -> 'PortfolioOptimizerBase':
        """
        The fit step. Learns statistical moments from the lookback window.
        """
        if returns.empty:
            raise ValueError(f"{self.name}: Input returns for fit cannot be empty.")

        # stores dict returned by the subclass's _compute_estimators
        self.estimators = self._compute_estimators(returns)
        self._is_fitted = True
        return self

    def transform(self,
                  signals: pd.Series,
                  current_weights: pd.Series
                  ) -> pd.Series:
        """
        The transform step. Uses the fitted moments to compute the target weights W_t.
        The current_data parameter is usually ignored here as weights depend on
        historical estimators, not current day data.
        """
        if not self._is_fitted:
            raise RuntimeError(f"{self.name} must be fitted before transformation.")

        target_weights = self._compute_weights(
            self.estimators,
            signals,
            current_weights
        )

        # Ensure weights sum to 1 before applying leverage or vol targeting
        target_weights /= target_weights.abs().sum()
        final_weights = target_weights.copy()

        # vol targeting
        if self.vol_target is not None:
            try:
                # Calculate the ex-ante annualized volatility of the normalized portfolio
                current_vol = self._compute_portfolio_volatility(target_weights)

                if current_vol > 1e-6:  # Check to avoid division by near zero
                    # Calculate the required scaling factor
                    scale_factor = self.vol_target / current_vol
                    # Scale the weights
                    final_weights = target_weights * scale_factor
                else:
                    # Fallback if volatility is negligible
                    print(
                        f"Warning: Current portfolio volatility is near zero. Falling back to fixed leverage scaling.")
                    final_weights = target_weights * self.leverage

            except RuntimeError as e:
                # Handles the case where the covariance matrix is missing (subclass error)
                print(f"Warning: {e}. Cannot scale by vol target. Falling back to fixed leverage scaling.")
                final_weights = target_weights * self.leverage
            except Exception as e:
                # Handles other potential errors (e.g., matrix mismatch, bad data)
                print(f"Error during volatility scaling: {e}. Falling back to fixed leverage scaling.")
                final_weights = target_weights * self.leverage

        # 4. Apply Simple Leverage (only if vol_target was not set)
        elif self.leverage != 1.0:
            final_weights = target_weights * self.leverage

        return final_weights
