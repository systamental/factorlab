import pandas as pd
from abc import abstractmethod
from typing import Dict, Any, Union

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
    """

    def __init__(self, window_size: int = 60, leverage: float = 1.0):

        super().__init__(name="PortfolioOptimizer",
                         description="Abstract base class for portfolio optimizers.")

        self.window_size = window_size
        self.leverage = 1.0  # Default leverage, can be overridden in subclasses
        self.estimators: Dict[str, Any] = {}
        self.weights: pd.DataFrame = pd.DataFrame()

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

    def fit(self, returns: pd.DataFrame, **kwargs) -> 'PortfolioOptimizerBase':
        """
        The fit step. Learns statistical moments from the lookback window.
        """
        if returns.empty:
            raise ValueError(f"{self.name}: Input returns for fit cannot be empty.")

        # Stores the dictionary returned by the subclass's _compute_estimators
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

        # Scale to the desired leverage
        target_weights *= self.leverage

        return target_weights
