import pandas as pd
import numpy as np
from typing import Dict, Any

from factorlab.portfolio_opt.base import PortfolioOptimizerBase
from factorlab.portfolio_opt.estimators.risk import RiskMetrics


class SignalWeighted(PortfolioOptimizerBase):
    """
    Implements a signal-weighted portfolio optimizer where weights are directly
    proportional to the trading signal (S_t).

    The final target weights W*_t are calculated as:
    W*_t = S_t / sum(|S_t|)
    """

    def __init__(self, window_size: int = 360, **kwargs):
        super().__init__(window_size=window_size, **kwargs)

        self.name = "SignalWeighted"
        self.description = "Portfolio optimizer using signal-weighted allocation."

    def _compute_estimators(self, returns: pd.DataFrame) -> Dict[str, Any]:
        """
        Gathers necessary information (asset names, date) for the weight calculation.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the asset columns and the last date.
        """
        # drop missing returns to get valid asset list
        returns = returns.dropna(axis=1, how='all')

        # risk estimator function
        try:
            risk_func = getattr(RiskMetrics, self.risk_estimator)
        except AttributeError:
            raise ValueError(
                f"RiskMetrics has no method named '{self.risk_estimator}'. "
                "Check available estimators."
            )

        # Execute the chosen risk estimator function
        cov_matrix = risk_func(returns, **self.risk_estimator_params)

        return {
            'assets': returns.columns.tolist(),
            'cov_matrix': cov_matrix
        }

    def _compute_weights(self,
                         estimators: Dict[str, Any],  # required by interface but unused here
                         signals: pd.Series,
                         current_weights: pd.Series  # required by interface but unused here
                         ) -> pd.DataFrame:
        """
        Computes the signal-weighted portfolio weights.

        Parameters
        ----------
        estimators : Dict[str, Any]
            Contains the calculated risk Series (not used in this strategy).
        signals : pd.Series
            The cross-sectional signal vector S_t.
        current_weights : pd.Series
            The current portfolio weights W*_t-1 (not used in this strategy).

        Returns
        -------
        pd.Series
            The target weights for the period.
        """
        # estimators dict
        assets = estimators['assets']

        if not assets:
            raise ValueError("Asset list not found in estimators.")

        # target weights proportional to signal magnitude
        target_weights = signals / np.abs(signals).sum()

        return target_weights.reindex(assets)
