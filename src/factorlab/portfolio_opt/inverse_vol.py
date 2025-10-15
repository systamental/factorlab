import pandas as pd
import numpy as np
from typing import Dict, Any

from factorlab.portfolio_opt.base import PortfolioOptimizerBase
from factorlab.portfolio_opt.estimators.risk import RiskMetrics


class InverseVolatility(PortfolioOptimizerBase):
    """
    Implements a risk-scaled portfolio optimizer where weights are inversely
    proportional to volatility, and then scaled by the trading signal (S_t).

    The final target weights W*_t are calculated as:
    W*_t = S_t * ( (1 / sigma_t) / sum(1 / sigma_t) )
    The base class handles the final normalization to the specified leverage.
    """

    def __init__(self, window_size: int = 360, **kwargs):
        super().__init__(window_size=window_size, **kwargs)

        self.name = "InverseVolatility"
        self.description = "Portfolio optimizer using inverse volatility weighting."

    def _compute_estimators(self, returns: pd.DataFrame) -> Dict[str, Any]:
        """
        Estimates the annualized volatility (sigma) for each asset from the
        historical returns provided by the Backtester.

        Parameters
        ----------
        returns : Dict[str, Any]
            Dictionary containing estimators calculated from historical returns.
        """
        # drop missing returns for covariance calculation
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
                         estimators: Dict[str, Any],
                         signals: pd.Series,
                         current_weights: pd.Series  # required by interface but unused here
                         ) -> pd.Series:
        """
        Computes the directional target weights based on the signal and
        inverse volatility risk parity.

        W_t = S_t * W_IV, where W_IV is the normalized inverse volatility weight.

        Parameters
        ----------
        estimators : Dict[str, Any]
            Contains the calculated risk Series.
        signals : pd.Series (S_t)
            The strategy's directional view for the period.
        current_weights : pd.Series (W*_t-1)
            The weights currently held.

        Returns
        -------
        pd.Series
            The target weights for the period.
        """
        # estimators dict
        assets = estimators['assets']
        cov_matrix = estimators['cov_matrix']

        # inverse vol
        sqrt_diag = np.sqrt(np.diag(cov_matrix))
        sqrt_diag = np.where(sqrt_diag == 0, np.nan, sqrt_diag)  # avoid division by zero
        ivp = 1. / sqrt_diag
        # normalize
        ivp /= np.nansum(ivp)

        # weight series
        ivp = pd.Series(ivp, index=assets)

        # target weights
        target_weights = signals.multiply(ivp, axis=0)

        return target_weights.reindex(assets)
