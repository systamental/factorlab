import pandas as pd
from typing import Dict, Any

from factorlab.portfolio_opt.base import PortfolioOptimizerBase
from factorlab.portfolio_opt.estimators.risk import RiskMetrics


class EqualWeighted(PortfolioOptimizerBase):
    """
    Portfolio Optimizer implementing the Equal Weighted (1/N) strategy.

    Each asset receives an equal allocation regardless of historical returns or risk.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.name = "EqualWeighted"
        self.description = "Allocates equal weight to all assets (1/N)."

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

        # assets and date
        return {
            'assets': returns.dropna(axis=1, how='all').columns.tolist(),
            'cov_matrix': cov_matrix
        }

    def _compute_weights(self,
                         estimators: Dict[str, Any],
                         signals: pd.Series,
                         current_weights: pd.Series  # required by interface but unused here
                         ) -> pd.Series:
        """
        Computes the directional target weights based on equal weighting.

        Parameters
        ----------
        estimators : Dict[str, Any]
            Dictionary containing estimators calculated from historical returns.
        signals : pd.Series
            The strategy's directional view for the period.
        current_weights : pd.Series
            The weights currently held.

        Returns
        -------
        pd.Series
            The target weights for the period.
        """
        # estimators dict
        assets = estimators['assets']

        if not assets:
            raise ValueError("Asset list not found in estimators.")

        # number of assets
        num_assets = len(assets)
        ew = 1.0 / num_assets

        # ew series
        ew = pd.Series(ew, index=assets)

        # target weights
        target_weights = signals.multiply(ew, axis=0)

        return target_weights.reindex(assets)
