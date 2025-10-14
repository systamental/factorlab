import pandas as pd
from typing import Dict, Any

from factorlab.portfolio_opt.base import PortfolioOptimizerBase


class EqualWeighted(PortfolioOptimizerBase):
    """
    Portfolio Optimizer implementing the Equal Weighted (1/N) strategy.

    Each asset receives an equal allocation regardless of historical returns or risk.
    """

    def __init__(self, window_size: int = 60, **kwargs):
        super().__init__(window_size=window_size, **kwargs)

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

        # assets and date
        return {
            'assets': returns.dropna(axis=1, how='all').columns.tolist(),
        }

    def _compute_weights(self,
                         estimators: Dict[str, Any],
                         signals: pd.Series,
                         current_weights: pd.Series  # required by interface but unused here
                         ) -> pd.DataFrame:
        """
        Calculates the 1/N weights.
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
