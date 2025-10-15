import pandas as pd
from factorlab.cost_model.base import CostModelBase


class FixedCommissionModel(CostModelBase):
    """
    Calculates cost based on a fixed commission or transaction rate applied to total turnover.
    """

    def __init__(self, rate: float = 0.001):
        """
        Parameters
        ----------
        rate : float, default 0.001
            The fixed percentage cost applied to every dollar of absolute turnover (0.1%).
        """
        super().__init__()
        self.name = "FixedCommissionModel"
        self.rate = rate

    def compute_cost(self,
                     current_weights: pd.Series,
                     target_weights: pd.Series,
                     prices: pd.Series,
                     portfolio_value: float) -> float:
        """
        Calculates the absolute dollar cost of rebalancing from current_weights to target_weights.

        Parameters
        ----------
        current_weights: pd.Series
            Weights before rebalancing (W_t-1).
        target_weights: pd.Series
            Weights after optimization (W_t).
        prices: pd.Series
            The prices (or closing prices) used for execution (not used in this model).
        portfolio_value: float
            The total dollar value of the portfolio at the time of trade decision.

        Returns
        -------
        float
            The total absolute dollar cost of the transaction.
        """

        # 1. Align weights (union of all assets present in either series)
        aligned_weights = pd.concat([current_weights, target_weights], axis=1).fillna(0)
        old_w = aligned_weights.iloc[:, 0]
        new_w = aligned_weights.iloc[:, 1]

        # 2. Turnover is the absolute change in weights (buys and sells)
        # Turnover is dimensionless (a fraction of the portfolio)
        turnover_fraction = (new_w - old_w).abs().sum()

        # 3. Cost as a fraction of portfolio value
        fractional_cost = turnover_fraction * self.rate

        # 4. Convert fraction to absolute dollar cost (Crucial Step)
        dollar_cost = fractional_cost * portfolio_value

        return dollar_cost
