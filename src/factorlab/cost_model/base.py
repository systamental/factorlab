import pandas as pd
from abc import ABC, abstractmethod


class CostModelBase(ABC):
    """
    Abstract Base Class for calculating transaction costs (slippage, commission, etc.).

    This class defines the interface for all transaction cost models used in the
    backtesting engine. It does not inherit from BaseTransform as cost calculation
    is a concurrent, stateless utility function applied at the point of trade execution.
    """

    def __init__(self, name: str = "CostModelBase"):
        self.name = name

    @abstractmethod
    def compute_cost(self,
                     current_weights: pd.Series,
                     target_weights: pd.Series,
                     prices: pd.Series,
                     portfolio_value: float) -> float:
        """
        Calculates the fractional transaction cost based on turnover between
        the old and new portfolio weights.

        Parameters
        ----------
        current_weights : pd.Series
            Weights held at the end of the previous period.
        target_weights : pd.Series
            Target weights determined by the optimizer for the current period.
        prices : pd.Series
            The prices (or closing prices) used for execution (needed for
            slippage or price-dependent cost models).
        portfolio_value : float
            The total dollar value of the portfolio at the time of trade decision.

        Returns
        -------
        float
            The total absolute dollar cost of the transaction, computed as the total fractional cost incurred
            (e.g., 0.001 for 0.1% cost) multiplied by the portfolio value.
        """
        raise NotImplementedError
