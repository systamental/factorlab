from __future__ import annotations
import pandas as pd
from factorlab.factors.trend.base import TrendFactor
from factorlab.transformations.returns import LogReturn


class EWMA(TrendFactor):
    """
    Computes the exponentially weighted moving average (EWMA) trend factor.
    """

    def __init__(self, smooth: bool = True, **kwargs):
        """
        Constructor.

        """
        super().__init__(smooth=smooth, **kwargs)
        self.name = 'EWMA'
        self.description = 'Exponentially Weighted Moving Average of price over a rolling window.'

    def _compute_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes the EWMA signal.
        """
        # price change
        price_transform = LogReturn(lags=1)
        trend_df = price_transform.compute(df)

        # rename column
        trend_df['trend'] = trend_df['ret']

        return trend_df[['trend']]

