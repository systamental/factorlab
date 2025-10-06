from __future__ import annotations
import pandas as pd
import numpy as np

from factorlab.factors.trend.base import TrendFactor
from factorlab.transformations.returns import Difference


class Divergence(TrendFactor):
    """
    Computes the divergence trend factor.
    """
    def __init__(self,
                 input_col: str = 'close',
                 output_col: str = 'trend',
                 smooth=True,
                 scale=False,
                 **kwargs):
        """
        Constructor.

        """
        super().__init__(smooth=smooth, scale=scale, **kwargs)
        self.name = 'Divergence'
        self.description = 'Exponentially Weighted Moving Average of sign of price changes over a rolling window.'
        self.input_col = input_col
        self.output_col = output_col

    def _compute_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes the divergence signal.
        """
        # price change
        price_transform = Difference(lags=1)
        trend_df = price_transform.compute(df)
        trend_df = np.sign(trend_df)

        trend_df[self.output_col] = trend_df['diff']

        return trend_df[[self.output_col]]
