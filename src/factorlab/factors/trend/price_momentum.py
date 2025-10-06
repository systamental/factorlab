from __future__ import annotations
import pandas as pd
import numpy as np
from factorlab.factors.trend.base import TrendFactor
from factorlab.transformations.returns import LogReturn


class PriceMomentum(TrendFactor):
    """
    Computes the price momentum trend factor.
    """
    def __init__(self, **kwargs):
        """
        Constructor.

        """
        super().__init__(**kwargs)
        self.name = 'PriceMomentum'
        self.description = 'Measures price momentum over a rolling window.'

    def _compute_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes the price momentum signal.
        """
        # price change
        price_transform = LogReturn(lags=self.window_size)
        trend_df = price_transform.compute(df) / np.sqrt(self.window_size)

        # rename column
        trend_df['trend'] = trend_df['ret']

        return trend_df[['trend']]
