from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Optional

from factorlab.factors.trend.base import TrendFactor
from factorlab.transformations.smoothing import WindowSmoother
from factorlab.utils import to_dataframe


class MWDifference(TrendFactor):
    """
    Computes the moving window difference trend factor.
    """
    def __init__(self,
                 input_col: str = 'close',
                 short_window_size: Optional[int] = None,
                 long_window_size: Optional[int] = None,
                 central_tendency: str = 'mean',
                 lags: int = 0,
                 scale: bool = False,
                 smooth: bool = False,
                 **kwargs):
        """
        Constructor.

        Parameters
        ----------
        short_window_size : int, optional, default=None
            Size of the short-term smoothing window.
        long_window_size : int, optional, default=None
            Size of the long-term smoothing window.
        central_tendency: str, {'mean', 'median'}, default 'mean'
            Central tendency measure for smoothing.
        lags: int, default 0
            Number of periods to lag the long-term window.

        """
        super().__init__(scale=scale, smooth=smooth, **kwargs)
        self.name = 'MWDifference'
        self.description = 'Moving Window Difference trend factor.'
        self.input_col = input_col
        self.short_window_size = short_window_size
        self.long_window_size = long_window_size
        self.central_tendency = central_tendency
        self.lags = lags

    def _compute_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes the moving window difference trend factor.
        """
        # short rolling window param
        if self.short_window_size is None:
            self.short_window_size = int(np.ceil(np.sqrt(self.window_size)))  # sqrt of long-term window

        # smoothing
        short_window_transform = WindowSmoother(input_cols=self.input_col,
                                                window_type=self.window_type,
                                                window_size=self.short_window_size,
                                                central_tendency=self.central_tendency)

        # long rolling window param
        long_window_transform = WindowSmoother(input_cols=self.input_col,
                                               window_type=self.window_type,
                                               window_size=self.window_size,
                                               central_tendency=self.central_tendency,
                                               lags=self.lags)

        # diff
        trend_df = short_window_transform.compute(df).iloc[:, -1] - long_window_transform.compute(df).iloc[:, -1]

        # to DataFrame
        trend_df = to_dataframe(trend_df, name='trend')

        return trend_df[['trend']]
