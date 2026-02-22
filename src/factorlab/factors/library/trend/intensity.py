from __future__ import annotations
import pandas as pd
from typing import List
from factorlab.factors.trend.base import TrendFactor


class Intensity(TrendFactor):
    """
    Computes intraday intensity trend factor.
    """
    def __init__(self,
                 open_col: str = 'open',
                 high_col: str = 'high',
                 low_col: str = 'low',
                 close_col: str = 'close',
                 smooth: bool = True,
                 vwap: bool = False,
                 scale: bool = False,
                 **kwargs):
        """
        Constructor.

        Parameters
        ----------
        smooth: bool, default True
            Whether to apply smoothing to the price series.
        vwap: bool, default False
            Whether to apply VWAP transformation to the price series.

        """
        super().__init__(smooth=smooth, vwap=vwap, scale=scale, **kwargs)
        self.name = 'Intensity'
        self.description = 'Intraday Intensity trend factor.'
        self.open_col = open_col
        self.high_col = high_col
        self.low_col = low_col
        self.close_col = close_col

    @property
    def inputs(self) -> List[str]:
        """
        Required input columns.
        Override in subclasses as needed.
        """
        return [self.open_col, self.high_col, self.low_col, self.close_col]

    def _compute_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute the intraday intensity trend factor.
        """
        # compute true range
        hilo = df[self.high_col] - df[self.low_col]
        if isinstance(df.index, pd.MultiIndex):
            hicl = abs(df[self.high_col].sort_index(level=1) - df[self.close_col].groupby(level=1).shift(1))
            locl = abs(df[self.low_col].sort_index(level=1) - df[self.close_col].groupby(level=1).shift(1))
        else:
            hicl = abs(df[self.high_col] - df[self.close_col].shift(1))
            locl = abs(df[self.low_col] - df[self.close_col].shift(1))
        tr = pd.concat([hilo, hicl, locl], axis=1).max(axis=1)

        # normalized chg
        chg = df[self.close_col] - df[self.open_col]
        trend_df = chg / tr

        # convert to DataFrame
        trend_df = trend_df.to_frame('trend')

        return trend_df[['trend']]
