from __future__ import annotations
import pandas as pd
import numpy as np
from typing import List, Optional

from factorlab.factors.trend.base import TrendFactor
from factorlab.transformations.smoothing import WindowSmoother
from factorlab.utils import to_dataframe


class Stochastic(TrendFactor):
    """
    Computes the stochastic indicator K and D.
    
    Parameters
    ----------
    high_col : str, default 'high'
        Column name for highest price.
    low_col : str, default 'low'
        Column name for lowest price.
    close_col : str, default 'close'
        Column name for closing price.
    stochastic: str, {'k', 'd', 'all'}, default 'd'
        Stochastic to return.
    short_window_size : int, optional, default=None
        Size of the short-term smoothing window.
    long_window_size : int, optional, default=None
        Size of the long-term smoothing window.
    central_tendency: str, {'mean', 'median'}, default 'mean'
        Central tendency measure for smoothing.
    vwap: bool, default False
        Whether to apply VWAP transformation to the price series.
    signal: bool, default True
        Converts stochastic to a signal between -1 and 1.
        Typically, stochastic is normalized to between 0 and 100.
    """
    def __init__(self,
                 high_col: str = 'high',
                 low_col: str = 'low',
                 close_col: str = 'close',
                 stochastic: str = 'd',
                 short_window_size: Optional[int] = None,
                 long_window_size: Optional[int] = None,
                 central_tendency: str = 'mean',
                 vwap: bool = False,
                 log: bool = False,
                 scale: bool = False,
                 signal: bool = True,
                 **kwargs):
        super().__init__(vwap=vwap, log=log, scale=scale, **kwargs)
        
        self.name = 'Stochastic' + stochastic.upper()
        self.description = 'Stochastic (' + stochastic.upper() + ') indicator.'
        self.high_col = high_col
        self.low_col = low_col
        self.close_col = close_col
        self.stochastic = stochastic
        self.short_window_size = short_window_size
        self.long_window_size = long_window_size
        self.central_tendency = central_tendency
        self.signal = signal

    @property
    def inputs(self) -> List[str]:
        """
        Required input columns.
        Override in subclasses as needed.
        """
        return ['high', 'low', 'close']

    def _compute_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes the RSI indicator.
        """
        # short window
        if self.short_window_size is None:
            self.short_window_size = max(2, int(np.ceil(np.sqrt(self.window_size))))  # sqrt of long-term window

        # compute k
        if isinstance(df.index, pd.MultiIndex):
            num = df[self.close_col].sort_index(level=1) - \
                  df[self.low_col].groupby(level=1).rolling(self.window_size).min().values
            denom = (df[self.high_col].groupby(level=1).rolling(self.window_size).max() -
                     df[self.low_col].groupby(level=1).rolling(self.window_size).min().values).droplevel(0)
            k = num/denom

        else:
            k = (df[self.close_col] - df[self.low_col].rolling(self.window_size).min()) / \
                (df[self.high_col].rolling(self.window_size).max() - df[self.low_col].rolling(self.window_size).min())

        # clip extreme values
        k = k.clip(0, 1)
        k = to_dataframe(k, name='k')

        # smoothing
        stoch_df = WindowSmoother(input_cols='k',
                                  output_cols='d',
                                  window_type=self.window_type,
                                  window_size=self.short_window_size,
                                  central_tendency=self.central_tendency).compute(k)

        # create df
        stoch_df = stoch_df[['k', 'd']]

        # convert to signal
        if self.signal:
            stoch_df = (stoch_df * 2) - 1

        # stochastic
        trend_df = stoch_df[[self.stochastic]].sort_index()
        trend_df.columns = ['trend']

        return trend_df[['trend']]
