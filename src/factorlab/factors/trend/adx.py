from __future__ import annotations
import pandas as pd
import numpy as np
from typing import List
from factorlab.factors.trend.base import TrendFactor
from factorlab.transformations.smoothing import WindowSmoother


class ADX(TrendFactor):
    """
    Computes the average directional index (ADX) of a price series, with optional signal conversion.
    """
    def __init__(self,
                 open_col: str = 'open',
                 high_col: str = 'high',
                 low_col: str = 'low',
                 close_col: str = 'close',
                 scale: bool = False,
                 smooth: bool = False,
                 central_tendency: str = 'mean',
                 signal: bool = True,
                 **kwargs):
        """
        Constructor.

        Parameters
        ----------
        open_col: str, default 'open'
            Column name for opening price.
        high_col: str, default 'high'
            Column name for highest price.
        low_col: str, default 'low'
            Column name for lowest price.
        close_col: str, default 'close'
            Column name for closing price.
        scale: bool, default False
            Whether to scale the trend values.
        smooth: bool, default False
            Whether to apply smoothing to the price series.
        central_tendency: str, {'mean', 'median'}, default 'mean'
            Central tendency measure for smoothing.
        signal: bool, default True
            Converts adx values to a signal in the range [-1, 1] if True.

        """
        super().__init__(scale=scale, smooth=smooth, **kwargs)
        self.name = 'ADX'
        self.description = 'Average Directional Index (ADX) Trend Factor'
        self.open_col = open_col
        self.high_col = high_col
        self.low_col = low_col
        self.close_col = close_col
        self.central_tendency = central_tendency
        self.signal = signal

    @property
    def inputs(self) -> List[str]:
        """
        Required input columns.
        Override in subclasses as needed.
        """
        return [self.open_col, self.high_col, self.low_col, self.close_col]

    def _compute_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes the ADX indicator.
        """
        # compute true range
        hilo = df[self.high_col] - df[self.low_col]

        if isinstance(df.index, pd.MultiIndex):
            hicl = abs(df[self.high_col].sort_index(level=1) - df[self.close_col].groupby(level=1).shift(1))
            locl = abs(df[self.low_col].sort_index(level=1) - df[self.close_col].groupby(level=1).shift(1))
        else:
            hicl = abs(df[self.high_col] - df[self.close_col].shift(1))
            locl = abs(df[self.low_col] - df[self.close_col].shift(1))
        tr = pd.concat([hilo, hicl, locl], axis=1).max(axis=1).to_frame('tr').sort_index()

        # high and low price change
        if isinstance(df.index, pd.MultiIndex):
            high_chg = df[self.high_col].groupby(level=1).diff()
            low_chg = df[self.low_col].groupby(level=1).shift(1) - df[self.low_col].groupby(level=1).shift(0)
        else:
            high_chg = df[self.high_col].diff()
            low_chg = df[self.low_col].shift(1) - df[self.low_col]

        # compute +DM and -DM
        dm_pos = pd.DataFrame(np.where(high_chg > low_chg, high_chg, 0), index=df.index, columns=['dm_pos'])
        dm_neg = pd.DataFrame(np.where(low_chg > high_chg, low_chg, 0), index=df.index, columns=['dm_neg'])

        # compute directional movement index
        dm_pos = WindowSmoother(input_cols='dm_pos',
                                output_cols='dm_pos_smoothed',
                                window_size=self.window_size,
                                window_type=self.window_type,
                                central_tendency=self.central_tendency).compute(dm_pos)[['dm_pos_smoothed']]
        dm_neg = WindowSmoother(input_cols='dm_neg',
                                output_cols='dm_neg_smoothed',
                                window_size=self.window_size,
                                window_type=self.window_type,
                                central_tendency=self.central_tendency).compute(dm_neg)[['dm_neg_smoothed']]
        tr = WindowSmoother(input_cols='tr',
                            output_cols='tr_smoothed',
                            window_size=self.window_size,
                            window_type=self.window_type,
                            central_tendency=self.central_tendency).compute(tr)[['tr_smoothed']]

        # compute directional index
        di_pos = 100 * dm_pos.div(tr.squeeze(), axis=0)
        di_neg = 100 * dm_neg.div(tr.squeeze(), axis=0)

        # compute directional index difference
        di_diff = di_pos.subtract(di_neg.squeeze(), axis=0)
        di_sum = di_pos.add(di_neg.squeeze(), axis=0)
        trend_df = 100 * di_diff.div(di_sum.squeeze(), axis=0)

        # compute ADX
        if self.signal:
            adx = WindowSmoother(input_cols=trend_df.columns[0],
                                 output_cols='trend',
                                 window_size=self.window_size,
                                 window_type=self.window_type,
                                 central_tendency=self.central_tendency).compute(trend_df)
            trend_df = (adx / 100).clip(-1, 1)
        else:
            adx = WindowSmoother(input_cols=trend_df.columns[0],
                                 output_cols='trend',
                                 window_size=self.window_size,
                                 window_type=self.window_type,
                                 central_tendency=self.central_tendency).compute(trend_df.abs())
            trend_df = adx

        return trend_df
