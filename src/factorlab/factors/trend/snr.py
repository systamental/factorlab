from __future__ import annotations
import pandas as pd
import numpy as np

from factorlab.factors.trend.base import TrendFactor
from factorlab.transformations.returns import Difference
from factorlab.utils import maybe_droplevel


class SNR(TrendFactor):
    """
    Computes the signal-to-noise ratio.
    """
    def __init__(self,
                 input_col: str = 'close',
                 scale=False,
                 smooth=False,
                 **kwargs):
        """
        Constructor.

        """
        super().__init__(scale=scale, smooth=smooth, **kwargs)
        self.name = 'SNR'
        self.description = 'Signal-to-noise ratio, measuring trend strength relative to volatility.'
        self.input_col = input_col

    def _compute_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes the divergence signal.
        """
        # chg
        chg = Difference(input_col=self.input_col,
                         output_col='diff',
                         lags=self.window_size).compute(df)[['diff']]

        # abs roll_chg
        if isinstance(df.index, pd.MultiIndex):
            abs_roll_chg = df.groupby(level=1).diff().abs().groupby(level=1).rolling(self.window_size).sum()
        else:
            abs_roll_chg = np.abs(df.diff()).rolling(self.window_size).sum()

        # ensure df are correct index
        abs_roll_chg = maybe_droplevel(abs_roll_chg)[[self.input_col]]
        abs_roll_chg.columns = ['diff']

        # compute snr
        trend_df = chg / abs_roll_chg

        # rename column
        trend_df['trend'] = trend_df['diff']

        return trend_df[['trend']]
