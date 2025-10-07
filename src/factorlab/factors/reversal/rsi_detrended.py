from __future__ import annotations
import pandas as pd

from factorlab.factors.reversal.base import ReversalFactor
from factorlab.factors.trend import RSI
from factorlab.features.residuals import Residuals


class RSIDetrended(ReversalFactor):
    """
    Computes the detrended RSI reversal factor.

    Parameters
    ----------
    short_window : int, default 14
        The short window size for RSI calculation.
    long_window : int, default 28
        The long window size for RSI calculation.
    """
    def __init__(self,
                 short_window: int = 7,
                 long_window: int = 30,
                 vwap=False,
                 log=False,
                 scale=False,
                 **kwargs):
        super().__init__(vwap=vwap, log=log, scale=scale, **kwargs)
        self.name = 'DetrendedRSI'
        self.description = 'Detrended RSI reversal factor.'
        self.short_window = short_window
        self.long_window = long_window

    def _compute_reversal(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes the detrended RSI reversal factor.
        """
        rsi_short = RSI(window_type=self.window_type,
                        window_size=self.short_window,
                        central_tendency=self.central_tendency,
                        vwap=False).compute(df).iloc[:, -1]
        rsi_long = RSI(window_type=self.window_type,
                       window_size=self.long_window,
                       central_tendency=self.central_tendency,
                       vwap=False).compute(df).iloc[:, -1]
        rsi = pd.concat([rsi_short, rsi_long], axis=1)

        resid = Residuals(target_col=rsi.columns[0],
                          feature_col=rsi.columns[1],
                          model='linear',
                          window_type='rolling',
                          window_size=self.window_size).compute(rsi)

        return resid
