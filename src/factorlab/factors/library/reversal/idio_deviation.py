from __future__ import annotations
import pandas as pd

from factorlab.factors.reversal.base import ReversalFactor
from factorlab.features.residuals import Residuals


class IdiosyncraticDeviation(ReversalFactor):
    """
    Computes the linear deviation reversal factor.
    """
    def __init__(self,
                 return_col: str = 'ret',
                 factor_col: str = 'market',
                 vwap=False,
                 **kwargs):

        super().__init__(vwap=vwap, **kwargs)
        self.name = 'IdiosyncraticDeviation'
        self.description = 'Idiosyncratic Deviation reversal factor.'
        self.return_col = return_col
        self.factor_col = factor_col

    def _compute_reversal(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes the linear deviation reversal factor.
        """
        ret = df[[self.return_col, self.factor_col]].copy()
        ret = ret.cumsum()

        # idiosyncratic returns
        resid = Residuals(target_col=self.return_col,
                         feature_col=self.factor_col,
                         model='linear',
                         window_type='rolling',
                         window_size=self.window_size).compute(ret)

        return resid
