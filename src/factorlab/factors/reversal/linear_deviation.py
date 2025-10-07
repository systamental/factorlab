from __future__ import annotations
import pandas as pd

from factorlab.factors.reversal.base import ReversalFactor
from factorlab.signal_generation.time_series_analysis import TimeSeriesAnalysis as TSA


class LinearDeviation(ReversalFactor):
    """
    Computes the linear deviation reversal factor.
    """
    def __init__(self, window_type='rolling', **kwargs):
        super().__init__(window_type=window_type, **kwargs)
        self.name = 'LinearDeviation'
        self.description = 'Linear Deviation reversal factor.'

    def _compute_reversal(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes the linear deviation reversal factor.
        """
        resid = TSA(df,
                    trend='ct',
                    window_type=self.window_type,
                    window_size=self.window_size).linear_regression(output='resid')

        # rename column
        resid.columns = ['reversal']

        return resid
