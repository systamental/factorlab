from __future__ import annotations
import pandas as pd
from factorlab.factors.trend.base import TrendFactor
from factorlab.signal_generation.time_series_analysis import TimeSeriesAnalysis as TSA


class TimeTrend(TrendFactor):
    """
    Computes the time trend factor by regressing price on a constant and time trend to estimate coefficients.

    Parameters
    ----------
    target_col: str, default 'close'
        Column name for price data used in regression.
    """
    def __init__(self,
                 target_col: str = 'close',
                 scale: bool = False,
                 **kwargs):
        super().__init__(scale=scale, **kwargs)
        self.name = 'TimeTrend'
        self.description = 'Time Trend factor computed via rolling linear regression of price on time.'
        self.target_col = target_col

    def _compute_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes the time trend signal.
        """
        # fit linear regression
        coeff = TSA(df[[self.target_col]], trend='ct', window_type='rolling',
                    window_size=self.window_size).linear_regression(output='params')

        # time trend
        trend_df = coeff[['trend']].copy()

        # rename column
        trend_df.rename(columns={'trend': 'trend'}, inplace=True)

        return trend_df[['trend']]
