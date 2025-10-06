from __future__ import annotations
import pandas as pd
from factorlab.factors.trend.base import TrendFactor
from factorlab.signal_generation.time_series_analysis import TimeSeriesAnalysis as TSA


class PriceAcceleration(TrendFactor):
    """
    Computes the price acceleration factor by regressing price on a constant, time trend
    and time trend squared to estimate coefficients.

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
        self.name = 'PriceAcceleration'
        self.description = ('Price Acceleration factor computed via rolling linear regression of price on time '
                            'and time squared.')
        self.target_col = target_col

    def _compute_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes the price acceleration signal.
        """
        # fit linear regression
        coeff = TSA(df[[self.target_col]], trend='ctt', window_type='rolling',
                    window_size=self.window_size).linear_regression(output='params')

        # price acceleration
        trend_df = coeff[['trend_squared']].copy()

        # rename column
        trend_df.rename(columns={'trend_squared': 'trend'}, inplace=True)

        return trend_df[['trend']]
