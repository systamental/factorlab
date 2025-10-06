from __future__ import annotations
import pandas as pd
from typing import List, Union

from factorlab.factors.trend.base import TrendFactor
from factorlab.signal_generation.time_series_analysis import TimeSeriesAnalysis as TSA


class AlphaMomentum(TrendFactor):
    """
    Constant term/coefficient (alpha) from fitting an OLS linear regression of price on the market portfolio (beta,
    i.e. cross-sectional average of returns).
    """
    def __init__(self,
                 return_col: str = 'ret',
                 factor_cols: Union[str, List[str]] = 'market',
                 scale: bool = False,
                 **kwargs):
        super().__init__(scale=scale, **kwargs)
        self.name = 'AlphaMomentum'
        self.description = ('Alpha Momentum factor computes alpha term via rolling linear regression '
                            'of price on market returns.')
        self.return_col = return_col
        self.factor_cols = factor_cols if isinstance(factor_cols, list) else [factor_cols]

    @property
    def inputs(self) -> List[str]:
        """
        Required input columns.
        Override in subclasses as needed.
        """
        return [self.return_col] + self.factor_cols

    def _compute_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes the alpha momentum signal.
        """
        # fit linear regression
        alpha = TSA(df[self.return_col],
                    df[self.factor_cols],
                    trend='c',
                    window_type='rolling',
                    window_size=self.window_size).linear_regression(output='params')

        # alpha
        trend_df = alpha[['const']].copy()
        trend_df.rename(columns={'const': 'trend'}, inplace=True)

        return trend_df
