from __future__ import annotations
import pandas as pd
from typing import List, Union

from factorlab.transformations.returns import Difference
from factorlab.factors.trend.base import TrendFactor
from factorlab.features.residuals import IdiosyncraticReturns


class IdiosyncraticEWMA(TrendFactor):
    """
    Exponential weighted moving average of idiosyncratic returns from a linear regression of asset returns
    on the market portfolio.

    Parameters
    ----------
    return_col: str, default 'ret'
        Column name for asset returns.
    factor_cols: list of str, default ['market']
        List of column names for factor returns (e.g. market return).
    smooth: bool, default True
        If True, applies smoothing to the computed trend.
    """
    def __init__(self,
                 return_col: str = 'ret',
                 factor_cols: Union[str, List[str]] = 'market',
                 smooth: bool = True, **kwargs):
        """
        Constructor.

        """
        super().__init__(smooth=smooth, **kwargs)
        self.name = 'IdiosyncraticEWMA'
        self.description = ('Idiosyncratic EWMA factor computes EWMA of idiosyncratic returns from '
                            'a linear regression of asset returns on the market portfolio.')
        self.return_col = return_col
        self.factor_cols = factor_cols if isinstance(factor_cols, list) else [factor_cols]

    @property
    def inputs(self) -> List[str]:
        """Required input columns."""
        return [self.return_col] + self.factor_cols

    def _compute_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes the alpha momentum signal.
        """
        # idiosyncratic returns
        trend_df = IdiosyncraticReturns(return_col=self.return_col,
                                        factor_cols=self.factor_cols,
                                        model='linear',
                                        incl_alpha=True,
                                        window_type='rolling',
                                        window_size=self.window_size).compute(df)
        # rename column
        trend_df.rename(columns={'idio_ret': 'trend'}, inplace=True)

        return trend_df[['trend']]
