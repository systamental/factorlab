from __future__ import annotations
import pandas as pd

from factorlab.factors.reversal.base import ReversalFactor
from factorlab.transformations.smoothing import WindowSmoother
from factorlab.utils import to_dataframe


class MWDeviation(ReversalFactor):
    """
    Computes the moving window deviation reversal factor.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = 'MWDeviation'
        self.description = 'Moving Window Deviation reversal factor.'

    def _compute_reversal(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes the moving window deviation reversal factor.
        """
        # smoothing
        window_transform = WindowSmoother(input_cols='close',
                                          output_cols='close_smooth',
                                          window_type=self.window_type,
                                          window_size=self.window_size,
                                          central_tendency=self.central_tendency)

        # smooth df
        mw_df = window_transform.compute(df)

        # deviation
        rev_df = mw_df[self.input_col] - mw_df['close_smooth']

        # format output
        rev_df = to_dataframe(rev_df, name='reversal')

        return rev_df
