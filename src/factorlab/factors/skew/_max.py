import pandas as pd
import numpy as np
from factorlab.utils import grouped, maybe_droplevel
from typing import Optional

from factorlab.factors.skew.base import SkewFactor


class Max(SkewFactor):
    """
    Computes max return over a rolling window.

    Extreme-return measures are widely used as proxies for skewness and lottery-like
    behavior in asset returns. For example, Bali, Cakici & Whitelaw (2011) show that
    MAX (the single largest return in a rolling window) predicts the cross-section
    of stock returns.

    Parameters
    ----------
    n_top : int, optional
        Number of top returns to average for the calculation.

    Notes
    -----
    Instance variables match the parameters. Use the class constructor
    to set them, and access them directly via attributes if needed.
    """

    def __init__(self,
                 n_top: Optional[int] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.name = 'Max'
        self.description = 'Skewness factor calculated using the maximum of asset returns.'
        self.n_top = n_top

    def _compute_skew(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes the extreme-return skew factor.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame containing the required return column.

        Returns
        -------
        pd.DataFrame
            A DataFrame with the computed extreme-return factor.
        """
        # Group the DataFrame by asset identifier
        g = grouped(df)

        # Compute the rolling maximum
        if self.n_top is not None and self.n_top > 1:
            # Compute the average of the top n returns in the rolling window
            max_df = g.rolling(self.window_size).apply(lambda x: np.mean(np.sort(x)[-self.n_top:]), raw=False)
        else:
            max_df = g.rolling(self.window_size).max()

        # Drop the first level of the MultiIndex
        max_df = maybe_droplevel(max_df, level=0)

        # rename the column to 'skew'
        max_df['skew'] = max_df[self.return_col]

        return max_df[['skew']]
