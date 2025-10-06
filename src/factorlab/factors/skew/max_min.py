import pandas as pd
import numpy as np
from typing import Optional
from factorlab.utils import grouped, maybe_droplevel

from factorlab.factors.skew.base import SkewFactor


class MaxMin(SkewFactor):
    """
    Computes max - abs(min) return over a rolling window.

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
        self.name = 'MaxMin'
        self.description = 'Skewness factor calculated using the max - abs(min) of asset returns.'
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
            max_ret = g.rolling(self.window_size).apply(lambda x: np.mean(np.sort(x)[-self.n_top:]), raw=False)
            min_ret = g.rolling(self.window_size).apply(lambda x: np.mean(np.sort(x)[:self.n_top]), raw=False).abs()
        else:
            max_ret = g.rolling(self.window_size).max()
            min_ret = g.rolling(self.window_size).min().abs()

        # Compute max - abs(min)
        max_min_ret = max_ret - min_ret.abs()
        # Drop the first level of the MultiIndex
        max_min_ret = maybe_droplevel(max_min_ret, level=0)

        # rename the column to 'skew'
        max_min_ret['skew'] = max_min_ret[self.return_col]

        return max_min_ret[['skew']]

