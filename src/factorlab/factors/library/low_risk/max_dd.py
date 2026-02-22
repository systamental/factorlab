import pandas as pd
import numpy as np
from typing import List

from factorlab.factors.low_risk.base import LowRiskFactor


class MaxDD(LowRiskFactor):
    """
    Computes the max drawdown of an asset over a specified window.

    Max drawdown is a measure of the largest single drop from peak to trough
    in the value of an asset, before a new peak is achieved. It is commonly used
    to assess the risk of an investment.

    Parameters
    ----------
    window_type : str, {'rolling', 'expanding'}, default 'rolling'
        Type of rolling window to use.
    window_size : int, default 30
        Rolling window size for calculations.

    Notes
    -----
    Instance variables match the parameters. Use the class constructor
    to set them, and access them directly via attributes if needed.
    """

    def __init__(self,
                 return_col: str = 'close',
                 ret_type: str = 'simple',
                 window_type: str = "rolling",
                 sign_flip: bool = False,
                 **kwargs):
        super().__init__(window_type=window_type, sign_flip=sign_flip, **kwargs)
        self.name = 'MaxDD'
        self.description = 'Max Drawdown factor calculated using factor returns.'
        self.return_col = return_col
        self.ret_type = ret_type

    @property
    def inputs(self) -> List[str]:
        """
        Required input columns.
        Override in subclasses as needed.
        """
        return [self.return_col]

    def _compute_low_risk(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes the beta factor based on the initialized method.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame containing the required returns and factor columns.

        Returns
        -------
        pd.DataFrame
            A new DataFrame with the computed beta values in the specified output column.

        Raises
        ------
        ValueError
            If the input DataFrame is missing the required columns for computation.
        """
        # compute max dd
        if self.ret_type == 'simple':
            returns_for_prod = df[self.return_col] + 1
            cum_ret = returns_for_prod.groupby(level='ticker').cumprod()
        else:
            cum_ret = np.exp(df[self.return_col].groupby(level='ticker').cumsum())

        if self.window_type == 'rolling':
            rolling_peak = cum_ret.groupby(level='ticker').rolling(window=self.window_size,
                                                                   min_periods=1).max().droplevel(0)
            dd_series = (cum_ret / rolling_peak) - 1
            max_dd = dd_series.groupby(level='ticker').rolling(window=self.window_size,
                                                               min_periods=1).min().droplevel(0)

        elif self.window_type == 'expanding':
            peak_ret = cum_ret.groupby(level='ticker').expanding().max().droplevel(0)
            dd_series = (cum_ret / peak_ret) - 1
            max_dd = dd_series.groupby(level='ticker').expanding().min().droplevel(0)

        else:
            raise ValueError("window_type must be 'rolling' or 'expanding'")

        max_dd = max_dd.to_frame(name='max_dd').sort_index()

        return max_dd
