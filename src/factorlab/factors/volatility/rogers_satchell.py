import pandas as pd
import numpy as np
from typing import List

from factorlab.utils import grouped, maybe_droplevel
from factorlab.factors.volatility.base import VolFactor


class RogersSatchell(VolFactor):
    """
    Computes the Rogers-Satchell volatility of an asset.

    This factor calculates the volatility of an asset's returns using the
    Rogers-Satchell volatility estimator, which is based on the open, high, low, and close prices
    of the asset over a specified window.

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = 'RogersSatchell'
        self.description = 'Computes the volatility of an asset\'s returns using the Rogers-Satchell estimator.'

    @property
    def inputs(self) -> List[str]:
        """
        Required input columns.
        Override in subclasses as needed.
        """
        return ['high', 'low', 'open', 'close']

    def _compute_vol(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes the Parkinson's volatility of the asset.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame containing the required columns.

        Returns
        -------
        pd.DataFrame
            A new DataFrame with the computed vol values in the specified output column.

        Raises
        ------
        ValueError
            If the input DataFrame is missing the required columns for computation.
        """
        # Calculate the Rogers-Satchell volatility
        rs = (np.log(df['high'] / df['close']) * np.log(df['high'] / df['open']) +
              np.log(df['low'] / df['close']) * np.log(df['low'] / df['open']))

        g = grouped(rs)

        vol_df = g.rolling(self.window_size).mean() ** 0.5

        # drop the grouping level if it exists
        vol_df = maybe_droplevel(vol_df)

        df[self.output_col] = vol_df

        return df
