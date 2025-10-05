import pandas as pd
import numpy as np
from typing import List

from factorlab.utils import grouped, maybe_droplevel
from factorlab.factors.volatility.base import VolFactor


class GarmanKlass(VolFactor):
    """
    Computes the Garman-Klass volatility of an asset.

    This factor calculates the volatility of an asset's returns using the
    Garman-Klass volatility estimator, which is based on the open, high,
    low, and close prices of the asset over a specified window.

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = 'GarmanKlass'
        self.description = 'Computes the volatility of an asset\'s returns using the Garman-Klass estimator.'

    @property
    def inputs(self) -> List[str]:
        """
        Required input columns.
        Override in subclasses as needed.
        """
        return ['open', 'high', 'low', 'close']

    def _compute_vol(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes the Garman-Klass volatility of the asset.

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
        # Calculate the Parkinson's volatility
        hl = np.log(df['high'] / df['low']) ** 2
        co = np.log(df['close'] / df['open']) ** 2

        const = 2 * np.log(2) - 1

        g_hl = grouped(hl)
        g_co = grouped(co)

        vol_df = (0.5 * g_hl.rolling(self.window_size).mean() - const * g_co.rolling(self.window_size).mean()) ** 0.5

        vol_df = maybe_droplevel(vol_df)

        df[self.output_col] = vol_df

        return df
