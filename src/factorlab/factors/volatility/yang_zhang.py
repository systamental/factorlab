import pandas as pd
from typing import List

from factorlab.factors.volatility.base import VolFactor


class YangZhang(VolFactor):
    """
    Computes the Yang-Zhang volatility of an asset.

    This factor calculates the volatility of an asset's returns using the
    Yang-Zhang volatility estimator, which is based on the open, high, low, and close prices
    of the asset over a specified window.

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = 'YangZhang'
        self.description = 'Computes the volatility of an asset\'s returns using the Yang-Zhang estimator.'

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
        # TODO: Implement the Yang-Zhang volatility calculation
