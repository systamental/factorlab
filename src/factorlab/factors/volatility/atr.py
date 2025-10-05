import pandas as pd
from typing import List

from factorlab.factors.volatility.base import VolFactor
from factorlab.transformations.dispersion import AverageTrueRange


class ATR(VolFactor):
    """
    Computes the volatility of an asset's returns.

    This factor calculates the standard deviation of asset returns over a specified
    rolling window, providing a measure of the asset's volatility.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = 'ATR'
        self.description = 'Computes the volatility of an asset\'s returns.'

    @property
    def inputs(self) -> List[str]:
        """
        Required input columns.
        Override in subclasses as needed.
        """
        return ['open', 'high', 'low', 'close']

    def _compute_vol(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes the volatility of the asset's returns.

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
        vol_df = AverageTrueRange(output_col=self.output_col,
                                  window_type=self.window_type,
                                  window_size=self.window_size).compute(df)

        return vol_df
