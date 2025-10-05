import pandas as pd

from factorlab.factors.volatility.base import VolFactor
from factorlab.transformations.dispersion import MedianAbsoluteDeviation


class MAD(VolFactor):
    """
    Computes the median absolute deviation of an asset's returns.

    This factor calculates the median absolute deviation of asset returns over a specified
    rolling window, providing a measure of the asset's volatility.

    """

    def __init__(self, window_type='rolling', **kwargs):
        super().__init__(window_type=window_type, **kwargs)
        self.name = 'MAD'
        self.description = 'Computes the median absolute deviation of an asset\'s returns.'

    def _compute_vol(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes the median absolute deviation of the asset's returns.

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
        vol_df = MedianAbsoluteDeviation(input_col=self.input_col,
                                         output_col=self.output_col,
                                         window_type=self.window_type,
                                         window_size=self.window_size).compute(df)

        return vol_df
