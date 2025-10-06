import pandas as pd

from factorlab.factors.skew.base import SkewFactor
from factorlab.transformations.moments import Skewness


class Skew(SkewFactor):
    """
    Computes the skew of an asset's returns.

    Skewness is a measure of the asymmetry of the probability distribution of a real-valued random variable
    about its mean. Positive skew indicates a distribution with an asymmetric tail extending toward more
    positive values, while negative skew indicates a distribution with an asymmetric tail extending toward more
    negative values.

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = 'Skew'
        self.description = 'Skewness factor calculated using the asset returns.'

    def _compute_skew(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes the skewness factor based on the initialized method.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame containing the required columns.

        Returns
        -------
        pd.DataFrame
            A new DataFrame with the computed skewness values in the specified output column.

        Raises
        ------
        ValueError
            If the input DataFrame is missing the required columns for computation.
        """
        skew_df = Skewness(input_col=self.return_col,
                           window_type=self.window_type,
                           window_size=self.window_size).compute(df)

        return skew_df
