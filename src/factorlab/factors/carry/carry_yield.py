import pandas as pd

from factorlab.factors.carry.base import CarryFactor


class Yield(CarryFactor):
    """
    Computes the carry of an asset.

    Carry can be calculated using either a spot and forward price premium difference or
    an interest or funding rate/yield.
    """

    def __init__(self, scale=False, **kwargs):
        """
        Constructor.

        """
        super().__init__(scale=scale, **kwargs)
        self.name = 'Carry'
        self.description = 'Carry factor calculated using either spot/forward prices or interest rates/yield.'

    def _compute_carry(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes the carry factor based on the initialized method.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame containing the required columns.

        Returns
        -------
        pd.DataFrame
            A new DataFrame with the computed carry values in the specified output column.

        Raises
        ------
        ValueError
            If the input DataFrame is missing the required columns for computation.
        """
        if self.rate_col is None:
            carry_df = ((df[self.fwd_col] / df[self.spot_col]) - 1).to_frame('carry')
        else:
            carry_df = df[self.rate_col].to_frame('carry')

        return carry_df
