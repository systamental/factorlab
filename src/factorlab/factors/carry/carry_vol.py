import pandas as pd

from factorlab.factors.carry.base import CarryFactor


class CarryVol(CarryFactor):
    """
    Computes the carry to volatility ratio of an asset.

    Carry can be calculated using either a spot and forward price premium difference or
    an interest or funding rate/yield.

    The carry to volatility ratio is a measure of the return (carry) of an asset relative to its risk (volatility).
    """

    def __init__(self, scale=True, **kwargs):
        """
        Constructor.

        """
        super().__init__(scale=scale, **kwargs)
        self.name = 'CarryVol'
        self.description = ('Carry to Volatility ratio calculated using either spot/forward prices or '
                            'interest rates/yield and scaled by volatility.')

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
