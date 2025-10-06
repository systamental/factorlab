import pandas as pd
from typing import List, Union

from factorlab.factors.skew.base import SkewFactor
from factorlab.features.residuals import IdiosyncraticReturns
from factorlab.transformations.moments import Skewness


class ISkew(SkewFactor):
    """
    Computes the skew of an asset's idiosyncratic returns.

    Skewness is a measure of the asymmetry of the probability distribution of a real-valued random variable
    about its mean. Positive skew indicates a distribution with an asymmetric tail extending toward more
    positive values, while negative skew indicates a distribution with an asymmetric tail extending toward more
    negative values.

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
                 factor_cols: Union[str, List[str]],
                 **kwargs):
        super().__init__(**kwargs)
        self.name = 'ISkew'
        self.description = 'Idiosyncratic skewness factor calculated using the asset idiosyncratic returns.'
        self.factor_cols = factor_cols if isinstance(factor_cols, list) else [factor_cols]

    @property
    def inputs(self) -> List[str]:
        """
        Required input columns.
        Override in subclasses as needed.
        """
        return [self.return_col] + self.factor_cols

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
        # idiosyncratic returns
        idio_ret = IdiosyncraticReturns(return_col=self.return_col,
                                        factor_cols=self.factor_cols,
                                        model='linear',
                                        incl_alpha=True,
                                        window_type='rolling',
                                        window_size=self.window_size).compute(df)

        skew_df = Skewness(input_col='idio_ret',
                           window_type=self.window_type,
                           window_size=self.window_size).compute(idio_ret)

        return skew_df[['skew']]
