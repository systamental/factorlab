import pandas as pd
from typing import List

from factorlab.factors.low_risk.base import LowRiskFactor
from factorlab.features.betas import Betas


class Beta(LowRiskFactor):
    """
    Computes the beta of an asset relative to market returns.

    Beta is a measure of an asset's sensitivity to movements in the overall market or
    a specific benchmark. A beta greater than 1 indicates that the asset tends to
    amplify market movements, while a beta less than 1 suggests that the asset is
    less volatile than the market.

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
                 return_col: str = 'ret',
                 factor_col: str = 'market',
                 **kwargs):
        super().__init__(**kwargs)
        self.name = 'Beta'
        self.description = 'Beta factor calculated using factor returns.'
        self.return_col = return_col
        self.factor_col = factor_col

    @property
    def inputs(self) -> List[str]:
        """
        Required input columns.
        Override in subclasses as needed.
        """
        return [self.return_col, self.factor_col]

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
        # compute beta
        beta_df = Betas(target_col=self.return_col,
                        feature_col=self.factor_col,
                        model='linear',
                        window_type='rolling',
                        window_size=self.window_size).compute(df)

        return beta_df
