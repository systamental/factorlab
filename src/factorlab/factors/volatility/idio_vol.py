import pandas as pd
from typing import List, Union

from factorlab.factors.volatility.base import VolFactor
from factorlab.features.residuals import IdiosyncraticReturns
from factorlab.transformations.dispersion import StandardDeviation


class IVol(VolFactor):
    """
    Computes the idiosyncratic vol of an asset's returns.

    Idiosyncratic volatility is the portion of an asset's total volatility that is attributable to
    factors specific to that asset, rather than to market-wide factors. It represents the risk that
    cannot be diversified away through holding a broad portfolio of assets.

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
        self.name = 'IVol'
        self.description = 'Idiosyncratic vol factor calculated using the asset idiosyncratic returns.'
        self.factor_cols = factor_cols if isinstance(factor_cols, list) else [factor_cols]

    @property
    def inputs(self) -> List[str]:
        """
        Required input columns.
        Override in subclasses as needed.
        """
        return ['close'] + self.factor_cols

    def _compute_vol(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes the vol factor based on the initialized method.

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
        # idiosyncratic returns
        idio_ret = IdiosyncraticReturns(return_col='ret',
                                        factor_cols=self.factor_cols,
                                        output_col='idio_ret',
                                        model='linear',
                                        incl_alpha=True,
                                        window_type='rolling',
                                        window_size=self.window_size).compute(df)

        vol_df = StandardDeviation(input_col='idio_ret',
                                   output_col=self.output_col,
                                   window_type=self.window_type,
                                   window_size=self.window_size).compute(idio_ret)

        return vol_df
