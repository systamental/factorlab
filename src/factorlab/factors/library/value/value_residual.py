from __future__ import annotations
import pandas as pd
from factorlab.features.residuals import Residuals
from factorlab.factors.value.base import ValueFactor


class ValueResidual(ValueFactor):
    """
    Computes the value factor as the residuals from regressing a fundamental metric on price.

    Parameters
    ----------
    window_type : str, {'rolling', 'ewm', 'expanding'}, default "rolling"
        Method for regression.
    window_size : int, default 90
        Window size for regression.

    """
    def __init__(self,
                 window_type: str = 'rolling',
                 window_size: int = 360,
                 **kwargs):

        super().__init__(**kwargs)
        self.name = 'ValueResidual'
        self.description = ('Value Residual factor computed as the residuals from regressing a fundamental metric '
                            'on price.')
        self.window_type = window_type
        self.window_size = window_size

    def _compute_value(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes the value ratio signal.
        """
        # compute value residuals
        value_df = Residuals(target_col=self.price_col,
                             feature_col=self.fundamental_col,
                             model='linear',
                             window_type=self.window_type,
                             window_size=self.window_size,
                             output_col=self.output_col).compute(df)

        return value_df
