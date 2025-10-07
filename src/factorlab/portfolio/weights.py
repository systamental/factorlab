import pandas as pd
from typing import Union, List
from factorlab.core.base_transform import BaseTransform
from factorlab.utils import to_dataframe


class Weights(BaseTransform):
    """
    Computes cross-sectional weights by normalizing a 'metric_col' (e.g., notional value, market cap)
    across assets at each time step.

    This is a stateless transformation.
    """

    def __init__(self,
                 metric_col: str,
                 output_col: str = 'weight'
                 ):

        # Corrected name and description for generic weight calculation
        super().__init__(name="Weights",
                         description="Calculates asset weights based on a metric column.")
        self.metric_col = metric_col
        self.output_col = output_col

    @property
    def inputs(self) -> List[str]:

        return [self.metric_col]

    def fit(self, X: Union[pd.Series, pd.DataFrame], y=None) -> 'Weights':
        """
        Minimal fit implementation for a stateless class.
        We only validate inputs and set the fitted flag.
        """
        df_input = to_dataframe(X)
        self.validate_inputs(df_input)
        self._is_fitted = True
        return self

    def transform(self, X: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
        """
        Calculates weights = metric_value / total_metric_value, grouped by time.
        """
        if not self._is_fitted:
            raise RuntimeError(f"Transform '{self.name}' must be fitted before calling transform()")

        df = to_dataframe(X).copy()
        self.validate_inputs(df)

        # compute total val across assets at each time step
        total_val = df[self.metric_col].abs().groupby(level=0).transform('sum')

        # avoid division by zero by replacing zero sums with a tiny number
        total_val_safe = total_val.replace(0, 1e-9)

        # compute weights
        df[self.output_col] = df[self.metric_col] / total_val_safe

        # Return the original DataFrame with the new weight column appended
        return df
