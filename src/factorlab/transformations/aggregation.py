import pandas as pd
from typing import Union

from factorlab.core.base_transform import BaseTransform
from factorlab.utils import to_dataframe


class WeightedSum(BaseTransform):
    """
    Computes a weighted sum (e.g., portfolio return or weighted average factor score)
    of a target column using a corresponding weight column.

    Parameters
    ----------
    input_col : str, default 'return'
        The column containing the values to be aggregated (e.g., stock returns).
    weight_col : str, default 'weight'
        The column containing the corresponding weights for each item.
    output_col : str, default 'weighted_sum'
        The name for the computed output Series.
    """

    def __init__(
            self,
            input_col: str = "return",
            weight_col: str = "weight",
            output_col: str = "weighted_sum"
    ):
        super().__init__(
            name="WeightedSum",
            description=f"Computes the weighted sum of two series."
        )
        self.input_col = input_col
        self.weight_col = weight_col
        self.output_col = output_col

    @property
    def inputs(self) -> list[str]:
        """Required input columns for this transform."""
        return [self.input_col, self.weight_col]

    def fit(self, X: pd.DataFrame, y=None) -> 'WeightedSum':
        """Stateless fit: validates input columns and marks the transform as fitted."""
        df_input = to_dataframe(X)
        self.validate_inputs(df_input)
        self._is_fitted = True
        return self

    def transform(self, X: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
        """
        Public interface for applying the agg transformation.
        Performs checks and prepares data before calling the internal _transform method.
        """
        if not self._is_fitted:
            raise RuntimeError(f"Transform '{self.name}' must be fitted before calling transform()")

        # validate and create copy
        df_input = to_dataframe(X).copy(deep=True)
        self.validate_inputs(df_input)

        # transform and return
        return self._transform(df_input)

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Computes the weighted aggregation for each time period in X.

        Args:
            X (pd.DataFrame): DataFrame containing both the target and weight columns.

        Returns:
            pd.DataFrame: A DataFrame with a single column containing the
                          aggregated time series.
        """
        df = X

        # weighted values
        weighted_values = df[self.input_col] * df[self.weight_col]

        # weighted sum
        df[self.output_col] = weighted_values.groupby(level=0).transform('sum')

        return df
