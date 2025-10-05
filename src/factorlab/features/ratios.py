from __future__ import annotations
import pandas as pd
from typing import Optional, List, Union
from factorlab.features.base import Feature
from factorlab.utils import safe_divide, to_dataframe


class Ratios(Feature):
    """
    Computes the ratio between two specified columns in a DataFrame.

    This feature is designed to be used in factor computations where a
    normalized ratio between two variables is required.
    """

    def __init__(self,
                 numerator_col: str,
                 denominator_col: str,
                 output_col: Optional[str] = None,
                 **kwargs):
        """
        Constructor.

        Parameters
        ----------
        numerator_col : str
            The name of the column to be used as the numerator.
        denominator_col : str
            The name of the column to be used as the denominator.
        **kwargs :
            Additional keyword arguments for the base class.
        """
        super().__init__(**kwargs)
        self.name = 'Ratio',
        self.description = 'Computes the ratio between two specified columns.',
        self.numerator_col = numerator_col
        self.denominator_col = denominator_col
        self.output_col = output_col if output_col else f'{numerator_col}_to_{denominator_col}_ratio'

    @property
    def inputs(self) -> List[str]:
        """
        Required input columns.
        """
        return [self.numerator_col, self.denominator_col]

    def fit(self,
            X: Union[pd.Series, pd.DataFrame],
            y: Optional[Union[pd.Series, pd.DataFrame]] = None) -> 'Ratios':
        """
        Fits the Ratio transformation. This is primarily stateless but calls fit
        on internal components for consistency.
        """
        df_input = to_dataframe(X)
        self.validate_inputs(df_input)
        self._is_fitted = True
        return self

    def transform(self, X: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
        """
        Public interface for computing the ratio.
        Performs checks and prepares data before calling the internal _transform method.
        """
        if not self._is_fitted:
            raise RuntimeError(f"Transform '{self.name}' must be fitted before calling transform()")

        # validate and create copy
        df_input = to_dataframe(X).copy(deep=True)
        self.validate_inputs(df_input)

        # transform and return
        return self._transform(df_input)

    def _transform(self, X: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
        """
        Applies the ratio calculation using the fitted configuration.

        Parameters
        ----------
        X : Union[pd.Series, pd.DataFrame]
            Input DataFrame or Series containing the necessary columns.

        Returns
        -------
        pd.DataFrame
            A DataFrame with the computed ratio.
        """
        df = X

        # compute ratio using safe_divide
        ratio = safe_divide(df[[self.numerator_col]], df[[self.denominator_col]])

        X[self.output_col] = ratio

        return X
