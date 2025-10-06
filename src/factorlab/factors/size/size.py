import pandas as pd
from typing import List, Union, Optional
from factorlab.factors.base import Factor
from factorlab.utils import to_dataframe


class Size(Factor):
    """
    Computes the size factor, optionally applying log transformation and/or
    smoothing to the input size metric (e.g., market capitalization).

    This factor is typically computed as the negative logarithm of market cap
    to align with common factor investing convention (smaller firm = higher size factor value).


    Parameters
    ----------
    size_col : str
        Column name for the size metric (e.g., market cap).
    sign_flip : bool, default True
        If True, flips the sign of the computed size values.
    **kwargs:
        Additional keyword arguments.

    """

    def __init__(self,
                 size_col: str,
                 sign_flip: bool = True,
                 **kwargs):
        super().__init__(name='Size',
                         description='Size factor',
                         category='Size',
                         tags=['size', 'market_cap'])

        self.size_col = size_col
        self.sign_flip = sign_flip
        self.kwargs = kwargs

    @property
    def inputs(self) -> List[str]:
        """Required input columns."""
        return [self.size_col]

    def _generate_name(self) -> str:
        """
        Generates a standardized name for the factor based on its parameters.
        """
        name_parts = [self.name, self.size_col]

        return "_".join(name_parts)

    def fit(self, X: Union[pd.Series, pd.DataFrame], y: Optional[Union[pd.Series, pd.DataFrame]] = None):
        """
        Initializes and fits any internal transformers.

        Parameters
        ----------
        X : Union[pd.Series, pd.DataFrame]
            Input data (e.g., OHLCV data).

        Returns
        -------
        self
            The fitted transformer instance.
        """
        df_input = to_dataframe(X)
        self.validate_inputs(df_input)
        self._is_fitted = True
        return self

    def transform(self, X: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
        """
        Public interface for applying the skew factor.
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
        Applies the size factor computation pipeline (Log, Smoothing, Inversion).

        Parameters
        ----------
        X : Union[pd.Series, pd.DataFrame]
            Input data containing the size column.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the computed size factor.
        """
        df = X[[self.size_col]].copy()

        # sign flip so that smaller size = higher factor value
        if self.sign_flip:
            df = -df

        # rename col
        X[self._generate_name()] = df[[self.size_col]]

        return X
