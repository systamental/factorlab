from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import List, Optional, Union

from factorlab.factors.base import Factor
from factorlab.utils import to_dataframe


class ValueFactor(Factor, ABC):
    """
    Abstract base class for all value factors in FactorLab.

    This class provides a common framework for value factor calculation,
    handling repetitive tasks like log transformations, non-linear value
    functions (e.g., Metcalfe), and delegates the final ratio calculation
    to subclasses.

    Parameters
    ----------
    fundamental_col : str
        Column name for the fundamental metric (e.g., earnings, book value).
    price_col : str
        Column name for prices.
    value_function: str, {'Metcalfe', 'Zipf', 'Metcalfe_gen', 'Metcalfe_sqrt', 'Sardoff'}, default None
        Function that maps the relationship between the fundamental and the price metrics.
    sign_flip : bool, default True
        If True, flips the sign of the computed value factor.
    **kwargs:
        Additional keyword arguments.
    """

    def __init__(self,
                 fundamental_col: str,
                 price_col: str,
                 output_col: Optional[str] = None,
                 value_function: Optional[str] = None,
                 sign_flip: bool = True,
                 **kwargs):

        # Ensure name is set for subclass to override later
        super().__init__(name=self.__class__.__name__,
                         description='Base class for value factors.',
                         category='Value')

        self.fundamental_col = fundamental_col
        self.price_col = price_col
        self.output_col = output_col if output_col else f'{fundamental_col}_to_{price_col}_ratio'
        self.value_function = value_function
        self.sign_flip = sign_flip
        self.kwargs = kwargs

    @property
    def inputs(self) -> List[str]:
        """Required input columns."""
        return [self.fundamental_col, self.price_col]

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

    def _apply_value_function(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the non-linear value function transformation to the fundamental column.
        """
        fund_col = self.fundamental_col

        if self.value_function is not None:
            if self.value_function == 'Metcalfe':
                df[fund_col] = df[fund_col] ** 2
            elif self.value_function == 'Zipf':
                # Use np.log for natural log, guarding against non-positive values
                df[fund_col] = df[fund_col] * np.log(df[fund_col].clip(lower=1e-9))
            elif self.value_function == 'Metcalfe_gen':
                df[fund_col] = df[fund_col] ** 1.5
            elif self.value_function == 'Metcalfe_sqrt':
                # This is equivalent to absolute value: sqrt(x^2) = |x|
                df[fund_col] = np.sqrt(df[fund_col] ** 2)

        return df

    @abstractmethod
    def _compute_value(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Abstract method to be implemented by subclasses.

        This method must contain the unique logic for computing the value factor
        (e.g., the specific ratio or difference) after pre-processing and transformations.

        The returned DataFrame must contain only the final factor column.
        """
        raise NotImplementedError

    def _generate_name(self) -> str:
        """
        Generates a standardized name for the factor based on its parameters.
        Subclasses should typically override this for more descriptive names.
        """
        name_parts = [self.name, self.fundamental_col, self.price_col]

        return "_".join(name_parts)

    def transform(self, X: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
        """
        Public interface for applying the value factor.
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
        Applies the common ValueFactor computation pipeline.

        Parameters
        ----------
        X : Union[pd.Series, pd.DataFrame]
            Input data containing the fundamental and price columns.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the computed value factor.
        """
        df = X

        # deep copy for safety
        df = df[self.inputs].copy(deep=True)

        # fcn transformation
        df = self._apply_value_function(df)

        # compute value factor
        factor_df = self._compute_value(df)

        # sign flip
        if self.sign_flip:
            factor_df *= -1

        X[self._generate_name()] = factor_df[self.output_col]

        return X
