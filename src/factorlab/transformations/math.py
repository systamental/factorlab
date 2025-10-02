import pandas as pd
import numpy as np
from typing import Union, Optional, List

from factorlab.core.base_transform import BaseTransform
from factorlab.utils import to_dataframe


class Log(BaseTransform):
    """
    Computes the natural logarithm of the input values.

    This is a stateless transform and is fully compatible with the fit/transform/fit_transform API.

    Parameters
    ----------
    input_cols : Union[str, List[str]], default 'close'
        Column name(s) of the price series to compute logs on.
    """

    def __init__(self, input_cols: Union[str, List[str]] = 'close'):
        super().__init__(name="Log", description="Computes the natural logarithm of input values.")

        # Ensure input_cols is always a list for consistent processing
        self.input_cols: List[str] = [input_cols] if isinstance(input_cols, str) else input_cols

    @property
    def inputs(self) -> List[str]:
        """Required input columns for this transform."""
        return self.input_cols

    def fit(self, X: Union[pd.Series, pd.DataFrame], y: Optional[Union[pd.Series, pd.DataFrame]] = None) -> 'Log':
        """Stateless fit: validates input and marks the transform as fitted."""
        df_input = to_dataframe(X)
        self.validate_inputs(df_input)
        self._is_fitted = True
        return self

    def transform(self, X: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
        """
        Public method to apply the natural logarithm transformation.

        It handles input validation, state checks, and delegates the core
        calculation to the private `_transform` method.
        """
        if not self._is_fitted:
            raise RuntimeError(f"Transform '{self.name}' must be fitted before calling transform()")

        df_input = to_dataframe(X)
        self.validate_inputs(df_input)

        # Slice and copy the required columns (ensuring immutability)
        df_slice = df_input[self.input_cols].copy(deep=True)

        return self._transform(df_slice)

    def _transform(self, df_slice: pd.DataFrame) -> pd.DataFrame:
        """
        Private method containing the core natural logarithm calculation logic.

        Parameters
        ----------
        df_slice : pd.DataFrame
            The input data slice containing only the columns to be transformed.

        Returns
        -------
        pd.DataFrame
            The transformed data with new column names.
        """
        # Replace non-positive values (<= 0) with NaN before taking the log
        df_slice = df_slice.mask(df_slice <= 0)
        log_df = np.log(df_slice)

        # Clean up infinities that might result from edge cases
        log_df = log_df.replace([np.inf, -np.inf], np.nan)

        # Rename columns to indicate log transformation
        rename_map = {col: f'log_{col}' for col in self.input_cols}
        log_df = log_df.rename(columns=rename_map)

        return log_df


class SquareRoot(BaseTransform):
    """
    Computes the square root of the input values.

    This is a stateless transform and is fully compatible with the fit/transform/fit_transform API.

    Parameters
    ----------
    input_cols : Union[str, List[str]], default 'close'
        Column name(s) of the price series to compute square root on.
    """

    def __init__(self, input_cols: Union[str, List[str]] = 'close'):
        super().__init__(name="SquareRoot", description="Computes the square root of the input values.")

        # Ensure input_cols is always a list for consistent processing
        self.input_cols: List[str] = [input_cols] if isinstance(input_cols, str) else input_cols

    @property
    def inputs(self) -> List[str]:
        """Required input columns for this transform."""
        return self.input_cols

    def fit(self,
            X: Union[pd.Series, pd.DataFrame],
            y: Optional[Union[pd.Series, pd.DataFrame]] = None) -> 'SquareRoot':
        """Stateless fit: validates input and marks the transform as fitted."""
        df_input = to_dataframe(X)
        self.validate_inputs(df_input)
        self._is_fitted = True
        return self

    def transform(self, X: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
        """
        Public method to apply the square root transformation.

        It handles input validation, state checks, and delegates the core
        calculation to the private `_transform` method.
        """
        if not self._is_fitted:
            raise RuntimeError(f"Transform '{self.name}' must be fitted before calling transform()")

        df_input = to_dataframe(X)
        self.validate_inputs(df_input)

        # Slice and copy the required columns (ensuring immutability)
        df_slice = df_input[self.input_cols].copy(deep=True)

        return self._transform(df_slice)

    def _transform(self, df_slice: pd.DataFrame) -> pd.DataFrame:
        """
        Private method containing the core square root calculation logic.

        Parameters
        ----------
        df_slice : pd.DataFrame
            The input data slice containing only the columns to be transformed.

        Returns
        -------
        pd.DataFrame
            The transformed data with new column names.
        """
        # Replace negative values with NaN as square root is undefined for real numbers
        df_slice = df_slice.mask(df_slice < 0)
        sqrt_df = np.sqrt(df_slice)

        # Clean up infinities
        sqrt_df = sqrt_df.replace([np.inf, -np.inf], np.nan)

        # Rename columns to indicate square root transformation
        rename_map = {col: f'sqrt_{col}' for col in self.input_cols}
        sqrt_df = sqrt_df.rename(columns=rename_map)

        return sqrt_df


class Square(BaseTransform):
    """
    Computes the square of the input values.

    This is a stateless transform and is fully compatible with the fit/transform/fit_transform API.

    Parameters
    ----------
    input_cols : Union[str, List[str]], default 'close'
        Column name(s) of the price series to compute square on.
    """

    def __init__(self, input_cols: Union[str, List[str]] = 'close'):
        super().__init__(name="Square", description="Computes the square of the input values.")

        # Ensure input_cols is always a list for consistent processing
        self.input_cols: List[str] = [input_cols] if isinstance(input_cols, str) else input_cols

    @property
    def inputs(self) -> List[str]:
        """Required input columns for this transform."""
        return self.input_cols

    def fit(self, X: Union[pd.Series, pd.DataFrame], y: Optional[Union[pd.Series, pd.DataFrame]] = None) -> 'Square':
        """Stateless fit: validates input and marks the transform as fitted."""
        df_input = to_dataframe(X)
        self.validate_inputs(df_input)
        self._is_fitted = True
        return self

    def transform(self, X: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
        """
        Public method to apply the square transformation.

        It handles input validation, state checks, and delegates the core
        calculation to the private `_transform` method.
        """
        if not self._is_fitted:
            raise RuntimeError(f"Transform '{self.name}' must be fitted before calling transform()")

        df_input = to_dataframe(X)
        self.validate_inputs(df_input)

        # Slice and copy the required columns (ensuring immutability)
        df_slice = df_input[self.input_cols].copy(deep=True)

        return self._transform(df_slice)

    def _transform(self, df_slice: pd.DataFrame) -> pd.DataFrame:
        """
        Private method containing the core square calculation logic.

        Parameters
        ----------
        df_slice : pd.DataFrame
            The input data slice containing only the columns to be transformed.

        Returns
        -------
        pd.DataFrame
            The transformed data with new column names.
        """
        # Apply square transformation using numpy
        sq_df = np.square(df_slice)

        # Rename columns to indicate square transformation
        rename_map = {col: f'sq_{col}' for col in self.input_cols}
        sq_df = sq_df.rename(columns=rename_map)

        return sq_df


class Power(BaseTransform):
    """
    Computes the power of the input values raised to a specified exponent.

    This is a stateless transform and is fully compatible with the fit/transform/fit_transform API.

    Parameters
    ----------
    exponent: Union[int, float], default 2
        Exponent used in the power transformation.
    input_cols : Union[str, List[str]], default 'close'
        Column name(s) of the price series to compute power on.
    """

    def __init__(self, exponent: Union[int, float] = 2, input_cols: Union[str, List[str]] = 'close'):
        # Allow exponent to be an integer or float for flexibility
        if not isinstance(exponent, (int, float)):
            raise ValueError("Exponent must be an integer or float.")

        super().__init__(name="Power", description=f"Computes the power of the input values raised to {exponent}.")

        self.exponent = exponent

        # Ensure input_cols is always a list for consistent processing
        self.input_cols: List[str] = [input_cols] if isinstance(input_cols, str) else input_cols

    @property
    def inputs(self) -> List[str]:
        """Required input columns for this transform."""
        return self.input_cols

    def fit(self, X: Union[pd.Series, pd.DataFrame], y: Optional[Union[pd.Series, pd.DataFrame]] = None) -> 'Power':
        """Stateless fit: validates input and marks the transform as fitted."""
        df_input = to_dataframe(X)
        self.validate_inputs(df_input)
        self._is_fitted = True
        return self

    def transform(self, X: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
        """
        Public method to apply the power transformation.

        It handles input validation, state checks, and delegates the core
        calculation to the private `_transform` method.
        """
        if not self._is_fitted:
            raise RuntimeError(f"Transform '{self.name}' must be fitted before calling transform()")

        df_input = to_dataframe(X)
        self.validate_inputs(df_input)

        # Slice and copy the required columns (ensuring immutability)
        df_slice = df_input[self.input_cols].copy(deep=True)

        return self._transform(df_slice)

    def _transform(self, df_slice: pd.DataFrame) -> pd.DataFrame:
        """
        Private method containing the core power calculation logic.

        Parameters
        ----------
        df_slice : pd.DataFrame
            The input data slice containing only the columns to be transformed.

        Returns
        -------
        pd.DataFrame
            The transformed data with new column names.
        """
        # Apply power transformation using numpy
        power_df = np.power(df_slice, self.exponent)

        # Clean up infinities that might result from edge cases
        power_df = power_df.replace([np.inf, -np.inf], np.nan)

        # Rename columns to indicate power transformation
        # Use a sanitized exponent name for the column prefix
        exp_name = str(self.exponent).replace('.', '_')
        rename_map = {col: f'power{exp_name}_{col}' for col in self.input_cols}
        power_df = power_df.rename(columns=rename_map)

        return power_df
