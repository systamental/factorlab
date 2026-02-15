import pandas as pd
import numpy as np
from typing import Union, Optional, List

from factorlab.core.base_transform import BaseTransform
from factorlab.utils import to_dataframe


class Log(BaseTransform):
    """
    Computes the natural logarithm of the input values.

    Implements the Phase 1: Accumulation Contract: returns the full input DataFrame
    with new 'log_' columns appended.

    Parameters
    ----------
    input_cols : Union[str, List[str]], default 'close'
        Column name(s) of the price series to compute logs on.
    """

    def __init__(self, input_cols: Union[str, List[str]] = 'close'):
        super().__init__(name="Log", description="Computes the natural logarithm of input values.")

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

        It handles input validation, state checks, and implements the Accumulation Contract.
        """
        if not self._is_fitted:
            raise RuntimeError(f"Transform '{self.name}' must be fitted before calling transform()")

        df_input = to_dataframe(X).copy(deep=True)
        self.validate_inputs(df_input)

        return self._transform(df_input)

    def _transform(self, df_input: pd.DataFrame) -> pd.DataFrame:
        """
        Private method containing the core natural logarithm calculation logic and accumulation.

        Parameters
        ----------
        df_input : pd.DataFrame
            The input data containing all columns (context).

        Returns
        -------
        pd.DataFrame
            The input data with the new log column(s) appended.
        """
        # log transformation logic
        df_slice = df_input[self.input_cols].mask(df_input <= 0)
        log_df = np.log(df_slice).replace([np.inf, -np.inf], np.nan)

        # rename cols
        rename_map = {col: f'log_{col}' for col in self.input_cols}
        for original_col, new_col in rename_map.items():
            df_input[new_col] = log_df[original_col]

        return df_input


class SquareRoot(BaseTransform):
    """
    Computes the square root of the input values.

    Implements the Phase 1: Accumulation Contract: returns the full input DataFrame
    with new 'sqrt_' columns appended.

    Parameters
    ----------
    input_cols : Union[str, List[str]], default 'close'
        Column name(s) of the price series to compute square root on.
    """

    def __init__(self, input_cols: Union[str, List[str]] = 'close'):
        super().__init__(name="SquareRoot", description="Computes the square root of the input values.")

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

        It handles input validation, state checks, and implements the Accumulation Contract.
        """
        if not self._is_fitted:
            raise RuntimeError(f"Transform '{self.name}' must be fitted before calling transform()")

        df_input = to_dataframe(X).copy(deep=True)
        self.validate_inputs(df_input)

        return self._transform(df_input)

    def _transform(self, df_input: pd.DataFrame) -> pd.DataFrame:
        """
        Private method containing the core square root calculation logic and accumulation.
        """
        # sqrt transformation logic
        df_slice = df_input[self.input_cols].mask(df_input < 0)
        sqrt_df = np.sqrt(df_slice).replace([np.inf, -np.inf], np.nan)

        # rename cols
        rename_map = {col: f'sqrt_{col}' for col in self.input_cols}
        for original_col, new_col in rename_map.items():
            df_input[new_col] = sqrt_df[original_col]

        return df_input


class Square(BaseTransform):
    """
    Computes the square of the input values.

    Implements the Phase 1: Accumulation Contract: returns the full input DataFrame
    with new 'sq_' columns appended.

    Parameters
    ----------
    input_cols : Union[str, List[str]], default 'close'
        Column name(s) of the price series to compute square on.
    """

    def __init__(self, input_cols: Union[str, List[str]] = 'close'):
        super().__init__(name="Square", description="Computes the square of the input values.")

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

        It handles input validation, state checks, and implements the Accumulation Contract.
        """
        if not self._is_fitted:
            raise RuntimeError(f"Transform '{self.name}' must be fitted before calling transform()")

        df_input = to_dataframe(X).copy(deep=True)
        self.validate_inputs(df_input)

        return self._transform(df_input)

    def _transform(self, df_input: pd.DataFrame) -> pd.DataFrame:
        """
        Private method containing the core square calculation logic and accumulation.
        """
        # square transformation logic
        sq_df = np.square(df_input[self.input_cols])

        # rename cols
        rename_map = {col: f'sq_{col}' for col in self.input_cols}
        for original_col, new_col in rename_map.items():
            df_input[new_col] = sq_df[original_col]

        return df_input


class Power(BaseTransform):
    """
    Computes the power of the input values raised to a specified exponent.

    Implements the Phase 1: Accumulation Contract: returns the full input DataFrame
    with new 'power_' columns appended.

    Parameters
    ----------
    exponent: Union[int, float], default 2
        Exponent used in the power transformation.
    input_cols : Union[str, List[str]], default 'close'
        Column name(s) of the price series to compute power on.
    """

    def __init__(self, exponent: Union[int, float] = 2, input_cols: Union[str, List[str]] = 'close'):
        if not isinstance(exponent, (int, float)):
            raise ValueError("Exponent must be an integer or float.")
        super().__init__(name="Power", description=f"Computes the power of the input values raised to {exponent}.")

        self.exponent = exponent
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

        It handles input validation, state checks, and implements the Accumulation Contract.
        """
        if not self._is_fitted:
            raise RuntimeError(f"Transform '{self.name}' must be fitted before calling transform()")

        df_input = to_dataframe(X).copy(deep=True)
        self.validate_inputs(df_input)

        # Delegate computation and accumulation
        return self._transform(df_input)

    def _transform(self, df_input: pd.DataFrame) -> pd.DataFrame:
        """
        Private method containing the core power calculation logic and accumulation.
        """
        # 1. Compute the power on the selected columns
        power_df = np.power(df_input[self.input_cols], self.exponent)
        power_df = power_df.replace([np.inf, -np.inf], np.nan)

        # 2. Define the new column names
        exp_name = str(self.exponent).replace('.', '_')
        rename_map = {col: f'power{exp_name}_{col}' for col in self.input_cols}

        # 3. Accumulate: Assign new columns directly to the input DataFrame copy
        for original_col, new_col in rename_map.items():
            df_input[new_col] = power_df[original_col]

        # 4. Return the full, expanded DataFrame
        return df_input
