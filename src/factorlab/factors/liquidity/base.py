from abc import ABC, abstractmethod
import pandas as pd
from pandas.core.groupby import DataFrameGroupBy
from typing import Optional, List, Union, Any

from factorlab.factors.base import Factor
from factorlab.utils import to_dataframe


class LiquidityFactor(Factor, ABC):
    """
    Abstract base class for liquidity factors in FactorLab.

    Provides a common framework for liquidity factor calculation, handling tasks
    like normalization, winsorization, and common pre-processing steps.

    Parameters
    ----------
    input_col : str, default 'ret'
        Column name for return data.
    window_type : {'rolling', 'ewm', 'expanding'}, default 'rolling'
        Type of rolling window to use.
    window_size : int, default 30
        Rolling window size for calculations.
    **kwargs :
        Additional keyword arguments for specific liquidity factor implementations.

    Notes
    -----
    Instance variables match the parameters. Use the class constructor to set
    them, and access them directly via attributes if needed.

    """

    def __init__(self,
                 input_col: str = 'ret',
                 window_type: str = "rolling",
                 window_size: int = 30,
                 smooth: bool = False,
                 **kwargs):
        super().__init__(name=self.__class__.__name__,
                         description='Base class for liquidity factors.',
                         category='Liquidity')

        self.input_col = input_col
        self.window_type = window_type
        self.window_size = window_size
        self.smooth = smooth
        self.kwargs = kwargs

    @property
    def inputs(self) -> List[str]:
        """
        Required input columns.
        Override in subclasses as needed.
        """
        return [self.input_col]

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

    @abstractmethod
    def _compute_liquidity(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Abstract method to be implemented by subclasses.

        This method should contain the unique logic for computing the liquidity factor,
        without any normalization or winsorization.
        """
        raise NotImplementedError

    def _generate_name(self) -> str:
        """
        Generates a standardized name for the low risk factor based on its parameters.
        """
        name_parts = [self.name]

        return "_".join(name_parts)

    def _get_ts_window_op(self, g: Union[pd.DataFrame, DataFrameGroupBy]) -> Any:
        """Helper to determine and initialize the correct window operation for time series."""
        if self.window_type == 'ewm':
            # Returns Rolling GroupBy object
            return g.ewm(span=self.window_size)
        elif self.window_type == 'rolling':
            # Returns Rolling GroupBy object
            return g.rolling(window=self.window_size)
        elif self.window_type == 'expanding':
            # Returns Expanding GroupBy object
            return g.expanding()
        else:
            raise ValueError(f"Unsupported window type: {self.window_type} for axis 'ts'. "
                             f"Must be 'rolling', 'expanding', or 'fixed'.")

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
        Applies the full LowRiskFactor computation pipeline.

        Parameters
        ----------
        X : Union[pd.Series, pd.DataFrame]
            Input data containing the required columns.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the computed low risk factor.
        """
        df = X.copy()

        # compute trend
        liq_df = self._compute_liquidity(df)

        # add to original df
        X[self._generate_name()] = liq_df

        return X
