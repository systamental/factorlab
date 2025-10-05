import typing
import pandas as pd
from pandas.core.groupby import DataFrameGroupBy
from typing import Any, Optional, Union

from factorlab.core.base_transform import BaseTransform
from factorlab.utils import grouped, maybe_droplevel, to_dataframe


class Rank(BaseTransform):
    """
    Computes ranks or percentile ranks of values along a specified axis.

    Parameters
    ----------
    input_col : str, default 'close'
        The column containing the values to be ranked.
    axis : {'ts', 'cs'}, default 'ts'
        Direction of ranking, either time-series ('ts') or cross-section ('cs').
    percentile : bool, default False
        If True, returns percentile ranks (0-1); otherwise, ordinal ranks.
    window_type : {'fixed', 'rolling', 'expanding'}, default 'expanding'
        Windowing method used for time-series ranking.
    window_size : int, default 30
        Size of the window for rolling computations.
    min_periods : int, default 2
        Minimum observations required in window to produce a value.
    """

    def __init__(self,
                 input_col: str = 'close',
                 output_col: str = 'rank',
                 axis: str = 'ts',
                 percentile: bool = False,
                 window_type: str = 'expanding',
                 window_size: int = 30,
                 min_periods: int = 2):
        super().__init__(name="Rank", description="Ranks values along time or cross-section.")

        self.input_col = input_col
        self.output_col = output_col
        self.axis = axis
        self.percentile = percentile
        self.window_type = window_type
        self.window_size = window_size
        self.min_periods = min_periods

    def fit(self,
            X: Union[pd.Series, pd.DataFrame],
            y: Optional[Union[pd.Series, pd.DataFrame]] = None) -> 'Rank':
        """Stateless fit: validates input and marks the transform as fitted."""
        df_input = to_dataframe(X)
        self.validate_inputs(df_input)
        self._is_fitted = True
        return self

    def _get_ts_window_op(self, g: Union[pd.DataFrame, DataFrameGroupBy]) -> Any:
        """Helper to determine and initialize the correct window operation for time series."""
        if self.window_type == 'rolling':
            # Returns Rolling GroupBy object
            return g.rolling(window=self.window_size, min_periods=self.min_periods)
        elif self.window_type == 'expanding':
            # Returns Expanding GroupBy object
            return g.expanding(min_periods=self.min_periods)
        else:
            raise ValueError(f"Unsupported window type: {self.window_type} for axis 'ts'. "
                             f"Must be 'rolling', 'expanding', or 'fixed'.")

    def transform(self, X: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
        """
        Public interface for applying the ranking transformation.
        Performs checks and prepares data before calling the internal _transform method.
        """
        if not self._is_fitted:
            raise RuntimeError(f"Transform '{self.name}' must be fitted before calling transform()")

        # validate and create copy
        df_input = to_dataframe(X).copy(deep=True)
        self.validate_inputs(df_input)

        # transform and return
        return self._transform(df_input)

    def _transform(self, df: typing.Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
        """Apply the ranking transformation."""
        multiindex = isinstance(df.index, pd.MultiIndex)

        # Time series ranking
        if self.axis == 'ts':
            g = grouped(df)  # Group by second level if MultiIndex

            if self.window_type == 'rolling':
                rank = g.rolling(window=self.window_size, min_periods=self.min_periods).rank(pct=self.percentile)

            elif self.window_type == 'expanding':
                rank = g.expanding(min_periods=self.min_periods).rank(pct=self.percentile)

            elif self.window_type == 'fixed':
                rank = g.rank(pct=self.percentile)

            else:
                raise ValueError(f"Unsupported window_type: {self.window_type}")

            # If MultiIndex, stack the result to maintain the original structure
            rank = maybe_droplevel(rank, level=0)

        # Cross-sectional ranking
        elif self.axis == 'cs':

            g = grouped(df, axis='cs')  # Group by first level if MultiIndex
            if multiindex:
                rank = g.rank(pct=self.percentile)
            else:
                rank = g.rank(axis=1, pct=self.percentile)

        else:
            raise ValueError(f"Unsupported axis: {self.axis}")

        df[self.output_col] = rank[self.input_col]

        return df
