import pandas as pd
from pandas.core.groupby import DataFrameGroupBy
from typing import Union, Optional, Any, List

from factorlab.utils import maybe_droplevel, to_dataframe, grouped
from factorlab.core.base_transform import BaseTransform


class WindowSmoother(BaseTransform):
    """
    Applies smoothing techniques using rolling, expanding, or exponentially weighted (EWM) windows
    to a specified column. The calculation supports both single-asset (single-index) and
    multi-asset (MultiIndex) data, applying window operations group-wise for the latter.

    Parameters
    ----------
    input_col: str, default 'close'
        Column of the input DataFrame to apply smoothing to.
    window_type : str, {'rolling', 'expanding', 'ewm'}, default 'rolling'
        Type of window applied for smoothing.
    window_size : int, default 30
        Size (periods/span) of the rolling or EWM window. Ignored if `window_type` is 'expanding'.
    central_tendency : str, {'mean', 'median'}, default 'mean'
        Measure of central tendency to apply to the window. 'median' is not supported for 'ewm'.
    output_col: str or None, default None
        Name of the output column. If None, the name is generated automatically.
    window_fcn : str or None, default None
        Rolling window function (e.g. 'hann', 'gaussian') if `window_type` is 'rolling'.
    lags : int, default 0
        Number of periods to lag the final result. For MultiIndex, the lag is applied group-wise.
    kwargs : dict
        Additional arguments passed to the pandas window method (e.g., `min_periods` for rolling).
    """

    def __init__(self,
                 input_cols: Union[str, List[str]] = 'close',
                 output_cols: Union[str, List[str]] = None,
                 window_type: str = 'rolling',
                 window_size: int = 30,
                 central_tendency: str = 'mean',

                 window_fcn: Optional[str] = None,
                 lags: int = 0,
                 **kwargs):
        super().__init__(name="WindowSmoother", description="Applies rolling, expanding, or ewm smoothing.")

        self.input_cols = input_cols if isinstance(input_cols, list) else [input_cols]
        self.output_cols = output_cols if output_cols is not None else \
            [f"{col}_{window_type}_{central_tendency}_{window_size}" for col in self.input_cols]
        self.window_type = window_type
        self.window_size = window_size
        self.central_tendency = central_tendency
        self.window_fcn = window_fcn
        self.lags = lags
        self.kwargs = kwargs

    def fit(self, X: Union[pd.Series, pd.DataFrame], y: Optional[Union[pd.Series, pd.DataFrame]] = None) \
            -> 'WindowSmoother':
        """Stateless fit: validates input and marks the transform as fitted."""
        df_input = to_dataframe(X)
        self.validate_inputs(df_input)
        self._is_fitted = True
        return self

    def _get_window_op(self, g: Union[pd.DataFrame, DataFrameGroupBy]) -> Any:
        """Helper to determine and initialize the correct window operation."""
        if self.window_type == 'rolling':
            return g.rolling(window=self.window_size, win_type=self.window_fcn, **self.kwargs)
        elif self.window_type == 'expanding':
            return g.expanding(**self.kwargs)
        elif self.window_type == 'ewm':
            if self.central_tendency == 'median':
                raise ValueError("Median is not supported for ewm smoothing.")
            return g.ewm(span=self.window_size, **self.kwargs)
        else:
            raise ValueError(f"Unsupported window_type: {self.window_type}. Must be 'rolling', 'expanding', or 'ewm'.")

    def transform(self, X: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
        """
        Public interface for applying the smoothing transformation.
        Performs checks and prepares data before calling the internal _transform method.
        """
        if not self._is_fitted:
            raise RuntimeError(f"Transform '{self.name}' must be fitted before calling transform()")

        # validate input
        df_input = to_dataframe(X).copy(deep=True)
        self.validate_inputs(df_input)

        # transform
        return self._transform(df_input)

    def _transform(self, X: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
        """Apply the smoothing transformation and append the new feature column."""
        if not self._is_fitted:
            raise RuntimeError(f"Transform '{self.name}' must be fitted before calling transform()")

        df = X
        multiindex = isinstance(df.index, pd.MultiIndex)

        # group
        g = grouped(df)

        # window operation
        window_op = self._get_window_op(g)

        # central tendency
        if self.central_tendency not in ['mean', 'median']:
            raise ValueError(f"Unsupported central_tendency: {self.central_tendency}. Must be 'mean' or 'median'.")

        # smoothing
        smooth_df = getattr(window_op, self.central_tendency)()

        # drop level 0 if MultiIndex to align with original df
        smooth_series = maybe_droplevel(smooth_df, level=0)

        # lagging
        if self.lags > 0:
            if multiindex:
                smooth_series = smooth_series.groupby(level=1).shift(self.lags)
            else:
                smooth_series = smooth_series.shift(self.lags)

        # assign to output column
        df[self.output_cols] = smooth_series[self.input_cols]

        return df
