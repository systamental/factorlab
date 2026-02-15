import pandas as pd
from pandas.core.groupby import DataFrameGroupBy
from typing import Union, Optional, Any

from factorlab.utils import to_dataframe, grouped, maybe_droplevel
from factorlab.core.base_transform import BaseTransform


class Dispersion(BaseTransform):
    """
    A factory class for applying various dispersion/risk transformations.
    This class acts as a facade, delegating the computation to the
    appropriate, specific dispersion class.

    Notes
    -----
    This factory class retains the `compute` method for backward compatibility
    and simplicity, as it delegates instantiation and transformation to the
    specific, individual dispersion classes (e.g., StandardDeviation).
    """

    def __init__(self,
                 method: str = 'std',
                 **kwargs):
        super().__init__(name="Dispersion", description="A factory for various dispersion measures.")
        self.method = method
        self.kwargs = kwargs

        # Map method names to their corresponding classes
        self._method_map = {
            'std': StandardDeviation,
            'variance': Variance,
            'iqr': InterquartileRange,
            'mad': MedianAbsoluteDeviation,
            'min_max': MinMax,
            'quantile': Quantile,
            'atr': AverageTrueRange,
        }

        if self.method not in self._method_map:
            raise ValueError(f"Invalid method '{self.method}', must be one of {list(self._method_map.keys())}")

        self._transformer = self._method_map[self.method](**self.kwargs)

    @property
    def inputs(self) -> list[str]:
        """Required input columns for this transform, delegated to the specific return transformer."""
        return self._transformer.inputs

    def fit(self, X: Union[pd.Series, pd.DataFrame], y=None) -> 'Dispersion':
        """Fit the delegated return transformer. For stateless transforms, marks as fitted."""
        self._transformer.fit(X, y)
        self._is_fitted = True
        return self

    def transform(self, X: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
        """Apply the delegated return transformation."""
        if not self._is_fitted:
            raise RuntimeError("Returns transformer must be fitted before transform()")
        return self._transformer.transform(X)


class StandardDeviation(BaseTransform):
    """
    Computes the standard deviation (volatility) over time series or cross-section.

    Standard Deviation is a measure of the dispersion of a set of values from its mean.
    This is a stateless transform and is fully compatible with the
    fit/transform/fit_transform API for pipeline chaining.

    Parameters
    ----------
    input_col: str, default 'ret'
        Returns column to use when computing standard deviation.
    output_col: str, default 'std'
        Name of the output column to store the computed standard deviation.
    axis: str, {'ts', 'cs'}, default 'ts'
        Axis over which to compute the standard deviation:
        'ts' (time series): Rolling, expanding, or fixed standard deviation per asset.
        'cs' (cross-section): Standard deviation across assets at each point in time.
    window_type: str, {'ewm', 'rolling', 'expanding', 'fixed'}, default 'rolling'
        Type of window to apply the standard deviation calculation over for 'ts' axis.
        'fixed' computes standard deviation over the entire time series.
    window_size: int, default 30
        Size of the window for rolling computations. Ignored if window_type is 'expanding' or 'fixed'.
    min_periods: int, default 2
        Minimum number of observations in the window required to have a value.
    """

    def __init__(self,
                 input_col: str = 'ret',
                 output_col: str = 'std',
                 axis: str = 'ts',
                 window_type: str = 'rolling',
                 window_size: int = 30,
                 min_periods: int = 2):
        super().__init__(name="StandardDeviation",
                         description="Computes standard deviation (volatility) over time series or cross-section.")

        self.input_col = input_col
        self.output_col = output_col
        self.axis = axis
        self.window_type = window_type
        self.window_size = window_size
        self.min_periods = min_periods

    def fit(self,
            X: Union[pd.Series, pd.DataFrame],
            y: Optional[Union[pd.Series, pd.DataFrame]] = None) -> 'StandardDeviation':
        """Stateless fit: validates input and marks the transform as fitted."""
        df_input = to_dataframe(X)
        self.validate_inputs(df_input)
        self._is_fitted = True
        return self

    def _get_ts_window_op(self, g: Union[pd.DataFrame, DataFrameGroupBy]) -> Any:
        """Helper to determine and initialize the correct window operation for time series."""
        if self.window_type == 'ewm':
            # Returns Rolling GroupBy object
            return g.ewm(span=self.window_size, min_periods=self.min_periods)
        elif self.window_type == 'rolling':
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
        Public interface for applying the standard deviation transformation.
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
        """Core logic to apply the standard deviation transformation and append the new feature."""
        df = X
        multiindex = isinstance(df.index, pd.MultiIndex)

        if self.axis == 'ts':
            if self.window_type == 'fixed':

                if multiindex:
                    fixed_std_series = df.groupby(level=1)[self.input_col].std()
                    df[self.output_col] = df.index.get_level_values(1).map(fixed_std_series)
                else:
                    df[self.output_col] = df[self.input_col].std()
                return df

            # moving window
            g = grouped(df)
            window_op = self._get_ts_window_op(g)

            # apply std
            std_df = window_op.std()

            # drop multiindex level if present
            X[self.output_col] = maybe_droplevel(std_df[self.input_col], level=0)
            return X

        elif self.axis == 'cs':
            # Cross-Sectional (cs) logic: Standard Deviation across assets at each timestamp (group by level 0)
            if not multiindex:
                raise ValueError("Cross-sectional standard deviation ('cs') requires a MultiIndex DataFrame.")
            X[self.output_col] = df.groupby(level=0)[self.input_col].transform('std')
            return X

        else:
            raise ValueError(f"Unsupported axis: {self.axis}. Must be 'ts' (time series) or 'cs' (cross-section).")


class Quantile(BaseTransform):
    """
    Computes quantiles over time series or cross-section.

    This is a stateless transform and is fully compatible with the
    fit/transform/fit_transform API for pipeline chaining.
    """

    def __init__(self,
                 input_col: str = 'ret',
                 output_col: str = 'quantile',
                 q: float = 0.5,
                 axis: str = 'ts',
                 window_type: str = 'expanding',
                 window_size: int = 30,
                 min_periods: int = 1):
        super().__init__(name="Quantile", description="Computes quantiles over time series or cross-section.")

        self.input_col = input_col
        self.output_col = output_col
        self.q = q
        self.axis = axis
        self.window_type = window_type
        self.window_size = window_size
        self.min_periods = min_periods

    def fit(self, X: Union[pd.Series, pd.DataFrame], y: Optional[Union[pd.Series, pd.DataFrame]] = None) -> 'Quantile':
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
        Public interface for applying the standard deviation transformation.
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
        """Core logic to apply the standard deviation transformation and append the new feature."""
        df = X
        multiindex = isinstance(df.index, pd.MultiIndex)

        if self.axis == 'ts':
            if self.window_type == 'fixed':

                if multiindex:
                    fixed_std_series = df.groupby(level=1)[self.input_col].quantile(self.q)
                    df[self.output_col] = df.index.get_level_values(1).map(fixed_std_series)
                else:
                    df[self.output_col] = df[self.input_col].quantile(self.q)

                return df

            # moving window
            g = grouped(df)
            window_op = self._get_ts_window_op(g)

            # apply std
            quant_df = window_op.quantile(self.q)

            # drop multiindex level if present
            X[self.output_col] = maybe_droplevel(quant_df[self.input_col], level=0)
            return X

        elif self.axis == 'cs':
            # Cross-Sectional (cs) logic: Standard Deviation across assets at each timestamp (group by level 0)
            if not multiindex:
                raise ValueError("Cross-sectional standard deviation ('cs') requires a MultiIndex DataFrame.")
            X[self.output_col] = df.groupby(level=0)[self.input_col].transform('quantile', q=self.q)
            return X

        else:
            raise ValueError(f"Unsupported axis: {self.axis}. Must be 'ts' (time series) or 'cs' (cross-section).")


class InterquartileRange(BaseTransform):
    """
    Computes the interquartile range (IQR) over time series or cross-section,
    normalized to approximate standard deviation (IQR / 1.349).


    This is a stateless transform and is fully compatible with the
    fit/transform/fit_transform API for pipeline chaining.
    """

    def __init__(self,
                 input_col: str = 'ret',
                 output_col: str = 'iqr',
                 axis: str = 'ts',
                 window_type: str = 'expanding',
                 window_size: int = 30,
                 min_periods: int = 1):
        super().__init__(name="IQR",
                         description="Computes interquartile range (IQR) over time series or cross-section.")

        self.input_col = input_col
        self.output_col = output_col
        self.axis = axis
        self.window_type = window_type
        self.window_size = window_size
        self.min_periods = min_periods

    def fit(self,
            X: Union[pd.Series, pd.DataFrame],
            y: Optional[Union[pd.Series, pd.DataFrame]] = None) -> 'InterquartileRange':
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
        Public interface for applying the IQR transformation.
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
        """Core logic to apply the Interquartile Range transformation and append the new feature."""
        df = X
        multiindex = isinstance(df.index, pd.MultiIndex)

        if self.axis == 'ts':
            if self.window_type == 'fixed':
                # Time-series (ts) fixed window: IQR over the entire history (or by asset if MultiIndex)

                if multiindex:
                    # Calculate IQR for each asset (level=1)
                    q75 = df.groupby(level=1)[self.input_col].quantile(0.75)
                    q25 = df.groupby(level=1)[self.input_col].quantile(0.25)
                    iqr_series = q75 - q25
                    iqr_series /= 1.349  # Normalize
                    # Map the result back to the original index
                    df[self.output_col] = df.index.get_level_values(1).map(iqr_series)
                else:
                    # Calculate IQR for the single time series
                    iqr_val = df[self.input_col].quantile(0.75) - df[self.input_col].quantile(0.25)
                    df[self.output_col] = (iqr_val / 1.349)  # Normalize

                return df

            # moving window (rolling or expanding)
            g = grouped(df)
            window_op = self._get_ts_window_op(g)

            # Apply 75th and 25th quantiles
            q75 = window_op[self.input_col].quantile(0.75)
            q25 = window_op[self.input_col].quantile(0.25)
            iqr_df = q75 - q25

            # Normalize the IQR
            iqr_df /= 1.349

            # Drop multiindex level if present (for single asset group) and assign
            X[self.output_col] = maybe_droplevel(iqr_df, level=0)
            return X

        elif self.axis == 'cs':
            # Cross-Sectional (cs) logic: IQR across assets at each timestamp (group by level 0)
            if not multiindex:
                raise ValueError("Cross-sectional IQR ('cs') requires a MultiIndex DataFrame.")

            # Calculate Q75 and Q25 across assets (level 0) and subtract using transform
            q75 = df.groupby(level=0)[self.input_col].transform('quantile', q=0.75)
            q25 = df.groupby(level=0)[self.input_col].transform('quantile', q=0.25)

            iqr_series = q75 - q25
            iqr_series /= 1.349  # Normalize to std dev

            X[self.output_col] = iqr_series
            return X

        else:
            raise ValueError(f"Unsupported axis: {self.axis}. Must be 'ts' (time series) or 'cs' (cross-section).")


class MedianAbsoluteDeviation(BaseTransform):
    """
    Computes the median absolute deviation (MAD) over time series or cross-section,
    normalized to approximate standard deviation (MAD / 0.6745).

    This is a stateless transform and is fully compatible with the
    fit/transform/fit_transform API for pipeline chaining.
    """

    def __init__(self,
                 input_col: str = 'ret',
                 output_col: str = 'mad',
                 axis: str = 'ts',
                 window_type: str = 'rolling',
                 window_size: int = 30,
                 min_periods: int = 1):
        super().__init__(name="MedianAbsoluteDeviation",
                         description="Computes median absolute deviation (MAD) over time series or cross-section.")

        self.input_col = input_col
        self.output_col = output_col
        self.axis = axis
        self.window_type = window_type
        self.window_size = window_size
        self.min_periods = min_periods
        self._norm_factor = 0.6745  # Normalize MAD to std dev for normal distribution

    def fit(self,
            X: Union[pd.Series, pd.DataFrame],
            y: Optional[Union[pd.Series, pd.DataFrame]] = None) -> 'MedianAbsoluteDeviation':
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

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Core logic to apply the Median Absolute Deviation transformation and append the new feature."""
        df = X
        multiindex = isinstance(df.index, pd.MultiIndex)

        if self.axis == 'ts':

            if self.window_type == 'fixed':

                if multiindex:
                    # Calculate IQR for each asset (level=1)
                    med = df.groupby(level=1)[self.input_col].median()
                    # Align median back to original index
                    df['__median'] = df.index.get_level_values(1).map(med)
                    abs_dev_series = (df[self.input_col] - df['__median']).abs()
                    # Calculate median of absolute deviations per asset
                    mad_series = abs_dev_series.groupby(level=1).median()
                    # Normalize and map back
                    df[self.output_col] = df.index.get_level_values(1).map(mad_series / self._norm_factor)
                    df.drop(columns=['__median'], inplace=True)

                else:
                    # Single time series MAD
                    abs_dev_series = (df[self.input_col] - df[self.input_col].median()).abs()
                    mad_val = abs_dev_series.median()
                    df[self.output_col] = mad_val / self._norm_factor

                return df

            # moving window (rolling or expanding)
            g = grouped(df)
            window_op = self._get_ts_window_op(g)
            median_windowed = window_op[self.input_col].median()

            # drop level 0 and reindex the median result to the original df index
            median_realigned = maybe_droplevel(median_windowed, level=0).reindex(df.index)

            # abs deviation from the moving median
            abs_dev_df = (df[self.input_col] - median_realigned).abs().to_frame(self.input_col)

            # calculate the moving window median of the abs dev
            g_abs_dev = grouped(abs_dev_df)

            if self.window_type == 'rolling':
                mad_df = g_abs_dev.rolling(window=self.window_size, min_periods=self.min_periods).median()
            else:  # expanding
                mad_df = g_abs_dev.expanding(min_periods=self.min_periods).median()

            # Drop multiindex level if present and assign the normalized result
            mad_result = maybe_droplevel(mad_df[self.input_col], level=0)
            X[self.output_col] = mad_result / self._norm_factor
            return X

        elif self.axis == 'cs':
            # Cross-Sectional (cs) logic: MAD across assets at each timestamp (group by level 0)
            if not multiindex:
                raise ValueError("Cross-sectional MAD ('cs') requires a MultiIndex DataFrame.")

            # median per timestamp
            median_series = df.groupby(level=0)[self.input_col].transform('median')
            # abs dev
            abs_dev_series = (df[self.input_col] - median_series).abs()
            # median of abs dev
            mad_series = abs_dev_series.groupby(level=0).transform('median')
            # Normalize and assign
            X[self.output_col] = mad_series / self._norm_factor
            return X

        else:
            raise ValueError(f"Unsupported axis: {self.axis}. Must be 'ts' (time series) or 'cs' (cross-section).")

    def transform(self, X: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
        """
        Public interface for applying the MAD transformation.
        Performs checks and prepares data before calling the internal _transform method.
        """
        if not self._is_fitted:
            raise RuntimeError(f"Transform '{self.name}' must be fitted before calling transform()")

        # validate and create copy
        df_input = to_dataframe(X).copy(deep=True)
        self.validate_inputs(df_input)

        # transform and return
        return self._transform(df_input)


class Variance(BaseTransform):
    """
    Computes the variance over time series or cross-section.

    This is a stateless transform and is fully compatible with the
    fit/transform/fit_transform API for pipeline chaining.

    Parameters
    ----------
    input_col: str, default 'ret'
        Returns column to use when computing variance.
    output_col: str, default 'var'
        Name of the output column to store the computed variance.
    axis: str, {'ts', 'cs'}, default 'ts'
        Axis over which to compute the variance:
        'ts' (time series): Rolling, expanding, or fixed variance per asset.
        'cs' (cross-section): Variance across assets at each point in time.
    window_type: str, {'ewm', 'rolling', 'expanding', 'fixed'}, default 'rolling'
        Type of window to apply the variance calculation over for 'ts' axis.
        'fixed' computes variance over the entire time series.
    window_size: int, default 30
        Size of the window for rolling computations. Ignored if window_type is 'expanding' or 'fixed'.
    min_periods: int, default 1
        Minimum number of observations in the window required to have a value.

    """

    def __init__(self,
                 input_col: str = 'ret',
                 output_col: str = 'var',
                 axis: str = 'ts',
                 window_type: str = 'expanding',
                 window_size: int = 30,
                 min_periods: int = 1):
        super().__init__(name="Variance", description="Computes variance over time series or cross-section.")

        self.input_col = input_col
        self.output_col = output_col
        self.axis = axis
        self.window_type = window_type
        self.window_size = window_size
        self.min_periods = min_periods

    def fit(self,
            X: Union[pd.Series, pd.DataFrame],
            y: Optional[Union[pd.Series, pd.DataFrame]] = None) -> 'Variance':
        """Stateless fit: validates input and marks the transform as fitted."""
        df_input = to_dataframe(X)
        self.validate_inputs(df_input)
        self._is_fitted = True
        return self

    def _get_ts_window_op(self, g: Union[pd.DataFrame, DataFrameGroupBy]) -> Any:
        """Helper to determine and initialize the correct window operation for time series."""
        if self.window_type == 'ewm':
            # Returns EWM GroupBy object
            return g.ewm(span=self.window_size, min_periods=self.min_periods)
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
        Public interface for applying the variance transformation.
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
        """Core logic to apply the variance transformation and append the new feature."""
        df = X
        multiindex = isinstance(df.index, pd.MultiIndex)

        if self.axis == 'ts':
            if self.window_type == 'fixed':

                if multiindex:
                    fixed_var_series = df.groupby(level=1)[self.input_col].var()
                    df[self.output_col] = df.index.get_level_values(1).map(fixed_var_series)
                else:
                    df[self.output_col] = df[self.input_col].var()

                return df

            # moving window
            g = grouped(df)
            window_op = self._get_ts_window_op(g)

            # apply std
            var_df = window_op.var()

            # drop multiindex level if present
            X[self.output_col] = maybe_droplevel(var_df[self.input_col], level=0)
            return X

        elif self.axis == 'cs':
            # Cross-Sectional (cs) logic: variance across assets at each timestamp (group by level 0)
            if not multiindex:
                raise ValueError("Cross-sectional standard deviation ('cs') requires a MultiIndex DataFrame.")

            # Use transform('std') to calculate variance across assets at each date,
            # ensuring the result maintains the original index structure.
            X[self.output_col] = df.groupby(level=0)[self.input_col].transform('var')
            return X

        else:
            raise ValueError(f"Unsupported axis: {self.axis}. Must be 'ts' (time series) or 'cs' (cross-section).")


class MinMax(BaseTransform):
    """
    Computes the range (max - min) over time series or cross-section.

    This is a stateless transform and is fully compatible with the
    fit/transform/fit_transform API for pipeline chaining.
    """

    def __init__(self,
                 input_col: str = 'ret',
                 output_col: str = 'range',
                 axis: str = 'ts',
                 window_type: str = 'expanding',
                 window_size: int = 30,
                 min_periods: int = 1):
        super().__init__(name="Range", description="Computes range (max - min) over time series or cross-section.")

        self.input_col = input_col
        self.output_col = output_col
        self.axis = axis
        self.window_type = window_type
        self.window_size = window_size
        self.min_periods = min_periods

    def fit(self, X: Union[pd.Series, pd.DataFrame], y: Optional[Union[pd.Series, pd.DataFrame]] = None) -> 'MinMax':
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
        Public interface for applying the IQR transformation.
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
        """Core logic to apply the Range transformation and append the new feature."""
        df = X
        multiindex = isinstance(df.index, pd.MultiIndex)

        if self.axis == 'ts':
            if self.window_type == 'fixed':
                # Time-series (ts) fixed window: min-max over the entire history (or by asset if MultiIndex)

                if multiindex:
                    # Calculate IQR for each asset (level=1)
                    max_vals = df.groupby(level=1)[self.input_col].max()
                    min_vals = df.groupby(level=1)[self.input_col].min()
                    range_vals = max_vals - min_vals
                    # Map the result back to the original index
                    df[self.output_col] = df.index.get_level_values(1).map(range_vals)
                else:
                    # Calculate min-max for the single time series
                    range_vals = df[self.input_col].max() - df[self.input_col].min()
                    X[self.output_col] = range_vals

                return X

            # moving window (rolling or expanding)
            g = grouped(df)
            window_op = self._get_ts_window_op(g)

            # Apply 75th and 25th quantiles
            max_vals = window_op[self.input_col].max()
            min_vals = window_op[self.input_col].min()
            range_vals = max_vals - min_vals

            # Drop multiindex level if present (for single asset group) and assign
            X[self.output_col] = maybe_droplevel(range_vals, level=0)
            return X

        elif self.axis == 'cs':
            # Cross-Sectional (cs) logic: MinMax across assets at each timestamp (group by level 0)
            if not multiindex:
                raise ValueError("Cross-sectional MinMax ('cs') requires a MultiIndex DataFrame.")

            # Calculate Q75 and Q25 across assets (level 0) and subtract using transform
            max_vals = df.groupby(level=0)[self.input_col].transform('max')
            min_vals = df.groupby(level=0)[self.input_col].transform('min')
            range_vals = max_vals - min_vals

            X[self.output_col] = range_vals
            return X

        else:
            raise ValueError(f"Unsupported axis: {self.axis}. Must be 'ts' (time series) or 'cs' (cross-section).")


class AverageTrueRange(BaseTransform):
    """
    Computes the Average True Range (ATR) over a time series using OHLC data.

    This is a stateless transform and is fully compatible with the
    fit/transform/fit_transform API for pipeline chaining.

    Parameters
    ----------
    open_col: str, default 'open'
        Column name for the opening price.
    high_col: str, default 'high'
        Column name for the high price.
    low_col: str, default 'low'
        Column name for the low price.
    close_col: str, default 'close'
        Column name for the closing price.
    output_col: str, default 'atr'
        Name of the output column to store the computed ATR.
    window_type: str, {'ewm', 'rolling', 'expanding', 'fixed'}, default 'expanding'
        Type of window to apply the ATR calculation over.
        'fixed' computes ATR over the entire time series.
        'ewm' computes exponentially weighted moving average ATR.
    window_size: int, default 30
        Size of the window for rolling computations. Ignored if window_type is 'expanding' or 'fixed'.
    min_periods: int, default 1
        Minimum number of observations in the window required to have a value.

    """

    def __init__(self,
                 open_col: str = 'open',
                 high_col: str = 'high',
                 low_col: str = 'low',
                 close_col: str = 'close',
                 output_col: str = 'atr',
                 window_type: str = 'expanding',
                 window_size: int = 30,
                 min_periods: int = 1):
        super().__init__(name="AverageTrueRange", description="Computes Average True Range (ATR) from OHLC data.")

        self.open_col = open_col
        self.high_col = high_col
        self.low_col = low_col
        self.close_col = close_col
        self.output_col = output_col
        self.window_type = window_type
        self.window_size = window_size
        self.min_periods = min_periods

    @property
    def inputs(self) -> list[str]:
        """Required input columns for this transform."""
        return [self.open_col, self.high_col, self.low_col, self.close_col]

    def fit(self, X: Union[pd.Series, pd.DataFrame], y: Optional[Union[pd.Series, pd.DataFrame]] = None) -> 'MinMax':
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
        Public interface for applying the IQR transformation.
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
        if not self._is_fitted:
            raise RuntimeError(f"Transform '{self.name}' must be fitted before calling transform()")

        df = to_dataframe(X).copy()
        self.validate_inputs(df)

        # Unstack if MultiIndex
        is_multi = isinstance(df.index, pd.MultiIndex)
        df_ohlc = df.unstack() if is_multi else df

        # Compute True Range
        high = df_ohlc['high']
        low = df_ohlc['low']
        close = df_ohlc['close']

        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()

        if is_multi:
            # Re-stack for max comparison across columns, then combine
            tr = pd.concat([tr1.stack(), tr2.stack(), tr3.stack()], axis=1).max(axis=1)
        else:
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Compute ATR using selected window
        g = grouped(tr.to_frame('tr'))

        if self.window_type == 'ewm':
            res = g.ewm(span=self.window_size, min_periods=self.min_periods).mean()
        elif self.window_type == 'rolling':
            res = g.rolling(window=self.window_size, min_periods=self.min_periods).mean()
        elif self.window_type == 'expanding':
            res = g.expanding(min_periods=self.min_periods).mean()
        elif self.window_type == 'ewm':
            res = g.ewm(span=self.window_size, min_periods=self.min_periods).mean()
        elif self.window_type == 'fixed':
            res = g.mean().tr
            if is_multi:
                df[self.output_col] = df.index.get_level_values(1).map(res)
            else:
                df[self.output_col] = res
            return df
        else:
            raise ValueError(f"Unsupported window type: {self.window_type}")

        atr = maybe_droplevel(res, level=0)
        X[self.output_col] = atr.tr
        return X
