import pandas as pd
import numpy as np
from typing import Union, Optional

from sklearn.preprocessing import power_transform

from factorlab.transformations.dispersion import (
    AverageTrueRange,
    StandardDeviation,
    InterquartileRange,
    MedianAbsoluteDeviation,
    MinMax
)
from factorlab.transformations.returns import Difference
from factorlab.utils import to_dataframe, grouped, maybe_droplevel
from factorlab.core.base_transform import BaseTransform


class Normalization(BaseTransform):
    """
    A factory class for applying various normalization transformations.
    This class acts as a facade, delegating the computation to the
    appropriate, specific normalization class.
    """
    def __init__(self,
                 method: str = 'z-score',
                 **kwargs):
        """
        Constructor.

        Parameters
        ----------
        method: str, {'zscore', 'min_max', 'percentile', 'robust_zscore',
                      'mod_zscore', 'atr_scaler', 'power_transform'}, default 'zscore'
            The normalization method to use.
        **kwargs:
            Additional keyword arguments to pass to the specific normalization class.
        """
        super().__init__(name='Normalization', description='A factory for various normalization methods.')
        self.method = method
        self.kwargs = kwargs

        # Map method names to their corresponding classes
        self._method_map = {
            'zscore': ZScore,
            'robust_zscore': RobustZScore,
            'mod_zscore': ModZScore,
            'percentile': Percentile,
            'min_max': MinMaxScaler,
            'atr_scaler': ATRScaler,
            'power_transform': PowerTransform
        }

        if self.method not in self._method_map:
            raise ValueError(f"Invalid method '{self.method}', must be one of {list(self._method_map.keys())}")

        self._transformer = self._method_map[self.method](**self.kwargs)

    @property
    def inputs(self) -> list[str]:
        """Required input columns for this transform, delegated to the specific return transformer."""
        return self._transformer.inputs

    def fit(self, X: Union[pd.Series, pd.DataFrame], y=None) -> 'Normalization':
        """Fit the delegated return transformer. For stateless transforms, marks as fitted."""
        self._transformer.fit(X, y)
        self._is_fitted = True
        return self

    def transform(self, X: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
        """Apply the delegated return transformation."""
        if not self._is_fitted:
            raise RuntimeError("Returns transformer must be fitted before transform()")
        return self._transformer.transform(X)

    def compute(self, X: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
        """Backward-compatible compute method."""
        return super().compute(X)


class Center(BaseTransform):
    """
    Centers the data by subtracting a central tendency measure (mean, median or mode).

    This transformation is useful for normalizing data to have a central tendency of zero.

    Parameters
    ----------
    input_col : str, default 'close'
        The column to be centered.
    output_col : str, default 'center'
        The name for the computed centered column.
    method : str or callable, default 'mean'
        Method to compute the center: 'mean', 'median', or a custom callable.
    axis : str, {'ts', 'cs'}, default 'ts'
        Axis to center across: 'ts' = time series, 'cs' = cross-section.
    window_type : str, {'ewm', 'rolling', 'expanding', 'fixed'}, default 'expanding'
        Type of window: 'ewm', 'rolling', 'expanding', or 'fixed'.
    window_size : int, default 30
        Number of periods in the rolling window (ignored for fixed).
    min_periods : int, default 1
        Minimum number of observations in window required to have a value.
    """

    def __init__(self,
                 input_col: str = 'close',
                 output_col: str = 'center',
                 method: str = 'mean',
                 axis: str = 'ts',
                 window_type: str = 'expanding',
                 window_size: int = 30,
                 min_periods: int = 1):
        super().__init__(name="Center", description="Centers the data by subtracting a central tendency measure.")

        self.input_col = input_col
        self.output_col = output_col
        self.method = method
        self.axis = axis
        self.window_type = window_type
        self.window_size = window_size
        self.min_periods = min_periods
        
    def fit(self,
            X: Union[pd.Series, pd.DataFrame],
            y: Optional[Union[pd.Series, pd.DataFrame]] = None) -> 'Center':
        """Stateless fit: validates input and marks the transform as fitted."""
        df_input = to_dataframe(X)
        self.validate_inputs(df_input)
        self._is_fitted = True
        return self

    def transform(self, X: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
        """
        Public interface for applying the signal transformation.
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
        Computes the center for the input DataFrame or Series.
        """
        df = to_dataframe(X).copy()
        
        if self.axis == 'ts':
            g = grouped(df)

            if self.window_type == 'ewm':
                center = g.ewm(span=self.window_size, min_periods=self.min_periods).mean()

            elif self.window_type == 'rolling':
                center = getattr(g.rolling(window=self.window_size, min_periods=self.min_periods), self.method)()

            elif self.window_type == 'expanding':
                center = getattr(g.expanding(min_periods=self.min_periods), self.method)()

            elif self.window_type == 'fixed':
                center = getattr(g, self.method)()

            else:
                raise ValueError(f"Invalid window_type: {self.window_type}"
                                 f". Must be one of 'ewm', 'rolling', 'expanding', or 'fixed'.")

            # Handle MultiIndex by dropping the first level if necessary
            center = maybe_droplevel(center, level=0)
            # Center the data by subtracting the computed center
            centered = df - center
            df[self.output_col] = centered[self.input_col]
            return df

        elif self.axis == 'cs':
            # fixed window
            if isinstance(df.index, pd.MultiIndex):
                center = getattr(df.groupby(level=0), self.method)()
            else:
                center = getattr(df, self.method)(axis=1)

            # handle MultiIndex
            center = maybe_droplevel(center, level=0)
            # center
            centered = df.sub(center, axis=0)
            centered[self.output_col] = centered[self.input_col]
            return centered

        else:
            raise ValueError(f"Invalid axis: {self.axis}. Must be 'ts' or 'cs'.")


class ZScore(BaseTransform):
    """
    Normalizes the data by computing the z-score.

    This transformation is useful for standardizing data to have a mean of 0 and standard deviation of 1.

    Parameters
    ----------
    input_col : str, default 'close'
        The column to be normalized.
    output_col : str, default 'zscore'
        The name for the computed z-score column.
    axis : str, {'ts', 'cs'}, default 'ts'
        Axis to normalize across: 'ts' = time series, 'cs' = cross-section.
    centering : bool, default True
        Whether to center the data before normalization. If True, subtracts the mean.
        If False, uses the raw values.
    window_type : str, {'ewm', 'rolling', 'expanding', 'fixed'}, default 'expanding'
        Type of window: 'ewm', 'rolling', 'expanding', or 'fixed'.
    window_size : int, default 30
        Number of periods in the rolling window (ignored for fixed).
    min_periods : int, default 1
        Minimum number of observations in window required to have a value.
    winsorize : int, optional
        If specified, applies winsorization to the data after computing z-scores.
        This can help reduce the influence of outliers.
    """

    def __init__(self,
                 input_col: str = 'close',
                 output_col: str = 'zscore',
                 axis: str = 'ts',
                 centering: bool = True,
                 window_type: str = 'expanding',
                 window_size: int = 30,
                 min_periods: int = 2,
                 winsorize: Optional[int] = None):
        super().__init__(name="ZScore",
                         description="Normalizes the data by computing the z-score.")

        self.input_col = input_col
        self.output_col = output_col
        self.axis = axis
        self.centering = centering
        self.window_type = window_type
        self.window_size = window_size
        self.min_periods = min_periods
        self.winsorize = winsorize
        self.center_transformer = Center(input_col=self.input_col,
                                         output_col='center',
                                         axis=self.axis,
                                         method='mean',
                                         window_type=self.window_type,
                                         window_size=self.window_size,
                                         min_periods=self.min_periods)
        self.std_transformer = StandardDeviation(input_col=self.input_col,
                                                 output_col='std',
                                                 axis=self.axis,
                                                 window_type=self.window_type,
                                                 window_size=self.window_size,
                                                 min_periods=self.min_periods)
        
    def fit(self,
            X: Union[pd.Series, pd.DataFrame],
            y: Optional[Union[pd.Series, pd.DataFrame]] = None) -> 'ZScore':
        """Stateless fit: validates input and marks the transform as fitted."""
        df_input = to_dataframe(X)
        self.validate_inputs(df_input)
        self.center_transformer.fit(df_input)
        self.std_transformer.fit(df_input)
        self._is_fitted = True
        return self

    def transform(self, X: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
        """
        Public interface for applying the transformation.
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
        Computes the zscore for the input DataFrame or Series.
        """
        df = X

        # Center the data if specified
        if self.centering:
            df = self.center_transformer.transform(df)
        else:
            df['center'] = df[self.input_col]

        # Compute standard deviation
        df = self.std_transformer.transform(df)

        # Compute z-scores
        df[self.output_col] = df['center'] / df['std'].replace(0, np.nan)

        # Apply winsorization if specified
        if self.winsorize is not None:
            df[self.output_col] = df[self.output_col].clip(lower=-self.winsorize, upper=self.winsorize)

        return df


class RobustZScore(BaseTransform):
    """
    Computes a robust z-score using median and IQR (Interquartile Range).

    This transformation is useful for standardizing data while being less sensitive to outliers.

    Parameters
    ----------
    axis : str, {'ts', 'cs'}, default 'ts'
        Axis to normalize across: 'ts' = time series, 'cs' = cross-section.
    centering : bool, default True
        Whether to center the data before normalization. If True, subtracts the median. If False, uses the raw values.
    window_type : str, {'rolling', 'expanding', 'fixed'}, default 'expanding'
        Type of window: 'rolling', 'expanding', or 'fixed'.
    window_size : int, default 30
        Number of periods in the rolling window (ignored for fixed).
    min_periods : int, default 1
        Minimum number of observations in window required to have a value.
    winsorize : int, optional
        If specified, applies winsorization to the data after computing robust z-scores.
        This can help reduce the influence of outliers.
    """

    def __init__(self,
                 input_col: str = 'close',
                 output_col: str = 'zscore',
                 axis: str = 'ts',
                 centering: bool = True,
                 window_type: str = 'expanding',
                 window_size: int = 30,
                 min_periods: int = 1,
                 winsorize: Optional[int] = None):
        super().__init__(name="RobustZScore",
                         description="Computes a robust z-score using median and MAD.")

        self.input_col = input_col
        self.output_col = output_col
        self.axis = axis
        self.centering = centering
        self.window_type = window_type
        self.window_size = window_size
        self.min_periods = min_periods
        self.winsorize = winsorize
        self.center_transformer = Center(input_col=self.input_col,
                                         output_col='center',
                                         axis=self.axis,
                                         method='median',
                                         window_type=self.window_type,
                                         window_size=self.window_size,
                                         min_periods=self.min_periods)

        self.iqr_transformer = InterquartileRange(input_col=self.input_col,
                                                  output_col='iqr',
                                                  axis=self.axis,
                                                  window_type=self.window_type,
                                                  window_size=self.window_size,
                                                  min_periods=self.min_periods)
        
    def fit(self,
            X: Union[pd.Series, pd.DataFrame],
            y: Optional[Union[pd.Series, pd.DataFrame]] = None) -> 'RobustZScore':
        """Stateless fit: validates input and marks the transform as fitted."""
        df_input = to_dataframe(X)
        self.validate_inputs(df_input)
        self.center_transformer.fit(df_input)
        self.iqr_transformer.fit(df_input)
        self._is_fitted = True
        return self

    def transform(self, X: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
        """
        Public interface for applying the transformation.
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
        Computes the robust zscore for the input DataFrame or Series.
        """
        df = X

        # Center the data if specified
        if self.centering:
            df = self.center_transformer.transform(df)
        else:
            df['center'] = df[self.input_col]

        # Compute standard deviation
        df = self.iqr_transformer.transform(df)

        # Compute z-scores
        df[self.output_col] = df['center'] / df['iqr'].replace(0, np.nan)

        # Apply winsorization if specified
        if self.winsorize is not None:
            df[self.output_col] = df[self.output_col].clip(lower=-self.winsorize, upper=self.winsorize)

        return df


class ModZScore(BaseTransform):
    """
    Computes the modified z-score using median and MAD (Median Absolute Deviation).

    This transformation is useful for standardizing data while being robust to outliers.

    Parameters
    ----------
    input_col : str, default 'close'
        The column to be normalized.
    output_col : str, default 'zscore'
        The name for the computed modified z-score column.
    axis : str, {'ts', 'cs'}, default 'ts'
        Axis to normalize across: 'ts' = time series, 'cs' = cross-section.
    centering : bool, default True
        Whether to center the data before normalization. If True, subtracts the median.
    window_type : str, {'ewm', 'rolling', 'expanding', 'fixed'}, default 'expanding'
        Type of window: 'ewm', 'rolling', 'expanding', or 'fixed'.
    window_size : int, default 30
        Number of periods in the rolling window (ignored for fixed).
    min_periods : int, default 1
        Minimum number of observations in window required to have a value.
    winsorize : int, optional
        If specified, applies winsorization to the data after computing modified z-scores.
        This can help reduce the influence of outliers.
    """

    def __init__(self,
                 input_col: str = 'close',
                 output_col: str = 'zscore',
                 axis: str = 'ts',
                 centering: bool = True,
                 window_type: str = 'expanding',
                 window_size: int = 30,
                 min_periods: int = 1,
                 winsorize: Optional[int] = None):
        super().__init__(name="ModZScore",
                         description="Computes the modified z-score using median and MAD.")

        self.input_col = input_col
        self.output_col = output_col
        self.axis = axis
        self.centering = centering
        self.window_type = window_type
        self.window_size = window_size
        self.min_periods = min_periods
        self.winsorize = winsorize
        self.center_transformer = Center(input_col=self.input_col,
                                         output_col='center',
                                         axis=self.axis,
                                         method='median',
                                         window_type=self.window_type,
                                         window_size=self.window_size,
                                         min_periods=self.min_periods)
        self.mad_transformer = MedianAbsoluteDeviation(input_col=self.input_col,
                                                       output_col='mad',
                                                       axis=self.axis,
                                                       window_type=self.window_type,
                                                       window_size=self.window_size,
                                                       min_periods=self.min_periods)
        
    def fit(self,
            X: Union[pd.Series, pd.DataFrame],
            y: Optional[Union[pd.Series, pd.DataFrame]] = None) -> 'ModZScore':
        """Stateless fit: validates input and marks the transform as fitted."""
        df_input = to_dataframe(X)
        self.validate_inputs(df_input)
        self.center_transformer.fit(df_input)
        self.mad_transformer.fit(df_input)
        self._is_fitted = True
        return self

    def transform(self, X: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
        """
        Public interface for applying the transformation.
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
        Computes the modified zscore for the input DataFrame or Series.
        """
        df = X

        # Center the data if specified
        if self.centering:
            df = self.center_transformer.transform(df)
        else:
            df['center'] = df[self.input_col]

        # Compute standard deviation
        df = self.mad_transformer.transform(df)

        # Compute z-scores
        df[self.output_col] = df['center'] / df['mad'].replace(0, np.nan)

        # Apply winsorization if specified
        if self.winsorize is not None:
            df[self.output_col] = df[self.output_col].clip(lower=-self.winsorize, upper=self.winsorize)

        return df


class Percentile(BaseTransform):
    """
    Computes the specified percentile over a time series or cross-section.

    Parameters
    ----------
    input_col : str, default 'close'
        The column to compute percentiles on.
    output_col : str, default 'percentile'
        The name for the computed percentile column.
    axis : str, {'ts', 'cs'}, default 'ts'
        Whether to compute time series ('ts') or cross-sectional ('cs') percentiles.
    window_type : str, {'rolling', 'expanding', 'fixed'}, default 'expanding'
        Type of window: 'rolling', 'expanding', or 'fixed'.
    window_size : int, default 30
        Number of periods in the rolling.
    min_periods : int, default 1
        Minimum number of observations in window required to have a value.
    """

    def __init__(self,
                 input_col: str = 'close',
                 output_col: str = 'percentile',
                 axis: str = 'ts',
                 window_type: str = 'expanding',
                 window_size: int = 30,
                 min_periods: int = 1):
        super().__init__(name="Percentile",
                         description="Computes percentile rank over time series or cross-section.")

        self.input_col = input_col
        self.output_col = output_col
        self.axis = axis
        self.window_type = window_type
        self.window_size = window_size
        self.min_periods = min_periods
        
    def fit(self,
            X: Union[pd.Series, pd.DataFrame],
            y: Optional[Union[pd.Series, pd.DataFrame]] = None) -> 'Percentile':
        """Stateless fit: validates input and marks the transform as fitted."""
        df_input = to_dataframe(X)
        self.validate_inputs(df_input)
        self._is_fitted = True
        return self

    def transform(self, X: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
        """
        Public interface for applying the transformation.
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
        Computes the percentile for the input DataFrame or Series.
        """
        df = X
        multiindex = isinstance(df.index, pd.MultiIndex)

        if self.axis == 'ts':
            g = grouped(df)

            if self.window_type == 'rolling':
                res = g.rolling(window=self.window_size, min_periods=self.min_periods).rank(pct=True)

            elif self.window_type == 'expanding':
                res = g.expanding(min_periods=self.min_periods).rank(pct=True)

            elif self.window_type == 'fixed':
                res = g.rank(pct=True)

            else:
                raise ValueError(f"Unsupported window type: {self.window_type}")

            # Drop level if MultiIndex
            if self.window_type != 'fixed':
                res = maybe_droplevel(res, level=0)

            df[self.output_col] = res[self.input_col]
            return df

        elif self.axis == 'cs':

            if not multiindex:
                raise ValueError("Cross-sectional percentile ('cs') requires a MultiIndex DataFrame.")
            else:
                percentile = df.groupby(level=0).rank(pct=True)

            df[self.output_col] = percentile[self.input_col]

            return df


class MinMaxScaler(BaseTransform):
    """
    Scales the data to a specified range, typically [0, 1].

    This transformation is useful for normalizing data to a common scale.

    Parameters
    ----------
    input_col : str, default 'close'
        The column to be scaled.
    output_col : str, default 'range'
        The name for the computed scaled column.
    axis : str, {'ts', 'cs'}, default 'ts'
        Axis to normalize across: 'ts' = time series, 'cs' = cross-section.
    centering : bool, default True
        Whether to center the data before scaling. If True, subtracts the minimum.
    window_type : str, {'ewm', 'rolling', 'expanding', 'fixed'}, default 'expanding'
        Type of window: 'ewm', 'rolling', 'expanding', or 'fixed'.
    window_size : int, default 30
        Number of periods in the rolling window (ignored for fixed).
    min_periods : int, default 1
        Minimum number of observations in window required to have a value.
    """

    def __init__(self,
                 input_col: str = 'close',
                 output_col: str = 'min_max_scaled',
                 axis: str = 'ts',
                 centering: bool = True,
                 window_type: str = 'expanding',
                 window_size: int = 30,
                 min_periods: int = 1):
        super().__init__(name="MinMaxScaler", description="Scales the data to a specified range.")

        self.input_col = input_col
        self.output_col = output_col
        self.axis = axis
        self.centering = centering
        self.window_type = window_type
        self.window_size = window_size
        self.min_periods = min_periods
        self.center_transformer = Center(input_col=self.input_col,
                                         output_col='center',
                                         axis=self.axis,
                                         method='min',
                                         window_type=self.window_type,
                                         window_size=self.window_size,
                                         min_periods=self.min_periods)
        self.range_transformer = MinMax(input_col=self.input_col,
                                        output_col='range',
                                        axis=self.axis,
                                        window_type=self.window_type,
                                        window_size=self.window_size,
                                        min_periods=self.min_periods)

    def fit(self,
            X: Union[pd.Series, pd.DataFrame],
            y: Optional[Union[pd.Series, pd.DataFrame]] = None) -> 'MinMaxScaler':
        """Stateless fit: validates input and marks the transform as fitted."""
        df_input = to_dataframe(X)
        self.validate_inputs(df_input)
        self.center_transformer.fit(df_input)
        self.range_transformer.fit(df_input)
        self._is_fitted = True
        return self

    def transform(self, X: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
        """
        Public interface for applying the transformation.
        Performs checks and prepares data before calling the internal _transform method.
        """
        if not self._is_fitted:
            raise RuntimeError(f"Transform '{self.name}' must be fitted before calling transform()")

        # validate and create copy
        df_input = to_dataframe(X).copy(deep=True)
        self.validate_inputs(df_input)

        # transform and return
        return self._transform(df_input)
    
    def _transform(self, X: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
        """
        Computes the min-max scaled values for the input DataFrame or Series.
        """
        df = X

        # Center the data if specified
        if self.centering:
            df = self.center_transformer.transform(df)
        else:
            df['center'] = df[self.input_col]

        # Compute standard deviation
        df = self.range_transformer.transform(df)

        # Compute z-scores
        df[self.output_col] = df['center'] / df['range'].replace(0, np.nan)
        # clip values to the range [0, 1]
        df[self.output_col] = df[self.output_col] .clip(lower=0, upper=1)

        return df


class ATRScaler(BaseTransform):
    """
    Scales the data using the Average True Range (ATR).

    This transformation is useful for normalizing data based on volatility.

    Parameters
    ----------
    window_type : str, {'ewm', 'rolling', 'expanding', 'fixed'}, default 'expanding'
        Type of window: 'ewm', 'rolling', 'expanding', or 'fixed'.
    window_size : int, default 30
        Number of periods in the rolling window (ignored for fixed).
    min_periods : int, default 1
        Minimum number of observations in window required to have a value.
    """

    def __init__(self,
                 input_col: str = 'close',
                 output_col: str = 'atr_scaled',
                 centering: bool = True,
                 window_type: str = 'expanding',
                 window_size: int = 30,
                 min_periods: int = 1):
        super().__init__(name="ATRScaler", description="Scales the data using the Average True Range (ATR).")

        self.input_col = input_col
        self.output_col = output_col
        self.centering = centering
        self.window_type = window_type
        self.window_size = window_size
        self.min_periods = min_periods
        self._diff_transformer = Difference(input_col=self.input_col,
                                            output_col='diff')
        self.center_transformer = Center(input_col='diff',
                                         output_col='center',
                                         method='median',
                                         window_type=self.window_type,
                                         window_size=self.window_size,
                                         min_periods=self.min_periods)
        self.atr_transformer = AverageTrueRange(output_col='atr',
                                                window_type=self.window_type,
                                                window_size=self.window_size,
                                                min_periods=self.min_periods)
        
    def fit(self,
            X: Union[pd.Series, pd.DataFrame],
            y: Optional[Union[pd.Series, pd.DataFrame]] = None) -> 'ATRScaler':
        """Stateless fit: validates input and marks the transform as fitted."""
        df_input = to_dataframe(X)
        self.validate_inputs(df_input)
        self._diff_transformer.fit(df_input)
        self.center_transformer.fit(df_input)
        self.atr_transformer.fit(df_input)
        self._is_fitted = True
        return self

    def transform(self, X: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
        """
        Public interface for applying the transformation.
        Performs checks and prepares data before calling the internal _transform method.
        """
        if not self._is_fitted:
            raise RuntimeError(f"Transform '{self.name}' must be fitted before calling transform()")

        # validate and create copy
        df_input = to_dataframe(X).copy(deep=True)
        self.validate_inputs(df_input)

        # transform and return
        return self._transform(df_input)
    
    def _transform(self, X: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
        """
        Computes the ATR scaled values for the input DataFrame or Series.
        """
        df = X

        # Compute the difference
        df = self._diff_transformer.transform(df)

        # Center the data if specified
        if self.centering:
            df = self.center_transformer.transform(df)
        else:
            df['center'] = df[self.input_col]

        # Compute ATR
        df = self.atr_transformer.transform(df)

        # Compute ATR scaled values
        df[self.output_col] = df['center'] / df['atr'].replace(0, np.nan)

        return df


class PowerTransform(BaseTransform):
    """
    Applies power transformations ('box-cox' or 'yeo-johnson') to time series or cross-sectional data.

    Parameters
    ----------
    input_col : str, default 'zscore'
        The column to be transformed.
    output_col : str, default 'power_zscore'
        The name for the computed transformed column.
    method : str, {'box-cox', 'yeo-johnson'}, default 'box-cox'
        Power transformation method.
    axis : str, {'ts', 'cs'}, default 'ts'
        Axis along which to apply the transformation.
    window_type : str, {'rolling', 'expanding', 'fixed'}, default 'expanding'
        Type of window applied to the transformation.
    window_size : int, default 30
        Size of the moving window.
    min_periods : int, default 2
        Minimum periods for rolling/expanding windows.
    adjustment : float, default 1e-6
        Adjustment for non-positive values (required for box-cox).
    """

    def __init__(self,
                 input_col: str = 'zscore',
                 output_col: str = 'power_zscore',
                 method: str = 'box-cox',
                 axis: str = 'ts',
                 window_type: str = 'expanding',
                 window_size: int = 30,
                 min_periods: int = 2,
                 adjustment: float = 1e-6):
        super().__init__(name="PowerTransform", description=f"Applies {method} transformation.")

        self.input_col = input_col
        self.output_col = output_col
        self.method = method
        self.axis = axis
        self.window_type = window_type
        self.window_size = window_size
        self.min_periods = min_periods
        self.adjustment = adjustment
        
    def fit(self,
            X: Union[pd.Series, pd.DataFrame],
            y: Optional[Union[pd.Series, pd.DataFrame]] = None) -> 'PowerTransform':
        """Stateless fit: validates input and marks the transform as fitted."""
        df_input = to_dataframe(X)
        self.validate_inputs(df_input)
        self._is_fitted = True
        return self

    def transform(self, X: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
        """
        Public interface for applying the transformation.
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
        Applies the power transformation to the input DataFrame or Series.
        """
        df = X[[self.input_col]]
        multiindex = isinstance(df.index, pd.MultiIndex)

        if self.axis == 'ts':
            df_unstacked = df.unstack() if multiindex else df
            df_transformed = pd.DataFrame(index=df_unstacked.index, columns=df_unstacked.columns)

            for col in df_unstacked.columns:
                series = df_unstacked[col]

                if self.window_type == 'rolling':
                    out = []
                    for i in range(self.window_size, len(series) + 1):
                        window = series.iloc[i - self.window_size:i]
                        adjusted = (window - window.min() + self.adjustment).to_frame() \
                            if self.method == 'box-cox' else window.to_frame()
                        transformed = power_transform(adjusted, method=self.method, standardize=True).flatten()
                        out.append(transformed[-1])
                    df_transformed.loc[df_transformed.index[-len(out):], col] = out

                elif self.window_type == 'expanding':
                    out = []
                    for i in range(self.min_periods, len(series) + 1):
                        window = series.iloc[:i]
                        adjusted = (window - window.min() + self.adjustment).to_frame() \
                            if self.method == 'box-cox' else window.to_frame()
                        transformed = power_transform(adjusted, method=self.method, standardize=True).flatten()
                        out.append(transformed[-1])
                    df_transformed.loc[df_transformed.index[-len(out):], col] = out

                elif self.window_type == 'fixed':
                    adjusted = (series - series.min() + self.adjustment).to_frame() \
                        if self.method == 'box-cox' else series.to_frame()
                    transformed = power_transform(adjusted, method=self.method, standardize=True).flatten()
                    df_transformed[col] = transformed

                else:
                    raise ValueError(f"Unsupported window_type: {self.window_type}")

            df_out = df_transformed.stack(future_stack=True).sort_index() if multiindex else df_transformed
            X[self.output_col] = df_out[self.input_col].astype(float)
            return X

        elif self.axis == 'cs':

            if multiindex:
                if self.method == 'box-cox':
                    df = df - df.groupby(level=0).min() + self.adjustment
                df = df.groupby(level=0, group_keys=False).apply(
                    lambda x: pd.DataFrame(
                        power_transform(x, method=self.method, standardize=True, copy=True),
                        index=x.index, columns=x.columns
                    )
                )
            else:
                raise ValueError("Cross-sectional power transformation ('cs') requires a MultiIndex DataFrame.")

            X[self.output_col] = df[self.input_col].astype(float)
            return X

        else:
            raise ValueError(f"Unsupported axis: {self.axis}")
