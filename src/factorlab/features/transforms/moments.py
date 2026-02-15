import pandas as pd
from typing import Union, Optional, Any
from factorlab.utils import to_dataframe, grouped, maybe_droplevel
from factorlab.core.base_transform import BaseTransform


class Skewness(BaseTransform):
    """
    Computes the skewness of a time series or cross-section of returns.

    Skewness is a measure of the asymmetry of the probability distribution of a
    real-valued random variable about its mean.

    Parameters
    ----------
    input_col: str, default 'ret'
        Returns column to use when computing skewness.
    axis: str, {'ts', 'cs'}, default 'ts'
        Axis over which to compute the skewness:
        'ts' (time series): Rolling, expanding, or fixed skewness per asset.
        'cs' (cross-section): Skewness across assets at each point in time.
    window_type: str, {'rolling', 'expanding', 'fixed'}, default 'rolling'
        Type of window to apply the skewness calculation over for 'ts' axis.
        'fixed' computes skewness over the entire time series.
    window_size: int, default 30
        Size of the window for rolling computations. Ignored if window_type is 'expanding' or 'fixed'.
    min_periods: int, default 2
        Minimum number of observations in the window required to have a value.
    """

    def __init__(self,
                 input_col: str = 'ret',
                 output_col: str = 'skew',
                 axis: str = 'ts',
                 window_type: str = 'rolling',
                 window_size: int = 30,
                 min_periods: int = 2):
        super().__init__(name="Skewness",
                         description="Computes the skewness of a return series.")

        self.input_col = input_col
        self.axis = axis
        self.window_type = window_type
        self.window_size = window_size
        self.min_periods = min_periods
        self.output_col = output_col

    def fit(self, X: Union[pd.Series, pd.DataFrame], y: Optional[Union[pd.Series, pd.DataFrame]] = None) -> 'Skewness':
        """Stateless fit: validates input and marks the transform as fitted."""
        df_input = to_dataframe(X)
        self.validate_inputs(df_input)
        self._is_fitted = True
        return self

    def _get_ts_window_op(self, g: Union[pd.DataFrame, pd.core.groupby.GroupBy]) -> Any:
        """Helper to determine and initialize the correct window operation for time series."""
        if self.window_type == 'rolling':
            # Returns Rolling GroupBy object
            return g.rolling(window=self.window_size, min_periods=self.min_periods)
        elif self.window_type == 'expanding':
            # Returns Expanding GroupBy object
            return g.expanding(min_periods=self.min_periods)
        else:
            raise ValueError(f"Unsupported window type: {self.window_type} for axis 'ts'.")

    def transform(self, X: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
        """
        Public interface for applying the skewness transformation.
        Performs checks and prepares data before calling the internal _transform method.
        """
        if not self._is_fitted:
            raise RuntimeError(f"Transform '{self.name}' must be fitted before calling transform()")
        df_input = to_dataframe(X).copy(deep=True)
        self.validate_inputs(df_input)

        return self._transform(df_input)

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Core logic to apply the skewness transformation and append the new feature."""
        df = X
        multiindex = isinstance(df.index, pd.MultiIndex)

        if self.axis == 'ts':
            if self.window_type == 'fixed':

                if multiindex:
                    fixed_skew_series = df.groupby(level=1)[self.input_col].skew()
                    df[self.output_col] = df.index.get_level_values(1).map(fixed_skew_series)
                else:
                    df[self.output_col] = df[self.input_col].skew()

                return df

            # rolling or expanding time series skewness
            g = grouped(df)
            window_op = self._get_ts_window_op(g)

            skew_df = window_op.skew()

            # drop level 0 if MultiIndex to align results
            df[self.output_col] = maybe_droplevel(skew_df[self.input_col], level=0)
            return df

        elif self.axis == 'cs':
            # Cross-Sectional (cs) logic: Skewness at each timestamp (group by level 0)
            if not multiindex:
                raise ValueError("Cross-sectional skewness ('cs') requires a MultiIndex DataFrame.")
            df[self.output_col] = df.groupby(level=0)[self.input_col].transform('skew')
            return df

        else:
            raise ValueError(f"Unsupported axis: {self.axis}. Must be 'ts' (time series) or 'cs' (cross-section).")


class Kurtosis(BaseTransform):
    """
    Computes the kurtosis of a time series or cross-section of returns.

    Kurtosis is a measure of the "tailedness" of the probability distribution of a
    real-valued random variable.

    Parameters
    ----------
    input_col: str, default 'ret'
        Returns column to use when computing kurtosis.
    axis: str, {'ts', 'cs'}, default 'ts'
        Axis over which to compute the kurtosis:
        'ts' (time series): Rolling, expanding, or fixed kurtosis per asset.
        'cs' (cross-section): Kurtosis across assets at each point in time.
    window_type: str, {'rolling', 'expanding', 'fixed'}, default 'rolling'
        Type of window to apply the kurtosis calculation over for 'ts' axis.
        'fixed' computes kurtosis over the entire time series.
    window_size: int, default 30
        Size of the window for rolling computations. Ignored if window_type is 'expanding' or 'fixed'.
    min_periods: int, default 2
        Minimum number of observations in the window required to have a value.
    """

    def __init__(self,
                 input_col: str = 'ret',
                 output_col: str = 'kurt',
                 axis: str = 'ts',
                 window_type: str = 'rolling',
                 window_size: int = 30,
                 min_periods: int = 2):
        super().__init__(name="Kurtosis",
                         description="Computes the kurtosis of a time series.")

        self.input_col = input_col
        self.axis = axis
        self.window_type = window_type
        self.window_size = window_size
        self.min_periods = min_periods
        self.output_col = output_col

    def fit(self, X: Union[pd.Series, pd.DataFrame], y: Optional[Union[pd.Series, pd.DataFrame]] = None) -> 'Kurtosis':
        """Stateless fit: validates input and marks the transform as fitted."""
        df_input = to_dataframe(X)
        self.validate_inputs(df_input)
        self._is_fitted = True
        return self

    def _get_ts_window_op(self, g: Union[pd.DataFrame, pd.core.groupby.GroupBy]) -> Any:
        """Helper to determine and initialize the correct window operation for time series."""
        if self.window_type == 'rolling':
            # Returns Rolling GroupBy object
            return g.rolling(window=self.window_size, min_periods=self.min_periods)
        elif self.window_type == 'expanding':
            # Returns Expanding GroupBy object
            return g.expanding(min_periods=self.min_periods)
        else:
            raise ValueError(f"Unsupported window type: {self.window_type} for axis 'ts'.")

    def transform(self, X: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
        """
        Public interface for applying the kurtosis transformation.
        Performs checks and prepares data before calling the internal _transform method.
        """
        if not self._is_fitted:
            raise RuntimeError(f"Transform '{self.name}' must be fitted before calling transform()")

        # validation
        df_input = to_dataframe(X).copy(deep=True)
        self.validate_inputs(df_input)

        # transform
        return self._transform(df_input)

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Core logic to apply the kurtosis transformation and append the new feature."""
        df = X
        multiindex = isinstance(df.index, pd.MultiIndex)

        if self.axis == 'ts':
            if self.window_type == 'fixed':
                # Fixed time series kurtosis (calculates kurtosis over all periods per asset)

                if multiindex:
                    # Calculate fixed kurtosis for each group (ticker)
                    fixed_kurt_series = df.groupby(level=1).apply(pd.DataFrame.kurt)[self.input_col]
                    # Map the single kurtosis value back to every row of its corresponding group
                    df[self.output_col] = df.index.get_level_values(1).map(fixed_kurt_series)
                else:
                    # Calculate fixed kurtosis for the whole series and replicate it
                    df[self.output_col] = df[self.input_col].kurt()

                return df

            # Rolling or Expanding Time Series Kurtosis
            g = grouped(df)
            window_op = self._get_ts_window_op(g)

            kurt_df = window_op.kurt()

            # Align the result by dropping level 0 if MultiIndex
            df[self.output_col] = maybe_droplevel(kurt_df[self.input_col], level=0)
            return df

        elif self.axis == 'cs':
            # Cross-Sectional (cs) logic: Kurtosis at each timestamp (group by level 0)
            if not multiindex:
                raise ValueError("Cross-sectional kurtosis ('cs') requires a MultiIndex DataFrame.")
            df[self.output_col] = df.groupby(level=0)[self.input_col].transform(lambda x: x.kurt())
            return df

        else:
            raise ValueError(f"Unsupported axis: {self.axis}. Must be 'ts' (time series) or 'cs' (cross-section).")
