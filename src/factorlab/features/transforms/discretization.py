import pandas as pd
import numpy as np
from typing import Union
from factorlab.utils import to_dataframe
from factorlab.core.base_transform import BaseTransform
from factorlab.transformations.normalization import Percentile
from sklearn.preprocessing import KBinsDiscretizer


class Quantize(BaseTransform):
    """
    Quantizes features into discrete bins based on percentiles.

    This transformation is useful for converting continuous variables into discrete categories. It computes the
    percentiles of the input data and then maps these percentiles to discrete bins.

    Parameters
    ----------
    input_col: str
        The name of the input column to be quantized. Default is 'zscore'.
    output_col: str
        The name of the output column to store the quantized values. Default is 'quantile'.
    axis: str
        The axis along which to compute the quantization. Default is 'ts' (time series).
    bins: int
        The number of discrete bins to create. Must be greater than 1.
    window_type: str
        The type of window to use for computing percentiles. Default is 'expanding'.
    window_size: int
        The size of the window for computing percentiles. Default is 30.
    min_periods: int
        The minimum number of observations in the window required to compute a percentile. Default is 2.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the quantized values.
    """
    def __init__(self,
                 input_col: str = 'zscore',
                 output_col: str = 'quantile',
                 axis: str = 'ts',
                 bins: int = 5,
                 window_type: str = 'expanding',
                 window_size: int = 30,
                 min_periods: int = 2):
        super().__init__(name="Quantize", description="Quantizes features into discrete bins.")
        
        if bins <= 1:
            raise ValueError("bins must be greater than 1.")
        self.input_col = input_col
        self.output_col = output_col
        self.axis = axis
        self.bins = bins
        self.window_type = window_type
        self.window_size = window_size
        self.min_periods = min_periods
        self.pct_transformer = Percentile(input_col=self.input_col,
                                          output_col=self.output_col,
                                          axis=self.axis,
                                          window_type=self.window_type,
                                          window_size=self.window_size,
                                          min_periods=self.min_periods)

    def fit(self, X: Union[pd.Series, pd.DataFrame],
            y: Union[pd.Series, pd.DataFrame] = None) -> 'Quantize':
        """Stateless fit: validates input and marks the transform as fitted."""
        df_input = to_dataframe(X)
        self.validate_inputs(df_input)
        self.pct_transformer.fit(df_input)
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
        """Apply the quantization transformation."""
        df = X

        # compute percentiles
        perc_df = self.pct_transformer.transform(df)

        # map percentiles to bins
        out = (perc_df * self.bins).apply(np.ceil).astype(float)

        df[self.output_col] = out[self.output_col]
        
        return df


class Discretize(BaseTransform):
    """
    Discretizes continuous features into ordinal bins using KBinsDiscretizer.

    Supports time-series and cross-sectional discretization over rolling, expanding, or fixed windows.
    This transformation is useful for converting continuous variables into discrete categories, which can be beneficial
    for certain types of financial analysis or modeling.

    Parameters
    ----------
    input_col: str
        The name of the input column to be discretized. Default is 'zscore'.
    output_col: str
        The name of the output column to store the discretized values. Default is 'discretized'.
    bins: int
        The number of bins to discretize the data into. Must be greater than 1.
    axis: str
        The axis along which to apply the discretization. Can be 'ts' (time series) or 'cs' (cross-sectional).
    method: str
        The method to use for binning. Options are 'quantile', 'uniform', or 'kmeans'.
    window_type: str
        The type of window to use for discretization. Options are 'rolling', 'expanding', or 'fixed'.
    window_size: int
        The size of the window for rolling or expanding discretization. Default is 30.
    min_obs: int
        The minimum number of observations required for expanding discretization. Default is 1.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the discretized values, with each value incremented by 1 to ensure positive bin indices.
    """

    def __init__(self,
                 input_col: str = 'zscore',
                 output_col: str = 'discretized',
                 bins: int = 5,
                 axis: str = 'ts',
                 method: str = 'quantile',
                 window_type: str = 'expanding',
                 window_size: int = 30,
                 min_obs: int = 1):
        super().__init__(name="Discretize", description="Discretizes continuous data using KBinsDiscretizer.")

        if bins <= 1:
            raise ValueError("Number of bins must be larger than 1.")
        if method not in {'quantile', 'uniform', 'kmeans'}:
            raise ValueError("Method must be one of 'quantile', 'uniform', or 'kmeans'.")

        self.input_col = input_col
        self.output_col = output_col
        self.bins = bins
        self.axis = axis
        self.method = method
        self.window_type = window_type
        self.window_size = window_size
        self.min_obs = min_obs
        self.discretizer = KBinsDiscretizer(n_bins=self.bins,
                                            strategy=self.method,
                                            encode='ordinal')

    def fit(self, X: Union[pd.Series, pd.DataFrame],
            y: Union[pd.Series, pd.DataFrame] = None) -> 'Discretize':
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
        """Apply the discretization transformation."""
        df = X[[self.input_col]]

        multiindex = isinstance(df.index, pd.MultiIndex)
        df_unstacked = df.unstack().dropna() if multiindex else df.dropna()

        if self.axis == 'ts':
            disc = None

            if self.window_type == 'rolling':
                for row in range(df_unstacked.shape[0] - self.window_size + 1):
                    window = df_unstacked.iloc[row: row + self.window_size]
                    transformed = self.discretizer.fit_transform(window)
                    if disc is None:
                        disc = transformed[-1]
                    else:
                        disc = np.vstack([disc, transformed[-1]])

            elif self.window_type == 'expanding':
                for row in range(self.min_obs, df_unstacked.shape[0] + 1):
                    window = df_unstacked.iloc[:row]
                    transformed = self.discretizer.fit_transform(window)
                    if disc is None:
                        disc = transformed[-1]
                    else:
                        disc = np.vstack([disc, transformed[-1]])

            elif self.window_type == 'fixed':
                disc = self.discretizer.fit_transform(df_unstacked)

            else:
                raise ValueError(f"Unsupported window_type: {self.window_type}")

            disc_df = pd.DataFrame(disc, index=df_unstacked.index[-len(disc):], columns=df_unstacked.columns)

            if multiindex:
                disc_df = disc_df.stack(future_stack=True)

            X[self.output_col] = disc_df.astype(float) + 1

            return X

        elif self.axis == 'cs':

            def discretize(data):
                mask = data.isna()
                trans = self.discretizer.fit_transform(data.fillna(0))
                trans_df = pd.DataFrame(trans, index=data.index, columns=data.columns)
                trans_df[mask] = np.nan
                return trans_df

            if isinstance(df.index, pd.MultiIndex):
                # If MultiIndex, group by the first level (e.g., date)
                discretized_df = df.groupby(level=0, group_keys=False).apply(lambda x: discretize(x))
            else:
                raise ValueError("Cross-sectional discretization requires a MultiIndex DataFrame.")

            X[self.output_col] = discretized_df.astype(float) + 1

            return X

        else:
            raise ValueError(f"Unsupported axis: {self.axis}")
