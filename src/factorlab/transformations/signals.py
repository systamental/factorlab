import pandas as pd
import numpy as np
from typing import Union, Optional
from scipy.stats import norm, logistic

from factorlab.utils import to_dataframe, grouped
from factorlab.core.base_transform import BaseTransform


class ScoresToSignals(BaseTransform):
    """
    Converts standardized scores to signals in the range [-1, 1].

    Parameters
    ----------
    input_col : str, default 'zscore'
        Name of the input column containing standardized scores.
    output_col : str, default 'signal'
        Name of the output column to store the resulting signals.
    method : str, {'norm',  'logistic', 'adj_norm', 'tanh', 'percentile', 'min_max'}, default 'norm'
            norm: normal cumulative distribution function.
            logistic: logistic cumulative distribution function.
            adj_norm: adjusted normal distribution.
            tanh: hyperbolic tangent.
            percentile: percentile rank.
            min-max: values between 0 and 1.

    """

    def __init__(self,
                 input_col: str = 'zscore',
                 output_col: str = 'signal',
                 method: str = 'norm'):
        super().__init__(name="ScoresToSignals",
                         description="Converts standardized scores to signals in the range [-1, 1].")

        self.input_col = input_col
        self.output_col = output_col
        self.method = method

    def fit(self,
            X: Union[pd.Series, pd.DataFrame],
            y: Optional[Union[pd.Series, pd.DataFrame]] = None) -> 'ScoresToSignals':
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
        """Apply the scores to signals transformation."""
        df = X[[self.input_col]]

        if self.method == 'norm':
            signals = pd.DataFrame(norm.cdf(df),
                                  index=df.index,
                                  columns=df.columns)

        elif self.method == 'logistic':
            signals = pd.DataFrame(logistic.cdf(df),
                                  index=df.index,
                                  columns=df.columns)

        elif self.method == 'adj_norm':
            signals = df * np.exp((-1 * df ** 2) / 4) / 0.89

        elif self.method == 'tanh':
            signals = np.tanh(df)

        elif self.method == 'percentile':
            signals = df

        elif self.method == 'min_max':
            signals = df

        else:
            raise ValueError(f"Unsupported method: {self.method}")

        if self.method in {'norm', 'logistic', 'min_max', 'percentile'}:
            signals = (signals * 2) - 1

        X[self.output_col] = signals
        return X


class QuantilesToSignals(BaseTransform):
    """
    Converts quantile ranks to signals in the range [-1, 1].

    Parameters
    ----------
    input_col : str, default 'quantile'
        Name of the input column containing quantile ranks.
    output_col : str, default 'signal'
        Name of the output column to store the resulting signals.
    bins : int, optional
        Number of quantile bins to use. If None, defaults to the median number of unique values in the DataFrame.
    axis : str, {'ts', 'cs'}, default 'ts'
        Axis along which to compute the quantiles. If 'ts', computes across time series;
        if 'cs', computes across cross-sections.

    """

    def __init__(self,
                 input_col: str = 'quantile',
                 output_col: str = 'signal',
                 bins: int = None,
                 axis: str = 'ts'):
        super().__init__(name="QuantilesToSignals",
                         description="Converts quantiles to signals in the range [-1, 1].")

        self.input_col = input_col
        self.output_col = output_col
        self.bins = bins
        self.axis = axis

    def fit(self,
            X: Union[pd.Series, pd.DataFrame],
            y: Optional[Union[pd.Series, pd.DataFrame]] = None) -> 'QuantilesToSignals':
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
        """Apply the quantiles to signals transformation."""
        df = X[[self.input_col]]

        # number of bins
        if self.bins is None:
            self.bins = df.nunique().median()

        # axis time series
        if self.axis == 'ts':
            g = grouped(df)
            ts_min = g.min()
            ts_max = g.max()
            ts_range = ts_max - ts_min

            signals = ((df - ts_min) / ts_range) * 2 - 1

        # axis cross-section
        else:

            if isinstance(df.index, pd.MultiIndex):
                # min number of observations in the cross-section
                df = df[(df.groupby(level=0).count() >= self.bins)].dropna()
                if df.empty:
                    raise ValueError("Number of bins is larger than the number of observations in the cross-section.")
                cs_min = df.groupby(level=0).min()
                cs_range = df.groupby(level=0).max() - df.groupby(level=0).min()
                signals = (df - cs_min) / cs_range * 2 - 1

            else:
                raise ValueError("Cross-section axis requires a MultiIndex DataFrame.")

        X[self.output_col] = signals

        return X


class RanksToSignals(BaseTransform):
    """
    Converts ranks to signals in the range [-1, 1].

    Parameters
    ----------
    input_col : str, default 'rank'
        Name of the input column containing ranks.
    output_col : str, default 'signal'
        Name of the output column to store the resulting signals.
    axis : str, {'ts', 'cs'}, default 'ts'
        Axis along which to compute the ranks. If 'ts', computes across time series;
        if 'cs', computes across cross-sections.
    """

    def __init__(self,
                 input_col: str = 'rank',
                 output_col: str = 'signal',
                 axis: str = 'ts'):
        super().__init__(name="RankToSignal",
                         description="Converts ranks to signals in the range [-1, 1].")

        self.input_col = input_col
        self.output_col = output_col
        self.axis = axis

    def fit(self,
            X: Union[pd.Series, pd.DataFrame],
            y: Optional[Union[pd.Series, pd.DataFrame]] = None) -> 'ScoresToSignals':
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
        """Apply the ranks to signals transformation."""
        if not self._is_fitted:
            raise RuntimeError(f"Transform '{self.name}' must be fitted before calling transform()")

        df = X[[self.input_col]]

        # axis time series
        if self.axis == 'ts':
            g = grouped(df)
            ts_min = g.min()
            ts_range = g.max() - g.min()

            signals = ((df - ts_min) / ts_range) * 2 - 1

        # axis cross-section
        else:
            if isinstance(df.index, pd.MultiIndex):
                cs_min = df.groupby(level=0).min()
                cs_range = df.groupby(level=0).max() - df.groupby(level=0).min()
                signals = (df - cs_min) / cs_range * 2 - 1
            else:
                raise ValueError("Cross-section axis requires a MultiIndex DataFrame.")

        X[self.output_col] = signals

        return X
