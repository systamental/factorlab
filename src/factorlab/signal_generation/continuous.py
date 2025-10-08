import pandas as pd
import numpy as np
from typing import Union
from scipy.stats import norm, logistic

from factorlab.signal_generation.base import BaseSignal
from factorlab.utils import grouped


class ZScoreSignal(BaseSignal):
    """
    Converts standardized z-scores to signals in the range [-1, 1].

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
                 method: str = 'norm',
                 **kwargs):
        super().__init__(input_col=input_col, output_col=output_col, **kwargs)

        self.name = "ZScoreToSignal"
        self.description = "Converts standardized z-scores to signals in the range [-1, 1]."
        self.input_col = input_col
        self.output_col = output_col
        self.method = method

    def _compute_signal(self, X: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
        """Apply the scores to signals transformation."""
        df = X.copy()[[self.input_col]]

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

        return signals


class QuantileSignal(BaseSignal):
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
                 axis: str = 'ts',
                 **kwargs):
        super().__init__(input_col=input_col, output_col=output_col, **kwargs)

        self.name = "QuantileSignal"
        self.description = "Converts quantiles to signals in the range [-1, 1]."
        self.input_col = input_col
        self.output_col = output_col
        self.bins = bins
        self.axis = axis

    def _compute_signal(self, X: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
        """Apply the quantiles to signals transformation."""
        df = X.copy()[[self.input_col]]

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

        return signals


class RankSignal(BaseSignal):
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
                 axis: str = 'ts',
                 **kwargs):
        super().__init__(input_col=input_col, output_col=output_col, **kwargs)

        self.name = "RankSignal"
        self.description = "Converts ranks to signals in the range [-1, 1]."
        self.input_col = input_col
        self.output_col = output_col
        self.axis = axis

    def _compute_signal(self, X: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
        """Apply the ranks to signals transformation."""
        df = X.copy()[[self.input_col]]

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

        return signals
