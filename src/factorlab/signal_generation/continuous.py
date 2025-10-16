import pandas as pd
import numpy as np
from typing import Union, Dict
from scipy.stats import norm, logistic

from factorlab.signal_generation.base import BaseSignal
from factorlab.utils import grouped


class BuyHoldSignal(BaseSignal):
    """
    Generates a constant buy-and-hold signal of 1.0 for all assets.

    Parameters
    ----------
    output_col : str, default 'signal'
        Name of the output column to store the resulting signals.
    """

    def __init__(self,
                 input_col: str = 'ret',
                 **kwargs):
        super().__init__(input_col=input_col, **kwargs)

        self.name = "BuyHoldSignal"
        self.description = "Generates a constant buy-and-hold signal of 1.0 for all assets."
        self.input_col = input_col

    def _compute_signal(self, X: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
        """Generate constant buy-and-hold signals."""
        df = X[self.input_col].copy()

        signals = np.sign(df.abs())  # 1.0 for all non-NaN entries
        return signals


class RawSignal(BaseSignal):
    """
    Uses raw scores as signals without any transformation.

    Parameters
    ----------
    input_col : str
        Name of the input column containing raw scores.
    output_col : str, default 'signal'
        Name of the output column to store the resulting signals.

    """

    def __init__(self,
                 input_col: str,
                 **kwargs):
        super().__init__(input_col=input_col, **kwargs)

        self.name = "ZScoreToSignal"
        self.description = "Converts standardized z-scores to signals in the range [-1, 1]."
        self.input_col = input_col

    def _compute_signal(self, X: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
        """Use raw scores for signals transformation."""
        return X[self.input_col].copy()


class ScoreSignal(BaseSignal):
    """
    Converts standardized cores to signals in the range [-1, 1].

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

    # transformation map
    _transformation_map: Dict[str, callable] = {
        'norm': lambda df: pd.DataFrame(norm.cdf(df), index=df.index, columns=df.columns),
        'logistic': lambda df: pd.DataFrame(logistic.cdf(df), index=df.index, columns=df.columns),
        'adj_norm': lambda df: df * np.exp((-1 * df ** 2) / 4) / 0.89,
        'tanh': np.tanh,
        'percentile': lambda df: df,
        'min_max': lambda df: df,
    }

    def __init__(self,
                 input_col: str = 'zscore',
                 method: str = 'norm',
                 **kwargs):
        super().__init__(input_col=input_col, **kwargs)

        self.name = "ZScoreToSignal"
        self.description = "Converts standardized z-scores to signals in the range [-1, 1]."
        self.input_col = input_col
        self.method = method

        if self.method not in self._transformation_map:
            raise ValueError(f"Unsupported method: {self.method}")

    def _compute_signal(self, X: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
        """Apply the scores to signals transformation."""
        df = X[[self.input_col]].copy()

        # signal transformation
        signals = self._transformation_map[self.method](df)

        # rescale signals result in [0, 1] range to [-1, 1].
        if self.method in {'norm', 'logistic', 'percentile', 'min_max'}:
            signals = (signals * 2) - 1

        return signals


class QuantileSignal(BaseSignal):
    """
    Converts quantile ranks to signals in the range [-1, 1].

    Parameters
    ----------
    input_col : str, default 'quantile'
        Name of the input column containing quantile ranks.
    bins : int, optional
        Number of quantile bins to use. If None, defaults to the median number of unique values in the DataFrame.
    axis : str, {'ts', 'cs'}, default 'ts'
        Axis along which to compute the quantiles. If 'ts', computes across time series;
        if 'cs', computes across cross-sections.

    """

    def __init__(self,
                 input_col: str = 'quantile',
                 bins: int = None,
                 **kwargs):
        super().__init__(input_col=input_col, **kwargs)

        self.name = "QuantileSignal"
        self.description = "Converts quantiles to signals in the range [-1, 1]."
        self.input_col = input_col
        self.bins = bins

    def _time_series_scale(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies Min-Max scaling (0 to 1) by group (ts)."""
        g = grouped(df)
        min_val = g.min()
        range_val = g.max() - g.min()

        signals = df.sub(min_val).div(range_val)
        signals.replace([np.inf, -np.inf], np.nan, inplace=True)

        return signals

    def _cross_sectional_scale(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handles cross-sectional scaling, considering MultiIndex structure and bin validation."""
        if not isinstance(df.index, pd.MultiIndex):
            raise ValueError("Cross-section axis requires a MultiIndex DataFrame.")

        # number of bins
        if self.bins is None:
            self.bins = df.nunique().median()

        # min number of observations in the cross-section
        df = df[(df.groupby(level=0).count() >= self.bins)].dropna()
        if df.empty:
            raise ValueError("Number of bins is larger than the number of observations in the cross-section.")

        cs_min = df.groupby(level=0).min()
        cs_range = df.groupby(level=0).max() - df.groupby(level=0).min()
        signals = (df - cs_min).div(cs_range)
        signals.replace([np.inf, -np.inf], np.nan, inplace=True)

        return signals

    def _compute_signal(self, X: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
        """Apply the quantiles to signals transformation."""
        df = X[[self.input_col]].copy()

        # scale
        if self.axis == 'ts':
            signals = self._time_series_scale(df)
        elif self.axis == 'cs':
            signals = self._cross_sectional_scale(df)
        else:
            raise ValueError(f"Unsupported axis: {self.axis}. Must be 'ts' or 'cs'.")

        # rescale
        return (signals * 2) - 1


class RankSignal(BaseSignal):
    """
    Converts ranks to signals in the range [-1, 1].

    Parameters
    ----------
    input_col : str, default 'rank'
        Name of the input column containing ranks.
    """

    def __init__(self,
                 input_col: str = 'rank',
                 **kwargs):
        super().__init__(input_col=input_col, **kwargs)

        self.name = "RankSignal"
        self.description = "Converts ranks to signals in the range [-1, 1]."
        self.input_col = input_col

    def _compute_signal(self, X: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
        """Apply the ranks to signals transformation."""
        df = X.copy()[[self.input_col]]

        # axis time series
        if self.axis == 'ts':
            g = grouped(df)
            ts_min = g.min()
            ts_range = g.max() - g.min()
            signals = ((df - ts_min) / ts_range)

        # axis cross-section
        else:
            if isinstance(df.index, pd.MultiIndex):
                cs_min = df.groupby(level=0).min()
                cs_range = df.groupby(level=0).max() - df.groupby(level=0).min()
                signals = (df - cs_min) / cs_range
            else:
                raise ValueError("Cross-section axis requires a MultiIndex DataFrame.")

        return (signals * 2) - 1
