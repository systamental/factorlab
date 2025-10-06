from __future__ import annotations
import pandas as pd
from factorlab.factors.trend.base import TrendFactor
from factorlab.transformations.normalization import Normalization
from factorlab.transformations.signals import ScoresToSignals


class Breakout(TrendFactor):
    """
    Computes the breakout trend factor by normalizing price over a rolling window.
    """
    def __init__(self,
                 input_col: str = 'close',
                 output_col: str = 'trend',
                 method: str = 'min_max',
                 signal: bool = True,
                 signal_method: str = 'min_max',
                 scale: bool = False,
                 **kwargs):
        """
        Constructor.

        Parameters
        ----------
        input_col: str, default 'close'
            Column name for closing price.
        output_col: str, default 'breakout'
            Column name for the computed breakout values.
        method: str, {'min_max', 'percentile', 'zscore'}, default 'min-max'
            Method to use for normalizing the price series.
        signal: bool, default True
            Whether to convert normalized values to a signal between -1 and 1.
        signal_method: str, {'min-max', 'percentile', 'norm', 'logistic', 'adj_norm'}, default 'min-max'
            Method to use for converting scores to signals.
        """
        super().__init__(scale=scale, **kwargs)
        self.name = 'Breakout'
        self.description = 'Measures price relative to its recent range.'
        self.input_col = input_col
        self.output_col = output_col
        self.method = method
        self.signal = signal
        self.signal_method = signal_method

    def _compute_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes the breakout signal.
        """
        # We use normalization to compute the breakout signal
        norm_transform = Normalization(input_col=self.input_col,
                                       output_col='scores',
                                       method=self.method,
                                       window_type='rolling',
                                       window_size=self.window_size)
        trend_df = norm_transform.compute(df)

        if self.signal:
            trend_df = ScoresToSignals(input_col='scores',
                                       output_col=self.output_col,
                                       method=self.signal_method).compute(trend_df)

        return trend_df[[self.output_col]]
