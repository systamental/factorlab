from __future__ import annotations
import pandas as pd
from factorlab.factors.trend.base import TrendFactor
from factorlab.transformations.normalization import Normalization
from factorlab.signal_generation.continuous import ScoreSignal


class Breakout(TrendFactor):
    """
    Computes the breakout trend factor by normalizing price over a rolling window.
    """
    def __init__(self,
                 input_col: str = 'close',
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
            trend_df = ScoreSignal(input_col='scores',
                                   output_col='trend',
                                   method=self.signal_method).compute(trend_df)
        else:
            trend_df.rename(columns={'scores': 'trend'}, inplace=True)

        return trend_df[['trend']]
