from __future__ import annotations
import pandas as pd
import numpy as np
from typing import List, Optional
from factorlab.factors.trend.base import TrendFactor
from factorlab.transformations.smoothing import WindowSmoother


class TripleEWMADifference(TrendFactor):
    """
    Computes the exponentially weighted moving average (EWMA) crossover trend factor for 3 different short
    and long window combinations.

    A CTA-momentum signal, based on the cross-over of multiple exponentially weighted moving averages (EWMA) with
    different half-lives.

    Computed as described in Dissecting Investment Strategies in the Cross-Section and Time Series:
    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2695101

    Span parameter is used instead of halflife in the ewm function.
    """
    def __init__(self,
                 input_col: str = 'close',
                 signal: bool = True,
                 scale: bool = False,
                 smooth: bool = False,
                 **kwargs):
        """
        Constructor.

        Parameters
        ----------
        signal: bool, default True
            Converts normalized ewma crossover values to signal between -1 and 1.

        """
        super().__init__(scale=scale, smooth=smooth, **kwargs)
        self.name = 'TripleEWMADifference'
        self.description = 'Triple EWMA Difference trend factor.'
        self.input_col = input_col
        self.signal = signal

    def _compute_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes the RSI indicator.
        """
        if not isinstance(df.index, pd.MultiIndex):
            raise ValueError("Input DataFrame must have a MultiIndex with levels ['date', 'asset_id'].")

        def span_to_halflife(span):
            alpha = 2 / (span + 1)
            return np.ceil(np.log(0.5) / np.log(1 - alpha))

        # half-life lists for short and long windows
        hl_s = [span_to_halflife(self.window_size), span_to_halflife(self.window_size) * 2,
                span_to_halflife(self.window_size) * 4]
        hl_l = [hl_s[0] * 3, hl_s[1] * 3, hl_s[2] * 3]

        # create emtpy df
        factor_df = pd.DataFrame()
        df_unstacked = df[self.input_col].unstack()

        # compute ewma diff for short, medium and long windows
        for i in range(len(hl_s)):
            factor_df[f"x_k{i}"] = (df_unstacked.ewm(halflife=hl_s[i]).mean() -
                                    df_unstacked.ewm(halflife=hl_l[i]).mean()).stack(future_stack=True)

        # scale by std of price
        for i in range(len(hl_s)):
            factor_df[f"y_k{i}"] = (factor_df[f"x_k{i}"].unstack() /
                                    df_unstacked.rolling(90).std()).stack(future_stack=True)

        # scale by normalized y_k diff
        for i in range(len(hl_s)):
            factor_df[f"z_k{i}"] = (factor_df[f"x_k{i}"].unstack() /
                                    factor_df[f"x_k{i}"].unstack().rolling(365).std()).stack(future_stack=True)

        # convert to signal
        if self.signal:
            for i in range(len(hl_s)):
                factor_df[f"signal_k{i}"] = (factor_df[f"z_k{i}"] * np.exp((-1 * factor_df[f"z_k{i}"] ** 2) / 4)) / 0.89

        # mean of short, medium and long window signals
        ewma_diff = factor_df.iloc[:, -3:].mean(axis=1)
        # replace inf
        ewma_diff.replace([np.inf, -np.inf], np.nan, inplace=True)

        # ffill NaNs
        trend_df = ewma_diff.groupby(level=1).ffill()
        trend_df = trend_df.to_frame('trend')

        return trend_df[['trend']]
