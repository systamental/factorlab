import pandas as pd
import numpy as np

from factorlab.factors.liquidity.base import LiquidityFactor
from factorlab.utils import grouped, maybe_droplevel


class HighLowSpreadEstimator(LiquidityFactor):
    """
    Computes the high-low spread estimator from Corwin & Schultz (2011),
    which estimates bid-ask spreads using high/low price ranges over two days.

    This nonparametric estimator is useful for liquidity analysis when quote data is unavailable.

    See: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1106193

    Parameters
    ----------
    high_col : str, default "high"
        Column name for high prices.
    low_col : str, default "low"
        Column name for low prices.
    close_col : str, default "close"
        Column name for close prices.
    output_col : str, default "hl_spread"
        Name of the output column to store the spread estimate.

    """

    def __init__(
            self,
            high_col: str = "high",
            low_col: str = "low",
            close_col: str = "close",
            output_col: str = "hl_spread",
            **kwargs
    ):
        super().__init__(**kwargs)

        self.name = "HighLowSpreadEstimator"
        self.description = "High-Low Spread Estimator from Corwin & Schultz (2011)"
        self.high_col = high_col
        self.low_col = low_col
        self.close_col = close_col
        self.output_col = output_col

    @property
    def inputs(self):
        return [self.high_col, self.low_col, self.close_col]

    def _compute_liquidity(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes the high-low spread estimator.

        Parameters
        ----------
        df: pd.DataFrame
            The input DataFrame containing the required high, low, and close price columns.

        Returns
        -------
        pd.Series
            A Series with the computed high-low spread estimates in the specified output column.

        """

        # preserve shape
        is_multi = isinstance(df.index, pd.MultiIndex)
        df_unstacked = df.unstack() if is_multi else df

        high = df_unstacked[self.high_col]
        low = df_unstacked[self.low_col]
        close = df_unstacked[self.close_col]

        # 2-day highs/lows and midpoint
        high_2d = high.rolling(2).max()
        low_2d = low.rolling(2).min()

        # adjust for overnight gaps
        prev_close = close.shift(1)
        gap_up = low - prev_close
        gap_down = high - prev_close

        high_adj = high.where(gap_down >= 0, high - gap_down)
        low_adj = low.where(gap_down >= 0, low - gap_down)
        high_adj = high_adj.where(gap_up <= 0, high_adj - gap_up)
        low_adj = low_adj.where(gap_up <= 0, low_adj - gap_up)

        # compute B and G
        B = (np.log(high_adj / low_adj)) ** 2 + (np.log(high_adj.shift(1) / low_adj.shift(1))) ** 2
        G = (np.log(high_2d / low_2d)) ** 2

        # compute alpha
        denom = (3 - 2 * np.sqrt(2))
        alpha = (np.sqrt(2 * B) - np.sqrt(B)) / denom - np.sqrt(G / denom)
        alpha = np.clip(alpha, 0, None)

        # compute high-low spread estimator
        S = (2 * (np.exp(alpha) - 1)) / (1 + np.exp(alpha))

        # moving window
        if self.smooth:
            g = grouped(S)
            window_op = self._get_ts_window_op(g)
            # apply mean
            S = window_op.mean()
            # drop multiindex level if present
            S = maybe_droplevel(S, level=0)

        # restore original index structure
        if is_multi:
            S = S.stack().reindex(df.index).rename(self.output_col)
        else:
            S = S.reindex(df.index).rename(self.output_col)

        return S
