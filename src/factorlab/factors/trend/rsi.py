from __future__ import annotations
import pandas as pd
from factorlab.factors.trend.base import TrendFactor
from factorlab.transformations.returns import LogReturn
from factorlab.transformations.smoothing import WindowSmoother
from factorlab.utils import to_dataframe


class RSI(TrendFactor):
    """
    Computes the RSI indicator.
    """
    def __init__(self,
                 input_col: str = 'close',
                 central_tendency: str = 'mean',
                 signal: bool = True,
                 scale: bool = False,
                 **kwargs):
        """
        Constructor.

        Parameters
        ----------
        central_tendency: str, {'mean', 'median'}, default 'mean'
            Central tendency measure for smoothing.
        signal: bool, default True
            Converts RSI to a signal between -1 and 1.
            Typically, RSI is normalized to between 0 and 100.
        """
        super().__init__(scale=scale, **kwargs)
        self.name = 'RSI'
        self.description = 'RSI (Relative Strength Index) indicator.'
        self.input_col = input_col
        self.central_tendency = central_tendency
        self.signal = signal

    def _compute_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes the RSI indicator.
        """
        price_transform = LogReturn(lags=1, output_col='ret')
        chg = price_transform.compute(df)

        # get up and down days
        up = chg.where(chg > 0).fillna(0)
        down = abs(chg.where(chg < 0).fillna(0))

        # up over down
        rs_up = WindowSmoother(input_cols='ret',
                               window_type=self.window_type,
                               window_size=self.window_size,
                               central_tendency=self.central_tendency).compute(up)
        rs_down = WindowSmoother(input_cols='ret',
                                 window_type=self.window_type,
                                 window_size=self.window_size,
                                 central_tendency=self.central_tendency).compute(down)

        rs = rs_up.iloc[:, -1] / rs_down.iloc[:, -1]

        # normalization to remove inf 0 div
        rs = 100 - (100 / (1 + rs))
        # signal
        if self.signal:
            rs = (rs - 50)/50

        # rsi
        trend_df = to_dataframe(rs, name='trend')

        return trend_df[['trend']]
