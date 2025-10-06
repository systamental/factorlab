from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Optional
from factorlab.factors.trend.base import TrendFactor
from factorlab.transformations.dispersion import Dispersion
from factorlab.transformations.returns import LogReturn


class Energy(TrendFactor):
    """
    Computes the energy trend factor, E = mc^2, where E is energy, m is mass (volatility or VaR)
    and c is the speed (price momentum).

    Parameters
    ----------
    mass_method: str, {'std', 'var'}, default 'std'
        Method to compute mass. 'std' uses volatility, 'var' uses Value at Risk.
    perc: float, default 0.05
        Percentile for VaR calculation if mass_method is 'var'.
    """
    def __init__(self,
                 input_col: str = 'close',
                 mass_method='std',
                 perc: Optional[float] = 0.05,
                 window_type: str = "rolling",
                 scale: bool = False,
                 smooth: bool = False,
                 **kwargs):
        super().__init__(window_type=window_type, scale=scale, smooth=smooth, **kwargs)
        self.name = 'Energy'
        self.description = 'Measures the energy of price movements, combining momentum and volatility.'
        self.input_col = input_col
        self.mass_method = mass_method
        self.perc = perc

    def _compute_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes the energy signal.
        """
        # price change
        price_transform = LogReturn(lags=self.window_size, input_col=self.input_col)
        speed = price_transform.compute(df)  # / np.sqrt(self.window_size)
        speed = speed[['ret']]

        # ret
        ret = LogReturn(lags=1, input_col=self.input_col).compute(df)
        ret = ret[['ret']]

        # mass
        if self.mass_method == 'std':  # volatility
            mass = Dispersion(method='std', window_type=self.window_type, window_size=self.scaling_window).compute(ret)
            mass = mass[['std']]
        else:
            # VaR
            left_tail = Dispersion(method='quantile',
                                   q=self.perc,
                                   window_type=self.window_type,
                                   window_size=self.scaling_window).compute(ret) * -1
            left_tail = left_tail[['quantile']]
            right_tail = Dispersion(method='quantile',
                                    q=1 - self.perc,
                                    window_type=self.window_type,
                                    window_size=self.scaling_window).compute(ret)
            right_tail = right_tail[['quantile']]

            mass = pd.DataFrame(np.where(speed > 0, left_tail, right_tail), speed.index, speed.columns)

        # energy
        trend_df = speed.multiply(mass.values, axis=0)
        # rename column
        trend_df.columns = ['trend']

        return trend_df[['trend']]
