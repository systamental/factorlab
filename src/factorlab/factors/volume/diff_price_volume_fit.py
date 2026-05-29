from __future__ import annotations

import numpy as np
import pandas as pd

from factorlab.factors.volume.base import VolumeFactor


class DiffPriceVolumeFit(VolumeFactor):
    def __init__(self, short_dist: int = 20, long_dist: int = 100, **kwargs):
        super().__init__(**kwargs)
        self.short_dist = short_dist
        self.long_dist = long_dist
        self.name = "DiffPriceVolumeFit"
        self.description = "Short minus long price-volume fit slope."

    def _generate_name(self) -> str:
        return self.output_col or f"{self.name}_{self.short_dist}_{self.long_dist}"

    def _pv_fit(self, df: pd.DataFrame, hist_length: int) -> pd.Series:
        close = df[self.price_col]
        volume = df[self.volume_col]

        x = self._safe_log(volume)
        y = self._safe_log(close)

        mean_x = self._rolling_mean(x, window=hist_length)
        mean_y = self._rolling_mean(y, window=hist_length)
        mean_xy = self._rolling_mean(x * y, window=hist_length)
        mean_x2 = self._rolling_mean(x * x, window=hist_length)

        cov_xy = mean_xy - (mean_x * mean_y)
        var_x = mean_x2 - (mean_x * mean_x)
        return cov_xy / var_x.replace(0, np.nan)

    def _compute_volume(self, df: pd.DataFrame) -> pd.Series:
        short = self._pv_fit(df, self.short_dist)
        long = self._pv_fit(df, self.long_dist)
        return short - long
