from __future__ import annotations

import numpy as np
import pandas as pd

from factorlab.factors.volume.base import VolumeFactor


class PriceVolumeFit(VolumeFactor):
    def __init__(self, hist_length: int = 50, **kwargs):
        super().__init__(**kwargs)
        self.hist_length = hist_length
        self.name = "PriceVolumeFit"
        self.description = "Rolling slope for log(price) on log(volume)."

    def _generate_name(self) -> str:
        return self.output_col or f"{self.name}_{self.hist_length}"

    def _compute_volume(self, df: pd.DataFrame) -> pd.Series:
        close = df[self.price_col]
        volume = df[self.volume_col]

        x = self._safe_log(volume)
        y = self._safe_log(close)

        mean_x = self._rolling_mean(x, window=self.hist_length)
        mean_y = self._rolling_mean(y, window=self.hist_length)
        mean_xy = self._rolling_mean(x * y, window=self.hist_length)
        mean_x2 = self._rolling_mean(x * x, window=self.hist_length)

        cov_xy = mean_xy - (mean_x * mean_y)
        var_x = mean_x2 - (mean_x * mean_x)
        return cov_xy / var_x.replace(0, np.nan)
