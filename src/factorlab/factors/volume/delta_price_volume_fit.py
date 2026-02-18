from __future__ import annotations

import numpy as np
import pandas as pd

from factorlab.factors.volume.base import VolumeFactor


class DeltaPriceVolumeFit(VolumeFactor):
    def __init__(self, hist_length: int = 20, delta_dist: int = 30, **kwargs):
        super().__init__(**kwargs)
        self.hist_length = hist_length
        self.delta_dist = delta_dist
        self.name = "DeltaPriceVolumeFit"
        self.description = "Current minus lagged price-volume fit slope."

    def _generate_name(self) -> str:
        return self.output_col or f"{self.name}_{self.hist_length}_{self.delta_dist}"

    def _compute_volume(self, df: pd.DataFrame) -> pd.Series:
        close = df[self.price_col]
        volume = df[self.volume_col]

        x = self._safe_log(volume)
        y = self._safe_log(close)

        mean_x = self._rolling_stat(x, window=self.hist_length, stat="mean")
        mean_y = self._rolling_stat(y, window=self.hist_length, stat="mean")
        mean_xy = self._rolling_stat(x * y, window=self.hist_length, stat="mean")
        mean_x2 = self._rolling_stat(x * x, window=self.hist_length, stat="mean")

        cov_xy = mean_xy - (mean_x * mean_y)
        var_x = mean_x2 - (mean_x * mean_x)
        pvf = cov_xy / var_x.replace(0, np.nan)

        return pvf - self._shift_by_asset(pvf, self.delta_dist)
