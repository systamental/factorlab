from __future__ import annotations

import numpy as np
import pandas as pd

from factorlab.factors.volume.base import VolumeFactor


class DeltaPositiveVolumeIndicator(VolumeFactor):
    def __init__(self, hist_length: int = 40, delta_dist: int = 35, **kwargs):
        super().__init__(**kwargs)
        self.hist_length = hist_length
        self.delta_dist = delta_dist
        self.name = "DeltaPositiveVolumeIndicator"
        self.description = "Current minus lagged positive-volume indicator."

    def _generate_name(self) -> str:
        return self.output_col or f"{self.name}_{self.hist_length}_{self.delta_dist}"

    def _compute_volume(self, df: pd.DataFrame) -> pd.Series:
        close = df[self.price_col]
        volume = df[self.volume_col]

        rel_change = self._pct_change_by_asset(close, periods=1)
        prev_volume = self._shift_by_asset(volume, 1)
        filtered = rel_change.where(volume > prev_volume, 0.0)

        avg_change = self._rolling_mean(filtered, window=self.hist_length)
        norm_window = max(2 * self.hist_length, 250)
        std_change = self._rolling_std(
            rel_change,
            window=norm_window,
            min_periods=self.hist_length,
        ).replace(0, np.nan)
        pvi = avg_change / std_change

        return pvi - self._shift_by_asset(pvi, self.delta_dist)
