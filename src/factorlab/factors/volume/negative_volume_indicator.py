from __future__ import annotations

import numpy as np
import pandas as pd

from factorlab.factors.volume.base import VolumeFactor


class NegativeVolumeIndicator(VolumeFactor):
    def __init__(self, hist_length: int = 40, **kwargs):
        super().__init__(**kwargs)
        self.hist_length = hist_length
        self.name = "NegativeVolumeIndicator"
        self.description = "Normalized average return on falling-volume bars."

    def _generate_name(self) -> str:
        return self.output_col or f"{self.name}_{self.hist_length}"

    def _compute_volume(self, df: pd.DataFrame) -> pd.Series:
        close = df[self.price_col]
        volume = df[self.volume_col]

        rel_change = self._pct_change_by_asset(close, periods=1)
        prev_volume = self._shift_by_asset(volume, 1)
        filtered = rel_change.where(volume < prev_volume, 0.0)

        avg_change = self._rolling_stat(filtered, window=self.hist_length, stat="mean")
        norm_window = max(2 * self.hist_length, 250)
        std_change = self._rolling_stat(
            rel_change,
            window=norm_window,
            stat="std",
            min_periods=self.hist_length,
        ).replace(0, np.nan)

        return avg_change / std_change
