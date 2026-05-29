from __future__ import annotations

import numpy as np
import pandas as pd

from factorlab.factors.volume.base import VolumeFactor


class OnBalanceVolume(VolumeFactor):
    def __init__(self, hist_length: int = 50, **kwargs):
        super().__init__(**kwargs)
        self.hist_length = hist_length
        self.name = "OnBalanceVolume"
        self.description = "Signed-volume over total-volume ratio."

    def _generate_name(self) -> str:
        return self.output_col or f"{self.name}_{self.hist_length}"

    def _compute_volume(self, df: pd.DataFrame) -> pd.Series:
        close = df[self.price_col]
        volume = df[self.volume_col]

        close_diff = self._diff_by_asset(close, 1)
        signed_volume = volume * np.sign(close_diff)

        signed_sum = self._rolling_stat(signed_volume, window=self.hist_length, stat="sum")
        total_sum = self._rolling_stat(volume, window=self.hist_length, stat="sum")

        return signed_sum / total_sum.replace(0, np.nan)
