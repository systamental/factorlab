from __future__ import annotations

import numpy as np
import pandas as pd

from factorlab.factors.volume.base import VolumeFactor


class VolumeMomentum(VolumeFactor):
    def __init__(self, hist_length: int = 20, multiplier: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.hist_length = hist_length
        self.multiplier = multiplier
        self.name = "VolumeMomentum"
        self.description = "Short-vs-long volume momentum ratio."

    def _generate_name(self) -> str:
        return self.output_col or f"{self.name}_{self.hist_length}_{self.multiplier}"

    def _compute_volume(self, df: pd.DataFrame) -> pd.Series:
        volume = df[self.volume_col]
        short_ma = self._rolling_mean(volume, window=self.hist_length)
        long_ma = self._rolling_mean(volume, window=self.hist_length * self.multiplier)
        ratio = short_ma / long_ma.replace(0, np.nan)
        return self._safe_log(ratio)
