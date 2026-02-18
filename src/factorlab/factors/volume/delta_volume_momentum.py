from __future__ import annotations

import numpy as np
import pandas as pd

from factorlab.factors.volume.base import VolumeFactor


class DeltaVolumeMomentum(VolumeFactor):
    def __init__(self, hist_length: int = 20, multiplier: int = 4, delta_len: int = 100, **kwargs):
        super().__init__(**kwargs)
        self.hist_length = hist_length
        self.multiplier = multiplier
        self.delta_len = delta_len
        self.name = "DeltaVolumeMomentum"
        self.description = "Current minus lagged volume momentum."

    def _generate_name(self) -> str:
        return self.output_col or f"{self.name}_{self.hist_length}_{self.multiplier}_{self.delta_len}"

    def _compute_volume(self, df: pd.DataFrame) -> pd.Series:
        volume = df[self.volume_col]
        short_ma = self._rolling_stat(volume, window=self.hist_length, stat="mean")
        long_ma = self._rolling_stat(volume, window=self.hist_length * self.multiplier, stat="mean")
        vmom = self._safe_log(short_ma / long_ma.replace(0, np.nan))
        return vmom - self._shift_by_asset(vmom, self.delta_len)
