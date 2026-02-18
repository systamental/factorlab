from __future__ import annotations

import numpy as np
import pandas as pd

from factorlab.factors.volume.base import VolumeFactor


class VolumeWeightedMAOverMA(VolumeFactor):
    def __init__(self, hist_length: int = 50, **kwargs):
        super().__init__(**kwargs)
        self.hist_length = hist_length
        self.name = "VolumeWeightedMAOverMA"
        self.description = "Log ratio of VWMA over MA."

    def _generate_name(self) -> str:
        return self.output_col or f"{self.name}_{self.hist_length}"

    def _compute_volume(self, df: pd.DataFrame) -> pd.Series:
        close = df[self.price_col]
        volume = df[self.volume_col]

        pv = close * volume
        vwma = self._rolling_stat(pv, window=self.hist_length, stat="sum") / self._rolling_stat(
            volume, window=self.hist_length, stat="sum"
        ).replace(0, np.nan)
        ma = self._rolling_stat(close, window=self.hist_length, stat="mean")

        return self._safe_log(vwma / ma.replace(0, np.nan))
