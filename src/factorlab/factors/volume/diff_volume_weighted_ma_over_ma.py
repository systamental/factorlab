from __future__ import annotations

import numpy as np
import pandas as pd

from factorlab.factors.volume.base import VolumeFactor


class DiffVolumeWeightedMAOverMA(VolumeFactor):
    def __init__(self, short_dist: int = 20, long_dist: int = 100, **kwargs):
        super().__init__(**kwargs)
        self.short_dist = short_dist
        self.long_dist = long_dist
        self.name = "DiffVolumeWeightedMAOverMA"
        self.description = "Short minus long VWMA-over-MA signal."

    def _generate_name(self) -> str:
        return self.output_col or f"{self.name}_{self.short_dist}_{self.long_dist}"

    def _vwma_over_ma(self, df: pd.DataFrame, hist_length: int) -> pd.Series:
        close = df[self.price_col]
        volume = df[self.volume_col]

        pv = close * volume
        vwma = self._rolling_stat(pv, window=hist_length, stat="sum") / self._rolling_stat(
            volume, window=hist_length, stat="sum"
        ).replace(0, np.nan)
        ma = self._rolling_stat(close, window=hist_length, stat="mean")

        return self._safe_log(vwma / ma.replace(0, np.nan))

    def _compute_volume(self, df: pd.DataFrame) -> pd.Series:
        short = self._vwma_over_ma(df, self.short_dist)
        long = self._vwma_over_ma(df, self.long_dist)
        return short - long
