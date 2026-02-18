from __future__ import annotations

import pandas as pd

from factorlab.factors.volume.base import VolumeFactor


class DeltaSumPriceVolume(VolumeFactor):
    def __init__(
        self,
        hist_length: int = 40,
        delta_dist: int = 35,
        norm_lookback: int = 250,
        norm_min_periods: int = 50,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hist_length = hist_length
        self.delta_dist = delta_dist
        self.norm_lookback = norm_lookback
        self.norm_min_periods = norm_min_periods
        self.name = "DeltaSumPriceVolume"
        self.description = "Current minus lagged sum-price-volume signal."

    def _generate_name(self) -> str:
        return self.output_col or f"{self.name}_{self.hist_length}_{self.delta_dist}"

    def _compute_volume(self, df: pd.DataFrame) -> pd.Series:
        spv = self._raw_sum_price_volume(
            df,
            hist_length=self.hist_length,
            norm_lookback=self.norm_lookback,
            norm_min_periods=self.norm_min_periods,
        )
        return spv - self._shift_by_asset(spv, self.delta_dist)
