from __future__ import annotations

import pandas as pd

from factorlab.factors.volume.base import VolumeFactor


class DiffPriceVolumeFit(VolumeFactor):
    def __init__(self, short_dist: int = 20, long_dist: int = 100, **kwargs):
        super().__init__(**kwargs)
        self.short_dist = short_dist
        self.long_dist = long_dist
        self.name = "DiffPriceVolumeFit"
        self.description = "Short minus long price-volume fit slope."

    def _generate_name(self) -> str:
        return self.output_col or f"{self.name}_{self.short_dist}_{self.long_dist}"

    def _compute_volume(self, df: pd.DataFrame) -> pd.Series:
        short = self._raw_price_volume_fit(df, self.short_dist)
        long = self._raw_price_volume_fit(df, self.long_dist)
        return short - long
