from __future__ import annotations

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
        pvf = self._raw_price_volume_fit(df, self.hist_length)
        return pvf - self._shift_by_asset(pvf, self.delta_dist)
