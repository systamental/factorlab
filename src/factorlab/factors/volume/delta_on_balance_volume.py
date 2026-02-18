from __future__ import annotations

import pandas as pd

from factorlab.factors.volume.base import VolumeFactor


class DeltaOnBalanceVolume(VolumeFactor):
    def __init__(self, hist_length: int = 50, delta_dist: int = 45, **kwargs):
        super().__init__(**kwargs)
        self.hist_length = hist_length
        self.delta_dist = delta_dist
        self.name = "DeltaOnBalanceVolume"
        self.description = "Current minus lagged on-balance-volume signal."

    def _generate_name(self) -> str:
        return self.output_col or f"{self.name}_{self.hist_length}_{self.delta_dist}"

    def _compute_volume(self, df: pd.DataFrame) -> pd.Series:
        obv = self._raw_on_balance_volume(df, self.hist_length)
        return obv - self._shift_by_asset(obv, self.delta_dist)
