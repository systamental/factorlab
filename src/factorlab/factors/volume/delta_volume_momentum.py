from __future__ import annotations

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
        vmom = self._raw_volume_momentum(df, self.hist_length, self.multiplier)
        return vmom - self._shift_by_asset(vmom, self.delta_len)
