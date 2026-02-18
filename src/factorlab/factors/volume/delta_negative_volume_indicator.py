from __future__ import annotations

import pandas as pd

from factorlab.factors.volume.base import VolumeFactor


class DeltaNegativeVolumeIndicator(VolumeFactor):
    def __init__(self, hist_length: int = 40, delta_dist: int = 35, **kwargs):
        super().__init__(**kwargs)
        self.hist_length = hist_length
        self.delta_dist = delta_dist
        self.name = "DeltaNegativeVolumeIndicator"
        self.description = "Current minus lagged negative-volume indicator."

    def _generate_name(self) -> str:
        return self.output_col or f"{self.name}_{self.hist_length}_{self.delta_dist}"

    def _compute_volume(self, df: pd.DataFrame) -> pd.Series:
        nvi = self._raw_negative_volume_indicator(df, self.hist_length)
        return nvi - self._shift_by_asset(nvi, self.delta_dist)
