from __future__ import annotations

import pandas as pd

from factorlab.factors.volume.base import VolumeFactor


class DeltaPositiveVolumeIndicator(VolumeFactor):
    def __init__(self, hist_length: int = 40, delta_dist: int = 35, **kwargs):
        super().__init__(**kwargs)
        self.hist_length = hist_length
        self.delta_dist = delta_dist
        self.name = "DeltaPositiveVolumeIndicator"
        self.description = "Current minus lagged positive-volume indicator."

    def _generate_name(self) -> str:
        return self.output_col or f"{self.name}_{self.hist_length}_{self.delta_dist}"

    def _compute_volume(self, df: pd.DataFrame) -> pd.Series:
        pvi = self._raw_positive_volume_indicator(df, self.hist_length)
        return pvi - self._shift_by_asset(pvi, self.delta_dist)
