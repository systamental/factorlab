from __future__ import annotations

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
        return self._raw_volume_weighted_ma_over_ma(df, self.hist_length)
