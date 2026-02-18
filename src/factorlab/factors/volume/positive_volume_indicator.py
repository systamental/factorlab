from __future__ import annotations

import pandas as pd

from factorlab.factors.volume.base import VolumeFactor


class PositiveVolumeIndicator(VolumeFactor):
    def __init__(self, hist_length: int = 40, **kwargs):
        super().__init__(**kwargs)
        self.hist_length = hist_length
        self.name = "PositiveVolumeIndicator"
        self.description = "Normalized average return on rising-volume bars."

    def _generate_name(self) -> str:
        return self.output_col or f"{self.name}_{self.hist_length}"

    def _compute_volume(self, df: pd.DataFrame) -> pd.Series:
        return self._raw_positive_volume_indicator(df, self.hist_length)
