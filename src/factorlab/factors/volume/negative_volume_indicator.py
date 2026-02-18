from __future__ import annotations

import pandas as pd

from factorlab.factors.volume.base import VolumeFactor


class NegativeVolumeIndicator(VolumeFactor):
    def __init__(self, hist_length: int = 40, **kwargs):
        super().__init__(**kwargs)
        self.hist_length = hist_length
        self.name = "NegativeVolumeIndicator"
        self.description = "Normalized average return on falling-volume bars."

    def _generate_name(self) -> str:
        return self.output_col or f"{self.name}_{self.hist_length}"

    def _compute_volume(self, df: pd.DataFrame) -> pd.Series:
        return self._raw_negative_volume_indicator(df, self.hist_length)
