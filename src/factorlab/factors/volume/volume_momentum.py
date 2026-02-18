from __future__ import annotations

import pandas as pd

from factorlab.factors.volume.base import VolumeFactor


class VolumeMomentum(VolumeFactor):
    def __init__(self, hist_length: int = 20, multiplier: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.hist_length = hist_length
        self.multiplier = multiplier
        self.name = "VolumeMomentum"
        self.description = "Short-vs-long volume momentum ratio."

    def _generate_name(self) -> str:
        return self.output_col or f"{self.name}_{self.hist_length}_{self.multiplier}"

    def _compute_volume(self, df: pd.DataFrame) -> pd.Series:
        return self._raw_volume_momentum(df, self.hist_length, self.multiplier)
