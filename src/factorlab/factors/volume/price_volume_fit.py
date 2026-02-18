from __future__ import annotations

import pandas as pd

from factorlab.factors.volume.base import VolumeFactor


class PriceVolumeFit(VolumeFactor):
    def __init__(self, hist_length: int = 50, **kwargs):
        super().__init__(**kwargs)
        self.hist_length = hist_length
        self.name = "PriceVolumeFit"
        self.description = "Rolling slope for log(price) on log(volume)."

    def _generate_name(self) -> str:
        return self.output_col or f"{self.name}_{self.hist_length}"

    def _compute_volume(self, df: pd.DataFrame) -> pd.Series:
        return self._raw_price_volume_fit(df, self.hist_length)
