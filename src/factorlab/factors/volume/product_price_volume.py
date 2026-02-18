from __future__ import annotations

import pandas as pd

from factorlab.factors.volume.base import VolumeFactor


class ProductPriceVolume(VolumeFactor):
    def __init__(
        self,
        hist_length: int = 25,
        norm_lookback: int = 250,
        norm_min_periods: int = 50,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hist_length = hist_length
        self.norm_lookback = norm_lookback
        self.norm_min_periods = norm_min_periods
        self.name = "ProductPriceVolume"
        self.description = "Smoothed product of normalized price and volume shocks."

    def _generate_name(self) -> str:
        return self.output_col or f"{self.name}_{self.hist_length}"

    def _compute_volume(self, df: pd.DataFrame) -> pd.Series:
        return self._raw_product_price_volume(
            df,
            hist_length=self.hist_length,
            norm_lookback=self.norm_lookback,
            norm_min_periods=self.norm_min_periods,
        )
