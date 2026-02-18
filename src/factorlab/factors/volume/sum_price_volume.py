from __future__ import annotations

import numpy as np
import pandas as pd

from factorlab.factors.volume.base import VolumeFactor


class SumPriceVolume(VolumeFactor):
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
        self.name = "SumPriceVolume"
        self.description = "Smoothed signed sum of normalized price/volume shocks."

    def _generate_name(self) -> str:
        return self.output_col or f"{self.name}_{self.hist_length}"

    def _normalized_volume_and_price_change(self, df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        close = df[self.price_col]
        volume = df[self.volume_col]

        prior_volume = self._shift_by_asset(volume, 1)
        median_volume = self._rolling_stat(
            prior_volume,
            window=self.norm_lookback,
            stat="median",
            min_periods=self.norm_min_periods,
        ).replace(0, np.nan)
        normalized_volume = volume / median_volume

        log_close = self._safe_log(close)
        price_change = self._diff_by_asset(log_close, 1)
        prior_change = self._shift_by_asset(price_change, 1)

        median_change = self._rolling_stat(
            prior_change,
            window=self.norm_lookback,
            stat="median",
            min_periods=self.norm_min_periods,
        )
        q75 = self._rolling_stat(
            prior_change,
            window=self.norm_lookback,
            stat="quantile",
            min_periods=self.norm_min_periods,
            q=0.75,
        )
        q25 = self._rolling_stat(
            prior_change,
            window=self.norm_lookback,
            stat="quantile",
            min_periods=self.norm_min_periods,
            q=0.25,
        )
        iqr = (q75 - q25).replace(0, np.nan)

        normalized_change = (price_change - median_change) / iqr
        return normalized_volume, normalized_change

    def _compute_volume(self, df: pd.DataFrame) -> pd.Series:
        normalized_volume, normalized_change = self._normalized_volume_and_price_change(df)
        precursor = normalized_volume + normalized_change.abs()
        precursor = precursor.where(normalized_change >= 0, -precursor)
        return self._rolling_stat(precursor, window=self.hist_length, stat="mean")
