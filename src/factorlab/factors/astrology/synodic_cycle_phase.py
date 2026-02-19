from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from factorlab.factors.astrology.base import AstrologyFactor
from factorlab.factors.astrology.common import SYNODIC_PAIRS, get_dates_planets, get_planet_longitude


class SynodicCyclePhase(AstrologyFactor):
    """Synodic phase angle and encodings for major planetary pairs."""

    def __init__(self, pairs: Optional[List[str]] = None, **kwargs):
        super().__init__(
            description="Synodic cycle phase features for major planetary pairs.",
            tags=["astrology", "synodic", "phase"],
            **kwargs,
        )
        self.pairs = pairs

    def _compute_astrology(self, ephemeris_df: pd.DataFrame) -> pd.DataFrame:
        dates, _ = get_dates_planets(ephemeris_df)
        pairs = self.pairs or list(SYNODIC_PAIRS.keys())

        results = {}
        for pair_name in pairs:
            if pair_name not in SYNODIC_PAIRS:
                continue

            p1, p2 = SYNODIC_PAIRS[pair_name]
            lon1 = get_planet_longitude(ephemeris_df, p1).reindex(dates)
            lon2 = get_planet_longitude(ephemeris_df, p2).reindex(dates)
            if lon1.empty or lon2.empty:
                continue

            phase = (lon1 - lon2) % 360
            phase_label = (phase // 45).astype(int).clip(0, 7)
            rad = phase * np.pi / 180.0

            results[f"{pair_name}_phase"] = phase
            results[f"{pair_name}_phase_label"] = phase_label
            results[f"{pair_name}_phase_sin"] = np.sin(rad)
            results[f"{pair_name}_phase_cos"] = np.cos(rad)

        return pd.DataFrame(results) if results else pd.DataFrame(index=dates)
