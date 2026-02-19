from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from factorlab.factors.astrology.base import AstrologyFactor
from factorlab.factors.astrology.common import get_dates_planets, get_planet_field


class PlanetarySpeedFeatures(AstrologyFactor):
    """Speed, station, and normalized motion features."""

    def __init__(
        self,
        planets: Optional[List[str]] = None,
        station_threshold: float = 0.05,
        **kwargs,
    ):
        super().__init__(
            description="Planetary speed and station-state features.",
            tags=["astrology", "speed", "retrograde"],
            **kwargs,
        )
        self.planets = planets
        self.station_threshold = station_threshold

    def _compute_astrology(self, ephemeris_df: pd.DataFrame) -> pd.DataFrame:
        dates, _ = get_dates_planets(ephemeris_df)
        planets = self.planets or [
            "mercury",
            "venus",
            "mars",
            "jupiter",
            "saturn",
            "uranus",
            "neptune",
            "pluto",
        ]

        results = {}
        for planet in planets:
            speed = get_planet_field(ephemeris_df, planet, "speed").reindex(dates)
            if speed.empty:
                continue

            abs_speed = speed.abs()
            speed_mean = speed.expanding().mean()
            speed_std = speed.expanding().std().replace(0, np.nan)

            results[f"{planet}_speed"] = speed
            results[f"{planet}_speed_pct"] = speed.expanding().rank(pct=True)
            results[f"{planet}_station"] = (abs_speed < self.station_threshold).astype(int)
            results[f"{planet}_speed_zscore"] = (speed - speed_mean) / speed_std

        return pd.DataFrame(results) if results else pd.DataFrame(index=dates)
