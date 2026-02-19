from __future__ import annotations

from typing import List, Optional

import pandas as pd

from factorlab.factors.astrology.base import AstrologyFactor
from factorlab.factors.astrology.common import get_dates_planets, get_planet_field


class RetrogradeIndicator(AstrologyFactor):
    """Planetary retrograde flags and aggregate count."""

    def __init__(self, planets: Optional[List[str]] = None, **kwargs):
        super().__init__(
            description="Planetary retrograde indicators.",
            tags=["astrology", "retrograde", "motion"],
            **kwargs,
        )
        self.planets = planets

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
            speed = get_planet_field(ephemeris_df, planet, "speed")
            if speed.empty:
                continue
            speed = speed.reindex(dates)
            results[f"{planet}_retrograde"] = (speed < 0).astype(int)

        df = pd.DataFrame(results)
        if not df.empty:
            df["retrograde_count"] = df.sum(axis=1)
        return df
