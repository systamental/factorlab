from __future__ import annotations

from typing import List, Optional

import pandas as pd

from factorlab.factors.astrology.base import AstrologyFactor
from factorlab.factors.astrology.common import (
    get_dates_planets,
    get_planet_longitude,
    get_zodiac_sign,
)


class PlanetaryIngress(AstrologyFactor):
    """Zodiac sign and ingress event flags per planet."""

    def __init__(self, planets: Optional[List[str]] = None, **kwargs):
        super().__init__(
            description="Planetary ingress events by zodiac sign boundary crossing.",
            tags=["astrology", "ingress", "zodiac"],
            **kwargs,
        )
        self.planets = planets

    def _compute_astrology(self, ephemeris_df: pd.DataFrame) -> pd.DataFrame:
        dates, available_planets = get_dates_planets(ephemeris_df)
        planets = self.planets or available_planets

        results = {}
        for planet in planets:
            lon = get_planet_longitude(ephemeris_df, planet)
            if lon.empty:
                continue

            lon = lon.reindex(dates)
            sign = get_zodiac_sign(lon)
            sign_prev = sign.shift(1)
            ingress = (sign != sign_prev).astype(int)
            ingress.iloc[0] = 0

            results[f"{planet}_sign"] = sign
            results[f"{planet}_ingress"] = ingress

        return pd.DataFrame(results)
