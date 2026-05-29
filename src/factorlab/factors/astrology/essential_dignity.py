from __future__ import annotations

from typing import List, Optional

import pandas as pd

from factorlab.factors.astrology.base import AstrologyFactor
from factorlab.factors.astrology.common import DIGNITY_TABLE, get_dates_planets, get_planet_longitude, get_zodiac_sign


class EssentialDignity(AstrologyFactor):
    """Essential dignity scores by planet and aggregate."""

    def __init__(self, planets: Optional[List[str]] = None, **kwargs):
        super().__init__(
            description="Planetary essential dignity and aggregate strength scores.",
            tags=["astrology", "dignity", "classical"],
            **kwargs,
        )
        self.planets = planets

    def _compute_astrology(self, ephemeris_df: pd.DataFrame) -> pd.DataFrame:
        dates, available_planets = get_dates_planets(ephemeris_df)
        planets = self.planets or [p for p in available_planets if p in DIGNITY_TABLE]

        results = {}
        for planet in planets:
            if planet not in DIGNITY_TABLE:
                continue
            lon = get_planet_longitude(ephemeris_df, planet).reindex(dates)
            if lon.empty:
                continue

            sign = get_zodiac_sign(lon)
            score = sign.map(lambda s: DIGNITY_TABLE[planet].get(s, 0))
            results[f"{planet}_dignity"] = score

        df = pd.DataFrame(results)
        if not df.empty:
            df["dignity_aggregate"] = df.sum(axis=1)
        return df
