from __future__ import annotations

from typing import List, Optional

import pandas as pd

from factorlab.factors.astrology.base import AstrologyFactor
from factorlab.factors.astrology.common import (
    ASPECT_ANGLES,
    DEFAULT_ORBS,
    aspect_weight,
    compute_aspect_distance,
    get_dates_planets,
    get_planet_longitude,
)


class PlanetaryAspects(AstrologyFactor):
    """Angular planetary aspect distances, flags, and weights."""

    def __init__(
        self,
        planets: Optional[List[str]] = None,
        aspect_types: Optional[List[str]] = None,
        orb: float = 10.0,
        **kwargs,
    ):
        super().__init__(
            description="Planetary aspect distances, in-orb flags, and aspect weights.",
            tags=["astrology", "aspects", "synodic"],
            **kwargs,
        )
        self.planets = planets
        self.aspect_types = aspect_types
        self.orb = orb

    def _compute_astrology(self, ephemeris_df: pd.DataFrame) -> pd.DataFrame:
        dates, available_planets = get_dates_planets(ephemeris_df)
        planets = self.planets or available_planets
        aspect_types = self.aspect_types or [
            "conjunction",
            "semi_square",
            "sextile",
            "square",
            "trine",
            "sesquiquadrate",
            "opposition",
        ]

        results = {}
        for i, p1 in enumerate(planets):
            lon1 = get_planet_longitude(ephemeris_df, p1)
            if lon1.empty:
                continue
            for p2 in planets[i + 1 :]:
                lon2 = get_planet_longitude(ephemeris_df, p2)
                if lon2.empty:
                    continue
                lon1_a, lon2_a = lon1.align(lon2, join="inner")

                for aspect_name in aspect_types:
                    angle = ASPECT_ANGLES[aspect_name]
                    aspect_orb = DEFAULT_ORBS.get(aspect_name, self.orb)
                    dist = compute_aspect_distance(lon1_a, lon2_a, angle)
                    in_orb = (dist <= aspect_orb).astype(int)
                    weight = aspect_weight(dist, aspect_orb)

                    prefix = f"{p1}_{p2}_{aspect_name}"
                    results[f"{prefix}_dist"] = dist
                    results[f"{prefix}_in_orb"] = in_orb
                    results[f"{prefix}_weight"] = weight

        return pd.DataFrame(results) if results else pd.DataFrame(index=dates)
