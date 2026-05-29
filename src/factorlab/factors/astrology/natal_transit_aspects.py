from __future__ import annotations

from typing import List, Optional, Union

import pandas as pd

from factorlab.factors.astrology.base import AstrologyFactor
from factorlab.factors.astrology.common import (
    ASPECT_ANGLES,
    aspect_weight,
    compute_aspect_distance,
    get_dates_planets,
    get_natal_positions,
    get_planet_longitude,
)


class NatalTransitAspects(AstrologyFactor):
    """Transit-to-natal aspect activations."""

    def __init__(
        self,
        planets: Optional[List[str]] = None,
        natal_date: Optional[Union[str, pd.Timestamp]] = None,
        aspect_types: Optional[List[str]] = None,
        orb: float = 10.0,
        **kwargs,
    ):
        super().__init__(
            natal_date=natal_date,
            description="Aspects between transiting planets and natal chart positions.",
            tags=["astrology", "natal", "transit"],
            **kwargs,
        )
        self.planets = planets
        self.aspect_types = aspect_types
        self.orb = orb
        self.natal_date_override = pd.Timestamp(natal_date) if natal_date else None

    def _compute_astrology(self, ephemeris_df: pd.DataFrame) -> pd.DataFrame:
        dates, available_planets = get_dates_planets(ephemeris_df)
        planets = self.planets or available_planets
        aspect_types = self.aspect_types or ["conjunction", "sextile", "square", "trine", "opposition"]
        natal_date = self.natal_date_override or self.natal_date

        natal_positions = get_natal_positions(natal_date, planets)
        if not natal_positions:
            return pd.DataFrame(index=dates)

        results = {}
        for planet in planets:
            if planet not in natal_positions:
                continue
            natal_lon = natal_positions[planet]
            transit_lon = get_planet_longitude(ephemeris_df, planet)
            if transit_lon.empty:
                continue

            transit_lon = transit_lon.reindex(dates)
            for aspect_name in aspect_types:
                angle = ASPECT_ANGLES[aspect_name]
                dist = compute_aspect_distance(transit_lon, natal_lon, angle)
                weight = aspect_weight(dist, self.orb)
                in_orb = (dist <= self.orb).astype(int)

                prefix = f"natal_{planet}_{aspect_name}"
                results[f"{prefix}_dist"] = dist
                results[f"{prefix}_in_orb"] = in_orb
                results[f"{prefix}_weight"] = weight

        return pd.DataFrame(results) if results else pd.DataFrame(index=dates)
