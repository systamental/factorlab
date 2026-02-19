from __future__ import annotations

import pandas as pd

from factorlab.factors.astrology.base import AstrologyFactor
from factorlab.factors.astrology.common import (
    ASPECT_ANGLES,
    BRADLEY_MIDTERM_PAIRS,
    BRADLEY_VALENCY,
    aspect_weight,
    compute_aspect_distance,
    get_dates_planets,
    get_planet_field,
    get_planet_longitude,
)


class BradleySiderograph(AstrologyFactor):
    """Bradley siderograph potential and component terms."""

    def __init__(self, multiplier: float = 1.0, **kwargs):
        super().__init__(
            description="Bradley siderograph composite potential.",
            tags=["astrology", "bradley", "cycles"],
            **kwargs,
        )
        self.multiplier = multiplier

    def _compute_astrology(self, ephemeris_df: pd.DataFrame) -> pd.DataFrame:
        dates, _ = get_dates_planets(ephemeris_df)
        long_terms = pd.Series(0.0, index=dates, name="long_terms")
        mid_terms = pd.Series(0.0, index=dates, name="mid_terms")

        for (p1, p2), valency_map in BRADLEY_VALENCY.items():
            lon1 = get_planet_longitude(ephemeris_df, p1)
            lon2 = get_planet_longitude(ephemeris_df, p2)
            if lon1.empty or lon2.empty:
                continue

            lon1 = lon1.reindex(dates)
            lon2 = lon2.reindex(dates)
            for aspect_name, valency in valency_map.items():
                angle = ASPECT_ANGLES[aspect_name]
                dist = compute_aspect_distance(lon1, lon2, angle)
                weight = aspect_weight(dist, orb=15.0)
                long_terms = long_terms + weight * valency

        for p1, p2 in BRADLEY_MIDTERM_PAIRS:
            lon1 = get_planet_longitude(ephemeris_df, p1)
            lon2 = get_planet_longitude(ephemeris_df, p2)
            if lon1.empty or lon2.empty:
                continue

            lon1 = lon1.reindex(dates)
            lon2 = lon2.reindex(dates)
            for aspect_name in ["conjunction", "sextile", "square", "trine", "opposition"]:
                angle = ASPECT_ANGLES[aspect_name]
                dist = compute_aspect_distance(lon1, lon2, angle)
                weight = aspect_weight(dist, orb=15.0)
                valency = 1 if aspect_name in ("sextile", "trine", "conjunction") else -1
                mid_terms = mid_terms + weight * valency

        mars_decl = get_planet_field(ephemeris_df, "mars", "declination").reindex(dates)
        venus_decl = get_planet_field(ephemeris_df, "venus", "declination").reindex(dates)
        declination_factor = (mars_decl.fillna(0) + venus_decl.fillna(0)) / 2

        sidereal_potential = self.multiplier * (long_terms + declination_factor) + mid_terms

        return pd.DataFrame(
            {
                "sidereal_potential": sidereal_potential,
                "long_terms": long_terms,
                "mid_terms": mid_terms,
                "declination_factor": declination_factor,
            }
        )
