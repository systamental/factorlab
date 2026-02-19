from __future__ import annotations

import pandas as pd

from factorlab.factors.astrology.base import AstrologyFactor
from factorlab.factors.astrology.common import (
    compute_aspect_distance,
    get_dates_planets,
    get_planet_field,
    get_planet_longitude,
)


class LunarFeatures(AstrologyFactor):
    """Lunar phase, lunation, and node-linked lunar features."""

    def __init__(self, **kwargs):
        super().__init__(
            description="Lunar phase, new/full moon, and node-linked lunar features.",
            tags=["astrology", "moon", "lunation"],
            **kwargs,
        )

    def _compute_astrology(self, ephemeris_df: pd.DataFrame) -> pd.DataFrame:
        dates, _ = get_dates_planets(ephemeris_df)
        moon_lon = get_planet_longitude(ephemeris_df, "moon").reindex(dates)
        sun_lon = get_planet_longitude(ephemeris_df, "sun").reindex(dates)

        phase = (moon_lon - sun_lon) % 360
        results = {
            "lunar_phase": phase,
            "new_moon": ((phase < 15) | (phase > 345)).astype(int),
            "full_moon": ((phase > 165) & (phase < 195)).astype(int),
        }

        moon_decl = get_planet_field(ephemeris_df, "moon", "declination").reindex(dates)
        if not moon_decl.empty:
            results["moon_decl_extreme"] = (moon_decl.abs() > 23.44).astype(int)

        north_node_lon = get_planet_longitude(ephemeris_df, "north_node").reindex(dates)
        if not north_node_lon.empty and not moon_lon.empty:
            node_dist = compute_aspect_distance(moon_lon, north_node_lon, 0)
            results["moon_north_node_conj"] = (node_dist < 10).astype(int)

        return pd.DataFrame(results)
