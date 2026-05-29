from __future__ import annotations

from typing import List, Optional, Tuple

import pandas as pd

from factorlab.factors.astrology.base import AstrologyFactor
from factorlab.factors.astrology.common import (
    compute_aspect_distance,
    deg_to_lowest_180,
    get_dates_planets,
    get_planet_longitude,
)


class MidpointActivations(AstrologyFactor):
    """Midpoint activation flags for trigger planets."""

    def __init__(
        self,
        midpoint_pairs: Optional[List[Tuple[str, str]]] = None,
        trigger_planets: Optional[List[str]] = None,
        orb: float = 2.0,
        **kwargs,
    ):
        super().__init__(
            description="Hamburg midpoint activation features.",
            tags=["astrology", "midpoints", "williams"],
            **kwargs,
        )
        self.midpoint_pairs = midpoint_pairs
        self.trigger_planets = trigger_planets
        self.orb = orb

    def _compute_astrology(self, ephemeris_df: pd.DataFrame) -> pd.DataFrame:
        dates, _ = get_dates_planets(ephemeris_df)
        midpoint_pairs = self.midpoint_pairs or [
            ("sun", "jupiter"),
            ("sun", "saturn"),
            ("sun", "moon"),
            ("jupiter", "saturn"),
            ("jupiter", "uranus"),
            ("saturn", "neptune"),
            ("venus", "jupiter"),
            ("mars", "jupiter"),
            ("mars", "saturn"),
        ]
        trigger_planets = self.trigger_planets or ["sun", "mars", "mercury", "moon"]

        results = {}
        for p1, p2 in midpoint_pairs:
            lon1 = get_planet_longitude(ephemeris_df, p1).reindex(dates)
            lon2 = get_planet_longitude(ephemeris_df, p2).reindex(dates)
            if lon1.empty or lon2.empty:
                continue

            diff = deg_to_lowest_180(lon1 - lon2)
            midpoint = (lon2 + diff / 2) % 360

            for trigger in trigger_planets:
                if trigger in (p1, p2):
                    continue
                trigger_lon = get_planet_longitude(ephemeris_df, trigger).reindex(dates)
                if trigger_lon.empty:
                    continue

                conj_dist = compute_aspect_distance(trigger_lon, midpoint, 0)
                opp_dist = compute_aspect_distance(trigger_lon, midpoint, 180)
                activation = ((conj_dist <= self.orb) | (opp_dist <= self.orb)).astype(int)
                results[f"midpoint_{p1}_{p2}_{trigger}"] = activation

        return pd.DataFrame(results) if results else pd.DataFrame(index=dates)
