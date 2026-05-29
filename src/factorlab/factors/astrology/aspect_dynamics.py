from __future__ import annotations

from typing import List, Optional

import pandas as pd

from factorlab.factors.astrology.base import AstrologyFactor
from factorlab.factors.astrology.common import (
    ASPECT_ANGLES,
    DEFAULT_ORBS,
    compute_aspect_distance,
    get_dates_planets,
    get_planet_longitude,
)


class AspectDynamics(AstrologyFactor):
    """Applying vs separating aspect-state flags."""

    def __init__(
        self,
        planets: Optional[List[str]] = None,
        aspect_types: Optional[List[str]] = None,
        orb: float = 10.0,
        **kwargs,
    ):
        super().__init__(
            description="Applying/separating state for planetary aspects.",
            tags=["astrology", "aspects", "dynamics"],
            **kwargs,
        )
        self.planets = planets
        self.aspect_types = aspect_types
        self.orb = orb

    def _compute_astrology(self, ephemeris_df: pd.DataFrame) -> pd.DataFrame:
        dates, available_planets = get_dates_planets(ephemeris_df)
        planets = self.planets or available_planets
        aspect_types = self.aspect_types or ["conjunction", "square", "trine", "opposition"]

        results = {}
        for i, p1 in enumerate(planets):
            lon1 = get_planet_longitude(ephemeris_df, p1).reindex(dates)
            if lon1.empty:
                continue
            for p2 in planets[i + 1 :]:
                lon2 = get_planet_longitude(ephemeris_df, p2).reindex(dates)
                if lon2.empty:
                    continue

                for aspect_name in aspect_types:
                    angle = ASPECT_ANGLES[aspect_name]
                    aspect_orb = DEFAULT_ORBS.get(aspect_name, self.orb)
                    dist = compute_aspect_distance(lon1, lon2, angle)
                    in_orb = dist <= aspect_orb

                    dist_change = dist.diff()
                    applying = (dist_change < 0).astype(int)
                    separating = (dist_change > 0).astype(int)

                    prefix = f"{p1}_{p2}_{aspect_name}"
                    results[f"{prefix}_applying"] = applying * in_orb.astype(int)
                    results[f"{prefix}_separating"] = separating * in_orb.astype(int)

        return pd.DataFrame(results) if results else pd.DataFrame(index=dates)
