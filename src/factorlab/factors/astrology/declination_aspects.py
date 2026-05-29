from __future__ import annotations

from typing import List, Optional

import pandas as pd

from factorlab.factors.astrology.base import AstrologyFactor
from factorlab.factors.astrology.common import get_dates_planets, get_planet_field


class DeclinationAspects(AstrologyFactor):
    """Declination parallel and contra-parallel aspect flags."""

    def __init__(
        self,
        planets: Optional[List[str]] = None,
        orb: float = 1.0,
        **kwargs,
    ):
        super().__init__(
            description="Declination parallel and contra-parallel aspects.",
            tags=["astrology", "declination", "jensen"],
            **kwargs,
        )
        self.planets = planets
        self.orb = orb

    def _compute_astrology(self, ephemeris_df: pd.DataFrame) -> pd.DataFrame:
        dates, available_planets = get_dates_planets(ephemeris_df)
        planets = self.planets or [p for p in available_planets if p != "north_node"]

        results = {}
        for i, p1 in enumerate(planets):
            decl1 = get_planet_field(ephemeris_df, p1, "declination").reindex(dates)
            if decl1.empty:
                continue
            for p2 in planets[i + 1 :]:
                decl2 = get_planet_field(ephemeris_df, p2, "declination").reindex(dates)
                if decl2.empty:
                    continue

                parallel_dist = (decl1 - decl2).abs()
                contra_dist = (decl1 + decl2).abs()

                results[f"{p1}_{p2}_parallel"] = (parallel_dist <= self.orb).astype(int)
                results[f"{p1}_{p2}_contra_parallel"] = (contra_dist <= self.orb).astype(int)

        return pd.DataFrame(results) if results else pd.DataFrame(index=dates)
