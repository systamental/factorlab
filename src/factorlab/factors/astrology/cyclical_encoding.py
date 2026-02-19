from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from factorlab.factors.astrology.base import AstrologyFactor
from factorlab.factors.astrology.common import get_dates_planets, get_planet_field


class CyclicalEncoding(AstrologyFactor):
    """Sin/cos encoding for circular planetary fields."""

    def __init__(
        self,
        planets: Optional[List[str]] = None,
        fields: Optional[List[str]] = None,
        **kwargs,
    ):
        super().__init__(
            description="Sin/cos cyclical encodings for planetary coordinates.",
            tags=["astrology", "encoding", "cyclical"],
            **kwargs,
        )
        self.planets = planets
        self.fields = fields

    def _compute_astrology(self, ephemeris_df: pd.DataFrame) -> pd.DataFrame:
        dates, available_planets = get_dates_planets(ephemeris_df)
        planets = self.planets or available_planets
        fields = self.fields or ["longitude"]

        results = {}
        for planet in planets:
            for field in fields:
                series = get_planet_field(ephemeris_df, planet, field)
                if series.empty:
                    continue
                series = series.reindex(dates)

                if field == "longitude":
                    rad = series * np.pi / 180.0
                elif field == "declination":
                    rad = (series + 90) * np.pi / 180.0
                else:
                    rad = series * np.pi / 180.0

                results[f"{planet}_{field}_sin"] = np.sin(rad)
                results[f"{planet}_{field}_cos"] = np.cos(rad)

        return pd.DataFrame(results)
