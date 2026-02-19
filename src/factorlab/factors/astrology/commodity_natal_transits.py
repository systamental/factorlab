from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from factorlab.factors.astrology.base import AstrologyFactor
from factorlab.factors.astrology.common import (
    ASPECT_ANGLES,
    COMMODITY_NATAL_DATES,
    aspect_weight,
    compute_aspect_distance,
    get_dates_planets,
    get_natal_positions,
    get_planet_longitude,
)


class CommodityNatalTransits(AstrologyFactor):
    """Ticker-specific transit scores using first-trade natal dates."""

    def __init__(
        self,
        ticker_natal_dates: Optional[Dict[str, str]] = None,
        planets: Optional[List[str]] = None,
        aspect_types: Optional[List[str]] = None,
        orb: float = 8.0,
        **kwargs,
    ):
        super().__init__(
            description="Commodity natal transit activation scores.",
            tags=["astrology", "natal", "commodities"],
            **kwargs,
        )
        self.ticker_natal_dates = ticker_natal_dates
        self.planets = planets
        self.aspect_types = aspect_types
        self.orb = orb

    def _compute_astrology(self, ephemeris_df: pd.DataFrame) -> pd.DataFrame:
        dates, _ = get_dates_planets(ephemeris_df)
        ticker_natal_dates = self.ticker_natal_dates or COMMODITY_NATAL_DATES
        planets = self.planets or ["jupiter", "saturn", "uranus", "neptune", "pluto"]
        aspect_types = self.aspect_types or ["conjunction", "square", "opposition"]

        all_results = {}
        for ticker, natal_str in ticker_natal_dates.items():
            natal_positions = get_natal_positions(pd.Timestamp(natal_str), planets)
            if not natal_positions:
                continue

            ticker_score = pd.Series(0.0, index=dates)
            for planet in planets:
                if planet not in natal_positions:
                    continue

                transit_lon = get_planet_longitude(ephemeris_df, planet).reindex(dates)
                if transit_lon.empty:
                    continue

                natal_lon = natal_positions[planet]
                for aspect_name in aspect_types:
                    angle = ASPECT_ANGLES[aspect_name]
                    dist = compute_aspect_distance(transit_lon, natal_lon, angle)
                    ticker_score = ticker_score + aspect_weight(dist, self.orb)

            all_results[f"natal_transit_{ticker}"] = ticker_score

        return pd.DataFrame(all_results) if all_results else pd.DataFrame(index=dates)
