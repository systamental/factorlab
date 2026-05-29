from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

from factorlab.factors.astrology.base import AstrologyFactor
from factorlab.factors.astrology.common import ASPECT_ANGLES, DEFAULT_ORBS, compute_aspect_distance, get_dates_planets, get_zodiac_sign

logger = logging.getLogger(__name__)


class HeliocentricFeatures(AstrologyFactor):
    """Heliocentric signs, phases, and in-orb aspect flags."""

    def __init__(
        self,
        helio_ephemeris_df: Optional[pd.DataFrame] = None,
        planets: Optional[List[str]] = None,
        aspect_types: Optional[List[str]] = None,
        orb: float = 10.0,
        **kwargs,
    ):
        super().__init__(
            description="Heliocentric feature set for lead-lag testing.",
            tags=["astrology", "heliocentric", "jensen"],
            **kwargs,
        )
        self.helio_ephemeris_df = helio_ephemeris_df
        self.planets = planets
        self.aspect_types = aspect_types
        self.orb = orb

    def _compute_astrology(self, ephemeris_df: pd.DataFrame) -> pd.DataFrame:
        dates, _ = get_dates_planets(ephemeris_df)
        helio_ephemeris_df = self.helio_ephemeris_df
        if helio_ephemeris_df is None:
            logger.warning("No heliocentric ephemeris provided; skipping.")
            return pd.DataFrame(index=dates)

        planets = self.planets
        if planets is None:
            planets = [
                p
                for p in helio_ephemeris_df.index.get_level_values("ticker").unique()
                if p not in ("sun", "moon", "north_node")
            ]

        aspect_types = self.aspect_types or ["conjunction", "sextile", "square", "trine", "opposition"]

        results = {}
        for planet in planets:
            try:
                lon = helio_ephemeris_df.xs(planet, level="ticker")["longitude"].reindex(dates)
            except KeyError:
                continue
            if lon.empty:
                continue

            results[f"helio_{planet}_sign"] = get_zodiac_sign(lon)
            rad = lon * np.pi / 180.0
            results[f"helio_{planet}_lon_sin"] = np.sin(rad)
            results[f"helio_{planet}_lon_cos"] = np.cos(rad)

        helio_pairs = [("jupiter", "saturn"), ("jupiter", "uranus"), ("saturn", "neptune"), ("saturn", "pluto")]
        for p1, p2 in helio_pairs:
            try:
                lon1 = helio_ephemeris_df.xs(p1, level="ticker")["longitude"].reindex(dates)
                lon2 = helio_ephemeris_df.xs(p2, level="ticker")["longitude"].reindex(dates)
            except KeyError:
                continue
            if lon1.empty or lon2.empty:
                continue

            phase = (lon1 - lon2) % 360
            rad = phase * np.pi / 180.0
            results[f"helio_{p1}_{p2}_phase"] = phase
            results[f"helio_{p1}_{p2}_phase_sin"] = np.sin(rad)
            results[f"helio_{p1}_{p2}_phase_cos"] = np.cos(rad)

            for aspect_name in aspect_types:
                angle = ASPECT_ANGLES[aspect_name]
                aspect_orb = DEFAULT_ORBS.get(aspect_name, self.orb)
                dist = compute_aspect_distance(lon1, lon2, angle)
                results[f"helio_{p1}_{p2}_{aspect_name}_in_orb"] = (dist <= aspect_orb).astype(int)

        return pd.DataFrame(results) if results else pd.DataFrame(index=dates)
