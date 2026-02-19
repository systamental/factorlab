from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from factorlab.factors.astrology.base import AstrologyFactor
from factorlab.factors.astrology.common import deg_to_lowest_180, get_dates_planets, get_planet_longitude

logger = logging.getLogger(__name__)


class PriceLongitudeAngles(AstrologyFactor):
    """Gann-style price projections from planetary longitude motion."""

    def __init__(
        self,
        planet: str = "sun",
        anchor_price: float = 1.0,
        scale: float = 1.0,
        mode: str = "single",
        planet2: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            description="Price-longitude projections using Gann fan ratios.",
            tags=["astrology", "gann", "price"],
            **kwargs,
        )
        self.planet = planet
        self.anchor_price = anchor_price
        self.scale = scale
        self.mode = mode
        self.planet2 = planet2

    def _compute_astrology(self, ephemeris_df: pd.DataFrame) -> pd.DataFrame:
        dates, _ = get_dates_planets(ephemeris_df)
        lon1 = get_planet_longitude(ephemeris_df, self.planet).reindex(dates)

        if self.mode == "single":
            lon = lon1
        elif self.mode in ("average", "synodic") and self.planet2 is not None:
            lon2 = get_planet_longitude(ephemeris_df, self.planet2).reindex(dates)
            if self.mode == "average":
                lon = (lon1 + lon2) / 2
            else:
                lon = deg_to_lowest_180(lon1 - lon2).abs()
        else:
            logger.warning("Invalid mode '%s' or missing planet2.", self.mode)
            return pd.DataFrame(index=dates)

        lon_diff = lon.diff()
        lon_diff = lon_diff.where(lon_diff.abs() < 180, lon_diff - np.sign(lon_diff) * 360)
        accumulated = lon_diff.cumsum().fillna(0)

        gann_ratios = {
            "1x1": 1.0,
            "1x2": 0.5,
            "2x1": 2.0,
            "1x3": 1.0 / 3.0,
            "3x1": 3.0,
            "1x4": 0.25,
            "4x1": 4.0,
            "1x8": 0.125,
            "8x1": 8.0,
        }

        results = {
            f"{self.planet}_gann_{name}": self.anchor_price + accumulated * self.scale * ratio
            for name, ratio in gann_ratios.items()
        }
        return pd.DataFrame(results)
