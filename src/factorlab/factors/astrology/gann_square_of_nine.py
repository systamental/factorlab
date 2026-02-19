from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from factorlab.factors.astrology.base import AstrologyFactor
from factorlab.factors.astrology.common import get_dates_planets, get_planet_longitude

logger = logging.getLogger(__name__)


class GannSquareOfNine(AstrologyFactor):
    """Square-of-Nine alignment and support/resistance distances."""

    def __init__(self, price_col: str = "close", anchor_planet: str = "sun", **kwargs):
        super().__init__(
            description="Gann Square of Nine support/resistance and alignment features.",
            tags=["astrology", "gann", "square_of_nine"],
            **kwargs,
        )
        self.price_col = price_col
        self.anchor_planet = anchor_planet

    def _compute_astrology(self, ephemeris_df: pd.DataFrame) -> pd.DataFrame:
        dates, _ = get_dates_planets(ephemeris_df)
        if self.price_df is None:
            logger.warning("No price data provided for Gann So9.")
            return pd.DataFrame(index=dates)

        if isinstance(self.price_df.index, pd.MultiIndex):
            try:
                price = self.price_df[self.price_col].groupby("date").mean().reindex(dates)
            except Exception:
                price = self.price_df[self.price_col].reindex(dates)
        else:
            price = self.price_df[self.price_col].reindex(dates)

        if price.empty or price.isna().all():
            return pd.DataFrame(index=dates)

        sqrt_price = np.sqrt(price.abs())
        price_degree = (sqrt_price % 1) * 360

        results = {"so9_price_degree": price_degree}

        cardinal_angles = [0, 45, 90, 135, 180, 225, 270, 315]
        min_dist = pd.Series(180.0, index=dates)
        for angle in cardinal_angles:
            dist = (price_degree - angle).abs()
            dist = dist.where(dist <= 180, 360 - dist)
            min_dist = min_dist.where(min_dist < dist, dist)
        results["so9_cardinal_dist"] = min_dist

        for rotation, label in [(1, "90"), (2, "180"), (3, "270"), (4, "360")]:
            sr_up = (sqrt_price + rotation * 0.5) ** 2
            sr_down = (sqrt_price - rotation * 0.5).clip(lower=0) ** 2
            results[f"so9_sr_up_{label}"] = sr_up
            results[f"so9_sr_down_{label}"] = sr_down

        sr_up_180 = (sqrt_price + 1) ** 2
        sr_down_180 = (sqrt_price - 1).clip(lower=0) ** 2
        results["so9_dist_up_pct"] = (sr_up_180 - price) / price
        results["so9_dist_down_pct"] = (price - sr_down_180) / price

        planet_lon = get_planet_longitude(ephemeris_df, self.anchor_planet).reindex(dates)
        if not planet_lon.empty:
            alignment_dist = (price_degree - planet_lon).abs()
            alignment_dist = alignment_dist.where(alignment_dist <= 180, 360 - alignment_dist)
            results[f"so9_{self.anchor_planet}_alignment"] = alignment_dist
            results[f"so9_{self.anchor_planet}_conjunct"] = (
                (alignment_dist < 5).astype(float).fillna(0).astype(int)
            )

        return pd.DataFrame(results)
