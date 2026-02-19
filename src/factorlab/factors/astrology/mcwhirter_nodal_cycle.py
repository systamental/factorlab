from __future__ import annotations

import pandas as pd

from factorlab.factors.astrology.base import AstrologyFactor
from factorlab.factors.astrology.common import MCWHIRTER_BULLISH_SIGNS, get_dates_planets, get_planet_longitude, get_zodiac_sign


class McWhirterNodalCycle(AstrologyFactor):
    """McWhirter 18.6-year nodal cycle regime flags."""

    def __init__(self, **kwargs):
        super().__init__(
            description="North Node trend and extreme-zone flags per McWhirter.",
            tags=["astrology", "mcwhirter", "node"],
            **kwargs,
        )

    def _compute_astrology(self, ephemeris_df: pd.DataFrame) -> pd.DataFrame:
        dates, _ = get_dates_planets(ephemeris_df)
        node_lon = get_planet_longitude(ephemeris_df, "north_node").reindex(dates)
        if node_lon.empty:
            return pd.DataFrame(index=dates)

        node_sign = get_zodiac_sign(node_lon)
        degree_in_sign = node_lon % 30

        results = {
            "node_sign": node_sign,
            "node_trend": node_sign.map(lambda s: 1 if s in MCWHIRTER_BULLISH_SIGNS else -1),
            "node_extreme_top": ((node_sign == 4) & (degree_in_sign < 10)).astype(int),
            "node_extreme_bottom": ((node_sign == 10) & (degree_in_sign < 10)).astype(int),
        }
        return pd.DataFrame(results)
