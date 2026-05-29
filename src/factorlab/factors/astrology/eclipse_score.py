from __future__ import annotations

import numpy as np
import pandas as pd

from factorlab.factors.astrology.base import AstrologyFactor
from factorlab.factors.astrology.common import compute_aspect_distance, get_dates_planets, get_planet_longitude


class EclipseScore(AstrologyFactor):
    """Lunation/eclipse detection and decayed impact score."""

    def __init__(self, decay_window: int = 14, eclipse_orb: float = 1.5, **kwargs):
        super().__init__(
            description="Solar/lunar eclipse proximity and weighted impact score.",
            tags=["astrology", "eclipse", "lunation"],
            **kwargs,
        )
        self.decay_window = decay_window
        self.eclipse_orb = eclipse_orb

    def _compute_astrology(self, ephemeris_df: pd.DataFrame) -> pd.DataFrame:
        dates, _ = get_dates_planets(ephemeris_df)
        sun_lon = get_planet_longitude(ephemeris_df, "sun").reindex(dates)
        moon_lon = get_planet_longitude(ephemeris_df, "moon").reindex(dates)
        node_lon = get_planet_longitude(ephemeris_df, "north_node").reindex(dates)

        if sun_lon.empty or moon_lon.empty or node_lon.empty:
            return pd.DataFrame(index=dates)

        elongation = (moon_lon - sun_lon) % 360
        lunation_orb = 12.0
        new_moon = (elongation < lunation_orb) | (elongation > (360 - lunation_orb))
        full_moon = (elongation > (180 - lunation_orb)) & (elongation < (180 + lunation_orb))

        sun_node_dist = compute_aspect_distance(sun_lon, node_lon, 0)
        sun_south_node_dist = compute_aspect_distance(sun_lon, node_lon, 180)
        near_node = (sun_node_dist <= self.eclipse_orb) | (sun_south_node_dist <= self.eclipse_orb)

        solar_eclipse = (new_moon & near_node).astype(int)
        lunar_eclipse = (full_moon & near_node).astype(int)
        eclipse_events = (solar_eclipse | lunar_eclipse).astype(float)

        score = pd.Series(0.0, index=dates)
        eclipse_dates = dates[eclipse_events > 0]
        if len(eclipse_dates) > 0:
            date_ordinals = np.array([d.toordinal() for d in dates])
            for d in eclipse_dates:
                delta = date_ordinals - d.toordinal()
                mask = (delta >= 0) & (delta <= self.decay_window)
                decay = 1.0 - delta[mask] / self.decay_window
                score.iloc[mask] = np.maximum(score.iloc[mask].values, decay)

        return pd.DataFrame(
            {
                "solar_eclipse": solar_eclipse,
                "lunar_eclipse": lunar_eclipse,
                "eclipse_score": score,
                "eclipse_weighted_score": score * (1 + solar_eclipse.reindex(dates).fillna(0)),
            }
        )
