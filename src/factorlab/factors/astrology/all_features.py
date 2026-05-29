from __future__ import annotations

from typing import Optional

import pandas as pd

from factorlab.factors.astrology.aspect_dynamics import AspectDynamics
from factorlab.factors.astrology.base import AstrologyFactor
from factorlab.factors.astrology.bradley_siderograph import BradleySiderograph
from factorlab.factors.astrology.commodity_natal_transits import CommodityNatalTransits
from factorlab.factors.astrology.cyclical_encoding import CyclicalEncoding
from factorlab.factors.astrology.declination_aspects import DeclinationAspects
from factorlab.factors.astrology.dewey_oscillators import DeweyOscillators
from factorlab.factors.astrology.eclipse_score import EclipseScore
from factorlab.factors.astrology.essential_dignity import EssentialDignity
from factorlab.factors.astrology.gann_square_of_nine import GannSquareOfNine
from factorlab.factors.astrology.heliocentric_features import HeliocentricFeatures
from factorlab.factors.astrology.lunar_features import LunarFeatures
from factorlab.factors.astrology.mcwhirter_nodal_cycle import McWhirterNodalCycle
from factorlab.factors.astrology.midpoint_activations import MidpointActivations
from factorlab.factors.astrology.natal_transit_aspects import NatalTransitAspects
from factorlab.factors.astrology.planetary_aspects import PlanetaryAspects
from factorlab.factors.astrology.planetary_ingress import PlanetaryIngress
from factorlab.factors.astrology.planetary_speed_features import PlanetarySpeedFeatures
from factorlab.factors.astrology.price_longitude_angles import PriceLongitudeAngles
from factorlab.factors.astrology.retrograde_indicator import RetrogradeIndicator
from factorlab.factors.astrology.synodic_cycle_phase import SynodicCyclePhase
from factorlab.factors.astrology.common import get_dates_planets


class AllAstrologyFeatures(AstrologyFactor):
    """Composite factor that returns the full astrology feature set."""

    def __init__(
        self,
        include_natal: bool = True,
        include_price_angles: bool = False,
        include_heliocentric: bool = False,
        include_commodity_natal: bool = False,
        include_gann_so9: bool = False,
        anchor_price: float = 1.0,
        scale: float = 1.0,
        helio_ephemeris_df: Optional[pd.DataFrame] = None,
        target_series: Optional[pd.Series] = None,
        **kwargs,
    ):
        super().__init__(
            description="Composite astrology factor bundle.",
            tags=["astrology", "composite", "feature_bundle"],
            **kwargs,
        )
        self.include_natal = include_natal
        self.include_price_angles = include_price_angles
        self.include_heliocentric = include_heliocentric
        self.include_commodity_natal = include_commodity_natal
        self.include_gann_so9 = include_gann_so9
        self.anchor_price = anchor_price
        self.scale = scale
        self.helio_ephemeris_df = helio_ephemeris_df
        self.target_series = target_series

    def _compute_astrology(self, ephemeris_df: pd.DataFrame) -> pd.DataFrame:
        dates, planets = get_dates_planets(ephemeris_df)
        features = []

        def _append(df: pd.DataFrame) -> None:
            if df is not None and not df.empty:
                features.append(df)

        _append(PlanetaryAspects().compute(ephemeris_df))
        _append(BradleySiderograph().compute(ephemeris_df))
        _append(RetrogradeIndicator().compute(ephemeris_df))
        _append(PlanetaryIngress().compute(ephemeris_df))
        _append(LunarFeatures().compute(ephemeris_df))
        _append(CyclicalEncoding().compute(ephemeris_df))

        if self.include_natal:
            _append(NatalTransitAspects(natal_date=self.natal_date).compute(ephemeris_df))

        if self.include_price_angles:
            for planet in ["sun", "jupiter", "saturn"]:
                if planet in planets:
                    _append(
                        PriceLongitudeAngles(
                            planet=planet,
                            anchor_price=self.anchor_price,
                            scale=self.scale,
                        ).compute(ephemeris_df)
                    )

        _append(SynodicCyclePhase().compute(ephemeris_df))
        _append(McWhirterNodalCycle().compute(ephemeris_df))
        _append(
            DeweyOscillators(
                data_driven=self.target_series is not None,
                target_series=self.target_series,
            ).compute(ephemeris_df)
        )
        _append(EssentialDignity().compute(ephemeris_df))
        _append(DeclinationAspects().compute(ephemeris_df))
        _append(EclipseScore().compute(ephemeris_df))
        _append(PlanetarySpeedFeatures().compute(ephemeris_df))
        _append(AspectDynamics().compute(ephemeris_df))
        _append(MidpointActivations().compute(ephemeris_df))

        if self.include_heliocentric and self.helio_ephemeris_df is not None:
            _append(HeliocentricFeatures(helio_ephemeris_df=self.helio_ephemeris_df).compute(ephemeris_df))

        if self.include_commodity_natal:
            _append(CommodityNatalTransits().compute(ephemeris_df))

        if self.include_gann_so9 and self.price_df is not None:
            _append(GannSquareOfNine(price_df=self.price_df).compute(ephemeris_df))

        if not features:
            return pd.DataFrame(index=dates)

        result = pd.concat(features, axis=1)
        return result.loc[:, ~result.columns.duplicated()]
