from __future__ import annotations

from typing import ClassVar, Dict, Optional, Type, Union

import pandas as pd

from factorlab.core.base_transform import BaseTransform
from factorlab.factors.base import Factor
from factorlab.factors.astrology.all_features import AllAstrologyFeatures
from factorlab.factors.astrology.aspect_dynamics import AspectDynamics
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


class Astrology(Factor):
    """Factory class for astrology factor indicators."""

    _METHOD_MAP: ClassVar[Dict[str, Type[BaseTransform]]] = {
        "planetary_aspects": PlanetaryAspects,
        "bradley_siderograph": BradleySiderograph,
        "retrograde_indicator": RetrogradeIndicator,
        "planetary_ingress": PlanetaryIngress,
        "natal_transit_aspects": NatalTransitAspects,
        "lunar_features": LunarFeatures,
        "cyclical_encoding": CyclicalEncoding,
        "price_longitude_angles": PriceLongitudeAngles,
        "synodic_cycle_phase": SynodicCyclePhase,
        "mcwhirter_nodal_cycle": McWhirterNodalCycle,
        "dewey_oscillators": DeweyOscillators,
        "essential_dignity": EssentialDignity,
        "declination_aspects": DeclinationAspects,
        "eclipse_score": EclipseScore,
        "planetary_speed_features": PlanetarySpeedFeatures,
        "heliocentric_features": HeliocentricFeatures,
        "aspect_dynamics": AspectDynamics,
        "gann_square_of_nine": GannSquareOfNine,
        "commodity_natal_transits": CommodityNatalTransits,
        "midpoint_activations": MidpointActivations,
        "all_features": AllAstrologyFeatures,
    }

    _ALIASES: ClassVar[Dict[str, str]] = {
        "aspects": "planetary_aspects",
        "bradley": "bradley_siderograph",
        "retrograde": "retrograde_indicator",
        "ingress": "planetary_ingress",
        "natal_transits": "natal_transit_aspects",
        "lunar": "lunar_features",
        "cyclical": "cyclical_encoding",
        "price_angles": "price_longitude_angles",
        "synodic": "synodic_cycle_phase",
        "mcwhirter": "mcwhirter_nodal_cycle",
        "dewey": "dewey_oscillators",
        "dignity": "essential_dignity",
        "declination": "declination_aspects",
        "eclipse": "eclipse_score",
        "speed": "planetary_speed_features",
        "helio": "heliocentric_features",
        "dynamics": "aspect_dynamics",
        "so9": "gann_square_of_nine",
        "commodity_natal": "commodity_natal_transits",
        "midpoints": "midpoint_activations",
        "all": "all_features",
    }

    @classmethod
    def get_factor_metadata(cls) -> pd.DataFrame:
        data = []
        for alias, factor_class in cls._METHOD_MAP.items():
            try:
                instance = factor_class()
                data.append(
                    {
                        "Alias": alias,
                        "Class": factor_class.__name__,
                        "Description": instance.description,
                    }
                )
            except Exception as exc:
                data.append(
                    {
                        "Alias": alias,
                        "Class": factor_class.__name__,
                        "Description": f"Instantiation Failed: {exc}",
                    }
                )

        return pd.DataFrame(data).set_index("Alias")

    def __init__(self, method: str = "all_features", **kwargs):
        super().__init__(
            name="Astrology",
            description="A factory for astrology factors.",
            category="Astrology",
            tags=["astrology", "ephemeris", "cycles"],
        )

        method = method.lower().strip()
        self.method = self._ALIASES.get(method, method)
        self.kwargs = kwargs

        if self.method not in self._METHOD_MAP:
            raise ValueError(
                f"Invalid astrology factor method '{self.method}'. "
                f"Method must be one of: {list(self._METHOD_MAP.keys())}"
            )

        factor_class = self._METHOD_MAP[self.method]
        self._factor: Factor = factor_class(**self.kwargs)

    @property
    def inputs(self) -> list[str]:
        return self._factor.inputs

    def fit(
        self,
        X: Union[pd.Series, pd.DataFrame],
        y: Optional[Union[pd.Series, pd.DataFrame]] = None,
    ) -> "Astrology":
        self._factor.fit(X, y)
        self._is_fitted = True
        return self

    def transform(self, data: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
        if not self._is_fitted:
            raise RuntimeError("Astrology transform must be fitted before calling transform().")

        return self._factor.transform(data)
