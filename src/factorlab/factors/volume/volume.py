import pandas as pd
from typing import ClassVar, Dict, Optional, Type, Union

from factorlab.core.base_transform import BaseTransform
from factorlab.factors.base import Factor
from factorlab.factors.volume.volume_momentum import VolumeMomentum
from factorlab.factors.volume.delta_volume_momentum import DeltaVolumeMomentum
from factorlab.factors.volume.volume_weighted_ma_over_ma import VolumeWeightedMAOverMA
from factorlab.factors.volume.diff_volume_weighted_ma_over_ma import DiffVolumeWeightedMAOverMA
from factorlab.factors.volume.price_volume_fit import PriceVolumeFit
from factorlab.factors.volume.diff_price_volume_fit import DiffPriceVolumeFit
from factorlab.factors.volume.delta_price_volume_fit import DeltaPriceVolumeFit
from factorlab.factors.volume.on_balance_volume import OnBalanceVolume
from factorlab.factors.volume.delta_on_balance_volume import DeltaOnBalanceVolume
from factorlab.factors.volume.positive_volume_indicator import PositiveVolumeIndicator
from factorlab.factors.volume.delta_positive_volume_indicator import DeltaPositiveVolumeIndicator
from factorlab.factors.volume.negative_volume_indicator import NegativeVolumeIndicator
from factorlab.factors.volume.delta_negative_volume_indicator import DeltaNegativeVolumeIndicator
from factorlab.factors.volume.product_price_volume import ProductPriceVolume
from factorlab.factors.volume.sum_price_volume import SumPriceVolume
from factorlab.factors.volume.delta_product_price_volume import DeltaProductPriceVolume
from factorlab.factors.volume.delta_sum_price_volume import DeltaSumPriceVolume
from factorlab.utils import to_dataframe


class Volume(Factor):
    """Factory class for volume factors."""

    _METHOD_MAP: ClassVar[Dict[str, Type[BaseTransform]]] = {
        "volume_momentum": VolumeMomentum,
        "delta_volume_momentum": DeltaVolumeMomentum,
        "volume_weighted_ma_over_ma": VolumeWeightedMAOverMA,
        "diff_volume_weighted_ma_over_ma": DiffVolumeWeightedMAOverMA,
        "price_volume_fit": PriceVolumeFit,
        "diff_price_volume_fit": DiffPriceVolumeFit,
        "delta_price_volume_fit": DeltaPriceVolumeFit,
        "on_balance_volume": OnBalanceVolume,
        "delta_on_balance_volume": DeltaOnBalanceVolume,
        "positive_volume_indicator": PositiveVolumeIndicator,
        "delta_positive_volume_indicator": DeltaPositiveVolumeIndicator,
        "negative_volume_indicator": NegativeVolumeIndicator,
        "delta_negative_volume_indicator": DeltaNegativeVolumeIndicator,
        "product_price_volume": ProductPriceVolume,
        "sum_price_volume": SumPriceVolume,
        "delta_product_price_volume": DeltaProductPriceVolume,
        "delta_sum_price_volume": DeltaSumPriceVolume,
    }

    _ALIASES: ClassVar[Dict[str, str]] = {
        "vmom": "volume_momentum",
        "dvmom": "delta_volume_momentum",
        "vwmama": "volume_weighted_ma_over_ma",
        "dvwmama": "diff_volume_weighted_ma_over_ma",
        "pvf": "price_volume_fit",
        "difpvf": "diff_price_volume_fit",
        "dpvf": "delta_price_volume_fit",
        "obv": "on_balance_volume",
        "dobv": "delta_on_balance_volume",
        "pvi": "positive_volume_indicator",
        "dpvi": "delta_positive_volume_indicator",
        "nvi": "negative_volume_indicator",
        "dnvi": "delta_negative_volume_indicator",
        "ppv": "product_price_volume",
        "spv": "sum_price_volume",
        "dppv": "delta_product_price_volume",
        "dspv": "delta_sum_price_volume",
    }

    @classmethod
    def get_factor_metadata(cls) -> pd.DataFrame:
        data = []
        for alias, factor_class in cls._METHOD_MAP.items():
            try:
                factor_instance = factor_class()
                data.append(
                    {
                        "Alias": alias,
                        "Class": factor_class.__name__,
                        "Description": factor_instance.description,
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

    def __init__(self, method: str = "volume_momentum", **kwargs):
        super().__init__(
            name="Volume",
            description="A factory for volume-based factors.",
            category="Volume",
        )

        method = method.lower().strip()
        self.method = self._ALIASES.get(method, method)
        self.kwargs = kwargs

        if self.method not in self._METHOD_MAP:
            raise ValueError(
                f"Invalid volume factor method '{self.method}'. "
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
    ) -> "Volume":
        df_input = to_dataframe(X)
        self.validate_inputs(df_input)
        self._factor.fit(df_input)
        self._is_fitted = True
        return self

    def transform(self, data: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
        if not self._is_fitted:
            raise RuntimeError("Volume transform must be fitted before calling transform().")

        return self._factor.transform(data)
