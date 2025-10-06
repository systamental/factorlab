import pandas as pd
from typing import Optional, Union, Dict, Type, ClassVar
from factorlab.core.base_transform import BaseTransform

from factorlab.factors.base import Factor
from factorlab.factors.trend.breakout import Breakout
from factorlab.factors.trend.price_acceleration import PriceAcceleration
from factorlab.factors.trend.price_momentum import PriceMomentum
from factorlab.factors.trend.ewma import EWMA
from factorlab.factors.trend.divergence import Divergence
from factorlab.factors.trend.time_trend import TimeTrend
from factorlab.factors.trend.alpha_momentum import AlphaMomentum
from factorlab.factors.trend.idio_ewma import IdiosyncraticEWMA
from factorlab.factors.trend.rsi import RSI
from factorlab.factors.trend.stochastic import Stochastic
from factorlab.factors.trend.intensity import Intensity
from factorlab.factors.trend.mw_difference import MWDifference
from factorlab.factors.trend.triple_ewma_diff import TripleEWMADifference
from factorlab.factors.trend.energy import Energy
from factorlab.factors.trend.snr import SNR
from factorlab.factors.trend.adx import ADX


class Trend(Factor):
    """
    A factory class for creating and computing various trend factors.

    This class acts as a facade, providing a simple and consistent interface
    for all trend factor calculations.

    Parameters
    ----------
    method: str, {'breakout', 'price_mom', 'ewma'}, default 'breakout'
        The trend factor method to compute.
    **kwargs:
        Additional keyword arguments to pass to the specific trend factor class.

    """
    # Map method names to their corresponding factor classes
    _METHOD_MAP: ClassVar[Dict[str, Type[BaseTransform]]] = {
        'breakout': Breakout,
        'price_momentum': PriceMomentum,
        'ewma': EWMA,
        'divergence': Divergence,
        'time_trend': TimeTrend,
        'price_acceleration': PriceAcceleration,
        'alpha_momentum': AlphaMomentum,
        'idio_ewma': IdiosyncraticEWMA,
        'rsi': RSI,
        'stochastic': Stochastic,
        'intensity': Intensity,
        'mw_difference': MWDifference,
        'triple_ewma_diff': TripleEWMADifference,
        'energy': Energy,
        'snr': SNR,
        'adx': ADX
    }

    @classmethod
    def get_factor_metadata(cls) -> pd.DataFrame:
        """
        Returns a DataFrame detailing all available factor aliases and descriptions
        by temporarily instantiating each class and accessing its 'name' and 'description' properties.
        """
        data = []
        for alias, factor_class in cls._METHOD_MAP.items():
            try:
                # 1. Create a temporary, dummy instance.
                # This ensures BaseTransform.__init__ runs and sets the default .name and .description.
                factor_instance = factor_class()

                # 2. Directly access the properties of the instance.
                data.append({
                    'Alias': alias,
                    'Class': factor_class.__name__,
                    'Description': factor_instance.description,
                })

            except Exception as e:
                # Important safety check: factors must be instantiable without arguments.
                print(f"Warning: Could not instantiate {factor_class.__name__} for metadata retrieval. Error: {e}")
                data.append({
                    'Alias': alias,
                    'Class': factor_class.__name__,
                    'Description': f'Instantiation Failed: {e}',
                })

        df = pd.DataFrame(data).set_index('Alias')
        return df

    def __init__(self,
                 method: str = 'price_mom',
                 **kwargs):

        super().__init__(name='Trend', description='A factory for various trend factors.', category='Trend')
        self.method = method
        self.kwargs = kwargs

        if self.method not in self._METHOD_MAP:
            raise ValueError(f"Invalid value factor method. "
                             f"Method must be one of: {list(self._METHOD_MAP.keys())}")

        trend_class = self._METHOD_MAP[self.method]
        self._factor: Factor = trend_class(**self.kwargs)

    def fit(self, X: Union[pd.Series, pd.DataFrame], y: Optional[Union[pd.Series, pd.DataFrame]] = None) -> 'Trend':
        """
        Fits the underlying factor (e.g., calculates log parameters, smoother parameters).

        Delegates the call to the internal factor instance.
        """
        self.validate_inputs(X)
        self._factor.fit(X)
        self._is_fitted = True
        return self

    def transform(self, data: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
        """
        Applies the transformation using the fitted underlying factor.

        Delegates the call to the internal factor instance.
        """
        if not self._is_fitted:
            raise RuntimeError("Value transform must be fitted before calling transform().")

        return self._factor.transform(data)

