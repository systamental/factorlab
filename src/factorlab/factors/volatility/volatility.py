import pandas as pd
from typing import Dict, Type, ClassVar, Optional, Union
from factorlab.utils import to_dataframe

from factorlab.core.base_transform import BaseTransform
from factorlab.factors.base import Factor
from factorlab.factors.volatility.std import STD
from factorlab.factors.volatility.iqr import IQR
from factorlab.factors.volatility.mad import MAD
from factorlab.factors.volatility.atr import ATR
from factorlab.factors.volatility.garman_klass import GarmanKlass
from factorlab.factors.volatility.parkinson import Parkinson
from factorlab.factors.volatility.rogers_satchell import RogersSatchell
from factorlab.factors.volatility.yang_zhang import YangZhang
from factorlab.factors.volatility.idio_vol import IVol


class Vol(Factor):
    """
    A factory class for creating and computing various vol factors.

    This class acts as a facade, providing a simple and consistent interface
    for all vol factor calculations.

    Parameters
    ----------
    method: str, {'std', 'iqr', 'mad'}, default 'std'
        The skew factor method to compute.
    **kwargs:
        Additional keyword arguments to pass to the specific vol factor class.
    """
    _METHOD_MAP: ClassVar[Dict[str, Type[BaseTransform]]] = {
        'std': STD,
        'iqr': IQR,
        'mad': MAD,
        'atr': ATR,
        'ivol': IVol,
        'garman_klass': GarmanKlass,
        'parkinson': Parkinson,
        'rogers_satchell': RogersSatchell,
        'yang_zhang': YangZhang
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
                # create an instance of the factor class
                factor_instance = factor_class()

                # access its name and description
                data.append({
                    'Alias': alias,
                    'Class': factor_class.__name__,
                    'Description': factor_instance.description,
                })

            except Exception as e:
                # check for instantiation errors
                print(f"Warning: Could not instantiate {factor_class.__name__} for metadata retrieval. Error: {e}")
                data.append({
                    'Alias': alias,
                    'Class': factor_class.__name__,
                    'Description': f'Instantiation Failed: {e}',
                })

        df = pd.DataFrame(data).set_index('Alias')
        return df

    def __init__(self,
                 method: str = 'std',
                 **kwargs):
        super().__init__(name='Vol', description='A factory for various vol factors.', category='Vol')
        self.method = method
        self.kwargs = kwargs

        if self.method not in self._METHOD_MAP:
            raise ValueError(f"Invalid value factor method. "
                             f"Method must be one of: {list(self._METHOD_MAP.keys())}")

        vol_class = self._METHOD_MAP[self.method]
        self._factor: Factor = vol_class(**self.kwargs)

    def fit(self, X: Union[pd.Series, pd.DataFrame], y: Optional[Union[pd.Series, pd.DataFrame]] = None) -> 'Vol':
        """
        Fits the underlying factor (e.g., calculates log parameters, smoother parameters).

        Delegates the call to the internal factor instance.
        """
        df_input = to_dataframe(X)
        self.validate_inputs(df_input)
        self._factor.fit(df_input)
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
