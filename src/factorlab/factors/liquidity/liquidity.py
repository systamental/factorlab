import pandas as pd
from typing import Optional, Union, Dict, Type, ClassVar

from factorlab.core.base_transform import BaseTransform
from factorlab.factors.base import Factor
from factorlab.factors.liquidity.amihud import Amihud
from factorlab.factors.liquidity.edge import EDGE
from factorlab.factors.liquidity.high_low_spread import HighLowSpreadEstimator
from factorlab.factors.liquidity.notional_value import NotionalValue


class Liquidity(Factor):
    """
    A factory class for creating and computing various low risk factors.

    This class acts as a facade, providing a simple and consistent interface
    for all low risk factor construction.

    Parameters
    ----------
    method: str, {'beta', 'max_dd'}, default 'beta'
        The low risk factor method to compute.
    **kwargs:
        Additional keyword arguments to pass to the specific low risk factor class.
    """

    # Map method names to their corresponding factor classes
    _METHOD_MAP: ClassVar[Dict[str, Type[BaseTransform]]] = {
        'amihud': Amihud,
        'edge': EDGE,
        'high_low_spread': HighLowSpreadEstimator,
        'notional_value': NotionalValue
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
                 method: str = 'amihud',
                 **kwargs):
        super().__init__(name='Liquidity',
                         description='A factory for various liquidity factors.',
                         category='Liquidity')
        self.method = method
        self.kwargs = kwargs

        if self.method not in self._METHOD_MAP:
            raise ValueError(f"Invalid liquidity factor method. "
                             f"Method must be one of: {list(self._METHOD_MAP.keys())}")

        lr_class = self._METHOD_MAP[self.method]
        self._factor: Factor = lr_class(**self.kwargs)

    def fit(self, X: Union[pd.Series, pd.DataFrame], y: Optional[Union[pd.Series, pd.DataFrame]] = None) -> 'Liquidity':
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
