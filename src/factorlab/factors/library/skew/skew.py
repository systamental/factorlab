import pandas as pd
from typing import Optional, Union
from factorlab.factors.base import Factor
from factorlab.factors.skew.skewness import Skew as SkewnessFactor
from factorlab.factors.skew._max import Max
from factorlab.factors.skew._min import Min
from factorlab.factors.skew.max_min import MaxMin
from factorlab.factors.skew.idio_skew import ISkew


class Skew(Factor):
    """
    A factory class for creating and computing various skew factors.

    This class acts as a facade, providing a simple and consistent interface
    for all skew factor calculations.

    Parameters
    ----------
    method: str, {'skew', 'max', 'min', 'max_min'}, default 'skew'
        The skew factor method to compute.
    **kwargs:
        Additional keyword arguments to pass to the specific skew factor class.
    """

    def __init__(self,
                 method: str = 'skew',
                 **kwargs):
        super().__init__(name='Skew', description='A factory for various skew factors.', category='Skew')
        self.method = method
        self.kwargs = kwargs

        # Map method names to their corresponding factor classes
        self._method_map = {
            'skew': SkewnessFactor,
            'max': Max,
            'min': Min,
            'max_min': MaxMin,
            'idio_skew': ISkew
        }

        if self.method not in self._method_map:
            raise ValueError(f"Invalid skew factor method. "
                             f"Method must be one of: {list(self._method_map.keys())}")

        skew_class = self._method_map[self.method]
        self._factor: Factor = skew_class(**self.kwargs)

    def fit(self, X: Union[pd.Series, pd.DataFrame], y: Optional[Union[pd.Series, pd.DataFrame]] = None) -> 'Skew':
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
            raise RuntimeError("Skew transform must be fitted before calling transform().")

        return self._factor.transform(data)

    def compute(self, X: pd.DataFrame) -> pd.DataFrame:
        """Backward-compatible compute method (auto-fit for stateless transforms)."""
        return super().compute(X)
