import pandas as pd
from typing import Optional, Union
from factorlab.factors.base import Factor
from factorlab.factors.carry.carry_yield import Yield
from factorlab.factors.carry.carry_vol import CarryVol


class Carry(Factor):
    """
    A factory class for creating and computing various carry factors.

    This class acts as a facade, providing a simple and consistent interface
    for all carry factor calculations.

    Parameters
    ----------
    method: str, {'carry', 'carry_vol'}, default 'carry'
        The carry factor method to compute.
    **kwargs:
        Additional keyword arguments to pass to the specific carry factor class.
    """

    def __init__(self,
                 method: str = 'carry',
                 **kwargs):
        super().__init__(name='Carry', description='A factory for various carry factors.', category='Carry')
        self.method = method
        self.kwargs = kwargs

        # Map method names to their corresponding factor classes
        self._method_map = {
            'carry': Yield,
            'carry_vol': CarryVol
        }

        if self.method not in self._method_map:
            raise ValueError(f"Invalid carry factor method. "
                             f"Method must be one of: {list(self._method_map.keys())}")

        carry_class = self._method_map[self.method]
        self._factor: Factor = carry_class(**self.kwargs)

    def fit(self, X: Union[pd.Series, pd.DataFrame], y: Optional[Union[pd.Series, pd.DataFrame]] = None) -> 'Carry':
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
