import pandas as pd
from typing import Optional, Union
from factorlab.factors.base import Factor
from factorlab.factors.value.value_ratio import ValueRatio
from factorlab.factors.value.value_residual import ValueResidual


class Value(Factor):
    """
    A factory class for creating and computing various value factors.

    This class acts as a facade, providing a simple and consistent interface
    for all value factor calculations.
    """

    def __init__(self,
                 method: str = 'value_ratio',
                 **kwargs):
        """
        Constructor.

        Parameters
        ----------
        method: str, {'value_ratio', 'value_residual'}, default 'value_ratio'
            The value factor method to compute.
        **kwargs:
            Additional keyword arguments to pass to the specific value factor class.
        """
        super().__init__(name='Value', description='A factory for various value factors.', category='Value')
        self.method = method
        self.kwargs = kwargs

        # Map method names to their corresponding factor classes
        self._method_map = {
            'value_ratio': ValueRatio,
            'value_residual': ValueResidual,
        }

        if self.method not in self._method_map:
            raise ValueError(f"Invalid value factor method. "
                             f"Method must be one of: {list(self._method_map.keys())}")

        value_class = self._method_map[self.method]
        self._factor: Factor = value_class(**self.kwargs)

    def fit(self, X: Union[pd.Series, pd.DataFrame], y: Optional[Union[pd.Series, pd.DataFrame]] = None) -> 'Value':
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
