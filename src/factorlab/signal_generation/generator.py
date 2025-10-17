import pandas as pd
from typing import Union, Dict, Any

from factorlab.core.base_transform import BaseTransform
from factorlab.signal_generation.discrete import (
    DiscreteZScoreSignal, DiscreteQuantileSignal, DiscreteRankSignal, Sign)
from factorlab.signal_generation.continuous import ScoreSignal, QuantileSignal, RankSignal, RawSignal, BuyHoldSignal


class SignalGenerator(BaseTransform):
    """
    Factory for converting continuous factor scores or forecasts into
    discrete or continuous trading signals.

    Delegates the work to specific signal generation classes.
    """

    def __init__(self, method: str = "zscore", **kwargs):
        super().__init__(name="Signal",
                         description="Converts scores into trading signals.")
        self.method = method
        self.kwargs = kwargs

        self._method_map: Dict[str, Any] = {
            'buy_hold': BuyHoldSignal,
            'raw': RawSignal,
            'zscore': ScoreSignal,
            'quantile': QuantileSignal,
            'rank': RankSignal,
            'sign': Sign,
            'discrete_zscore': DiscreteZScoreSignal,
            'discrete_quantile': DiscreteQuantileSignal,
            'discrete_rank': DiscreteRankSignal,
        }

        if self.method not in self._method_map:
            raise ValueError(f"Invalid signal method '{self.method}', must be one of {list(self._method_map.keys())}")

        # Instantiate the specific signal transformer using Composition
        self._transformer = self._method_map[self.method](**self.kwargs)

    def fit(self, X: Union[pd.Series, pd.DataFrame], y=None) -> 'SignalGenerator':
        """Fit the delegated signal transformer."""
        self.validate_inputs(X)
        self._transformer.fit(X)
        self._is_fitted = True
        return self

    def transform(self, X: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
        """Apply the delegated signal generation."""
        if not self._is_fitted:
            raise RuntimeError("SignalGenerator must be fitted before transform()")
        return self._transformer.transform(X)
