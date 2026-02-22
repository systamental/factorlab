from __future__ import annotations
import pandas as pd
from factorlab.features.ratios import Ratios
from factorlab.factors.value.base import ValueFactor


class ValueRatio(ValueFactor):
    """
    Computes the value factor as a ratio of a fundamental metric to price.

    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = 'ValueRatio'
        self.description = 'Value Ratio factor computed as the ratio of a fundamental metric to price.'

    def _compute_value(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes the value ratio signal.
        """
        # compute value ratio
        value_df = Ratios(numerator_col=self.fundamental_col,
                          denominator_col=self.price_col,
                          output_col=self.output_col).compute(df)

        return value_df
