import pandas as pd
import numpy as np
from typing import List

from factorlab.factors.liquidity.base import LiquidityFactor
from factorlab.utils import grouped, maybe_droplevel


class NotionalValue(LiquidityFactor):
    """
    Computes the Amihud (2002) illiquidity measure:
    Return impact per unit of volume traded.
    """

    def __init__(
            self,
            price_col: str = "close",
            volume_col: str = "volume",
            output_col: str = "notional_value",
            **kwargs
    ):
        super().__init__(**kwargs)

        self.name = 'NotionalValue'
        self.description = 'Notional Value based liquidity measure.'
        self.price_col = price_col
        self.volume_col = volume_col
        self.output_col = output_col

    @property
    def inputs(self) -> List[str]:
        """
        Required input columns.
        Override in subclasses as needed.
        """
        return [self.price_col, self.volume_col]

    def _compute_liquidity(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes the notional value measure, aka dollar volume when price is in USD * volume.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame containing the required price and volume columns.

        Returns
        -------
        pd.Series
            A Series with the computed notional value in the specified output column.
        """
        df[self.output_col] = df[self.price_col] * df[self.volume_col].replace(0, np.nan)

        # moving window
        if self.smooth:
            g = grouped(df)
            window_op = self._get_ts_window_op(g)
            # apply mean
            df = window_op.mean()
            # drop multiindex level if present
            df = maybe_droplevel(df, level=0)

        return df[self.output_col]
