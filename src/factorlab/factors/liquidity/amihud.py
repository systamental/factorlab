import pandas as pd
import numpy as np
from typing import List

from factorlab.factors.liquidity.base import LiquidityFactor
from factorlab.utils import grouped, maybe_droplevel


class Amihud(LiquidityFactor):
    """
    Computes the Amihud (2002) illiquidity measure:
    Return impact per unit of volume traded.
    """

    def __init__(
            self,
            return_col: str = 'ret',
            price_col: str = "close",
            volume_col: str = "volume",
            output_col: str = "amihud",
            scale: bool = False,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.name = 'AmihudIlliquidity'
        self.description = 'Amihud (2002) illiquidity measure: Return impact per unit of volume traded.'
        self.return_col = return_col
        self.price_col = price_col
        self.volume_col = volume_col
        self.output_col = output_col
        self.scale = scale  # Higher Amihud = Less Liquid

    @property
    def inputs(self) -> List[str]:
        """
        Required input columns.
        Override in subclasses as needed.
        """
        return [self.return_col, self.price_col, self.volume_col]

    def _compute_liquidity(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes the Amihud illiquidity measure.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame containing the required return and volume columns.

        Returns
        -------
        pd.Series
            A Series with the computed Amihud illiquidity values in the specified output column.
        """
        df['notional_value'] = df[self.price_col] * df[self.volume_col].replace(0, np.nan)
        df[self.output_col] = np.abs(df[self.return_col]) / df['notional_value']

        # moving window
        if self.smooth:
            g = grouped(df)
            window_op = self._get_ts_window_op(g)
            # apply mean
            df = window_op.mean()
            # drop multiindex level if present
            df = maybe_droplevel(df, level=0)

        if self.scale:
            df[self.output_col] *= 1e6  # Scale to basis points per million dollars

        return df[self.output_col]
