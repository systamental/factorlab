import pandas as pd
from pandas.core.groupby import DataFrameGroupBy
import numpy as np
from typing import List, Any, Union

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
            window_type: str = "ewm",
            window_size: int = 30,
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
        self.window_type = window_type
        self.window_size = window_size
        self.scale = scale  # Higher Amihud = Less Liquid

    @property
    def inputs(self) -> List[str]:
        """
        Required input columns.
        Override in subclasses as needed.
        """
        return [self.input_col]

    def _get_ts_window_op(self, g: Union[pd.DataFrame, DataFrameGroupBy]) -> Any:
        """Helper to determine and initialize the correct window operation for time series."""
        if self.window_type == 'ewm':
            # Returns Rolling GroupBy object
            return g.ewm(span=self.window_size)
        elif self.window_type == 'rolling':
            # Returns Rolling GroupBy object
            return g.rolling(window=self.window_size)
        elif self.window_type == 'expanding':
            # Returns Expanding GroupBy object
            return g.expanding()
        else:
            raise ValueError(f"Unsupported window type: {self.window_type} for axis 'ts'. "
                             f"Must be 'rolling', 'expanding', or 'fixed'.")

    def _compute_liquidity(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes the Amihud illiquidity measure.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame containing the required return and volume columns.

        Returns
        -------
        pd.DataFrame
            A new DataFrame with the computed Amihud illiquidity values in the specified output column.
        """
        df['notional_value'] = df[self.price_col] * df[self.volume_col].replace(0, np.nan)
        df[self.output_col] = np.abs(df[self.return_col]) / df['notional_value']

        # moving window
        g = grouped(df)
        window_op = self._get_ts_window_op(g)
        # apply mean
        mean_df = window_op.mean()

        # drop multiindex level if present
        mean_df = maybe_droplevel(mean_df, level=0)

        if self.scale:
            mean_df[self.output_col] *= 1e6  # Scale to basis points per million dollars

        return mean_df[[self.output_col]]
