import pandas as pd
from typing import List, Optional, Union

from factorlab.core.base_transform import BaseTransform
from factorlab.utils import grouped, maybe_droplevel, to_dataframe


class VWAP(BaseTransform):
    """
    Computes a simplified Volume Weighted Average Price (VWAP) using OHLC data.

    VWAP_t = (Close + Typical Price) / 2
    Typical Price = (Open + High + Low) / 3

    This is a stateless transform (no parameters learned) and is fully compatible
    with the fit/transform/fir_transform API for pipeline chaining.

    Parameters
    ----------
    open_col : str, default 'open'
        Column name for opening price.
    high_col : str, default 'high'
        Column name for highest price.
    low_col : str, default 'low'
        Column name for lowest price.
    close_col : str, default 'close'
        Column name for closing price.
    """

    def __init__(
        self,
        open_col: str = "open",
        high_col: str = "high",
        low_col: str = "low",
        close_col: str = "close",
    ):
        super().__init__(
            name="VWAP",
            description="Computes simplified VWAP from OHLC data."
        )

        self.open_col = open_col
        self.high_col = high_col
        self.low_col = low_col
        self.close_col = close_col

    @property
    def inputs(self) -> List[str]:
        """Required input columns for this transform."""
        return [self.open_col, self.high_col, self.low_col, self.close_col]

    def fit(self, X: Union[pd.Series, pd.DataFrame], y: Optional[Union[pd.Series, pd.DataFrame]] = None) -> 'VWAP':
        """Stateless fit: validates input and marks the transform as fitted."""
        df_input = to_dataframe(X)
        self.validate_inputs(df_input)
        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Compute the VWAP and return a new DataFrame with the output column."""
        if not self._is_fitted:
            raise RuntimeError(f"Transform '{self.name}' must be fitted before calling transform()")

        df_input = to_dataframe(X)
        self.validate_inputs(df_input)

        # Slice and copy the required columns (ensuring immutability)
        df_slice = df_input[self.inputs].copy(deep=True)

        # typical price
        typical_price = (df_slice[self.open_col] + df_slice[self.high_col] + df_slice[self.low_col]) / 3

        # vwap
        vwap = (df_input[self.close_col] + typical_price) / 2

        vwap_df = to_dataframe(vwap, 'vwap')

        return vwap_df


class NotionalValue(BaseTransform):
    """
    Computes a notional value using OHLCV data.

    NotionalValue_t = price_t * volume_t

    This is a stateless transform (no parameters learned) and is fully compatible
    with the fit/transform/fir_transform API for pipeline chaining.

    Parameters
    ----------
    price_col: str, default 'close'
        Column name of the price series.
    volume_col: str, default 'volume'
        Column name of the volume series.
    window_size: int, default 30
        The rolling window size for aggregation.'=
    agg_method: str, default 'mean'
        Aggregation method: 'mean' or 'sum'.
    """

    def __init__(
        self,
        price_col: str = 'close',
        volume_col: str = 'volume',
        window_size: int = 30,
        agg_method: str = "mean"
    ):
        super().__init__(
            name="NotionalValue",
            description="Computes notional value from OHLCV data."
        )
        self.price_col = price_col
        self.volume_col = volume_col
        self.window_size = window_size
        self.agg_method = agg_method

    @property
    def inputs(self) -> List[str]:
        """Required input columns for this transform."""
        return [self.price_col, self.volume_col]

    def fit(self,
            X: Union[pd.Series, pd.DataFrame],
            y: Optional[Union[pd.Series, pd.DataFrame]] = None) -> 'NotionalValue':
        """Stateless fit: validates input and marks the transform as fitted."""
        df_input = to_dataframe(X)
        self.validate_inputs(df_input)
        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Compute the VWAP and return a new DataFrame with the output column."""
        if not self._is_fitted:
            raise RuntimeError(f"Transform '{self.name}' must be fitted before calling transform()")

        df_input = to_dataframe(X)
        self.validate_inputs(df_input)

        # Slice and copy the required columns (ensuring immutability)
        df_slice = df_input[[self.price_col, self.volume_col]].copy(deep=True)

        # notional value
        nv = df_slice[self.price_col] * df_slice[self.volume_col]

        # aggregate
        g = grouped(nv)
        if self.agg_method == "mean":
            nv_agg = g.rolling(window=self.window_size, min_periods=1).mean()
        elif self.agg_method == "sum":
            nv_agg = g.rolling(window=self.window_size, min_periods=1).sum()
        else:
            raise ValueError(f"Unsupported aggregation method: {self.agg_method}")
        nv_agg = maybe_droplevel(nv_agg)

        notional_df = to_dataframe(nv_agg, 'notional_value')

        return notional_df
