import pandas as pd
from typing import List, Optional, Union

from factorlab.core.base_transform import BaseTransform
from factorlab.utils import grouped, maybe_droplevel, to_dataframe


class VWAP(BaseTransform):
    """
    Computes a simplified Volume Weighted Average Price (VWAP) using OHLC data.

    VWAP_t = (Close + Typical Price) / 2
    Typical Price = (Open + High + Low) / 3

    Implements the Phase 1: Accumulation Contract: returns the full input DataFrame
    with the new 'vwap' column appended.

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
        output_col: str = "vwap"
    ):
        super().__init__(
            name="VWAP",
            description="Computes simplified VWAP from OHLC data."
        )

        self.open_col = open_col
        self.high_col = high_col
        self.low_col = low_col
        self.close_col = close_col
        self.output_col = output_col

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

    def _transform(self, df_input: pd.DataFrame) -> pd.DataFrame:
        """
        Private method containing the core VWAP calculation logic and accumulation.

        Parameters
        ----------
        df_input : pd.DataFrame
            The input data containing all columns (context).

        Returns
        -------
        pd.DataFrame
            The input data with the new 'vwap' column appended.
        """
        # typical price
        typical_price = (
            df_input[self.open_col] +
            df_input[self.high_col] +
            df_input[self.low_col]
        ) / 3

        # vwap
        vwap = (df_input[self.close_col] + typical_price) / 2

        df_input[self.output_col] = vwap

        return df_input

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Public method to compute the VWAP.

        It handles input validation, state checks, and implements the Accumulation Contract.
        """
        if not self._is_fitted:
            raise RuntimeError(f"Transform '{self.name}' must be fitted before calling transform()")

        df_input = to_dataframe(X).copy(deep=True)
        self.validate_inputs(df_input)

        return self._transform(df_input)


class NotionalValue(BaseTransform):
    """
    Computes a notional value using OHLCV data.

    NotionalValue_t = price_t * volume_t
    The result is then aggregated over a rolling window.

    Implements the Phase 1: Accumulation Contract: returns the full input DataFrame
    with the new 'notional_value' column appended.

    Parameters
    ----------
    price_col: str, default 'close'
        Column name of the price series.
    volume_col: str, default 'volume'
        Column name of the volume series.
    window_size: int, default 30
        The rolling window size for aggregation.
    agg_method: str, default 'mean'
        Aggregation method: 'mean' or 'sum'.
    """

    def __init__(
        self,
        price_col: str = 'close',
        volume_col: str = 'volume',
        output_col: str = 'notional_value',
        window_size: int = 30,
        agg_method: str = "mean"
    ):
        super().__init__(
            name="NotionalValue",
            description="Computes notional value from OHLCV data."
        )

        self.price_col = price_col
        self.volume_col = volume_col
        self.output_col = output_col
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

    def _transform(self, df_input: pd.DataFrame) -> pd.DataFrame:
        """
        Private method containing the core Notional Value calculation logic and accumulation.

        Parameters
        ----------
        df_input : pd.DataFrame
            The input data containing all columns (context).

        Returns
        -------
        pd.DataFrame
            The input data with the new 'notional_value' column appended.
        """
        # compute notional value
        nv = df_input[self.price_col] * df_input[self.volume_col]

        # aggregate over rolling window
        g = grouped(nv)
        if self.agg_method == "mean":
            nv_agg = g.rolling(window=self.window_size, min_periods=1).mean()
        elif self.agg_method == "sum":
            nv_agg = g.rolling(window=self.window_size, min_periods=1).sum()
        else:
            raise ValueError(f"Unsupported aggregation method: {self.agg_method}")

        nv_agg = maybe_droplevel(nv_agg)

        df_input[self.output_col] = nv_agg

        return df_input

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Public method to compute the aggregated Notional Value.

        It handles input validation, state checks, and implements the Accumulation Contract.
        """
        if not self._is_fitted:
            raise RuntimeError(f"Transform '{self.name}' must be fitted before calling transform()")

        df_input = to_dataframe(X).copy(deep=True)
        self.validate_inputs(df_input)

        return self._transform(df_input)
