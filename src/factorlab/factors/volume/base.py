from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Union

import numpy as np
import pandas as pd

from factorlab.factors.base import Factor
from factorlab.utils import to_dataframe


class VolumeFactor(Factor, ABC):
    """Base class for volume/price interaction factors."""

    def __init__(
        self,
        price_col: str = "close",
        volume_col: str = "volume",
        output_col: Optional[str] = None,
        compress: bool = True,
        compression_window: int = 250,
        compression_min_periods: int = 30,
        compression_strength: float = 1.0,
        **kwargs: Any,
    ):
        super().__init__(
            name=self.__class__.__name__,
            description="Base class for volume factors.",
            category="Volume",
            tags=["volume", "flow", "microstructure"],
        )
        self.price_col = price_col
        self.volume_col = volume_col
        self.output_col = output_col
        self.compress = compress
        self.compression_window = compression_window
        self.compression_min_periods = compression_min_periods
        self.compression_strength = compression_strength
        self.kwargs = kwargs

    @property
    def inputs(self) -> List[str]:
        return [self.price_col, self.volume_col]

    def fit(
        self,
        X: Union[pd.Series, pd.DataFrame],
        y: Optional[Union[pd.Series, pd.DataFrame]] = None,
    ) -> "VolumeFactor":
        df_input = to_dataframe(X)
        self.validate_inputs(df_input)
        self._is_fitted = True
        return self

    def transform(self, X: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
        if not self._is_fitted:
            raise RuntimeError(f"Transform '{self.name}' must be fitted before calling transform()")

        df = to_dataframe(X).copy(deep=True)
        self.validate_inputs(df)
        df = df.sort_index()

        factor = self._compute_volume(df)
        if self.compress:
            factor = self._compress(factor)

        df[self._generate_name()] = factor.clip(-50, 50)
        return df

    @abstractmethod
    def _compute_volume(self, df: pd.DataFrame) -> pd.Series:
        raise NotImplementedError

    def _generate_name(self) -> str:
        return self.output_col or self.name

    def _is_multiindex(self, series: pd.Series) -> bool:
        return isinstance(series.index, pd.MultiIndex)

    def _safe_log(self, series: pd.Series) -> pd.Series:
        return np.log(series.where(series > 0, np.nan))

    def _shift_by_asset(self, series: pd.Series, periods: int) -> pd.Series:
        if self._is_multiindex(series):
            return series.groupby(level=1).shift(periods)
        return series.shift(periods)

    def _pct_change_by_asset(self, series: pd.Series, periods: int = 1) -> pd.Series:
        if self._is_multiindex(series):
            return series.groupby(level=1).pct_change(periods=periods, fill_method=None)
        return series.pct_change(periods=periods, fill_method=None)

    def _diff_by_asset(self, series: pd.Series, periods: int = 1) -> pd.Series:
        if self._is_multiindex(series):
            return series.groupby(level=1).diff(periods=periods)
        return series.diff(periods=periods)

    def _rolling_stat(
        self,
        series: pd.Series,
        window: int,
        stat: str,
        min_periods: Optional[int] = None,
        **kwargs: Any,
    ) -> pd.Series:
        min_periods = window if min_periods is None else min_periods

        if self._is_multiindex(series):
            rolled = getattr(
                series.groupby(level=1).rolling(window=window, min_periods=min_periods),
                stat,
            )(**kwargs)
            return rolled.droplevel(0).sort_index()

        return getattr(series.rolling(window=window, min_periods=min_periods), stat)(**kwargs)

    def _compress(self, raw: pd.Series) -> pd.Series:
        robust_scale = self._rolling_stat(
            raw.abs(),
            window=self.compression_window,
            stat="median",
            min_periods=self.compression_min_periods,
        ).replace(0, np.nan)

        normalized = raw / robust_scale
        return 50.0 * np.tanh(self.compression_strength * normalized)
