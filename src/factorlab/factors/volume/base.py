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

    # Shared raw components
    def _raw_volume_momentum(self, df: pd.DataFrame, hist_length: int, multiplier: int) -> pd.Series:
        volume = df[self.volume_col]
        short_ma = self._rolling_stat(volume, window=hist_length, stat="mean")
        long_ma = self._rolling_stat(volume, window=hist_length * multiplier, stat="mean")
        return self._safe_log(short_ma / long_ma.replace(0, np.nan))

    def _raw_volume_weighted_ma_over_ma(self, df: pd.DataFrame, hist_length: int) -> pd.Series:
        close = df[self.price_col]
        volume = df[self.volume_col]
        pv = close * volume

        vwma = self._rolling_stat(pv, window=hist_length, stat="sum") / self._rolling_stat(
            volume, window=hist_length, stat="sum"
        ).replace(0, np.nan)
        ma = self._rolling_stat(close, window=hist_length, stat="mean")

        return self._safe_log(vwma / ma.replace(0, np.nan))

    def _raw_price_volume_fit(self, df: pd.DataFrame, hist_length: int) -> pd.Series:
        close = df[self.price_col]
        volume = df[self.volume_col]

        x = self._safe_log(volume)
        y = self._safe_log(close)

        mean_x = self._rolling_stat(x, window=hist_length, stat="mean")
        mean_y = self._rolling_stat(y, window=hist_length, stat="mean")
        mean_xy = self._rolling_stat(x * y, window=hist_length, stat="mean")
        mean_x2 = self._rolling_stat(x * x, window=hist_length, stat="mean")

        cov_xy = mean_xy - (mean_x * mean_y)
        var_x = mean_x2 - (mean_x * mean_x)
        return cov_xy / var_x.replace(0, np.nan)

    def _raw_on_balance_volume(self, df: pd.DataFrame, hist_length: int) -> pd.Series:
        close = df[self.price_col]
        volume = df[self.volume_col]

        close_diff = self._diff_by_asset(close, 1)
        signed_volume = volume * np.sign(close_diff)

        signed_sum = self._rolling_stat(signed_volume, window=hist_length, stat="sum")
        total_sum = self._rolling_stat(volume, window=hist_length, stat="sum")
        return signed_sum / total_sum.replace(0, np.nan)

    def _raw_positive_volume_indicator(self, df: pd.DataFrame, hist_length: int) -> pd.Series:
        close = df[self.price_col]
        volume = df[self.volume_col]

        rel_change = self._pct_change_by_asset(close, periods=1)
        prev_volume = self._shift_by_asset(volume, 1)
        filtered = rel_change.where(volume > prev_volume, 0.0)

        avg_change = self._rolling_stat(filtered, window=hist_length, stat="mean")
        norm_window = max(2 * hist_length, 250)
        std_change = self._rolling_stat(
            rel_change,
            window=norm_window,
            stat="std",
            min_periods=hist_length,
        ).replace(0, np.nan)
        return avg_change / std_change

    def _raw_negative_volume_indicator(self, df: pd.DataFrame, hist_length: int) -> pd.Series:
        close = df[self.price_col]
        volume = df[self.volume_col]

        rel_change = self._pct_change_by_asset(close, periods=1)
        prev_volume = self._shift_by_asset(volume, 1)
        filtered = rel_change.where(volume < prev_volume, 0.0)

        avg_change = self._rolling_stat(filtered, window=hist_length, stat="mean")
        norm_window = max(2 * hist_length, 250)
        std_change = self._rolling_stat(
            rel_change,
            window=norm_window,
            stat="std",
            min_periods=hist_length,
        ).replace(0, np.nan)
        return avg_change / std_change

    def _normalized_volume_and_price_change(
        self,
        df: pd.DataFrame,
        norm_lookback: int,
        norm_min_periods: int,
    ) -> tuple[pd.Series, pd.Series]:
        close = df[self.price_col]
        volume = df[self.volume_col]

        prior_volume = self._shift_by_asset(volume, 1)
        median_volume = self._rolling_stat(
            prior_volume,
            window=norm_lookback,
            stat="median",
            min_periods=norm_min_periods,
        ).replace(0, np.nan)
        normalized_volume = volume / median_volume

        log_close = self._safe_log(close)
        price_change = self._diff_by_asset(log_close, 1)
        prior_change = self._shift_by_asset(price_change, 1)

        median_change = self._rolling_stat(
            prior_change,
            window=norm_lookback,
            stat="median",
            min_periods=norm_min_periods,
        )
        q75 = self._rolling_stat(
            prior_change,
            window=norm_lookback,
            stat="quantile",
            min_periods=norm_min_periods,
            q=0.75,
        )
        q25 = self._rolling_stat(
            prior_change,
            window=norm_lookback,
            stat="quantile",
            min_periods=norm_min_periods,
            q=0.25,
        )
        iqr = (q75 - q25).replace(0, np.nan)

        normalized_change = (price_change - median_change) / iqr
        return normalized_volume, normalized_change

    def _raw_product_price_volume(
        self,
        df: pd.DataFrame,
        hist_length: int,
        norm_lookback: int = 250,
        norm_min_periods: int = 50,
    ) -> pd.Series:
        norm_vol, norm_change = self._normalized_volume_and_price_change(df, norm_lookback, norm_min_periods)
        precursor = norm_vol * norm_change
        return self._rolling_stat(precursor, window=hist_length, stat="mean")

    def _raw_sum_price_volume(
        self,
        df: pd.DataFrame,
        hist_length: int,
        norm_lookback: int = 250,
        norm_min_periods: int = 50,
    ) -> pd.Series:
        norm_vol, norm_change = self._normalized_volume_and_price_change(df, norm_lookback, norm_min_periods)
        precursor = norm_vol + norm_change.abs()
        precursor = precursor.where(norm_change >= 0, -precursor)
        return self._rolling_stat(precursor, window=hist_length, stat="mean")
