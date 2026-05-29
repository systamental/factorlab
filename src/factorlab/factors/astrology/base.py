from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Union

import pandas as pd

from factorlab.factors.base import Factor
from factorlab.factors.astrology.common import BTC_NATAL_DATE
from factorlab.utils import to_dataframe


class AstrologyFactor(Factor, ABC):
    """Base class for astrology factor indicators built on ephemeris data."""

    def __init__(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        price_df: Optional[pd.DataFrame] = None,
        natal_date: Optional[Union[str, pd.Timestamp]] = None,
        broadcast_to_assets: bool = False,
        output_prefix: Optional[str] = None,
    ):
        super().__init__(
            name=name or self.__class__.__name__,
            description=description or "Astrology factor.",
            category="Astrology",
            tags=tags or ["astrology", "ephemeris", "cycles"],
        )
        self.price_df = price_df
        self.natal_date = pd.Timestamp(natal_date) if natal_date else pd.Timestamp(BTC_NATAL_DATE)
        self.broadcast_to_assets = broadcast_to_assets
        self.output_prefix = output_prefix

    @property
    def inputs(self) -> List[str]:
        return []

    def _validate_ephemeris(self, ephemeris_df: pd.DataFrame) -> None:
        if not isinstance(ephemeris_df.index, pd.MultiIndex):
            raise ValueError("Expected ephemeris input indexed by MultiIndex(date, ticker).")

        required_index_levels = {"date", "ticker"}
        if not required_index_levels.issubset(set(ephemeris_df.index.names)):
            raise ValueError(
                "Ephemeris index must contain levels named 'date' and 'ticker'."
            )

        required_cols = {"longitude"}
        missing = required_cols - set(ephemeris_df.columns)
        if missing:
            raise ValueError(f"Missing required ephemeris columns: {missing}")

    def fit(
        self,
        X: Union[pd.Series, pd.DataFrame],
        y: Optional[Union[pd.Series, pd.DataFrame]] = None,
    ) -> "AstrologyFactor":
        ephemeris_df = to_dataframe(X)
        self._validate_ephemeris(ephemeris_df)
        self._is_fitted = True
        return self

    @abstractmethod
    def _compute_astrology(self, ephemeris_df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def _apply_output_prefix(self, features: pd.DataFrame) -> pd.DataFrame:
        if not self.output_prefix or features.empty:
            return features
        out = features.copy()
        out.columns = [f"{self.output_prefix}{c}" for c in out.columns]
        return out

    def _broadcast_by_asset(
        self,
        features: pd.DataFrame,
        ephemeris_df: pd.DataFrame,
    ) -> pd.DataFrame:
        if features.empty:
            return pd.DataFrame(index=ephemeris_df.index)

        date_index = ephemeris_df.index.get_level_values("date")
        expanded = features.reindex(date_index)
        expanded.index = ephemeris_df.index
        return expanded

    def transform(self, X: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
        if not self._is_fitted:
            raise RuntimeError(f"Transform '{self.name}' must be fitted before calling transform().")

        ephemeris_df = to_dataframe(X).sort_index()
        self._validate_ephemeris(ephemeris_df)
        features = self._compute_astrology(ephemeris_df)
        features = self._apply_output_prefix(features)

        if self.broadcast_to_assets:
            return self._broadcast_by_asset(features, ephemeris_df)
        return features
