from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator, Optional, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator

from factorlab.learning.utils import (
    apply_purge,
    extract_dates,
    mask_from_dates,
    unique_sorted_dates,
    validate_panel_xy,
)


class BasePanelSplit(BaseCrossValidator, ABC):
    """
    Base panel splitter for FactorLab indices of shape (date, ticker).
    """

    def __init__(
        self,
        date_level: int = 0,
        lookahead: int = 0,
        embargo: int = 0,
        drop_nas: bool = True,
    ):
        if lookahead < 0:
            raise ValueError("lookahead must be >= 0.")
        if embargo < 0:
            raise ValueError("embargo must be >= 0.")

        self.date_level = date_level
        self.lookahead = lookahead
        self.embargo = embargo
        self.drop_nas = drop_nas

    def _prepare_xy(
        self,
        X: pd.DataFrame,
        y: Optional[Union[pd.Series, pd.DataFrame]],
    ) -> tuple[pd.DataFrame, pd.DatetimeIndex, pd.DatetimeIndex]:
        validate_panel_xy(X=X, y=y, date_level=self.date_level, require_multiindex=True)

        if y is None:
            Xy = X.copy(deep=True)
        else:
            y_df = y.to_frame() if isinstance(y, pd.Series) else y
            Xy = pd.concat([X, y_df], axis=1)
            if self.drop_nas:
                Xy = Xy.dropna()
            else:
                Xy = Xy.dropna(subset=y_df.columns)
                Xy = Xy.dropna(subset=X.columns, how="all")

        dates = extract_dates(Xy.index, date_level=self.date_level)
        unique_dates = unique_sorted_dates(Xy.index, date_level=self.date_level)
        return Xy, dates, unique_dates

    @abstractmethod
    def split(
        self,
        X: pd.DataFrame,
        y: Optional[Union[pd.Series, pd.DataFrame]] = None,
        groups=None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        raise NotImplementedError

    def get_n_splits(self, X=None, y=None, groups=None):
        if X is None:
            raise ValueError("X is required to compute n_splits.")
        return len(list(self.split(X=X, y=y, groups=groups)))


class ExpandingKFoldPanelSplit(BasePanelSplit):
    """
    Time-respecting K-Fold splitter for panel data.
    """

    def __init__(
        self,
        n_splits: int = 5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if n_splits < 1:
            raise ValueError("n_splits must be >= 1.")
        self.n_splits = n_splits

    def split(self, X, y=None, groups=None):
        if groups is not None:
            raise ValueError("groups is not supported by this splitter.")
        Xy, dates, unique_dates = self._prepare_xy(X=X, y=y)

        intervals = np.array_split(unique_dates, self.n_splits + 1)
        for i in range(self.n_splits):
            train_dates = np.concatenate(intervals[: i + 1])
            test_dates = intervals[i + 1]
            if len(test_dates) == 0:
                continue
            train_dates = apply_purge(
                train_dates=pd.DatetimeIndex(train_dates),
                unique_dates=unique_dates,
                test_start=pd.Timestamp(test_dates[0]),
                lookahead=self.lookahead,
                embargo=self.embargo,
            )
            if len(train_dates) == 0:
                continue

            train_idx = np.where(mask_from_dates(dates, train_dates))[0]
            test_idx = np.where(mask_from_dates(dates, pd.DatetimeIndex(test_dates)))[0]
            if len(test_idx) == 0:
                continue
            yield train_idx, test_idx


class RollingKFoldPanelSplit(BasePanelSplit):
    """
    K-Fold splitter over time intervals.
    """

    def __init__(self, n_splits: int = 5, **kwargs):
        super().__init__(**kwargs)
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2.")
        self.n_splits = n_splits

    def split(self, X, y=None, groups=None):
        if groups is not None:
            raise ValueError("groups is not supported by this splitter.")
        _, dates, unique_dates = self._prepare_xy(X=X, y=y)
        intervals = np.array_split(unique_dates, self.n_splits)

        for i in range(self.n_splits):
            test_dates = pd.DatetimeIndex(intervals[i])
            if len(test_dates) == 0:
                continue

            train_dates = pd.DatetimeIndex(np.concatenate(intervals[:i]))
            train_dates = apply_purge(
                train_dates=train_dates,
                unique_dates=unique_dates,
                test_start=test_dates.min(),
                lookahead=self.lookahead,
                embargo=self.embargo,
            )
            if len(train_dates) == 0:
                continue

            train_idx = np.where(mask_from_dates(dates, train_dates))[0]
            test_idx = np.where(mask_from_dates(dates, test_dates))[0]
            if len(test_idx) == 0:
                continue
            yield train_idx, test_idx


class ExpandingIncrementPanelSplit(BasePanelSplit):
    """
    Walk-forward splitter with fixed train expansion intervals and fixed test size.
    """

    def __init__(
        self,
        train_intervals: int = 21,
        test_size: int = 21,
        min_train_periods: int = 252,
        max_train_periods: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if train_intervals < 1:
            raise ValueError("train_intervals must be >= 1.")
        if test_size < 1:
            raise ValueError("test_size must be >= 1.")
        if min_train_periods < 1:
            raise ValueError("min_train_periods must be >= 1.")
        if max_train_periods is not None and max_train_periods < 1:
            raise ValueError("max_train_periods must be >= 1 when provided.")

        self.train_intervals = train_intervals
        self.test_size = test_size
        self.min_train_periods = min_train_periods
        self.max_train_periods = max_train_periods

    def split(self, X, y=None, groups=None):
        if groups is not None:
            raise ValueError("groups is not supported by this splitter.")
        _, dates, unique_dates = self._prepare_xy(X=X, y=y)
        n_dates = len(unique_dates)

        train_end = self.min_train_periods
        while train_end + self.lookahead + self.embargo < n_dates:
            test_start = train_end + self.lookahead + self.embargo
            test_end = min(test_start + self.test_size, n_dates)
            if test_start >= test_end:
                break

            if self.max_train_periods is None:
                train_start = 0
            else:
                train_start = max(0, train_end - self.max_train_periods)

            train_dates = unique_dates[train_start:train_end]
            test_dates = unique_dates[test_start:test_end]
            if len(train_dates) == 0 or len(test_dates) == 0:
                break

            train_idx = np.where(mask_from_dates(dates, train_dates))[0]
            test_idx = np.where(mask_from_dates(dates, test_dates))[0]
            if len(test_idx) > 0:
                yield train_idx, test_idx

            train_end += self.train_intervals


class ExpandingFrequencyPanelSplit(BasePanelSplit):
    """
    Walk-forward splitter driven by calendar frequencies.
    """

    _FREQ_MAP = {
        "D": pd.DateOffset(days=1),
        "W": pd.DateOffset(weeks=1),
        "M": pd.DateOffset(months=1),
        "Q": pd.DateOffset(months=3),
        "Y": pd.DateOffset(years=1),
    }

    def __init__(
        self,
        expansion_freq: str = "M",
        test_freq: str = "M",
        min_train_periods: int = 252,
        max_train_periods: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if expansion_freq not in self._FREQ_MAP:
            raise ValueError("expansion_freq must be one of {'D','W','M','Q','Y'}.")
        if test_freq not in self._FREQ_MAP:
            raise ValueError("test_freq must be one of {'D','W','M','Q','Y'}.")
        if min_train_periods < 1:
            raise ValueError("min_train_periods must be >= 1.")
        if max_train_periods is not None and max_train_periods < 1:
            raise ValueError("max_train_periods must be >= 1 when provided.")

        self.expansion_freq = expansion_freq
        self.test_freq = test_freq
        self.min_train_periods = min_train_periods
        self.max_train_periods = max_train_periods

    def split(self, X, y=None, groups=None):
        if groups is not None:
            raise ValueError("groups is not supported by this splitter.")
        _, dates, unique_dates = self._prepare_xy(X=X, y=y)

        exp_offset = self._FREQ_MAP[self.expansion_freq]
        test_offset = self._FREQ_MAP[self.test_freq]

        train_end = self.min_train_periods
        n_dates = len(unique_dates)
        while train_end + self.lookahead + self.embargo < n_dates:
            if self.max_train_periods is None:
                train_start = 0
            else:
                train_start = max(0, train_end - self.max_train_periods)

            train_dates = unique_dates[train_start:train_end]
            if len(train_dates) == 0:
                break

            test_start_idx = train_end + self.lookahead + self.embargo
            if test_start_idx >= n_dates:
                break
            test_start_date = unique_dates[test_start_idx]
            test_end_date = test_start_date + test_offset
            test_dates = unique_dates[(unique_dates >= test_start_date) & (unique_dates < test_end_date)]
            if len(test_dates) == 0:
                test_dates = unique_dates[test_start_idx: test_start_idx + 1]
            if len(test_dates) == 0:
                break

            train_idx = np.where(mask_from_dates(dates, train_dates))[0]
            test_idx = np.where(mask_from_dates(dates, test_dates))[0]
            if len(test_idx) > 0:
                yield train_idx, test_idx

            train_last_date = unique_dates[train_end - 1]
            next_cutoff_date = train_last_date + exp_offset
            next_train_end = int(unique_dates.searchsorted(next_cutoff_date, side="right"))
            if next_train_end <= train_end:
                next_train_end = train_end + 1
            if next_train_end > n_dates:
                break
            train_end = next_train_end


__all__ = [
    "BasePanelSplit",
    "ExpandingKFoldPanelSplit",
    "RollingKFoldPanelSplit",
    "ExpandingIncrementPanelSplit",
    "ExpandingFrequencyPanelSplit",
]
