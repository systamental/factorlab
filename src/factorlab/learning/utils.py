from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd


def extract_dates(index: pd.Index, date_level: int = 0) -> pd.DatetimeIndex:
    """
    Extract a DatetimeIndex from an index or MultiIndex.
    """
    if isinstance(index, pd.MultiIndex):
        dates = index.get_level_values(date_level)
    else:
        dates = index

    if not pd.api.types.is_datetime64_any_dtype(dates):
        dates = pd.to_datetime(dates)

    return pd.DatetimeIndex(dates)


def unique_sorted_dates(index: pd.Index, date_level: int = 0) -> pd.DatetimeIndex:
    """
    Return unique sorted dates from an index or MultiIndex.
    """
    return pd.DatetimeIndex(pd.Index(extract_dates(index, date_level=date_level)).unique()).sort_values()


def validate_panel_xy(
    X: pd.DataFrame,
    y: Optional[Union[pd.Series, pd.DataFrame]] = None,
    date_level: int = 0,
    require_multiindex: bool = True,
) -> None:
    """
    Validate panel-style X/y inputs.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame.")

    if require_multiindex and not isinstance(X.index, pd.MultiIndex):
        raise ValueError("X must use a MultiIndex with a datetime date level.")

    _ = extract_dates(X.index, date_level=date_level)
    if not X.index.is_monotonic_increasing:
        raise ValueError("X index must be sorted in increasing order before splitting.")

    if X.index.has_duplicates:
        raise ValueError("X index contains duplicates. Remove duplicates before splitting.")

    if y is None:
        return

    if not isinstance(y, (pd.Series, pd.DataFrame)):
        raise TypeError("y must be a pandas Series or DataFrame.")
    if not X.index.equals(y.index):
        raise ValueError("X and y must share the exact same index.")
    if not y.index.is_monotonic_increasing:
        raise ValueError("y index must be sorted in increasing order before splitting.")
    if y.index.has_duplicates:
        raise ValueError("y index contains duplicates. Remove duplicates before splitting.")
    _ = extract_dates(y.index, date_level=date_level)


def apply_purge(
    train_dates: pd.DatetimeIndex,
    unique_dates: pd.DatetimeIndex,
    test_start: pd.Timestamp,
    lookahead: int = 0,
    embargo: int = 0,
) -> pd.DatetimeIndex:
    """
    Purge training dates whose labels can overlap with the test window.

    The cutoff rule is:
    train_date_position < test_start_position - lookahead - embargo
    """
    if lookahead < 0:
        raise ValueError("lookahead must be >= 0.")
    if embargo < 0:
        raise ValueError("embargo must be >= 0.")

    if len(train_dates) == 0:
        return train_dates

    test_start_pos = int(unique_dates.get_loc(test_start))
    cutoff = test_start_pos - int(lookahead) - int(embargo)
    if cutoff <= 0:
        return train_dates[:0]

    allowed = set(unique_dates[:cutoff])
    return pd.DatetimeIndex([d for d in train_dates if d in allowed])


def mask_from_dates(dates: pd.DatetimeIndex, selected_dates: pd.DatetimeIndex) -> np.ndarray:
    """
    Build a boolean mask selecting rows whose dates are in selected_dates.
    """
    return dates.isin(selected_dates)


def resolve_feature_columns(
    df: pd.DataFrame,
    feature_cols: Optional[Sequence[str]] = None,
    exclude_cols: Optional[Sequence[str]] = None,
    require_numeric: bool = True,
) -> list[str]:
    """
    Resolve feature columns with consistent validation rules.
    """
    if feature_cols is None:
        if require_numeric:
            cols = list(df.select_dtypes(include=[np.number]).columns)
        else:
            cols = list(df.columns)
    else:
        cols = list(feature_cols)
        missing = set(cols) - set(df.columns)
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")

    exclude = {c for c in (exclude_cols or []) if c is not None}
    cols = [c for c in cols if c not in exclude]
    if len(cols) == 0:
        raise ValueError("No valid feature columns resolved after exclusions.")
    return cols
