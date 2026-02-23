from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Union

import numpy as np
import pandas as pd

from factorlab.learning.utils import extract_dates
from factorlab.targets.base import Target
from factorlab.core.utils.utils import to_dataframe


class ForwardReturnTarget(Target):
    """
    Build a forward return target column from a price column.

    This target is designed for supervised forecasting in walk-forward workflows.
    """

    def __init__(
        self,
        input_col: str = "close",
        output_col: str = "target",
        horizon: int = 1,
        method: str = "pct",
        group_level: int = 1,
    ):
        super().__init__(
            name="ForwardReturnTarget",
            description="Forward return target from a price column.",
        )
        if horizon < 1:
            raise ValueError("horizon must be >= 1.")
        if method not in {"pct", "log", "diff"}:
            raise ValueError("method must be one of {'pct', 'log', 'diff'}.")

        self.input_col = input_col
        self.output_col = output_col
        self.horizon = int(horizon)
        self.method = method
        self.group_level = group_level

    @property
    def inputs(self) -> List[str]:
        return [self.input_col]

    def fit(
        self,
        X: Union[pd.Series, pd.DataFrame],
        y: Optional[Union[pd.Series, pd.DataFrame]] = None,
    ) -> "ForwardReturnTarget":
        df = to_dataframe(X)
        self.validate_inputs(df)
        self._is_fitted = True
        return self

    def _forward_price(self, df: pd.DataFrame) -> pd.Series:
        price = df[self.input_col]
        if isinstance(df.index, pd.MultiIndex):
            return price.groupby(level=self.group_level, observed=True).shift(-self.horizon)
        return price.shift(-self.horizon)

    def _compute_target(self, df: pd.DataFrame) -> pd.Series:
        price = df[self.input_col]
        fwd_price = self._forward_price(df)

        if self.method == "pct":
            target = fwd_price.div(price) - 1.0
        elif self.method == "log":
            ratio = fwd_price.div(price)
            target = np.log(ratio.where(ratio > 0))
        else:
            target = fwd_price - price

        return target.replace([np.inf, -np.inf], np.nan)

    def transform(self, X: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
        if not self._is_fitted:
            raise RuntimeError(f"Transform '{self.name}' must be fitted before calling transform().")

        df = to_dataframe(X).copy(deep=True)
        self.validate_inputs(df)
        df[self.output_col] = self._compute_target(df)
        return df


class ForwardDirectionTarget(ForwardReturnTarget):
    """
    Build a binary forward-direction target from a forward return.
    """

    def __init__(
        self,
        input_col: str = "close",
        output_col: str = "target",
        horizon: int = 1,
        threshold: float = 0.0,
        group_level: int = 1,
    ):
        super().__init__(
            input_col=input_col,
            output_col=output_col,
            horizon=horizon,
            method="pct",
            group_level=group_level,
        )
        self.name = "ForwardDirectionTarget"
        self.description = "Binary target from forward returns."
        self.threshold = float(threshold)

    def _compute_target(self, df: pd.DataFrame) -> pd.Series:
        fwd_ret = super()._compute_target(df)
        return (fwd_ret > self.threshold).astype(float)


@dataclass(frozen=True)
class ForwardTargetSpec:
    """
    Declarative target specification for fold-local walk-forward training.

    The spec can build a target series on demand and compute the rows that are
    safe to use for training at a given fold boundary.
    """

    input_col: str = "close"
    output_col: str = "target"
    horizon: int = 1
    kind: str = "return"
    method: str = "pct"
    threshold: float = 0.0
    group_level: int = 1
    clip_lower: Optional[float] = None
    clip_upper: Optional[float] = None
    label_func: Optional[Callable[[pd.Series], pd.Series]] = None

    def __post_init__(self) -> None:
        if self.horizon < 1:
            raise ValueError("ForwardTargetSpec.horizon must be >= 1.")
        if self.kind not in {"return", "direction"}:
            raise ValueError("ForwardTargetSpec.kind must be one of {'return', 'direction'}.")
        if self.kind == "return" and self.method not in {"pct", "log", "diff"}:
            raise ValueError("ForwardTargetSpec.method must be one of {'pct', 'log', 'diff'}.")
        if self.clip_lower is not None and self.clip_upper is not None and self.clip_lower > self.clip_upper:
            raise ValueError("clip_lower must be <= clip_upper when both are set.")

    def _build_transform(self) -> Target:
        if self.kind == "direction":
            return ForwardDirectionTarget(
                input_col=self.input_col,
                output_col=self.output_col,
                horizon=self.horizon,
                threshold=self.threshold,
                group_level=self.group_level,
            )
        return ForwardReturnTarget(
            input_col=self.input_col,
            output_col=self.output_col,
            horizon=self.horizon,
            method=self.method,
            group_level=self.group_level,
        )

    def build(self, X: Union[pd.Series, pd.DataFrame]) -> pd.Series:
        """
        Build the target series for the provided data.
        """
        df = to_dataframe(X).copy(deep=True)
        transform = self._build_transform()
        transform.fit(df)
        out = transform.transform(df)[self.output_col]

        if self.clip_lower is not None or self.clip_upper is not None:
            out = out.clip(lower=self.clip_lower, upper=self.clip_upper)
        if self.label_func is not None:
            out = self.label_func(out)

        return out.rename(self.output_col)

    def label_end_dates(self, index: pd.Index, date_level: int = 0) -> pd.Series:
        """
        Return the timestamp used by each row's forward label.
        """
        if isinstance(index, pd.MultiIndex):
            row_dates = pd.Series(extract_dates(index, date_level=date_level), index=index)
            return row_dates.groupby(level=self.group_level, observed=True).shift(-self.horizon)

        row_dates = pd.Series(extract_dates(index, date_level=date_level), index=index)
        return row_dates.shift(-self.horizon)

    def trainable_mask(
        self,
        index: pd.Index,
        train_index: pd.Index,
        date_level: int = 0,
    ) -> np.ndarray:
        """
        Return a boolean mask for rows safe to train on at this fold.

        A row is trainable if:
        - It belongs to the fold's train_index.
        - Its forward label end-date is defined.
        - Its forward label end-date is not after the fold train_end date.
        """
        if len(train_index) == 0:
            return np.zeros(len(index), dtype=bool)

        train_dates = extract_dates(train_index, date_level=date_level)
        train_end = pd.Timestamp(train_dates.max())
        label_end = self.label_end_dates(index=index, date_level=date_level)

        return (
            pd.Index(index).isin(train_index)
            & label_end.notna().to_numpy()
            & (pd.DatetimeIndex(label_end.values) <= train_end)
        )
