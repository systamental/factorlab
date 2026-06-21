from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from factorlab.core.pipeline import Pipeline
from factorlab.learning.splitters import BasePanelSplit
from factorlab.learning.utils import extract_dates
from factorlab.targets.forward import ForwardTargetSpec


@dataclass
class FoldRunInfo:
    fold: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    n_train: int
    n_test: int
    lookahead: int
    embargo: int
    horizon: int
    n_trainable: Optional[int] = None


class WalkForwardRunner:
    """
    Execute a Pipeline with explicit train/test fold boundaries.

    This runner is strategy-level orchestration: each fold is fitted on train rows and
    transformed up to the fold test end date, then only test rows are collected.
    """

    def __init__(
        self,
        pipeline: Pipeline,
        splitter: BasePanelSplit,
        y: Optional[Union[pd.Series, pd.DataFrame]] = None,
        target_spec: Optional[ForwardTargetSpec] = None,
        strict_temporal: bool = True,
        date_level: int = 0,
        show_progress: bool = False,
    ):
        if y is not None and target_spec is not None:
            raise ValueError("Provide either y or target_spec, not both.")

        self.pipeline = pipeline
        self.splitter = splitter
        self.y = y
        self.target_spec = target_spec
        self.strict_temporal = strict_temporal
        self.date_level = date_level
        self.show_progress = show_progress

        self.fold_info: list[FoldRunInfo] = []
        self.output: Optional[pd.DataFrame] = None

    @staticmethod
    def _clone_pipeline_steps(pipeline: Pipeline) -> list[tuple[str, object]]:
        return [(name, deepcopy(transformer)) for name, transformer in pipeline.steps]

    @staticmethod
    def _count_valid_targets(y: Union[pd.Series, pd.DataFrame]) -> int:
        if isinstance(y, pd.DataFrame):
            return int(y.notna().all(axis=1).sum())
        return int(y.notna().sum())

    def _resolve_fold_target(
        self,
        X_context: pd.DataFrame,
        train_index: pd.Index,
    ) -> Optional[Union[pd.Series, pd.DataFrame]]:
        if self.target_spec is not None:
            y_context = self.target_spec.build(X_context)
            trainable = self.target_spec.trainable_mask(
                index=X_context.index,
                train_index=train_index,
                date_level=self.date_level,
            )
            return y_context.where(trainable)

        if self.y is None:
            return None

        y_context = self.y.reindex(X_context.index)
        if isinstance(y_context, pd.DataFrame):
            if y_context.shape[1] != 1:
                raise ValueError("WalkForwardRunner currently supports a single y target column.")
        return y_context

    def _fit_transform_fold(
        self,
        X: pd.DataFrame,
        train_index: pd.Index,
        test_index: pd.Index,
    ) -> tuple[pd.DataFrame, Optional[int]]:
        dates = extract_dates(X.index, date_level=self.date_level)
        test_dates = pd.DatetimeIndex(test_index.get_level_values(self.date_level) if isinstance(test_index, pd.MultiIndex) else test_index)
        test_end = pd.Timestamp(test_dates.max())

        context_mask = dates <= test_end
        Xt = X.loc[context_mask].copy(deep=True)
        y_context = self._resolve_fold_target(X_context=Xt, train_index=train_index)
        n_trainable = None

        steps = self._clone_pipeline_steps(self.pipeline)
        for _, transformer in steps:
            train_mask = Xt.index.isin(train_index)
            X_train = Xt.loc[train_mask]

            if y_context is None:
                transformer.fit(X_train)
            else:
                y_train = y_context.reindex(X_train.index)
                n_trainable = self._count_valid_targets(y_train)
                transformer.fit(X_train, y_train)

            Xt = transformer.transform(Xt)
            if y_context is not None and not y_context.index.equals(Xt.index):
                y_context = y_context.reindex(Xt.index)

        return Xt.loc[test_index], n_trainable

    def run(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("WalkForwardRunner expects X as a pandas DataFrame.")
        if self.y is not None and not isinstance(self.y, (pd.Series, pd.DataFrame)):
            raise TypeError("y must be a pandas Series or DataFrame when provided.")
        if self.y is not None and not X.index.equals(self.y.index):
            raise ValueError("WalkForwardRunner requires y to share the exact same index as X.")
        if self.strict_temporal and self.target_spec is not None:
            if int(getattr(self.splitter, "lookahead", 0)) < int(self.target_spec.horizon):
                raise ValueError(
                    "Splitter lookahead is shorter than target_spec.horizon. "
                    "Increase lookahead or reduce horizon to avoid temporal leakage."
                )

        out = X.copy(deep=True)
        self.fold_info = []

        split_y = None if self.target_spec is not None else self.y
        splits = list(self.splitter.split(X, split_y))
        iterator = tqdm(enumerate(splits), total=len(splits), desc="WalkForward Runner", disable=not self.show_progress)

        for fold_id, (train_idx, test_idx) in iterator:
            if len(train_idx) == 0 or len(test_idx) == 0:
                continue

            train_index = X.iloc[train_idx].index
            test_index = X.iloc[test_idx].index

            train_dates = extract_dates(train_index, date_level=self.date_level)
            test_dates = extract_dates(test_index, date_level=self.date_level)
            if pd.Timestamp(train_dates.max()) >= pd.Timestamp(test_dates.min()):
                raise ValueError(
                    "Temporal integrity violation: train_end must be strictly earlier than test_start."
                )

            fold_out, n_trainable = self._fit_transform_fold(X=X, train_index=train_index, test_index=test_index)

            for col in fold_out.columns:
                if col not in out.columns:
                    out[col] = np.nan
            out.loc[test_index, fold_out.columns] = fold_out

            self.fold_info.append(
                FoldRunInfo(
                    fold=fold_id,
                    train_start=pd.Timestamp(train_dates.min()),
                    train_end=pd.Timestamp(train_dates.max()),
                    test_start=pd.Timestamp(test_dates.min()),
                    test_end=pd.Timestamp(test_dates.max()),
                    n_train=int(len(train_idx)),
                    n_test=int(len(test_idx)),
                    lookahead=int(getattr(self.splitter, "lookahead", 0)),
                    embargo=int(getattr(self.splitter, "embargo", 0)),
                    horizon=int(self.target_spec.horizon if self.target_spec is not None else 0),
                    n_trainable=n_trainable,
                )
            )

        self.output = out
        return out
