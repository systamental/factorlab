from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.base import clone
from tqdm import tqdm

from factorlab.core.base_transform import BaseTransform
from factorlab.learning.utils import extract_dates, unique_sorted_dates
from factorlab.targets.base import Target
from factorlab.core.utils.utils import to_dataframe


@dataclass
class FoldInfo:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    n_train: int
    n_test: int


class WalkForwardLearner(BaseTransform):
    """
    Supervised forecasting transform that performs date-based walk-forward fitting.

    This class is designed to be inserted directly into ``Pipeline`` after factor
    construction and before signal generation.
    """

    def __init__(
        self,
        model: Any,
        feature_cols: Sequence[str],
        target_transform: Optional[Target] = None,
        target_col: Optional[str] = None,
        prediction_col: str = "forecast",
        window_type: str = "expanding",
        min_train_periods: int = 60,
        train_periods: Optional[int] = None,
        retrain_interval: int = 1,
        min_train_samples: int = 100,
        prediction_method: str = "predict",
        proba_index: int = 1,
        label_lookahead: Optional[int] = None,
        date_level: int = 0,
        show_progress: bool = False,
    ):
        super().__init__(
            name=f"WalkForwardLearner({model.__class__.__name__})",
            description="Date-based walk-forward supervised learning step.",
        )

        if not feature_cols:
            raise ValueError("feature_cols cannot be empty.")
        if target_transform is None and target_col is None:
            raise ValueError("Provide either target_transform or target_col.")
        if window_type not in {"expanding", "rolling"}:
            raise ValueError("window_type must be one of {'expanding', 'rolling'}.")
        if min_train_periods < 1:
            raise ValueError("min_train_periods must be >= 1.")
        if retrain_interval < 1:
            raise ValueError("retrain_interval must be >= 1.")
        if min_train_samples < 1:
            raise ValueError("min_train_samples must be >= 1.")
        if window_type == "rolling" and (train_periods is None or train_periods < 1):
            raise ValueError("train_periods must be >= 1 when window_type='rolling'.")

        self.model = model
        self.feature_cols = list(feature_cols)
        self.target_transform = target_transform
        self.target_col = target_col
        self.prediction_col = prediction_col
        self.window_type = window_type
        self.min_train_periods = min_train_periods
        self.train_periods = train_periods
        self.retrain_interval = retrain_interval
        self.min_train_samples = min_train_samples
        self.prediction_method = prediction_method
        self.proba_index = proba_index
        self.label_lookahead = label_lookahead
        self.date_level = date_level
        self.show_progress = show_progress

        self.fold_info: List[FoldInfo] = []

    @property
    def inputs(self) -> List[str]:
        required = set(self.feature_cols)
        if self.target_transform is not None:
            required.update(self.target_transform.inputs)
        elif self.target_col is not None:
            required.add(self.target_col)
        return sorted(required)

    @staticmethod
    def _safe_clone(model: Any) -> Any:
        try:
            return clone(model)
        except Exception:
            return deepcopy(model)

    def _resolve_target_column(self, df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
        if self.target_transform is not None:
            self.target_transform.fit(df)
            out = self.target_transform.transform(df)
            if not hasattr(self.target_transform, "output_col"):
                raise ValueError("target_transform must expose an 'output_col' attribute.")
            return out, self.target_transform.output_col

        if self.target_col is None:
            raise ValueError("target_col is required when target_transform is not provided.")
        if self.target_col not in df.columns:
            raise ValueError(f"target_col '{self.target_col}' is missing from input data.")
        return df, self.target_col

    def _predict(self, model: Any, X: pd.DataFrame) -> np.ndarray:
        if self.prediction_method == "predict_proba":
            if not hasattr(model, "predict_proba"):
                raise AttributeError(f"{model.__class__.__name__} does not implement predict_proba().")
            proba = np.asarray(model.predict_proba(X))
            if proba.ndim != 2:
                raise ValueError("predict_proba output must be 2-dimensional.")
            if self.proba_index >= proba.shape[1]:
                raise ValueError(
                    f"proba_index {self.proba_index} is out of bounds for predict_proba output with "
                    f"{proba.shape[1]} columns."
                )
            return proba[:, self.proba_index]

        if not hasattr(model, self.prediction_method):
            raise AttributeError(f"{model.__class__.__name__} does not implement {self.prediction_method}().")

        pred = np.asarray(getattr(model, self.prediction_method)(X))
        if pred.ndim == 1:
            return pred
        if pred.ndim == 2 and pred.shape[1] == 1:
            return pred.reshape(-1)
        raise ValueError("Prediction output must be 1-dimensional for this forecaster.")

    def _resolve_lookahead(self) -> int:
        if self.label_lookahead is not None:
            if self.label_lookahead < 0:
                raise ValueError("label_lookahead must be >= 0.")
            return self.label_lookahead

        if self.target_transform is not None and hasattr(self.target_transform, "horizon"):
            return int(getattr(self.target_transform, "horizon"))

        return 0

    def fit(
        self,
        X: Union[pd.Series, pd.DataFrame],
        y: Optional[Union[pd.Series, pd.DataFrame]] = None,
    ) -> "WalkForwardLearner":
        df = to_dataframe(X).copy(deep=True)
        self.validate_inputs(df)
        self._resolve_target_column(df)
        self._is_fitted = True
        return self

    def transform(self, X: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
        if not self._is_fitted:
            raise RuntimeError(f"Transform '{self.name}' must be fitted before calling transform().")

        df = to_dataframe(X).copy(deep=True)
        self.validate_inputs(df)
        df_target, target_col = self._resolve_target_column(df)

        dates = extract_dates(df.index, date_level=self.date_level)
        unique_dates = unique_sorted_dates(df.index, date_level=self.date_level)

        lookahead = self._resolve_lookahead()
        first_pred_idx = self.min_train_periods + lookahead
        if first_pred_idx >= len(unique_dates):
            raise ValueError(
                f"Not enough dates for walk-forward forecasting: need more than "
                f"{first_pred_idx} unique dates, got {len(unique_dates)}."
            )

        out = df.copy(deep=True)
        out[self.prediction_col] = np.nan
        self.fold_info = []

        split_starts = range(first_pred_idx, len(unique_dates), self.retrain_interval)
        iterator = tqdm(split_starts, desc="WalkForward Learning", disable=not self.show_progress)

        for pred_idx in iterator:
            pred_dates = unique_dates[pred_idx: pred_idx + self.retrain_interval]
            if pred_dates.empty:
                continue

            train_end_idx = pred_idx - lookahead
            if train_end_idx <= 0:
                continue

            if self.window_type == "expanding":
                train_start_idx = 0
            else:
                train_start_idx = max(0, train_end_idx - int(self.train_periods))

            train_dates = unique_dates[train_start_idx:train_end_idx]
            if train_dates.empty:
                continue

            train_mask = dates.isin(train_dates)
            test_mask = dates.isin(pred_dates)

            train_df = df_target.loc[train_mask, self.feature_cols + [target_col]]
            train_valid = train_df[self.feature_cols].notna().all(axis=1) & train_df[target_col].notna()
            if int(train_valid.sum()) < self.min_train_samples:
                continue

            X_train = train_df.loc[train_valid, self.feature_cols]
            y_train = train_df.loc[train_valid, target_col]

            model = self._safe_clone(self.model)
            model.fit(X_train, y_train)

            test_df = out.loc[test_mask, self.feature_cols]
            test_valid = test_df.notna().all(axis=1)
            if int(test_valid.sum()) > 0:
                preds = self._predict(model, test_df.loc[test_valid, self.feature_cols])
                out.loc[test_df.loc[test_valid].index, self.prediction_col] = preds

            self.fold_info.append(
                FoldInfo(
                    train_start=train_dates.min(),
                    train_end=train_dates.max(),
                    test_start=pred_dates.min(),
                    test_end=pred_dates.max(),
                    n_train=int(train_valid.sum()),
                    n_test=int(test_valid.sum()),
                )
            )

        return out
