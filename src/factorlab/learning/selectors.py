from __future__ import annotations

from copy import deepcopy
from itertools import combinations
from typing import Any, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, contingency
from sklearn.base import clone
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression

from factorlab.core.base_transform import BaseTransform
from factorlab.learning.model_registry import build_regressor, extract_feature_importance
from factorlab.learning.utils import resolve_feature_columns
from factorlab.core.utils.utils import to_dataframe


class FeatureSelector(BaseTransform):
    """
    Pipeline-native feature selection transform.

    Fit learns ranked/selected features from a training fold; transform can optionally
    drop unselected features from the output DataFrame.
    """

    _METHOD_ALIASES = {
        "spearman_rank": "spearman",
        "kendall_tau": "kendall",
        "mutual_info": "mutual_info_regression",
    }
    _METHODS = {
        "spearman",
        "kendall",
        "cramer_v",
        "tschuprow",
        "pearson_cc",
        "chi2",
        "mutual_info_regression",
        "mutual_info_classification",
        "variance",
        "model_importance",
        "lars",
        "lasso",
        "ridge",
        "elastic_net",
        "random_forest",
        "xgboost",
        "catboost",
        "mrmr",
        "mifs",
        "mrmr_mifs",
        "spearman_mrmr",
        "rfe",
        "stepwise",
        "forward",
        "backward",
        "exhaustive",
    }

    def __init__(
        self,
        method: str = "spearman",
        feature_cols: Optional[Sequence[str]] = None,
        target_col: Optional[str] = None,
        n_features: Optional[int] = None,
        score_threshold: Optional[float] = None,
        model: Optional[Any] = None,
        drop_unselected: bool = True,
        exclude_cols: Optional[Sequence[str]] = None,
        absolute_scores: bool = True,
        feature_bins: int = 5,
        target_bins: int = 3,
        mifs_beta: float = 0.5,
        random_state: int = 42,
        method_kwargs: Optional[dict[str, Any]] = None,
    ):
        canonical_method = self._METHOD_ALIASES.get(method, method)
        super().__init__(
            name=f"FeatureSelector({canonical_method})",
            description="Select top features using fold-local training data.",
        )
        if canonical_method not in self._METHODS:
            raise ValueError(f"method must be one of {sorted(self._METHODS)}.")
        if n_features is not None and n_features < 1:
            raise ValueError("n_features must be >= 1 when provided.")
        if feature_bins < 2:
            raise ValueError("feature_bins must be >= 2.")
        if target_bins < 2:
            raise ValueError("target_bins must be >= 2.")
        if canonical_method == "model_importance" and model is None:
            raise ValueError("model must be provided when method='model_importance'.")

        self.method = canonical_method
        self.feature_cols = list(feature_cols) if feature_cols is not None else None
        self.target_col = target_col
        self.n_features = n_features
        self.score_threshold = score_threshold
        self.model = model
        self.drop_unselected = drop_unselected
        self.exclude_cols = list(exclude_cols) if exclude_cols is not None else []
        self.absolute_scores = absolute_scores
        self.feature_bins = feature_bins
        self.target_bins = target_bins
        self.mifs_beta = float(mifs_beta)
        self.random_state = int(random_state)
        self.method_kwargs = dict(method_kwargs or {})

        self.resolved_feature_cols_: list[str] = []
        self.selected_features_: list[str] = []
        self.feature_scores_: pd.DataFrame = pd.DataFrame()

    @property
    def inputs(self) -> List[str]:
        required = list(self.feature_cols) if self.feature_cols is not None else []
        if self.target_col is not None:
            required.append(self.target_col)
        return required

    @staticmethod
    def _safe_clone(model: Any) -> Any:
        try:
            return clone(model)
        except Exception:
            return deepcopy(model)

    def _resolve_target(
        self,
        X_df: pd.DataFrame,
        y: Optional[Union[pd.Series, pd.DataFrame]],
    ) -> pd.Series:
        if y is not None:
            y_df = to_dataframe(y).copy(deep=True)
            if y_df.shape[1] != 1:
                raise ValueError("y must contain exactly one target column.")
            y_series = y_df.iloc[:, 0].reindex(X_df.index)
            return y_series

        if self.target_col is None:
            raise ValueError("target_col must be provided when y is not supplied.")
        if self.target_col not in X_df.columns:
            raise ValueError(f"target_col '{self.target_col}' not found in input columns.")
        return X_df[self.target_col]

    @staticmethod
    def _quantize_series(series: pd.Series, bins: int) -> pd.Series:
        valid = series.dropna()
        if valid.nunique() <= 1:
            return pd.Series(np.nan, index=series.index)
        q = min(int(bins), int(valid.nunique()))
        ranked = valid.rank(method="first")
        quant = pd.qcut(ranked, q=q, labels=False, duplicates="drop")
        out = pd.Series(np.nan, index=series.index, dtype=float)
        out.loc[valid.index] = quant.astype(float)
        return out

    def _categorical_association_scores(self, X_feat: pd.DataFrame, y: pd.Series) -> pd.Series:
        yq = self._quantize_series(y, bins=self.target_bins)
        scores: dict[str, float] = {}
        for col in X_feat.columns:
            xq = self._quantize_series(X_feat[col], bins=self.feature_bins)
            valid = xq.notna() & yq.notna()
            if int(valid.sum()) == 0:
                scores[col] = np.nan
                continue
            table = pd.crosstab(xq.loc[valid], yq.loc[valid])
            if table.empty:
                scores[col] = np.nan
                continue
            if self.method == "cramer_v":
                scores[col] = float(contingency.association(table, method="cramer"))
            elif self.method == "tschuprow":
                scores[col] = float(contingency.association(table, method="tschuprow"))
            elif self.method == "pearson_cc":
                scores[col] = float(contingency.association(table, method="pearson"))
            elif self.method == "chi2":
                scores[col] = float(chi2_contingency(table)[0])
            else:
                raise ValueError(f"Unsupported association method '{self.method}'.")
        return pd.Series(scores, dtype=float)

    def _build_model_for_method(self, X_feat: pd.DataFrame) -> Any:
        if self.method in {
            "model_importance",
            "lars",
            "lasso",
            "ridge",
            "elastic_net",
            "random_forest",
            "xgboost",
            "catboost",
        }:
            custom_model = self.model if self.method == "model_importance" else None
            method = "linear_regression" if self.method == "model_importance" else self.method
            return build_regressor(
                method=method,
                n_features=int(self.n_features or X_feat.shape[1]),
                random_state=self.random_state,
                custom_model=custom_model,
                method_kwargs=self.method_kwargs,
            )
        return None

    @staticmethod
    def _safe_abs_spearman_corr(a: pd.Series, b: pd.Series) -> float:
        corr = a.corr(b, method="spearman")
        if pd.isna(corr):
            return 0.0
        return float(abs(corr))

    def _greedy_redundancy_selection(
        self,
        X_feat: pd.DataFrame,
        y: pd.Series,
        mode: str,
    ) -> pd.Series:
        if self.n_features is None:
            top_k = min(10, X_feat.shape[1])
        else:
            top_k = min(int(self.n_features), int(X_feat.shape[1]))
        if top_k < 1:
            raise ValueError("n_features must resolve to >= 1 for redundancy-aware methods.")

        if mode == "spearman_mrmr":
            relevance = X_feat.apply(lambda col: self._safe_abs_spearman_corr(col, y))
            redundancy = X_feat.corr(method="spearman").abs().fillna(0.0)
        else:
            valid = X_feat.notna().all(axis=1) & y.notna()
            if int(valid.sum()) == 0:
                raise ValueError("No valid non-NaN rows available for relevance scoring.")
            Xv = X_feat.loc[valid]
            yv = y.loc[valid]
            mi = mutual_info_regression(Xv, yv, random_state=self.random_state)
            relevance = pd.Series(mi, index=X_feat.columns).fillna(0.0)
            redundancy = X_feat.corr(method="pearson").abs().fillna(0.0)

        remaining = list(X_feat.columns)
        selected: list[str] = []
        scores: dict[str, float] = {}

        while remaining and len(selected) < top_k:
            if not selected:
                best = max(remaining, key=lambda c: float(relevance.get(c, 0.0)))
                score = float(relevance.get(best, 0.0))
            else:
                candidate_scores: dict[str, float] = {}
                for col in remaining:
                    rel = float(relevance.get(col, 0.0))
                    red_vals = [float(redundancy.loc[col, s]) for s in selected if s in redundancy.columns]
                    red_mean = float(np.mean(red_vals)) if red_vals else 0.0
                    red_sum = float(np.sum(red_vals)) if red_vals else 0.0

                    if mode in {"mrmr", "spearman_mrmr"}:
                        candidate_scores[col] = rel - red_mean
                    elif mode == "mifs":
                        candidate_scores[col] = rel - self.mifs_beta * red_sum
                    elif mode == "mrmr_mifs":
                        candidate_scores[col] = rel - 0.5 * (red_mean + self.mifs_beta * red_sum)
                    else:
                        raise ValueError(f"Unknown redundancy mode '{mode}'.")
                best = max(candidate_scores, key=candidate_scores.get)
                score = float(candidate_scores[best])

            selected.append(best)
            remaining.remove(best)
            scores[best] = score

        return pd.Series(scores, dtype=float).sort_values(ascending=False)

    def _score_features(
        self,
        X_feat: pd.DataFrame,
        y: Optional[pd.Series],
    ) -> pd.Series:
        if self.method == "variance":
            scores = X_feat.var(ddof=0)
            return scores.fillna(0.0)

        if y is None:
            raise ValueError(f"Method '{self.method}' requires a target series.")

        if self.method in {"spearman", "kendall"}:
            corr_method = "spearman" if self.method == "spearman" else "kendall"
            scores = X_feat.apply(lambda col: col.corr(y, method=corr_method))
            return scores.fillna(0.0)

        if self.method in {"cramer_v", "tschuprow", "pearson_cc", "chi2"}:
            return self._categorical_association_scores(X_feat=X_feat, y=y).fillna(0.0)

        if self.method in {"mrmr", "mifs", "mrmr_mifs", "spearman_mrmr"}:
            return self._greedy_redundancy_selection(X_feat=X_feat, y=y, mode=self.method).fillna(0.0)

        valid = X_feat.notna().all(axis=1) & y.notna()
        if int(valid.sum()) == 0:
            raise ValueError("No valid non-NaN rows available for feature scoring.")

        Xv = X_feat.loc[valid]
        yv = y.loc[valid]

        if self.method == "mutual_info_regression":
            vals = mutual_info_regression(Xv, yv, random_state=self.random_state)
            return pd.Series(vals, index=X_feat.columns, dtype=float)

        if self.method == "mutual_info_classification":
            y_codes = pd.Series(yv, index=Xv.index).astype("category").cat.codes.to_numpy()
            vals = mutual_info_classif(Xv, y_codes, random_state=self.random_state)
            return pd.Series(vals, index=X_feat.columns, dtype=float)

        if self.method == "rfe":
            estimator = self._safe_clone(self.model if self.model is not None else LinearRegression())
            n_select = self.n_features if self.n_features is not None else max(1, X_feat.shape[1] // 2)
            step = self.method_kwargs.get("step", 1)
            selector = RFE(estimator=estimator, n_features_to_select=int(n_select), step=step)
            selector.fit(Xv, yv)
            ranking = pd.Series(selector.ranking_, index=X_feat.columns, dtype=float)
            # Lower ranking is better; convert to descending score where selected features receive highest values.
            return (ranking.max() + 1 - ranking).astype(float)

        if self.method in {"forward", "backward", "stepwise"}:
            estimator = self._safe_clone(self.model if self.model is not None else LinearRegression())
            n_select = self.n_features if self.n_features is not None else max(1, X_feat.shape[1] // 2)
            direction = "backward" if self.method == "backward" else "forward"
            selector = SequentialFeatureSelector(
                estimator=estimator,
                n_features_to_select=int(n_select),
                direction=direction,
            )
            selector.fit(Xv, yv)
            support = pd.Series(selector.get_support(), index=X_feat.columns)
            return support.astype(float)

        if self.method == "exhaustive":
            estimator = self._safe_clone(self.model if self.model is not None else LinearRegression())
            n_select = self.n_features if self.n_features is not None else max(1, X_feat.shape[1] // 2)
            n_select = min(int(n_select), int(X_feat.shape[1]))
            best_combo: Optional[tuple[str, ...]] = None
            best_score = -np.inf
            for combo in combinations(list(X_feat.columns), n_select):
                estimator_i = self._safe_clone(estimator)
                estimator_i.fit(Xv.loc[:, combo], yv)
                score = float(estimator_i.score(Xv.loc[:, combo], yv))
                if score > best_score:
                    best_score = score
                    best_combo = combo
            if best_combo is None:
                raise ValueError("Exhaustive selection could not identify a valid feature subset.")
            support = pd.Series(0.0, index=X_feat.columns, dtype=float)
            support.loc[list(best_combo)] = 1.0
            return support

        model = self._build_model_for_method(X_feat=X_feat)
        if model is None:
            raise ValueError(f"Unsupported feature-selection method '{self.method}'.")
        model.fit(Xv, yv)
        return extract_feature_importance(model, list(X_feat.columns))["feature_importance"]

    def fit(
        self,
        X: Union[pd.Series, pd.DataFrame],
        y: Optional[Union[pd.Series, pd.DataFrame]] = None,
    ) -> "FeatureSelector":
        df = to_dataframe(X).copy(deep=True)
        self.resolved_feature_cols_ = resolve_feature_columns(
            df=df,
            feature_cols=self.feature_cols,
            exclude_cols=self.exclude_cols + [self.target_col],
            require_numeric=True,
        )
        X_feat = df[self.resolved_feature_cols_]

        target = None if self.method == "variance" else self._resolve_target(df, y=y)
        scores = self._score_features(X_feat, y=target)
        if self.absolute_scores:
            scores = scores.abs()
        scores = scores.sort_values(ascending=False)

        if self.score_threshold is not None:
            scores = scores.loc[scores >= float(self.score_threshold)]
        if self.n_features is not None:
            scores = scores.head(int(self.n_features))
        if scores.empty:
            raise ValueError("Feature selection produced zero selected features.")

        self.selected_features_ = list(scores.index)
        self.feature_scores_ = scores.to_frame(name="score")
        self._is_fitted = True
        return self

    def transform(self, X: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
        if not self._is_fitted:
            raise RuntimeError(f"Transform '{self.name}' must be fitted before calling transform().")

        df = to_dataframe(X).copy(deep=True)
        missing = set(self.selected_features_) - set(df.columns)
        if missing:
            raise ValueError(f"Input is missing selected features required by transform: {missing}")

        if not self.drop_unselected:
            return df

        drop_cols = [c for c in self.resolved_feature_cols_ if c not in self.selected_features_]
        if drop_cols:
            df = df.drop(columns=drop_cols, errors="ignore")
        return df
