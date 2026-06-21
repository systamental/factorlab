from __future__ import annotations

from copy import deepcopy
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import (
    ElasticNet,
    ElasticNetCV,
    Lars,
    LarsCV,
    Lasso,
    LassoCV,
    LassoLars,
    LassoLarsCV,
    LassoLarsIC,
    LinearRegression,
    LogisticRegression,
    LogisticRegressionCV,
    Ridge,
    RidgeCV,
)


def _safe_clone(model: Any) -> Any:
    try:
        return clone(model)
    except Exception:
        return deepcopy(model)


def build_regressor(
    method: str,
    n_features: Optional[int] = None,
    random_state: int = 42,
    custom_model: Optional[Any] = None,
    method_kwargs: Optional[dict[str, Any]] = None,
) -> Any:
    """
    Build a regressor by canonical method string.
    """
    kwargs = dict(method_kwargs or {})
    method = method.lower()

    if custom_model is not None:
        return _safe_clone(custom_model)
    if method in {"ols", "linear_regression"}:
        return LinearRegression(**kwargs)
    if method == "lasso":
        return Lasso(**kwargs)
    if method == "lasso_cv":
        return LassoCV(**kwargs)
    if method == "lasso_lars":
        return LassoLars(**kwargs)
    if method == "lasso_lars_cv":
        return LassoLarsCV(**kwargs)
    if method == "lasso_lars_ic":
        return LassoLarsIC(**kwargs)
    if method == "lars":
        if n_features is not None and "n_nonzero_coefs" not in kwargs:
            kwargs["n_nonzero_coefs"] = int(n_features)
        return Lars(**kwargs)
    if method == "lars_cv":
        return LarsCV(**kwargs)
    if method == "ridge":
        return Ridge(**kwargs)
    if method == "ridge_cv":
        return RidgeCV(**kwargs)
    if method == "elastic_net":
        return ElasticNet(**kwargs)
    if method == "elastic_net_cv":
        return ElasticNetCV(**kwargs)
    if method == "random_forest":
        kwargs.setdefault("random_state", random_state)
        return RandomForestRegressor(**kwargs)
    if method == "xgboost":
        try:
            from xgboost import XGBRegressor
        except Exception as exc:
            raise ImportError("xgboost is required for method='xgboost'.") from exc
        kwargs.setdefault("random_state", random_state)
        return XGBRegressor(**kwargs)
    if method == "catboost":
        try:
            from catboost import CatBoostRegressor
        except Exception as exc:
            raise ImportError("catboost is required for method='catboost'.") from exc
        kwargs.setdefault("random_seed", random_state)
        kwargs.setdefault("verbose", False)
        return CatBoostRegressor(**kwargs)

    raise ValueError(f"Unknown regression method '{method}'.")


def build_classifier(
    method: str,
    random_state: int = 42,
    custom_model: Optional[Any] = None,
    method_kwargs: Optional[dict[str, Any]] = None,
) -> Any:
    """
    Build a classifier by canonical method string.
    """
    kwargs = dict(method_kwargs or {})
    method = method.lower()

    if custom_model is not None:
        return _safe_clone(custom_model)
    if method in {"logit", "logistic", "logistic_regression"}:
        return LogisticRegression(**kwargs)
    if method in {"logit_cv", "logistic_cv", "logistic_regression_cv"}:
        return LogisticRegressionCV(**kwargs)
    if method in {"random_forest", "random_forest_classifier"}:
        kwargs.setdefault("random_state", random_state)
        return RandomForestClassifier(**kwargs)
    if method in {"xgboost", "xgboost_classifier"}:
        try:
            from xgboost import XGBClassifier
        except Exception as exc:
            raise ImportError("xgboost is required for method='xgboost_classifier'.") from exc
        kwargs.setdefault("random_state", random_state)
        return XGBClassifier(**kwargs)
    if method in {"catboost", "catboost_classifier"}:
        try:
            from catboost import CatBoostClassifier
        except Exception as exc:
            raise ImportError("catboost is required for method='catboost_classifier'.") from exc
        kwargs.setdefault("random_seed", random_state)
        kwargs.setdefault("verbose", False)
        return CatBoostClassifier(**kwargs)

    raise ValueError(f"Unknown classification method '{method}'.")


def extract_feature_importance(model: Any, feature_names: list[str]) -> pd.DataFrame:
    """
    Extract feature importance/coefficient ranking as DataFrame.
    """
    if hasattr(model, "feature_importances_"):
        vals = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_, dtype=float)
        if coef.ndim == 1:
            vals = coef
        else:
            vals = np.abs(coef).mean(axis=0)
    else:
        raise ValueError(
            "Model must expose 'feature_importances_' or 'coef_' for importance extraction."
        )

    score = pd.Series(np.abs(vals), index=feature_names, dtype=float).sort_values(ascending=False)
    return score.to_frame(name="feature_importance")
