import numpy as np
import pandas as pd
import pytest

from factorlab.learning import (
    CatBoostClassifierLearner,
    CatBoostRegressorLearner,
    ClassificationLearner,
    LinearRegressionLearner,
    LogisticRegressionCVLearner,
    RandomForestRegressorLearner,
    RegressionLearner,
    SupervisedPCALearner,
    TorchRegressorLearner,
    XGBoostClassifierLearner,
    XGBoostRegressorLearner,
)


def _make_regression_df(n: int = 160, seed: int = 21) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = 0.7 * x1 - 0.2 * x2 + rng.normal(scale=0.05, size=n)
    return pd.DataFrame({"x1": x1, "x2": x2, "target": y}, index=idx)


def _make_classification_df(n: int = 180, seed: int = 44) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    latent = 0.9 * x1 - 0.5 * x2 + rng.normal(scale=0.2, size=n)
    y = (latent > 0).astype(int)
    return pd.DataFrame({"x1": x1, "x2": x2, "target": y}, index=idx)


@pytest.mark.parametrize(
    "method,kwargs,n_features",
    [
        ("linear_regression", {}, None),
        ("ridge", {"alpha": 1.0}, None),
        ("lasso", {"alpha": 0.01}, None),
        ("elastic_net", {"alpha": 0.01, "l1_ratio": 0.5}, None),
        ("lars", {}, 1),
        ("ridge_cv", {}, None),
        ("random_forest", {"n_estimators": 20, "max_depth": 4}, None),
    ],
)
def test_regression_learner_methods_fit_transform(method: str, kwargs: dict, n_features: int | None):
    df = _make_regression_df()
    learner = RegressionLearner(
        method=method,
        feature_cols=["x1", "x2"],
        target_col="target",
        output_col="forecast",
        n_features=n_features,
        **kwargs,
    )
    learner.fit(df)
    out = learner.transform(df)

    assert "forecast" in out.columns
    assert out["forecast"].notna().sum() == len(df)


@pytest.mark.parametrize(
    "method,kwargs,prediction_method",
    [
        ("logistic_regression", {}, "predict_proba"),
        ("logistic_regression_cv", {"cv": 3}, "predict_proba"),
        ("random_forest", {"n_estimators": 20, "max_depth": 4}, "predict_proba"),
        ("random_forest", {"n_estimators": 20, "max_depth": 4}, "predict"),
    ],
)
def test_classification_learner_methods_fit_transform(
    method: str,
    kwargs: dict,
    prediction_method: str,
):
    df = _make_classification_df()
    learner = ClassificationLearner(
        method=method,
        feature_cols=["x1", "x2"],
        target_col="target",
        output_col="forecast",
        prediction_method=prediction_method,
        **kwargs,
    )
    learner.fit(df)
    out = learner.transform(df)

    assert "forecast" in out.columns
    valid = out["forecast"].dropna()
    assert len(valid) == len(df)
    if prediction_method == "predict_proba":
        assert ((valid >= 0.0) & (valid <= 1.0)).all()
    else:
        assert set(valid.unique()).issubset({0.0, 1.0})


def test_linear_regression_learner_fit_transform():
    df = _make_regression_df()
    learner = LinearRegressionLearner(
        feature_cols=["x1", "x2"],
        target_col="target",
        output_col="forecast",
    )
    learner.fit(df)
    out = learner.transform(df)

    assert "forecast" in out.columns
    assert out["forecast"].notna().sum() == len(df)


def test_random_forest_learner_fit_transform():
    df = _make_regression_df()
    learner = RandomForestRegressorLearner(
        n_estimators=20,
        max_depth=4,
        feature_cols=["x1", "x2"],
        target_col="target",
        output_col="forecast",
    )
    learner.fit(df)
    out = learner.transform(df)

    assert "forecast" in out.columns
    assert out["forecast"].notna().sum() == len(df)


def test_logistic_regression_cv_learner_fit_transform():
    df = _make_classification_df()
    learner = LogisticRegressionCVLearner(
        feature_cols=["x1", "x2"],
        target_col="target",
        output_col="forecast",
        cv=3,
    )
    learner.fit(df)
    out = learner.transform(df)

    assert "forecast" in out.columns
    valid = out["forecast"].dropna()
    assert len(valid) == len(df)
    assert ((valid >= 0.0) & (valid <= 1.0)).all()


def test_torch_learner_optional_dependency():
    df = _make_regression_df()
    learner = TorchRegressorLearner(
        feature_cols=["x1", "x2"],
        target_col="target",
        output_col="forecast",
        epochs=2,
    )
    try:
        learner.fit(df)
    except ImportError:
        pytest.skip("torch is not installed in test environment.")
    out = learner.transform(df)
    assert "forecast" in out.columns


def test_supervised_pca_learner_fit_transform():
    df = _make_regression_df(n=200, seed=42)
    df["x3"] = 0.5 * df["x1"] + 0.5 * df["x2"] + np.random.default_rng(1).normal(scale=0.01, size=len(df))

    learner = SupervisedPCALearner(
        feature_cols=["x1", "x2", "x3"],
        target_col="target",
        output_col="forecast",
        selection_method="lasso",
        n_features=2,
        n_components=1,
    )
    learner.fit(df)
    out = learner.transform(df)

    assert "forecast" in out.columns
    assert out["forecast"].notna().sum() == len(df)


def test_xgboost_regressor_learner_optional_dependency():
    df = _make_regression_df()
    try:
        learner = XGBoostRegressorLearner(
            feature_cols=["x1", "x2"],
            target_col="target",
            output_col="forecast",
            n_estimators=20,
            max_depth=3,
        )
    except ImportError:
        pytest.skip("xgboost is not installed in test environment.")

    learner.fit(df)
    out = learner.transform(df)
    assert out["forecast"].notna().sum() == len(df)


def test_xgboost_classifier_learner_optional_dependency():
    df = _make_classification_df()
    try:
        learner = XGBoostClassifierLearner(
            feature_cols=["x1", "x2"],
            target_col="target",
            output_col="forecast",
            n_estimators=20,
            max_depth=3,
        )
    except ImportError:
        pytest.skip("xgboost is not installed in test environment.")

    learner.fit(df)
    out = learner.transform(df)
    valid = out["forecast"].dropna()
    assert len(valid) == len(df)
    assert ((valid >= 0.0) & (valid <= 1.0)).all()


def test_catboost_regressor_learner_optional_dependency():
    df = _make_regression_df()
    try:
        learner = CatBoostRegressorLearner(
            feature_cols=["x1", "x2"],
            target_col="target",
            output_col="forecast",
            iterations=30,
            depth=3,
        )
    except ImportError:
        pytest.skip("catboost is not installed in test environment.")

    learner.fit(df)
    out = learner.transform(df)
    assert out["forecast"].notna().sum() == len(df)


def test_catboost_classifier_learner_optional_dependency():
    df = _make_classification_df()
    try:
        learner = CatBoostClassifierLearner(
            feature_cols=["x1", "x2"],
            target_col="target",
            output_col="forecast",
            iterations=30,
            depth=3,
        )
    except ImportError:
        pytest.skip("catboost is not installed in test environment.")

    learner.fit(df)
    out = learner.transform(df)
    valid = out["forecast"].dropna()
    assert len(valid) == len(df)
    assert ((valid >= 0.0) & (valid <= 1.0)).all()
