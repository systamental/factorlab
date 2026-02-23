import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from factorlab.core.pipeline import Pipeline
from factorlab.learning import FeatureSelector
from factorlab.learning.sklearn_wrapper import SKLearnWrapper


def _make_df(n: int = 120, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2021-01-01", periods=n, freq="D")
    f1 = rng.normal(size=n)
    f2 = rng.normal(size=n)
    noise = rng.normal(scale=0.05, size=n)
    target = 0.8 * f1 + 0.1 * noise
    return pd.DataFrame({"f1": f1, "f2": f2, "target": target}, index=idx)


def test_feature_selector_spearman_selects_informative_feature():
    df = _make_df()
    selector = FeatureSelector(
        method="spearman",
        feature_cols=["f1", "f2"],
        n_features=1,
        drop_unselected=True,
    )
    selector.fit(df, y=df["target"])
    out = selector.transform(df)

    assert selector.selected_features_ == ["f1"]
    assert "f1" in out.columns
    assert "f2" not in out.columns
    assert "target" in out.columns


def test_feature_selector_variance_works_without_target():
    idx = pd.date_range("2022-01-01", periods=40, freq="D")
    df = pd.DataFrame(
        {
            "f1": np.linspace(0.0, 1.0, len(idx)),
            "f2": 1.0,
            "target": np.linspace(0.0, 1.0, len(idx)),
        },
        index=idx,
    )
    selector = FeatureSelector(
        method="variance",
        feature_cols=["f1", "f2"],
        n_features=1,
        drop_unselected=True,
    )
    selector.fit(df)
    out = selector.transform(df)

    assert selector.selected_features_ == ["f1"]
    assert "f2" not in out.columns


def test_pipeline_feature_selector_with_auto_sklearn_features():
    df = _make_df()
    pipeline = Pipeline(
        steps=[
            (
                "selector",
                FeatureSelector(
                    method="spearman",
                    feature_cols=["f1", "f2"],
                    target_col="target",
                    n_features=1,
                    drop_unselected=True,
                ),
            ),
            (
                "learner",
                SKLearnWrapper(
                    model=LinearRegression(),
                    feature_cols=None,
                    target_col="target",
                    output_col="forecast",
                ),
            ),
        ]
    )
    out = pipeline.fit_transform(df)

    assert "forecast" in out.columns
    assert out["forecast"].notna().sum() > 0


def test_feature_selector_supports_legacy_alias_names():
    df = _make_df()
    selector = FeatureSelector(
        method="spearman_rank",
        feature_cols=["f1", "f2"],
        target_col="target",
        n_features=1,
    )
    selector.fit(df)
    assert selector.method == "spearman"
    assert selector.selected_features_ == ["f1"]


def test_feature_selector_categorical_association_methods():
    df = _make_df()
    for method in ("cramer_v", "tschuprow", "pearson_cc", "chi2"):
        selector = FeatureSelector(
            method=method,
            feature_cols=["f1", "f2"],
            target_col="target",
            n_features=1,
            feature_bins=4,
            target_bins=3,
        )
        selector.fit(df)
        assert len(selector.selected_features_) == 1
        assert selector.selected_features_[0] in {"f1", "f2"}


def test_feature_selector_model_methods():
    df = _make_df()
    methods = [
        ("lars", {}),
        ("lasso", {"alpha": 0.01}),
        ("ridge", {"alpha": 1.0}),
        ("elastic_net", {"alpha": 0.01, "l1_ratio": 0.5}),
        ("random_forest", {"n_estimators": 20, "max_depth": 3}),
        ("rfe", {}),
        ("forward", {}),
        ("backward", {}),
        ("stepwise", {}),
        ("exhaustive", {}),
    ]
    for method, kwargs in methods:
        selector = FeatureSelector(
            method=method,
            feature_cols=["f1", "f2"],
            target_col="target",
            n_features=1,
            method_kwargs=kwargs,
        )
        selector.fit(df)
        assert selector.selected_features_ == ["f1"]


def test_feature_selector_xgboost_optional_dependency():
    df = _make_df()
    selector = FeatureSelector(
        method="xgboost",
        feature_cols=["f1", "f2"],
        target_col="target",
        n_features=1,
        method_kwargs={"n_estimators": 20, "max_depth": 3},
    )
    try:
        selector.fit(df)
    except ImportError:
        pytest.skip("xgboost is not installed in test environment.")
    assert selector.selected_features_ == ["f1"]


def test_feature_selector_catboost_optional_dependency():
    df = _make_df()
    selector = FeatureSelector(
        method="catboost",
        feature_cols=["f1", "f2"],
        target_col="target",
        n_features=1,
        method_kwargs={"iterations": 25, "depth": 3, "learning_rate": 0.2},
    )
    try:
        selector.fit(df)
    except ImportError:
        pytest.skip("catboost is not installed in test environment.")
    assert selector.selected_features_ == ["f1"]


def test_feature_selector_redundancy_methods():
    df = _make_df()
    for method in ("mrmr", "mifs", "mrmr_mifs", "spearman_mrmr"):
        selector = FeatureSelector(
            method=method,
            feature_cols=["f1", "f2"],
            target_col="target",
            n_features=1,
        )
        selector.fit(df)
        assert selector.selected_features_ == ["f1"]
