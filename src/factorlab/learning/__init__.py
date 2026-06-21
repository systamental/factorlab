from __future__ import annotations

from typing import Any

from factorlab.learning.supervised.learners import (
    CatBoostClassifierLearner,
    CatBoostRegressorLearner,
    ClassificationLearner,
    ElasticNetLearner,
    LassoLearner,
    LinearRegressionLearner,
    LogisticRegressionCVLearner,
    LogisticRegressionLearner,
    MLPClassifierLearner,
    MLPRegressorLearner,
    RegressionLearner,
    RandomForestClassifierLearner,
    RandomForestRegressorLearner,
    RidgeLearner,
    SupervisedPCALearner,
    TorchRegressorLearner,
    XGBoostClassifierLearner,
    XGBoostRegressorLearner,
)
from factorlab.learning.supervised.walk_forward_learner import WalkForwardLearner
from factorlab.learning.selectors import FeatureSelector
from factorlab.learning.splitters import (
    BasePanelSplit,
    ExpandingFrequencyPanelSplit,
    ExpandingIncrementPanelSplit,
    ExpandingKFoldPanelSplit,
    RollingKFoldPanelSplit,
)
from factorlab.learning.time_series_analysis import (
    LagFeatures,
    StatsmodelsOLSLearner,
    TimeSeriesAnalysis,
    TimeSeriesDiagnostics,
    add_lags,
    expanding_window,
    rolling_window,
)
from factorlab.learning.unsupervised.unsupervised_learning import (
    PCATransform,
    PPCATransform,
    R2PCATransform,
)

__all__ = [
    "BasePanelSplit",
    "ExpandingKFoldPanelSplit",
    "RollingKFoldPanelSplit",
    "ExpandingIncrementPanelSplit",
    "ExpandingFrequencyPanelSplit",
    "WalkForwardLearner",
    "FeatureSelector",
    "WalkForwardGridSearch",
    "parameter_grid",
    "set_pipeline_params",
    "RegressionLearner",
    "ClassificationLearner",
    "LinearRegressionLearner",
    "RidgeLearner",
    "LassoLearner",
    "ElasticNetLearner",
    "RandomForestRegressorLearner",
    "CatBoostRegressorLearner",
    "RandomForestClassifierLearner",
    "LogisticRegressionLearner",
    "LogisticRegressionCVLearner",
    "XGBoostClassifierLearner",
    "CatBoostClassifierLearner",
    "XGBoostRegressorLearner",
    "MLPRegressorLearner",
    "MLPClassifierLearner",
    "TorchRegressorLearner",
    "SupervisedPCALearner",
    "add_lags",
    "rolling_window",
    "expanding_window",
    "LagFeatures",
    "StatsmodelsOLSLearner",
    "TimeSeriesAnalysis",
    "TimeSeriesDiagnostics",
    "PCATransform",
    "PPCATransform",
    "R2PCATransform",
]


def __getattr__(name: str) -> Any:
    """
    Lazily expose search utilities to avoid import cycles with targets.forward.
    """
    if name in {"WalkForwardGridSearch", "parameter_grid", "set_pipeline_params"}:
        from factorlab.learning.search import (
            WalkForwardGridSearch,
            parameter_grid,
            set_pipeline_params,
        )

        lazy_exports = {
            "WalkForwardGridSearch": WalkForwardGridSearch,
            "parameter_grid": parameter_grid,
            "set_pipeline_params": set_pipeline_params,
        }
        value = lazy_exports[name]
        globals()[name] = value
        return value
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
