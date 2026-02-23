from __future__ import annotations

from copy import deepcopy
from typing import Any, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor

from factorlab.learning.base import SupervisedLearner
from factorlab.learning.model_registry import build_classifier, build_regressor
from factorlab.learning.selectors import FeatureSelector
from factorlab.learning.sklearn_wrapper import SKLearnWrapper
from factorlab.learning.utils import resolve_feature_columns
from factorlab.core.utils.utils import to_dataframe


class RegressionLearner(SKLearnWrapper):
    """
    Pipeline-native regression learner built from the shared model registry.

    Use ``method`` for full model coverage (linear/lasso/lars/ridge/elastic-net/
    random-forest/xgboost/catboost).
    """

    def __init__(
        self,
        method: str = "linear_regression",
        feature_cols: Optional[List[str]] = None,
        target_col: Optional[str] = None,
        output_col: str = "forecast",
        exclude_cols: Optional[List[str]] = None,
        model: Optional[Any] = None,
        n_features: Optional[int] = None,
        random_state: int = 42,
        prediction_method: str = "predict",
        **model_kwargs: Any,
    ):
        estimator = build_regressor(
            method=method,
            n_features=n_features,
            random_state=random_state,
            custom_model=model,
            method_kwargs=model_kwargs,
        )
        self.method = method
        self.model_kwargs = dict(model_kwargs)

        super().__init__(
            model=estimator,
            feature_cols=feature_cols,
            target_col=target_col,
            output_col=output_col,
            prediction_method=prediction_method,
            exclude_cols=exclude_cols,
        )


class ClassificationLearner(SKLearnWrapper):
    """
    Pipeline-native classification learner built from the shared model registry.

    Use ``method`` for full model coverage (logistic/logistic-cv/random-forest/
    xgboost/catboost).
    """

    def __init__(
        self,
        method: str = "logistic_regression",
        feature_cols: Optional[List[str]] = None,
        target_col: Optional[str] = None,
        output_col: str = "forecast_proba",
        exclude_cols: Optional[List[str]] = None,
        model: Optional[Any] = None,
        random_state: int = 42,
        prediction_method: str = "predict_proba",
        **model_kwargs: Any,
    ):
        estimator = build_classifier(
            method=method,
            random_state=random_state,
            custom_model=model,
            method_kwargs=model_kwargs,
        )
        self.method = method
        self.model_kwargs = dict(model_kwargs)

        super().__init__(
            model=estimator,
            feature_cols=feature_cols,
            target_col=target_col,
            output_col=output_col,
            prediction_method=prediction_method,
            exclude_cols=exclude_cols,
        )


class LinearRegressionLearner(RegressionLearner):
    def __init__(
        self,
        feature_cols: Optional[List[str]] = None,
        target_col: Optional[str] = None,
        output_col: str = "forecast",
        exclude_cols: Optional[List[str]] = None,
        **model_kwargs: Any,
    ):
        super().__init__(
            method="linear_regression",
            feature_cols=feature_cols,
            target_col=target_col,
            output_col=output_col,
            exclude_cols=exclude_cols,
            **model_kwargs,
        )


class RidgeLearner(RegressionLearner):
    def __init__(
        self,
        alpha: float = 1.0,
        feature_cols: Optional[List[str]] = None,
        target_col: Optional[str] = None,
        output_col: str = "forecast",
        exclude_cols: Optional[List[str]] = None,
        **model_kwargs: Any,
    ):
        super().__init__(
            method="ridge",
            feature_cols=feature_cols,
            target_col=target_col,
            output_col=output_col,
            exclude_cols=exclude_cols,
            alpha=alpha,
            **model_kwargs,
        )


class LassoLearner(RegressionLearner):
    def __init__(
        self,
        alpha: float = 0.05,
        feature_cols: Optional[List[str]] = None,
        target_col: Optional[str] = None,
        output_col: str = "forecast",
        exclude_cols: Optional[List[str]] = None,
        **model_kwargs: Any,
    ):
        super().__init__(
            method="lasso",
            feature_cols=feature_cols,
            target_col=target_col,
            output_col=output_col,
            exclude_cols=exclude_cols,
            alpha=alpha,
            **model_kwargs,
        )


class ElasticNetLearner(RegressionLearner):
    def __init__(
        self,
        alpha: float = 0.05,
        l1_ratio: float = 0.5,
        feature_cols: Optional[List[str]] = None,
        target_col: Optional[str] = None,
        output_col: str = "forecast",
        exclude_cols: Optional[List[str]] = None,
        **model_kwargs: Any,
    ):
        super().__init__(
            method="elastic_net",
            feature_cols=feature_cols,
            target_col=target_col,
            output_col=output_col,
            exclude_cols=exclude_cols,
            alpha=alpha,
            l1_ratio=l1_ratio,
            **model_kwargs,
        )


class RandomForestRegressorLearner(RegressionLearner):
    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: Optional[int] = None,
        random_state: int = 42,
        feature_cols: Optional[List[str]] = None,
        target_col: Optional[str] = None,
        output_col: str = "forecast",
        exclude_cols: Optional[List[str]] = None,
        **model_kwargs: Any,
    ):
        super().__init__(
            method="random_forest",
            feature_cols=feature_cols,
            target_col=target_col,
            output_col=output_col,
            exclude_cols=exclude_cols,
            random_state=random_state,
            n_estimators=n_estimators,
            max_depth=max_depth,
            **model_kwargs,
        )


class XGBoostRegressorLearner(RegressionLearner):
    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 6,
        random_state: int = 42,
        feature_cols: Optional[List[str]] = None,
        target_col: Optional[str] = None,
        output_col: str = "forecast",
        exclude_cols: Optional[List[str]] = None,
        **model_kwargs: Any,
    ):
        super().__init__(
            method="xgboost",
            feature_cols=feature_cols,
            target_col=target_col,
            output_col=output_col,
            exclude_cols=exclude_cols,
            random_state=random_state,
            n_estimators=n_estimators,
            max_depth=max_depth,
            **model_kwargs,
        )


class CatBoostRegressorLearner(RegressionLearner):
    def __init__(
        self,
        iterations: int = 300,
        depth: int = 6,
        learning_rate: float = 0.05,
        random_state: int = 42,
        feature_cols: Optional[List[str]] = None,
        target_col: Optional[str] = None,
        output_col: str = "forecast",
        exclude_cols: Optional[List[str]] = None,
        **model_kwargs: Any,
    ):
        super().__init__(
            method="catboost",
            feature_cols=feature_cols,
            target_col=target_col,
            output_col=output_col,
            exclude_cols=exclude_cols,
            random_state=random_state,
            iterations=iterations,
            depth=depth,
            learning_rate=learning_rate,
            **model_kwargs,
        )


class LogisticRegressionLearner(ClassificationLearner):
    def __init__(
        self,
        feature_cols: Optional[List[str]] = None,
        target_col: Optional[str] = None,
        output_col: str = "forecast_proba",
        exclude_cols: Optional[List[str]] = None,
        **model_kwargs: Any,
    ):
        super().__init__(
            method="logistic_regression",
            feature_cols=feature_cols,
            target_col=target_col,
            output_col=output_col,
            exclude_cols=exclude_cols,
            prediction_method="predict_proba",
            **model_kwargs,
        )


class LogisticRegressionCVLearner(ClassificationLearner):
    def __init__(
        self,
        feature_cols: Optional[List[str]] = None,
        target_col: Optional[str] = None,
        output_col: str = "forecast_proba",
        exclude_cols: Optional[List[str]] = None,
        **model_kwargs: Any,
    ):
        super().__init__(
            method="logistic_regression_cv",
            feature_cols=feature_cols,
            target_col=target_col,
            output_col=output_col,
            exclude_cols=exclude_cols,
            prediction_method="predict_proba",
            **model_kwargs,
        )


class RandomForestClassifierLearner(ClassificationLearner):
    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: Optional[int] = None,
        random_state: int = 42,
        feature_cols: Optional[List[str]] = None,
        target_col: Optional[str] = None,
        output_col: str = "forecast_proba",
        exclude_cols: Optional[List[str]] = None,
        prediction_method: str = "predict_proba",
        **model_kwargs: Any,
    ):
        super().__init__(
            method="random_forest",
            feature_cols=feature_cols,
            target_col=target_col,
            output_col=output_col,
            exclude_cols=exclude_cols,
            random_state=random_state,
            prediction_method=prediction_method,
            n_estimators=n_estimators,
            max_depth=max_depth,
            **model_kwargs,
        )


class XGBoostClassifierLearner(ClassificationLearner):
    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 6,
        random_state: int = 42,
        feature_cols: Optional[List[str]] = None,
        target_col: Optional[str] = None,
        output_col: str = "forecast_proba",
        exclude_cols: Optional[List[str]] = None,
        prediction_method: str = "predict_proba",
        **model_kwargs: Any,
    ):
        super().__init__(
            method="xgboost",
            feature_cols=feature_cols,
            target_col=target_col,
            output_col=output_col,
            exclude_cols=exclude_cols,
            random_state=random_state,
            prediction_method=prediction_method,
            n_estimators=n_estimators,
            max_depth=max_depth,
            **model_kwargs,
        )


class CatBoostClassifierLearner(ClassificationLearner):
    def __init__(
        self,
        iterations: int = 300,
        depth: int = 6,
        learning_rate: float = 0.05,
        random_state: int = 42,
        feature_cols: Optional[List[str]] = None,
        target_col: Optional[str] = None,
        output_col: str = "forecast_proba",
        exclude_cols: Optional[List[str]] = None,
        prediction_method: str = "predict_proba",
        **model_kwargs: Any,
    ):
        super().__init__(
            method="catboost",
            feature_cols=feature_cols,
            target_col=target_col,
            output_col=output_col,
            exclude_cols=exclude_cols,
            random_state=random_state,
            prediction_method=prediction_method,
            iterations=iterations,
            depth=depth,
            learning_rate=learning_rate,
            **model_kwargs,
        )


class MLPRegressorLearner(SKLearnWrapper):
    def __init__(
        self,
        hidden_layer_sizes: tuple[int, ...] = (64, 32),
        random_state: int = 42,
        feature_cols: Optional[List[str]] = None,
        target_col: Optional[str] = None,
        output_col: str = "forecast",
        exclude_cols: Optional[List[str]] = None,
        **model_kwargs: Any,
    ):
        super().__init__(
            model=MLPRegressor(
                hidden_layer_sizes=hidden_layer_sizes,
                random_state=random_state,
                **model_kwargs,
            ),
            feature_cols=feature_cols,
            target_col=target_col,
            output_col=output_col,
            prediction_method="predict",
            exclude_cols=exclude_cols,
        )


class MLPClassifierLearner(SKLearnWrapper):
    def __init__(
        self,
        hidden_layer_sizes: tuple[int, ...] = (64, 32),
        random_state: int = 42,
        feature_cols: Optional[List[str]] = None,
        target_col: Optional[str] = None,
        output_col: str = "forecast_proba",
        exclude_cols: Optional[List[str]] = None,
        **model_kwargs: Any,
    ):
        super().__init__(
            model=MLPClassifier(
                hidden_layer_sizes=hidden_layer_sizes,
                random_state=random_state,
                **model_kwargs,
            ),
            feature_cols=feature_cols,
            target_col=target_col,
            output_col=output_col,
            prediction_method="predict_proba",
            exclude_cols=exclude_cols,
        )


class TorchMLPRegressor:
    """
    Optional lightweight torch estimator with sklearn-like fit/predict.
    """

    def __init__(
        self,
        input_dim: Optional[int] = None,
        hidden_dim: int = 64,
        epochs: int = 50,
        lr: float = 1e-3,
        batch_size: int = 128,
        random_state: int = 42,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.random_state = random_state
        self._net = None

    def fit(self, X, y):
        try:
            import torch
        except Exception as exc:
            raise ImportError("torch is required for TorchMLPRegressor.") from exc

        X_np = np.asarray(X, dtype=np.float32)
        y_np = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        in_dim = int(X_np.shape[1] if self.input_dim is None else self.input_dim)

        torch.manual_seed(self.random_state)
        self._net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, 1),
        )
        optimizer = torch.optim.Adam(self._net.parameters(), lr=self.lr)
        loss_fn = torch.nn.MSELoss()

        dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(X_np),
            torch.from_numpy(y_np),
        )
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=min(self.batch_size, len(dataset)),
            shuffle=True,
        )

        self._net.train()
        for _ in range(int(self.epochs)):
            for xb, yb in loader:
                optimizer.zero_grad()
                pred = self._net(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                optimizer.step()
        return self

    def predict(self, X):
        if self._net is None:
            raise RuntimeError("TorchMLPRegressor must be fitted before predict().")
        import torch

        X_np = np.asarray(X, dtype=np.float32)
        self._net.eval()
        with torch.no_grad():
            pred = self._net(torch.from_numpy(X_np)).cpu().numpy().reshape(-1)
        return pred


class TorchRegressorLearner(SKLearnWrapper):
    def __init__(
        self,
        feature_cols: Optional[List[str]] = None,
        target_col: Optional[str] = None,
        output_col: str = "forecast",
        exclude_cols: Optional[List[str]] = None,
        **model_kwargs: Any,
    ):
        super().__init__(
            model=TorchMLPRegressor(**model_kwargs),
            feature_cols=feature_cols,
            target_col=target_col,
            output_col=output_col,
            prediction_method="predict",
            exclude_cols=exclude_cols,
        )


class SupervisedPCALearner(SupervisedLearner):
    """
    Pipeline-native supervised PCA learner.

    Workflow:
    1) Select a fold-local feature subset with FeatureSelector.
    2) Fit PCA on selected features.
    3) Fit regression model on principal components.
    """

    def __init__(
        self,
        feature_cols: Optional[Sequence[str]] = None,
        target_col: Optional[str] = None,
        output_col: str = "forecast",
        exclude_cols: Optional[Sequence[str]] = None,
        selection_method: str = "lasso",
        n_features: int = 30,
        n_components: Optional[int] = None,
        selector_kwargs: Optional[dict[str, Any]] = None,
        model: Optional[Any] = None,
        random_state: int = 42,
    ):
        super().__init__(
            name="SupervisedPCALearner",
            description="Feature selection + PCA + supervised regression learner.",
        )
        if n_features < 1:
            raise ValueError("n_features must be >= 1.")
        if n_components is not None and n_components < 1:
            raise ValueError("n_components must be >= 1 when provided.")

        self.feature_cols = list(feature_cols) if feature_cols is not None else None
        self.target_col = target_col
        self.output_col = output_col
        self.exclude_cols = list(exclude_cols) if exclude_cols is not None else []
        self.selection_method = selection_method
        self.n_features = int(n_features)
        self.n_components = n_components
        self.selector_kwargs = dict(selector_kwargs or {})
        self.model = model if model is not None else LinearRegression()
        self.random_state = int(random_state)

        self._resolved_feature_cols: list[str] = []
        self._selected_feature_cols: list[str] = []
        self._selector: Optional[FeatureSelector] = None
        self._pca: Optional[PCA] = None
        self._model = None

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
        df: pd.DataFrame,
        y: Optional[Union[pd.Series, pd.DataFrame]],
    ) -> pd.Series:
        if y is not None:
            y_df = to_dataframe(y).copy(deep=True)
            if y_df.shape[1] != 1:
                raise ValueError("y must contain exactly one target column.")
            return y_df.iloc[:, 0].reindex(df.index)

        if self.target_col is None:
            raise ValueError("target_col must be provided when fit(..., y=None).")
        if self.target_col not in df.columns:
            raise ValueError(f"target_col '{self.target_col}' not found in input columns.")
        return df[self.target_col]

    def fit(
        self,
        X: Union[pd.Series, pd.DataFrame],
        y: Optional[Union[pd.Series, pd.DataFrame]] = None,
    ) -> "SupervisedPCALearner":
        df = to_dataframe(X).copy(deep=True)
        self._resolved_feature_cols = resolve_feature_columns(
            df=df,
            feature_cols=self.feature_cols,
            exclude_cols=self.exclude_cols + [self.target_col, self.output_col],
            require_numeric=True,
        )
        target = self._resolve_target(df=df, y=y)

        selector = FeatureSelector(
            method=self.selection_method,
            feature_cols=self._resolved_feature_cols,
            n_features=min(self.n_features, len(self._resolved_feature_cols)),
            drop_unselected=True,
            absolute_scores=True,
            random_state=self.random_state,
            method_kwargs=self.selector_kwargs,
        )
        selector.fit(df, target)
        self._selector = selector
        self._selected_feature_cols = list(selector.selected_features_)
        if len(self._selected_feature_cols) == 0:
            raise ValueError("SupervisedPCALearner selected zero features.")

        X_selected = df[self._selected_feature_cols]
        valid = X_selected.notna().all(axis=1) & target.notna()
        if int(valid.sum()) == 0:
            raise ValueError("No valid non-NaN rows available for SupervisedPCALearner fit.")

        X_train = X_selected.loc[valid]
        y_train = target.loc[valid]
        n_components = self.n_components or min(self.n_features, X_train.shape[1])
        n_components = max(1, min(int(n_components), int(X_train.shape[1])))

        self._pca = PCA(n_components=n_components, random_state=self.random_state)
        X_pcs = self._pca.fit_transform(X_train)

        self._model = self._safe_clone(self.model)
        self._model.fit(X_pcs, y_train)
        self._is_fitted = True
        return self

    def transform(self, X: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
        if not self._is_fitted:
            raise RuntimeError(f"Transform '{self.name}' must be fitted before calling transform().")
        if self._selector is None or self._pca is None or self._model is None:
            raise RuntimeError("SupervisedPCALearner internal components are not fitted.")

        df = to_dataframe(X).copy(deep=True)
        missing = set(self._selected_feature_cols) - set(df.columns)
        if missing:
            raise ValueError(f"Missing selected features for transform: {missing}")

        X_selected = df[self._selected_feature_cols]
        valid = X_selected.notna().all(axis=1)
        pred = pd.Series(np.nan, index=df.index, dtype=float)
        if int(valid.sum()) > 0:
            X_pcs = self._pca.transform(X_selected.loc[valid])
            pred.loc[valid] = np.asarray(self._model.predict(X_pcs), dtype=float).reshape(-1)

        df[self.output_col] = pred
        return df
