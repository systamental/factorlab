import pandas as pd
from typing import Optional, Union, List
from sklearn.base import BaseEstimator

from factorlab.learning.base import SupervisedLearner
from factorlab.learning.utils import resolve_feature_columns
from factorlab.core.utils.utils import to_dataframe


class SKLearnWrapper(SupervisedLearner):
    """
    Wraps a scikit-learn compatible model (estimator) to integrate it
    as a prediction step within the FactorLab Pipeline.

    This class provides the necessary 'fit' and 'transform' methods to delegate
    work to the internal ML model, ensuring compatibility with the Pipeline
    and BaseTransform interface.
    """

    def __init__(self,
                 model: BaseEstimator,
                 feature_cols: Optional[List[str]] = None,
                 target_col: Optional[str] = None,
                 output_col: str = 'forecast_score',
                 prediction_method: str = 'predict',
                 exclude_cols: Optional[List[str]] = None):
        """
        Args:
            model: An instantiated scikit-learn estimator (e.g., RandomForestClassifier()).
            feature_cols: List of column names to use as features (X). These must be present
                          in the input DataFrame passed to transform().
            target_col: Optional name of the column to use as the target (y) when
                        `fit(..., y=None)` is used. If `fit(..., y=...)` is provided,
                        this can be omitted.
            output_col: Name of the new column that will hold the prediction score/signal.
            prediction_method: 'predict' (for classification/regression output) or
                               'predict_proba' (for probability scores).
        """
        # Assign a descriptive name incorporating the specific model being used
        super().__init__(
            name=f"SKLearnWrapper({model.__class__.__name__})",
            description=f"Trains and applies {model.__class__.__name__} to generate a forecast score."
        )

        self.model = model
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.output_col = output_col
        self.prediction_method = prediction_method
        self.exclude_cols = exclude_cols or []
        self._resolved_feature_cols: List[str] = []

        # Input validation for the underlying model's capability
        if not hasattr(self.model, 'fit') or not hasattr(self.model, prediction_method):
            raise ValueError(f"Model must have 'fit' and '{prediction_method}' methods.")

    @property
    def inputs(self) -> List[str]:
        required = list(self.feature_cols) if self.feature_cols is not None else []
        if self.target_col is not None:
            required.append(self.target_col)
        return required

    def fit(
        self,
        X: Union[pd.Series, pd.DataFrame],
        y: Optional[Union[pd.Series, pd.DataFrame]] = None,
    ) -> 'SKLearnWrapper':
        """Fits the underlying ML model using the provided training data."""
        df_input = to_dataframe(X).copy(deep=True)
        self._resolved_feature_cols = resolve_feature_columns(
            df=df_input,
            feature_cols=self.feature_cols,
            exclude_cols=list(self.exclude_cols) + [self.target_col, self.output_col],
            require_numeric=True,
        )
        X_train = df_input[self._resolved_feature_cols]
        if y is None:
            if self.target_col is None:
                raise ValueError("target_col must be provided when fit(..., y=None).")
            if self.target_col not in df_input.columns:
                raise ValueError(f"target_col '{self.target_col}' not found in input columns.")
            y_train = df_input[self.target_col]
        else:
            y_df = to_dataframe(y).copy(deep=True)
            if y_df.shape[1] != 1:
                raise ValueError("y must contain exactly one target column.")
            y_series = y_df.iloc[:, 0]
            y_train = y_series.reindex(X_train.index)

        valid = X_train.notna().all(axis=1) & y_train.notna()
        if int(valid.sum()) == 0:
            raise ValueError("No valid non-NaN rows available for model fit.")
        X_train = X_train.loc[valid]
        y_train = y_train.loc[valid]

        # Fit the model
        self.model.fit(X_train, y_train)
        self._is_fitted = True
        return self

    def transform(self, X: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
        """Applies the underlying ML model to generate a forecast score column."""
        if not self._is_fitted:
            raise RuntimeError(f"Transform '{self.name}' must be fitted before calling transform()")
        if len(self._resolved_feature_cols) == 0:
            raise RuntimeError("No resolved feature columns found. Fit must resolve feature columns first.")

        df = to_dataframe(X).copy(deep=True)
        missing_features = set(self._resolved_feature_cols) - set(df.columns)
        if missing_features:
            raise ValueError(f"Missing feature columns for transform: {missing_features}")

        X_test = df[self._resolved_feature_cols]
        valid = X_test.notna().all(axis=1)

        # Select the appropriate prediction method
        scores = pd.Series(index=df.index, dtype=float)
        if valid.any():
            X_eval = X_test.loc[valid]
            if self.prediction_method == 'predict_proba' and hasattr(self.model, 'predict_proba'):
                # Generally, the probability of the positive class is index 1
                pred = self.model.predict_proba(X_eval)[:, 1]
            elif hasattr(self.model, self.prediction_method):
                # For 'predict' or other custom methods
                pred = getattr(self.model, self.prediction_method)(X_eval)
            else:
                raise AttributeError(
                    f"Model {self.model.__class__.__name__} does not have method '{self.prediction_method}'")
            scores.loc[valid] = pred

        # Add the resulting score column to the DataFrame
        df[self.output_col] = scores

        return df
