import pandas as pd
from typing import Union, List
from sklearn.base import BaseEstimator

from factorlab.core.base_transform import BaseTransform
from factorlab.utils import to_dataframe


class SKLearnWrapper(BaseTransform):
    """
    Wraps a scikit-learn compatible model (estimator) to integrate it
    as a prediction step within the FactorLab Pipeline.

    This class provides the necessary 'fit' and 'transform' methods to delegate
    work to the internal ML model, ensuring compatibility with the Pipeline
    and BaseTransform interface.
    """

    def __init__(self,
                 model: BaseEstimator,
                 feature_cols: List[str],
                 target_col: str,
                 output_col: str = 'forecast_score',
                 prediction_method: str = 'predict'):
        """
        Args:
            model: An instantiated scikit-learn estimator (e.g., RandomForestClassifier()).
            feature_cols: List of column names to use as features (X). These must be present
                          in the input DataFrame passed to transform().
            target_col: Name of the column to use as the target (y).
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

        # Input validation for the underlying model's capability
        if not hasattr(self.model, 'fit') or not hasattr(self.model, prediction_method):
            raise ValueError(f"Model must have 'fit' and '{prediction_method}' methods.")

    @property
    def inputs(self) -> List[str]:
        # The fit operation requires both features and the target.
        # The transform operation only requires features.
        return self.feature_cols + [self.target_col]

    def fit(self, X: Union[pd.Series, pd.DataFrame], y=None) -> 'SKLearnWrapper':
        """Fits the underlying ML model using the provided training data."""
        df_input = to_dataframe(X).copy(deep=True)
        self.validate_inputs(df_input)

        X_train = df_input[self.feature_cols]
        # Extract the target column for fitting
        y_train = df_input[self.target_col]

        # Fit the model
        self.model.fit(X_train, y_train)
        self._is_fitted = True
        print(f"SKLearnWrapper: Successfully fitted {self.model.__class__.__name__}.")
        return self

    def transform(self, X: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
        """Applies the underlying ML model to generate a forecast score column."""
        if not self._is_fitted:
            raise RuntimeError(f"Transform '{self.name}' must be fitted before calling transform()")

        df = to_dataframe(X).copy(deep=True)
        # We only need the features for transformation, but validate all inputs for safety
        self.validate_inputs(df)

        X_test = df[self.feature_cols]

        # Select the appropriate prediction method
        if self.prediction_method == 'predict_proba' and hasattr(self.model, 'predict_proba'):
            # Generally, the probability of the positive class is index 1
            scores = self.model.predict_proba(X_test)[:, 1]
        elif hasattr(self.model, self.prediction_method):
            # For 'predict' or other custom methods
            scores = getattr(self.model, self.prediction_method)(X_test)
        else:
            raise AttributeError(
                f"Model {self.model.__class__.__name__} does not have method '{self.prediction_method}'")

        # Add the resulting score column to the DataFrame
        df[self.output_col] = scores

        return df
