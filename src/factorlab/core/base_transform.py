from abc import ABC, abstractmethod
import pandas as pd
import uuid
from typing import List, Optional, Union, Any


class BaseTransform(ABC):
    """
    Abstract base class for transformation components in a quant pipeline.

    Adopts the Scikit-learn fit/transform contract to ensure proper separation
    of learning (fit) and application (transform) for robust backtesting
    and prevention of look-ahead bias.
    """

    def __init__(self, name: Optional[str] = None, description: Optional[str] = None):
        self.name = name or self.__class__.__name__
        self.description = description or ""
        self.transform_id = str(uuid.uuid4())

        # State management
        self._is_fitted: bool = False
        self._fitted_params: dict[str, Any] = {}

    @property
    def inputs(self) -> List[str]:
        return []

    def validate_inputs(self, df: pd.DataFrame):
        missing = set(self.inputs) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns for feature '{self.name}': {missing}")

    @abstractmethod
    def fit(self,
            X: Union[pd.Series, pd.DataFrame],
            y: Optional[Union[pd.Series, pd.DataFrame]] = None) -> 'BaseTransform':
        """
        Base implementation of fit. Handles the internal state management.
        Subclasses should use super().fit() to run this logic.
        """
        raise NotImplementedError("Each transform must implement fit().")

    @abstractmethod
    def transform(self,
                  X: Union[pd.Series, pd.DataFrame]
                  ) -> pd.DataFrame:
        """
        Apply transformation using learned parameters.
        Raises RuntimeError if not fitted.
        """
        raise NotImplementedError("Each transform must implement transform().")

    def fit_transform(self,
                      X: Union[pd.Series, pd.DataFrame],
                      y: Optional[Union[pd.Series, pd.DataFrame]] = None) -> pd.DataFrame:
        """
        Convenience method: fit the transform and immediately apply it.
        """
        return self.fit(X, y).transform(X)

    def compute(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Backward compatibility: alias for transform(). Auto-fit stateless transforms.
        """
        if not self._is_fitted:
            self.fit(X)
        return self.transform(X)

    def get_metadata(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "class": self.__class__.__name__,
            "transform_id": self.transform_id,
            "inputs": self.inputs,
            "is_fitted": self._is_fitted
        }

    def __repr__(self):
        return f"<Transform {self.name} ({self.__class__.__name__}) [Fitted: {self._is_fitted}]>"
