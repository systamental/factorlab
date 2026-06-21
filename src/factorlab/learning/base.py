from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Union

import pandas as pd

from factorlab.core.base_transform import BaseTransform


class SupervisedLearner(BaseTransform, ABC):
    """
    Base protocol for supervised learner steps used inside the Pipeline.
    """

    @abstractmethod
    def fit(
        self,
        X: Union[pd.Series, pd.DataFrame],
        y: Optional[Union[pd.Series, pd.DataFrame]] = None,
    ) -> "SupervisedLearner":
        raise NotImplementedError

    @abstractmethod
    def transform(self, X: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
        raise NotImplementedError
