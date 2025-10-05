from typing import Optional, List, Union
import pandas as pd
from abc import abstractmethod
from factorlab.features.base import Feature


class Factor(Feature):
    """
    Abstract base class for alpha or risk factors in quantitative research.

    Factors are used to capture specific characteristics of financial data
    that can be used to predict returns or assess risk.

    """

    category: Optional[str]
    tags: List[str]

    def __init__(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None
    ):
        super().__init__(name=name, description=description)
        self.category = category
        self.tags = tags or []

    @abstractmethod
    def transform(self, data: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
        """
        Transform the input data to compute the factor values.

        This method MUST be implemented by all subclasses and contains the
        core factor calculation logic.

        Parameters
        ----------
        data : Union[pd.Series, pd.DataFrame]
            Input data (e.g., OHLCV data).

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the computed factor values.
        """
        raise NotImplementedError("Each factor must implement transform().")

    def get_metadata(self) -> dict:
        """
        Extend feature metadata to include factor-specific fields like category and tags.
        """
        metadata = super().get_metadata()
        metadata.update({
            "category": self.category,
            "tags": self.tags,
        })
        return metadata

    def __repr__(self):
        return f"<Factor {self.name} ({self.__class__.__name__})>"
