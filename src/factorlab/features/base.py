from abc import ABC
import uuid
from typing import Optional
from factorlab.core.base_transform import BaseTransform


class Feature(BaseTransform, ABC):
    """
    Abstract base class for all feature engineering components in FactorLab.

    A Feature is a composite transformation that produces a descriptive variable
    from input data.

    Inherits from BaseTransform, ensuring a unified interface for all data-processing components.

    NOTE: Concrete feature classes must implement the fit() and transform() methods
    inherited from BaseTransform.
    """

    def __init__(self,
                 name: Optional[str] = None,
                 description: Optional[str] = None):

        super().__init__(name=name, description=description)
        self.feature_id = str(uuid.uuid4())  # Unique ID for traceability

    def get_metadata(self) -> dict:
        """
        Return feature metadata for logging, reproducibility, or UI.
        """
        # Get metadata from parent class and add feature-specific metadata
        metadata = super().get_metadata()
        metadata.update({
            "feature_id": self.feature_id,
        })

        return metadata
