import pandas as pd
from abc import abstractmethod
from typing import Union, List

from factorlab.core.base_transform import BaseTransform
from factorlab.utils import to_dataframe


class BaseSignal(BaseTransform):
    """
    Base class for all signal generation methods, discrete or continuous.
    Enforces the required input parameters and BaseTransform interface.
    """

    def __init__(self,
                 input_col: str,
                 output_col: str = 'signal',
                 axis: str = 'ts'
                 ):
        super().__init__(name="BaseSignal", description="Generates a trading signal from a score column.")

        self.input_col = input_col
        self.output_col = output_col
        self.axis = axis

        if self.axis not in {'cs', 'ts'}:
            raise ValueError("Axis must be 'cs' (cross-sectional) or 'ts' (time-series).")

    @property
    def inputs(self) -> List[str]:
        # Requires the raw factor score or forecast column
        return [self.input_col]

    def fit(self, X: Union[pd.Series, pd.DataFrame], y=None) -> 'BaseSignal':
        """Minimal fit implementation: only validates inputs."""
        df_input = to_dataframe(X)
        self.validate_inputs(df_input)
        self._is_fitted = True
        return self

    @abstractmethod
    def _compute_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Abstract method to be implemented by subclasses.

        This method should contain the unique logic for computing the trend factor,
        without any normalization or winsorization.
        """
        raise NotImplementedError

    def transform(self, X: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
        """
        Public interface for applying the vol factor.
        Performs checks and prepares data before calling the internal _transform method.
        """
        if not self._is_fitted:
            raise RuntimeError(f"Transform '{self.name}' must be fitted before calling transform()")

        # validate and create copy
        df_input = to_dataframe(X).copy(deep=True)
        self.validate_inputs(df_input)

        # transform and return
        return self._transform(df_input)

    def _transform(self, X: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
        """
        Applies the full TrendFactor computation pipeline.

        Parameters
        ----------
        X : Union[pd.Series, pd.DataFrame]
            Input data containing the required columns.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the computed trend factor.
        """
        df = X.copy()

        # compute trend
        signal_df = self._compute_signal(df)

        # add to original df
        X[self.output_col] = signal_df[self.input_col]

        return X
