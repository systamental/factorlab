from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import List, Optional, Union

from factorlab.factors.base import Factor
from factorlab.transformations.dispersion import Dispersion
from factorlab.transformations.returns import LogReturn
from factorlab.utils import safe_divide, to_dataframe


class CarryFactor(Factor, ABC):
    """
    Abstract base class for all carry factors in FactorLab.

    This class provides a common framework for carry factor calculation,
    handling repetitive tasks like normalization, smoothing, and common
    pre-processing steps using the fit/transform design pattern.
    """

    def __init__(self,
                 spot_col: str,
                 rate_col: Optional[str] = None,
                 fwd_col: Optional[str] = None,
                 annualize: bool = True,
                 ann_factor: int = 365,
                 sign_flip: bool = True,
                 scale: bool = False,
                 scaling_method: str = 'std',
                 scaling_window: int = 365,
                 **kwargs):
        """
        Constructor. (Parameters are the same as original)
        """
        super().__init__(name=self.__class__.__name__,
                         description='Base class for carry factors.',
                         category='Carry')

        self.spot_col = spot_col
        self.rate_col = rate_col
        self.fwd_col = fwd_col
        self.annualize = annualize
        self.ann_factor = ann_factor
        self.sign_flip = sign_flip
        self.scale = scale
        self.scaling_method = scaling_method
        self.scaling_window = scaling_window
        self.kwargs = kwargs
        self.scaler: Optional[Dispersion] = None
        self.log_return_transformer: Optional[LogReturn] = None
        if self.scale:
            self.log_return_transformer = LogReturn(input_col=self.spot_col, output_col='ret', lags=1)
            self.scaler = Dispersion(
                input_col='ret',
                output_col='disp',
                method=self.scaling_method,
                window_type='rolling',
                window_size=self.scaling_window
            )

    @property
    def inputs(self) -> List[str]:
        """
        Required input columns.
        Override in subclasses as needed.
        """
        return [self.spot_col, self.fwd_col] if self.spot_col and self.fwd_col else [self.rate_col, self.spot_col]

    def _generate_name(self) -> str:
        """
        Generates a standardized name for the factor based on its parameters.
        """
        name_parts = [self.name]

        if self.scale:
            name_parts.append(f"scaled_{self.scaling_method}_{self.scaling_window}")

        return "_".join(name_parts)

    def fit(self, X: Union[pd.Series, pd.DataFrame], y: Optional[Union[pd.Series, pd.DataFrame]] = None):
        """
        Initializes and fits any internal transformers.

        Parameters
        ----------
        X : Union[pd.Series, pd.DataFrame]
            Input data (e.g., OHLCV data).

        Returns
        -------
        self
            The fitted transformer instance.
        """
        df_input = to_dataframe(X)
        self.validate_inputs(df_input)

        # scaling transformer
        if self.scale:
            # returns
            chg_df = self.log_return_transformer.compute(X)
            self.scaler.fit(chg_df)

        self._is_fitted = True
        return self

    @abstractmethod
    def _compute_carry(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Abstract method to be implemented by subclasses.

        This method must contain the unique logic for computing the raw carry factor.
        """
        raise NotImplementedError

    def transform(self, X: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
        """
        Public interface for applying the skew factor.
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
        Applies the full CarryFactor computation pipeline.

        Parameters
        ----------
        X : Union[pd.Series, pd.DataFrame]
            Input data containing the required columns.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the computed carry factor.
        """
        df = X

        # zero checks (critical for division)
        if self.spot_col in df.columns:
            df = df[df[self.spot_col] != 0]
        if self.fwd_col and self.fwd_col in df.columns:
            df = df[df[self.fwd_col] != 0]

        # compute carry
        carry_df = self._compute_carry(df)

        # scaling
        if self.scale:
            chg_df = self.log_return_transformer.compute(df)
            scaling_factor = self.scaler.transform(chg_df)

            carry_df = safe_divide(carry_df, scaling_factor.iloc[:, -1].to_frame('carry'))

        # annualize
        if self.annualize:
            carry_df = carry_df * np.sqrt(self.ann_factor)

        # sign flip if needed
        if self.sign_flip:
            carry_df *= -1

        # add to original df
        X[self._generate_name()] = carry_df.iloc[:, -1]

        return X

