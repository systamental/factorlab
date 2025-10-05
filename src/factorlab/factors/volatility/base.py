from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import List, Optional, Union

from factorlab.factors.base import Factor
from factorlab.utils import to_dataframe


class VolFactor(Factor, ABC):
    """
    Abstract base class for all vol factors in FactorLab.

    This class provides a common framework for vol factor calculation,
    handling repetitive tasks smoothing, and common
    pre-processing steps.

    Parameters
    ----------
    input_col : str
        Column name for returns.
    output_col : str
        Column name for the computed vol values.
    ann_factor : float, default 365
        Annualization factor for scaling the vol values.
    sign_flip : bool, default True
        If True, flips the sign of the computed skew values.
    window_type: str, {'rolling', 'ewm'}, default 'ewm'
        Type of rolling window to use for smoothing.
    window_size: int, default 30
        Rolling window size for smoothing.
    **kwargs:
        Additional keyword arguments for specific trend factor implementations.

    Notes
    -----
    Instance variables match the parameters. Use the class constructor to set
    them, and access them directly via attributes if needed.

    Examples
    --------
    >>> factor = Volatility(input_col='ret', sign_flip=True, smooth=True, window_type='ewm', window_size=30)
    >>> factor.input_col
    'returns'
    >>> factor.sign_flip
    True
    """

    def __init__(self,
                 input_col: str = 'ret',
                 output_col: str = 'vol',
                 annualize: bool = True,
                 ann_factor: float = 365,
                 sign_flip: bool = False,
                 window_type: str = "ewm",
                 window_size: int = 30,
                 **kwargs):
        super().__init__(name=self.__class__.__name__,
                         description='Base class for vol factors.',
                         category='Vol')

        self.input_col = input_col
        self.output_col = output_col
        self.annualize = annualize
        self.ann_factor = ann_factor
        self.sign_flip = sign_flip
        self.window_type = window_type
        self.window_size = window_size
        self.kwargs = kwargs
        self._log_transformer = None

    @property
    def inputs(self) -> List[str]:
        """
        Required input columns.
        Override in subclasses as needed.
        """
        return [self.input_col]

    def fit(self,
            X: Union[pd.Series, pd.DataFrame],
            y: Optional[Union[pd.Series, pd.DataFrame]] = None) -> 'VolFactor':
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
        self._is_fitted = True
        return self

    @abstractmethod
    def _compute_vol(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Abstract method to be implemented by subclasses.

        This method should contain the unique logic for computing the vol factor,
        without any normalization or smoothing.
        """
        raise NotImplementedError

    def _generate_name(self) -> str:
        """
        Generates a standardized name for the trend factor based on its parameters.
        """
        name_parts = [self.name]
        
        # add window 
        if hasattr(self, 'window_type') and self.window_type:
            name_parts.append(self.window_type)
        if hasattr(self, 'window_size') and self.window_size:
            name_parts.append(str(self.window_size))

        return "_".join(name_parts)

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
        Applies the full VolFactor computation pipeline.

        Parameters
        ----------
        X : Union[pd.Series, pd.DataFrame]
            Input data containing the required columns.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the computed vol factor.
        """
        df = X

        # compute vol
        vol_df = self._compute_vol(df)

        # annualize
        if self.annualize:
            vol_df = vol_df * np.sqrt(self.ann_factor)

        # sign flip if needed
        if self.sign_flip:
            vol_df *= -1
        
        df[self._generate_name()] = vol_df[self.output_col]

        return df
