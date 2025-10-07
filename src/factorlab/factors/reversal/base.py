from abc import ABC, abstractmethod
import pandas as pd
from typing import Optional, List, Union

from factorlab.factors.base import Factor
from factorlab.transformations.dispersion import Dispersion
from factorlab.transformations.smoothing import WindowSmoother
from factorlab.transformations.returns import Returns
from factorlab.utils import to_dataframe


class ReversalFactor(Factor, ABC):
    """
    Abstract base class for reversal factors in FactorLab.

    Provides a common framework for reversal factor calculation, handling tasks
    like normalization, winsorization, and common pre-processing steps.

    Parameters
    ----------
    input_col : str, default 'close'
        Column name for price data.
    central_tendency : {'mean', 'median'}, default 'mean'
        Central tendency measure for smoothing.
    window_type : {'rolling', 'ewm'}, default 'ewm'
        Type of rolling window to use.
    window_size : int, default 30
        Rolling window size for calculations.
    min_periods : int, optional
        Minimum periods required for rolling calculations. Defaults to window_size if None.
    sign_flip : bool, default True
        Whether to flip the sign of the computed reversal values.
    winsorize : float, optional
        Winsorization threshold if set.
    **kwargs :
        Additional keyword arguments for specific trend factor implementations.

    Notes
    -----
    Instance variables match the parameters. Use the class constructor to set
    them, and access them directly via attributes if needed.

    Examples
    --------
    >>> factor = LinearDeviation(window_size=20, scale=True)
    >>> factor.window_size
    20
    >>> factor.scale
    True
    """
    def __init__(self,
                 input_col: str = 'close',
                 central_tendency: str = 'mean',
                 window_type: str = "ewm",
                 window_size: int = 30,
                 min_periods: Optional[int] = None,
                 sign_flip: bool = True,
                 winsorize: Optional[float] = None,
                 **kwargs):
        super().__init__(name=self.__class__.__name__,
                         description='Base class for reversal factors.',
                         category='Reversal')

        self.input_col = input_col
        self.central_tendency = central_tendency
        self.window_type = window_type
        self.window_size = window_size
        self.min_periods = min_periods if min_periods is not None else window_size
        self.sign_flip = sign_flip
        self.winsorize = winsorize
        self.kwargs = kwargs
        self.return_transformer: Optional[Returns] = None
        self.scaler: Optional[Dispersion] = None
        self.smoother: Optional[WindowSmoother] = None

    @property
    def inputs(self) -> List[str]:
        """
        Required input columns.
        Override in subclasses as needed.
        """
        return ['close']

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
        self._is_fitted = True
        return self

    @abstractmethod
    def _compute_reversal(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Abstract method to be implemented by subclasses.

        This method should contain the unique logic for computing the reversal factor,
        without any normalization or winsorization.
        """
        raise NotImplementedError

    def _generate_name(self) -> str:
        """
        Generates a standardized name for the factor based on its parameters.
        """
        name_parts = [self.name]

        # Add window size
        if hasattr(self, 'window_size') and self.window_size:
            name_parts.append(str(self.window_size))

        return "_".join(name_parts)

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
        Applies the full ReversalFactor computation pipeline.

        Parameters
        ----------
        X : Union[pd.Series, pd.DataFrame]
            Input data containing the required columns.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the computed reversal factor.
        """
        df = X

        # compute trend
        rev_df = self._compute_reversal(df)

        # flip sign if needed
        if self.sign_flip:
            rev_df = rev_df * -1

        if self.winsorize is not None:
            rev_df = rev_df.clip(-self.winsorize, self.winsorize)

        # add to original df
        X[self._generate_name()] = rev_df.iloc[:, -1]

        return X
