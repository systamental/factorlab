from abc import ABC, abstractmethod
import pandas as pd
from typing import List, Union, Optional
from factorlab.factors.base import Factor
from factorlab.transformations.smoothing import WindowSmoother
from factorlab.utils import to_dataframe


class SkewFactor(Factor, ABC):
    """
    Abstract base class for all skew factors in FactorLab.

    This class provides a common framework for skew factor calculation,
    handling repetitive tasks smoothing, and common
    pre-processing steps.

    Parameters
    ----------
    return_col : str
        Column name for returns.
    sign_flip : bool, default True
        If True, flips the sign of the computed skew values.
    window_type : str, {'rolling', 'expanding'}, default 'rolling'
        Type of rolling window to use.
    window_size : int, default 30
        Rolling window size for calculations.
    smooth: bool, default True
        Whether to apply smoothing to the returns series.
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

    """

    def __init__(self,
                 return_col: str,
                 sign_flip: bool = False,
                 window_type: str = 'rolling',
                 window_size: int = 30,
                 smooth: bool = False,
                 central_tendency: str = 'mean',
                 smoothing_method: str = "ewm",
                 smoothing_window: int = 5,
                 **kwargs):

        super().__init__(name=self.__class__.__name__,
                         description='Base class for skew factors.',
                         category='Skew')
        
        self.return_col = return_col
        self.sign_flip = sign_flip
        self.window_type = window_type
        self.window_size = window_size
        self.smooth = smooth
        self.central_tendency = central_tendency
        self.smoothing_method = smoothing_method
        self.smoothing_window = smoothing_window
        self.kwargs = kwargs
        self.smoother = None

    @property
    def inputs(self) -> List[str]:
        """
        Required input columns.
        Override in subclasses as needed.
        """
        return [self.return_col]

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

        # smooth transform
        if self.smooth:
            self.smoother = WindowSmoother(input_cols='skew',
                                           output_cols=f"skew_smoothed",
                                           window_type=self.smoothing_method,
                                           window_size=self.smoothing_window,
                                           central_tendency=self.central_tendency)
            self.smoother.fit(X)

        self._is_fitted = True
        return self

    @abstractmethod
    def _compute_skew(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Abstract method to be implemented by subclasses.

        This method should contain the unique logic for computing the skew factor,
        without any normalization or smoothing.
        """
        raise NotImplementedError

    def _generate_name(self) -> str:
        """
        Generates a standardized name for the trend factor based on its parameters.
        """
        name_parts = [self.name]

        # add window size
        if hasattr(self, 'window_size') and self.window_size:
            name_parts.append(str(self.window_size))

        if self.smooth:
            name_parts.append(f"smooth_{self.smoothing_method}_{self.smoothing_window}_{self.central_tendency}")

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
        Transform the input data to compute the skew factor values.

        This method MUST be implemented by all subclasses and contains the
        core skew calculation logic.

        Parameters
        ----------
        X : Union[pd.Series, pd.DataFrame]
            Input data (e.g., OHLCV data).

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the computed skew factor values.
        """
        df = X

        # compute skew
        skew_df = self._compute_skew(df)

        # smoothing
        if self.smooth:
            skew_df = self.smoother.transform(skew_df)

        # sign flip
        if self.sign_flip:
            skew_df *= -1

        # add to original df
        X[self._generate_name()] = skew_df.iloc[:, -1]

        return X
