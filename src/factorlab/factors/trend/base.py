from abc import ABC, abstractmethod
import pandas as pd
from typing import Optional, List, Union
from factorlab.factors.base import Factor
from factorlab.transformations.dispersion import Dispersion
from factorlab.transformations.smoothing import WindowSmoother
from factorlab.transformations.returns import Returns
from factorlab.utils import safe_divide, to_dataframe


class TrendFactor(Factor, ABC):
    """
    Abstract base class for trend factors in FactorLab.

    Provides a common framework for trend factor calculation, handling tasks
    like normalization, winsorization, and common pre-processing steps.

    Parameters
    ----------
    window_type : {'rolling', 'ewm'}, default 'ewm'
        Type of rolling window to use.
    window_size : int, default 30
        Rolling window size for calculations.
    scale : bool, default True
        Whether to scale the trend values.
    scale_method : {'std', 'variance', 'iqr', 'mad', 'range', 'atr'}, default 'std'
        Method to use for scaling.
    scaling_window : int, default 365
        Window size for scaling calculations.
    smooth : bool, default False
        Whether to apply smoothing to the price series.
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
    >>> factor = PriceMomentum(window_size=20, scale=True)
    >>> factor.window_size
    20
    >>> factor.scale
    True
    """
    def __init__(self,
                 input_col: str = 'close',
                 window_type: str = "ewm",
                 window_size: int = 30,
                 scale: bool = True,
                 scaling_method: str = 'std',
                 scaling_window: int = 365,
                 smooth: bool = False,
                 winsorize: Optional[float] = None,
                 **kwargs):
        super().__init__(name=self.__class__.__name__,
                         description='Base class for trend factors.',
                         category='Trend')

        self.input_col = input_col
        self.window_type = window_type
        self.window_size = window_size
        self.scale = scale
        self.scaling_method = scaling_method
        self.scaling_window = scaling_window
        self.smooth = smooth
        self.winsorize = winsorize
        self.kwargs = kwargs
        self.return_transformer: Optional[Returns] = None
        self.scaler: Optional[Dispersion] = None
        self.smoother: Optional[WindowSmoother] = None

        if self.scale:
            self.return_transformer = Returns(method='log', lags=1)
            self.scaler = Dispersion(
                input_col='ret',
                output_col='scaling_factor',
                method=self.scaling_method,
                window_type='rolling',
                window_size=self.scaling_window
            )

        if self.smooth:
            self.smoother = WindowSmoother(
                input_cols='trend',
                output_cols='trend_smooth',
                window_type=self.window_type,
                window_size=self.window_size
            )

    @property
    def inputs(self) -> List[str]:
        """
        Required input columns.
        Override in subclasses as needed.
        """
        return [self.input_col]

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

        # scaling
        if self.scale:
            self.return_transformer.fit(df_input)
            returns_data = self.return_transformer.transform(df_input)
            self.scaler.fit(returns_data)

        # smoothing
        if self.smooth:
            self.smoother.fit(df_input)

        self._is_fitted = True
        return self

    @abstractmethod
    def _compute_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Abstract method to be implemented by subclasses.

        This method should contain the unique logic for computing the trend factor,
        without any normalization or winsorization.
        """
        raise NotImplementedError

    def _generate_name(self) -> str:
        """
        Generates a standardized name for the trend factor based on its parameters.
        """
        name_parts = [self.name]

        # Add window size
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
        df = X

        # compute trend
        trend_df = self._compute_trend(df)

        # scaling
        if self.scale:
            scaling_factor = self.scaler.compute(df)
            trend_df = safe_divide(trend_df[['trend']], scaling_factor[['scaling_factor']])

        # smoothing
        if self.smooth:
            trend_df = self.smoother.transform(trend_df)

        if self.winsorize is not None:
            trend_df = trend_df.clip(-self.winsorize, self.winsorize)

        # add to original df
        X[self._generate_name()] = trend_df.iloc[:, -1]

        return X
