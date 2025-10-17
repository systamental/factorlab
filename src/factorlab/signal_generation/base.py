import pandas as pd
from abc import abstractmethod
from typing import Union, List

from factorlab.core.base_transform import BaseTransform
from factorlab.utils import to_dataframe


class BaseSignal(BaseTransform):
    """
    Base class for all signal generation methods, discrete or continuous.
    Enforces the required input parameters and BaseTransform interface.

    Parameters
    ----------
    input_col : str
        The name of the input column containing the raw factor score or forecast.
    output_col : str, default 'signal'
        The name of the output column to store the generated signal.
    axis : str, default 'ts'
        The axis along which to compute the signal: 'cs' for cross-sectional,
        'ts' for time-series.
    lags : int, optional
        The number of time periods to lag the signal. If None, no lag is applied.
    leverage : float, optional
        A scaling factor to apply to the final signal. If None, no scaling is applied.
    direction : str, optional
        If 'long', only positive signals are retained; if 'short', only negative
        signals are retained. If None, both long and short signals are kept.
    """

    def __init__(self,
                 input_col: str,
                 output_col: str = 'signal',
                 axis: str = 'ts',
                 lags: int = None,
                 leverage: float = None,
                 direction: str = None
                 ):
        super().__init__(name="BaseSignal", description="Generates a trading signal from a score column.")

        self.input_col = input_col
        self.output_col = output_col
        self.axis = axis
        self.lags = lags
        self.leverage = leverage
        self.direction = direction

        if self.axis not in {'cs', 'ts'}:
            raise ValueError("Axis must be 'cs' (cross-sectional) or 'ts' (time-series).")
        if self.direction is not None and self.direction not in ['long', 'short']:
            raise ValueError("Direction must be 'long' or 'short'.")

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

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Applies the full Signal computation pipeline."""
        df = X.copy()

        # compute the raw signal (must be implemented by subclass)
        signal_df = self._compute_signal(df)

        # post-processing layers
        signal_df = self._apply_lags(signal_df)
        signal_df = self._apply_direction(signal_df)
        signal_df = self._apply_leverage(signal_df)

        # 3. add final signal column to the original DataFrame
        X[self.output_col] = signal_df.squeeze()

        return X

    def _apply_lags(self, signal_df: pd.DataFrame) -> pd.DataFrame:
        """Applies time lags to the signal column."""
        if self.lags is not None and self.lags > 0:
            if isinstance(signal_df.index, pd.MultiIndex):
                # Lag groups by asset (level=1)
                return signal_df.groupby(level=1).shift(self.lags)
            else:
                return signal_df.shift(self.lags)
        return signal_df

    def _apply_direction(self, signal_df: pd.DataFrame) -> pd.DataFrame:
        """Filters signals based on the 'long' or 'short' direction."""
        if self.direction == 'long':
            signal_df.clip(lower=0, inplace=True)
        elif self.direction == 'short':
            signal_df.clip(upper=0, inplace=True)
        return signal_df

    def _apply_leverage(self, signal_df: pd.DataFrame) -> pd.DataFrame:
        """Scales the signals by the leverage factor."""
        if self.leverage is not None:
            return signal_df * self.leverage
        return signal_df
