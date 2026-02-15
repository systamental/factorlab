import pandas as pd
from typing import Union

from factorlab.signal_generation.base import BaseSignal
from factorlab.utils import to_dataframe


class Sign(BaseSignal):
    """
    Generates a discrete signal (-1, 0, 1) based on the sign of the input score.
    """

    def __init__(self,
                 input_col: str = 'score',
                 output_col: str = 'signal',
                 **kwargs):

        super().__init__(input_col=input_col, output_col=output_col, **kwargs)
        self.name = "SignSignal"
        self.description = "Generates a discrete trading signal from the sign of the score."
        self.input_col = input_col
        self.output_col = output_col

    def _compute_signal(self, X: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
        """
        Assigns a signal based on the sign of the score column.
        """
        df = X.copy()

        # 3. Initialize signal column to 0 (Neutral)
        df[self.output_col] = 0.0

        # 4. Assign Long Signal (1)
        df.loc[df[self.input_col] > 0, self.output_col] = 1.0

        # 5. Assign Short Signal (-1)
        df.loc[df[self.input_col] < 0, self.output_col] = -1.0

        return df[self.output_col]


class DiscreteZScoreSignal(BaseSignal):
    """
    Generates a discrete signal (-1, 0, 1) based on cross-sectional Z-score thresholds.
    """

    def __init__(self,
                 input_col: str = 'score',
                 long_z_score: float = 1.5,
                 short_z_score: float = -1.5,
                 output_col: str = 'signal',
                 **kwargs):

        super().__init__(input_col=input_col, output_col=output_col, **kwargs)
        self.name = "ZScoreSignal"
        self.description = "Generates a discrete trading signal from Z-score thresholds."
        self.input_col = input_col
        self.long_z_score = long_z_score
        self.short_z_score = short_z_score
        self.output_col = output_col

        if long_z_score <= short_z_score:
            raise ValueError("long_z_score must be greater than short_z_score.")

    def _compute_signal(self, X: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
        """
        Calculates the cross-sectional Z-score for the score column at each time step and assigns a signal.
        """
        df = X.copy()

        # 3. Initialize signal column to 0 (Neutral)
        df[self.output_col] = 0.0

        # 4. Assign Long Signal (1)
        df.loc[df[self.input_col] >= self.long_z_score, self.output_col] = 1.0

        # 5. Assign Short Signal (-1)
        df.loc[df[self.input_col] <= self.short_z_score, self.output_col] = -1.0

        return df[self.output_col]


class DiscreteQuantileSignal(BaseSignal):
    """
    Generates a discrete signal (-1, 0, 1) based on cross-sectional or time-series quantile thresholds.
    """

    def __init__(self,
                 input_col: str = 'quantile',
                 output_col: str = 'signal',
                 axis: str = 'ts',
                 **kwargs):

        super().__init__(input_col=input_col, output_col=output_col, axis=axis, **kwargs)
        self.name = "QuantileSignal"
        self.description = "Generates a discrete trading signal from quantile thresholds."
        self.score_col = input_col
        self.output_col = output_col
        self.axis = axis

    def _compute_signal(self, X: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
        """
        Calculates quantile thresholds along the specified axis (CS or TS) and assigns a signal.
        """
        df = X.copy()

        top_quantile = df[self.input_col].max()
        bottom_quantile = df[self.input_col].min()

        # signal col
        df[self.output_col] = 0.0

        # long signal
        df.loc[df[self.input_col] == top_quantile, self.output_col] = 1.0

        # short signal
        df.loc[df[self.input_col] == bottom_quantile, self.output_col] = -1.0

        return df[self.output_col]


class DiscreteRankSignal(BaseSignal):
    """
    Generates a discrete signal (-1, 0, 1) based on cross-sectional or time-series quantile thresholds.

    Parameters
    ----------
    input_col : str
        The name of the input column containing the rank values.
    n_assets : int
        The number of top/bottom assets to assign long/short signals.
    output_col : str
        The name of the output column to store the generated signals.
    axis : str, {'ts', 'cs'}
        The axis along which to compute the ranks. If 'ts', computes across time series;
        if 'cs', computes across cross-sections.

    """

    def __init__(self,
                 input_col: str = 'rank',
                 n_assets: int = 10,
                 output_col: str = 'signal',
                 axis: str = 'ts',
                 **kwargs):

        super().__init__(input_col=input_col, output_col=output_col, axis=axis, **kwargs)
        self.name = "RankSignal"
        self.description = "Generates a discrete trading signal from rank thresholds."
        self.score_col = input_col
        self.n_assets = n_assets
        self.output_col = output_col
        self.axis = axis

    def _compute_signal(self, X: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
        """
        Calculates quantile thresholds along the specified axis (CS or TS) and assigns a signal.
        """
        if not self._is_fitted:
            raise RuntimeError(f"Transform '{self.name}' must be fitted before calling transform()")

        df = to_dataframe(X).copy()
        self.validate_inputs(df)

        if self.axis == 'ts':
            raise NotImplementedError(f"Time-series axis not implemented for RankSignal. Use 'cs' axis instead or "
                                      f"QuantileSignal or ZScoreSignal for time-series signals.")
        else:
            # cs axis
            # check cs count, drop if not enough assets
            df = df[df.groupby(level=0).transform('count') > 2 * self.n_assets].dropna()

            df['lower_thresh'] = df[self.input_col].groupby(level=0).transform('min') + self.n_assets
            df['upper_thresh'] = df[self.input_col].groupby(level=0).transform('max') - self.n_assets

            # signal col
            df[self.output_col] = 0.0
            df.loc[df[self.input_col] >= df['upper_thresh'], self.output_col] = 1.0
            df.loc[df[self.input_col] <= df['lower_thresh'], self.output_col] = -1.0

        return df.drop(columns=['lower_thresh', 'upper_thresh'])[self.output_col]
