from __future__ import annotations
import pandas as pd
from typing import Optional, List, Union

from factorlab.features.base import Feature
from factorlab.utils import to_dataframe
from factorlab.signal_generation.time_series_analysis import TimeSeriesAnalysis as TSA


class Betas(Feature):
    """
    Computes the betas from regressing target_col on feature_col.

    Parameters
    ----------
    target_col : str
        The name of the column to be used as the target (dependent variable).
    feature_col : str
        The name of the column to be used as the feature (independent variable).
    model: str, default 'linear'
        The regression model to use.
    window_type : str, {'rolling', 'ewm'}, default "rolling"
        Window type for regression.
    window_size : int, default 60
        Window size for regression.
    **kwargs :
        Additional keyword arguments for the base class.

    Notes
    -----
    Instance variables match the parameters. Use the class constructor
    to set them, and access them directly via attributes if needed.
    """

    def __init__(self,
                 target_col: str,
                 feature_col: str,
                 output_col: str = 'beta',
                 model: str = 'linear',
                 window_type: str = 'rolling',
                 window_size: int = 60,
                 **kwargs):
        super().__init__(**kwargs)
        self.name = 'Beta',
        self.description = 'Computes the betas from regressing target_col on feature_col.',
        self.target_col = target_col
        self.feature_col = feature_col
        self.output_col = output_col
        self.model = model
        self.window_type = window_type
        self.window_size = window_size

    @property
    def inputs(self) -> List[str]:
        """
        Required input columns.
        """
        return [self.target_col, self.feature_col]

    def fit(self,
            X: Union[pd.Series, pd.DataFrame],
            y: Optional[Union[pd.Series, pd.DataFrame]] = None) -> 'Betas':
        """
        Fits the Betas transformation. For rolling regressions, this is typically stateless,
        but we use it to fit any internal, stateful components.
        """
        df_input = to_dataframe(X)
        self.validate_inputs(df_input)
        self._is_fitted = True
        return self

    def transform(self, X: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
        """
        Public interface for computing the residual.
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
        Computes the betas from regressing target_col on feature_col using the fitted configuration.

        Parameters
        ----------
        X : Union[pd.Series, pd.DataFrame]
            Input DataFrame containing the necessary columns.

        Returns
        -------
        pd.DataFrame
            A DataFrame with the computed betas.
        """
        df = X

        # remove tickers with insufficient data
        df = df.groupby(level='ticker').filter(lambda x: len(x) >= self.window_size)

        #  linear regression
        if self.model == 'linear':
            beta = TSA(df[self.target_col],
                       df[self.feature_col],
                       window_type=self.window_type,
                       window_size=self.window_size,
                       trend='c').linear_regression(output='params')

            beta = beta.iloc[:, 1:]
        else:
            raise NotImplementedError(f"Model {self.model} not implemented.")

        # add to df
        X[self.output_col] = beta

        return X
