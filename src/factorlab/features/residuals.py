from __future__ import annotations
import pandas as pd
from typing import Optional, List, Union

from factorlab.features.base import Feature
from factorlab.utils import to_dataframe
from factorlab.signal_generation.time_series_analysis import TimeSeriesAnalysis as TSA


class Residuals(Feature):
    """
    Computes the residuals from regressing target_col on feature_col.

    Parameters
    ----------
    target_col : str
        The name of the column to be used as the target (dependent variable).
    feature_col : str
        The name of the column to be used as the feature (independent variable).
    output_col : str, default 'idio_ret'
        The name of the output column to store the computed idiosyncratic returns.
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
                 output_col: str = 'residual',
                 model: str = 'linear',
                 window_type: str = 'rolling',
                 window_size: int = 360,
                 **kwargs):
        super().__init__(**kwargs)
        self.name = 'Residual',
        self.description = 'Computes the residual from regressing target_col on feature_col.',
        self.target_col = target_col
        self.feature_col = feature_col
        self.output_col = output_col
        self.model = model
        self.window_type = window_type
        self.window_size = window_size

    @property
    def inputs(self) -> List[str]:
        """Required input columns."""
        return [self.target_col, self.feature_col]

    def fit(self,
            X: Union[pd.Series, pd.DataFrame],
            y: Optional[Union[pd.Series, pd.DataFrame]] = None) -> 'Residuals':
        """
        Fits the Residuals transformation.
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
        Applies the residual calculation using the fitted configuration.
        """
        df = X

        # filter min window size for TSA
        df = df.groupby(level='ticker').filter(lambda x: len(x) >= self.window_size)

        # compute residuals
        if self.model == 'linear':
            resid = TSA(df[self.target_col],
                        df[self.feature_col],
                        window_type=self.window_type,
                        window_size=self.window_size,
                        trend='c').linear_regression(output='resid')
        else:
            raise NotImplementedError(f"Model {self.model} not implemented.")

        # Convert to df
        X[self.output_col] = resid

        return X


class IdiosyncraticReturns(Feature):
    """
    Computes the idiosyncratic returns from regressing asset returns on the market returns.

    This feature is commonly used in financial analysis to isolate the portion of an asset's returns
    that is not explained by market movements. It represents the asset-specific risk and return,
    independent of broader market trends used for factors like IVOL and ISKEW.

    Parameters
    ----------
    return_col : str
        The name of the column to be used as the asset returns.
    factor_cols: str or list, default 'market'
        Column name(s) of the systematic factor returns (e.g., 'MKT-RF').
    output_col : str, default 'idio_ret'
        The name of the output column to store the computed idiosyncratic returns.
    model: str, default 'linear'
        The regression model to use.
    incl_alpha : bool, default True
        If True, includes the alpha (intercept term) to the residual returns. This separates the returns into
        1) beta returns and 2) idiosyncratic returns (alpha + residuals).
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
                 return_col: str,
                 factor_cols: Union[str, List[str]],
                 output_col: str = 'idio_ret',
                 model: str = 'linear',
                 incl_alpha: bool = True,
                 window_type: str = 'rolling',
                 window_size: int = 30,
                 **kwargs):
        super().__init__(**kwargs)
        self.name = 'IdiosyncraticReturns',
        self.description = 'Computes the idiosyncratic returns from regressing asset returns on the market returns.',
        self.return_col = return_col
        self.factor_cols = factor_cols if isinstance(factor_cols, list) else [factor_cols]
        self.output_col = output_col
        self.model = model
        self.incl_alpha = incl_alpha
        self.window_type = window_type
        self.window_size = window_size

    @property
    def inputs(self) -> List[str]:
        """Required input columns."""
        return [self.return_col] + self.factor_cols

    def fit(self,
            X: Union[pd.Series, pd.DataFrame],
            y: Optional[Union[pd.Series, pd.DataFrame]] = None) -> 'IdiosyncraticReturns':
        """
        Fits the IdiosyncraticReturns transformation (primarily stateless).
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
        Applies the idiosyncratic return calculation.
        """
        df = X

        # Ensure minimum window size for TSA
        df = df.groupby(level='ticker').filter(lambda x: len(x) >= self.window_size)

        # fit linear regression
        if self.model == 'linear':
            if self.incl_alpha:
                beta = TSA(df[self.return_col],
                           df[self.factor_cols],
                           window_type=self.window_type,
                           window_size=self.window_size,
                           trend='c').linear_regression(output='params')
                # compute alpha returns
                beta = beta.iloc[:, 1:]
                beta_ret = beta.squeeze() * df[self.return_col].squeeze()
                idio_ret = df[self.return_col].squeeze() - beta_ret
            else:
                idio_ret = TSA(df[self.return_col],
                               df[self.factor_cols],
                               window_type=self.window_type,
                               window_size=self.window_size,
                               trend='c').linear_regression(output='resid')
        else:
            raise NotImplementedError(f"Model {self.model} not implemented.")

        # add to df
        X[self.output_col] = idio_ret

        return X
