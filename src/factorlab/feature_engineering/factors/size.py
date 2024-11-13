from __future__ import annotations
import pandas as pd
from typing import Optional

from factorlab.feature_engineering.transformations import Transform


class Size:
    """
    Size factor.
    """
    def __init__(self,
                 df: pd.DataFrame,
                 size_metric: str,
                 log: bool = True,
                 ):
        """
        Constructor

        Parameters
        ----------
        df : pd.DataFrame
            Size data.
        size_metric : str
            Size metric to use.
        log : bool, optional
            Logarithm transformation. Default is True.
        """
        self.df = df.to_frame() if isinstance(df, pd.Series) else df
        self.size_metric = size_metric
        self.log = log
        self.size = None
        self.convert_to_multiindex()

    def convert_to_multiindex(self) -> pd.DataFrame:
        """
        Converts DataFrame to MultiIndex.

        Returns
        -------
        df: pd.DataFrame
            DataFrame with MultiIndex.
        """
        if not isinstance(self.df.index, pd.MultiIndex):
            self.df = self.df.stack()
            self.df.columns = [self.size_metric]

        return self.df

    def compute_size_factor(self,
                            smoothing: bool = False,
                            sm_window_type: str = 'rolling',
                            sm_window_size: int = 3,
                            sm_central_tendency: str = 'mean',
                            sm_window_fcn: Optional[str] = None,
                            ) -> pd.DataFrame:
        """
        Computes size factor.

        Parameters
        ----------
        smoothing : bool, optional
            Smoothing. Default is False.
        sm_window_type : str, optional
            Type of window for smoothing. Default is 'rolling'.
        sm_window_size : int, optional
            Number of observations in moving window for smoothing. Default is 3.
        sm_central_tendency : str, optional
            Measure of central tendency used for the smoothing rolling window. Default is 'mean'.
        sm_window_fcn : str, optional
            Provide a window function. If None, observations are equally-weighted in the rolling computation.
            See scipy.signal.windows for more information.

        Returns
        -------
        size: pd.DataFrame
            Size factor.
        """
        # name
        name = 'size'

        # log
        if self.log:
            self.size = Transform(self.df).log()

        # smoothing
        if smoothing:
            self.size = Transform(self.size).smooth(sm_window_size, window_type=sm_window_type,
                                                central_tendency=sm_central_tendency, window_fcn=sm_window_fcn)
            # name
            name = f"{name}_{sm_central_tendency}_{sm_window_size}"

        # size factor
        self.size = self.size[[self.size_metric]]
        self.size.columns = [name]

        return self.size
