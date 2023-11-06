import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from importlib import resources
from typing import Optional, Union
from scipy.stats import chi2_contingency, spearmanr, kendalltau, contingency
from sklearn.feature_selection import mutual_info_classif

from factorlab.time_series_analysis import linear_reg, fm_summary
from factorlab.transform import Transform

class Factor:
    """
    Screening methods for raw data or features.
    """
    def __init__(self,
                 factors: pd.DataFrame,
                 ret: pd.Series,
                 strategy: str = 'ts_ls',
                 factor_bins: int = 5,
                 target_bins: int = 2,
                 window_type: Optional[str] = 'expanding',
                 window_size: Optional[int] = 90,
                 ):
        """
        Constructor

        Parameters
        ----------
        factors: pd.Series or pd.DataFrame - Single or MultiIndex
            Dataframe with DatetimeIndex (level 0), tickers (level 1) and factors (cols).
        ret: pd.Series or pd.DataFrame - Single or MultiIndex
            Dataframe or series with DatetimeIndex (level 0), tickers (level 1) and returns (cols).
        strategy: str, {'ts_ls' 'ts_l', 'cs_ls', 'cs_l', default 'ts_ls'
            Time series or cross-sectional strategy, long/short or long-only.
        factor_bins: int, default 5
            Number of bins to create for factors.
        target_bins: int, default 2
            Number of bins to create for forward returns.
        window_type: str, {'fixed', 'expanding', 'rolling'}, default 'expanding'
            Window type for normalization.
        window_size: int
            Minimal number of observations to include in moving window (rolling or expanding).
        """
        self.factors = factors.astype(float)
        self.ret = ret.astype(float)
        self.strategy = strategy
        self.factor_bins = factor_bins
        self.target_bins = target_bins
        self.window_type = window_type
        self.window_size = window_size
        if isinstance(self.factors, pd.Series):
            self.factors = self.factors.to_frame()
        if factor_bins <= 1 or target_bins <= 1:
            raise ValueError("Number of bins must be larger than 1.")
