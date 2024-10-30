import pandas as pd
import numpy as np
from typing import Union, Optional
from factorlab.feature_engineering.transformations import Transform


class Target:
    """
    Target variable transformation class.
    """
    def __init__(self,
                 df: Union[pd.Series, pd.DataFrame],
                 strategy: str = 'ts',
                 bins: int = 3,
                 lead: int = 1,
                 vwap: bool = False,
                 normalize: bool = False,
                 power_transform: bool = False,
                 quantize: bool = False,
                 rank: bool = False,
                 window_type: str = 'rolling',
                 window_size: int = 360,
                 **kwargs
                 ):
        """
        Constructor

        Parameters
        ----------
        df: pd.Series or pd.DataFrame - Single or MultiIndex
            DataFrame MultiIndex with DatetimeIndex (level 0), ticker (level 1) and prices (cols).
        strategy: str, {'ts', 'cs'}, default 'ts'
            Time series (directional) or cross-sectional (market neutral) strategy.
        bins: int, default 3
            Number of bins to use for quantization.
        lead: int, default 1
            Number of periods to lead the target variable.
        method: str, {'simple', 'log', 'diff'}, default 'log'
            Method to compute price changes.
            simple: computes simple returns.
            log: computes log returns.
            diff: computes difference.
        vwap: bool, default False
            Computes volume-weighted average price.
        normalize: bool, default False
            Normalizes the target variable by dividing the price change by a measure of dispersion.
        power_transform: bool, default False
            Transforms the normalized target variable using a power transformation.
        quantize: bool, default False
            Quantizes the target variable into discrete bins.
        rank: bool, default False
            Ranks the target variable in ascending order.
        window_type: str, {'fixed', 'expanding', 'rolling', 'ewm'}, default 'expanding'
            Provide a window type. If None, all observations are used in the calculation.
        window_size: int, default 10
            Size of the moving window. This is the minimum number of observations used for the rolling or expanding
            window statistic.
        """
        self.df = df
        self.strategy = strategy
        self.bins = bins if bins > 1 else self._raise_value_error()
        self.lead = lead
        self.vwap = vwap
        self.normalize = normalize
        self.power_transform = power_transform
        self.quantize = quantize
        self.rank = rank
        self.window_type = window_type
        self.window_size = window_size
        self.kwargs = kwargs
        # self.price_chg = None
        # self.norm_price_chg = None
        # self.trans_price_chg = None
        # self.quantiles = None
        # self.rank = None
        self.target = None

    def _raise_value_error(self):
        raise ValueError(f"Number of bins must be larger than 1.")

    def compute_price_chg(self, method: str = 'log') -> Union[pd.Series, pd.DataFrame]:
        """
        Computes price change.

        Parameters
        ----------
        method: str, {'simple', 'log', 'diff'}, default 'log'
            Method to compute price changes.

        Returns
        -------
        price_chg: pd.Series or pd.DataFrame
            Price change.
        """
        # convert dtype to float
        self.df = self.df.astype(float).copy()

        # vwap
        if self.vwap:
            self.df = Transform(self.df).vwap()[['vwap']]

        # simple returns
        if method == 'simple':
            if isinstance(self.df.index, pd.MultiIndex):
                self.target = self.df.groupby(level=1).pct_change(self.lead, fill_method=None)
            else:
                self.target = self.df.pct_change(self.lead, fill_method=None)

        # log returns
        elif method == 'log':
            # remove negative values
            self.df[self.df <= 0] = np.nan
            if isinstance(self.df.index, pd.MultiIndex):
                self.target = np.log(self.df).groupby(level=1).diff(self.lead)
            else:
                self.target = np.log(self.df).diff(self.lead)

        # price chg
        else:
            if isinstance(self.df.index, pd.MultiIndex):
                self.target = self.df.groupby(level=1).diff(self.lead)
            else:
                self.target = self.df.diff(self.lead)

        # shift price chg forward
        if isinstance(self.df.index, pd.MultiIndex):
            self.target = self.target.groupby(level=1).shift(self.lead * -1).dropna(how='all').sort_index()
        else:
            self.target = self.target.shift(self.lead * -1).dropna(how='all').sort_index()

        return self.target

    def normalize_price_chg(self,
                            method: str = 'z-score',
                            centering: bool = False,
                            ts_norm: bool = False,
                            winsorize: Optional[int] = None
                            ) -> Union[pd.Series, pd.DataFrame]:
        """
        Normalizes price change.

        Parameters
        ---------
        method: str, {'z-score', 'iqr', 'mod_z', 'atr', 'min-max', 'percentile'}, default 'z-score'
            Normalization method.
            z-score: subtracts mean and divides by standard deviation.
            iqr: subtracts median and divides by interquartile range.
            mod_z: modified z-score using median absolute deviation.
            atr: subtracts mean and divides by average true range.
            min-max: rescales to values between 0 and 1 by subtracting the min and dividing by the range of values.
            percentile: converts values to percentiles.
        centering: bool, default False
            Centers values using the appropriate measure of central tendency for the normalization method. Otherwise,
            0 is used.
        ts_norm: bool, default False
            Normalizes factors over the time series before normalizing over the cross-section.
        winsorize: int, default None
            Winsorizes/clips values to between positive and negative values of specified integer.

        Returns
        -------
        norm_price_chg: pd.Series or pd.DataFrame
            Normalized price change.
        """
        # check method
        if method not in ['z-score', 'iqr', 'mod_z', 'atr', 'min-max', 'percentile']:
            raise ValueError('Invalid normalization method. Valid methods are: z-score, iqr, mod_z, atr, min-max, '
                             'percentile')

        # normalize price chg
        if self.strategy == 'cs' and ts_norm:
            price_chg_norm_ts = Transform(self.target).normalize(method=method,
                                                                 axis='ts',
                                                                 centering=centering,
                                                                 window_type=self.window_type,
                                                                 window_size=self.window_size,
                                                                 winsorize=winsorize)

            self.target = Transform(price_chg_norm_ts).normalize(method=method,
                                                                 axis='cs',
                                                                 centering=centering,
                                                                 window_type=self.window_type,
                                                                 window_size=self.window_size,
                                                                 winsorize=winsorize)

        else:
            self.target = Transform(self.target).normalize(method=method,
                                                           axis=self.strategy,
                                                           centering=centering,
                                                           window_type=self.window_type,
                                                           window_size=self.window_size,
                                                           winsorize=winsorize)

        # drop NaN values and sort
        self.target = self.target.dropna(how='all').sort_index()

        return self.target

    def transform(self, method: str = 'yeo-johnson') -> Union[pd.Series, pd.DataFrame]:
        """
        Transforms the normalized price change using a power transformation.

        Parameters
        ----------
        method: str, {'yeo-johnson', 'box-cox'}, default 'yeo-johnson'
            Power transformation method.
            yeo-johnson: works with positive and negative values.
            box-cox: works with positive values only.

        Returns
        -------
        trans_price_chg: pd.Series or pd.DataFrame
            Transformed price change.
        """
        self.target = Transform(self.target).power_transform(method=method,
                                                             axis=self.strategy,
                                                             window_type=self.window_type,
                                                             window_size=self.window_size
                                                             )

        # drop NaN values and sort
        self.target = self.target.dropna(how='all').sort_index()

        return self.target

    def quantize_target(self) -> Union[pd.Series, pd.DataFrame]:
        """
        Quantizes the target variable into discrete bins.
        """
        if self.bins <= 1 or self.bins is None:
            raise ValueError('Number of bins must be larger than 1. Please increase number of bins.')

        self.target = Transform(self.target).quantize(bins=self.bins,
                                                      axis=self.strategy,
                                                      window_type=self.window_type,
                                                      window_size=self.window_size)

        # drop NaN values and sort
        self.target = self.target.dropna(how='all').sort_index()

        return self.target

    def rank_target(self):
        """
        Ranks the target variable in ascending order.
        """
        self.target = Transform(self.target).rank(axis=self.strategy,
                                                  window_type=self.window_type,
                                                  window_size=self.window_size)

        # drop NaN values and sort
        self.target = self.target.dropna(how='all').sort_index()

        return self.target

    def compute_target(self,
                       method: str = 'log',
                       norm_method: str = 'z-score',
                       centering: bool = False,
                       ts_norm: bool = False,
                       winsorize: Optional[int] = None,
                       transform_method: str = 'yeo-johnson'
                       ):
        """
        Computes the target variable transformation.
        """
        # compute price change
        self.compute_price_chg(method=method)

        # normalize
        if self.normalize:
            self.normalize_price_chg(method=norm_method,
                                     centering=centering,
                                     ts_norm=ts_norm,
                                     winsorize=winsorize)

        # transform
        if self.power_transform:
            self.transform(method=transform_method)

        # quantize
        if self.quantize:
            self.quantize_target()

        # rank
        if self.rank:
            self.rank_target()

        # drop NaN values and sort
        self.target = self.target.dropna(how='all').sort_index()

        return self.target
