from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Union, Optional

from factorlab.signal_generation.time_series_analysis import TimeSeriesAnalysis as TSA
from factorlab.feature_engineering.transformations import Transform


class Value:
    """
    Value factor.
    """
    def __init__(self,
                 df: pd.DataFrame,
                 ratio: Optional[str] = None,
                 method: str = 'ratio',
                 value_fcn: Optional[str] = None,
                 log: bool = True,
                 ts_norm: bool = False,
                 norm_method: str = 'z-score',
                 smoothing: Optional[str] = None,
                 sm_central_tendency: str = 'mean',
                 sm_window_size: int = 7,
                 res_window_type: str = 'expanding',
                 res_window_size: int = None,
                 ):
        """
        Constructor

        Parameters
        ----------
        df: pd.DataFrame - MultiIndex
            DataFrame MultiIndex with DatetimeIndex (level 0), ticker (level 1) and prices (cols).
        method: str, {'ratio', 'lin_reg'}, default 'ratio'
            Method used to compute value factor.
            Ratio computes a ratio of market cap to value metric.
            Residual estimates a residual by regressing market cap on the value metric.
        value_fcn: str, {'Metcalfe', 'Zipf', 'Metcalfe_gen, 'Metcalfe_sqrt', 'Sardoff'}, default 'Metcalfe'
            Function that maps the relationship between the value metric and the price. Defaults to Sardoff.
        ratio: str, default None
            Name of the column to use for ratio.
        log: bool, default False
            Converts to log price.
        ts_norm: bool, default False
            Normalize factor over the time series.
        norm_method: str, {'z-score', 'cdf', iqr', 'mod_z', 'min-max', 'percentile'}, default 'z-score'
            Method to use for normalization.
        smoothing: str, {'ewm', 'rolling'}, default None
            Smoothing method to use.
        sm_window_size: int, default 5
            Length of lookback window for smoothing.
        sm_central_tendency: str, {'mean', 'median'}, default 'mean'
            Central tendency to use for smoothing.
        res_window_type: str, default 'expanding'
            Window type to use for regression.
        res_window_size: int, default 7
            Number of observations in moving window.
        """
        self.df = df.to_frame() if isinstance(df, pd.Series) else df
        self.method = method
        self.value_fcn = value_fcn
        self.ratio = ratio
        self.log = log
        self.ts_norm = ts_norm
        self.norm_method = norm_method
        self.smoothing = smoothing
        self.sm_central_tendency = sm_central_tendency
        self.sm_window_size = sm_window_size
        self.res_window_type = res_window_type
        self.res_window_size = res_window_size
        self.resid = None
        self.value_factor = None
        self.check_fields()
        self.convert_to_multiindex()

    def check_fields(self) -> None:
        """
        Checks if required fields are in dataframe.

        Returns
        -------
        None
        """
        # check fields
        if all([col not in self.df.columns for col in ['ratio', 'spot', 'mkt_cap']]):
            raise ValueError("'spot' price, 'mkt_cap' or 'ratio' series must be provided to compute value factor.")

    def convert_to_multiindex(self) -> pd.DataFrame:
        """
        Converts DataFrame to MultiIndex.

        Returns
        -------
        df: pd.DataFrame
            DataFrame with MultiIndex.
        """
        if not isinstance(self.df.index, pd.MultiIndex):
            self.df = self.df.stack(future_stack=True)

        return self.df

    def remove_empty_cols(self, price: str = 'mkt_cap', value_metric: str = 'add_act') -> pd.DataFrame:
        """
        Removes empty columns from dataframe.

        Parameters
        ----------
        price: str
            Name of the column to use for price.
        value_metric: str
            Name of the column to use for value metric.

        Returns
        -------
        df: pd.DataFrame
            DataFrame with DatetimeIndex (level 0), ticker (level 1) and prices (cols).
        """
        # check value metric
        if value_metric not in self.df.columns:
            raise ValueError("A value metric series must be provided to compute value factor.")
        else:
            self.df = self.df.loc[:, [price, value_metric]]

        # remove empty cols
        self.df = self.df.unstack().dropna(how='all', axis=1).stack(future_stack=True)

        # keep common tickers
        tickers = list(set(self.df[price].unstack().columns).intersection(self.df[value_metric].unstack().columns))
        self.df = self.df.loc[pd.IndexSlice[:, tickers], [price, value_metric]].dropna().sort_index()

        return self.df

    def value_function(self, value_metric: str) -> None:
        """
        Computes the value function.

        Parameters
        ----------
        value_metric: str
            Name of the column to use for value metric.

        Returns
        -------
        df: pd.DataFrame
            DataFrame with DatetimeIndex (level 0), ticker (level 1) and values (cols).

        """
        # value fcn
        if self.value_fcn is not None:
            if self.value_fcn == 'Metcalfe':
                self.df[value_metric] = self.df[value_metric] ** 2
            elif self.value_fcn == 'Zipf':
                self.df[value_metric] = self.df[value_metric] * np.log(self.df[value_metric])
            elif self.value_fcn == 'Metcalfe_gen':
                self.df[value_metric] = self.df[value_metric] ** 1.5
            elif self.value_fcn == 'Metcalfe_sqrt':
                self.df[value_metric] = np.sqrt(self.df[value_metric] ** 2)

        return self.df

    def compute_ratio(self, price: str, value_metric: str) -> pd.DataFrame:
        """
        Computes ratio of market cap to value metric.

        Parameters
        ----------
        price: str
            Name of the column to use for price.
        value_metric: str
            Name of the column to use for value metric.

        Returns
        -------
        ratio: pd.DataFrame
            DataFrame with DatetimeIndex (level 0), tickers (level 1) and ratio (cols).
        """
        # check if ratio is already provided
        if self.ratio is None:

            # remove empty cols
            self.remove_empty_cols(price, value_metric)
            # value fcn
            self.value_function(value_metric)
            # log
            if self.log:
                self.df = Transform(self.df).log()
            # ratio
            self.ratio = self.df[value_metric].divide(self.df[price])

        return self.ratio

    def compute_residual(self, price: str, value_metric: str) -> pd.DataFrame:
        """
        Computes residual of market cap to value metric.

        Parameters
        ----------
        price: str
            Name of the column to use for price.
        value_metric: str
            Name of the column to use for value metric.

        Returns
        -------
        resid: pd.DataFrame
            DataFrame with DatetimeIndex (level 0), tickers (level 1) and residual (cols).
        """
        # remove empty cols
        self.remove_empty_cols(price, value_metric)

        # window size
        if self.res_window_size is None:
            self.res_window_size = int(self.df.unstack().shape[0] / 3)

        # value fcn
        self.value_function(value_metric)

        # log
        if self.log:
            self.df = Transform(self.df).log()

        # fit linear regression
        self.resid = TSA(self.df[price], self.df[value_metric], window_type=self.res_window_type, trend='c',
                         window_size=self.res_window_size).linear_regression(output='resid')

        return self.resid

    def name_factor(self, method: str) -> str:
        """
        Returns factor name.

        Parameters
        ----------
        method: str
            Name of method used to compute value factor.

        Returns
        -------
        names: str
            Factor name.
        """
        # method
        if self.method == 'ratio':
            name = 'ratio'
        else:
            name = 'resid'
        # norm
        if self.ts_norm:
            name = f"{name}_{self.norm_method}"
        # smoothing
        if self.smoothing is not None:
            name = f"{name}_{self.smoothing}_{self.sm_window_size}"
        # add method
        name = f"{method}_{name}"

        return name

    def compute_value_factor(self, price: str, value_metric: str, name: str) -> pd.DataFrame:
        """
        Computes value factor.

        Parameters
        ----------
        price: str
            Name of the column to use for price.
        value_metric: str
            Name of the column to use for value metric.
        name: str
            Name of the value factor.

        Returns
        -------
        value_factor: pd.DataFrame
            DataFrame with DatetimeIndex (level 0), tickers (level 1) and value factor (cols).
        """
        # compute value factor
        if self.method == 'ratio':  # ratio
            self.value_factor = self.compute_ratio(price, value_metric)
        else:
            # fit linear regression
            self.value_factor = self.compute_residual(price, value_metric)

        # normalize
        if self.ts_norm:
            self.value_factor = Transform(self.value_factor).normalize(axis='ts', method=self.norm_method,
                                                                       window_type='expanding')

        # smoothing
        if self.smoothing is not None:
            self.value_factor = Transform(self.value_factor).smooth(self.sm_window_size, window_type=self.smoothing,
                                                               central_tendency=self.sm_central_tendency)

        # name factor
        factor_name = self.name_factor(name)

        # return df
        if isinstance(self.value_factor, pd.Series):
            self.value_factor = self.value_factor.to_frame(factor_name).dropna().sort_index()
        else:
            self.value_factor = self.value_factor.dropna().sort_index()
            self.value_factor.columns = [factor_name]

        return self.value_factor

    def nvt(self,
            price: str = 'mkt_cap',
            value_metric: str = 'tfr_val_usd',
            name: str = 'nvt'
            ) -> Union[pd.Series, pd.DataFrame]:
        """
        Computes the network value to transactions value factor.

        Parameters
        ----------
        price: str
            Name of the column to use for price.
        value_metric: str
            Name of the column to use for value metric.
        name: str
            Name of the value factor.

        Returns
        -------
        nvt: pd.Series or pd.DataFrame
            Series or DataFrame with DatetimeIndex (level 0), tickers (level 1) and NVT value factor (cols).
        """
        # compute value factor
        nvt = self.compute_value_factor(price, value_metric, name)

        return nvt

    def nvm(self,
            price: str = 'mkt_cap',
            value_metric: str = 'add_act',
            name: str = 'nvm'
            ) -> Union[pd.Series, pd.DataFrame]:
        """
        Computes the network value to transactions value factor.

        Parameters
        ----------
        price: str, default 'mkt_cap'
            Name of the column to use for price.
        value_metric: str, default 'add_act'
            Name of the column to use for value metric.
        name: str, default 'nvm'
            Name of the value factor.

        Returns
        -------
        nvm: pd.Series or pd.DataFrame
            Series or DataFrame with DatetimeIndex (level 0), tickers (level 1) and NVM value factor (cols).
        """
        # compute value factor
        nvm = self.compute_value_factor(price, value_metric, name)

        return nvm

    def nvsf(self,
             price: str = 'mkt_cap',
             value_metric: str = 'supply_circ',
             name: str = 'nvsf'
             ) -> Union[pd.Series, pd.DataFrame]:

        # set log to false
        if self.log:
            self.log = False

        # sf ratio
        self.df['sf'] = (self.df[value_metric].unstack().diff(periods=365) /
                         self.df[value_metric].unstack()).stack(future_stack=True) * -1
        value_metric = 'sf'
        # compute value factor
        nvsf = self.compute_value_factor(price, value_metric, name)

        return nvsf

    def nvc(self,
            price: str = 'mkt_cap',
            value_metric: str = 'hashrate',
            name: str = 'nvc'
            ) -> Union[pd.Series, pd.DataFrame]:
        """
        Computes the Network Value to Production Costs ratio.

        Parameters
        ----------
        price: str, default 'mkt_cap'
            Name of the column to use for price.
        value_metric: str, default 'hashrate'
            Name of the column to use for
        name: str, default 'nvc'
            Name of the value factor.

        Returns
        -------
        nvc: pd.Series or pd.DataFrame
            Series or DataFrame with DatetimeIndex (level 0), tickers (level 1) and NVC value factor (cols).
        """
        # compute value factor
        nvc = self.compute_value_factor(price, value_metric, name)

        return nvc

    def npm(self,
            price: str = 'mkt_cap',
            lookback: int = 365,
            name: str = 'npm'
            ) -> Union[pd.Series, pd.DataFrame]:
        """

        Parameters
        ----------
        price: str, default 'mkt_cap'
            Name of the column to use for price series.
        lookback: int, default 365
            Lookback period for price momentum.
        name: str, default 'npm'
            Name of the value factor.

        Returns
        -------

        """
        # log
        log_price = Transform(self.df[price]).log()
        # negative price mom
        npm = log_price.groupby(level=1).diff(lookback).to_frame() * -1
        # rename
        npm.columns = [f"{name}_{lookback}"]

        return npm
