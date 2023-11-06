from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Union, Optional

from factorlab.time_series_analysis import linear_reg
from factorlab.transform import Transform


class Quality:
    """
    Quality factor.
    """
    def __init__(self,
                 df: pd.DataFrame,
                 smoothing: str = 'smw',
                 window_type: str = 'rolling',
                 window_size: int = 7
                 ):
        """
        Constructor

        Parameters
        ----------
        df: pd.DataFrame - MultiIndex
            DataFrame MultiIndex with DatetimeIndex (level 0), ticker (level 1) and market and/or on-chain data (cols).
        log: bool, default False
            Converts to log price.
        smoothing: str, {'median', 'smw', 'ewm'}, default None
            Smoothing method to use.
        method: str, {'ratio', 'lin_reg'}, default 'ratio'
            Method used to compute value factor.
        window_type: str, default 'expanding'
            Window type to use for regression.
        window_size: int, default 7
            Number of observations in moving window.
        """
        self.df = df.astype(float)
        self.smoothing = smoothing
        self.window_type = window_type
        self.window_size = window_size

    def network_size(self) -> pd.DataFrame:
        """
        Computes the cross-sectional rank of network size values.

        Parameters
        ----------
        None

        Returns
        -------
        df: pd.Series or pd.DataFrame - MultiIndex
            Series or DataFrame with DatetimeIndex (level 0), tickers (level 1) and network size rank (cols).
        """
        # rank cross-section
        df = self.df.groupby('date').rank()

        return df

    def network_growth(self) -> pd.DataFrame:
        """
        Computes the network growth over an n-period lookback window.

        Parameters
        ----------
        None

        Returns
        -------
        df: pd.Series or pd.DataFrame - MultiIndex
            Series or DataFrame with DatetimeIndex (level 0), tickers (level 1) and network size rank (cols).
        """
        df = self.df

        # smoothing
        if self.smoothing is not None:
            df = Transform(df).smooth(lookback=self.window_size, method=self.smoothing)

        # growth
        df = np.log(df).groupby('ticker').diff(periods=self.window_size)

        return df.sort_index()


    def network_persistence(self) -> pd.DataFrame:
        """

        Returns
        -------
        pers_df: pd.DataFrame or pd.Series - MultiIndex

        """
        df = self.df

        # smoothing
        if self.smoothing is not None:
            df = Transform(df).smooth(lookback=self.window_size, method=self.smoothing)

        # log diff
        df = np.log(df).groupby('ticker').diff()

        # fit linear regression
        if isinstance(df.index, pd.MultiIndex):  # multiindex
            pers_df = pd.DataFrame()
            for col in df.columns:
                reg_df = df.dropna().groupby(level=1, group_keys=False).apply(
                    lambda x: linear_reg(x[col], x[col].shift(1), window_type=self.window_type, output='coef',
                                         log=False, trend='c', lookback=self.window_size))
                pers_df = pd.concat([pers_df, reg_df[[col]]], axis=1)

        else:  # single index
            pers_df = pd.DataFrame()
            for col in df.columns:
                reg_df = linear_reg(df[col], df[col].shift(1), window_type=self.window_type, output='coef', log=False,
                                         trend='c', lookback=self.window_size)
                pers_df = pd.concat([pers_df, reg_df[[col]]], axis=1)

        return pers_df.sort_index()


    def network_vol(self) -> pd.DataFrame:
        """

        Returns
        -------
        vol: pd.DataFrame or pd.Series - MultiIndex

        """
        df = self.df

        # smoothing
        if self.smoothing is not None:
            df = Transform(df).smooth(lookback=self.window_size, method=self.smoothing)

        # log diff
        df = np.log(df).groupby('ticker').diff()
        # vol
        vol = df.groupby('ticker').rolling(self.window_size).std().droplevel(0)

        return vol.sort_index()

    def market_beta(self,
                    ret_df: Union[pd.DataFrame, pd.Series],
                    mkt_ret: Union[pd.DataFrame, pd.Series]
                    ) -> pd.DataFrame:
        """

        Parameters
        ----------
        ret_df
        mkt_ret

        Returns
        -------

        """
        # fit linear regression
        if isinstance(ret_df.index, pd.MultiIndex):  # multiindex
            beta = pd.DataFrame()
            for col in ret_df.unstack().columns:
                    reg_df = linear_reg(ret_df.unstack()[col],  mkt_ret, window_type=self.window_type, output='coef',
                                        log=False, trend='c', lookback=self.window_size)
                    beta = pd.concat([beta, reg_df.iloc[:, 0].to_frame(col)], axis=1)

        else:  # single index
            beta = pd.DataFrame()
            for col in ret_df.columns:
                reg_df = linear_reg(ret_df[col], mkt_ret, window_type=self.window_type, output='coef', log=False,
                                         trend='c', lookback=self.window_size)
                beta = pd.concat([beta, reg_df.iloc[:, 0].to_frame(col)], axis=1)

        # stack df
        beta = beta.stack().reset_index()
        beta.columns = ['date', 'ticker', 'beta']
        beta.set_index(['date', 'ticker'], inplace=True)

        return beta

