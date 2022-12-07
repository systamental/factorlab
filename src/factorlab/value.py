from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Union, Optional
from time_series_analysis import linear_reg
from transform import Transform


class Value:
    """
    Value factor.
    """

    def __init__(self,
                 df: pd.DataFrame,
                 log: bool = True,
                 smoothing: str = 'smw',
                 window_type: str = 'expanding',
                 window_size: int = 7,
                 method: str = 'ratio',
                 ):
        """
        Constructor

        Parameters
        ----------
        df: pd.DataFrame - MultiIndex
            DataFrame MultiIndex with DatetimeIndex (level 0), ticker (level 1) and prices (cols).
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
        self.log = log
        self.smoothing = smoothing
        self.window_type = window_type
        self.window_size = window_size
        self.method = method

    def nvt(self,
            nv: str = 'mkt_cap',
            trans_val: str = 'tfr_val_usd'
            ) -> Union[pd.Series, pd.DataFrame]:
        """
        Computes the network value to transactions value factor.

        Parameters
        ----------
        nv: str
            Name of the column to use for network value.
        trans_val: str
            Name of the column to use for transactions value/volume.

        Returns
        -------
        nvt: pd.Series or pd.DataFrame
            Series or DataFrame with DatetimeIndex (level 0), tickers (level 1) and NVT value factor (cols).
        """
        df = self.df[[nv, trans_val]].dropna()

        # log
        if self.log:
            df = Transform(df).log()

        # wrangle df
        if isinstance(df.index, pd.MultiIndex):
            # remove empty cols
            df = df.unstack().dropna(how='all', axis=1).stack()
            # keep common tickers
            if list(df[nv].unstack().columns) != list(df[trans_val].unstack().columns):
                tickers = [ticker for ticker in list(df[nv].unstack().columns) if ticker in
                           list(df[trans_val].unstack().columns)]
                df = df.loc[pd.IndexSlice[:, tickers], :]
        else:
            # remove empty cols
            df = df.dropna(how='all', axis=1)
            # keep common tickers
            if list(df[nv].columns) != list(df[trans_val].columns):
                tickers = [ticker for ticker in list(df[nv].columns) if ticker in
                           list(df[trans_val].columns)]
                df = df.loc[:, tickers]

        # smoothing
        if self.smoothing is not None:
            df[trans_val] = Transform(df[trans_val]).smooth(lookback=self.window_size, method=self.smoothing)

        # method
        if self.method == 'ratio':  # ratio
            nvt = df[trans_val].divide(df[nv]).to_frame('nvt_ratio_' + str(self.window_size))
        else:
            # fit linear regression
            if isinstance(df.index, pd.MultiIndex):  # multiindex
                resid = df.groupby(level=1, group_keys=False).apply(
                    lambda x: linear_reg(x[nv], x[trans_val], window_type=self.window_type, output='resid', log=False,
                                         trend='c', lookback=self.window_size)) * -1
            else:  # single index
                resid = linear_reg(df[nv], df[trans_val], window_type=self.window_type, output='resid', log=False,
                                   trend='c', lookback=self.window_size) * -1

            # z-score residual
            nvt = Transform(resid).normalize_ts(method='z-score', window_type=self.window_type).\
                rename(columns={'resid': 'nvt_resid_' + str(self.window_size)})

        return nvt

    def nvm(self,
            nv: str = 'mkt_cap',
            act_users: str = 'add_act',
            law: Optional[str] = 'Metcalfe',
            ) -> Union[pd.Series, pd.DataFrame]:
        """
        Computes the network value to Metcalfe value factor.

        Parameters
        ----------
        nv: str
            Name of the column to use for network value.
        act_users: str
            Name of the column to use as a proxy for active users.
        law: str, {'Metcalfe', 'Zipf', 'Metcalfe_gen, 'Metcalfe_sqrt', 'Sardoff'}, default 'Metcalfe'
            Law to use for network growth function. Defaults to Sardoff, or active users (n).

        Returns
        -------
        nvm: pd.Series or pd.DataFrame
            Series or DataFrame with DatetimeIndex (level 0), tickers (level 1) and NVM value factor (cols).
        """
        df = self.df[[nv, act_users]].dropna()

        # wrangle df
        if isinstance(df.index, pd.MultiIndex):
            # remove empty cols
            df = df.unstack().dropna(how='all', axis=1).stack()
            # keep common tickers
            if list(df[nv].unstack().columns) != list(df[act_users].unstack().columns):
                tickers = [ticker for ticker in list(df[nv].unstack().columns) if ticker in
                           list(df[act_users].unstack().columns)]
                df = df.loc[pd.IndexSlice[:, tickers], :]
        else:
            # remove empty cols
            df = df.dropna(how='all', axis=1)
            # keep common tickers
            if list(df[nv].columns) != list(df[act_users].columns):
                tickers = [ticker for ticker in list(df[nv].columns) if ticker in
                           list(df[act_users].columns)]
                df = df.loc[:, tickers]

        # smoothing
        if self.smoothing is not None:
            df[act_users] = Transform(df[act_users]).smooth(lookback=self.window_size, method=self.smoothing)

        # law
        if law == 'Metcalfe':
            df[act_users] = df[act_users] ** 2
        elif law == 'Zipf':
            df[act_users] = df[act_users] * np.log(df[act_users])
        elif law == 'Metcalfe_gen':
            df[act_users] = df[act_users] ** 1.5
        elif law == 'Metcalfe_sqrt':
            df[act_users] = np.sqrt(df[act_users] ** 2)
        else:
            df[act_users] = df[act_users]

        # log
        if self.log:
            df = Transform(df).log()

        # method
        if self.method == 'ratio':  # ratio
            if self.smoothing is not None:
                nvm = df[act_users].divide(df[nv]).to_frame('nvm_ratio_' + str(self.window_size))
            else:
                nvm = df[act_users].divide(df[nv]).to_frame('nvm_ratio')

        else:
            # fit linear regression
            if isinstance(df.index, pd.MultiIndex):  # multiindex
                resid = df.groupby(level=1, group_keys=False).apply(
                    lambda x: linear_reg(x[nv], x[act_users], window_type=self.window_type, output='resid', log=False,
                                         trend='c', lookback=self.window_size)) * -1
            else:  # single index
                resid = linear_reg(df[nv], df[act_users], window_type=self.window_type, output='resid', log=False,
                                   trend='c', lookback=self.window_size) * -1

            # z-score residual
            if self.smoothing is not None:
                nvm = Transform(resid).normalize_ts(method='z-score', window_type=self.window_type). \
                    rename(columns={'resid': 'nvm_resid_' + str(self.window_size)})
            else:
                nvm = Transform(resid).normalize_ts(method='z-score', window_type=self.window_type). \
                    rename(columns={'resid': 'nvm_resid'})

        return nvm

    def nvsf(self,
             nv: str = 'mkt_cap',
             supply: str = 'supply_circ'
             ) -> Union[pd.Series, pd.DataFrame]:
        """
        Computes the Stocks to Flow ratio.

        Parameters
        ----------
        nv: str
            Name of the column to use for network value.
        supply: str
            Name of the column to use as a proxy for supply.

        Returns
        -------
        nvm: pd.Series or pd.DataFrame
            Series or DataFrame with DatetimeIndex (level 0), tickers (level 1) and SF ratio (cols).
        """
        df = self.df[[nv, supply]].dropna()

        # wrangle df
        if isinstance(df.index, pd.MultiIndex):
            # remove empty cols
            df = df.unstack().dropna(how='all', axis=1).stack()
            # keep common tickers
            if list(df[nv].unstack().columns) != list(df[supply].unstack().columns):
                tickers = [ticker for ticker in list(df[nv].unstack().columns) if ticker in
                           list(df[supply].unstack().columns)]
                df = df.loc[pd.IndexSlice[:, tickers], :]
        else:
            # remove empty cols
            df = df.dropna(how='all', axis=1)
            # keep common tickers
            if list(df[nv].columns) != list(df[supply].columns):
                tickers = [ticker for ticker in list(df[nv].columns) if ticker in
                           list(df[supply].columns)]
                df = df.loc[:, tickers]

        # sf ratio
        if isinstance(df.index, pd.MultiIndex):  # multiindex
            df['sf'] = (1 / (df[supply].unstack().diff(periods=self.window_size) / df[supply].unstack())).stack()
        else:
            df['sf'] = 1 / (df[supply].diff(periods=self.window_size) / df[supply])

        # method
        if self.method == 'ratio':  # ratio
            psf = (1 / df['sf']) * -1
            psf = psf.to_frame('iss_rate_' + str(self.window_size))

        else:  # lin reg
            df = Transform(df).log()
            if isinstance(df.index, pd.MultiIndex):  # multiindex
                resid = df.groupby(level=1, group_keys=False).apply(
                    lambda x: linear_reg(x[nv], x['sf'], window_type=self.window_type, output='resid', log=False,
                                         trend='c')) * -1
            else:  # single index
                resid = linear_reg(df[nv], df['sf'], window_type=self.window_type, output='resid', log=False,
                                   trend='c') * -1

            # z-score residual
            psf = Transform(resid).normalize_ts(method='z-score', window_type=self.window_type).\
                rename(columns={'resid': 'nvsf_resid_' + str(self.window_size)})

        return psf

    def nvc(self,
            nv: str = 'mkt_cap',
            cost: str = 'hashrate'
            ) -> Union[pd.Series, pd.DataFrame]:
        """
        Computes the Network Value to Production Costs ratio.

        Parameters
        ----------
        nv: str
            Name of the column to use for network value.
        cost: str
            Name of the column to use as a proxy for production costs.

        Returns
        -------
        nvm: pd.Series or pd.DataFrame
            Series or DataFrame with DatetimeIndex (level 0), tickers (level 1) and NVC ratio (cols).
        """
        df = self.df[[nv, cost]].dropna()

        # log
        if self.log:
            df = Transform(df).log()

        # wrangle df
        if isinstance(df.index, pd.MultiIndex):
            # remove empty cols
            df = df.unstack().dropna(how='all', axis=1).stack()
            # keep common tickers
            if list(df[nv].unstack().columns) != list(df[cost].unstack().columns):
                tickers = [ticker for ticker in list(df[nv].unstack().columns) if ticker in
                           list(df[cost].unstack().columns)]
                df = df.loc[pd.IndexSlice[:, tickers], :]
        else:
            # remove empty cols
            df = df.dropna(how='all', axis=1)
            # keep common tickers
            if list(df[nv].columns) != list(df[cost].columns):
                tickers = [ticker for ticker in list(df[nv].columns) if ticker in
                           list(df[cost].columns)]
                df = df.loc[:, tickers]

        # smoothing
        if self.smoothing is not None:
            df[cost] = Transform(df[cost]).smooth(lookback=self.window_size, method=self.smoothing)

        # method
        if self.method == 'ratio':  # ratio
            nvc = df[cost].divide(df[nv]).to_frame('nvc_ratio_' + str(self.window_size))
        else:
            # fit linear regression
            if isinstance(df.index, pd.MultiIndex):  # multiindex
                resid = df.groupby(level=1, group_keys=False).apply(
                    lambda x: linear_reg(x[nv], x[cost], window_type=self.window_type, output='resid', log=False,
                                         trend='c', lookback=self.window_size)) * -1
            else:  # single index
                resid = linear_reg(df[nv], df[cost], window_type=self.window_type, output='resid', log=False, trend='c',
                                   lookback=self.window_size) * -1

            # z-score residual
            nvc = Transform(resid).normalize_ts(method='z-score', window_type=self.window_type).\
                rename(columns={'resid': 'nvc_resid_' + str(self.window_size)})

        return nvc
