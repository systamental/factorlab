from __future__ import annotations
import pandas as pd
import numpy as np
import inspect
from typing import Union, Optional

from factorlab.feature_analysis.time_series_analysis import linear_reg
from factorlab.feature_engineering.transformations import Transform


class Trend:
    """
    Trend factor.
    """
    def __init__(self,
                 df: pd.DataFrame,
                 vwap: bool = True,
                 log: bool = True,
                 st_lookback: int = None,
                 lookback: int = 20,
                 lt_lookback: int = None,
                 sm_window_type: str = 'rolling',
                 sm_central_tendency: str = 'mean',
                 sm_window_fcn: Optional[str] = None,
                 lags: int = 0,
                 ):
        """
        Constructor

        Parameters
        ----------
        df: pd.DataFrame - MultiIndex
            DataFrame MultiIndex with DatetimeIndex (level 0), ticker (level 1) and prices (cols).
         vwap: bool, default False
            Compute signal on vwap price.
        log: bool, default False
            Converts to log price.
        st_lookback: int
            Number of observations in short-term moving window.
        lookback: int
            Number of observations in moving window.
        lt_lookback: int
            Number of observations in long-term moving window.
        sm_window_type: str, {'rolling', 'ewm'}, default 'rolling'
            Smoothing window type.
        sm_central_tendency: str, {'mean', 'median'}, default 'mean'
            Measure of central tendency used for the smoothing rolling window.
        sm_window_fcn: str, optional, default None
            Smoothing window function.
        lags: int, default 0
            Number of periods to lag values by.
        """
        # convert data types
        if isinstance(df, pd.Series):
            df = df.to_frame('close')
        if not isinstance(df.index, pd.MultiIndex):
            df = df.stack().to_frame()
        # check fields
        if 'close' not in df.columns:
            raise ValueError("Close price series must be provided in dataframe.")

        self.df = df.copy()
        self.vwap = vwap
        self.log = log
        self.price = self.compute_price()
        self.st_lookback = st_lookback
        self.lookback = lookback
        self.lt_lookback = lt_lookback
        self.sm_window_type = sm_window_type
        self.sm_central_tendency = sm_central_tendency
        self.sm_window_fcn = sm_window_fcn
        self.lags = lags
        self.trend = None

    def compute_price(self) -> pd.DataFrame:
        """
        Computes the price series.

        Returns
        -------
        price: pd.DataFrame - MultiIndex
            DataFrame with DatetimeIndex (level 0), ticker (level 1) and price (cols).
        """
        # compute price
        if self.vwap:
            self.price = Transform(self.df).vwap()[['vwap']].copy()
        else:
            self.price = self.df[['close']].copy()
        if self.log:
            self.price = Transform(self.price).log()

        return self.price

    def breakout(self, method: str = 'cdf') -> pd.DataFrame:
        """
         Compute breakout signal.

        Parameters
        ----------
        method: str, {'min-max', 'percentile', 'cdf'}, default 'min-max'
            Method to use to normalize price series between 0 and 1.

         Returns
         -------
         signal: pd.DataFrame - MultiIndex
             Series with DatetimeIndex (level 0), ticker (level 1) and breakout signal (cols).
         """
        # check method
        if method not in ['min-max', 'percentile', 'cdf']:
            raise ValueError('Invalid method. Method must be: min-max, percentile, cdf.')

        # normalize
        self.trend = Transform(self.price).normalize_ts(method=method, window_type='rolling', window_size=self.lookback)

        # convert to breakout signal
        self.trend = (self.trend * 2) - 1

        # name
        self.trend.columns = [f"{inspect.currentframe().f_code.co_name}_{self.lookback}"]
        # sort index
        self.trend = self.trend.sort_index()

        return self.trend

    def price_mom(self) -> pd.DataFrame:
        """
        Computes the price momentum trend factor.

        Returns
        -------
        price_mom: pd.DataFrame - MultiIndex
           Series with DatetimeIndex (level 0), tickers (level 1) and price momentum trend factor values (cols).
        """
        # log
        if self.log:
            self.price = np.exp(self.price)

        # price mom
        self.trend = Transform(self.price).returns(method='simple', lags=self.lookback)
        # name
        self.trend.columns = [f"{inspect.currentframe().f_code.co_name}_{self.lookback}"]
        # sort index
        self.trend = self.trend.sort_index()

        return self.trend

    def divergence(self) -> pd.DataFrame:
        """
        Compute divergence measure.

        Returns
        -------
        div: pd.Series or pd.DataFrame - MultiIndex
           Series or dataframe with DatetimeIndex (level 0), tickers (level 1) and divergence values (cols).
        """
        # compute sign
        sign = np.sign(self.price.groupby(level=1).diff())

        # divergence
        self.trend = Transform(sign).smooth(self.lookback, window_type=self.sm_window_type,
                     central_tendency=self.sm_central_tendency, window_fcn=self.sm_window_fcn)
        # name
        self.trend.columns = [f"{inspect.currentframe().f_code.co_name}_{self.lookback}"]
        # sort index
        self.trend = self.trend.sort_index()

        return self.trend

    def time_trend(self) -> pd.DataFrame:
        """
        Computes the time trend factor by regressing price on a constant and time trend to estimate coefficients.

        Returns
        -------
        coeff: pd.Series or pd.DataFrame - MultiIndex
            Series with DatetimeIndex (level 0), tickers (level 1) and time trend coefficient of price regressed
            on a constant and time trend over the lookback window.
        """
        # fit linear regression
        coeff = self.price.groupby(level=1, group_keys=False).apply(lambda x: linear_reg(x, None, window_type='rolling',
                                    output='coef', log=False, trend='ct', window_size=self.lookback))

        # time trend
        self.trend = coeff[['trend']]
        # name
        self.trend.columns = [f"{inspect.currentframe().f_code.co_name}_{self.lookback}"]
        # sort index
        self.trend = self.trend.sort_index()

        return self.trend

    def price_acc(self) -> pd.DataFrame:
        """
        Compute the price acceleration factor by regressing price on a constant, time trend
        and time trend squared to estimate coefficients.

        Returns
        -------
        coeff: pd.Series or pd.DataFrame - MultiIndex
            Series with DatetimeIndex (level 0), tickers (level 1) and price acceleration (time trend squared) of
            price regressed on a constant, time trend and time trend squared over the lookback window.
        """
        # fit linear regression
        coeff = self.price.groupby(level=1, group_keys=False).apply(lambda x: linear_reg(x, None, window_type='rolling',
                                   output='coef', log=False, trend='ctt', window_size=self.lookback))

        # price acceleration
        self.trend = coeff[['trend_squared']]
        # name
        self.trend.columns = [f"{inspect.currentframe().f_code.co_name}_{self.lookback}"]
        # sort index
        self.trend = self.trend.sort_index()

        return self.trend

    def alpha_mom(self) -> pd.DataFrame:
        """
        Constant term (alpha) from fitting an OLS linear regression of price on the market beta,
        i.e. cross-sectional average).

        Returns
        -------
        resid: pd.Series or pd.DataFrame - MultiIndex
            DataFrame with DatetimeIndex (level 0), tickers (level 1) and residuals.
       """
        # compute ret
        if self.log:
            ret = self.price.groupby(level=1, group_keys=False).diff()
        else:
            ret = Transform(self.price).returns()

        # compute mean ret for all assets
        mkt_ret = ret.groupby(level=0).mean()
        # merge y, X
        data = pd.merge(ret, mkt_ret, right_index=True, left_index=True).dropna()
        data.columns = ['y', 'X']

        # fit linear regression
        alpha = data.groupby(level=1, group_keys=False).apply(
            lambda x: linear_reg(x.y, x.X, window_type='rolling', output='coef', log=False, trend='c',
                                 window_size=self.lookback))[['const']]
        # alpha
        self.trend = alpha
        # name
        self.trend.columns = [f"{inspect.currentframe().f_code.co_name}_{self.lookback}"]
        # sort index
        self.trend = self.trend.sort_index()

        return self.trend

    def rsi(self, signal: bool = True) -> pd.DataFrame:
        """
        Computes the RSI indicator.

        Parameters
        ----------
        signal: bool, default True
            Converts RSI to a signal between -1 and 1.
            Typically, RSI is normalized to between 0 and 100.

        Returns
        -------
        rsi: pd.Series or pd.DataFrame - MultiIndex
            Series or dataframe with DatetimeIndex (level 0), ticker (level 1) and RSI indicator values (cols).
        """
        # log
        if self.log is False:
            self.price = Transform(self.price).log()

        # compute price returns and up/down days
        ret = self.price.groupby(level=1, group_keys=False).diff()

        # get up and down days
        up = ret.where(ret > 0).fillna(0)
        down = abs(ret.where(ret < 0).fillna(0))

        # smoothing
        rs = Transform(up).smooth(self.lookback, window_type=self.sm_window_type,
                                  central_tendency=self.sm_central_tendency, window_fcn=self.sm_window_fcn) / \
            Transform(down).smooth(self.lookback, window_type=self.sm_window_type,
                                  central_tendency=self.sm_central_tendency, window_fcn=self.sm_window_fcn)

        # normalization to remove inf 0 div
        rs = 100 - (100 / (1 + rs))
        # signal
        if signal:
            rs = (rs - 50)/50

        # rsi
        self.trend = rs
        # name
        self.trend.columns = [f"{inspect.currentframe().f_code.co_name}_{self.lookback}"]
        # sort index
        self.trend = self.trend.sort_index()

        return self.trend

    def stochastic(self,
                   stochastic: str = 'd',
                   signal: bool = True
                   ) -> Union[pd.Series, pd.DataFrame]:
        """
        Computes the stochastic indicator K and D.

        Parameters
        ----------
        stochastic: str, {'k', 'd', 'all'}, default 'd'
            Stochastic to return.
        signal: bool, default True
            Converts stochastic to a signal between -1 and 1.

        Returns
        -------
        stochastic k, d: pd.Series or pd.DataFrame - MultiIndex
            DataFrame with DatetimeIndex and Stochastic indicator.
        """
        # check df fields
        if 'high' not in self.df.columns or 'low' not in self.df.columns:
            raise ValueError("High and low price series must be provided in dataframe.")

        # st lookback
        if self.st_lookback is None:
            self.st_lookback = max(2, round(self.lookback / 4))

        # compute k
        if isinstance(self.df.index, pd.MultiIndex):
            num = self.df.close.sort_index(level=1) - self.df.low.groupby(level=1).rolling(self.lookback).min().values
            denom = (self.df.high.groupby(level=1).rolling(self.lookback).max() -
                     self.df.low.groupby(level=1).rolling(self.lookback).min().values).droplevel(0)
            k = num/denom

        else:
            k = (self.df.close - self.df.low.rolling(self.lookback).min()) / \
                (self.df.high.rolling(self.lookback).max() - self.df.low.rolling(self.lookback).min())

        # clip extreme values
        k = k.clip(0, 1)
        # smoothing
        d = Transform(k).smooth(self.st_lookback, window_type=self.sm_window_type,
                                central_tendency=self.sm_central_tendency)

        # create df
        stoch_df = pd.concat([k, d], axis=1)
        stoch_df.columns = ['k', 'd']

        # convert to signal
        if signal:
            stoch_df = (stoch_df * 2) - 1

        # stochastic
        self.trend = stoch_df[[stochastic]]
        # name
        self.trend.columns = [f"{inspect.currentframe().f_code.co_name}_{self.lookback}"]
        # sort index
        self.trend = self.trend.sort_index()

        return self.trend

    def intensity(self) -> pd.DataFrame:
        """
        Computes intraday intensity trend factor.

        Returns
        -------
        intensity: pd.Series or pd.DataFrame - MultiIndex
            Series or dataframe with DatetimeIndex (level 0), tickers (level 1) and
            Intensity trend factor values (cols).
        """
        # check df fields
        if 'high' not in self.df.columns or 'low' not in self.df.columns:
            raise ValueError("High and low price series must be provided in dataframe.")

        # compute true range
        hilo = self.df.high - self.df.close

        if isinstance(self.df.index, pd.MultiIndex):
            hicl = abs(self.df.high.sort_index(level=1) - self.df.close.groupby(level=1).shift(1))
            locl = abs(self.df.low.sort_index(level=1) - self.df.close.groupby(level=1).shift(1))
        else:
            hicl = abs(self.df.high - self.df.close.shift(1))
            locl = abs(self.df.low - self.df.close.shift(1))

        # compute ATR, chg and intensity
        tr = pd.concat([hilo, hicl, locl], axis=1).max(axis=1)  # compute ATR
        today_chg = self.df.close - self.df.open  # compute today's change
        intensity = today_chg / tr  # compute intensity

        # intensity
        self.trend = Transform(intensity).smooth(self.lookback, window_type=self.sm_window_type,
                                  central_tendency=self.sm_central_tendency, window_fcn=self.sm_window_fcn)
        # name
        self.trend.columns = [f"{inspect.currentframe().f_code.co_name}_{self.lookback}"]
        # sort index
        self.trend = self.trend.sort_index()

        return self.trend

    def mw_diff(self) -> pd.DataFrame:
        """
        Computes the moving window difference trend factor.

        Returns
        -------
        mw_diff: pd.Series or pd.DataFrame - MultiIndex
            Series with DatetimeIndex (level 0), ticker (level 1) and
             moving window difference trend factor values (cols).
        """
        # short rolling window param
        if self.st_lookback is None:
            self.st_lookback = int(np.ceil(self.lookback / 3))

        # smoothing
        mw_diff = Transform(self.price).smooth(self.st_lookback, window_type=self.sm_window_type,
                                  central_tendency=self.sm_central_tendency, window_fcn=self.sm_window_fcn,
                                       lags=self.lags) - \
            Transform(self.price).smooth(self.lookback, window_type=self.sm_window_type,
                                 central_tendency=self.sm_central_tendency, window_fcn=self.sm_window_fcn,
                                 lags=self.lags)

        # mw_diff
        self.trend = mw_diff
        # name
        self.trend.columns = [f"{inspect.currentframe().f_code.co_name}_{self.lookback}"]
        # sort index
        self.trend = self.trend.sort_index()

        return self.trend

    def ewma_wxover(self,
                    s_k: list = [2, 4, 8],
                    l_k: list = [6, 12, 24],
                    signal: bool = False
                    ) -> pd.DataFrame:
        """
        Computes the moving window difference trend factor.

        Computed as described in Dissecting Investment Strategies in the Cross-Section and Time Series:
        https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2695101

        Parameters
        ----------
        s_k: list of int, default [2, 4, 8]
            Represents n for short window where halflife is given by log(0.5)/log(1 − 1/n).
        l_k: list of int, default [6, 12, 24]
            Represents n for long window where halflife is given by log(0.5)/log(1 − 1/n).
        signal: bool, False
            Converts normalized ewma crossover values to signal between [-1,1].

        Returns
        -------
        ewma_xover: pd.Series or pd.DataFrame
            Series or DataFrame with DatetimeIndex (level 0), tickers (level 1) and
            ewma crossover trend factor values (cols).
        """
        # half-life lists for short and long windows
        hl_s = [np.log(0.5) / np.log(1 - 1 / i) for i in s_k]
        hl_l = [np.log(0.5) / np.log(1 - 1 / i) for i in l_k]

        # create emtpy df
        factor_df = pd.DataFrame()

        # compute ewma diff for short, medium and long windows
        for i in range(0, len(s_k)):
            factor_df['x_k' + str(i)] = (self.price.unstack().ewm(halflife=hl_s[i]).mean() -
                                         self.price.unstack().ewm(halflife=hl_l[i]).mean()).stack()

        # normalize by std of price
        for i in range(0, len(s_k)):
            factor_df['y_k' + str(i)] = (
                        factor_df['x_k' + str(i)].unstack() / self.price.unstack().rolling(90).std()).stack()

        # normalize by normalized y_k diff
        for i in range(0, len(s_k)):
            factor_df['z_k' + str(i)] = (factor_df['x_k' + str(i)].unstack() /
                                         factor_df['x_k' + str(i)].unstack().rolling(365).std()).stack()

        # convert to signal
        if signal:
            for i in range(0, len(s_k)):
                factor_df['signal_k' + str(i)] = (factor_df['z_k' + str(i)] *
                                                  np.exp((-1 * factor_df['z_k' + str(i)] ** 2) / 4)) / 0.89

        # mean of short, medium and long window signals
        ewma_xover = factor_df.iloc[:, -3:].mean(axis=1)
        # replace inf
        ewma_xover.replace([np.inf, -np.inf], np.nan, inplace=True)

        # ffill NaNs
        ewma_xover = ewma_xover.groupby(level=1).ffill()
        # name
        self.trend = ewma_xover.to_frame(f"{inspect.currentframe().f_code.co_name}")
        # sort index
        self.trend = self.trend.sort_index()

        return self.trend

    def energy(self, mass_method='vol', perc: Optional[float] = 0.05) -> pd.DataFrame:
        """
        Computes the energy trend factor, E = mc^2, where E is energy, m is mass (volatility or VaR)
        and c is the speed (price momentum).

        Parameters
        ----------
        mass_method: str, {'vol', 'VaR'}, default 'vol'
            Method to use to compute mass.
        perc: float, default 0.05
            Percentile to use to compute VaR.

        Returns
        -------
        energy: pd.Series or pd.DataFrame - MultiIndex
            Series or dataframe with DatetimeIndex (level 0), tickers (level 1) and
            energy trend factor values (cols).
        """
        # compute speed
        speed = self.price_mom()

        # ret
        if self.log:
            ret = self.price.groupby(level=1, group_keys=False).diff()
        else:
            ret = Transform(self.price).returns()

        # mass
        # volatility
        if mass_method == 'vol':
            mass = ret.groupby(level=1, group_keys=False).rolling(self.lookback).std().droplevel(0)
        # VaR
        else:
            if speed > 0:
                mass = ret.groupby(level=1, group_keys=False).rolling(self.lookback).quantile(perc).droplevel(0)
            else:
                mass = ret.groupby(level=1, group_keys=False).rolling(self.lookback).quantile(1-perc).droplevel(0)

        # energy
        self.trend = (mass.iloc[:, 0].unstack() * (speed.iloc[:, 0].unstack()**2)).stack().to_frame()
        # name
        self.trend.columns = [f"{inspect.currentframe().f_code.co_name}_{self.lookback}"]
        # sort index
        self.trend = self.trend.sort_index()

        return self.trend
