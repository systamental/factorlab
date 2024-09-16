from __future__ import annotations
import pandas as pd
import numpy as np
import inspect
from typing import Union, Optional
from scipy.stats import norm, logistic

from factorlab.signal_generation.time_series_analysis import TimeSeriesAnalysis as TSA
from factorlab.feature_engineering.transformations import Transform


class Trend:
    """
    Trend factor.
    """
    def __init__(self,
                 df: pd.DataFrame,
                 vwap: bool = True,
                 log: bool = True,
                 normalize: bool = True,
                 central_tendency: str = 'mean',
                 disp_method: str = 'std',
                 window_size: int = 20,
                 short_window_size: int = None,
                 long_window_size: int = None,
                 window_type: str = 'rolling',
                 window_fcn: Optional[str] = None,
                 lags: int = 0,
                 **kwargs
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
        normalize: bool, default True
            Normalize trend factor.
        central_tendency: str, {'mean', 'median'}, default 'mean'
            Measure of central tendency used for the smoothing rolling window.
        disp_method: str, {'std', 'iqr', 'mad', 'atr'}, default 'std'
            Method for computing dispersion.
        window_size: int, default 20
            Number of observations in moving window.
        short_window_size: int, default None
            Number of observations in short-term moving window.
        long_window_size: int, default None
            Number of observations in long-term moving window.
        window_type: str, {'rolling', 'ewm'}, default 'rolling'
            Smoothing window type.
        window_fcn: str, optional, default None
            Smoothing window function.
        lags: int, default 0
            Number of periods to lag values by.
        """
        self.df = df.to_frame().copy() if isinstance(df, pd.Series) else df.copy()
        self.vwap = vwap
        self.log = log
        self.normalize = normalize
        self.central_tendency = central_tendency
        self.disp_method = disp_method
        self.price = self.compute_price()
        self.window_size = window_size
        self.short_window_size = short_window_size
        self.long_window_size = long_window_size
        self.window_type = window_type
        self.window_fcn = window_fcn
        self.lags = lags
        self.kwargs = kwargs
        self.trend = None
        self.disp = None

    def compute_price(self) -> pd.DataFrame:
        """
        Computes the price series.

        Returns
        -------
        price: pd.DataFrame
            DataFrame with DatetimeIndex and price (cols).
        """
        # compute price
        if self.vwap:
            self.price = Transform(self.df).vwap()[['vwap']].copy()
        else:
            self.price = self.df.copy()
        if self.log:
            self.price = Transform(self.price).log()

        return self.price

    def compute_dispersion(self) -> pd.DataFrame:
        """
        Computes the dispersion of the price series.

        Returns
        -------
        disp: pd.DataFrame
            DataFrame with DatetimeIndex and dispersion values (cols).
        """
        # check method
        if self.disp_method not in ['std', 'iqr', 'mad', 'atr']:
            raise ValueError('Invalid dispersion method. Method must be: std, iqr, range, atr, mad.')

        # compute dispersion
        if self.disp_method == 'atr':
            # compute atr
            self.disp = Transform(self.df).dispersion(method=self.disp_method,
                                                      window_type=self.window_type,
                                                      window_size=self.window_size)
        else:
            # price chg
            chg = Transform(self.price).diff()
            # dispersion
            self.disp = Transform(chg).dispersion(method=self.disp_method,
                                                  window_type=self.window_type,
                                                  window_size=self.window_size)

        return self.disp

    def breakout(self, method: str = 'min-max') -> pd.DataFrame:
        """
         Computes the breakout trend factor.

        Parameters
        ----------
        method: str, {'min-max', 'percentile', 'norm'}, default 'min-max'
            Method to use to normalize price series between 0 and 1.

         Returns
         -------
        breakout: pd.DataFrame
            DataFrame with DatetimeIndex and breakout signal values (cols).
         """
        # check method
        if method not in ['min-max', 'percentile', 'norm', 'logistic', 'adj_norm']:
            raise ValueError('Invalid method. Method must be: min-max, percentile, norm, logistic or adj_norm.')

        # uniform distribution
        if method == 'min-max':
            self.trend = Transform(self.price).normalize(method='min-max',
                                                         window_type='rolling',
                                                         window_size=self.window_size)

        # percentile rank
        elif method == 'percentile':
            self.trend = Transform(self.price).normalize(method='percentile',
                                                         window_type='rolling',
                                                         window_size=self.window_size)

        # normal distribution
        elif method == 'norm':
            self.trend = Transform(self.price).normalize(method='z-score',
                                                         window_type=self.window_type,
                                                         window_size=self.window_size)
            # convert to cdf
            self.trend = pd.DataFrame(norm.cdf(self.trend), index=self.trend.index, columns=self.trend.columns)

        # logistic
        elif method == 'logistic':
            self.trend = Transform(self.price).normalize(method='z-score',
                                                         window_type=self.window_type,
                                                         window_size=self.window_size)
            # convert to cdf
            self.trend = pd.DataFrame(logistic.cdf(self.trend), index=self.trend.index, columns=self.trend.columns)

        # adjusted normal distribution
        else:
            self.trend = Transform(self.price).normalize(method='adj_norm',
                                                         window_type=self.window_type,
                                                         window_size=self.window_size)

        # convert to signal
        if method == 'adj_norm':
            self.trend = self.trend * np.exp((-1 * self.trend ** 2) / 4) / 0.89
        else:
            self.trend = (self.trend * 2) - 1

        # name
        if self.trend.shape[1] == 1:
            self.trend.columns = [f"{inspect.currentframe().f_code.co_name}_{self.window_size}"]

        # sort index
        self.trend = self.trend.sort_index()

        return self.trend

    def price_mom(self) -> pd.DataFrame:
        """
        Computes the price momentum trend factor.

        Returns
        -------
        mom: pd.DataFrame
            DataFrame with DatetimeIndex and price momentum values (cols).
        """
        # compute price returns
        self.trend = Transform(self.price).diff(lags=self.window_size)

        # normalize
        if self.normalize:
            self.compute_dispersion()
            self.trend = self.trend.div(self.disp, axis=0)

        # name
        if self.trend.shape[1] == 1:
            self.trend.columns = [f"{inspect.currentframe().f_code.co_name}_{self.window_size}"]

        # sort index
        self.trend = self.trend.sort_index()

        return self.trend

    def ewma(self) -> pd.DataFrame:
        """
        Computes the exponentially weighted moving average (EWMA) trend factor.

        Returns
        -------
        ewma: pd.DataFrame
            DataFrame with DatetimeIndex and EWMA values (cols).
        """
        # chg
        chg = Transform(self.price).diff()

        # normalize
        if self.normalize:
            self.compute_dispersion()
            chg = chg.div(self.disp, axis=0)

        # smoothing
        self.trend = Transform(chg).smooth(self.window_size,
                                           window_type='ewm',
                                           **self.kwargs)

        # name
        if self.trend.shape[1] == 1:
            self.trend.columns = [f"{inspect.currentframe().f_code.co_name}_{self.window_size}"]

        # sort index
        self.trend = self.trend.sort_index()

        return self.trend

    def divergence(self) -> pd.DataFrame:
        """
        Computes the divergence trend factor.

        Returns
        -------
        divergence: pd.DataFrame
            DataFrame with DatetimeIndex and divergence values (cols).
        """
        # compute sign
        if isinstance(self.price.index, pd.MultiIndex):
            sign = np.sign(self.price.groupby(level=1).diff())
        else:
            sign = np.sign(self.price.diff())

        # divergence
        self.trend = Transform(sign).smooth(self.window_size,
                                            window_type=self.window_type,
                                            central_tendency=self.central_tendency,
                                            window_fcn=self.window_fcn)

        # name
        if self.trend.shape[1] == 1:
            self.trend.columns = [f"{inspect.currentframe().f_code.co_name}_{self.window_size}"]

        # sort index
        self.trend = self.trend.sort_index()

        return self.trend

    def time_trend(self) -> pd.DataFrame:
        """
        Computes the time trend factor by regressing price on a constant and time trend to estimate coefficients.

        Returns
        -------
        trend: pd.DataFrame
            DataFrame with DatetimeIndex and time trend values (cols).
        """
        # check df index
        if isinstance(self.price.index, pd.MultiIndex) and self.price.shape[1] > 1:
            raise ValueError("Select a price series/col.")
        elif not isinstance(self.price.index, pd.MultiIndex):
            self.price = self.price.stack().to_frame()
            self.price.index.names = ['date', 'ticker']

        # fit linear regression
        coeff = TSA(self.price, trend='ct', window_type='rolling',
                    window_size=self.window_size).linear_regression(output='params')

        # time trend
        self.trend = coeff[['trend']]

        # normalize
        if self.normalize:
            self.compute_dispersion()
            self.trend = self.trend.div(self.disp.squeeze(), axis=0)

        # single index
        if isinstance(self.df.index, pd.MultiIndex) is False:
            self.trend = self.trend.iloc[:, 0].unstack()

        # name
        if self.trend.shape[1] == 1:
            self.trend.columns = [f"{inspect.currentframe().f_code.co_name}_{self.window_size}"]

        # sort index
        self.trend = self.trend.sort_index()

        return self.trend

    def price_acc(self) -> pd.DataFrame:
        """
        Computes the price acceleration factor by regressing price on a constant, time trend
        and time trend squared to estimate coefficients.

        Returns
        -------
        coeff: pd.DataFrame
            DataFrame with DatetimeIndex and price acceleration values (cols).
        """
        # check df index
        if isinstance(self.price.index, pd.MultiIndex) and self.price.shape[1] > 1:
            raise ValueError("Select a price series/col.")
        elif not isinstance(self.price.index, pd.MultiIndex):
            self.price = self.price.stack().to_frame()
            self.price.index.names = ['date', 'ticker']

        # fit linear regression
        coeff = TSA(self.price, trend='ctt', window_type='rolling',
                    window_size=self.window_size).linear_regression(output='params')

        # price acceleration
        self.trend = coeff[['trend_squared']]

        # normalize
        if self.normalize:
            self.compute_dispersion()
            self.trend = self.trend.div(self.disp.squeeze(), axis=0)

        # single index
        if isinstance(self.df.index, pd.MultiIndex) is False:
            self.trend = self.trend.iloc[:, 0].unstack()

        # name
        if self.trend.shape[1] == 1:
            self.trend.columns = [f"{inspect.currentframe().f_code.co_name}_{self.window_size}"]

        # sort index
        self.trend = self.trend.sort_index()

        return self.trend

    def alpha_mom(self) -> pd.DataFrame:
        """
        Constant term/coefficient (alpha) from fitting an OLS linear regression of price on the market portfolio (beta,
        i.e. cross-sectional average of returns).

        Returns
        -------
        alpha: pd.DataFrame
            DataFrame with DatetimeIndex and alpha values (cols).
       """
        # check df index
        if isinstance(self.price.index, pd.MultiIndex) and self.price.shape[1] > 1:
            raise ValueError("Select a price series/col.")
        elif not isinstance(self.price.index, pd.MultiIndex):
            self.price = self.price.stack().to_frame()
            self.price.index.names = ['date', 'ticker']

        # compute ret
        ret = self.price.groupby(level=1, group_keys=False).diff()

        # compute mean ret for all assets
        mkt_ret = ret.groupby(level=0).mean().reindex(ret.index, level=0)

        # fit linear regression
        alpha = TSA(ret, mkt_ret, trend='c', window_type='rolling',
                    window_size=self.window_size).linear_regression(output='params')

        # alpha
        self.trend = alpha[['const']]

        # normalize
        if self.normalize:
            self.compute_dispersion()
            self.trend = self.trend.div(self.disp.squeeze(), axis=0)

        # single index
        if isinstance(self.df.index, pd.MultiIndex) is False:
            self.trend = self.trend.iloc[:, 0].unstack()

        # name
        if self.trend.shape[1] == 1:
            self.trend.columns = [f"{inspect.currentframe().f_code.co_name}_{self.window_size}"]

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
        rsi: pd.DataFrame - MultiIndex
            DataFrame with DatetimeIndex and RSI indicator (cols).
        """
        # log
        if self.log is False:
            self.price = Transform(self.price).log()

        # compute price returns and up/down days
        if isinstance(self.price.index, pd.MultiIndex):
            ret = self.price.groupby(level=1, group_keys=False).diff()
        else:
            ret = self.price.diff()

        # get up and down days
        up = ret.where(ret > 0).fillna(0)
        down = abs(ret.where(ret < 0).fillna(0))

        # smoothing
        rs = Transform(up).smooth(self.window_size,
                                  window_type=self.window_type,
                                  central_tendency=self.central_tendency,
                                  window_fcn=self.window_fcn) / \
            Transform(down).smooth(self.window_size,
                                   window_type=self.window_type,
                                   central_tendency=self.central_tendency,
                                   window_fcn=self.window_fcn)

        # normalization to remove inf 0 div
        rs = 100 - (100 / (1 + rs))
        # signal
        if signal:
            rs = (rs - 50)/50

        # rsi
        self.trend = rs

        # name
        if self.trend.shape[1] == 1:
            self.trend.columns = [f"{inspect.currentframe().f_code.co_name}_{self.window_size}"]

        # sort index
        self.trend = self.trend.sort_index()

        return self.trend

    def stochastic(self, stochastic: str = 'd', signal: bool = True) -> Union[pd.Series, pd.DataFrame]:
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

        # short window
        if self.short_window_size is None:
            self.short_window_size = max(2, int(np.ceil(np.sqrt(self.window_size))))  # sqrt of long-term window

        # compute k
        if isinstance(self.df.index, pd.MultiIndex):
            num = self.df.close.sort_index(level=1) - \
                  self.df.low.groupby(level=1).rolling(self.window_size).min().values
            denom = (self.df.high.groupby(level=1).rolling(self.window_size).max() -
                     self.df.low.groupby(level=1).rolling(self.window_size).min().values).droplevel(0)
            k = num/denom

        else:
            k = (self.df.close - self.df.low.rolling(self.window_size).min()) / \
                (self.df.high.rolling(self.window_size).max() - self.df.low.rolling(self.window_size).min())

        # clip extreme values
        k = k.clip(0, 1)
        # smoothing
        d = Transform(k).smooth(self.short_window_size,
                                window_type=self.window_type,
                                central_tendency=self.central_tendency)

        # create df
        stoch_df = pd.concat([k, d], axis=1)
        stoch_df.columns = ['k', 'd']

        # convert to signal
        if signal:
            stoch_df = (stoch_df * 2) - 1

        # stochastic
        self.trend = stoch_df[[stochastic]]

        # name
        if self.trend.shape[1] == 1:
            self.trend.columns = [f"{inspect.currentframe().f_code.co_name}_{self.window_size}"]

        # sort index
        self.trend = self.trend.sort_index()

        return self.trend

    def intensity(self) -> pd.DataFrame:
        """
        Computes intraday intensity trend factor.

        Returns
        -------
        intensity: pd.DataFrame
            DataFrame with DatetimeIndex and intensity values (cols).
        """
        # check df fields
        if 'high' not in self.df.columns or 'low' not in self.df.columns:
            raise ValueError("High and low price series must be provided in dataframe.")

        # compute true range
        hilo = self.df.high - self.df.low
        if isinstance(self.df.index, pd.MultiIndex):
            hicl = abs(self.df.high.sort_index(level=1) - self.df.close.groupby(level=1).shift(1))
            locl = abs(self.df.low.sort_index(level=1) - self.df.close.groupby(level=1).shift(1))
        else:
            hicl = abs(self.df.high - self.df.close.shift(1))
            locl = abs(self.df.low - self.df.close.shift(1))
        tr = pd.concat([hilo, hicl, locl], axis=1).max(axis=1)

        # normalized chg
        chg = self.df.close - self.df.open
        intensity = chg / tr

        # intensity
        self.trend = Transform(intensity).smooth(self.window_size,
                                                 window_type=self.window_type,
                                                 central_tendency=self.central_tendency,
                                                 window_fcn=self.window_fcn)

        # name
        self.trend = self.trend.to_frame(f"{inspect.currentframe().f_code.co_name}_{self.window_size}")

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
        if self.short_window_size is None:
            self.short_window_size = int(np.ceil(np.sqrt(self.window_size)))  # sqrt of long-term window

        # smoothing
        self.trend = Transform(self.price).smooth(self.short_window_size,
                                                  window_type=self.window_type,
                                                  central_tendency=self.central_tendency,
                                                  window_fcn=self.window_fcn) - \
            Transform(self.price).smooth(self.window_size,
                                         window_type=self.window_type,
                                         central_tendency=self.central_tendency,
                                         window_fcn=self.window_fcn,
                                         lags=self.short_window_size)

        # normalize
        if self.normalize:
            self.compute_dispersion()
            self.trend = self.trend.div(self.disp, axis=0)

        # name
        if self.trend.shape[1] == 1:
            self.trend.columns = [f"{inspect.currentframe().f_code.co_name}_"
                                  f"{self.short_window_size}_{self.window_size}"]

        # sort index
        self.trend = self.trend.sort_index()

        return self.trend

    def ewma_diff(self,
                  s_k: list = [2, 4, 8],
                  l_k: list = [6, 12, 24],
                  signal: bool = False
                  ) -> pd.DataFrame:
        """
        Computes the exponentially weighted moving average (EWMA) crossover trend factor.

        A CTA-momentum signal, based on the cross-over of multiple exponentially weighted moving averages (EWMA) with
        different half-lives.

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
        # check df index
        if isinstance(self.price.index, pd.MultiIndex) and self.price.shape[1] > 1:
            raise ValueError("Select a price series/col.")
        elif not isinstance(self.price.index, pd.MultiIndex):
            self.price = self.price.stack().to_frame()
            self.price.index.names = ['date', 'ticker']

        # half-life lists for short and long windows
        hl_s = [np.log(0.5) / np.log(1 - 1 / i) for i in s_k]
        hl_l = [np.log(0.5) / np.log(1 - 1 / i) for i in l_k]

        # create emtpy df
        factor_df = pd.DataFrame()

        # compute ewma diff for short, medium and long windows
        for i in range(0, len(s_k)):
            factor_df[f"x_k{i}"] = (self.price.unstack().ewm(halflife=hl_s[i]).mean() -
                                    self.price.unstack().ewm(halflife=hl_l[i]).mean()).stack(future_stack=True)

        # normalize by std of price
        for i in range(0, len(s_k)):
            factor_df[f"y_k{i}"] = (factor_df[f"x_k{i}"].unstack() /
                                    self.price.unstack().rolling(90).std()).stack(future_stack=True)

        # normalize by normalized y_k diff
        for i in range(0, len(s_k)):
            factor_df[f"z_k{i}"] = (factor_df[f"x_k{i}"].unstack() /
                                    factor_df[f"x_k{i}"].unstack().rolling(365).std()).stack(future_stack=True)

        # convert to signal
        if signal:
            for i in range(0, len(s_k)):
                factor_df[f"signal_k{i}"] = (factor_df[f"z_k{i}"] * np.exp((-1 * factor_df[f"z_k{i}"] ** 2) / 4)) / 0.89

        # mean of short, medium and long window signals
        ewma_diff = factor_df.iloc[:, -3:].mean(axis=1)
        # replace inf
        ewma_diff.replace([np.inf, -np.inf], np.nan, inplace=True)

        # ffill NaNs
        if isinstance(self.price.index, pd.MultiIndex):
            self.trend = ewma_diff.groupby(level=1).ffill()
        else:
            self.trend = ewma_diff.ffill()

        # name
        self.trend = ewma_diff.to_frame(f"{inspect.currentframe().f_code.co_name}_{s_k[0]}")

        # sort index
        self.trend = self.trend.sort_index()

        # single index
        if isinstance(self.df.index, pd.MultiIndex) is False:
            self.trend = self.trend.iloc[:, 0].unstack()

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
        # compute speed and returns
        speed = Transform(self.price).diff(lags=self.window_size)
        ret = Transform(self.price).diff()

        # mass
        if mass_method == 'vol':  # volatility
            mass = Transform(ret).compute_std(window_type=self.window_type, window_size=self.window_size)
        else:
            # VaR
            left_tail = Transform(ret).compute_quantile(perc, window_type='rolling', window_size=self.window_size) * -1
            right_tail = Transform(ret).compute_quantile(1 - perc, window_type='rolling', window_size=self.window_size)
            mass = pd.DataFrame(np.where(speed > 0, left_tail, right_tail), speed.index, speed.columns)

        # energy
        self.trend = speed.multiply(mass.values, axis=0)

        # name
        if self.trend.shape[1] == 1:
            self.trend.columns = [f"{inspect.currentframe().f_code.co_name}_{self.window_size}"]

        # sort index
        self.trend = self.trend.sort_index()

        return self.trend

    def snr(self) -> pd.DataFrame:
        """
        Computes the signal-to-noise ratio.

        Returns
        -------
        snr: pd.DataFrame
            DataFrame with DatetimeIndex and signal-to-noise ratio values (cols).
        """
        # sign, chg and roll_chg
        if isinstance(self.price.index, pd.MultiIndex):
            chg = self.price.groupby(level=1).diff(self.window_size).sort_index()
            abs_roll_chg = self.price.groupby(level=1).diff().abs().\
                groupby(level=1).rolling(self.window_size).sum().droplevel(0).sort_index()
        else:
            chg = self.price.diff(self.window_size)
            abs_roll_chg = np.abs(self.price.diff()).rolling(self.window_size).sum()

        # compute snr
        self.trend = chg / abs_roll_chg

        # name
        if self.trend.shape[1] == 1:
            self.trend.columns = [f"{inspect.currentframe().f_code.co_name}_{self.window_size}"]

        # sort index
        self.trend = self.trend.sort_index()

        return self.trend

    def adx(self, signal: bool = True) -> pd.DataFrame:
        """
        Computes the average directional index (ADX) of a price series.

        Parameters
        ----------
        signal: bool, default True
            Converts DI- and DI+ to a signal between -1 and 1.
            Otherwise, returns ADX.

        Returns
        -------
        adx: pd.DataFrame
            DataFrame with DatetimeIndex and average directional index values (cols).
        """
        # check df fields
        if 'high' not in self.df.columns or 'low' not in self.df.columns:
            raise ValueError("High and low price series must be provided in dataframe.")

        # compute true range
        hilo = self.df.high - self.df.low
        if isinstance(self.df.index, pd.MultiIndex):
            hicl = abs(self.df.high.sort_index(level=1) - self.df.close.groupby(level=1).shift(1))
            locl = abs(self.df.low.sort_index(level=1) - self.df.close.groupby(level=1).shift(1))
        else:
            hicl = abs(self.df.high - self.df.close.shift(1))
            locl = abs(self.df.low - self.df.close.shift(1))
        tr = pd.concat([hilo, hicl, locl], axis=1).max(axis=1).to_frame('tr').sort_index()

        # high and low price change
        if isinstance(self.df.index, pd.MultiIndex):
            high_chg = self.df.high.groupby(level=1).diff()
            low_chg = self.df.low.groupby(level=1).shift(1) - self.df.low.groupby(level=1).shift(0)
        else:
            high_chg = self.df.high.diff()
            low_chg = self.df.low.shift(1) - self.df.low

        # compute +DM and -DM
        dm_pos = pd.DataFrame(np.where(high_chg > low_chg, high_chg, 0), index=self.df.index, columns=['dm_pos'])
        dm_neg = pd.DataFrame(np.where(low_chg > high_chg, low_chg, 0), index=self.df.index, columns=['dm_neg'])

        # compute directional movement index
        dm_pos = Transform(dm_pos).smooth(self.window_size, window_type=self.window_type,
                                          central_tendency=self.central_tendency, window_fcn=self.window_fcn)
        dm_neg = Transform(dm_neg).smooth(self.window_size, window_type=self.window_type,
                                          central_tendency=self.central_tendency, window_fcn=self.window_fcn)
        tr = Transform(tr).smooth(self.window_size, window_type=self.window_type,
                                  central_tendency=self.central_tendency, window_fcn=self.window_fcn)

        # compute directional index
        di_pos = 100 * dm_pos.div(tr.squeeze(), axis=0)
        di_neg = 100 * dm_neg.div(tr.squeeze(), axis=0)

        # compute directional index difference
        di_diff = di_pos.subtract(di_neg.squeeze(), axis=0)
        di_sum = di_pos.add(di_neg.squeeze(), axis=0)
        self.trend = 100 * di_diff.div(di_sum.squeeze(), axis=0)

        # compute ADX
        adx = Transform(self.trend.abs()).smooth(self.window_size,
                                                 window_type=self.window_type,
                                                 central_tendency=self.central_tendency,
                                                 window_fcn=self.window_fcn)

        # name
        if self.trend.shape[1] == 1:
            self.trend.columns = [f"{inspect.currentframe().f_code.co_name}_{self.window_size}"]

        # sort index
        self.trend = self.trend.sort_index()

        if signal:
            self.trend = (self.trend / 100).clip(-1, 1)
        else:
            self.trend = adx

        return self.trend


# TODO: add hurst exponent, fractal dimension, detrended fluctuation analysis, wavelet transform, etc.
# def he(series: pd.Series, window_size: int):
#     """
#
#     Parameters
#     ----------
#     series: pd.Series
#         Series with time series.
#     window_size: int
#         Lookback window for hurst exponent calculation.
#
#     Returns
#     -------
#     hurst_exp: float
#         Hurst exponent.
#     """
#     # create the range of lag values
#     lags = range(2, window_size)
#
#     # calculate the array of the variances of the lagged differences
#     tau = [np.sqrt(np.std(np.subtract(series[lag:].values, series[:-lag].values))) for lag in lags]
#
#     # use a linear fit to estimate the Hurst Exponent
#     poly = np.polyfit(np.log(lags), np.log(tau), 1)
#
#     return poly[0] * 2.0
#
#
# def hurst(df: pd.DataFrame, window_size: int = 365) -> pd.DataFrame:
#     """
#     Computes the hurst exponent of a time series.
#
#     Parameters
#     ----------
#     df: pd.DataFrame
#         DataFrame with time series.
#     window_size: int
#         Lookback window for hurst exponent calculation.#
#     Returns
#     -------
#     hurst: pd.DataFrame
#         DataFrame with Hurst exponents for each series/col.
#     """
#     # convert to df
#     if isinstance(df, pd.Series):
#         df = df.to_frame()
#
#     # store results in df
#     res_df = pd.DataFrame()
#
#     # loop through cols
#     for col in df.columns:
#         hurst_exp = {'hurst': he(df[col], window_size)}
#         res_df = res_df.append(hurst_exp, ignore_index=True)
#
#     # add index
#     res_df.index = [df.columns]
#
#     return res_df.sort_values(by='hurst')
