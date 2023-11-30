from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Union


from factorlab.feature_analysis.time_series_analysis import linear_reg
from factorlab.feature_engineering.transform import Transform


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
                 smoothing: str = None,
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
        smoothing: str, {'median', 'smw', 'ewm'}, default None
            Smoothing method to use.
        lags: int, default 0
            Number of periods to lag values by.
        """
        self.df = df.astype(float)
        self.vwap = vwap
        self.log = log
        self.st_lookback = st_lookback
        self.lookback = lookback
        self.lt_lookback = lt_lookback
        self.smoothing = smoothing
        self.lags = lags

    def breakout(self) -> Union[pd.Series, pd.DataFrame]:
        """
         Compute breakout signal.

         Returns
         -------
         signal: pd.Series or pd.DataFrame - MultiIndex
             Series with DatetimeIndex (level 0), ticker (level 1) and breakout signal (cols).
         """
        df = self.df

        # vwap
        if self.vwap:
            df = Transform(df).vwap()['vwap'].to_frame()

        # create signal df
        signal = pd.DataFrame(data=0, index=df.index, columns=df.columns)

        # compute rolling high and low for lookback window
        # MultiIndex
        if isinstance(df.index, pd.MultiIndex):
            high = df.groupby(level=1).rolling(self.lookback).max().shift(1).droplevel(0)
            low = df.groupby(level=1).rolling(self.lookback).min().shift(1).droplevel(0)

        # single index
        else:
            high = df.rolling(self.lookback).max().shift(1)
            low = df.rolling(self.lookback).min().shift(1)

        # compute breakout signal
        signal[df.sort_index(level=1) > high] = 1
        signal[df.sort_index(level=1) < low] = -1

        # MultiIndex
        if isinstance(df.index, pd.MultiIndex):
            # replace fcn
            rep_fcn = lambda x: x.replace(to_replace=0, method='ffill')
            # ffill 0s to create always invested strategy
            signal = signal.groupby(level=1, group_keys=False).apply(rep_fcn)
        else:
            # ffill 0s to create always invested strategy
            signal.replace(to_replace=0, method='ffill', inplace=True)

        return signal.sort_index()

    def price_mom(self):
        """
        Computes the price momentum trend factor.

        Returns
        -------
        price_mom: pd.Series or pd.DataFrame - MultiIndex
           Series with DatetimeIndex (level 0), tickers (level 1) and price momentum trend factor values (cols).
        """
        df = self.df

        # vwap
        if self.vwap:
            df = Transform(df).vwap()['vwap'].to_frame()

        # log
        df = Transform(df).log()

        # price mom
        if isinstance(df.index, pd.MultiIndex):
            price_mom = df.groupby(level=1).diff(self.lookback)
        else:
            price_mom = df.diff(self.lookback)  # price series

        return price_mom.sort_index()

    def divergence(self) -> Union[pd.Series, pd.DataFrame]:
        """
        Compute divergence measure.

        Returns
        -------
        div: pd.Series or pd.DataFrame - MultiIndex
           Series or dataframe with DatetimeIndex (level 0), tickers (level 1) and divergence values (cols).
        """
        df = self.df

        # vwap
        if self.vwap:
            df = Transform(df).vwap()['vwap'].to_frame()

        # log
        df = Transform(df).log()

        # compute sign and divergence
        if isinstance(df.index, pd.MultiIndex):  # multiindex
            sign = np.sign(df.groupby(level=1).diff())
        else:
            sign = np.sign(df.diff)

        div = Transform(sign).smooth(lookback=self.lookback, method=self.smoothing)

        return div

    def time_trend(self, coef: Union[str, list] = 'trend') -> Union[pd.Series, pd.DataFrame]:
        """
        Compute the price dynamics indicator by regressing price on a time trend to estimate coefficients.

        Parameters
        ----------
        coef: str, {'const', 'trend', ['const', 'trend'], ['const', 'trend', 'trend_squared'],
            ['const', 'trend', 'trend_squared', 'trend_cubed']}, default 'trend_squared'
            Coefficient values to keep from regression.

        Returns
        -------
        coeff: pd.Series or pd.DataFrame - MultiIndex
            Series with DatetimeIndex (level 0), tickers (level 1) and selected coefficients of price regressed
            on a time trend over the lookback window.
        """
        df = self.df

        # vwap
        if self.vwap:
            df = Transform(df).vwap()['vwap'].to_frame()

        # log
        if self.log:
            df = Transform(df).log()

        # fit linear regression
        if isinstance(df.index, pd.MultiIndex):  # multiindex
            coeff = df.groupby(level=1, group_keys=False).apply(
                lambda x: linear_reg(x, None, window_type='rolling', output='coef', log=False, trend='ct',
                                     lookback=self.lookback))
        else:  # single index
            coeff = linear_reg(df, None, window_type='rolling', output='coef', log=False, trend='ct',
                               lookback=self.lookback)

        # return coef col
        coeff = coeff[coef]

        return coeff.sort_index()

    def price_dynamics(self, coef: Union[str, list] = 'trend_squared') -> Union[pd.Series, pd.DataFrame]:
        """
        Compute the price dynamics indicator by regressing price on a time trend to estimate coefficients.

        Parameters
        ----------
        coef: str, {'const', 'trend', 'trend_squared', 'price_dyn',
                    ['const', 'trend'], ['const', 'trend', 'trend_squared'],
            ['const', 'trend', 'trend_squared', 'trend_cubed']}, default 'trend_squared'
            Coefficient values to keep from regression.

        Returns
        -------
        coeff: pd.Series or pd.DataFrame - MultiIndex
            Series with DatetimeIndex (level 0), tickers (level 1) and selected coefficients of price regressed
            on a time trend over the lookback window.
        """
        df = self.df

        # vwap
        if self.vwap:
            df = Transform(df).vwap()['vwap'].to_frame()

        # log
        if self.log:
            df = Transform(df).log()

        # fit linear regression
        if isinstance(df.index, pd.MultiIndex):  # multiindex
            coeff = df.groupby(level=1, group_keys=False).apply(
                lambda x: linear_reg(x, None, window_type='rolling', output='coef', log=False, trend='ctt',
                                     lookback=self.lookback))
        else:  # single index
            coeff = linear_reg(df, None, window_type='rolling', output='coef', log=False, trend='ctt',
                               lookback=self.lookback)

        # price dyn
        if coef == 'price_dyn':
            coeff = Transform(coeff).normalize_ts(window_type='expanding').mean(axis=1).to_frame('price_dyn')

        # return coef col
        coeff = coeff[coef]

        return coeff.sort_index()

    def alpha_mom(self) -> Union[pd.Series, pd.DataFrame]:
        """
        Constant term (alpha) from fitting an OLS linear regression of price on the market beta,
        i.e. cross-sectional average, index or PC1).

        Returns
        -------
        resid: pd.Series or pd.DataFrame - MultiIndex
            DataFrame with DatetimeIndex (level 0), tickers (level 1) and residuals.
       """
        df = self.df

        # vwap
        if self.vwap:
            df = Transform(df).vwap()['vwap'].to_frame('price')
        else:
            df = df.close.to_frame('price')
        #
        # # log
        # df = Transform(df).log()

        # compute ret
        y = Transform(df).returns().price.rename('y')
        # compute market beta ret X and market ret y
        X = y.groupby(level=0).mean().rename('X')
        # merge y, X
        data = pd.merge(y, X, right_index=True, left_index=True).dropna()

        # fit linear regression
        alpha = data.groupby(level=1, group_keys=False).apply(
            lambda x: linear_reg(x.y, x.X, window_type='rolling', output='coef', log=False, trend='c',
                                 lookback=self.lookback))['const']
        # change col name
        alpha.columns = ['alpha']

        return alpha.sort_index()

    def rsi(self, signal: bool = True) -> Union[pd.Series, pd.DataFrame]:
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
        df = self.df

        # vwap
        if self.vwap:
            df = Transform(df).vwap()['vwap'].to_frame()

        # log
        df = Transform(df).log()

        # compute price returns and up/down days
        if isinstance(df.index, pd.MultiIndex):  # multiindex
            ret = df.groupby(level=1, group_keys=False).diff()
        else:  # single index
            ret = df.diff()

        # get up and down days
        up = ret.where(ret > 0).fillna(0)
        down = abs(ret.where(ret < 0).fillna(0))

        # smoothing
        rs = Transform(up).smooth(lookback=self.lookback, method=self.smoothing) / \
            Transform(down).smooth(lookback=self.lookback, method=self.smoothing)

        # normalization to remove inf 0 div
        rs = 100 - (100 / (1 + rs))
        # signal
        if signal:
            rs = (rs - 50)/50

        return rs.sort_index()

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
        df = self.df

        # st lookback
        if self.st_lookback is None:
            self.st_lookback = max(2, round(self.lookback / 4))

        # compute k
        if isinstance(df.index, pd.MultiIndex):
            num = df.close.sort_index(level=1) - df.low.groupby(level=1).rolling(self.lookback).min().values
            denom = (df.high.groupby(level=1).rolling(self.lookback).max() -
                     df.low.groupby(level=1).rolling(self.lookback).min().values).droplevel(0)
            k = num/denom

        else:
            k = (df.close - df.low.rolling(self.lookback).min()) / \
                (df.high.rolling(self.lookback).max() - df.low.rolling(self.lookback).min())

        # clip extreme values
        k = k.clip(0, 1)
        # smoothing
        d = Transform(k).smooth(lookback=self.st_lookback, method=self.smoothing)

        # convert to signal
        if signal:
            k, d = (k * 2)-1, (d*2)-1

        if isinstance(k, pd.Series):
            sdf = pd.DataFrame({'stochastic_k': k, 'stochastic_d': d})
        else:
            sdf = pd.DataFrame(columns=pd.MultiIndex.from_product([['stochastic_k', 'stochastic_d'],
                                                                   self.df.close.columns]))
            sdf['stochastic_k'] = k
            sdf['stochastic_d'] = d

        # stochastic
        if stochastic == 'k' or stochastic == 'd':
            sdf = sdf['stochastic_' + stochastic]

        return sdf.sort_index()

    def intensity(self) -> Union[pd.Series, pd.DataFrame]:
        """
        Computes intraday intensity trend factor.

        Returns
        -------
        intensity: pd.Series or pd.DataFrame - MultiIndex
            Series or dataframe with DatetimeIndex (level 0), tickers (level 1) and
            Intensity trend factor values (cols).
        """
        df = self.df

        # compute true range
        hilo = df.high - df.close

        if isinstance(df.index, pd.MultiIndex):
            hicl = abs(df.high.sort_index(level=1) - df.close.groupby(level=1).shift(1))
            locl = abs(df.low.sort_index(level=1) - df.close.groupby(level=1).shift(1))
        else:
            hicl = abs(df.high - df.close.shift(1))
            locl = abs(df.low - df.close.shift(1))

        # compute ATR, chg and intensity
        tr = pd.concat([hilo, hicl, locl], axis=1).max(axis=1)  # compute ATR
        today_chg = df.close - df.open  # compute today's change
        intensity = today_chg / tr  # compute intensity

        # smoothing
        intensity_smooth = Transform(intensity).smooth(lookback=self.lookback, method=self.smoothing)

        return intensity_smooth.sort_index()

    def mw_diff(self) -> Union[pd.Series, pd.DataFrame]:
        """
        Computes the moving window difference trend factor.

        Returns
        -------
        mw_diff: pd.Series or pd.DataFrame - MultiIndex
            Series with DatetimeIndex (level 0), ticker (level 1) and
             moving window difference trend factor values (cols).
        """
        df = self.df

        # vwap
        if self.vwap:
            df = Transform(df).vwap()['vwap'].to_frame()

        # log
        if self.log:
            df = Transform(df).log()

        # short rolling window param
        if self.st_lookback is None:
            self.st_lookback = int(np.ceil(self.lookback / 3))

        # smoothing
        mw_diff = Transform(df).smooth(lookback=self.st_lookback, method=self.smoothing) - \
            Transform(df).smooth(lookback=self.lookback, method=self.smoothing, lags=self.lags)

        return mw_diff.sort_index()

    def ewma_wxover(self,
                    s_k: list = [2, 4, 8],
                    l_k: list = [6, 12, 24],
                    signal: bool = False
                    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Computes the moving window difference trend factor.

        Computed as described in Dissecting Investment Strategies in the Cross Section and Time Series:
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
        df = self.df

        # vwap
        if self.vwap:
            df = Transform(df).vwap()['vwap'].to_frame('price')
        else:
            df = df.close.to_frame('price')

        # log
        if self.log:
            df = Transform(df).log()

        # half-life lists for short and long windows
        hl_s = [np.log(0.5) / np.log(1 - 1 / i) for i in s_k]
        hl_l = [np.log(0.5) / np.log(1 - 1 / i) for i in l_k]

        # create emtpy df
        factor_df = pd.DataFrame()

        # compute ewma diff for short, medium and long windows
        for i in range(0, len(s_k)):
            if isinstance(df.index, pd.MultiIndex):
                factor_df['x_k' + str(i)] = (df.groupby(level=1).ewm(halflife=hl_s[i]).mean() -
                                             df.groupby(level=1).ewm(halflife=hl_l[i]).mean()).droplevel(0)
            else:
                factor_df['x_k' + str(i)] = df.ewm(halflife=hl_s[i]).mean() - \
                                            df.ewm(halflife=hl_l[i]).mean()

        # normalize by std of price
        for i in range(0, len(s_k)):
            if isinstance(factor_df.index, pd.MultiIndex):
                factor_df['y_k' + str(i)] = factor_df['x_k' + str(i)].\
                    divide(df.price.groupby(level=1).rolling(90).std().droplevel(0), axis=0)
            else:
                factor_df['y_k' + str(i)] = factor_df['x_k' + str(i)].divide(df.price.rolling(90).std(), axis=0)

        # normalize by normalized y_k diff
        for i in range(0, len(s_k)):
            if isinstance(self.df .index, pd.MultiIndex):
                factor_df['z_k' + str(i)] = factor_df['x_k' + str(i)] / \
                                            factor_df['x_k' + str(i)].groupby(level=1).rolling(365).std().values
            else:
                factor_df['z_k' + str(i)] = factor_df['x_k' + str(i)] / factor_df['x_k' + str(i)].rolling(365).std()

        # convert to signal
        if signal:
            for i in range(0, len(s_k)):
                factor_df['signal_k' + str(i)] = (factor_df['z_k' + str(i)] * np.exp(
                    (-1 * factor_df['z_k' + str(i)] ** 2) / 4)) / 0.89

        # mean of short, medium and long window signals
        ewma_xover = factor_df.iloc[:, -3:].mean(axis=1)
        # replace inf
        ewma_xover.replace([np.inf, -np.inf], np.nan, inplace=True)
        # ffill NaNs
        if isinstance(ewma_xover.index, pd.MultiIndex):
            ewma_xover = ewma_xover.groupby(level=1).ffill()
        else:
            ewma_xover = ewma_xover.ffill()

        return ewma_xover.rename('ewma_xover').sort_index()
