import pandas as pd
import numpy as np
from typing import Union, Optional
from scipy import stats
from sklearn.preprocessing import KBinsDiscretizer


class Transform:
    """
    Transformations on raw data, returns or features.
    """

    def __init__(self, df: Union[pd.Series, pd.DataFrame]):
        """
        Constructor

        Parameters
        ----------
        df: series or pd.DataFrame - Single or MultiIndex
            DataFrame MultiIndex with DatetimeIndex (level 0), ticker (level 1), field values (cols).
        """
        self.df = df.astype(float)

    def vwap(self) -> Union[pd.Series, pd.DataFrame]:
        """
        Computes volume-weighted average price from OHLCV prices.

        Returns
        -------
        vwap: pd.Series or pd.DataFrame
            Series or dataframe with DatetimeIndex (level 0), tickers (level 1) and
            volume-weighted average price from OHLCV prices (cols).
        """
        df = self.df

        # check if OHLC
        if not all([col in self.df.columns for col in ['open', 'high', 'low', 'close']]):
            raise ValueError("Dataframe must have open, high, low and close fields to compute vwap.")

        # compute vwap
        df['vwap'] = (df.close + (df.open + df.high + df.low)/3) / 2

        return df

    def log(self) -> Union[pd.Series, pd.DataFrame]:
        """
        Computes log of price.

        Returns
        -------
        df: pd.Series or pd.DataFrame - MultiIndex
            Series or dataframe with DatetimeIndex (level 0), tickers (level 1) and log of price (cols).
        """
        df = self.df

        # MultiIndex
        if isinstance(df.index, pd.MultiIndex):
            rep_fcn = lambda x: np.log(x).replace([np.inf, -np.inf], np.nan)
            df = df.groupby(level=1, group_keys=False).apply(rep_fcn).ffill()
        else:  # single
            df = np.log(df).replace([np.inf, -np.inf], np.nan).ffill()

        return df

    def returns(self,
                lags: int = 1,
                forward: bool = False,
                method: str = 'log',
                market: Optional[bool] = None,
                mkt_weighting: Optional[str] = None
                ) -> pd.DataFrame:
        """
        Computes the log returns of a price series.

        Parameters
        ----------
        lags: int, default 1
            Interval over which to compute returns: 1 computes daily return, 5 weekly returns, etc.
        forward: bool, default False
            Shifts returns forward by number of periods specified in lags.
        method: str, {'simple', 'log'}, default 'log'
            Method to compute returns, continuous or simple
        market: bool, Optional, default None
            Computes the returns of the entire universe of asset prices, or the market return.
        mkt_weighting: str, optional, default None
            Weighting method to use to compute market return.

        Returns
        -------
        ret: Series or DataFrame - Single or MultiIndex
            Series or DataFrame with DatetimeIndex and returns series.
        """
        df = self.df

        if isinstance(df, pd.Series):
            df = df.to_frame()
        if isinstance(df.index, pd.MultiIndex):
            df = df.unstack()

        if method == 'simple':
            # simple return
            ret = df.pct_change(lags)
        else:
            # log returns
            ret = np.log(df) - np.log(df).shift(lags)

        if forward:
            # forward ret
            ret = ret.shift(lags * -1)

        if market:  # market return
            if mkt_weighting == 'inv_vol':
                pass
            else:
                ret = ret.mean(axis=1)

        if isinstance(self.df.index, pd.MultiIndex):
            ret = ret.stack()

        return ret

    def target_vol(self, ann_vol: float = 0.15, ann_factor: float = 252) -> pd.DataFrame:
        """
        Set volatility of returns to be equal to a specific vol target.

        Parameters
        ----------
        ann_vol: float, default 0.15
            Target annualized volatility.
        ann_factor: float, {12, 52, 252, 365}, default 365
            Annualization factor.

        Returns
        -------
        df: DataFrame
            DataFrame with vol-adjusted returns.
        """
        df = self.df

        if isinstance(df, pd.Series):
            df = df.to_frame()
        if isinstance(df.index, pd.MultiIndex):
            df = df.unstack()

        # set target vol
        norm_factor = 1 / ((df.std() / ann_vol) * np.sqrt(ann_factor))
        df = df * norm_factor

        if isinstance(self.df.index, pd.MultiIndex):
            df = df.stack()

        return df

    def smooth(self,
               lookback: int = None,
               method: str = 'smw',
               lags: int = 0
               ) -> Union[pd.Series, pd.DataFrame]:
        """
        Smooths values using 'smoothing' method.

        Parameters
        ----------
        lookback: int
            Number of observations in moving window.
        method: str, {'median', 'swm', 'ewm'}, default 'swm'
            Method used for smoothing/filtering.
        lags: int, default None
            Number of periods by which to lag values.

        Returns
        -------
        df: pd.Series or pd.DataFrame
            Series or DataFrame with DatetimeIndex (level 0), tickers (level 1) and smoothed values (cols).
        """
        df = self.df

        if isinstance(df.index, pd.MultiIndex):  # multiindex
            if method == 'median':
                df = df.groupby(level=1).rolling(lookback, min_periods=2).median().shift(lags).droplevel(0)
            elif method == 'ewm':
                df = df.groupby(level=1).ewm(span=lookback, min_periods=2).mean().shift(lags).droplevel(0)
            else:
                df = df.groupby(level=1).rolling(lookback, min_periods=2).mean().shift(lags).droplevel(0)

        else:  # single index
            if method == 'median':
                df = df.rolling(lookback, min_periods=2).median().shift(lags)
            elif method == 'ewm':
                df = df.ewm(span=lookback, min_periods=2).mean().shift(lags)
            else:
                df = df.rolling(lookback, min_periods=2).mean().shift(lags)

        return df

    @staticmethod
    def normalize(df: Union[pd.Series, pd.DataFrame],
                  centering: bool = True,
                  method: str = 'z-score',
                  window_type: str = 'fixed',
                  lookback: int = 10,
                  min_periods: int = 2,
                  winsorize: Optional[int] = None
                  ) -> pd.DataFrame:
        """
        Normalizes features.

        Parameters
        ----------
        df: pd.DataFrame or pd.Series
            Dataframe or series to normalize.
        centering: bool, default True
            Centers values using the appropriate measure of central tendency used for the selected method. Otherwise,
            0 is used.
        method: str, {'z-score', 'cdf', iqr', 'mod_z', 'min-max', 'percentile'}, default 'z-score'
                z-score: subtracts mean and divides by standard deviation.
                cdf: cumulative distribution function rescales z-scores to values between 0 and 1.
                iqr:  subtracts median and divides by interquartile range.
                mod_z: modified z-score using median absolute deviation.
                min-max: rescales to values between 0 and 1 by subtracting the min and dividing by the range.
                percentile: converts values to their percentile rank relative to the observations in the
                defined window type.
        window_type: str, {'fixed', 'expanding', 'rolling}, default 'fixed'
            Provide a window type. If None, all observations are used in the calculation.
        lookback: int, default 10
            Size of the moving window. This is the minimum number of observations used for the rolling or
            expanding statistic.
        min_periods: int, default 2
            Minimum number of observations in window required to have a value; otherwise, result is np.nan.
        winsorize: int, default None
            Winsorizes/clips values to between positive and negative values of specified integer.

        Returns
        -------
        norm_features: Series or DataFrame
            Dataframe with DatetimeIndex and normalized features
        """
        # convert type and to df
        df = df.astype(float)
        if isinstance(df, pd.Series):
            df = df.to_frame()

        # window type
        if window_type == 'rolling':
            mov_df = getattr(df, window_type)(lookback, min_periods=min_periods)
        elif window_type == 'expanding':
            mov_df = getattr(df, window_type)(min_periods=min_periods)
        else:
            mov_df = df

        # centering
        if centering is True and method in ['z-score', 'cdf']:
            center = mov_df.mean()
        elif centering is True and method in ['iqr', 'mod_z']:
            center = mov_df.median()
        else:
            center = 0

        # methods
        norm_df = None
        if method == 'z-score':
            norm_df = (df - center) / mov_df.std()
        elif method == 'iqr':
            norm_df = (df - center) / (mov_df.quantile(0.75) - mov_df.quantile(0.25))
        elif method == 'mod_z':
            ad = (df - center).abs()
            if window_type != 'fixed':
                mad = getattr(ad, window_type)(lookback).median()
            else:
                mad = ad.median()
            norm_df = 0.6745 * (df - center) / mad
        elif method == 'cdf':
            norm_df = (df - center) / mov_df.std()
            norm_df = pd.DataFrame(stats.norm.cdf(norm_df), index=norm_df.index, columns=norm_df.columns)
        elif method == 'min-max':
            norm_df = (df - mov_df.min()) / (mov_df.max() - mov_df.min())
        elif method == 'percentile':
            if window_type != 'fixed':
                norm_df = mov_df.apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
            else:
                norm_df = mov_df.apply(lambda x: pd.Series(x).rank(pct=True))

        # winsorize
        if winsorize is not None and method in ['z-score', 'iqr', 'mod_z']:
            norm_df = norm_df.clip(winsorize * -1, winsorize)

        return norm_df

    def normalize_ts(self,
                     centering: bool = True,
                     method: str = 'z-score',
                     window_type: str = 'fixed',
                     lookback: int = 10,
                     min_periods: int = 2,
                     winsorize: Optional[int] = None
                     ) -> pd.DataFrame:
        """
        Normalizes features over the time series (rows).

        Parameters
        ----------
        centering: bool, default True
            Centers values using the appropriate measure of central tendency used for the selected method. Otherwise,
            0 is used.
        method: str, {'z-score', 'cdf', iqr', 'min-max', 'percentile'}, default 'z-score'
            z-score: subtracts mean and divides by standard deviation.
            iqr:  subtracts median and divides by interquartile range.
            mod_z: modified z-score using median absolute deviation.
            cdf: cumulative distribution function rescales z-scores to values between 0 and 1.
            percentile: converts values to their percentile rank relative to the observations in the
            defined window type.
            min-max: rescales to values between 0 and 1 by subtracting the min and dividing by the range.
        window_type: str, {'fixed', 'expanding', 'rolling}, default 'fixed'
            Provide a window type. If None, all observations are used in the calculation.
        lookback: int, default 10
            Size of the moving window. This is the minimum number of observations used for the rolling or
            expanding statistic.
        min_periods: int, default 2
            Minimum number of observations in window required to have a value; otherwise, result is np.nan.
        winsorize: int, default None
            Winsorizes/clips values to between positive and negative values of specified integer.

        Returns
        -------
        norm_features: Series or DataFrame - MultiIndex
            DatetimeIndex (level 0), tickers (level 1) and normalized features
        """
        if isinstance(self.df.index, pd.MultiIndex):
            norm_df = self.normalize(self.df.unstack(),
                                     centering=centering,
                                     window_type=window_type,
                                     lookback=lookback,
                                     min_periods=min_periods,
                                     method=method,
                                     winsorize=winsorize,
                                     ).stack()
        else:
            norm_df = self.normalize(self.df,
                                     centering=centering,
                                     window_type=window_type,
                                     lookback=lookback,
                                     min_periods=min_periods,
                                     method=method,
                                     winsorize=winsorize
                                     )

        return norm_df

    def normalize_cs(self,
                     centering: bool = True,
                     method: str = 'z-score'
                     ) -> Union[pd.Series, pd.DataFrame]:
        """
        Normalizes features over the cross-section (cols).

        Parameters
        ----------
        centering: bool, default True
            Centers values using the appropriate measure of central tendency used for the selected method. Otherwise,
            0 is used.
        method: str, {'z-score', 'cdf', iqr', 'min-max', 'percentile'}, default 'z-score'
                z-score: subtracts mean and divides by standard deviation.
                cdf: cumulative distribution function rescales z-scores to values between 0 and 1.
                iqr:  subtracts median and divides by interquartile range.
                mod_z: modified z-score using median absolute deviation.
                min-max: rescales to values between 0 and 1 by subtracting the min and dividing by the range.
                percentile: converts values to their percentile rank relative to the observations in the
                defined window type.

        Returns
        -------
        norm_features: Series or DataFrame - MultiIndex
            DatetimeIndex (level 0), tickers (level 1) and normalized features
        """
        norm_df = self.df.groupby(level=0, group_keys=False).apply(
            lambda x: self.normalize(x, centering=centering, method=method))

        return norm_df

    @staticmethod
    def quantize(df: Union[pd.Series, pd.DataFrame],
                 bins: int = 5,
                 window_type: str = 'fixed',
                 lookback: int = 10,
                 _min_periods: int = 2,
                 axis: int = 0,
                 ) -> pd.DataFrame:
        """
        Quantizes factors or targets. Quantization is the process of transforming a continuous variable
        into a discrete one by creating a set of bins (or equivalently contiguous intervals/cuttoffs) that spans
        the range of the variable’s values. Quantization creates an equal number of values in each bin.
        See Discretize function for other types of discretization.

        Parameters
        ----------
        df: pd.DataFrame or pd.Series
            Dataframe or series to quantize.
        bins: int, default 5
            Number of bins.
        window_type: str, {'fixed', 'expanding', 'rolling}, default 'fixed'
            Provide a window type. If None, all observations are used in the calculation.
        lookback: int, default 10
            Size of the moving window. This is the minimum number of observations used for the rolling or
            expanding statistic.
        _min_periods: int, default 2
            Minimum number of observations in window required to have a value; otherwise, result is np.nan.
        axis: int, default 0
            Axis along which to quantize. Defaults to each column.

        Returns
        -------
        quant_df: DataFrame
            Series or DataFrame with DatetimeIndex (index) and quantized features (columns).
        """
        if bins <= 1 or bins is None:
            raise ValueError('Number of bins must be larger than 1. Please increase number of bins.')
        if isinstance(df, pd.Series):
            df = df.to_frame()  # convert to df

        # quantize function
        def quantize(x):
            return pd.qcut(x, bins, labels=False, duplicates='drop') + 1

        def quant_mw(x):
            return (pd.qcut(x, bins, labels=False, duplicates='drop') + 1)[-1]

        # window type
        if window_type == 'rolling':
            quant_df = getattr(df, window_type)(lookback, min_periods=_min_periods).apply(quant_mw)
        elif window_type == 'expanding':
            quant_df = getattr(df, window_type)(min_periods=_min_periods).apply(quant_mw)
        else:
            quant_df = df.apply(quantize, axis=axis)

        return quant_df

    def quantize_ts(self,
                    bins: int = 5,
                    window_type: str = 'fixed',
                    lookback: int = 10,
                    _min_periods: int = 2) -> Union[pd.Series, pd.DataFrame]:
        """
        Quantizes features over the time series (rows).

        Parameters
        ----------
        bins: int, default 5
            Number of bins.
        window_type: str, {'fixed', 'expanding', 'rolling}, default 'fixed'
            Provide a window type. If None, all observations are used in the calculation.
        lookback: int, default 10
            Size of the moving window. This is the minimum number of observations used for the rolling or
            expanding statistic.
        _min_periods: int, default 2
            Minimum number of observations in window required to have a value; otherwise, result is np.nan.

        Returns
        -------
        quant_df: DataFrame
            Series or DataFrame with DatetimeIndex (level 0), tickers (level 1) and quantized features (columns).
        """
        # # drop empty cols
        # if isinstance(self.df, pd.DataFrame):
        #     df = self.df.dropna(how='all', axis=1)
        # else:
        #     df = self.df

        if isinstance(self.df.index, pd.MultiIndex):
            quant_df = self.quantize(self.df.unstack().dropna(how='all', axis=1),
                                     bins=bins,
                                     window_type=window_type,
                                     lookback=lookback,
                                     _min_periods=_min_periods).stack()
        else:
            quant_df = self.quantize(self.df.dropna(how='all', axis=1),
                                     bins=bins,
                                     window_type=window_type,
                                     lookback=lookback,
                                     _min_periods=_min_periods)

        return quant_df

    def quantize_cs(self, bins: int = 5) -> Union[pd.Series, pd.DataFrame]:
        """
        Quantizes features over the cross-section (cols).

        Parameters
        ----------
        bins: int, default 5
            Number of bins.

        Returns
        -------
        quant_df: DataFrame
            Series or DataFrame with DatetimeIndex (level 0), tickers (level 1), and quantized features (columns).
        """
        # df to store quantized cols
        quant_df = pd.DataFrame()
        # loop through cols
        for col in self.df.columns:
            df = self.quantize(self.df[col].unstack().dropna(thresh=bins, axis=0), axis=1).stack().to_frame(col)
            quant_df = pd.concat([quant_df, df], axis=1)

        return quant_df.sort_index()

    @staticmethod
    def discretize(df: Union[pd.Series, pd.DataFrame],
                   discretization: str = 'quantile',
                   bins: int = 5,
                   window_type: str = 'fixed',
                   lookback: int = 7) -> pd.DataFrame:
        """
        Discretizes normalized factors or targets. Discretization is the process of transforming a continuous variable
        into a discrete one by creating a set of bins (or equivalently contiguous intervals/cuttoffs) that spans
        the range of the variable’s values.

        Parameters
        ----------
        df: pd.DataFrame
        discretization: str, {'quantile', 'uniform', 'kmeans'}, default 'quantile'
            quantile: all bins have the same number of values.
            uniform: all bins have identical widths.
            kmeans: values in each bin have the same nearest center of a 1D k-means cluster.
        bins: int, default 5
            Number of bins.
        window_type: str, {'fixed', 'expanding', 'rolling}, default 'fixed'
            Provide a window type. If None, all observations are used in the calculation.
        lookback: int, default 7
            Size of the moving window. This is the minimum number of observations used for the rolling or
            expanding statistic.

        Returns
        -------
        disc_features: DataFrame
            Series or DataFrame with DatetimeIndex and discretized features.
        """
        if bins <= 1 or bins is None:
            raise ValueError('Number of bins must be larger than 1. Please increase number of bins.')
        if isinstance(df, pd.Series):
            df = df.to_frame()  # convert series to df

        # discretize features
        discretize = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy=discretization)
        # window size
        window_size = lookback

        # rolling or expanding window
        if window_type != 'fixed':
            # create empty df to store values
            disc_df = pd.DataFrame(index=df.index, columns=df.columns)
            # discretize features in window
            while window_size <= df.shape[0]:
                # discretize features
                # rolling window
                if window_type == 'rolling':
                    disc_df.iloc[window_size - 1, :] = \
                        discretize.fit_transform(df.iloc[window_size - lookback:window_size])[-1]
                # expanding window
                if window_type == 'expanding':
                    disc_df.iloc[window_size - 1, :] = discretize.fit_transform(df.iloc[:window_size])[-1]
                window_size += 1
        # fixed window type
        else:
            disc_df = pd.DataFrame(discretize.fit_transform(df), index=df.index, columns=df.columns)

        # convert type to float
        disc_df = disc_df.astype(float) + 1

        return disc_df.round(2)

    def discretize_ts(self,
                      discretization: str = 'quantile',
                      bins: int = 5,
                      window_type: str = 'fixed',
                      lookback: int = 7) -> pd.DataFrame:
        """
        Discretizes features over the time series (rows).

        Parameters
        ----------
        discretization: str, {'quantile', 'uniform', 'kmeans'}, default 'quantile'
            quantile: all bins have the same number of values.
            uniform: all bins have identical widths.
            kmeans: values in each bin have the same nearest center of a 1D k-means cluster.
        bins: int, default 5
            Number of bins.
        window_type: str, {'fixed', 'expanding', 'rolling}, default 'fixed'
            Provide a window type. If None, all observations are used in the calculation.
        lookback: int, default 7
            Size of the moving window. This is the minimum number of observations used for the rolling or
            expanding statistic.

        Returns
        -------
        disc_features: DataFrame
            Series or DataFrame with DatetimeIndex and discretized features.
        """
        if isinstance(self.df.index, pd.MultiIndex):
            df = self.df.unstack().dropna(how='all', axis=1).dropna()  # dropna
            disc_df = self.discretize(df,
                                      discretization=discretization,
                                      bins=bins,
                                      window_type=window_type,
                                      lookback=lookback).stack()
        else:
            df = self.df.dropna(how='all', axis=1).dropna()  # dropna
            disc_df = self.discretize(df,
                                      discretization=discretization,
                                      bins=bins,
                                      window_type=window_type,
                                      lookback=lookback)
        return disc_df

    def discretize_cs(self,
                      discretization: str = 'quantile',
                      bins: int = 5
                      ) -> Union[pd.Series, pd.DataFrame]:
        """
        Discretizes features over the time series (rows).

        Parameters
        ----------
        discretization: str, {'quantile', 'uniform', 'kmeans'}, default 'quantile'
            quantile: all bins have the same number of values.
            uniform: all bins have identical widths.
            kmeans: values in each bin have the same nearest center of a 1D k-means cluster.
        bins: int, default 5
            Number of bins.

        Returns
        -------
        disc_features: DataFrame
            Series or DataFrame with DatetimeIndex and discretized features.
        """
        df = self.df.dropna(how='all', axis=1).dropna()  # dropna

        disc_df = df.groupby(level=0, group_keys=False).apply(
            lambda x: self.discretize(x, discretization=discretization, bins=bins))

        return disc_df
