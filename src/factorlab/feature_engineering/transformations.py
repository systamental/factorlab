import pandas as pd
import numpy as np
from typing import Union, Optional
from scipy import stats
from sklearn.preprocessing import KBinsDiscretizer


class Transform:
    """
    Transformations on raw data, returns or features.
    """
    def __init__(self, data: Union[pd.Series, pd.DataFrame, np.array]):
        """
        Constructor

        Parameters
        ----------
        data: pd.Series, pd.DataFrame or np.array
            Data to transform.
        """
        self.df = data.astype(float) if isinstance(data, pd.DataFrame) else data.astype(float).to_frame() \
            if isinstance(data, pd.Series) else None
        self.arr = data.to_numpy(dtype=np.float64) if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series) \
            else data.astype(float).values
        self.trans_df = None

    def vwap(self) -> Union[pd.Series, pd.DataFrame]:
        """
        Computes volume-weighted average price from OHLCV prices and add vwap price to dataframe.

        Returns
        -------
        vwap: pd.Series or pd.DataFrame
            Series or dataframe with DatetimeIndex (level 0), tickers (level 1) and
            volume-weighted average price from OHLCV prices (cols).
        """
        # transformed df
        self.trans_df = self.df.copy()

        # check if OHLC
        if not all([col in self.df.columns for col in ['open', 'high', 'low', 'close']]):
            raise ValueError("Dataframe must have open, high, low and close fields to compute vwap.")

        # compute vwap
        self.trans_df['vwap'] = (self.trans_df.close +
                                (self.trans_df.open + self.trans_df.high + self.trans_df.low)/3)/2

        return self.trans_df

    def log(self) -> Union[pd.Series, pd.DataFrame]:
        """
        Computes log of price.

        Returns
        -------
        df: pd.Series or pd.DataFrame - MultiIndex
            Series or dataframe with DatetimeIndex (level 0), tickers (level 1) and log of price (cols).
        """
        # transformed df
        self.trans_df = self.df.copy()
        # remove negative values
        self.trans_df[self.trans_df <= 0] = np.nan
        # log and replace inf
        self.trans_df = np.log(self.trans_df).replace([np.inf, -np.inf], np.nan)

        return self.trans_df

    def returns(self,
                lags: int = 1,
                forward: bool = False,
                method: str = 'log',
                market: Optional[bool] = False,
                mkt_field: str = 'close',
                mkt_weighting: Optional[str] = None
                ) -> pd.DataFrame:
        """
        Computes the log returns of a price series.

        Parameters
        ----------
        lags: int, default 1
            Interval over which to compute returns: 1 computes daily return, 5 weekly returns, 30 monthly returns, etc.
        forward: bool, default False
            Shifts returns forward by number of periods specified in lags.
        method: str, {'simple', 'log'}, default 'log'
            Method to compute returns, continuous or simple
        market: bool, Optional, default False
            Computes the returns of the entire universe of asset prices, or the market return.
        mkt_field: str, optional, default None
            Field to use to compute market return.
        mkt_weighting: str, optional, default None
            Weighting method to use to compute market return.

        Returns
        -------
        ret: Series or DataFrame - Single or MultiIndex
            Series or DataFrame with DatetimeIndex and returns series.
        """
        # unstack
        if isinstance(self.df.index, pd.MultiIndex):
            df = self.df.unstack().copy()
        else:
            df = self.df.copy()

        # simple returns
        if method == 'simple':
            ret = df.pct_change(lags)
        # log returns
        else:
            # remove negative values
            df[df <= 0] = np.nan
            ret = np.log(df) - np.log(df).shift(lags)

        # forward returns
        if forward:
            ret = ret.shift(lags * -1)

        # stack to multiindex
        if isinstance(ret.index, pd.DatetimeIndex):
            ret = ret.stack()

        # market return
        # TODO: write tests
        if market:
            if mkt_weighting is None:
                ret = ret.unstack()[mkt_field].mean(axis=1).to_frame('mkt_ret')
            else:
                pass  # TODO: add other market computations

        return ret

    @staticmethod
    def compute_std(df: Union[pd.Series, pd.DataFrame],
                    window_type: str = 'fixed',
                    window_size: int = 30,
                    min_periods: int = 2,
                    window_fcn: str = None
                    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Computes standard deviation.

        Parameters
        ----------
        df: pd.DataFrame or pd.Series
            Dataframe or series to compute standard deviation.
        window_type: str, {'fixed', 'expanding', 'rolling}, default 'fixed'
            Provide a window type. If None, all observations are used in the calculation.
        window_size: int, default 30
            Size of the moving window. This is the minimum number of observations used for the rolling or expanding
            window statistic.
        min_periods: int, default 2
            Minimum number of observations in window required to have a value; otherwise, result is np.nan.
        window_fcn: str, default None
            Provide a window function. If None, observations are equally-weighted in the rolling computation.

        Returns
        -------
        stdev: pd.Series or pd.DataFrame
            Series or DataFrame with DatetimeIndex (level 0), ticker (level 1) and standard deviation of values (cols).
        """
        if window_type == 'rolling':
            stdev = df.rolling(window=window_size, min_periods=min_periods, win_type=window_fcn).std()
        elif window_type == 'expanding':
            stdev = df.expanding(min_periods=min_periods).std()
        else:
            stdev = df.std()

        return stdev

    @staticmethod
    def compute_iqr(df: Union[pd.Series, pd.DataFrame],
                    window_type: str = 'fixed',
                    window_size: int = 30,
                    min_periods: int = 2,
                    window_fcn: str = None
                    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Computes interquartile range.

        Parameters
        ----------
        df: pd.DataFrame or pd.Series
            Dataframe or series to compute interquartile range.
        window_type: str, {'fixed', 'expanding', 'rolling}, default 'fixed'
            Provide a window type. If None, all observations are used in the calculation.
        window_size: int, default 30
            Size of the moving window. This is the minimum number of observations used for the rolling or expanding
            window statistic.
        min_periods: int, default 2
            Minimum number of observations in window required to have a value; otherwise, result is np.nan.
        window_fcn: str, default None
            Provide a window function. If None, observations are equally-weighted in the rolling computation.

        Returns
        -------
        iqr: pd.Series or pd.DataFrame
            Series or DataFrame with DatetimeIndex (level 0), ticker (level 1) and interquartile range of values (cols).
        """
        if window_type == 'rolling':
            iqr = df.rolling(window=window_size, min_periods=min_periods, win_type=window_fcn).quantile(0.75) - \
                    df.rolling(window=window_size, min_periods=min_periods, win_type=window_fcn).quantile(0.25)
        elif window_type == 'expanding':
            iqr = df.expanding(min_periods=min_periods).quantile(0.75) - \
                    df.expanding(min_periods=min_periods).quantile(0.25)
        else:
            iqr = df.quantile(0.75) - df.quantile(0.25)

        return iqr

    @staticmethod
    def compute_mad(df: Union[pd.Series, pd.DataFrame],
                    window_type: str = 'fixed',
                    window_size: int = 30,
                    min_periods: int = 2,
                    window_fcn: str = None
                    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Computes median absolute deviation.

        Parameters
        ----------
        df: pd.DataFrame or pd.Series
            Dataframe or series to compute median absolute deviation.
        window_type: str, {'fixed', 'expanding', 'rolling}, default 'fixed'
            Provide a window type. If None, all observations are used in the calculation.
        window_size: int, default 30
            Size of the moving window. This is the minimum number of observations used for the rolling or expanding
            window statistic.
        min_periods: int, default 2
            Minimum number of observations in window required to have a value; otherwise, result is np.nan.
        window_fcn: str, default None
            Provide a window function. If None, observations are equally-weighted in the rolling computation.

        Returns
        -------
        mad: pd.Series or pd.DataFrame
            Series or DataFrame with DatetimeIndex (level 0), ticker (level 1) and
            median absolute deviation of values (cols).
        """
        if window_type == 'rolling':
            abs_dev = (df - df.rolling(window=window_size, min_periods=min_periods, win_type=window_fcn).median()).abs()
            mad = abs_dev.rolling(window=window_size, min_periods=min_periods, win_type=window_fcn).median()
        elif window_type == 'expanding':
            abs_dev = (df - df.expanding(min_periods=min_periods).median()).abs()
            mad = abs_dev.expanding(min_periods=min_periods).median()
        else:
            abs_dev = (df - df.median()).abs()
            mad = abs_dev.median()

        return mad

    @staticmethod
    def compute_range(df: Union[pd.Series, pd.DataFrame],
                      window_type: str = 'fixed',
                      window_size: int = 30,
                      min_periods: int = 2,
                      window_fcn: str = None
                      ) -> Union[pd.Series, pd.DataFrame]:
        """
        Computes range.

        Parameters
        ----------
        df: pd.DataFrame or pd.Series
            Dataframe or series to compute range.
        window_type: str, {'fixed', 'expanding', 'rolling}, default 'fixed'
            Provide a window type. If None, all observations are used in the calculation.
        window_size: int, default 30
            Size of the moving window. This is the minimum number of observations used for the rolling or expanding
            window statistic.
        min_periods: int, default 2
            Minimum number of observations in window required to have a value; otherwise, result is np.nan.
        window_fcn: str, default None
            Provide a window function. If None, observations are equally-weighted in the rolling computation.

        Returns
        -------
        range: pd.Series or pd.DataFrame
            Series or DataFrame with DatetimeIndex (level 0), ticker (level 1) and range of values (cols).
        """
        if window_type == 'rolling':
            min_max = df.rolling(window=window_size, min_periods=min_periods, win_type=window_fcn).max() - \
                    df.rolling(window=window_size, min_periods=min_periods, win_type=window_fcn).min()
        elif window_type == 'expanding':
            min_max = df.expanding(min_periods=min_periods).max() - \
                    df.expanding(min_periods=min_periods).min()
        else:
            min_max = df.max() - df.min()

        return min_max

    @staticmethod
    def compute_var(df: Union[pd.Series, pd.DataFrame],
                    window_type: str = 'fixed',
                    window_size: int = 30,
                    min_periods: int = 2,
                    window_fcn: str = None
                    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Computes variance.

        Parameters
        ----------
        df: pd.DataFrame or pd.Series
            Dataframe or series to compute variance.
        window_type: str, {'fixed', 'expanding', 'rolling}, default 'fixed'
            Provide a window type. If None, all observations are used in the calculation.
        window_size: int, default 30
            Size of the moving window. This is the minimum number of observations used for the rolling or expanding
            window statistic.
        min_periods: int, default 2
            Minimum number of observations in window required to have a value; otherwise, result is np.nan.
        window_fcn: str, default None
            Provide a window function. If None, observations are equally-weighted in the rolling computation.

        Returns
        -------
        var: pd.Series or pd.DataFrame
            Series or DataFrame with DatetimeIndex (level 0), ticker (level 1) and variance of values (cols).
        """
        if window_type == 'rolling':
            var = getattr(df, 'rolling')(window=window_size, min_periods=min_periods, win_type=window_fcn).var()
        elif window_type == 'expanding':
            var = getattr(df, 'expanding')(min_periods=min_periods).var()
        else:
            var = df.var()

        return var

    @staticmethod
    def compute_atr(df: Union[pd.Series, pd.DataFrame],
                    window_type: str = 'fixed',
                    window_size: int = 30,
                    min_periods: int = 2,
                    window_fcn: str = None
                    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Computes average true range.

        Parameters
        ----------
        df: pd.DataFrame or pd.Series
            Dataframe or series to compute average true range.
        window_type: str, {'fixed', 'expanding', 'rolling}, default 'fixed'
            Provide a window type. If None, all observations are used in the calculation.
        window_size: int, default 30
            Size of the moving window. This is the minimum number of observations used for the rolling or expanding
            window statistic.
        min_periods: int, default 2
            Minimum number of observations in window required to have a value; otherwise, result is np.nan.
        window_fcn: str, default None
            Provide a window function. If None, observations are equally-weighted in the rolling computation.

        Returns
        -------
        atr: pd.Series or pd.DataFrame
            Series or DataFrame with DatetimeIndex (level 0), ticker (level 1) and average true range of values (cols).
        """
        # check if OHLC
        if not all([col in df.columns for col in ['open', 'high', 'low', 'close']]):
            raise ValueError("Dataframe must have open, high, low and close fields to compute atr.")

        # compute atr
        df['tr1'] = df.high - df.low
        df['tr2'] = np.abs(df.high - df.close.shift())
        df['tr3'] = np.abs(df.low - df.close.shift())
        df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)

        # window type
        if window_type == 'rolling':
            atr = df.tr.rolling(window=window_size, min_periods=min_periods, window_type=window_fcn).mean()
        elif window_type == 'expanding':
            atr = df.tr.expanding(min_periods=min_periods).mean()
        else:
            atr = df.tr.mean()

        return atr

    def dispersion(self,
                   method: str = 'std',
                   axis: str = 'ts',
                   window_type: str = 'fixed',
                   window_size: int = 30,
                   min_periods: int = 1,
                   window_fcn: str = None,
                   ) -> Union[pd.Series, pd.DataFrame]:
        """
        Computes dispersion of series.

        Parameters
        ----------
        method: str, {'std', 'iqr', 'range', 'atr', 'range', 'mad'}, default 'std'
            Method for computing dispersion. Options are 'std' or 'beta'.
        axis: str, {'ts', 'cs'}, default 'ts'
            Axis along which to compute dispersion, time series or cross-section.
        window_type: str, {'fixed', 'expanding', 'rolling}, default 'fixed'
            Provide a window type. If None, all observations are used in the calculation.
        window_size: int, default 30
            Size of the moving window. This is the minimum number of observations used for the rolling or expanding
            window statistic.
        min_periods: int, default 1
            Minimum number of observations in window required to have a value; otherwise, result is np.nan.
        window_fcn: str, default None
            Provide a window function. If None, observations are equally-weighted in the rolling computation.
            See scipy.signal.windows for more information.

        Returns
        -------
        disp: pd.Series or pd.DataFrame
            Series or DataFrame with DatetimeIndex (level 0), ticker (level 1) and dispersion values (cols).
        """
        # axis ts
        if axis == 'ts':

            # unstack
            if isinstance(self.df.index, pd.MultiIndex):
                self.trans_df = self.df.unstack().copy()
            else:
                self.trans_df = self.df.copy()

            # dispersion
            if window_type == 'rolling':
                self.trans_df = getattr(self, f"compute_{method}")(self.trans_df, window_type='rolling',
                                                                   window_size=window_size, min_periods=min_periods,
                                                                   window_fcn=window_fcn)
            elif window_type == 'expanding':
                self.trans_df = getattr(self, f"compute_{method}")(self.trans_df, window_type='expanding',
                                                                   window_size=window_size, min_periods=min_periods)
            else:
                self.trans_df = getattr(self, f"compute_{method}")(self.trans_df, window_type='fixed')

            # stack to multiindex
            if isinstance(self.df.index, pd.MultiIndex):
                self.trans_df = self.trans_df.stack()

        # axis cs
        else:
            # df
            self.trans_df = self.df.copy()
            # axis 1 - cross-section
            if isinstance(self.df.index, pd.MultiIndex):
                self.trans_df = self.trans_df.groupby(level=0).apply(getattr(self, f"compute_{method}"))
            else:
                self.trans_df = self.trans_df.apply(getattr(self, f"compute_{method}"), axis=1)

        return self.trans_df

    def target_vol(self,
                   ann_vol: float = 0.15,
                   ann_factor: float = 365
                   ) -> pd.DataFrame:
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
        # df
        df = self.df.copy()

        # unstack
        if isinstance(self.df.index, pd.MultiIndex):
            df = df.unstack()

        # set target vol
        norm_factor = 1 / ((df.std() / ann_vol) * np.sqrt(ann_factor))
        df = df * norm_factor

        # stack to multiindex
        if isinstance(df.index, pd.MultiIndex):
            df = df.stack()

        return df

    def smooth(self,
               window_size: int,
               window_type: str = 'rolling',
               window_fcn: str = None,
               central_tendency: str = 'mean',
               lags: int = 0,
               **kwargs: dict
               ) -> Union[pd.Series, pd.DataFrame]:
        """
        Smooths values using 'smoothing' method.

        Parameters
        ----------
        window_size: int
            Number of observations in moving window.
        window_type: str, {'expanding', 'rolling', 'ewm'}, default 'rolling'
            Provide a window type. If None, all observations are used in the calculation.
        window_fcn: str, default None
            Provide a rolling window function. If None, observations are equally-weighted in the rolling computation.
            See scipy.signal.windows for more information.
        central_tendency: str, {'mean', 'median'}, default 'mean'
            Measure of central tendency used for the rolling window.
        lags: int, default None
            Number of periods by which to lag values.
        kwargs: dict

        Returns
        -------
        df: pd.Series or pd.DataFrame
            Series or DataFrame with DatetimeIndex (level 0), tickers (level 1) and smoothed values (cols).
        """
        # TODO: refactor to allow for more window functions in rolling
        # multiindex
        if isinstance(self.df.index, pd.MultiIndex):
            # unstack
            self.trans_df = self.df.unstack().copy()
        else:
            self.trans_df = self.df.copy()

        # smoothing
        if window_type == 'ewm':
            if central_tendency == 'median':
                raise ValueError("Median is not supported for ewm smoothing.")
            else:
                self.trans_df = getattr(getattr(self.trans_df, window_type)(span=window_size),
                                    central_tendency)(**kwargs).shift(lags)
        elif window_type == 'rolling':
            self.trans_df = getattr(getattr(self.trans_df, window_type)(window=window_size, win_type=window_fcn),
                                    central_tendency)(**kwargs).shift(lags)
        elif window_type == 'expanding':
            self.trans_df = getattr(getattr(self.trans_df, window_type)(), central_tendency)(**kwargs).shift(lags)

        # stack to multiindex
        if isinstance(self.df.index, pd.MultiIndex):
            self.trans_df = self.trans_df.stack()

        return self.trans_df

    # TODO: correct normalize for look-ahead bias
    @staticmethod
    def normalize(df: Union[pd.Series, pd.DataFrame],
                  centering: bool = True,
                  method: str = 'z-score',
                  window_type: str = 'fixed',
                  window_size: int = 10,
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
        window_size: int, default 10
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
        # window type
        if window_type == 'rolling':
            mov_df = getattr(df, window_type)(window_size, min_periods=min_periods)
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
                mad = getattr(ad, window_type)(window_size).median()
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
                     window_size: int = 10,
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
        method: str, {'z-score', 'cdf', iqr', 'min-max', 'percentile', 'mod_z}, default 'z-score'
            z-score: subtracts mean and divides by standard deviation.
            cdf: cumulative distribution function rescales z-scores to values between 0 and 1.
            iqr:  subtracts median and divides by interquartile range.
            min-max: rescales to values between 0 and 1 by subtracting the min and dividing by the range.
            percentile: converts values to their percentile rank relative to the observations in the
            defined window type.
            mod_z: modified z-score using median absolute deviation.
        window_type: str, {'fixed', 'expanding', 'rolling}, default 'fixed'
            Provide a window type. If None, all observations are used in the calculation.
        window_size: int, default 10
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
        # multiindex
        if isinstance(self.df.index, pd.MultiIndex):
            norm_df = self.normalize(self.df.unstack(),
                                     centering=centering,
                                     window_type=window_type,
                                     window_size=window_size,
                                     min_periods=min_periods,
                                     method=method,
                                     winsorize=winsorize,
                                     ).stack()
        # single index
        else:
            norm_df = self.normalize(self.df,
                                     centering=centering,
                                     window_type=window_type,
                                     window_size=window_size,
                                     min_periods=min_periods,
                                     method=method,
                                     winsorize=winsorize
                                     )

        return norm_df

    def normalize_cs(self,
                     centering: bool = True,
                     method: str = 'z-score',
                     min_periods: int = 2,
                     winsorize: Optional[int] = None
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
        min_periods: int, default 2
            Minimum number of observations in window required to have a value; otherwise, result is np.nan.
        winsorize: int, default None
            Winsorizes/clips values to between positive and negative values of specified integer.

        Returns
        -------
        norm_features: Series or DataFrame - MultiIndex
            DatetimeIndex (level 0), tickers (level 1) and normalized features
        """
        norm_df = self.df.groupby(level=0, group_keys=False).apply(
            lambda x: self.normalize(x, centering=centering, method=method,
                                     min_periods=min_periods, winsorize=winsorize))

        return norm_df

    @staticmethod
    def quantize_np(df: Union[pd.Series, pd.DataFrame], bins: int = 5) -> np.array:
        """
        Quantizes factors or targets using numpy rather than pandas qcut.

        Quantization is the process of transforming a continuous variable into a discrete one by creating a set of bins
        (or equivalently contiguous intervals/cuttoffs) that spans the range of the variable’s values.
        Quantization creates an equal number of values in each bin.
        See Discretize function for other types of discretization.

        Parameters
        ----------
        df: pd.DataFrame or pd.Series
            Dataframe or series to quantize.
        bins:
            Number of bins.

        Returns
        -------
        quantiles: np.array
            Array with quantized data.
        """
        # percentile rank
        perc = df.rank(pct=True)
        # bins
        labels = np.arange(0, 1 + 1 / bins, 1 / bins)
        # mask nans
        mask = np.isnan(df)
        # quantize
        quantiles = np.digitize(perc, labels, right=True)
        quantiles = np.where(mask, df, quantiles)

        return quantiles

    def quantize(self, df, bins: int = 5, window_type='fixed', min_periods=2, window_size=30, axis=0):
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
        min_periods: int, default 2
            Minimum number of observations in window required to have a value; otherwise, result is np.nan.
        window_size: int, default 30
            Size of the moving window. This is the minimum number of observations used for the rolling or
        axis: int, default 0
            Axis along which to quantize. Defaults to each column.

        Returns
        -------
        quant_df: DataFrame
            Series or DataFrame with DatetimeIndex (index) and quantized features (columns).
        """
        # quantize using moving window
        def quant_mw(x, _bins=bins):
            return self.quantize_np(x, bins=_bins)[-1]

        # window type
        if window_type == 'rolling':
            quant_df = getattr(df, window_type)(window_size, min_periods=min_periods).apply(quant_mw)
        elif window_type == 'expanding':
            quant_df = getattr(df, window_type)(min_periods=min_periods).apply(quant_mw)
        else:
            quant_df = df.apply(self.quantize_np, axis=axis, result_type='broadcast')

        return quant_df

    def quantize_ts(self, bins: int = 5, window_type='fixed', window_size=30, min_periods=2) -> Union[pd.DataFrame]:
        """
        Quantizes features over the time series (rows).

        Parameters
        ----------
        bins: int, default 5
            Number of bins.
        window_type: str, {'fixed', 'expanding', 'rolling}, default 'fixed'
            Provide a window type. If None, all observations are used in the calculation.
        window_size: int, default 30
            Size of the moving window. This is the minimum number of observations used for the rolling or
        min_periods: int, default 2
            Minimum number of observations in window required to have a value; otherwise, result is np.nan.

        Returns
        -------
        quant_df: DataFrame
            Series or DataFrame with DatetimeIndex (level 0), tickers (level 1) and quantized features (columns).
        """
        # multiindex
        if isinstance(self.df.index, pd.MultiIndex):
            quant_df = self.quantize(self.df.unstack(),
                                     bins=bins,
                                     window_type=window_type,
                                     window_size=window_size,
                                     min_periods=min_periods,
                                     axis=0).stack()
        # single index
        else:
            quant_df = self.quantize(self.df,
                                     bins=bins,
                                     window_type=window_type,
                                     min_periods=min_periods,
                                     axis=0)

        return quant_df

    def quantize_cs(self, bins: int = 5) -> Union[pd.DataFrame]:
        """
        Quantizes features over the cross-section (cols).

        Returns
        -------
        quant_df: DataFrame
            Series or DataFrame with DatetimeIndex (level 0), tickers (level 1), and quantized features (columns).
        """
        # multiindex
        if isinstance(self.df.index, pd.MultiIndex):
            quant_df = pd.DataFrame()
            for col in self.df.columns:
                quant_df = pd.concat([quant_df,
                                      self.quantize(self.df[[col]].unstack(), bins=bins, axis=1).stack()], axis=1)
        # single index
        else:
            quant_df = self.quantize(self.df, bins=bins, axis=1)

        return quant_df

    # @staticmethod
    # def quantize(df: Union[pd.Series, pd.DataFrame],
    #              bins: int = 5,
    #              window_type: str = 'fixed',
    #              lookback: int = 10,
    #              _min_periods: int = 2,
    #              axis: int = 0,
    #              ) -> pd.DataFrame:
    #     """
    #     Quantizes factors or targets. Quantization is the process of transforming a continuous variable
    #     into a discrete one by creating a set of bins (or equivalently contiguous intervals/cuttoffs) that spans
    #     the range of the variable’s values. Quantization creates an equal number of values in each bin.
    #     See Discretize function for other types of discretization.
    #
    #     Parameters
    #     ----------
    #     df: pd.DataFrame or pd.Series
    #         Dataframe or series to quantize.
    #     bins: int, default 5
    #         Number of bins.
    #     window_type: str, {'fixed', 'expanding', 'rolling}, default 'fixed'
    #         Provide a window type. If None, all observations are used in the calculation.
    #     lookback: int, default 10
    #         Size of the moving window. This is the minimum number of observations used for the rolling or
    #         expanding statistic.
    #     _min_periods: int, default 2
    #         Minimum number of observations in window required to have a value; otherwise, result is np.nan.
    #     axis: int, default 0
    #         Axis along which to quantize. Defaults to each column.
    #
    #     Returns
    #     -------
    #     quant_df: DataFrame
    #         Series or DataFrame with DatetimeIndex (index) and quantized features (columns).
    #     """
    #     # check bins
    #     if bins <= 1 or bins is None:
    #         raise ValueError('Number of bins must be larger than 1. Please increase number of bins.')
    #     # convert to df
    #     if isinstance(df, pd.Series):
    #         df = df.to_frame().copy()
    #
    #     # quantize function
    #     # def quantize(x, _bins=bins, _axis=axis):
    #     #
    #     #     # percentile rank
    #     #     perc = x.rank(pct=True, axis=_axis)
    #     #     # bins
    #     #     labels = np.arange(0, 1, 1 / _bins)
    #     #     quantiles = np.digitize(perc, labels, right=True)
    #     #
    #     #     return quantiles
    #     #
    #     # def quant_mw(x):
    #     #     return quantize(x)[-1]
    #
    #     # quantize function
    #     def quantize(x):
    #         return pd.qcut(x, bins, labels=False, duplicates='drop') + 1
    #
    #     def quant_mw(x):
    #         return (pd.qcut(x, bins, labels=False, duplicates='drop') + 1)[-1]
    #
    #     # window type
    #     if window_type == 'rolling':
    #         quant_df = getattr(df, window_type)(lookback, min_periods=_min_periods).apply(quant_mw)
    #     elif window_type == 'expanding':
    #         quant_df = getattr(df, window_type)(min_periods=_min_periods).apply(quant_mw)
    #     else:
    #         quant_df = df.apply(quantize, axis=axis)
    #
    #     return quant_df
    #
    # def quantize_ts(self,
    #                 bins: int = 5,
    #                 window_type: str = 'fixed',
    #                 lookback: int = 10,
    #                 _min_periods: int = 2) -> Union[pd.Series, pd.DataFrame]:
    #     """
    #     Quantizes features over the time series (rows).
    #
    #     Parameters
    #     ----------
    #     bins: int, default 5
    #         Number of bins.
    #     window_type: str, {'fixed', 'expanding', 'rolling}, default 'fixed'
    #         Provide a window type. If None, all observations are used in the calculation.
    #     lookback: int, default 10
    #         Size of the moving window. This is the minimum number of observations used for the rolling or
    #         expanding statistic.
    #     _min_periods: int, default 2
    #         Minimum number of observations in window required to have a value; otherwise, result is np.nan.
    #
    #     Returns
    #     -------
    #     quant_df: DataFrame
    #         Series or DataFrame with DatetimeIndex (level 0), tickers (level 1) and quantized features (columns).
    #     """
    #     # multiindex
    #     if isinstance(self.df.index, pd.MultiIndex):
    #         quant_df = self.quantize(self.df.unstack().dropna(how='all', axis=1),
    #                                  bins=bins,
    #                                  window_type=window_type,
    #                                  lookback=lookback,
    #                                  _min_periods=_min_periods).stack()
    #     # single index
    #     else:
    #         quant_df = self.quantize(self.df.dropna(how='all', axis=1),
    #                                  bins=bins,
    #                                  window_type=window_type,
    #                                  lookback=lookback,
    #                                  _min_periods=_min_periods)
    #
    #     return quant_df
    #
    # def quantize_cs(self, bins: int = 5) -> Union[pd.Series, pd.DataFrame]:
    #     """
    #     Quantizes features over the cross-section (cols).
    #
    #     Parameters
    #     ----------
    #     bins: int, default 5
    #         Number of bins.
    #
    #     Returns
    #     -------
    #     quant_df: DataFrame
    #         Series or DataFrame with DatetimeIndex (level 0), tickers (level 1), and quantized features (columns).
    #     """
    #     # df to store quantized cols
    #     quant_df = pd.DataFrame()
    #
    #     # multiindex
    #     if isinstance(self.df.index, pd.MultiIndex):
    #         # loop through cols
    #         for col in self.df.columns:
    #             df = self.quantize(self.df[col].unstack().dropna(thresh=bins, axis=0), axis=1).stack().to_frame(col)
    #             quant_df = pd.concat([quant_df, df], axis=1)
    #
    #     # single index
    #     else:
    #         quant_df = self.quantize(self.df.dropna(thresh=bins, axis=0), bins=5, axis=1)
    #
    #     return quant_df.sort_index()

    @staticmethod
    def discretize(df: Union[pd.Series, pd.DataFrame],
                   discretization: str = 'quantile',
                   bins: int = 5,
                   window_type: str = 'fixed',
                   window_size: int = 7) -> pd.DataFrame:
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
        window_size: int, default 7
            Size of the moving window. This is the minimum number of observations used for the rolling or
            expanding statistic.

        Returns
        -------
        disc_features: DataFrame
            Series or DataFrame with DatetimeIndex and discretized features.
        """
        # check bins
        if bins <= 1 or bins is None:
            raise ValueError('Number of bins must be larger than 1. Please increase number of bins.')
        # convert to df
        if isinstance(df, pd.Series):
            df = df.to_frame().copy()

        # discretize features
        discretize = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy=discretization)
        # window size
        window_size = window_size

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
                        discretize.fit_transform(df.iloc[window_size - window_size:window_size])[-1]
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
                      window_size: int = 7) -> pd.DataFrame:
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
        window_size: int, default 7
            Size of the moving window. This is the minimum number of observations used for the rolling or
            expanding statistic.

        Returns
        -------
        disc_features: DataFrame
            Series or DataFrame with DatetimeIndex and discretized features.
        """
        # multiindex
        if isinstance(self.df.index, pd.MultiIndex):
            df = self.df.unstack().dropna(how='all', axis=1).dropna()  # dropna
            disc_df = self.discretize(df,
                                      discretization=discretization,
                                      bins=bins,
                                      window_type=window_type,
                                      window_size=window_size).stack()
        # single index
        else:
            df = self.df.dropna(how='all', axis=1).dropna()  # dropna
            disc_df = self.discretize(df,
                                      discretization=discretization,
                                      bins=bins,
                                      window_type=window_type,
                                      window_size=window_size)
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
        # dropna
        df = self.df.dropna(how='all', axis=1).dropna()
        # discretize
        disc_df = df.groupby(level=0, group_keys=False).apply(
            lambda x: self.discretize(x, discretization=discretization, bins=bins))

        return disc_df

    def orthogonalize(self) -> pd.DataFrame:
        """
        Orthogonalize factors.

        As described by Klein and Chow (2013) in Orthogonalized Factors and Systematic Risk Decompositions:
        https://www.sciencedirect.com/science/article/abs/pii/S1062976913000185
        They propose an optimal simultaneous orthogonal transformation of factors, following the so-called symmetric
        procedure of Schweinler and Wigner (1970) and Löwdin (1970).  The data transformation allows the identification
        of the underlying uncorrelated components of common factors without changing their correlation with the original
        factors. It also facilitates the systematic risk decomposition by disentangling the coefficient of determination
        (R²) based on factor volatilities, which makes it easier to distinguish the marginal risk contribution of each
        common risk factor to asset returns.

        Returns
        -------
        orthogonal_factors: pd.DataFrame
            Orthogonalized factors.
        """
        # compute cov matrix
        M = np.cov(self.arr.T)
        # factorize cov matrix M
        u, s, vh = np.linalg.svd(M)
        # solve for symmetric matrix
        S = u @ np.diag(s ** (-0.5)) @ vh
        # rescale symmetric matrix to original variances
        M[M < 0] = np.nan  # remove negative values
        S_rs = S @ (np.diag(np.sqrt(M)) * np.eye(S.shape[0], S.shape[1]))
        # convert to orthogonal matrix
        orthogonal_factors = self.arr @ S_rs

        return pd.DataFrame(orthogonal_factors, index=self.df.index[-orthogonal_factors.shape[0]:],
                            columns=self.df.columns)
