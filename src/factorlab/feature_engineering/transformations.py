import pandas as pd
import numpy as np
from typing import Union, Optional
from scipy import stats
from sklearn.preprocessing import KBinsDiscretizer


class Transform:
    """
    Transformations on raw data, returns or features.
    """
    def __init__(self, data: Union[pd.Series, pd.DataFrame, np.array], **kwargs: dict):
        """
        Constructor

        Parameters
        ----------
        data: pd.Series, pd.DataFrame or np.array
            Data to transform.
        kwargs: dict
        """
        self.raw_data = data
        self.df = None
        self.arr = None
        self.trans_df = None
        self.index = None
        self.freq = None
        self.kwargs = kwargs
        self.preprocess_data()

    def preprocess_data(self) -> None:
        """
        Preprocesses data.
        """
        # df
        if isinstance(self.raw_data, pd.DataFrame) or isinstance(self.raw_data, pd.Series):
            self.df = self.raw_data.astype('float64').copy()
            self.arr = self.raw_data.to_numpy(dtype='float64').copy()
            self.index = self.raw_data.index
            if isinstance(self.index, pd.MultiIndex):
                self.freq = pd.infer_freq(self.index.get_level_values(0).unique())
            else:
                self.freq = pd.infer_freq(self.index)
        # series
        elif isinstance(self.raw_data, pd.Series):
            self.df = self.df.to_frame()
        # array
        elif isinstance(self.raw_data, np.ndarray):
            self.arr = self.raw_data.astype(float).copy()
            self.df = pd.DataFrame(self.arr).copy()

        self.trans_df = self.df.copy()

    def vwap(self) -> pd.DataFrame:
        """
        Computes volume weighted average price.

        Returns
        -------
        df: pd.DataFrame
            DataFrame with vwap.
        """
        # check if OHLC
        if not all([col in self.df.columns for col in ['open', 'high', 'low', 'close']]):
            raise ValueError("Dataframe must have open, high, low and close fields to compute vwap.")

        # compute vwap
        self.trans_df['vwap'] = (self.df.close +
                                (self.df.open + self.df.high + self.df.low)/3)/2

        return self.trans_df

    def log(self) -> Union[pd.Series, pd.DataFrame]:
        """
        Computes log transformation.

        Returns
        -------
        df: pd.Series or pd.DataFrame
            Series or DataFrame with log-transformed values.
        """
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
                ) -> Union[pd.Series, pd.DataFrame]:
        """
        Computes price returns.


        Parameters
        ----------
        lags: int, default 1
            Number of periods to lag the returns.
        forward: bool, default False
            Forward returns.
        method: str, {'simple', 'log'}, default 'log'
            Type of returns.
        market: bool, default False
            Market returns.
        mkt_field: str, default 'close'
            Market field to use.
        mkt_weighting: str, default None
            Market weighting method.

        Returns
        -------
        df: pd.DataFrame
            DataFrame with returns.
        """
        # simple returns
        if method == 'simple':
            if isinstance(self.index, pd.MultiIndex):
                self.trans_df = self.df.groupby(level=1).pct_change(lags)
            else:
                self.trans_df = self.df.pct_change(lags)

        # log returns
        else:
            # remove negative values
            self.df[self.df <= 0] = np.nan
            if isinstance(self.index, pd.MultiIndex):
                self.trans_df = np.log(self.df).groupby(level=1).diff(lags)
            else:
                self.trans_df = np.log(self.df).diff(lags)

        # forward returns
        if forward:
            if isinstance(self.index, pd.MultiIndex):
                self.trans_df = self.trans_df.groupby(level=1).shift(lags * -1)
            else:
                self.trans_df = self.trans_df.shift(lags * -1)

        # market return
        if market:
            if mkt_weighting is None:
                if isinstance(self.index, pd.MultiIndex):
                    self.trans_df = self.trans_df.groupby(level=0).mean()[mkt_field].to_frame('mkt_ret')
                else:
                    self.trans_df = self.trans_df[mkt_field].to_frame('mkt_ret')
            else:
                pass  # TODO: add other market computations

        return self.trans_df.sort_index()

    @staticmethod
    def returns_to_price(returns: Union[pd.Series, pd.DataFrame],
                         ret_type: str = 'simple',
                         start_val: float = 1.0
                         ) -> Union[pd.Series, pd.DataFrame]:
        """
        Converts returns to price series.

        Parameters
        ----------
        returns: pd.Series or pd.DataFrame
            Series or DataFrame with returns.
        ret_type: str, {'simple', 'log'}, default 'simple'
            Type of returns.
        start_val: float, default 1.0
            Starting value for the price series.

        Returns
        -------
        price: pd.Series or pd.DataFrame
            Series or DataFrame with price series.
        """
        # simple returns
        if ret_type == 'simple':
            if isinstance(returns.index, pd.MultiIndex):
                cum_ret = (1 + returns).groupby(level=1).cumprod()
            else:
                cum_ret = (1 + returns).cumprod()
        else:
            if isinstance(returns.index, pd.MultiIndex):
                cum_ret = np.exp(returns).groupby(level=1).cumprod()
            else:
                cum_ret = np.exp(returns.cumsum())

        # start val
        if start_val == 0:
            price = cum_ret - 1
        else:
            price = cum_ret * start_val

        return price

    def target_vol(self,
                   ann_vol: float = 0.15,
                   ann_factor: float = 365
                   ) -> Union[pd.Series, pd.DataFrame]:
        """
        Set volatility of series to be equal to a specific vol target.

        Parameters
        ----------
        ann_vol: float, default 0.15
            Target annualized volatility.
        ann_factor: float, {12, 52, 252, 365}, default 365
            Annualization factor.

        Returns
        -------
        df: pd.Series or pd.DataFrame
            Series or DataFrame with target volatility.
        """
        # set target vol
        if isinstance(self.df.index, pd.MultiIndex):
            norm_factor = 1 / ((self.df.groupby(level=1).std() / ann_vol) * np.sqrt(ann_factor)).sort_index()
        else:
            norm_factor = 1 / ((self.df.std() / ann_vol) * np.sqrt(ann_factor))

        self.trans_df = self.df * norm_factor

        return self.trans_df

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
            Series or DataFrame with smoothed values.
        """
        # TODO: refactor to allow for more window functions in rolling

        # smoothing
        if window_type == 'ewm':
            if central_tendency == 'median':
                raise ValueError("Median is not supported for ewm smoothing.")
            else:
                if isinstance(self.df.index, pd.MultiIndex):
                    self.trans_df = getattr(getattr(self.df.groupby(level=1), window_type)(span=window_size, **kwargs),
                                        central_tendency)().droplevel(0).groupby(level=1).shift(lags).sort_index()
                else:
                    self.trans_df = getattr(getattr(self.df, window_type)(span=window_size, **kwargs),
                                            central_tendency)().shift(lags)

        elif window_type == 'rolling':
            if isinstance(self.df.index, pd.MultiIndex):
                self.trans_df = getattr(getattr(self.df.groupby(level=1), window_type)(window=window_size,
                                                                                   win_type=window_fcn, **kwargs),
                                    central_tendency)().droplevel(0).groupby(level=1).shift(lags).sort_index()
            else:
                self.trans_df = getattr(getattr(self.df, window_type)(window=window_size, win_type=window_fcn,
                                                                      **kwargs), central_tendency)().shift(lags)

        elif window_type == 'expanding':
            if isinstance(self.df.index, pd.MultiIndex):
                self.trans_df = getattr(getattr(self.df.groupby(level=1), window_type)(), central_tendency)()\
                    .droplevel(0).groupby(level=1).shift(lags).sort_index()
            else:
                self.trans_df = getattr(getattr(self.df, window_type)(), central_tendency)().shift(lags)

        return self.trans_df

    def center(self,
               axis: str = 'ts',
               central_tendency: str = 'mean',
               window_type: str = 'fixed',
               window_size: int = 30,
               min_periods: int = 2
               ) -> Union[pd.Series, pd.DataFrame]:
        """
        Centers values.

        Parameters
        ----------
        axis: int, {0, 1}, default 0
            Axis along which to compute standard deviation, time series or cross-section.
        central_tendency: str, {'mean', 'median', 'min'}, default 'mean'
            Measure of central tendency used for the rolling window.
        window_type: str, {'fixed', 'expanding', 'rolling}, default 'fixed'
            Provide a window type. If None, all observations are used in the calculation.
        window_size: int, default 30
            Size of the moving window. This is the minimum number of observations used for the rolling or expanding
            window statistic.
        min_periods: int, default 2
            Minimum number of observations in window required to have a value; otherwise, result is np.nan.

        Returns
        -------
        centered: pd.Series or pd.DataFrame
            Series or DataFrame with centered values.
        """
        # axis time series
        if axis == 'ts':

            # rolling window
            if window_type == 'rolling':
                if isinstance(self.df.index, pd.MultiIndex):
                    self.trans_df = self.df - \
                                    getattr(self.df.groupby(level=1).rolling(window=window_size,
                                                                             min_periods=min_periods),
                                            central_tendency)().droplevel(0).sort_index()
                else:
                    self.trans_df = self.df - getattr(self.df.rolling(window=window_size, min_periods=min_periods),
                                                     central_tendency)()

            # expanding window
            elif window_type == 'expanding':
                if isinstance(self.df.index, pd.MultiIndex):
                    self.trans_df = self.df - \
                                    getattr(self.df.groupby(level=1).expanding(min_periods=min_periods),
                                            central_tendency)().droplevel(0).sort_index()
                else:
                    self.trans_df = self.df - getattr(self.df.expanding(min_periods=min_periods), central_tendency)()

            # fixed window
            else:
                if isinstance(self.df.index, pd.MultiIndex):
                    self.trans_df = self.df - getattr(self.df.groupby(level=1), central_tendency)().sort_index()
                else:
                    self.trans_df = self.df - getattr(self.df, central_tendency)()

        # axis 1 (cross-section)
        else:
            # fixed window
            if isinstance(self.df.index, pd.MultiIndex):
                self.trans_df = self.df - getattr(self.df.groupby(level=0), central_tendency)()
            else:
                self.trans_df = self.df.subtract(getattr(self.df, central_tendency)(axis=1), axis=0)

        return self.trans_df

    def compute_std(self,
                    axis: str = 'ts',
                    window_type: str = 'fixed',
                    window_size: int = 30,
                    min_periods: int = 2,
                    window_fcn: str = None
                    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Computes standard deviation.

        Parameters
        ----------
        axis: int, {0, 1}, default 0
            Axis along which to compute standard deviation, time series or cross-section.
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
        std: pd.Series or pd.DataFrame
            Series or DataFrame with standard deviation of values.
        """
        # axis time series
        if axis == 'ts':
            # rolling window
            if window_type == 'rolling':
                if isinstance(self.df.index, pd.MultiIndex):
                    self.trans_df = self.df.groupby(level=1).rolling(window=window_size, min_periods=min_periods,
                                                        win_type=window_fcn).std().droplevel(0).sort_index()
                else:
                    self.trans_df = self.df.rolling(window=window_size, min_periods=min_periods,
                                                    win_type=window_fcn).std()
            # expanding window
            elif window_type == 'expanding':
                if isinstance(self.df.index, pd.MultiIndex):
                    self.trans_df = self.df.groupby(level=1).expanding(min_periods=min_periods).std().\
                        droplevel(0).sort_index()
                else:
                    self.trans_df = self.df.expanding(min_periods=min_periods).std()
            # fixed window
            else:
                if isinstance(self.df.index, pd.MultiIndex):
                    self.trans_df = self.df.groupby(level=1).std()
                else:
                    self.trans_df = self.df.std()

        # axis 1 (cross-section)
        else:
            # fixed window
            if isinstance(self.df.index, pd.MultiIndex):
                self.trans_df = self.df.groupby(level=0).std()
            else:
                self.trans_df = self.df.std(axis=1)

        return self.trans_df

    def compute_iqr(self,
                    axis: str = 'ts',
                    window_type: str = 'fixed',
                    window_size: int = 30,
                    min_periods: int = 2,
                    window_fcn: str = None
                    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Computes inter-quartile range.

        Parameters
        ----------
        axis: str, {'ts', 'cs'}, default 'ts'
            Axis along which to compute interquartile range, time series or cross-section.
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
            Series or DataFrame with interquartile range of values.
        """
        # axis time series
        if axis == 'ts':
            # rolling window
            if window_type == 'rolling':
                if isinstance(self.df.index, pd.MultiIndex):
                    self.trans_df = (self.df.groupby(level=1).rolling(window=window_size, min_periods=min_periods,
                                                       win_type=window_fcn).quantile(0.75) -
                        self.df.groupby(level=1).rolling(window=window_size, min_periods=min_periods,
                                                    win_type=window_fcn).quantile(0.25)).droplevel(0).sort_index()
                else:
                    self.trans_df = self.df.rolling(window=window_size, min_periods=min_periods,
                                                    win_type=window_fcn).quantile(0.75) - \
                            self.df.rolling(window=window_size, min_periods=min_periods,
                                            win_type=window_fcn).quantile(0.25)
            # expanding window
            elif window_type == 'expanding':
                if isinstance(self.df.index, pd.MultiIndex):
                    self.trans_df = (self.df.groupby(level=1).expanding(min_periods=min_periods).quantile(0.75) -
                        self.df.groupby(level=1).expanding(min_periods=min_periods).quantile(0.25)).droplevel(0).\
                        sort_index()
                else:
                    self.trans_df = self.df.expanding(min_periods=min_periods).quantile(0.75) - \
                        self.df.expanding(min_periods=min_periods).quantile(0.25)
            # fixed window
            else:
                if isinstance(self.df.index, pd.MultiIndex):
                    self.trans_df = self.df.groupby(level=1).quantile(0.75) - self.df.groupby(level=1).quantile(0.25)
                else:
                    self.trans_df = self.df.quantile(0.75) - self.df.quantile(0.25)

        # axis 1 (cross-section)
        else:
            if isinstance(self.df.index, pd.MultiIndex):
                self.trans_df = self.df.groupby(level=0).quantile(0.75) - self.df.groupby(level=0).quantile(0.25)
            else:
                self.trans_df = self.df.quantile(0.75, axis=1) - self.df.quantile(0.25, axis=1)

        return self.trans_df

    def compute_mad(self,
                    axis: str = 'ts',
                    window_type: str = 'fixed',
                    window_size: int = 30,
                    min_periods: int = 2,
                    window_fcn: str = None
                    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Computes median absolute deviation.

        Parameters
        ----------
        axis: str, {'ts', 'cs'}, default 'ts'
            Axis along which to compute median absolute deviation, time series or cross-section.
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
            Series or DataFrame with median absolute deviation of values.
        """
        # axis time series
        if axis == 'ts':
            # rolling window
            if window_type == 'rolling':
                if isinstance(self.df.index, pd.MultiIndex):
                    abs_dev = (self.df - self.df.groupby(level=1).rolling(window=window_size, min_periods=min_periods,
                                                       win_type=window_fcn).median().droplevel(0)).abs()
                    self.trans_df = abs_dev.groupby(level=1).rolling(window=window_size, min_periods=min_periods,
                                                         win_type=window_fcn).median().droplevel(0).sort_index()
                else:
                    abs_dev = (self.df - self.df.rolling(window=window_size, min_periods=min_periods,
                                               win_type=window_fcn).median()).abs()
                    self.trans_df = abs_dev.rolling(window=window_size, min_periods=min_periods,
                                                    win_type=window_fcn).median()
            # expanding window
            elif window_type == 'expanding':
                if isinstance(self.df.index, pd.MultiIndex):
                    abs_dev = (self.df - self.df.groupby(level=1).expanding(min_periods=min_periods).median().
                               droplevel(0)).abs()
                    self.trans_df = abs_dev.groupby(level=1).expanding(min_periods=min_periods).median().droplevel(0).\
                        sort_index()
                else:
                    abs_dev = (self.df - self.df.expanding(min_periods=min_periods).median()).abs()
                    self.trans_df = abs_dev.expanding(min_periods=min_periods).median()
            # fixed window
            else:
                if isinstance(self.df.index, pd.MultiIndex):
                    abs_dev = (self.df - self.df.groupby(level=1).median()).abs()
                    self.trans_df = abs_dev.groupby(level=1).median()
                else:
                    self.trans_df = (self.df - self.df.median()).abs().median()

        # axis 1 (cross-section)
        else:
            if isinstance(self.df.index, pd.MultiIndex):
                self.trans_df = (self.df - self.df.groupby(level=0).median()).abs().groupby(level=0).median()
            else:
                self.trans_df = (self.df - self.df.median(axis=1)).abs().median(axis=1)

        return self.trans_df

    def compute_range(self,
                      axis: str = 'ts',
                      window_type: str = 'fixed',
                      window_size: int = 30,
                      min_periods: int = 2,
                      window_fcn: str = None
                      ) -> Union[pd.Series, pd.DataFrame]:
        """
        Computes range.

        Parameters
        ----------
        axis: str, {'ts', 'cs'}, default 'ts'
            Axis along which to compute range, time series or cross-section.
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
            Series or DataFrame with range of values.
        """
        # axis time series
        if axis == 'ts':
            # rolling window
            if window_type == 'rolling':
                if isinstance(self.df.index, pd.MultiIndex):
                    self.trans_df = (self.df.groupby(level=1).rolling(window=window_size, min_periods=min_periods,
                                                       win_type=window_fcn).max() -
                        self.df.groupby(level=1).rolling(window=window_size, min_periods=min_periods,
                                                    win_type=window_fcn).min()).droplevel(0).sort_index()
                else:
                    self.trans_df = self.df.rolling(window=window_size, min_periods=min_periods,
                                                    win_type=window_fcn).max() - \
                                    self.df.rolling(window=window_size, min_periods=min_periods,
                                                    win_type=window_fcn).min()
            elif window_type == 'expanding':
                if isinstance(self.df.index, pd.MultiIndex):
                    self.trans_df = (self.df.groupby(level=1).expanding(min_periods=min_periods).max() -
                                     self.df.groupby(level=1).expanding(min_periods=min_periods).min()).\
                        droplevel(0).sort_index()
                else:
                    self.trans_df = self.df.expanding(min_periods=min_periods).max() - \
                                    self.df.expanding(min_periods=min_periods).min()
            else:
                if isinstance(self.df.index, pd.MultiIndex):
                    self.trans_df = self.df.groupby(level=1).max() - self.df.groupby(level=1).min()
                else:
                    self.trans_df = self.df.max() - self.df.min()

        # axis 1 (cross-section)
        else:
            if isinstance(self.df.index, pd.MultiIndex):
                self.trans_df = self.df.groupby(level=0).max() - self.df.groupby(level=0).min()
            else:
                self.trans_df = self.df.max(axis=1) - self.df.min(axis=1)

        return self.trans_df

    def compute_var(self,
                    axis: str = 'ts',
                    window_type: str = 'fixed',
                    window_size: int = 30,
                    min_periods: int = 2,
                    window_fcn: str = None
                    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Computes variance.

        Parameters
        ----------
        axis: str, {'ts', 'cs'}, default 'ts'
            Axis along which to compute variance, time series or cross-section.
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
            Series or DataFrame with variance of values.
        """
        # axis time series
        if axis == 'ts':
            # rolling window
            if window_type == 'rolling':
                if isinstance(self.df.index, pd.MultiIndex):
                    self.trans_df = self.df.groupby(level=1).rolling(window=window_size, min_periods=min_periods,
                                                        win_type=window_fcn).var().droplevel(0).sort_index()
                else:
                    self.trans_df = self.df.rolling(window=window_size, min_periods=min_periods,
                                                    win_type=window_fcn).var()
            # expanding window
            elif window_type == 'expanding':
                if isinstance(self.df.index, pd.MultiIndex):
                    self.trans_df = self.df.groupby(level=1).expanding(min_periods=min_periods).var().\
                        droplevel(0).sort_index()
                else:
                    self.trans_df = self.df.expanding(min_periods=min_periods).var()
            # fixed window
            else:
                if isinstance(self.df.index, pd.MultiIndex):
                    self.trans_df = self.df.groupby(level=1).var()
                else:
                    self.trans_df = self.df.var()

        # axis 1 (cross-section)
        else:
            # fixed window
            if isinstance(self.df.index, pd.MultiIndex):
                self.trans_df = self.df.groupby(level=0).var()
            else:
                self.trans_df = self.df.var(axis=1)

        return self.trans_df

    def compute_atr(self,
                    axis: str = 'ts',
                    window_type: str = 'fixed',
                    window_size: int = 30,
                    min_periods: int = 2,
                    window_fcn: str = None
                    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Computes average true range.

        Parameters
        ----------
        axis: str, {'ts', 'cs'}, default 'ts'
            Axis along which to compute average true range, time series or cross-section.
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
            Series or DataFrame with average true range of values.
        """
        # check if OHLC
        if not all([col in self.df.columns for col in ['open', 'high', 'low', 'close']]):
            raise ValueError("Dataframe must have open, high, low and close fields to compute atr.")

        # unstack
        if isinstance(self.df.index, pd.MultiIndex):
            df = self.df.unstack().copy()
        else:
            df = self.df.copy()

        # axis time series
        if axis == 'ts':

            # compute true range
            tr1 = df.high - df.low
            tr2 = df.high - df.close.shift()
            tr3 = df.low - df.close.shift()

            # multiindex
            if isinstance(self.df.index, pd.MultiIndex):
                tr = pd.concat([tr1.stack(), tr2.stack(), tr3.stack()], axis=1).max(axis=1)
                # self.trans_df = tr.groupby(level=1).mean().to_frame('atr')
            else:
                tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                # self.trans_df = tr.mean()

            # rolling window
            if window_type == 'rolling':
                # multiindex
                if isinstance(self.df.index, pd.MultiIndex):
                    self.trans_df = tr.groupby(level=1).rolling(window=window_size, min_periods=min_periods,
                                               win_type=window_fcn).mean().droplevel(0).sort_index().to_frame('atr')
                else:
                    self.trans_df = tr.rolling(window=window_size, min_periods=min_periods,
                                               win_type=window_fcn).mean().to_frame('atr')
            # expanding window
            elif window_type == 'expanding':
                # multiindex
                if isinstance(self.df.index, pd.MultiIndex):
                    self.trans_df = tr.groupby(level=1).expanding(min_periods=min_periods).\
                        mean().droplevel(0).sort_index().to_frame('atr')
                else:
                    self.trans_df = tr.expanding(min_periods=min_periods).mean().to_frame('atr')
            # fixed window
            else:
                # multiindex
                if isinstance(self.df.index, pd.MultiIndex):
                    self.trans_df = tr.groupby(level=1).mean().to_frame('atr')
                else:
                    self.trans_df = tr.mean()

        # axis 1 (cross-section)
        else:
            raise ValueError("Cross-section not supported for ATR computation.")

        return self.trans_df

    def compute_percentile(self,
                           axis: str = 'ts',
                           window_type: str = 'fixed',
                           window_size: int = 30,
                           min_periods: int = 2,
                           window_fcn: str = None
                           ) -> Union[pd.Series, pd.DataFrame]:
        """
        Computes percentile.

        Parameters
        ----------
        axis: int, {0, 1}, default 0
            Axis along which to compute standard deviation, time series or cross-section.
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
        percentile: pd.Series or pd.DataFrame
            Series or DataFrame with percentile of values.
        """
        # axis time series
        if axis == 'ts':
            # rolling window
            if window_type == 'rolling':
                if isinstance(self.df.index, pd.MultiIndex):
                    self.trans_df = self.df.groupby(level=1).rolling(window=window_size, min_periods=min_periods,
                                                        win_type=window_fcn).rank(pct=True).droplevel(0).sort_index()
                else:
                    self.trans_df = self.df.rolling(window=window_size, min_periods=min_periods,
                                                    win_type=window_fcn).rank(pct=True)
            # expanding window
            elif window_type == 'expanding':
                if isinstance(self.df.index, pd.MultiIndex):
                    self.trans_df = self.df.groupby(level=1).expanding(min_periods=min_periods).rank(pct=True).\
                        droplevel(0).sort_index()
                else:
                    self.trans_df = self.df.expanding(min_periods=min_periods).rank(pct=True)
            # fixed window
            else:
                if isinstance(self.df.index, pd.MultiIndex):
                    self.trans_df = self.df.groupby(level=1).rank(pct=True)
                else:
                    self.trans_df = self.df.rank(pct=True)

        # axis 1 (cross-section)
        else:
            # fixed window
            if isinstance(self.df.index, pd.MultiIndex):
                self.trans_df = self.df.groupby(level=0).rank(pct=True)
            else:
                self.trans_df = self.df.rank(pct=True, axis=1)

        return self.trans_df

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
            Method for computing dispersion.
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
        df: pd.Series or pd.DataFrame
            Series or DataFrame with dispersion of values.
        """

        self.trans_df = getattr(self, f"compute_{method}")(axis=axis, window_type=window_type,
                                                           window_size=window_size, min_periods=min_periods,
                                                           window_fcn=window_fcn)

        return self.trans_df

    def normalize(self,
                  method: str = 'z-score',
                  axis: str = 'ts',
                  centering: bool = True,
                  window_type: str = 'fixed',
                  window_size: int = 10,
                  min_periods: int = 2,
                  winsorize: Optional[int] = None
                  ) -> Union[pd.Series, pd.DataFrame]:
        """
        Normalizes features.

        Parameters
        ----------
        method: str, {'z-score', 'iqr', 'mod_z', 'min-max', 'percentile'}, default 'z-score'
            z-score: subtracts mean and divides by standard deviation.
            iqr:  subtracts median and divides by interquartile range.
            mod_z: modified z-score using median absolute deviation.
            min-max: rescales to values between 0 and 1 by subtracting the min and dividing by the range.
            percentile: converts values to their percentile rank relative to the observations in the
            defined window type.
        axis: str, {'ts', 'cs'}, default 'ts'
            Axis along which to compute dispersion, time series or cross-section.
        centering: bool, default True
            Centers values using the appropriate measure of central tendency used for the selected method. Otherwise,
            0 is used.
        window_type: str, {'fixed', 'expanding', 'rolling}, default 'fixed'
            Provide a window type. If None, all observations are used in the calculation.
        window_size: int, default 10
            Size of the moving window. This is the minimum number of observations used for the rolling or expanding
            window statistic.
        min_periods: int, default 2
            Minimum number of observations in window required to have a value; otherwise, result is np.nan.
        winsorize: int, default None
            Winsorizes/clips values to between positive and negative values of specified integer.

        Returns
        -------
        norm_features: pd.Series or pd.DataFrame
            Series or DataFrame with dispersion of values.
        """
        #  method
        if method == 'z-score':
            # center
            if centering:
                center = self.center(axis=axis, central_tendency='mean', window_type=window_type,
                                     window_size=window_size, min_periods=min_periods)
            else:
                center = self.df
            # normalize
            if axis == 'cs':
                self.trans_df = center.divide(self.compute_std(axis=axis, window_type=window_type,
                                                               window_size=window_size, min_periods=min_periods),
                                              axis=0)
            else:
                self.trans_df = center / self.compute_std(axis=axis, window_type=window_type, window_size=window_size,
                                                          min_periods=min_periods)
        elif method == 'iqr':
            # center
            if centering:
                center = self.center(axis=axis, central_tendency='median', window_type=window_type,
                                     window_size=window_size, min_periods=min_periods)
            else:
                center = self.df
            # normalize
            if axis == 'cs':
                self.trans_df = center.divide(self.compute_iqr(axis=axis, window_type=window_type,
                                                               window_size=window_size, min_periods=min_periods),
                                              axis=0)
            else:
                self.trans_df = center / self.compute_iqr(axis=axis, window_type=window_type, window_size=window_size,
                                                          min_periods=min_periods)
        elif method == 'mod_z':
            # center
            if centering:
                center = self.center(axis=axis, central_tendency='median', window_type=window_type,
                                     window_size=window_size, min_periods=min_periods)
            else:
                center = self.df
            # normalize
            if axis == 'cs':
                self.trans_df = center.divide(self.compute_mad(axis=axis, window_type=window_type,
                                                               window_size=window_size, min_periods=min_periods),
                                              axis=0)
            else:
                self.trans_df = center / self.compute_mad(axis=axis, window_type=window_type, window_size=window_size,
                                                          min_periods=min_periods)
            self.trans_df = 0.6745 * self.trans_df

        elif method == 'min-max':
            # center
            if centering:
                center = self.center(axis=axis, central_tendency='min', window_type=window_type,
                                     window_size=window_size, min_periods=min_periods)
            else:
                center = self.df
            # normalize
            if axis == 'cs':
                self.trans_df = center.divide(self.compute_range(axis=axis, window_type=window_type,
                                                               window_size=window_size, min_periods=min_periods),
                                              axis=0)
            else:
                self.trans_df = center / self.compute_range(axis=axis, window_type=window_type,
                                                          window_size=window_size,
                                                          min_periods=min_periods)

        elif method == 'percentile':
            self.trans_df = self.compute_percentile(axis=axis, window_type=window_type, window_size=window_size,
                                                    min_periods=min_periods)

        # winsorize
        if winsorize is not None and method in ['z-score', 'iqr', 'mod_z']:
            self.trans_df = self.trans_df.clip(winsorize * -1, winsorize)

        return self.trans_df

    def quantize(self,
                 bins: int = 5,
                 axis: str = 'ts',
                 window_type: str = 'fixed',
                 window_size: int = 30,
                 min_periods: int = 2
                 ) -> Union[pd.Series, pd.DataFrame]:
        """
        Quantize features.

        Parameters
        ----------
        bins: int, default 5
            Number of bins to use for quantization.
        axis: str, {'ts', 'cs'}, default 'ts'
            Axis along which to compute dispersion, time series or cross-section.
        window_type: str, {'fixed', 'expanding', 'rolling}, default 'fixed'
            Provide a window type. If None, all observations are used in the calculation.
        window_size: int, default 30
            Size of the moving window. This is the minimum number of observations used for the rolling or expanding
            window statistic.
        min_periods: int, default 2
            Minimum number of observations in window required to have a value; otherwise, result is np.nan.

        Returns
        -------
        quantized_features: pd.Series or pd.DataFrame
            Series or DataFrame with quantized values.
        """
        if bins <= 1 or bins is None:
            raise ValueError('Number of bins must be larger than 1. Please increase number of bins.')

        # compute percentile
        perc = self.compute_percentile(axis=axis, window_type=window_type, window_size=window_size,
                                       min_periods=min_periods)

        # quantize
        labels = np.arange(0, 1 + 1 / bins, 1 / bins)
        # mask nans
        mask = np.isnan(perc)
        # quantize
        quantiles = np.digitize(perc, labels, right=True)
        self.trans_df = np.where(mask, perc, quantiles)

        # add to df
        if self.index is not None:
            self.trans_df = pd.DataFrame(self.trans_df, index=perc.index, columns=perc.columns)

        return self.trans_df

    #     elif method == 'cdf':
    #         norm_df = (df - center) / mov_df.std()
    #         norm_df = pd.DataFrame(stats.norm.cdf(norm_df), index=norm_df.index, columns=norm_df.columns)

    @staticmethod
    def quant(df: Union[pd.Series, pd.DataFrame],
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
        # check bins
        if bins <= 1 or bins is None:
            raise ValueError('Number of bins must be larger than 1. Please increase number of bins.')
        # convert to df
        if isinstance(df, pd.Series):
            df = df.to_frame().copy()

        # quantize function
        # def quantize(x, _bins=bins, _axis=axis):
        #
        #     # percentile rank
        #     perc = x.rank(pct=True, axis=_axis)
        #     # bins
        #     labels = np.arange(0, 1, 1 / _bins)
        #     quantiles = np.digitize(perc, labels, right=True)
        #
        #     return quantiles
        #
        # def quant_mw(x):
        #     return quantize(x)[-1]

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

    def quant_ts(self,
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
        # multiindex
        if isinstance(self.df.index, pd.MultiIndex):
            quant_df = self.quant(self.df.unstack().dropna(how='all', axis=1),
                                     bins=bins,
                                     window_type=window_type,
                                     lookback=lookback,
                                     _min_periods=_min_periods).stack()
        # single index
        else:
            quant_df = self.quant(self.df.dropna(how='all', axis=1),
                                     bins=bins,
                                     window_type=window_type,
                                     lookback=lookback,
                                     _min_periods=_min_periods)

        return quant_df

    def quant_cs(self, bins: int = 5) -> Union[pd.Series, pd.DataFrame]:
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

        # multiindex
        if isinstance(self.df.index, pd.MultiIndex):
            # loop through cols
            for col in self.df.columns:
                df = self.quant(self.df[col].unstack().dropna(thresh=bins, axis=0), axis=1).stack().to_frame(col)
                quant_df = pd.concat([quant_df, df], axis=1)

        # single index
        else:
            quant_df = self.quant(self.df.dropna(thresh=bins, axis=0), bins=5, axis=1)

        return quant_df.sort_index()

    def disc(self,
             bins: int = 5,
             axis: str = 'ts',
             method: str = 'quantile',
             window_type: str = 'fixed',
             window_size: int = 30,
             min_obs: int = 1
             ) -> pd.DataFrame:
        """
        Discretizes normalized factors or targets. Discretization is the process of transforming a continuous variable
        into a discrete one by creating a set of bins (or equivalently contiguous intervals/cuttoffs) that spans
        the range of the variable’s values.

        Parameters
        ----------
        bins: int, default 5
            Number of bins.
        axis: str, {'ts', 'cs'}, default 'ts'
            Axis along which to compute median absolute deviation, time series or cross-section.
        method: str, {'quantile', 'uniform', 'kmeans'}, default 'quantile'
            quantile: all bins have the same number of values.
            uniform: all bins have identical widths.
            kmeans: values in each bin have the same nearest center of a 1D k-means cluster.
        window_type: str, {'fixed', 'expanding', 'rolling}, default 'fixed'
            Provide a window type. If None, all observations are used in the calculation.
        window_size: int, default 30
            Size of the moving window. This is the minimum number of observations used for the rolling or
            expanding statistic.
        min_obs: int, default 1
            Minimum number of observations in window required to have a value; otherwise, result is np.nan.

        Returns
        -------
        disc_features: DataFrame
            Series or DataFrame with DatetimeIndex and discretized features.
        """
        # check bins
        if bins <= 1 or bins is None:
            raise ValueError('Number of bins must be larger than 1. Please increase number of bins.')

        # discretize features
        discretize = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy=method)
        # store discretized features
        disc = None

        # axis time series
        if axis == 'ts':

            if isinstance(self.index, pd.MultiIndex):
                self.trans_df = self.df.unstack().copy()

            if window_type == 'rolling':

                # loop through rows of df
                for row in range(self.trans_df.shape[0] - window_size + 1):
                    # discretize features
                    res = discretize.fit_transform(self.trans_df.iloc[row: row + window_size])
                    # store results
                    if row == 0:
                        disc = res[-1]
                    else:
                        disc = np.vstack([disc, res[-1]])

            elif window_type == 'expanding':

                # loop through rows of df
                for row in range(min_obs, self.trans_df.shape[0] + 1):
                    # discretize features
                    res = discretize.fit_transform(self.trans_df.iloc[:row])
                    # store results
                    if row == min_obs:
                        disc = res[-1]
                    else:
                        disc = np.vstack([disc, res[-1]])

            else:  # fixed window
                disc = discretize.fit_transform(self.trans_df)

            # convert to df
            disc = pd.DataFrame(disc, index=self.trans_df.index[-disc.shape[0]:], columns=self.trans_df.columns)

            if isinstance(self.index, pd.MultiIndex):
                disc = disc.stack()

        # axis 1 (cross-section)
        else:
            def discretization(df):
                return pd.DataFrame(discretize.fit_transform(df), index=df.index, columns=df.columns)

            disc = self.trans_df.groupby(level=0, group_keys=False).apply(lambda x: discretization(x))

        # convert type to float
        disc = disc.astype(float) + 1

        return disc

    @staticmethod
    def discretize(df: pd.DataFrame,
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
