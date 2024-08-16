import pandas as pd
import numpy as np
from typing import Union, Optional
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

    def diff(self, lags: int = 1) -> Union[pd.Series, pd.DataFrame]:
        """
        Computes difference.

        Parameters
        ----------
        lags: int, default 1
            Number of periods to lag the difference.

        Returns
        -------
        df: pd.Series or pd.DataFrame
            Series or DataFrame with difference.
        """
        if isinstance(self.index, pd.MultiIndex):
            self.trans_df = self.df.groupby(level=1).diff(lags)
        else:
            self.trans_df = self.df.diff(lags)

        return self.trans_df.sort_index()

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
                self.trans_df = self.df.groupby(level=1).pct_change(lags, fill_method=None)
            else:
                self.trans_df = self.df.pct_change(lags, fill_method=None)

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
                # TODO: add other market computations
                raise ValueError("Market weighting not supported yet.")

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

        # axis cross-section
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

        # axis cross-section
        else:
            # fixed window
            if isinstance(self.df.index, pd.MultiIndex):
                self.trans_df = self.df.groupby(level=0).std()
            else:
                self.trans_df = self.df.std(axis=1)

        # convert to df
        if isinstance(self.trans_df, pd.Series):
            self.trans_df = self.trans_df.to_frame()

        return self.trans_df

    def compute_quantile(self,
                         q: float,
                         axis: str = 'ts',
                         window_type: str = 'fixed',
                         window_size: int = 30,
                         min_periods: int = 2,
                         window_fcn: str = None
                         ) -> Union[pd.Series, pd.DataFrame]:
        """
        Computes quantile.

        Parameters
        ----------
        q: float
            Quantile to compute.
        axis : str, {'ts', 'cs'}, default 'ts'
            Axis along which to compute quantile, time series or cross-section.
        window_type: str, {'fixed', 'expanding', 'rolling}, default 'fixed'
            Provide a window type. If None, all observations are used in the calculation.
        window_size: int, default 30
            Size of the moving window. This is the minimum number of observations used for the rolling or expanding
        min_periods: int, default 2
            Minimum number of observations in window required to have a value; otherwise, result is np.nan.
        window_fcn: str, default None
            Provide a window function. If None, observations are equally-weighted in the rolling computation.

        Returns
        -------
        quantile: pd.Series or pd.DataFrame
            Series or DataFrame with quantile of values.
        """
        # axis time series
        if axis == 'ts':
            # rolling window
            if window_type == 'rolling':
                if isinstance(self.df.index, pd.MultiIndex):
                    self.trans_df = self.df.groupby(level=1).rolling(window=window_size, min_periods=min_periods,
                                                       win_type=window_fcn).quantile(q).droplevel(0).sort_index()
                else:
                    self.trans_df = self.df.rolling(window=window_size, min_periods=min_periods,
                                                    win_type=window_fcn).quantile(q)
            # expanding window
            elif window_type == 'expanding':
                if isinstance(self.df.index, pd.MultiIndex):
                    self.trans_df = self.df.groupby(level=1).expanding(min_periods=min_periods).quantile(q).\
                        droplevel(0).sort_index()

                else:
                    self.trans_df = self.df.expanding(min_periods=min_periods).quantile(q)
            # fixed window
            else:
                if isinstance(self.df.index, pd.MultiIndex):
                    self.trans_df = self.df.groupby(level=1).quantile(q)
                else:
                    self.trans_df = self.df.quantile(q)

        # axis cross-section
        else:
            if isinstance(self.df.index, pd.MultiIndex):
                self.trans_df = self.df.groupby(level=0).quantile(q)
            else:
                self.trans_df = self.df.quantile(q, axis=1)

        # convert to df
        if isinstance(self.trans_df, pd.Series):
            self.trans_df = self.trans_df.to_frame()

        return self.trans_df

    def compute_iqr(self,
                    axis: str = 'ts',
                    window_type: str = 'fixed',
                    window_size: int = 30,
                    min_periods: int = 2,
                    window_fcn: str = None
                    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Computes interquartile range.

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

        # axis cross-section
        else:
            if isinstance(self.df.index, pd.MultiIndex):
                self.trans_df = self.df.groupby(level=0).quantile(0.75) - self.df.groupby(level=0).quantile(0.25)
            else:
                self.trans_df = self.df.quantile(0.75, axis=1) - self.df.quantile(0.25, axis=1)

        # convert to df
        if isinstance(self.trans_df, pd.Series):
            self.trans_df = self.trans_df.to_frame()

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

        # axis cross-section
        else:
            if isinstance(self.df.index, pd.MultiIndex):
                self.trans_df = (self.df - self.df.groupby(level=0).median()).abs().groupby(level=0).median()
            else:
                self.trans_df = (self.df - self.df.median(axis=1)).abs().median(axis=1)

        # convert to df
        if isinstance(self.trans_df, pd.Series):
            self.trans_df = self.trans_df.to_frame()

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

        # axis cross-section
        else:
            if isinstance(self.df.index, pd.MultiIndex):
                self.trans_df = self.df.groupby(level=0).max() - self.df.groupby(level=0).min()
            else:
                self.trans_df = self.df.max(axis=1) - self.df.min(axis=1)

        # convert to df
        if isinstance(self.trans_df, pd.Series):
            self.trans_df = self.trans_df.to_frame()

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

        # axis cross-section
        else:
            # fixed window
            if isinstance(self.df.index, pd.MultiIndex):
                self.trans_df = self.df.groupby(level=0).var()
            else:
                self.trans_df = self.df.var(axis=1)

        # convert to df
        if isinstance(self.trans_df, pd.Series):
            self.trans_df = self.trans_df.to_frame()

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
            else:
                tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

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

        # axis cross-section
        else:
            raise ValueError("Cross-section not supported for ATR computation.")

        # convert to df
        if isinstance(self.trans_df, pd.Series):
            self.trans_df = self.trans_df.to_frame()

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

        # axis cross-section
        else:
            # fixed window
            if isinstance(self.df.index, pd.MultiIndex):
                self.trans_df = self.df.groupby(level=0).rank(pct=True)
            else:
                self.trans_df = self.df.rank(pct=True, axis=1)

        # convert to df
        if isinstance(self.trans_df, pd.Series):
            self.trans_df = self.trans_df.to_frame()

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
        if method == 'percentile':
            self.trans_df = self.compute_percentile(axis=axis, window_type=window_type,
                                                    window_size=window_size, min_periods=min_periods)
        else:
            # centering
            if centering:
                if method == 'iqr' or method == 'mod_z':
                    center = self.center(axis=axis, central_tendency='median', window_type=window_type,
                                         window_size=window_size, min_periods=min_periods)
                elif method == 'min-max':
                    center = self.center(axis=axis, central_tendency='min', window_type=window_type,
                                         window_size=window_size, min_periods=min_periods)
                else:
                    center = self.center(axis=axis, central_tendency='mean', window_type=window_type,
                                         window_size=window_size, min_periods=min_periods)
            else:
                center = self.df

            # disp
            if method == 'iqr':
                disp = self.compute_iqr(axis=axis, window_type=window_type, window_size=window_size,
                                        min_periods=min_periods)
            elif method == 'mod_z':
                disp = self.compute_mad(axis=axis, window_type=window_type, window_size=window_size,
                                        min_periods=min_periods)
            elif method == 'min-max':
                disp = self.compute_range(axis=axis, window_type=window_type, window_size=window_size,
                                          min_periods=min_periods)
            else:
                disp = self.compute_std(axis=axis, window_type=window_type, window_size=window_size,
                                        min_periods=min_periods)

            # normalize
            if isinstance(self.df.index, pd.MultiIndex):  # multiindex
                self.trans_df = center / disp
            else:  # single index
                if axis == 'ts':
                    if window_type == 'fixed':
                        self.trans_df = center / disp.iloc[:, 0]
                    else:
                        self.trans_df = center / disp
                else:
                    self.trans_df = center / disp.values

        if method == 'mod_z':
            self.trans_df = 0.6745 * self.trans_df

        # winsorize
        if winsorize is not None and method in ['z-score', 'iqr', 'mod_z']:
            self.trans_df = self.trans_df.clip(winsorize * -1, winsorize)

        # convert to df
        if isinstance(self.trans_df, pd.Series):
            self.trans_df = self.trans_df.to_frame()

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
        self.trans_df = (perc * bins).apply(np.ceil).astype(float)

        # add to df
        if self.index is not None:
            self.trans_df = pd.DataFrame(self.trans_df, index=perc.index, columns=perc.columns)

        return self.trans_df

    def discretize(self,
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

        # axis cross-section
        else:
            def discretization(df):
                return pd.DataFrame(discretize.fit_transform(df), index=df.index, columns=df.columns)

            disc = self.trans_df.groupby(level=0, group_keys=False).apply(lambda x: discretization(x))

        # convert type to float
        disc = disc.astype(float) + 1

        return disc
