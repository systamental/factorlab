import pandas as pd
import numpy as np
from typing import Optional
from scipy.stats import norm, logistic

from factorlab.feature_engineering.transformations import Transform


class Signal:
    """
    Signal construction class.
    """
    def __init__(self,
                 returns: pd.Series,
                 factors: pd.DataFrame,
                 strategy: str = 'ts_ls',
                 return_bins: int = 3,
                 factor_bins: int = 5,
                 n_factors: int = 10,
                 disc_thresh: float = 0,
                 window_type: str = 'expanding',
                 window_size: int = 90,
                 ):
        """
        Constructor

        Parameters
        ----------
        returns: pd.Series or pd.DataFrame - Single or MultiIndex
            Dataframe or series with DatetimeIndex (level 0), tickers (level 1) and returns (cols).
        factors: pd.Series or pd.DataFrame - Single or MultiIndex
            Dataframe with DatetimeIndex (level 0), tickers (level 1) and factors (cols).
        strategy: str, {'ts_ls' 'ts_l', 'cs_ls', 'cs_l', 'dual_ls', 'dual_l', default 'ts_ls'
            Time series, cross-sectional or dual strategy, long/short, long-only or short-only.
        factor_bins: int, default 5
            Number of bins to create for factors.
        return_bins: int, default 3
            Number of bins to create for returns.
        n_factors: int, default 10
            Number of factors to use for cross-sectional strategies.
        disc_thresh: float, default 0
            Threshold cutoff for converting continuous signal values to discrete signals.
        window_type: str, {'fixed', 'expanding', 'rolling'}, default 'expanding'
            Window type for normalization.
        window_size: int, default 90
            Minimal number of observations to include in moving window (rolling or expanding).
        """
        self.returns = returns.astype(float).to_frame() if isinstance(returns, pd.Series) else returns
        self.factors = factors.to_frame() if isinstance(factors, pd.Series) else factors
        self.strategy = strategy
        self.factor_bins = factor_bins if factor_bins > 1 else self._raise_value_error()
        self.return_bins = return_bins if return_bins > 1 else self._raise_value_error()
        self.n_factors = n_factors
        self.disc_thresh = disc_thresh
        self.window_type = window_type
        self.window_size = window_size
        self.norm_factors = None
        self.norm_ret = None
        self.factor_quantiles = None
        self.ret_quantiles = None
        self.signals = None
        self.signal_rets = None
        self.signal_disp = None

    def _raise_value_error(self):
        raise ValueError(f"Number of bins must be larger than 1.")

    def normalize(self,
                  method: str = 'z-score',
                  centering: bool = True,
                  ts_norm: bool = False,
                  winsorize: Optional[int] = None
                  ) -> pd.DataFrame:
        """
        Normalizes factors and/or targets.

        Parameters
        ---------
        method: str, {'z-score', 'iqr', 'mod_z', 'min-max', 'percentile'}, default 'z-score'
            z-score: subtracts mean and divides by standard deviation.
            iqr:  subtracts median and divides by interquartile range.
            mod_z: modified z-score using median absolute deviation.
            min-max: rescales to values between 0 and 1 by subtracting the min and dividing by the range.
            percentile: converts values to their percentile rank relative to the observations in the
            defined window type.
        centering: bool, default True
            Centers values using the appropriate measure of central tendency used for the selected method. Otherwise,
            0 is used.
        ts_norm: bool, default False
            Normalizes factors over the time series before normalizing over the cross-section.
        winsorize: int, default None
            Winsorizes/clips values to between positive and negative values of specified integer.

        Returns
        -------
        norm_factors: pd.DataFrame
            Normalized factors with DatetimeIndex and normalized values (cols).
        """
        # time series
        if self.strategy.split('_')[0] == 'ts':
            self.norm_factors = Transform(self.factors).normalize(method=method, axis='ts', centering=centering,
                                                                  window_type=self.window_type,
                                                                  window_size=self.window_size, winsorize=winsorize)
            self.norm_ret = Transform(self.returns).normalize(method=method, axis='ts', centering=centering,
                                                          window_type=self.window_type, window_size=self.window_size,
                                                          winsorize=winsorize)

        # cross-sectional
        elif self.strategy.split('_')[0] == 'cs':
            if ts_norm:
                factor_norm_ts = Transform(self.factors).normalize(method=method, axis='ts', centering=centering,
                                                                   window_type=self.window_type,
                                                                   window_size=self.window_size, winsorize=winsorize)
                ret_norm_ts = Transform(self.returns).normalize(method=method, axis='ts', centering=centering,
                                                            window_type=self.window_type,
                                                            window_size=self.window_size, winsorize=winsorize)
                self.norm_factors = Transform(factor_norm_ts).normalize(method=method, axis='cs', centering=centering,
                                                                        winsorize=winsorize)
                self.norm_ret = Transform(ret_norm_ts).normalize(method=method, axis='cs', centering=centering,
                                                                 winsorize=winsorize)
            else:
                self.norm_factors = Transform(self.factors).normalize(method=method, axis='cs', centering=centering,
                                                                      winsorize=winsorize)
                self.norm_ret = Transform(self.returns).normalize(method=method, axis='cs', centering=centering,
                                                              winsorize=winsorize)

        return self.norm_factors

    def quantize(self, ts_norm: bool = False) -> pd.DataFrame:
        """
        Quantizes factors and/or targets.

        Parameters
        ---------
        ts_norm: bool, default False
            Normalizes factors over the time series before quantization over the cross-section.

        Returns
        -------
        factor_quantiles: pd.Series or pd.DataFrame
            Quantized factors with DatetimeIndex and quantized values (cols).
        """
        # time series
        if self.strategy.split('_')[0] == 'ts':
            self.factor_quantiles = Transform(self.factors).quantize(bins=self.factor_bins, axis='ts',
                                                                     window_type=self.window_type,
                                                                     window_size=self.window_size)
            self.ret_quantiles = Transform(self.returns).quantize(bins=self.return_bins, axis='ts',
                                                              window_type=self.window_type,
                                                              window_size=self.window_size)
        # cross-sectional
        elif self.strategy.split('_')[0] == 'cs':
            if ts_norm:
                norm_factors_ts = Transform(self.factors).normalize(window_type=self.window_type, axis='ts',
                                                                    window_size=self.window_size)
                norm_ret_ts = Transform(self.returns).normalize(window_type=self.window_type, axis='ts',
                                                            window_size=self.window_size)
                self.factor_quantiles = Transform(norm_factors_ts).quantize(bins=self.factor_bins, axis='cs')
                self.ret_quantiles = Transform(norm_ret_ts).quantize(bins=self.return_bins, axis='cs')

            else:
                self.factor_quantiles = Transform(self.factors).quantize(bins=self.factor_bins, axis='cs')
                self.ret_quantiles = Transform(self.returns).quantize(bins=self.return_bins, axis='cs')

        return self.factor_quantiles

    def convert_to_signals(self,
                           transformation: Optional[str] = 'norm',
                           ts_norm: bool = False,
                           winsorize: Optional[int] = None
                           ) -> pd.DataFrame:
        """
        Converts raw factors to signals using a probability density function.

        Factor scores are converted to continuous signals between 1 and -1.

        Parameters
        ----------
        transformation: str, {'norm', 'percentile', 'min-max', 'logistic', 'adj_norm', 'sign'}, default 'norm'
            Transformation to convert raw factor values to signals between 1 and -1.
        ts_norm: bool, default False
            Normalizes factors over the time series before normalizing over the cross-section.
        winsorize: int, optional, default None
            Winsorizes/clips values to between positive and negative values of specified integer.

        Returns
        -------
        signals: pd.Series or pd.DataFrame - Single or MultiIndex
            Continuous signals with DatetimeIndex (level 0), tickers (level 1) and signal values (cols).
        """
        # transformation
        # normal distribution
        if transformation == 'norm':
            self.normalize(method='z-score', centering=True, ts_norm=ts_norm, winsorize=winsorize)
            # cumulative distribution function of normal distribution
            self.signals = pd.DataFrame(norm.cdf(self.norm_factors), index=self.norm_factors.index,
                                        columns=self.norm_factors.columns)

        # uniform distribution
        elif transformation == 'percentile':
            self.normalize(method='percentile', centering=True, ts_norm=ts_norm, winsorize=winsorize)
            self.signals = self.norm_factors

        # min-max
        elif transformation == 'min-max':
            self.normalize(method='min-max', centering=True, ts_norm=ts_norm, winsorize=winsorize)
            self.signals = self.norm_factors

        # logistic
        elif transformation == 'logistic':
            self.normalize(method='z-score', centering=True, ts_norm=ts_norm, winsorize=winsorize)
            # cdf of logistic distribution
            self.signals = pd.DataFrame(logistic.cdf(self.norm_factors), index=self.norm_factors.index,
                                        columns=self.norm_factors.columns)

        # adjusted normal distribution
        elif transformation == 'adj_norm':
            self.normalize(method='adj_norm', centering=True, ts_norm=ts_norm, winsorize=winsorize)
            self.signals = self.norm_factors * np.exp((-1 * self.norm_factors ** 2) / 4) / 0.89

        # sign
        elif transformation == 'sign':
            self.signals = np.sign(self.factors)

        else:
            self.signals = self.factors

        # convert to signals
        if transformation not in ['adj_norm', 'sign']:
            self.signals = (self.signals * 2) - 1

        return self.signals

    def discretize_signals(self,
                           transformation: str = 'norm',
                           ts_norm: bool = False,
                           winsorize: Optional[int] = None
                           ) -> pd.DataFrame:
        """
        Converts factor signals to discrete signals, 1, 0 or -1.

        Parameters
        ----------
        transformation: str, {'norm', 'percentile', 'min-max', 'logistic', 'adj_norm', 'sign'}, default 'norm'
            Transformation to convert raw factor values to signals between 1 and -1.
        ts_norm: bool, default False
            Normalizes factors over the time series before normalizing over the cross-section.
        winsorize: int, default 3
            Winsorizes/clips values to between [clip *-1, clip].

        Returns
        -------
        disc_signal: pd.Series or pd.DataFrame - Single or MultiIndex
            Discrete signals (-1, 0, 1) with DatetimeIndex (level 0), tickers (level 1) and signal values (cols).
        """
        # signals
        self.convert_to_signals(transformation=transformation, ts_norm=ts_norm, winsorize=winsorize)

        # discretize
        self.signals = self.signals.apply(lambda x: np.where(np.abs(x) >= self.disc_thresh, np.sign(x), 0))

        return self.signals

    def signals_to_quantiles(self,
                             transformation: str = 'norm',
                             ts_norm: bool = False,
                             winsorize: Optional[int] = None
                             ) -> pd.DataFrame:
        """
        Converts signals to signal quantiles.

        Parameters
        ----------
        transformation: str, {'norm', 'percentile', 'min-max', 'logistic', 'adj_norm', 'sign'}, default 'norm'
            Transformation to convert raw factor values to signals between 1 and -1.
        ts_norm: bool, default False
            Normalizes factors over the time series before quantization over the cross-section.
        winsorize: int, optional, default None
            Winsorizes/clips values to between positive and negative values of specified integer.

        Returns
        -------
        signal_quantiles: pd.Series or pd.DataFrame - Single or MultiIndex
            Signal quantiles with DatetimeIndex (level 0), tickers (level 1) and signal quantile values (cols).
        """
        # signals
        self.convert_to_signals(transformation=transformation, ts_norm=ts_norm, winsorize=winsorize)

        # quantiles
        self.factor_quantiles = Transform(self.signals).quantize(bins=self.factor_bins,
                                                                 axis=self.strategy.split('_')[0],
                                                                 window_type=self.window_type,
                                                                 window_size=self.window_size)

        # median centered
        med = np.median(range(1, self.factor_bins + 1))
        self.signals = (self.factor_quantiles - med) / (med - 1)

        return self.signals

    def signals_to_rank(self,
                        transformation: Optional[str] = None,
                        ts_norm: bool = False,
                        winsorize: Optional[int] = None
                        ) -> pd.DataFrame:
        """
        Ranks signals in the cross-section for top/bottom n factors for each time period.

        Parameters
        ----------
        transformation: str, {'norm', 'percentile', 'min-max', 'logistic', 'adj_norm', 'sign'}, default 'norm'
            Transformation to convert raw factor values to signals between 1 and -1.
        ts_norm: bool, default False
            Normalizes factors over the time series before quantization over the cross-section.
        winsorize: int, optional, default None
            Winsorizes/clips values to between positive and negative values of specified integer.

        Returns
        -------
        rank_df: pd.Series or pd.DataFrame - Single or MultiIndex
            Signal ranks with DatetimeIndex (level 0), tickers (level 1) and signal rank values (cols).
        """
        # time series
        if self.strategy.split('_')[0] == 'ts':
            raise ValueError("Signal rank is only available for cross-sectional strategies.")
        else:
            # signals
            self.convert_to_signals(transformation=transformation, ts_norm=ts_norm, winsorize=winsorize)

            # min n cross-section cutoff
            self.signals = self.signals[(self.signals.groupby(level=0).count() >= self.n_factors * 2)].dropna()

            # group by level 0 index (date) and rank values
            ranks = self.signals.groupby(level=0).rank(method='first')

            # upper/lower threshold values
            upper_thresh = ranks.groupby(level=0).max() - self.n_factors
            lower_thresh = ranks.groupby(level=0).min() + self.n_factors
            # sort top/bottom n
            bottom_n = ranks.lt(lower_thresh)
            top_n = ranks.gt(upper_thresh)

            # assign 1 to highest n values, -1 to lowest n values, and 0 to the rest
            self.signals = pd.DataFrame(data=0, index=self.signals.index, columns=self.signals.columns)
            self.signals[top_n] = 1
            self.signals[bottom_n] = -1

        return self.signals

    def compute_signals(self,
                        signal_type: str = 'signal',
                        transformation: str = 'norm',
                        ts_norm: bool = False,
                        winsorize: int = 3,
                        leverage: Optional[int] = None,
                        lags: Optional[int] = None
                        ) -> pd.DataFrame:
        """
        Computes signals by converting raw data to normalized signals (alpha factors).

        Parameters
        ----------
        signal_type: str, {'signal', 'signal_quantiles', 'disc_signal', 'sign', 'signal_rank'}, default 'signal'
            signal: factor inputs are converted to signals between -1 and 1 for l/s strategies, between 0 and 1 for long
            only strategies, and between -1 and 0 for short only strategies.
            disc_signal: factor inputs are converted to discrete signals -1, 0 and 1 for l/s strategies,
            0 or 1 for long only strategies, and -1 or 0 for short only strategies.
            signal_quantiles: factor inputs are converted to quantized signals between -1 and 1 for l/s strategies,
             between 0 and 1 for long-only strategies, and between -1 and 0 for short only strategies, with n bins.
            signal_rank: factor inputs are converted to signal ranks between -1 and 1, 0 or 1 for long only
            strategies, and -1 or 0 for short only strategies, with n factors.
        transformation: str, {'norm', 'percentile', 'min-max', 'logistic', 'adj_norm', 'sign'}, default 'norm'
            Transformation to convert raw factor values to signals between 1 and -1.
        ts_norm: bool, default False
            Normalizes factors over the time series before quantization over the cross-section.
        winsorize: int, default 3
            Max/min value to use for winsorization/clipping for signals when method is z-score, iqr or mod z.
        leverage: int, default None
            Multiplies factors by integer to increase leverage.
        lags: int, optional, default None
            Number of periods to lag signals/forward returns.

        Returns
        -------
        signals: pd.DataFrame
            Dataframe with DatetimeIndex (level 0), tickers (level 1) and computed signals (cols).
        """
        # raw factors, signal_type None
        if signal_type is None:
            self.signals = self.factors

        # signals, signal_type 'signal'
        if signal_type == 'signal':
            self.convert_to_signals(transformation=transformation, ts_norm=ts_norm, winsorize=winsorize)

        # quantized signals, signal_type 'signal_quantiles'
        if signal_type == 'signal_quantiles':
            self.signals_to_quantiles(transformation=transformation, ts_norm=ts_norm, winsorize=winsorize)

        # discrete signals, signal_type 'disc_signal'
        if signal_type == 'disc_signal':
            self.discretize_signals(transformation=transformation, ts_norm=ts_norm, winsorize=winsorize)

        # sign, signal_type 'sign'
        if signal_type == 'sign':
            self.convert_to_signals(transformation='sign', ts_norm=ts_norm, winsorize=winsorize)

        # signal rank, signal_type 'signal_rank'
        if signal_type == 'signal_rank':
            self.signals_to_rank(transformation=None, ts_norm=ts_norm, winsorize=winsorize)

        # long or short only
        if self.strategy.split('_')[1] == 'l':
            self.signals = self.signals.clip(lower=0)
        elif self.strategy.split('_')[1] == 's':
            self.signals = self.signals.clip(upper=0)

        # leverage
        if leverage is not None:
            self.signals *= leverage

        # lags
        if lags is not None:
            if isinstance(self.signals.index, pd.MultiIndex):
                self.signals = self.signals.groupby(level=1).shift(lags)
            else:
                self.signals = self.signals.shift(lags)

        return self.signals

    def compute_dual_signals(self,
                             summary_stat: str = 'mean',
                             signal_type: str = 'signal',
                             transformation: Optional[str] = 'norm',
                             ts_norm: bool = False,
                             winsorize: int = 3,
                             leverage: Optional[int] = None,
                             lags: Optional[int] = None
                             ):
        """
        Computes dual signals by converting raw data to normalized signals (alpha factors).

        Parameters
        ----------
        summary_stat: str, {'mean', 'median', 'min', 'max', 'sum', 'prod'}
            Summary statistic to compute dual signals.
        signal_type: str, {'signal', 'disc_signal', 'signal_quantiles', 'sign', 'signal_rank'}, default 'signal'
            signal: factor inputs are converted to signals between -1 and 1 for l/s strategies, between 0 and 1 for long
            only strategies, and between -1 and 0 for short only strategies.
            disc_signal: factor inputs are converted to discrete signals -1, 0 and 1 for l/s strategies,
            0 or 1 for long only strategies, and -1 or 0 for short only strategies.
            signal_quantiles: factor inputs are converted to quantized signals between -1 and 1 for l/s strategies,
            between 0 and 1 for long-only strategies, and between -1 and 0 for short only strategies, with n bins.
            signal_rank: factor inputs are converted to signal ranks between -1 and 1, 0 or 1 for long only
            strategies, and -1 or 0 for short only strategies, with n factors.
        transformation: str, {'norm', 'percentile', 'min-max', 'logistic', 'adj_norm', 'sign'}, default 'norm'
            Probability density function to convert raw factor values to signals between 1 and -1.
        ts_norm: bool, default False
            Normalizes factors over the time series before quantization over the cross-section.
        winsorize: int, default 3
            Max/min value to use for winsorization/clipping for signals when method is z-score, iqr or mod z.
        leverage: int, default None
            Multiplies factors by integer to increase leverage.
        lags: int, optional, default None
            Number of periods to lag signals/forward returns.

        Returns
        -------
        dual_signals: pd.DataFrame
            Dataframe with DatetimeIndex (level 0), tickers (level 1) and computed dual signals (cols).
        """
        # check strategy
        if self.strategy.split('_')[0] != 'dual':
            raise ValueError("Dual strategy must be selected to compute dual signals.")

        # strategy placeholder
        strategy = self.strategy

        # normalize factors
        self.strategy = 'ts' + '_' + strategy.split('_')[1]
        norm_factors_ts = self.normalize(method='z-score', centering=True, ts_norm=ts_norm, winsorize=winsorize)
        self.strategy = 'cs' + '_' + strategy.split('_')[1]
        norm_factors_cs = self.normalize(method='z-score', centering=True, ts_norm=ts_norm, winsorize=winsorize)

        # compute dual factors
        self.norm_factors = None
        factors = pd.concat([norm_factors_ts, norm_factors_cs], axis=1, join='inner')
        for factor in norm_factors_ts.columns:
            self.norm_factors = pd.concat([self.norm_factors,
                                           getattr(factors[factor], summary_stat)(axis=1).to_frame(factor)],
                                          axis=1)

        # compute dual signals
        self.strategy = 'dual' + '_' + strategy.split('_')[1]
        self.signals = self.compute_signals(signal_type=signal_type, transformation=transformation, ts_norm=ts_norm,
                                            winsorize=winsorize, leverage=leverage, lags=lags)

        return self.signals

    def compute_signal_returns(self,
                               signal_type: str = 'signal',
                               transformation: str = 'norm',
                               ts_norm: bool = False,
                               winsorize: int = 3,
                               leverage: Optional[int] = None,
                               lags: int = 1,
                               dual_summary_stat: str = 'mean',
                               ):
        """
        Compute the signal returns.

        Parameters
        ----------
        signal_type: str, {'signal', 'disc_signal', 'signal_quantiles', 'sign', 'signal_rank'}, default 'signal'
            signal: factor inputs are converted to signals between -1 and 1 for l/s strategies, between 0 and 1 for long
            only strategies, and between -1 and 0 for short only strategies.
            disc_signal: factor inputs are converted to discrete signals -1, 0 and 1 for l/s strategies,
            0 or 1 for long only strategies, and -1 or 0 for short only strategies.
            signal_quantiles: factor inputs are converted to quantized signals between -1 and 1 for l/s strategies,
             between 0 and 1 for long-only strategies, and between -1 and 0 for short only strategies, with n bins.
            signal_rank: factor inputs are converted to signal ranks between -1 and 1, 0 or 1 for long only
            strategies, and -1 or 0 for short only strategies, with n factors.
        transformation: str, {'norm', 'percentile', 'min-max', 'logistic', 'adj_norm', 'sign'}, default 'norm'
            Probability density function to convert raw factor values to signals between 1 and -1.
        ts_norm: bool, default False
            Normalizes factors over the time series before quantization over the cross-section.
        winsorize: int, default 3
            Max/min value to use for winsorization/clipping for signals when method is z-score, iqr or mod z.
        leverage: int, default None
            Multiplies factors by integer to increase leverage.
        lags: int, default 1
            Number of periods to lag signals/forward returns.
        dual_summary_stat: str, {'mean', 'median', 'min', 'max', 'sum', 'prod'}, default 'mean'
            Summary statistic to compute dual signals.

        Returns
        -------
        signal_rets: pd.DataFrame
            Dataframe with DatetimeIndex (level 0), tickers (level 1) and computed signal returns (cols).
        """
        # compute signals
        if self.strategy.split('_')[0] == 'dual':
            self.compute_dual_signals(signal_type=signal_type, transformation=transformation, ts_norm=ts_norm,
                                      winsorize=winsorize, leverage=leverage, lags=lags,
                                      summary_stat=dual_summary_stat)
        else:
            self.compute_signals(signal_type=signal_type, transformation=transformation, ts_norm=ts_norm,
                                 winsorize=winsorize, leverage=leverage, lags=lags)

        # concat signals and returns
        df = pd.concat([self.signals, self.returns], axis=1, join='inner')
        # multiply signals by returns
        self.signal_rets = df.iloc[:, :-1].mul(df.iloc[:, -1].values, axis=0).dropna(how='all')

        return self.signal_rets

    def signal_dispersion(self, method: str = 'sign'):
        """
        Computes dispersion in the signal cross-section.

        Parameters
        ----------
        method: str, {'sign', 'stdev', 'skew', 'range'}
            Method used to compute factor dispersion.

        Returns
        -------
        disp: Series
            Series with DatetimeIndex and dispersion measure.
        """
        if method == 'sign':
            if isinstance(self.signals.index, pd.MultiIndex):
                pos, neg = np.sign(self.signals[self.signals > 0]).groupby(level=0).sum(), \
                    np.sign(self.signals[self.signals < 0]).groupby(level=0).sum().abs()
            else:
                pos, neg = np.sign(self.signals[self.signals > 0]).sum(axis=1), \
                    np.sign(self.signals[self.signals < 0]).sum(axis=1).abs()

            self.signal_disp = (pos - neg) / (pos + neg)

        elif method == 'std':
            if isinstance(self.signals.index, pd.MultiIndex):
                self.signal_disp = self.signals.groupby(level=0).std()
            else:
                self.signal_disp = self.signals.std(axis=1)

        elif method == 'skew':
            if isinstance(self.signals.index, pd.MultiIndex):
                self.signal_disp = self.signals.groupby(level=0).skew()
            else:
                self.signal_disp = self.signals.skew(axis=1)

        elif method == 'range':
            if isinstance(self.signals.index, pd.MultiIndex):
                self.signal_disp = self.signals.groupby(level=0).max() - self.signals.groupby(level=0).min()
            else:
                self.signal_disp = self.signals.max(axis=1) - self.signals.min(axis=1)

        return self.signal_disp

    def signal_autocorrelation(self, lags: int = 1):
        """
        Computes the autocorrelation of the signal cross-section.

        Parameters
        ----------
        lags: int, default 1
            Number of periods to lag signals/forward returns.

        Returns
        -------
        autocorr: Series
            Series with DatetimeIndex and autocorrelation measure.
        """
        if isinstance(self.signals.index, pd.MultiIndex):
            self.signal_disp = self.signals.groupby(level=0).apply(lambda x: x.corrwith(x.shift(lags)))
        else:
            self.signal_disp = self.signals.apply(lambda x: x.corrwith(x.shift(lags)))

        return self.signal_disp
