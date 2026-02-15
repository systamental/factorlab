import pandas as pd
import numpy as np
from typing import Optional, Union

from factorlab.feature_engineering.transformations import Transform


class Signal:
    """
    Signal construction class.
    """
    def __init__(self,
                 factors: pd.DataFrame,
                 returns: Optional[Union[pd.Series, pd.DataFrame]] = None,
                 strategy: str = 'time_series',
                 direction: str = 'long_short',
                 signal: str = 'continuous',
                 signal_thresh: float = 0.6,
                 bins: int = 5,
                 n_factors: Optional[int] = None,
                 normalize: bool = True,
                 transform: bool = False,
                 quantize: bool = False,
                 rank: bool = False,
                 combine: bool = False,
                 window_type: str = 'expanding',
                 window_size: int = 360,
                 ):
        """
        Constructor

        Parameters
        ----------
        factors: pd.DataFrame - Single or MultiIndex
            Dataframe with DatetimeIndex (level 0), tickers (level 1) and factors (cols).
        returns: optional, pd.Series or pd.DataFrame - Single or MultiIndex, default None
            Dataframe or series with DatetimeIndex (level 0), tickers (level 1) and returns (cols).
        strategy: str, {'time_series', 'cross_sectional', 'dual'}, default 'time_series'
            Time series (aka directional), cross-sectional (market-neutral) or dual strategy.
        direction: str, {'long_short', 'long', 'short'}, default 'long_short'
            Long/short, long or short strategy.
        signal: str, {'continuous', 'discrete'}, default 'continuous'
            Signal to compute.
        signal_thresh: float, default 0
            Threshold cutoff for converting continuous signal values to discrete signals.
        bins: int, default 5
            Number of bins to use for quantization.
        n_factors: int, optional, default None
            Number of factors to use for converting to ranking to discrete signals in cross-sectional strategies.
        normalize: bool, default True
            Normalize factors.
        transform: bool, default False
            Power transform factors.
        quantize: bool, default False
            Quantize factors.
        rank: bool, default False
            Rank factors.
        combine: bool, default False
            Combine factors using a summary statistic.
        window_type: str, {'fixed', 'expanding', 'rolling'}, default 'expanding'
            Window type for normalization.
        window_size: int, default 90
            Minimal number of observations to include in moving window (rolling or expanding).
        """
        self.factors = factors
        self.returns = returns
        self.strategy = strategy
        self.direction = direction
        self.signal = signal
        self.signal_thresh = signal_thresh
        self.bins = bins
        self.n_factors = n_factors
        self.normalize = normalize
        self.transform = transform
        self.quantize = quantize
        self.rank = rank
        self.combine = combine
        self.window_type = window_type
        self.window_size = window_size
        self.preprocess_data()
        self.signals = None
        self.signal_rets = None
        self.signal_disp = None
        self.signal_corr = None

    def check_params(self):
        """
        Check signal parameters.
        """
        if self.strategy not in ['time_series', 'cross_sectional', 'dual']:
            raise ValueError(f"Strategy type must be either time series, cross-sectional or dual.")
        if self.direction not in ['long_short', 'long', 'short']:
            raise ValueError(f"Direction must be either long_short, long_only or short_only.")
        if self.signal not in ['continuous', 'discrete']:
            raise ValueError(f"Signal type must be either continuous or discrete.")
        if self.signal_thresh > 1 or self.signal_thresh < 0:
            raise ValueError(f"Signal threshold must be between 0 and 1.")
        if self.bins < 2:
            raise ValueError(f"Number of bins must be larger than 1.")
        if self.window_type not in ['fixed', 'expanding', 'rolling']:
            raise ValueError(f"Window type must be either fixed, expanding or rolling.")
        if self.window_size < 2:
            raise ValueError(f"Window size must be larger than 1.")

    def preprocess_data(self) -> None:
        """
        Preprocess the data for signal generation.
        """
        # check signal parameters
        self.check_params()

        # factors
        if isinstance(self.factors, pd.Series):
            self.factors = self.factors.to_frame().astype('float64')
        elif isinstance(self.factors, pd.DataFrame):
            self.factors = self.factors.astype('float64')

        # returns
        if isinstance(self.returns, pd.Series):
            self.returns = self.returns.to_frame().astype('float64')
        elif isinstance(self.returns, pd.DataFrame):
            self.returns = self.returns.astype('float64')

    def normalize_factors(self,
                          method: str = 'z-score',
                          centering: bool = False,
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
        factors: pd.DataFrame
            Normalized factors with DatetimeIndex and normalized values (cols).
        """
        if self.normalize:

            # time series
            if self.strategy == 'time_series':
                self.factors = Transform(self.factors).normalize(method=method,
                                                                 axis='ts',
                                                                 centering=centering,
                                                                 window_type=self.window_type,
                                                                 window_size=self.window_size,
                                                                 winsorize=winsorize)

            # cross-sectional
            elif self.strategy == 'cross_sectional':
                if ts_norm:
                    factor_norm_ts = Transform(self.factors).normalize(method=method,
                                                                       axis='ts',
                                                                       centering=centering,
                                                                       window_type=self.window_type,
                                                                       window_size=self.window_size,
                                                                       winsorize=winsorize)

                    self.factors = Transform(factor_norm_ts).normalize(method=method,
                                                                       axis='cs',
                                                                       centering=centering,
                                                                       winsorize=winsorize)

                else:
                    self.factors = Transform(self.factors).normalize(method=method,
                                                                     axis='cs',
                                                                     centering=centering,
                                                                     winsorize=winsorize)

        return self.factors

    def transform_factors(self, method: str = 'yeo-johnson') -> pd.DataFrame:
        """
        Power transforms factors.

        Parameters
        ---------
        method: str, {'yeo-johnson', 'box-cox'}, default 'yeo-johnson'
            Transformation method to use for power transformation.

        Returns
        -------
        factors: pd.DataFrame
            Power transformed factors with DatetimeIndex and transformed values (cols).
        """
        if self.transform:

            # time series
            if self.strategy == 'time_series':
                self.factors = Transform(self.factors).power_transform(axis='ts',
                                                                       method=method,
                                                                       window_type=self.window_type,
                                                                       window_size=self.window_size)

            # cross-sectional
            elif self.strategy == 'cross_sectional':
                self.factors = Transform(self.factors).power_transform(axis='cs',
                                                                       method=method)

            return self.factors

    def quantize_factors(self) -> pd.DataFrame:
        """
        Quantize factors.

        Returns
        -------
        factors: pd.Series or pd.DataFrame
            Quantized factors with DatetimeIndex and quantized values (cols).
        """
        if self.quantize:

            # time series
            if self.strategy == 'time_series':
                self.factors = Transform(self.factors).quantize(bins=self.bins,
                                                                axis='ts',
                                                                window_type=self.window_type,
                                                                window_size=self.window_size)

            # cross-sectional
            elif self.strategy == 'cross_sectional':
                self.factors = Transform(self.factors).quantize(bins=self.bins,
                                                                axis='cs')

            return self.factors

    def rank_factors(self) -> pd.DataFrame:
        """
        Ranks factors in the cross-section.

        Returns
        -------
        signals: pd.Series or pd.DataFrame - Single or MultiIndex
            Signal ranks with DatetimeIndex (level 0), tickers (level 1) and signal rank values (cols).
        """
        if self.rank:

            # time series
            if self.strategy == 'time_series':
                self.factors = Transform(self.factors).rank(axis='ts',
                                                            percentile=True,
                                                            window_type=self.window_type,
                                                            window_size=self.window_size)

            # cross-sectional
            else:
                # min n cross-section cutoff
                if self.n_factors is not None:
                    self.factors = self.factors[(self.factors.groupby(level=0).count() >= self.n_factors * 2)].dropna()

                self.factors = Transform(self.factors).rank(axis='cs')

        return self.factors

    def combine_factors(self, method: str = 'mean') -> pd.DataFrame:
        """
        Combines factors using a summary statistic.

        Parameters
        ---------
        method: str, {'mean', 'median', 'min', 'max', 'sum', 'prod', 'value-weighted'}
            Summary statistic to compute combined factors.

        Returns
        -------
        combined_factor: pd.DataFrame
            Combined factor with DatetimeIndex and combined factor values (cols).
        """
        if self.combine:

            if method == 'mean':
                self.factors = self.factors.mean(axis=1)
            elif method == 'median':
                self.factors = self.factors.median(axis=1)
            elif method == 'min':
                self.factors = self.factors.min(axis=1)
            elif method == 'max':
                self.factors = self.factors.max(axis=1)
            elif method == 'sum':
                self.factors = self.factors.sum(axis=1)
            elif method == 'prod':
                self.factors = self.factors.prod(axis=1)
            elif method == 'value-weighted':
                self.factors = self.factors.div(self.factors.abs().sum(axis=1), axis=0).sum(axis=1)
            else:
                raise ValueError(f"Combine method {method} is not available.")

            self.factors = self.factors.to_frame('combined_factor')

            return self.factors

    def factors_to_signals(self, transformation: str = 'norm') -> pd.DataFrame:
        """
        Converts raw factors to continuous signals between 1 and -1.

        Parameters
        ----------
        transformation: str, {'norm', 'logistic', 'adj_norm', 'percentile', 'min-max', 'sign'}, default 'norm'
            Transformation to convert raw factor values to signals between 1 and -1.

        Returns
        -------
        signals: pd.Series or pd.DataFrame - Single or MultiIndex
            Continuous signals with DatetimeIndex (level 0), tickers (level 1) and signal values (cols).
        """
        # convert to signals
        if transformation in ['norm', 'logistic', 'adj_norm', 'min-max', 'percentile']:
            self.signals = Transform(self.factors).scores_to_signals(transformation=transformation)

        # sign
        elif transformation == 'sign':
            self.signals = np.sign(self.factors)

        else:
            self.signals = self.factors

        return self.signals

    def quantiles_to_signals(self) -> pd.DataFrame:
        """
        Converts factor quantiles to signals.

        Returns
        -------
        signals: pd.Series or pd.DataFrame - Single or MultiIndex
            Continuous signals with DatetimeIndex (level 0), tickers (level 1) and signal values (cols).
        """
        # time series
        if self.strategy == 'time_series':
            self.signals = Transform(self.factors).quantiles_to_signals(axis='ts',
                                                                        bins=self.bins)

        # cross-sectional
        elif self.strategy == 'cross_sectional':
            self.signals = Transform(self.factors).quantiles_to_signals(axis='cs',
                                                                        bins=self.bins)

        return self.signals

    def ranks_to_signals(self) -> pd.DataFrame:
        """
        Converts factor ranks to signals.

        Returns
        -------
        signals: pd.Series or pd.DataFrame - Single or MultiIndex
            Continuous signals with DatetimeIndex (level 0), tickers (level 1) and signal values (cols).
        """
        # time series
        if self.strategy == 'time_series':
            self.signals = Transform(self.factors).ranks_to_signals(axis='ts')

        # cross-sectional
        elif self.strategy == 'cross_sectional':
            self.signals = Transform(self.factors).ranks_to_signals(axis='cs')

        return self.signals

    def discretize_signals(self) -> pd.DataFrame:
        """
        Discretize continuous factor signals to discrete signals in [-1, 0, 1].

        Returns
        -------
        signals: pd.Series or pd.DataFrame - Single or MultiIndex
            Discrete signals [-1, 0, 1] with DatetimeIndex (level 0), tickers (level 1) and signal values (cols).
        """
        # cross-sectional rank n factors cutoff
        if self.n_factors is not None and self.strategy == 'cross_sectional':

            # upper/lower threshold values
            upper_thresh = self.factors.groupby(level=0).max() - self.n_factors
            lower_thresh = self.factors.groupby(level=0).min() + self.n_factors

            # sort top/bottom n
            bottom_n = self.factors.lt(lower_thresh)
            top_n = self.factors.gt(upper_thresh)

            # assign 1 to highest n values, -1 to lowest n values, and 0 to the rest
            self.signals = pd.DataFrame(data=0.0, index=self.factors.index, columns=self.factors.columns)
            self.signals[top_n] = 1.0
            self.signals[bottom_n] = -1.0

        # continuous signal threshold cutoff
        else:
            self.signals = self.signals.apply(lambda x: np.where(np.abs(x) >= self.signal_thresh, np.sign(x), 0))

        return self.signals

    def filter_direction(self):
        """
        Filters signals based on strategy direction.

        Returns
        -------
        signals: pd.DataFrame
            DataFrame with DatetimeIndex (level 0), tickers (level 1) and filtered signals (cols).
        """
        # long or short only
        if self.direction == 'long':
            self.signals = self.signals.clip(lower=0)
        elif self.direction == 'short':
            self.signals = self.signals.clip(upper=0)

        return self.signals

    def compute_signals(self,
                        signal_type: str = 'signal',
                        transformation: str = 'norm',
                        norm_method: str = 'z-score',
                        norm_transformation: str = 'yeo-johnson',
                        centering: bool = False,
                        ts_norm: bool = False,
                        winsorize: int = 3,
                        leverage: Optional[int] = None,
                        lags: Optional[int] = None
                        ) -> pd.DataFrame:
        """
        Computes signals by converting raw data to signals (alpha factors).

        Parameters
        ----------
        signal_type: str, {'signal', 'signal_quantiles', 'signal_ranks'}, default 'signal'
            signal: factor values are converted to signals between -1 and 1 for long-short strategies,
            between 0 and 1 for long-only strategies, and between -1 and 0 for short-only strategies.
            signal_quantiles: factor values are converted to quantized signals between -1 and 1 for long-short
            strategies, between 0 and 1 for long-only strategies, and between -1 and 0 for short-only strategies,
            with n bins.
            signal_ranks: factor values are converted to signal ranks between -1 and 1, 0 or 1 for long-only
            strategies, and -1 or 0 for short-only strategies, with n factors.
        transformation: str, {'norm', 'percentile', 'min-max', 'logistic', 'adj_norm', 'sign'}, default 'norm'
            Transformation to convert raw factor values to signals between 1 and -1.
        norm_method: str, {'z-score', 'iqr', 'mod_z', 'min-max', 'percentile'}, default 'z-score'
            Normalization method to use for raw factor values.
        norm_transformation: str, {'yeo-johnson', 'box-cox'}, default 'yeo-johnson'
            Transformation to use for normalized factor values.
        centering: bool, default False
            Centers values using the appropriate measure of central tendency used for the selected method. Otherwise,
            0 is used.
        ts_norm: bool, default False
            Normalizes factors over the time series before normalizing over the cross-section.
        winsorize: int, default 3
            Max/min value to use for winsorization/clipping for signals when method is z-score, iqr or mod z.
        leverage: int, default None
            Multiplies signals by leverage factor.
        lags: int, optional, default None
            Number of periods to lag signals.

        Returns
        -------
        signals: pd.DataFrame
            DataFrame with DatetimeIndex (level 0), tickers (level 1) and computed signals (cols).
        """
        if self.signals is None:

            # normalize
            self.normalize_factors(method=norm_method, centering=centering, ts_norm=ts_norm, winsorize=winsorize)
            # transform
            self.transform_factors(method=norm_transformation)
            # quantize
            self.quantize_factors()
            # rank
            self.rank_factors()
            # combine
            self.combine_factors()

            # raw factors, signal_type None
            if signal_type is None:
                self.signals = self.factors.copy()

            # signals, signal_type 'signal'
            elif signal_type == 'signal':
                if self.normalize is False:
                    raise ValueError("Normalization must be enabled to compute signals. Set normalize=True.")
                else:
                    self.factors_to_signals(transformation=transformation)

            # quantized signals, signal_type 'signal_quantiles'
            elif signal_type == 'signal_quantiles':
                if self.quantize is False:
                    raise ValueError("Quantization must be enabled to compute quantized signals. Set quantize=True.")
                else:
                    self.quantiles_to_signals()

            # ranked signals, signal_type 'signal_rank'
            elif signal_type == 'signal_ranks':
                if self.rank is False:
                    raise ValueError("Ranking must be enabled to compute ranked signals. Set rank=True.")
                else:
                    self.ranks_to_signals()

            # discrete signals
            if self.signal == 'discrete':
                self.discretize_signals()

            # filter signals for direction
            if self.direction in ['long', 'short']:
                self.filter_direction()

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
                             transformation: str = 'norm',
                             norm_method: str = 'z-score',
                             norm_transformation: str = 'yeo-johnson',
                             centering: bool = False,
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
        signal_type: str, {'signal', 'signal_quantiles', 'signal_ranks'}, default 'signal'
            signal: factor values are converted to signals between -1 and 1 for long-short strategies,
            between 0 and 1 for long-only strategies, and between -1 and 0 for short-only strategies.
            signal_quantiles: factor values are converted to quantized signals between -1 and 1 for long-short
            strategies, between 0 and 1 for long-only strategies, and between -1 and 0 for short-only strategies,
            with n bins.
            signal_ranks: factor values are converted to signal ranks between -1 and 1, 0 or 1 for long-only
            strategies, and -1 or 0 for short-only strategies, with n factors.
        transformation: str, {'norm', 'percentile', 'min-max', 'logistic', 'adj_norm', 'sign'}, default 'norm'
            Transformation to convert raw factor values to signals between 1 and -1.
        norm_method: str, {'z-score', 'iqr', 'mod_z', 'min-max', 'percentile'}, default 'z-score'
            Normalization method to use for raw factor values.
        norm_transformation: str, {'yeo-johnson', 'box-cox'}, default 'yeo-johnson'
            Transformation to use for normalized factor values.
        centering: bool, default False
            Centers values using the appropriate measure of central tendency used for the selected method. Otherwise,
            0 is used.
        ts_norm: bool, default False
            Normalizes factors over the time series before normalizing over the cross-section.
        winsorize: int, default 3
            Max/min value to use for winsorization/clipping for signals when method is z-score, iqr or mod z.
        leverage: int, default None
            Multiplies signals by leverage factor.
        lags: int, optional, default None
            Number of periods to lag signals.

        Returns
        -------
        signals: pd.DataFrame
            DataFrame with DatetimeIndex (level 0), tickers (level 1) and computed dual signals (cols).
        """
        # check strategy
        if self.strategy != 'dual':
            raise ValueError("Dual strategy must be selected to compute dual signals.")

        # factors
        self.strategy = 'time_series'
        ts_factors = self.compute_signals(signal_type=None, transformation=transformation, norm_method=norm_method,
                                          norm_transformation=norm_transformation, centering=centering, ts_norm=ts_norm,
                                          winsorize=winsorize, leverage=leverage, lags=lags)

        self.strategy = 'cross_sectional'
        cs_factors = self.compute_signals(signal_type=None, transformation=transformation, norm_method=norm_method,
                                          norm_transformation=norm_transformation, centering=centering, ts_norm=ts_norm,
                                          winsorize=winsorize, leverage=leverage, lags=lags)

        # compute dual factors
        self.factors, self.signals = None, None
        factors = pd.concat([ts_factors, cs_factors], axis=1, join='inner')
        for factor in ts_factors.columns:
            self.factors = pd.concat([self.factors,
                                      getattr(factors[factor], summary_stat)(axis=1).to_frame(factor)], axis=1)

        # compute dual signals
        self.strategy = 'dual'
        self.signals = self.compute_signals(signal_type=signal_type, transformation=transformation,
                                            norm_method=norm_method, norm_transformation=norm_transformation,
                                            centering=centering, ts_norm=ts_norm, winsorize=winsorize,
                                            leverage=leverage, lags=lags)

        return self.signals

    def rebalance_signals(self, rebal_freq: Optional[Union[str, int]] = None) -> pd.DataFrame:
        """
        Rebalance signals based on rebalancing frequency.

        Returns
        -------
        signals: pd.DataFrame
            Rebalanced portfolio weights with DatetimeIndex and weights (cols).
        """
        # frequency dictionary
        freq_dict = {'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3, 'friday': 4, 'saturday': 5,
                     'sunday': 6, '15th': 15, 'month_end': 'is_month_end', 'month_start': 'is_month_start'}

        # rebalancing
        if rebal_freq is not None:

            if isinstance(self.signals.index, pd.MultiIndex):
                signals = self.signals.unstack().copy()
            else:
                signals = self.signals.copy()

            # day of the week
            if rebal_freq in list(freq_dict.keys())[:7]:
                rebal_df = signals[signals.index.dayofweek == freq_dict[rebal_freq]]
            # mid-month
            elif rebal_freq == '15th':
                rebal_df = signals[signals.index.day == 15]
            # fixed period
            elif isinstance(rebal_freq, int):
                rebal_df = signals.iloc[::rebal_freq, :]
            # month start, month end
            else:
                rebal_df = signals[getattr(signals.index, freq_dict[rebal_freq])]

            # reindex and forward fill
            signals = rebal_df.reindex(signals.index).ffill().dropna(how='all')

            if isinstance(self.signals.index, pd.MultiIndex):
                signals = signals.stack(future_stack=True)
                self.signals = signals.reindex(index=self.signals.index)

            else:
                self.signals = signals

        return self.signals

    def compute_tcosts(self, t_cost: Optional[float] = None) -> pd.DataFrame:
        """
        Computes transactions costs from changes in weights.

        Returns
        -------
        t_costs: pd.Series
            Series with DatetimeIndex (level 0), tickers (level 1) and transaction costs (cols).
        """
        # no t-costs
        if t_cost is None:
            t_costs = pd.DataFrame(data=0.0, index=self.signals.index, columns=self.signals.columns)
        # t-costs
        elif isinstance(self.signals.index, pd.MultiIndex):
            t_costs = self.signals.groupby(level=1).diff().abs() * t_cost
        else:
            t_costs = self.signals.diff().abs() * t_cost

        # sort index
        t_costs = t_costs.sort_index()

        return t_costs

    def compute_gross_returns(self) -> pd.DataFrame:
        """
        Compute gross returns.

        Returns
        -------
        gross_returns: pd.DataFrame
            Gross returns.
        """
        # concat signals and returns
        df = pd.concat([self.signals, self.returns], axis=1)
        # multiply signals by returns
        self.signal_rets = df.iloc[:, :-1].mul(df.iloc[:, -1].values, axis=0).dropna(how='all')

        return self.signal_rets

    def compute_net_returns(self, t_costs: pd.DataFrame) -> pd.DataFrame:
        """
        Compute net returns.

        Returns
        -------
        net_returns: pd.DataFrame
            Net returns.
        """
        self.signal_rets = self.signal_rets.subtract(t_costs, axis=0).dropna(how='all')

        return self.signal_rets

    def compute_signal_returns(self,
                               summary_stat: str = 'mean',
                               signal_type: str = 'signal',
                               transformation: str = 'norm',
                               norm_method: str = 'z-score',
                               norm_transformation: str = 'yeo-johnson',
                               centering: bool = False,
                               ts_norm: bool = False,
                               winsorize: int = 3,
                               leverage: Optional[int] = None,
                               lags:  int = 1,
                               rebal_freq: Optional[Union[str, int]] = None,
                               t_cost: Optional[float] = None,
                               ) -> pd.DataFrame:
        """
        Compute the signal returns.

        Parameters
        ----------
        summary_stat: str, {'mean', 'median', 'min', 'max', 'sum', 'prod'}
            Summary statistic to compute dual signals.
        signal_type: str, {'signal', 'signal_quantiles', 'signal_ranks'}, default 'signal'
            signal: factor values are converted to signals between -1 and 1 for long-short strategies,
            between 0 and 1 for long-only strategies, and between -1 and 0 for short-only strategies.
            signal_quantiles: factor values are converted to quantized signals between -1 and 1 for long-short
            strategies, between 0 and 1 for long-only strategies, and between -1 and 0 for short-only strategies,
            with n bins.
            signal_ranks: factor values are converted to signal ranks between -1 and 1, 0 or 1 for long-only
            strategies, and -1 or 0 for short-only strategies, with n factors.
        transformation: str, {'norm', 'percentile', 'min-max', 'logistic', 'adj_norm', 'sign'}, default 'norm'
            Transformation to convert raw factor values to signals between 1 and -1.
        norm_method: str, {'z-score', 'iqr', 'mod_z', 'min-max', 'percentile'}, default 'z-score'
            Normalization method to use for raw factor values.
        norm_transformation: str, {'yeo-johnson', 'box-cox'}, default 'yeo-johnson'
            Transformation to use for normalized factor values.
        centering: bool, default False
            Centers values using the appropriate measure of central tendency used for the selected method. Otherwise,
            0 is used.
        ts_norm: bool, default False
            Normalizes factors over the time series before normalizing over the cross-section.
        winsorize: int, default 3
            Max/min value to use for winsorization/clipping for signals when method is z-score, iqr or mod z.
        leverage: int, default None
            Multiplies signals by leverage factor.
        lags: int, optional, default 1
            Number of periods to lag signals.
        rebal_freq: str, optional, default None
            Rebalancing frequency.
        t_cost: float, optional, default None
            Transaction costs.

        Returns
        -------
        signal_rets: pd.DataFrame
            Dataframe with DatetimeIndex (level 0), tickers (level 1) and computed signal returns (cols).
        """
        if self.returns is None:
            raise ValueError("Returns must be provided to compute signal returns.")

        # compute signals
        if self.signals is None:
            if self.strategy == 'dual':
                self.compute_dual_signals(summary_stat=summary_stat, signal_type=signal_type,
                                          transformation=transformation, norm_method=norm_method,
                                          norm_transformation=norm_transformation, centering=centering,
                                          ts_norm=ts_norm, winsorize=winsorize, leverage=leverage, lags=lags)
            else:
                self.compute_signals(signal_type=signal_type, transformation=transformation, norm_method=norm_method,
                                     norm_transformation=norm_transformation, centering=centering, ts_norm=ts_norm,
                                     winsorize=winsorize, leverage=leverage, lags=lags)

        # rebalance signals
        self.rebalance_signals(rebal_freq=rebal_freq)

        # compute transaction costs
        t_costs = self.compute_tcosts(t_cost=t_cost)

        # compute gross returns
        self.compute_gross_returns()

        # compute net returns
        self.compute_net_returns(t_costs=t_costs)

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

    def signal_correlation(self, method: str = 'spearman'):
        """
        Computes the correlation of the signal cross-section.

        Parameters
        ----------
        method: str, {'pearson', 'spearman', 'kendall'}
            Method used to compute factor correlation.

        Returns
        -------
        corr: DataFrame
            DataFrame with correlation matrix.
        """
        self.signal_corr = self.signals.corr(method=method)

        return self.signal_corr
