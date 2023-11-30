import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from importlib import resources
from typing import Optional, Union
from scipy.stats import chi2_contingency, spearmanr, kendalltau, contingency
from sklearn.feature_selection import mutual_info_classif

from factorlab.feature_analysis.time_series_analysis import linear_reg, fm_summary
from factorlab.feature_engineering.transform import Transform
from factorlab.feature_analysis.performance import Performance


class Factor:
    """
    Analysis of alpha or risk factor.
    """
    def __init__(self,
                 factors: pd.DataFrame,
                 ret: pd.Series,
                 strategy: str = 'ts_ls',
                 factor_bins: int = 5,
                 target_bins: int = 2,
                 window_type: Optional[str] = 'expanding',
                 window_size: Optional[int] = 90,
                 ):
        """
        Constructor

        Parameters
        ----------
        factors: pd.Series or pd.DataFrame - Single or MultiIndex
            Dataframe with DatetimeIndex (level 0), tickers (level 1) and factors (cols).
        ret: pd.Series or pd.DataFrame - Single or MultiIndex
            Dataframe or series with DatetimeIndex (level 0), tickers (level 1) and returns (cols).
        strategy: str, {'ts_ls' 'ts_l', 'cs_ls', 'cs_l', default 'ts_ls'
            Time series or cross-sectional strategy, long/short or long-only.
        factor_bins: int, default 5
            Number of bins to create for factors.
        target_bins: int, default 2
            Number of bins to create for forward returns.
        window_type: str, {'fixed', 'expanding', 'rolling'}, default 'expanding'
            Window type for normalization.
        window_size: int
            Minimal number of observations to include in moving window (rolling or expanding).
        """
        self.factors = factors.astype(float)
        self.ret = ret.astype(float)
        self.strategy = strategy
        self.factor_bins = factor_bins
        self.target_bins = target_bins
        self.window_type = window_type
        self.window_size = window_size
        if isinstance(self.factors, pd.Series):
            self.factors = self.factors.to_frame()
        if factor_bins <= 1 or target_bins <= 1:
            raise ValueError("Number of bins must be larger than 1.")

    def normalize(self,
                  centering: bool = True,
                  method: str = 'z-score',
                  winsorize: Optional[int] = None
                  ) -> Union[pd.Series, pd.DataFrame]:
        """
        Normalizes factors and/or targets.

        Parameters
        ---------
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
        winsorize: int, default None
            Winsorizes/clips values to between positive and negative values of specified integer.

        Returns
        -------
        norm: pd.Series or pd.DataFrame - MultiIndex
            Dataframe with DatetimeIndex (level 0), tickers (level 1) and normalized values (cols).
        """
        # normalize features and fwd ret
        norm = Transform(self.factors).normalize_ts(centering=centering, method=method, window_type=self.window_type,
                                                    lookback=self.window_size, winsorize=winsorize)
        if self.strategy[:2] == 'cs':
            norm = Transform(norm).normalize_cs(centering=centering, method=method)

        # clip negative values for long-only
        if self.strategy.split('_')[1] == 'l':
            norm = norm.clip(0)

        return norm

    def quantize(self,
                 df: Union[pd.Series, pd.DataFrame],
                 bins: int = 5,
                 method: str = 'cdf',
                 ts_norm: bool = False
                 ) -> Union[pd.Series, pd.DataFrame]:
        """
        Quantizes factors and/or targets.

        Parameters
        ---------
        df: pd.Series or pd.DataFrame - Single or MultiIndex
            Dataframe with DatetimeIndex (level 0), tickers (level 1) and values to quantize.
        bins: int, default 5
            Number of quantiles or bins.
        method: str, {'z-score', 'cdf', iqr', 'mod_z', 'min-max', 'percentile'}, default 'z-score'
            Normalization method to use.
        ts_norm: bool, default False
            Normalizes factors over the time series before quantization over the cross section.

        Returns
        -------
        quantiles: pd.Series or pd.DataFrame - MultiIndex
            Dataframe with DatetimeIndex (level 0), tickers (level 1) and quantiles (cols).
        """
        # convert to df
        if isinstance(df, pd.Series):
            df = df.to_frame()

        # quantize features and fwd ret
        if self.strategy[:2] == 'ts':
            quantiles = Transform(df).quantize_ts(bins=bins, window_type=self.window_type, lookback=self.window_size)
        else:
            if ts_norm:
                norm_factors = Transform(df).normalize_ts(method=method, window_type=self.window_type,
                                                          lookback=self.window_size)
                quantiles = Transform(norm_factors).quantize_cs(bins=bins)
            else:
                quantiles = Transform(df).quantize_cs(bins=bins)

        return quantiles

    def signal(self,
               centering: bool = True,
               method: str = 'cdf',
               ts_norm: bool = False,
               clip: int = 3,
               ) -> Union[pd.Series, pd.DataFrame]:
        """
        Converts raw factors to signals.

        Z-score, iqr and mod Z methods are converted to signals [-1, 1].

        Parameters
        ----------
        centering: bool, default True
            Centers values using the appropriate measure of central tendency used for the selected method. Otherwise,
            0 is used.
        method: str, {'min-max', 'percentile', 'cdf'}, default 'percentile'
            Method to convert raw factor values to signals between [-1,1]
        ts_norm: bool, default False
            Normalizes factors over the time series before quantization over the cross section.
        clip: int, default 3
            Winsorizes/clips values to between [clip *-1, clip].

        Returns
        -------
        signals: pd.Series or pd.DataFrame
            Signals with DatetimeIndex (level 0), tickers (level 1) and signals (cols).
        """
        # normalize
        if self.strategy[:2] == 'ts':  # time series
            signals = Transform(self.factors).normalize_ts(centering=centering, window_type=self.window_type,
                                                           method=method, lookback=self.window_size, winsorize=clip)
        else:  # cross sectional
            if ts_norm:
                norm_factors = Transform(self.factors).normalize_ts(centering=centering, method=method,
                                                                    window_type=self.window_type,
                                                                    lookback=self.window_size)
                signals = Transform(norm_factors).normalize_cs(centering=centering, method=method)
            else:
                signals = Transform(self.factors).normalize_cs(centering=centering, method=method)

        # winsorize z-score, mod z and iqr norm methods
        if method in ['z-score', 'iqr', 'mod_z']:
            signals = signals / clip
        else:  # cdf, min-max and percentile norm methods
            signals = (signals * 2) - 1
        # winsorize signals
        signals = signals.clip(-1, 1)

        # long only
        if self.strategy.split('_')[1] == 'l':
            signals += 1

        return signals

    def signal_quantiles(self,
                         centering: bool = True,
                         method: str = 'cdf',
                         clip: int = 3,
                         ts_norm: bool = False,
                         ) -> Union[pd.Series, pd.DataFrame]:
        """
        Converts factors to signal quantiles.

        Parameters
        ----------
        centering: bool, default True
            Centers values using the appropriate measure of central tendency used for the selected method. Otherwise,
            0 is used.
        method: str, {'min-max', 'percentile', 'cdf'}, default 'percentile'
            Method to convert raw factor values to signals between [-1,1]
        clip: int, default 3
            Winsorizes/clips values to between [clip *-1, clip].
        ts_norm: bool, default False
            Normalizes factors over the time series before quantization over the cross section.

        Returns
        -------
        signal_quantiles: pd.Series or pd.DataFrame - Single or MultiIndex
            Signal quantiles with DatetimeIndex (level 0), tickers (level 1) and signal quantile values (cols).
        """
        # signals
        if self.strategy[:2] == 'ts':
            signals = self.signal(centering=centering, method=method, clip=clip)
            quantiles = self.quantize(signals, bins=self.factor_bins)
        else:
            if ts_norm:
                norm_factors = Transform(self.factors).normalize_ts(centering=centering, method=method,
                                                                    window_type=self.window_type,
                                                                    lookback=self.window_size)
                quantiles = self.quantize(norm_factors, bins=self.factor_bins)
            else:
                quantiles = self.quantize(self.factors, bins=self.factor_bins)

        # time series
        if self.strategy[:2] == 'ts':
            signal_quantiles = (quantiles - np.median(range(1, self.factor_bins + 1))) / \
                      (np.median(range(1, self.factor_bins + 1)) - 1)
        else:  # cross sectional
            factor_quant_centered = quantiles - quantiles.groupby(level=0).median()
            signal_quantiles = factor_quant_centered / factor_quant_centered.groupby(level=0).max()

        # clip negative values for long-only
        if self.strategy.split('_')[1] == 'l':
            signal_quantiles += 1

        return signal_quantiles

    def filter(
            self,
            ts_norm: bool = False,
            metrics: Union[str, list] = ['spearman_r'],
            rank_on: Optional[str] = 'spearman_r'
    ) -> pd.DataFrame:
        """
        Computes measures of dependence for factor/target pairs.

        The information coefficient (IC) is often used to assess the predictive power of a factor.
        It measures the degree of correlation between factor quantiles and forward returns. The higher (lower) the IC,
        the stronger the relationship between higher (lower) factor values and higher (lower) returns.
        The most common way to compute the IC is with the Spearman Rank correlation.
        Unlike measures of linear correlation (e.g. Pearson), it captures the monotonicity and non-linearity of the
        relationship between factor quantiles and forward returns.

        We also compare the Spearman Rank correlation to other statistical measures of association between
        discretized/categorical variables, e.g. Cramer's V, Chi-square, etc. Correlation measures in what way
        two variables are related, whereas, association measures how related the variables are.

        Parameters
        ts_norm: bool, default False
            Normalizes factors over the time series before quantization over the cross section.
        metrics: str, {'all', 'spearman_r'}, default 'spearman_r'
            Metric to compute.
        rank_on: str, optional, default 'spearman_r'
            Ranks the factors in descending order by selected metric.

        Returns
        -------
        metrics_df: pd.DataFrame
            Dataframe with factors (rows) and stats (cols), ranked by metric.
        """
        # quantize factors and fwd ret
        factor_quantiles = self.quantize(self.factors, bins=self.factor_bins, ts_norm=ts_norm)
        target_quantiles = self.quantize(self.ret, bins=self.target_bins, ts_norm=ts_norm)
        # merge
        quantiles_df = pd.concat([factor_quantiles, target_quantiles], axis=1, join='inner')

        # create empty df for correlation measures
        metrics_df = pd.DataFrame(index=self.factors.columns)
        # metrics list
        metrics_list = ['spearman_r', 'p-val', 'cramer_v', 'chi2', 'mutual_info', 'kendall_tau', 'tschuprow',
                        'pearson_cc', 'autocorrelation']

        # metrics
        if metrics == 'all':
            metrics = metrics_list
        elif metrics is None:
            metrics = ['spearman_r']
        else:
            if isinstance(metrics, str):
                metrics = [metrics]
            for metric in metrics:
                if metric not in metrics_list:
                    raise ValueError(f"{metric} is not available. Select available metrics from {metrics_list}.")

        # loop through factors
        for col in self.factors.columns:
            data = pd.concat([quantiles_df[col], quantiles_df[self.ret.name]], axis=1, join='inner').dropna()
            # add metrics
            if 'spearman_r' in metrics:
                metrics_df.loc[col, 'spearman_r'] = spearmanr(data[col], data[self.ret.name])[0]
            if 'p-val' in metrics:
                metrics_df.loc[col, 'p-val'] = spearmanr(data[col], data[self.ret.name])[1]
            if 'kendall_tau' in metrics:
                metrics_df.loc[col, 'kendall_tau'] = kendalltau(data[col], data[self.ret.name])[0]
            if 'cramer_v' in metrics:
                # contingency table
                cont_table = pd.crosstab(data[col], data[self.ret.name])
                metrics_df.loc[col, 'cramer_v'] = contingency.association(cont_table, method='cramer')
            if 'tschuprow_t' in metrics:
                # contingency table
                cont_table = pd.crosstab(data[col], data[self.ret.name])
                metrics_df.loc[col, 'tschuprow_t'] = contingency.association(cont_table, method='tschuprow')
            if 'pearson_cc' in metrics:
                # contingency table
                cont_table = pd.crosstab(data[col], data[self.ret.name])
                metrics_df.loc[col, 'pearson_cc'] = contingency.association(cont_table, method='pearson')
            if 'chi2' in metrics:
                # contingency table
                cont_table = pd.crosstab(data[col], data[self.ret.name])
                metrics_df.loc[col, 'chi2'] = chi2_contingency(cont_table)[0]
            if 'mutual_info' in metrics:
                metrics_df.loc[col, 'mutual_info'] = mutual_info_classif(data[[col]], data[self.ret.name])
            if 'autocorrelation' in metrics:
                idx = self.factors.groupby(level=1, group_keys=False).shift(1).index
                metrics_df.loc[col, 'autocorrelation'] = spearmanr(self.factors[col].reindex(idx),
                                                                   self.factors.groupby(level=1, group_keys=False).
                                                                   shift(1)[col], nan_policy='omit')[0]

        # sort by IC and round values to 2 decimals
        if rank_on is not None:
            metrics_df = metrics_df.sort_values(by=rank_on, ascending=False).round(decimals=2)

        return metrics_df

    def ic(self,
           factor: str,
           ):
        """
        Computes the Information Coefficient (IC) for factor and forward returns over time.

        Parameters
        ----------
        factor: str
            Name of factor (column) for which compute IC.

        Returns
        -------
        ic : pd.DataFrame
            Information coefficient between factor and forward returns over time.
        """
        # create df
        df = pd.concat([self.factors[factor].groupby('ticker').shift(1), self.ret.to_frame('ret')], join='inner',
                       axis=1)
        df = df.unstack().dropna(thresh=10).stack()  # set time series min nobs thresh

        # compute spearman rank corr
        def spearman_r(data):
            stat = data.apply(lambda x: spearmanr(data.iloc[:, 0], data.iloc[:, 1], nan_policy='omit')[0])
            return stat

        # cs strategy
        if self.strategy[:2] == 'cs':
            ic_df = df.groupby('date').apply(spearman_r).rolling(self.window_size).mean().iloc[:, :-1]
        # ts strategy
        else:
            # create dates list to iterate over
            dates = df.index.droplevel(1).unique().to_list()
            # empty np array
            corr_arr = np.full((len(dates), 2), np.nan)

            # loop through rows of df
            for i in range(len(dates) - self.window_size + 1):
                # get corr
                corr = spearmanr(df.loc[dates[i]: dates[i + self.window_size - 1], df.iloc[:, 0].name],
                                 df.loc[dates[i]: dates[i + self.window_size - 1], df.iloc[:, 1].name],
                                 nan_policy='omit')
                # add corr arr
                corr_arr[i + self.window_size - 1, 0] = corr[0]

            # create ic df
            ic_df = pd.DataFrame(corr_arr, index=dates).iloc[:, 0].to_frame(self.factors[factor].name)
            ic_df.index.name = 'date'

        return ic_df

    def plot_ic(self,
                factor: str,
                source: str = None,
                colors: Optional[str] = None,
                ):
        """
        Plots information coefficient (Spearman Rank correlation) over a specified lookback window.

        Parameters
        ----------
        factor: str
            Name of factor (column) for which to compute IC.
        source: str, default None
            Adds source info to bottom of plot.
        colors: str, {'colors_dark', 'colors_mid', 'colors_light'}, default None
            Color scheme to use.
        """
        df = self.ic(factor=factor).dropna(how='all')

        # line plot in Systamental style
        # plot size
        fig, ax = plt.subplots(figsize=(15, 7))

        # line colors
        if colors == 'colors_dark':
            colors = ['#00588D', '#A81829', '#005F73', '#005F52', '#714C00', '#4C5900', '#78405F', '#674E1F', '#3F5661']
        elif colors == 'colors_light':
            colors = ['#5DA4DF', '#FF6B6C', '#25ADC2', '#4DAD9E', '#C89608', '#9DA521', '#C98CAC', '#B99966', '#89A2AE']
        else:
            colors = ['#006BA2', '#DB444B', '#3EBCD2', '#379A8B', '#EBB434', '#B4BA39', '#9A607F', '#D1B07C', '#758D99']

        # plot
        df.plot(color=colors, alpha=0.8, ax=ax)

        # legend
        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

        # grid
        ax.set_axisbelow(True)
        ax.grid(which="major", axis='y', color='#758D99', alpha=0.6)
        ax.set_facecolor("whitesmoke")

        # remove splines
        ax.spines[['top', 'right', 'left']].set_visible(False)

        # format x-axis
        ax.set_xlim(df.index.get_level_values('date')[0], df.index.get_level_values('date')[-1])

        # Reformat y-axis tick labels
        ax.set_ylabel('IC')
        ax.yaxis.tick_right()

        # add systamental logo
        with resources.path("factorlab", "systamental_logo.png") as f:
            img_path = f
        img = Image.open(img_path)
        plt.figimage(img, origin='upper')

        # Add in title and subtitle
        plt.rcParams['font.family'] = 'georgia'
        ax.text(x=0.13, y=.92, s="Information Coefficient", transform=fig.transFigure, ha='left', fontsize=14,
                weight='bold', alpha=.8, fontdict=None)
        ax.text(x=0.13, y=.89, s="IC, 365 Period Rolling Window", transform=fig.transFigure, ha='left', fontsize=12,
                alpha=.8, fontdict=None)

        # Set source text
        if source is not None:
            ax.text(x=0.13, y=0.05, s=f"""Source: {source}""", transform=fig.transFigure, ha='left', fontsize=10,
                    alpha=.8, fontdict=None)

    def regression(self,
                   method: str = 'pooled',
                   multivariate: bool = False,
                   norm: Optional[bool] = False,
                   min_obs: int = 10
                   ):
        """
        Regression of forward returns on factors.

        Parameters
        ----------
        method: str, {'pooled', 'fama-macbeth'}, default 'pooled'
            Regression method.
        multivariate: bool, default False
            Runs a multivariate regression on all factors. Otherwise, runs univariate regressions on each factor.
        norm: bool, optional, default False
            Normalizes factors using z-score method before regressing on forward returns.
        min_obs: int, default 10
            Minimum number of observations in the cross-section for Fama Macbeth regression.

        Returns
        -------
        out: pd.DataFrame
            Dataframe/table containing regression summary results.
        """
        # factors
        factors = self.factors

        # convert to df
        if isinstance(factors, pd.Series):
            factors = factors.to_frame()

        # normalization
        if norm:
            factors = Transform(self.factors).normalize_ts(centering=True, method='z-score',
                                                           window_type=self.window_type, lookback=self.window_size)

        # pooled reg
        if method == 'pooled':
            if not multivariate:
                # compute beta, t-stat and rsq for multiple factors
                stats_df = pd.DataFrame(index=factors.columns, columns=['coef', 'pval', 'rsq'])
                for col in factors.columns:
                    for stat in ['coef', 'pval', 'rsq']:
                        res = linear_reg(self.ret, factors[col], window_type='fixed', output=stat, cov_type='HAC',
                                         cov_kwds={'maxlags': 1})
                        if stat != 'rsq':
                            stats_df.loc[col, stat] = res[0]
                        else:
                            stats_df.loc[col, stat] = res

                table = stats_df.sort_values(by='coef', ascending=False).rename(columns={'coef': 'beta'})

            else:
                table = linear_reg(self.ret, factors, window_type='fixed', output='summary', cov_type='HAC',
                                   cov_kwds={'maxlags': 1})

        else:  # Fama Macbeth
            if not multivariate:
                # compute beta, std error and t-stat
                table = pd.DataFrame()
                for col in factors.columns:
                    res = fm_summary(self.ret, factors[col], min_obs=min_obs)
                    table = pd.concat([table, res.loc[col].to_frame().T])
                table = table.sort_values(by='beta', ascending=False)
            else:
                table = fm_summary(self.ret, factors,  min_obs=min_obs)

        return table.round(decimals=4)

    def compute_factors(self, signal_type: Optional[str] = None,
                        centering: bool = True,
                        norm_method: str = 'cdf',
                        ts_norm: bool = False,
                        clip: int = 3,
                        leverage: Optional[int] = None,
                        tails: Optional[str] = None,
                        rebalancing: Optional[Union[str, int]] = None,
                        ) -> Union[pd.Series, pd.DataFrame]:
        """
        Computes alpha factors by converting raw data to normalized factors or signals.

        Parameters
        ----------
        signal_type: str, {'norm', 'signal', 'signal_quantiles'}, default 'norm'
            norm: factor inputs are normalized into z-scores.
            signal: factor inputs are converted to signals between -1 and 1, and 0 and 2 for l/s and
            long-only strategies, respectively.
            signal_quantiles: factor inputs are converted to quantized signals between -1 and 1, or 0 and 2,
            with n bins.
        centering: bool, default True
            Centers values using the appropriate measure of central tendency used for the selected method. Otherwise,
            0 is used.
        norm_method: str, {'z-score', 'cdf', iqr', 'mod_z', 'min-max', 'percentile'}, default 'z-score'
                z-score: subtracts mean and divides by standard deviation.
                cdf: cumulative distribution function rescales z-scores to values between 0 and 1.
                iqr:  subtracts median and divides by interquartile range.
                mod_z: modified z-score using median absolute deviation.
                min-max: rescales to values between 0 and 1 by subtracting the min and dividing by the range.
                percentile: converts values to their percentile rank relative to the observations in the
                defined window type.
        ts_norm: bool, default False
            Normalizes factors over the time series before quantization over the cross section.
        clip: int, default 3
            Max/min value to use for winsorization/clipping for signals when method is z-score, iqr or mod z.
        leverage: int, default None
            Multiplies factors by integer to increase leverage.
        tails: str, {'two', 'left', 'right'}, optional, default None
            Keeps only tail bins and ignores middle bins, 'two' for both tails, 'left' for left, 'right' for right
        rebalancing: str or int, default None
            Rebalancing frequency. Can be day of week, e.g. 'monday', 'tuesday', etc, start, middle or end of month,
            e.g. 'month_end', '15th', or 'month_start', or an int for the number of days between rebalancing.

        Returns
        -------
        factors: pd.DataFrame or pd.Series
            Series or dataframe with DatetimeIndex (level 0), tickers (level 1) and computed factors (cols).
        """
        # make a copy of factors
        factors = self.factors.copy()

        # raw factors, signal_type None
        if signal_type is None:
            if self.strategy.split('_')[1] == 'l':
                factors = factors.clip(0)

        # normalized factors, signal_type 'norm'
        if signal_type == 'norm':
            factors = self.normalize(centering=centering, method=norm_method)

        # signal factors, signal_type 'signal'
        if signal_type == 'signal':
            factors = self.signal(centering=centering, method=norm_method,  ts_norm=ts_norm, clip=clip)

        # quantized signal factors, signal_type 'signal_quantiles'
        if signal_type == 'signal_quantiles':
            factors = self.signal_quantiles(centering=centering, method=norm_method, ts_norm=ts_norm, clip=clip)

        # tails
        if tails is not None:
            if signal_type != 'signal_quantiles':
                raise ValueError("Tails parameter requires 'signal_quantiles' signal type")
            # convert to tails (optional)
            if tails == 'two':
                factors = factors[(factors == factors.min()) | (factors == factors.max())].fillna(0)
            elif tails == 'left':
                if self.strategy.split('_')[1] == 'l':
                    raise ValueError("Long-only strategies cannot be long the left tail."
                                     "The resulting signals will be 0.\n")
                else:
                    factors = factors[factors == factors.min()].fillna(0)
            elif tails == 'right':
                factors = factors[factors == factors.max()].fillna(0)

        # leverage
        if leverage is not None:
            factors *= leverage

        # rebalancing
        if rebalancing is not None:
            rebalancing_dict = {'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3, 'friday': 4, 'saturday': 5,
                                'sunday': 6, '15th': 15, 'month_end': 'is_month_end', 'month_start': 'is_month_start'}
            if rebalancing in list(rebalancing_dict.keys())[:7]:
                factors = factors.unstack()[factors.unstack().index.dayofweek == rebalancing_dict[rebalancing]]. \
                    reindex(factors.unstack().index).ffill().stack()
            elif rebalancing == '15th':
                factors = factors.unstack()[factors.unstack().index.day == 15]. \
                    reindex(factors.unstack().index).ffill().stack()
            elif isinstance(rebalancing, int):
                factors = factors.unstack().iloc[::rebalancing, :]. \
                    reindex(factors.unstack().index).ffill().stack()
            else:
                factors = factors.unstack()[getattr(factors.unstack().index, rebalancing_dict[rebalancing])]. \
                    reindex(factors.unstack().index).ffill().stack()

        return factors

    def compute_inv_vol_weights(self,
                                method: str = 'smw',
                                vol_target: float = 0.1,
                                ann_factor: int = 365,
                                ) -> pd.DataFrame:
        """
        Parameters
        ----------
        method: str, {'smw', 'ewm'}, default 'smw'
            Type of moving window for volatility computation.
        vol_target: float, default 0.1
            Target volatility for vol scaling factor.
        ann_factor: int
            Annualization factor, e.g. 365 for daily, 52 for weekly, 12 for monthly, etc.

        Returns
        -------
        w_df: pd.DataFrame
            Dataframe with DatetimeIndex (level 0), tickers (level 1) and inverse volatility weights (cols).
        """
        # std
        if method == 'ewm':
            std_df = self.ret.groupby(level=1).ewm(self.window_size, min_periods=self.window_size).\
                std().droplevel(0)
        else:
            std_df = self.ret.groupby(level=1).rolling(self.window_size).std().droplevel(0)

        # inv vol factor
        inv_vol_df = vol_target / (std_df * np.sqrt(ann_factor))

        return inv_vol_df.sort_index()

    @staticmethod
    def tcosts(factors: pd.DataFrame,
               t_cost: float = None):
        """
        Computes transactions costs from changes in factor values (signals).

        Parameters
        ----------
        factors: pd.DataFrame
            Dataframe with DatetimeIndex (level 0), tickers (level 1) and factor values (cols).
        t_cost: float, default 0.001 per transaction, 10 bps
            Per transaction cost.

        Returns
        -------
        tcost_df: pd.DataFrame or pd.Series
            Dataframe or series with DatetimeIndex (level 0), tickers (level 1) and transaction costs (cols).
        """
        if t_cost is None:
            t_cost = 0
        if isinstance(factors, pd.Series):
            factors = factors.to_frame()
        # compute t-costs
        tcost_df = abs(factors.groupby(level=1).diff().shift(1)).groupby(level=0).mean() * t_cost

        return tcost_df

    def returns(self,
                signal_type: Optional[str] = None,
                centering: bool = True,
                tails: Optional[str] = None,
                leverage: Optional[int] = None,
                norm_method: Optional[str] = 'cdf',
                clip: int = 3,
                ts_norm: bool = False,
                rebalancing: Optional[Union[str, int]] = None,
                t_cost: Optional[float] = None,
                weighting: str = 'ew',
                vol_target: float = 0.1,
                ann_factor: int = 365,
                ) -> Union[pd.Series, pd.DataFrame]:
        """
        Computes factors returns.

        Factor returns are defined as F,t-1 * ret, where F is the fxn factor matrix and ret is an nx1 vector of asset
        returns.

        Parameters
        ----------
        signal_type: str, {'norm', 'signal', 'signal_quantiles'}, default 'norm'
            norm: factor inputs are normalized into z-scores.
            signal: factor inputs are converted to signals between -1 and 1, and 0 and 2 for l/s and
            long-only strategies, respectively.
            signal_quantiles: factor inputs are converted to quantized signals between -1 and 1, or 0 and 2,
            with n bins.
        centering: bool, default True
            Centers values using the appropriate measure of central tendency used for the selected method. Otherwise,
            0 is used.
        leverage: int, default None
            Multiplies factors by integer to increase leverage
        norm_method: str, {'z-score', 'cdf', iqr', 'mod_z', 'min-max', 'percentile'}, default 'z-score'
                z-score: subtracts mean and divides by standard deviation.
                cdf: cumulative distribution function rescales z-scores to values between 0 and 1.
                iqr:  subtracts median and divides by interquartile range.
                mod_z: modified z-score using median absolute deviation.
                min-max: rescales to values between 0 and 1 by subtracting the min and dividing by the range.
                percentile: converts values to their percentile rank relative to the observations in the
                defined window type.
        clip: int, default 3
            Max/min value to use for winsorization/clipping for signals when method is z-score, iqr or mod z.
        ts_norm: bool, default False
            Normalizes factors over the time series before quantization over the cross section.
        rebalancing: str or int, default None
            Rebalancing frequency. Can be day of week, e.g. 'monday', 'tuesday', etc, start, middle or end of month,
            e.g. 'month_end', '15th', or 'month_start', or an int for the number of days between rebalancing.
        tails: str, {'two', 'left', 'right'}, optional, default None
            Keeps only tail bins and ignores middle bins, 'two' for both tails, 'left' for left, 'right' for right
        t_cost: float, optional, default None
            Per transaction cost.
        weighting: str, {'ew', 'iv'}, default 'ew'
            Weights used to compute portfolio returns.
        vol_target: float, default 0.10
            Target annualized volatility.
        ann_factor: int, {12, 52, 252, 365}, default 365
            Annualization factor.

        Returns
        -------
        ret: pd.DataFrame or pd.Series
            Series or dataframe with DatetimeIndex (level 0), tickers (level 1) and factor returns (cols).
        """
        # compute factors
        factors = self.compute_factors(signal_type=signal_type, centering=centering, norm_method=norm_method, clip=clip,
                                       ts_norm=ts_norm, leverage=leverage, tails=tails, rebalancing=rebalancing)

        # compute weights
        if weighting == 'iv':  # vol-adj weights
            inv_vol_df = self.compute_inv_vol_weights(vol_target=vol_target, ann_factor=ann_factor)  # inv vol weights
            # scale factors
            scaled_factors_df = pd.concat([factors, inv_vol_df], axis=1)
            factors = scaled_factors_df.iloc[:, :-1].mul(scaled_factors_df.iloc[:, -1], axis=0)

        # compute factor ret
        df = pd.concat([factors.groupby('ticker').shift(1), self.ret], axis=1)
        ret_df = df.iloc[:, :-1].mul(df.iloc[:, -1], axis=0)
        ret_df = ret_df.groupby(level=0).mean()

        # compute net ret
        tcost_df = self.tcosts(factors.groupby('ticker').shift(1), t_cost=t_cost)  # t-costs
        net_ret_df = ret_df.subtract(tcost_df, axis=0).dropna(how='all')
        # replace NaNs with 0s
        net_ret_df.iloc[0] = 0

        return net_ret_df

    def tcosts_be(self,
                  signal_type: Optional[str] = None,
                  centering: bool = True,
                  leverage: Optional[int] = None,
                  norm_method: Optional[str] = 'z-score',
                  clip: int = 3,
                  ts_norm: bool = False,
                  rebalancing: Optional[Union[str, int]] = None,
                  tails: Optional[str] = None,
                  weighting: str = 'ew',
                  vol_target: float = 0.1,
                  ann_factor: int = 365,
                  plot_tcosts: bool = False,
                  source: Optional[str] = None
                  ):
        """
        Computes breakeven transaction costs.

        Finds value for transaction costs that would erode factor returns down to 0.

        Parameters
        ----------
        signal_type: str, {'norm', 'signal', 'signal_quantiles'}, default 'norm'
            norm: factor inputs are normalized into z-scores.
            signal: factor inputs are converted to signals between -1 and 1, and 0 and 2 for l/s and
            long-only strategies, respectively.
            signal_quantiles: factor inputs are converted to quantized signals between -1 and 1, or 0 and 2,
            with n bins.
        centering: bool, default True
            Centers values using the appropriate measure of central tendency used for the selected method. Otherwise,
            0 is used.
        leverage: int, default None
            Multiplies factors by integer to increase leverage
        norm_method: str, {'z-score', 'cdf', iqr', 'mod_z', 'min-max', 'percentile'}, default 'z-score'
                z-score: subtracts mean and divides by standard deviation.
                cdf: cumulative distribution function rescales z-scores to values between 0 and 1.
                iqr:  subtracts median and divides by interquartile range.
                mod_z: modified z-score using median absolute deviation.
                min-max: rescales to values between 0 and 1 by subtracting the min and dividing by the range.
                percentile: converts values to their percentile rank relative to the observations in the
                defined window type.
        clip: int, default 3
            Max/min value to use for winsorization/clipping for signals when method is z-score, iqr or mod z.
        ts_norm: bool, default False
            Normalizes factors over the time series before quantization over the cross section.
        rebalancing: str or int, default None
            Rebalancing frequency. Can be day of week, e.g. 'monday', 'tuesday', etc, start, middle or end of month,
            e.g. 'month_end', '15th', or 'month_start', or an int for the number of days between rebalancing.
        tails: str, {'two', 'left', 'right'}, optional, default None
            Keeps only tail bins and ignores middle bins, 'two' for both tails, 'left' for left, 'right' for right
        weighting: str, {'ew', 'iv'}, default 'ew'
            Weights used to compute portfolio returns.
        vol_target: float, default 0.10
            Target annualized volatility.
        ann_factor: int, {12, 52, 252, 365}, default 365
            Annualization factor.
        plot_tcosts: bool, default False
            Plots breakeven transaction costs, sorted by values.
        source: str, default None
            Adds source info to bottom of plot.

        Returns
        -------
        be_tcosts: pd.Series
            Breakeven transaction costs for each factor.
        """
        # compute factors
        factors = self.compute_factors(signal_type=signal_type, centering=centering, leverage=leverage,
                                       norm_method=norm_method, clip=clip, ts_norm=ts_norm, rebalancing=rebalancing,
                                       tails=tails)
        if isinstance(factors, pd.Series):
            factors = factors.to_frame()

        # compute weights
        if weighting == 'iv':  # vol-adj weights
            inv_vol_df = self.compute_inv_vol_weights(vol_target=vol_target, ann_factor=ann_factor)  # inv vol weights
            # scale factors
            scaled_factors_df = pd.concat([factors, inv_vol_df], axis=1)
            factors = scaled_factors_df.iloc[:, :-1].mul(scaled_factors_df.iloc[:, -1], axis=0)

        # gross factor ret
        df = pd.concat([factors.groupby(level=1).shift(1), self.ret], axis=1, join='inner')
        ret_df = df.iloc[:, :-1].mul(df.iloc[:, -1], axis=0).groupby(level=0).mean()
        cum_ret = ret_df.dropna().cumsum().iloc[-1]

        # compute turnover
        turn = abs(factors.groupby(level=1).diff().shift(1)).groupby(level=0).mean().dropna().cumsum().iloc[-1]

        # breakeven transaction costs, bps
        be_tcosts = (cum_ret / turn) * 10000

        # plot
        if plot_tcosts:
            # bar plot in Systamental style
            # plot size
            fig, ax = plt.subplots(figsize=(15, 7))

            # line colors
            colors = ['#98DAFF', '#FFA39F', '#6FE4FB', '#86E5D4', '#FFCB4D', '#D7DB5A', '#FFC2E3', '#F2CF9A', '#BFD8E5']

            # plot
            be_tcosts.sort_values().plot(kind='barh', color=colors[7], ax=ax, rot=1)

            # grid
            ax.set_axisbelow(True)
            ax.grid(which="major", axis='x', color='#758D99', alpha=0.6, zorder=0)
            ax.set_facecolor("whitesmoke")
            ax.set_xlabel('Breakeven transaction cost, bps')

            # remove splines
            ax.spines[['top', 'right', 'left']].set_visible(False)

            # Reformat y-axis tick labels
            ax.yaxis.tick_right()

            # add systamental logo
            with resources.path("factorlab", "systamental_logo.png") as f:
                img_path = f
            img = Image.open(img_path)
            plt.figimage(img, origin='upper')

            # Add in title and subtitle
            plt.rcParams['font.family'] = 'georgia'
            ax.text(x=0.13, y=.92, s="Transaction Cost Analysis", transform=fig.transFigure, ha='left', fontsize=14,
                    weight='bold', alpha=.8, fontdict=None)
            ax.text(x=0.13, y=.89, s="Breakeven, bps", transform=fig.transFigure, ha='left', fontsize=12,
                    alpha=.8, fontdict=None)

            # Set source text
            if source is not None:
                ax.text(x=0.13, y=0.05, s=f"""Source: {source}""", transform=fig.transFigure, ha='left', fontsize=10,
                        alpha=.8, fontdict=None)

        return be_tcosts

    def quantiles(self,
                  factor: str,
                  metric: str = 'ret',
                  mkt_ret: pd.Series = None,
                  signal_type: str = 'signal',
                  norm_method: str = 'cdf',
                  clip: int = 3,
                  ts_norm: bool = False,
                  rebalancing: Optional[Union[str, int]] = None,
                  ann_factor: Optional[float] = 365,
                  **kwargs
                  ):
        """
        Computes quantile returns for each factor quantile.

        Parameters
        ----------
        factor: str,
            Name of col/factor to group by quantile.
        metric: str,
            Metric to compute for each quantile.
        mkt_ret: pd.Series, default None
            Series with DatetimeIndex and market returns (cols).
        signal_type: str, {'norm', 'signal', 'signal_quantiles'}, default 'norm'
            norm: factor inputs are normalized into z-scores.
            signal: factor inputs are converted to signals between -1 and 1, and 0 and 2 for l/s and
            long-only strategies, respectively.
            signal_quantiles: factor inputs are converted to quantized signals between -1 and 1, or 0 and 2,
            with n bins.
        norm_method: str, {'min-max', 'percentile', 'cdf'}, default 'percentile'
            Method to convert raw factor values to signals between [-1,1]
        clip: int, default 3
            Winsorizes/clips values to between [clip *-1, clip].
        ts_norm: bool, default False
            Normalizes factors over the time series before quantization over the cross section.
        rebalancing: str or int, default None
            Rebalancing frequency. Can be day of week, e.g. 'monday', 'tuesday', etc, start, middle or end of month,
            e.g. 'month_end', '15th', or 'month_start', or an int for the number of days between rebalancing.
        ann_factor: int
            Annualization factor, e.g. 365 for daily, 52 for weekly, 12 for monthly, etc.

        Returns
        -------
        df: pd.DataFrame
            Dataframe with performance metrics by quantile.
        """
        # metrics
        metrics_dict = {'ret': 'Return', 'cumulative_ret': 'Cumulative returns', 'ann_ret': 'Annual return',
                        'ann_vol': 'Annual volatility', 'skewness': 'Skewness', 'kurtosis': 'Kurtosis',
                        'drawdown': 'Drawdown', 'max_dd': 'Max Drawdown', 'value_at_risk': 'VaR', 'tail_ratio':
                        'Tail ratio', 'sharpe_ratio': 'Sharpe ratio', 'sortino_ratio': 'Sortino ratio',
                        'calmar_ratio': 'Calmar ratio', 'omega_ratio': 'Omega ratio'}

        # check metric
        if metric not in metrics_dict.keys():
            raise ValueError(f"Metric not available. Available metrics are: {list(metrics_dict.keys())}")

        # compute factors
        factors = self.compute_factors(signal_type=signal_type, norm_method=norm_method, ts_norm=ts_norm, clip=clip,
                                       rebalancing=rebalancing)
        # quantize signals
        quantiles = self.quantize(factors[[factor]], bins=self.factor_bins)
        # merge
        quant_ret_df = pd.concat([quantiles.groupby('ticker').shift(1).dropna().astype(int), self.ret],
                                 axis=1, join='inner')

        # quantile rets
        quant_ret_ts_df = pd.DataFrame()
        for quant in range(1, self.factor_bins + 1):
            quant_ret_ts_df = pd.concat([quant_ret_ts_df,
                                         quant_ret_df[quant_ret_df[factor] == quant].iloc[:, -1].groupby(
                                             'date').mean().to_frame(quant)], axis=1)
        quant_ret_ts_df['top_vs_bottom'] = quant_ret_ts_df[self.factor_bins] - quant_ret_ts_df[1]
        quant_ret_ts_df.iloc[0] = 0  # make first period ret 0
        # compute performance metric
        if metric == 'ret':
            df = quant_ret_ts_df
        else:
            df = getattr(Performance(quant_ret_ts_df, mkt_ret=mkt_ret, ret_type='log', ann_factor=ann_factor),
                         metric)(**kwargs)
        # convert to df
        if isinstance(df, pd.Series):
            df = df.to_frame(metric)

        return df

    def dispersion(self, method: str = 'sign'):
        """
        Computes dispersion in the cross-section of factor values.

        Parameters
        ----------
        method: str, {'sign', 'stdev', 'skew', 'range'}
            Method used to compute factor dispersion.

        Returns
        -------
        disp: Series
            Series with DatetimeIndex and dispersion measure.
        """
        disp = None
        if method == 'sign':
            pos, neg = np.sign(self.factors[self.factors > 0]).groupby(level=0).sum(), abs(
                np.sign(self.factors[self.factors < 0]).groupby(level=0).sum())
            disp = pos - neg
        elif method == 'stdev':
            disp = self.factors.groupby(level=0).std()
        elif method == 'skew':
            disp = self.factors.groupby(level=0).skew()
        elif method == 'range':
            disp = self.factors.groupby(level=0).max() - self.factors.groupby(level=0).min()

        return disp
