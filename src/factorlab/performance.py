import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from importlib import resources
from typing import Union, Optional, Dict, List
from factorlab.time_series_analysis import linear_reg


class Performance:
    """
    Performance metrics for asset or strategy returns.
    """
    def __init__(self,
                 returns: Union[pd.Series, pd.DataFrame],
                 mkt_ret: pd.Series = None,
                 ret_type: str = 'simple',
                 ann_factor: Optional[float] = 365,
                 remove_missing: bool = False
                 ):
        """
        Constructor

        Parameters
        ----------
        returns: pd.Series or pd.DataFrame
            Dataframe or series with DatetimeIndex and returns (cols).
        mkt_ret: pd.Series, default None
            Series with DatetimeIndex and market returns (cols).
        ann_factor: int
            Annualization factor, e.g. 365 for daily, 52 for weekly, 12 for monthly, etc.
        ret_type: str, {'simple', 'log'}, default 'simple'
            Return type.
        remove_missing: bool, default False
            Removes missing values so all returns have equal number of observations.
        """
        if remove_missing:
            self.returns = returns.dropna()
        if isinstance(returns, pd.Series):
            self.returns = returns.to_frame()
        else:
            self.returns = returns
        self.mkt_ret = mkt_ret
        self.ret_type = ret_type
        self.ann_factor = ann_factor

    def cumulative_ret(self, start_val: Optional[int] = 0) -> Union[pd.Series, pd.DataFrame]:
        """
        Computes cumulative returns for an asset or strategy.

        Parameters
        ----------
        start_val: int, optional, default 0
            Start value for cumulative returns.

        Returns
        -------
        cum_ret: pd.Series or pd.DataFrame
            Dataframe or series with DatetimeIndex and cumulative returns.
        """
        ret = self.returns

        if self.ret_type == 'simple':
            cum_ret = (1 + ret).cumprod().astype(float)
        else:
            cum_ret = np.exp(ret.cumsum().astype(float))

        # start val
        if start_val == 0:
            cum_ret = cum_ret - 1
        else:
            cum_ret = cum_ret * start_val

        return cum_ret.ffill()

    def ann_ret(self) -> float:
        """
        Computes the compound annual growth rate of returns, equivalent to CAGR.

        Returns
        -------
        dd: float
            Annualized return for each asset or strategy.
        """
        # cum return
        cum_ret = self.cumulative_ret(start_val=1).iloc[-1]
        # number of years of returns
        num_years = self.returns.count() / self.ann_factor
        # num_years = len(self.returns) / self.ann_factor
        # annualized returns
        ann_ret = cum_ret ** (1 / num_years) - 1

        return ann_ret

    def drawdown(self) -> Union[pd.Series, pd.DataFrame]:
        """
        Computes drawdowns for an asset or strategy.

        Returns
        -------
        dd: pd.Series or pd.DataFrame
            Dataframe or series with DatetimeIndex and drawdowns.

        """
        # cumulative return
        cum_ret = self.cumulative_ret(start_val=100)
        # drawdown
        dd = (cum_ret / cum_ret.expanding().max()) - 1

        return dd

    def max_dd(self) -> float:
        """
        Computes the maximum drawdown for an asset or strategy.

        Returns
        -------
        dd: float
            Maximum drawdown, returns as negative decimal (float).
        """
        dd = self.drawdown().min()

        return dd

    def ann_vol(self,
                window_type: str = 'fixed',
                window_size: int = 365) -> Union[float, pd.Series, pd.DataFrame]:
        """
        Computes the annualized volatility (standard deviation) of returns.

        Parameters
        ----------
        window_type: str, {'fixed', 'expanding', 'rolling'}, default 'fixed'
            Window type for calculation.
        window_size: int, default 365
            Minimum number of observations to include in moving window.

        Returns
        -------
        ann_vol: float, pd.Series or pd.DataFrame
            Annualized volatility for each asset or strategy.
        """
        if window_type == 'fixed':
            ann_vol = self.returns.std() * np.sqrt(self.ann_factor)
        elif window_type == 'expanding':
            ann_vol = getattr(self.returns, window_type)().std() * np.sqrt(self.ann_factor)
        else:
            ann_vol = getattr(self.returns, window_type)(window=window_size).std() * np.sqrt(self.ann_factor)

        return ann_vol

    def skewness(self,
                 window_type: str = 'fixed',
                 window_size: int = 365) -> Union[float, pd.Series, pd.DataFrame]:
        """
        Computes the skewness of returns (asymmetry).

        Parameters
        ----------
        window_type: str, {'fixed', 'expanding', 'rolling'}, default 'fixed'
            Window type for calculation.
        window_size: int, default 365
            Minimum number of observations to include in moving window.

        Returns
        -------
        skew: float, pd.Series or pd.DataFrame
            Skewnesss of returns.
        """
        if window_type == 'fixed':
            skew = self.returns.skew()
        elif window_type == 'expanding':
            skew = getattr(self.returns, window_type)().skew()
        else:
            skew = getattr(self.returns, window_type)(window=window_size).skew()

        return skew

    def kurtosis(self,
                 window_type: str = 'fixed',
                 window_size: int = 365) -> Union[float, pd.Series, pd.DataFrame]:
        """
        Computes the kurtosis of returns (tails).

        Parameters
        ----------
        window_type: str, {'fixed', 'expanding', 'rolling'}, default 'fixed'
            Window type for calculation.
        window_size: int, default 365
            Minimum number of observations to include in moving window.

        Returns
        -------
        kurt: float, pd.Series or pd.DataFrame
            Kurtosis of returns.
        """
        if window_type == 'fixed':
            kurt = self.returns.kurt()
        elif window_type == 'expanding':
            kurt = getattr(self.returns, window_type)().kurt()
        else:
            kurt = getattr(self.returns, window_type)(window=window_size).kurt()

        return kurt

    def value_at_risk(self,
                      perc: float = 0.05,
                      window_type: str = 'fixed',
                      window_size: int = 365) -> Union[float, pd.Series, pd.DataFrame]:
        """
        Value at risk (VaR) of an asset or strategy.

        Parameters
        ----------
        perc : float, optional, default 0.05
            Percentage threshold for the left tail of the distribution of returns.
        window_type: str, {'fixed', 'expanding', 'rolling'}, default 'fixed'
            Window type for calculation.
        window_size: int, default 365
            Minimum number of observations to include in moving window.

        Returns
        -------
        VaR : float, pd.Series or pd.DataFrame
            The VaR value.
        """
        if window_type == 'fixed':
            var = self.returns.quantile(perc)
        elif window_type == 'expanding':
            var = getattr(self.returns, window_type)().quantile(perc)
        else:
            var = getattr(self.returns, window_type)(window=window_size).quantile(perc)

        return var

    def tail_ratio(self,
                   perc: float = 0.05,
                   window_type: str = 'fixed',
                   window_size: int = 365) -> Union[float, pd.Series, pd.DataFrame]:
        """
        Computes the tail ratio of returns.

        Parameters
        ----------
        perc : float, optional, default 0.05
            Percentage threshold for the tails of the distribution of returns.
        window_type: str, {'fixed', 'expanding', 'rolling'}, default 'fixed'
            Window type for calculation.
        window_size: int, default 365
            Minimum number of observations to include in moving window.

        Measure of return asymmetry which allows us to compare large gains to large losses,
        e.g the ratio of the 99th percentile to 1st percentile.

        Returns
        -------
        tr: float, pd.Series or pd.DataFrame
            Tail ratio.
        """
        if window_type == 'fixed':
            tr = np.abs(self.returns.quantile(1 - perc)) / np.abs(self.returns.quantile(perc))
        elif window_type == 'expanding':
            tr = np.abs(getattr(self.returns, window_type)().quantile(1 - perc) /
                        getattr(self.returns, window_type)().quantile(perc))
        else:
            tr = np.abs(getattr(self.returns, window_type)(window=window_size).quantile(1 - perc) /
                        getattr(self.returns, window_type)(window=window_size).quantile(perc))

        return tr

    def sharpe_ratio(self,
                     window_type: str = 'fixed',
                     window_size: int = 365) -> Union[float, pd.Series, pd.DataFrame]:
        """
        Computes the Sharpe ratio of asset or strategy returns.

        Parameters
        ----------
        window_type: str, {'fixed', 'expanding', 'rolling'}, default 'fixed'
            Window type for calculation.
        window_size: int, default 365
            Minimum number of observations to include in moving window.

        Returns
        -------
        sr: float, pd.Series or pd.DataFrame
            Sharpe ratio for each asset or strategy.
        """
        if window_type == 'fixed':
            try:
                sr = (self.returns.mean() / self.returns.std()) * np.sqrt(self.ann_factor)
            except ZeroDivisionError:
                sr = np.nan
        elif window_type == 'expanding':
            try:
                sr = (getattr(self.returns, window_type)().mean() /
                      getattr(self.returns, window_type)().std()) * np.sqrt(self.ann_factor)
            except ZeroDivisionError:
                sr = np.nan
        else:
            try:
                sr = (getattr(self.returns, window_type)(window=window_size).mean() /
                      getattr(self.returns, window_type)(window=window_size).std()) \
                     * np.sqrt(self.ann_factor)
            except ZeroDivisionError:
                sr = np.nan

        return sr

    def sortino_ratio(self,
                      window_type: str = 'fixed',
                      window_size: int = 365) -> Union[float, pd.Series, pd.DataFrame]:
        """
        Computes the Sortino ratio of asset or strategy returns.

        Parameters
        ----------
        window_type: str, {'fixed', 'expanding', 'rolling'}, default 'fixed'
            Window type for calculation.
        window_size: int, default 365
            Minimum number of observations to include in moving window.

        Returns
        -------
        sr: float, pd.Series or pd.DataFrame
            Sortino ratio for each asset or strategy.
        """
        if window_type == 'fixed':
            sr = (self.returns.mean() / self.returns[self.returns < 0].std()) * np.sqrt(self.ann_factor)
        elif window_type == 'expanding':
            sr = (getattr(self.returns, window_type)().mean() /
                  getattr(self.returns[self.returns < 0], window_type)().std()) * np.sqrt(self.ann_factor)
        else:
            sr = (getattr(self.returns, window_type)(window=window_size).mean() /
                  getattr(self.returns[self.returns < 0], window_type)(window=window_size).std()) \
                  * np.sqrt(self.ann_factor)

        return sr

    def calmar_ratio(self) -> Union[float, pd.Series, pd.DataFrame]:
        """
        Computes the Calmar ratio of asset or strategy returns.

        Returns
        -------
        calmar: float, pd.Series or pd.DataFrame
            calmar ratio for each asset or strategy.
        """
        cr = self.ann_ret() / (self.max_dd() * -1)

        return cr

    def omega_ratio(self,
                    window_type: str = 'fixed',
                    window_size: int = 365) -> Union[float, pd.Series, pd.DataFrame]:
        """
        Computes Omega ratio of asset or strategy returns.

        Parameters
        ----------
        window_type: str, {'fixed', 'expanding', 'rolling'}, default 'fixed'
            Window type for calculation.
        window_size: int, default 365
            Minimum number of observations to include in moving window.

        Returns
        -------
        omega: float, pd.Series or pd.DataFrame
            Omega ratio.
        """
        if window_type == 'fixed':
            omega = self.returns[self.returns > 0].sum() / (self.returns[self.returns < 0].sum() * -1)
        elif window_type == 'expanding':
            omega = getattr(self.returns[self.returns > 0], window_type)().sum() / \
                    getattr(self.returns[self.returns < 0], window_type)().sum() * -1
        else:
            omega = getattr(self.returns[self.returns > 0], window_type)(window=window_size).sum() / \
                    getattr(self.returns[self.returns < 0], window_type)(window=window_size).sum() * -1

        return omega

    def stability(self,
                  series: str = None,
                  window_type: str = 'fixed',
                  window_size: int = 365) -> Union[float, pd.Series, pd.DataFrame]:
        """
        Computes the stability of a cumulative return.

        series: str, default None
            Name of the col/series to compute stability.
        window_type: str, {'fixed', 'expanding', 'rolling'}, default 'fixed'
            Window type for calculation.
        window_size: int, default 365
            Minimum number of observations to include in moving window.

        Returns
        -------
        stability: float, pd.Series or pd.DataFrame
            Stability defined as R-squared of cumulative log returns regressed on a constant and time trend.
        """
        if series is None:
            raise ValueError("Select a an asset or strategy return to compute stability.")

        # cum ret curve
        cum_log_ret = np.log(self.cumulative_ret(start_val=100))[series]

        if window_type == 'fixed':
            stability = linear_reg(cum_log_ret, None, window_type='fixed', output='rsq', log=False, trend='ct')
        elif window_type == 'expanding':
            stability = linear_reg(cum_log_ret, None, window_type='expanding', output='rsq', log=False, trend='ct')
        else:
            stability = linear_reg(cum_log_ret, None, window_type='rolling', output='rsq', log=False, trend='ct',
                                   lookback=window_size)

        return stability

    def alpha_beta(self,
                   series: str = None,
                   window_type: str = 'fixed',
                   window_size: int = 365) -> Dict[str, Union[float, pd.Series]]:
        """
        Regresses asset or strategy returns on the market return, proxied by the equally weighted average of
        the cross-section of returns.

        Parameters
        ----------
        series: str, default None
            Name of the col/series to compute annualized alpha.
        window_type: str, {'fixed', 'expanding', 'rolling'}, default 'fixed'
            Window type for calculation.
        window_size: int, default 365
            Minimum number of observations to include in moving window.

        Returns
        -------
        stats: dict
            Dictionary with parameter estimate values for alpha and beta.
        """
        if series is None:
            raise ValueError("Select a an asset or strategy return to alpha and beta stats.")
        # compute market return
        if self.mkt_ret is None:
            raise ValueError("To estimate market beta and alpha, a market return must be provided.")
        else:
            mkt_ret = self.mkt_ret

        # fit OLS regression
        if window_type == 'fixed':
            coef = linear_reg(self.returns[series], mkt_ret, window_type='fixed', output='coef', log=False, trend='c')
            p_vals = linear_reg(self.returns[series], mkt_ret, window_type='fixed', output='pval', log=False, trend='c')
        elif window_type == 'expanding':
            coef = linear_reg(self.returns[series], mkt_ret, window_type='expanding', output='coef', log=False,
                              trend='c')
            p_vals = linear_reg(self.returns[series], mkt_ret, window_type='expanding', output='pval', log=False,
                                trend='c')
        else:
            coef = linear_reg(self.returns[series], mkt_ret, window_type='rolling', output='coef', log=False, trend='c',
                              lookback=window_size)
            p_vals = linear_reg(self.returns[series], mkt_ret, window_type='rolling', output='pval', log=False,
                                trend='c', lookback=window_size)

        # extract alpha and betas
        alpha = ((coef['const'] + 1) ** self.ann_factor) - 1
        beta = coef['mkt_ret']
        alpha_pval, beta_pval = p_vals['const'], p_vals['mkt_ret']

        return {'alpha': alpha, 'alpha_pval': alpha_pval, 'beta': beta, 'beta_pval': beta_pval}

    def table(self,
              metrics: Union[str, List[str]] = 'all',
              rank_on: str = None
              ) -> pd.DataFrame:
        """
        Computes key performance metrics for asset or strategy returns.

        Parameters
        ----------
        metrics: str or list, {'ret', 'risk', 'risk_adj_ret', 'alpha', 'key_metrics', 'all'}, default 'all'
            Performance metrics to compute.
        rank_on: str, default None
            Sorts factors in descending order of performance metric selected. None does not rank factors.

        Returns
        -------
        metrics: DataFrame
            DataFrame with computed performance metrics, ranked by selected metric.
        """
        # list of metrics
        if metrics == 'ret':
            metrics = ['Cumulative return', 'Annual return']
        elif metrics == 'risk':
            metrics = ['Annual volatility', 'Skewness', 'Kurtosis', 'Max drawdown', 'VaR', 'Tail ratio']
        elif metrics == 'risk_adj_ret':
            metrics = ['Sharpe ratio', 'Sortino ratio', 'Calmar ratio', 'Omega ratio', 'Stability']
        elif metrics == 'alpha':
            metrics = ['Annual alpha', 'Alpha p-val', 'Beta', 'Beta p-val']
        elif metrics == 'key_metrics':
            metrics = ['Cumulative return', 'Annual return', 'Annual volatility', 'Max drawdown', 'Sharpe ratio',
                       'Calmar ratio', 'Stability', 'Annual alpha', 'Alpha p-val', 'Beta', 'Beta p-val']
        elif metrics == 'all':
            metrics = ['Cumulative return', 'Annual return', 'Annual volatility', 'Skewness', 'Kurtosis',
                       'Max drawdown', 'VaR', 'Tail ratio', 'Sharpe ratio', 'Sortino ratio', 'Calmar ratio',
                       'Omega ratio', 'Stability', 'Annual alpha', 'Alpha p-val', 'Beta', 'Beta p-val']
        else:
            metrics = metrics

        # convert series to df
        if isinstance(self.returns, pd.Series):
            rets = self.returns.to_frame()
        else:
            rets = self.returns

        # create metrics df and add performance metrics
        metrics_df = pd.DataFrame(index=rets.columns)

        # compute metrics
        for metric in metrics:
            if metric == 'Cumulative return':
                metrics_df[metric] = self.cumulative_ret().iloc[-1]
            if metric == 'Annual return':
                metrics_df[metric] = self.ann_ret()
            if metric == 'Annual volatility':
                metrics_df[metric] = self.ann_vol()
            if metric == 'Skewness':
                metrics_df[metric] = self.skewness()
            if metric == 'Kurtosis':
                metrics_df[metric] = self.kurtosis()
            if metric == 'Max drawdown':
                metrics_df[metric] = self.max_dd()
            if metric == 'VaR':
                metrics_df[metric] = self.value_at_risk()
            if metric == 'Tail ratio':
                metrics_df[metric] = self.tail_ratio()
            if metric == 'Sharpe ratio':
                metrics_df[metric] = self.sharpe_ratio()
            if metric == 'Sortino ratio':
                metrics_df[metric] = self.sortino_ratio()
            if metric == 'Calmar ratio':
                metrics_df[metric] = self.calmar_ratio()
            if metric == 'Omega ratio':
                metrics_df[metric] = self.omega_ratio()
            if metric == 'Stability':
                for col in rets.columns:
                    metrics_df.loc[col, metric] = self.stability(series=col)
            if metric == 'Annual alpha':
                for col in rets.columns:
                    metrics_df.loc[col, metric] = self.alpha_beta(series=col)['alpha']
            if metric == 'Alpha p-val':
                for col in rets.columns:
                    metrics_df.loc[col, metric] = self.alpha_beta(series=col)['alpha_pval']
            if metric == 'Beta':
                for col in rets.columns:
                    metrics_df.loc[col, metric] = self.alpha_beta(series=col)['beta']
            if metric == 'Beta p-val':
                for col in rets.columns:
                    metrics_df.loc[col, metric] = self.alpha_beta(series=col)['beta_pval']

        # sort by sharpe ratio and round values to 2 decimals
        if rank_on is not None:
            metrics_df = metrics_df.sort_values(by=rank_on, ascending=False)

        return metrics_df.astype(float).round(decimals=2)

    def plot_metric(self,
                    metric: str = 'cumulative_ret',
                    window_type: str = 'expanding',
                    window_size: int = 365,
                    colors: Optional[str] = None,
                    source: Optional[str] = None,
                    **kwargs) -> None:
        """
        Plots metrics time series using a moving window (expanding or rolling).

        Parameters
        ----------
        metric: str
            Name of metric to plot.
        window_type: str, {'fixed', 'expanding', 'rolling'}, default 'fixed'
            Window type for calculation.
        window_size: int, default 365
            Minimum number of observations to include in moving window.
        colors: str, {'colors_dark', 'colors_mid', 'colors_light'}, default None
            Color scheme to use.
        source: str, default None
            Adds source info to bottom of plot.
        """
        metrics_dict = {'cumulative_ret': 'Cumulative returns', 'ann_vol': 'Annual volatility', 'skewness': 'Skewness',
                        'kurtosis': 'Kurtosis', 'drawdown': 'Drawdown', 'max_dd': 'Max Drawdown',
                        'value_at_risk': 'VaR', 'tail_ratio': 'Tail ratio', 'sharpe_ratio': 'Sharpe ratio',
                        'sortino_ratio': 'Sortino ratio', 'omega_ratio': 'Omega ratio', 'stability': 'Stability',
                        'alpha': 'Annual alpha', 'alpha_pval': 'Alpha p-val', 'beta': 'Beta', 'beta_pval': 'Beta p-val'}

        if window_type == 'fixed':
            raise ValueError("Cannot plot performance metrics for a fixed window type.")

        elif window_type == 'expanding':
            if metric in ['cumulative_ret', 'drawdown']:
                metric_df = getattr(self, metric)(**kwargs)
            elif metric == 'alpha':
                metric_df = getattr(self, 'alpha_beta')(**kwargs)['alpha']
            elif metric == 'alpha_pval':
                metric_df = getattr(self, 'alpha_beta')(**kwargs)['alpha_pval']
            elif metric == 'beta':
                metric_df = getattr(self, 'alpha_beta')(**kwargs)['beta']
            elif metric == 'beta_pval':
                metric_df = getattr(self, 'alpha_beta')(**kwargs)['beta_pval']
            else:
                metric_df = getattr(self, metric)(window_type='expanding', **kwargs)

        else:
            if metric == 'alpha':
                metric_df = getattr(self, 'alpha_beta')(window_type='rolling',
                                                        window_size=window_size, **kwargs)['alpha']
            elif metric == 'alpha_pval':
                metric_df = getattr(self, 'alpha_beta')(window_type='rolling',
                                                        window_size=window_size, **kwargs)['alpha_pval']
            elif metric == 'beta':
                metric_df = getattr(self, 'alpha_beta')(window_type='rolling',
                                                        window_size=window_size, **kwargs)['beta']
            elif metric == 'beta_pval':
                metric_df = getattr(self, 'alpha_beta')(window_type='rolling',
                                                        window_size=window_size, **kwargs)['beta_pval']
            else:
                metric_df = getattr(self, metric)(window_type='rolling',
                                                  window_size=window_size, **kwargs)

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
        metric_df.dropna().plot(color=colors, linewidth=2, rot=0, ax=ax)

        # font
        plt.rcParams['font.family'] = 'georgia'

        # legend
        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

        # grid
        ax.grid(which="major", axis='y', color='#758D99', alpha=0.6, zorder=1)
        ax.set_facecolor("whitesmoke")

        # remove splines
        ax.spines[['top', 'right', 'left']].set_visible(False)

        # format x-axis
        ax.set_xlim(metric_df.index.get_level_values('date')[0], metric_df.index.get_level_values('date')[-1])

        # Reformat y-axis tick labels
        ax.set_ylabel(metrics_dict[metric])
        ax.yaxis.tick_right()

        # add systamental logo
        with resources.path("factorlab", "systamental_logo.png") as f:
            img_path = f
        img = Image.open(img_path)
        plt.figimage(img, origin='upper')

        # Add in title and subtitle
        ax.text(x=0.13, y=.92, s=f"{metrics_dict[metric]}", transform=fig.transFigure, ha='left', fontsize=14,
                weight='bold', alpha=.8, fontdict=None)
        if window_type == 'rolling':
            sub_title = f"{window_size} Period {window_type.title()} Window"
            ax.text(x=0.13, y=.89, s=sub_title, transform=fig.transFigure, ha='left', fontsize=12, alpha=.8,
                    fontdict=None)

        # Set source text
        if source is not None:
            ax.text(x=0.13, y=0.05, s=f"""Source: {source}""", transform=fig.transFigure, ha='left', fontsize=10,
                    alpha=.8, fontdict=None)

    def returns_heatmap(self,
                        series: str = None,
                        ) -> None:
        """
        Creates a heatmap of monthly and yearly returns.

        Parameters
        ----------
        series: str, default None
            Name of the col/series to compute monthly returns.

        """
        fig, ax = plt.subplots(figsize=(15, 7))
        # returns, %
        if self.ret_type == 'simple':
            ret_df = np.log(self.returns + 1) * 100
        else:
            ret_df = self.returns * 100
        # reset index
        ret_df.reset_index(inplace=True)
        # get year and month
        ret_df["year"] = ret_df.date.apply(lambda x: x.year)
        ret_df["month"] = ret_df.date.apply(lambda x: x.strftime("%B"))
        # create table
        table = ret_df.pivot_table(index="year", columns="month", values=series, aggfunc="sum").fillna(0)
        # rename cols, index
        table.columns.name, table.index.name = '', ''
        cols = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
                "November", "December"]
        table = table.reindex(columns=cols)
        table.columns = [col[:3] for col in cols]
        # compute yearly return
        table.loc[:, 'Year'] = table.sum(axis=1)
        table = table.round(decimals=2)  # round
        # plot heatmap
        sns.heatmap(table, annot=True, cmap='RdYlGn', center=0, square=True, cbar=False, fmt='g')

        # add systamental logo
        with resources.path("factorlab", "systamental_logo.png") as f:
            img_path = f
        img = Image.open(img_path)
        plt.figimage(img, origin='upper')

        # Adding title
        ax.set_title('Monthly Returns (%)', loc='left', weight='bold', pad=20, fontsize=14, family='georgia')
        ax.text(x=0, y=-0.05, s=f"{series.title()}", ha='left', fontsize=12, alpha=.8, fontdict=None, family='georgia')
