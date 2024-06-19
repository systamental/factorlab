import numpy as np
import pandas as pd

from typing import Union, Optional
from factorlab.signal_generation.time_series_analysis import TimeSeriesAnalysis as TSA


class Metrics:
    """
    Performance metrics for asset or strategy returns.
    """
    def __init__(self,
                 returns: Union[pd.Series, pd.DataFrame],
                 risk_free_rate: Optional[Union[pd.DataFrame, pd.Series, float]] = None,
                 as_excess_returns: bool = False,
                 mkt_ret: Optional[Union[pd.Series, pd.DataFrame]] = None,
                 ret_type: str = 'log',
                 window_type: str = 'fixed',
                 window_size: Optional[int] = None,
                 ann_factor: Optional[int] = None
                 ):
        """
        Constructor

        Parameters
        ----------
        returns: pd.Series or pd.DataFrame
            Dataframe or series with DatetimeIndex and returns (cols).
        risk_free_rate: pd.Series, pd.DataFrame, float, default None
            Risk-free rate series for net returns computation.
        as_excess_returns: bool, default False
            Whether to metrics with excess returns.
        mkt_ret: pd.Series, default None
            Market return to compute beta.
        ret_type: str, {'simple', 'log'}, default 'simple'
            Return type.
        window_type: str, {'fixed', 'expanding', 'rolling'}, default 'fixed'
            Window type for calculation.
        window_size: int, default 365
            Minimum number of observations to include in moving window.
        ann_factor: int, default None
            Annualization factor.
        """
        self.returns = returns
        self.risk_free_rate = risk_free_rate
        self.as_excess_returns = as_excess_returns
        self.mkt_ret = mkt_ret
        self.ret_type = ret_type
        self.window_type = window_type
        self.window_size = window_size
        self.ann_factor = ann_factor
        self.freq = None
        self.preprocess_data()

    def preprocess_data(self):
        """
        Preprocess the data for the metrics computation.
        """
        # returns
        if isinstance(self.returns.index, pd.MultiIndex):  # check index
            raise ValueError('MultiIndex not supported. Convert to single index.')
        if not isinstance(self.returns, pd.DataFrame) and not isinstance(self.returns, pd.Series):  # check data type
            raise ValueError('rets must be a pd.DataFrame or pd.Series')
        if isinstance(self.returns, pd.Series):  # convert to df
            self.returns = self.returns.to_frame()
        self.returns.index = pd.to_datetime(self.returns.index)  # convert to index to datetime
        self.returns = self.returns.dropna(how='all')  # drop missing rows

        # freq
        self.freq = pd.infer_freq(self.returns.index)

        # ann_factor
        if self.ann_factor is None:
            self.ann_factor = self.returns.groupby(self.returns.index.year).count().max().mode()[0]

        # risk-free rate
        if self.risk_free_rate is None:
            self.risk_free_rate = 0.0
        elif isinstance(self.risk_free_rate, (pd.Series, pd.DataFrame)):
            self.risk_free_rate = self.returns.join(self.risk_free_rate).ffill().iloc[:, -1]  # join and ffill
            self.risk_free_rate = np.log(1 + self.risk_free_rate) / self.ann_factor  # de-annualize

        # market return
        if self.mkt_ret is None:
            self.mkt_ret = self.returns.mean(axis=1)

        # window size
        if self.window_size is None:
            self.window_size = self.ann_factor

        # excess returns
        self.excess_returns()

    def excess_returns(self) -> Union[pd.Series, pd.DataFrame]:
        """
        Computes excess returns.

        Returns
        -------
        excess_returns: pd.Series or pd.DataFrame
            Excess returns.
        """
        if self.as_excess_returns:
            self.returns = self.returns.sub(self.risk_free_rate, axis=0)

    def cumulative_returns(self, start_val: Optional[int] = 0) -> Union[pd.Series, pd.DataFrame]:
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
        # cumulative returns
        if self.ret_type == 'simple':
            cum_ret = (1 + self.returns).cumprod()
        else:
            cum_ret = np.exp(self.returns.cumsum())

        # start val
        if start_val == 0:
            cum_ret = cum_ret - 1
        else:
            cum_ret = cum_ret * start_val

        return cum_ret

    def annualized_return(self) -> Union[pd.Series, float]:
        """
        Computes the compound annual growth rate of returns, equivalent to CAGR.

        Returns
        -------
        dd: float or pd.Series
            Annualized return for each asset or strategy.
        """
        # cum return
        cum_ret = self.cumulative_returns(start_val=1).iloc[-1]
        # number of years of returns
        num_years = self.returns.count() / self.ann_factor
        # annualized returns
        ann_ret = cum_ret ** (1 / num_years) - 1

        return ann_ret

    def winning_percentage(self) -> Union[pd.Series, pd.DataFrame]:
        """
        Computes the winning percentage of returns.

        Returns
        -------
        wp: pd.Series or pd.DataFrame
            Winning percentage.
        """
        if self.window_type == 'rolling':
            wp = getattr(self.returns, self.window_type)(window=self.window_size).apply(lambda x: (x > 0).mean())
        elif self.window_type == 'expanding':
            wp = getattr(self.returns, self.window_type)().apply(lambda x: (x > 0).mean())
        else:
            wp = self.returns.apply(lambda x: (x > 0).mean())

        return wp

    def drawdown(self) -> Union[pd.Series, pd.DataFrame]:
        """
        Computes drawdowns for an asset or strategy.

        Returns
        -------
        dd: pd.Series or pd.DataFrame
            Dataframe or series with DatetimeIndex and drawdowns.

        """
        # cumulative return
        cum_ret = self.cumulative_returns(start_val=100)
        # drawdown
        dd = (cum_ret / cum_ret.expanding().max()) - 1

        return dd

    def max_drawdown(self) -> Union[float, pd.Series, pd.DataFrame]:
        """
        Computes the maximum drawdown for an asset or strategy.

        Returns
        -------
        dd: float
            Maximum drawdown, returns as negative decimal (float).
        """
        # max drawdown
        if self.window_type == 'rolling':
            max_dd = getattr(self.drawdown(), self.window_type)(window=self.window_size).min()
        elif self.window_type == 'expanding':
            max_dd = getattr(self.drawdown(), self.window_type)().min()
        else:
            max_dd = self.drawdown().min()

        return max_dd

    def conditional_drawdown_risk(self, alpha: float = 0.05) -> Union[float, pd.Series, pd.DataFrame]:
        """
        Computes the conditional drawdown risk (CDaR) of an asset or strategy.

        Parameters
        ----------
        alpha : float, optional, default 0.05
            Confidence level/threshold for the tail of the distribution of drawdowns.

        Returns
        -------
        cdar : float, pd.Series or pd.DataFrame
            Conditional drawdown risk.
        """
        # max drawdown
        dd = self.drawdown()
        max_dd = dd.expanding().min()

        # max drawdown at confidence level
        max_dd_quantile = max_dd.quantile(alpha)
        cdar = max_dd[max_dd <= max_dd_quantile].mean()

        return cdar

    def annualized_vol(self) -> Union[float, pd.Series, pd.DataFrame]:
        """
        Computes the annualized volatility (standard deviation) of returns.

        Returns
        -------
        ann_vol: float, pd.Series or pd.DataFrame
            Annualized volatility for each asset or strategy.
        """
        if self.window_type == 'rolling':
            ann_vol = getattr(self.returns, self.window_type)(window=self.window_size).std() * np.sqrt(self.ann_factor)
        elif self.window_type == 'expanding':
            ann_vol = getattr(self.returns, self.window_type)().std() * np.sqrt(self.ann_factor)
        else:
            ann_vol = self.returns.std() * np.sqrt(self.ann_factor)

        return ann_vol

    def skewness(self) -> Union[float, pd.Series, pd.DataFrame]:
        """
        Computes the skewness of returns (asymmetry).

        Returns
        -------
        skew: float, pd.Series or pd.DataFrame
            Skewnesss of returns.
        """
        if self.window_type == 'rolling':
            skew = getattr(self.returns, self.window_type)(window=self.window_size).skew()
        elif self.window_type == 'expanding':
            skew = getattr(self.returns, self.window_type)().skew()
        else:
            skew = self.returns.skew()

        return skew

    def kurtosis(self) -> Union[float, pd.Series, pd.DataFrame]:
        """
        Computes the kurtosis of returns (tails).

        Returns
        -------
        kurt: float, pd.Series or pd.DataFrame
            Kurtosis of returns.
        """
        if self.window_type == 'rolling':
            kurt = getattr(self.returns, self.window_type)(window=self.window_size).kurt()
        elif self.window_type == 'expanding':
            kurt = getattr(self.returns, self.window_type)().kurt()
        else:
            kurt = self.returns.kurt()

        return kurt

    def value_at_risk(self, alpha: float = 0.05) -> Union[float, pd.Series, pd.DataFrame]:
        """
        Value at risk (VaR) of an asset or strategy.

        Parameters
        ----------
        alpha : float, optional, default 0.05
            Confidence level/threshold for the tail of the distribution of returns.

        Returns
        -------
        VaR : float, pd.Series or pd.DataFrame
            The VaR value.
        """
        if self.window_type == 'rolling':
            var = getattr(self.returns, self.window_type)(window=self.window_size).quantile(alpha)
        elif self.window_type == 'expanding':
            var = getattr(self.returns, self.window_type)().quantile(alpha)
        else:
            var = self.returns.quantile(alpha)

        return var

    def expected_shortfall(self, alpha: float = 0.05) -> Union[float, pd.Series, pd.DataFrame]:
        """
        Expected shortfall (ES) of an asset or strategy.

        Parameters
        ----------
        alpha : float, optional, default 0.05
            Confidence level/threshold for the tail of the distribution of returns.

        Returns
        -------
        ES : float, pd.Series or pd.DataFrame
            The ES value.
        """
        if self.window_type == 'rolling':
            es = getattr(self.returns[self.returns < self.value_at_risk(alpha=alpha)],
                         self.window_type)(window=self.window_size).mean()
        elif self.window_type == 'expanding':
            es = getattr(self.returns[self.returns < self.value_at_risk(alpha=alpha)],
                         self.window_type)().mean()
        else:
            es = self.returns[self.returns < self.value_at_risk(alpha=alpha)].mean()

        return es

    def tail_ratio(self, alpha: float = 0.05) -> Union[float, pd.Series, pd.DataFrame]:
        """
        Computes the tail ratio of returns, a measure of return asymmetry which allows us to compare
        large gains to large losses, e.g the ratio of the 99th percentile to 1st percentile.

        Parameters
        ----------
        alpha : float, optional, default 0.05
            Confidence level/threshold for the tails of the distribution of returns.

        Returns
        -------
        tr: float, pd.Series or pd.DataFrame
            Tail ratio.
        """
        if self.window_type == 'rolling':
            tr = np.abs(getattr(self.returns, self.window_type)(window=self.window_size).quantile(1 - alpha) /
                        getattr(self.returns, self.window_type)(window=self.window_size).quantile(alpha))
        elif self.window_type == 'expanding':
            tr = np.abs(getattr(self.returns, self.window_type)().quantile(1 - alpha) /
                        getattr(self.returns, self.window_type)().quantile(alpha))
        else:
            tr = np.abs(self.returns.quantile(1 - alpha)) / np.abs(self.returns.quantile(alpha))

        return tr

    def sharpe_ratio(self) -> Union[float, pd.Series, pd.DataFrame]:
        """
        Computes the Sharpe ratio of asset or strategy returns.

        Returns
        -------
        sr: float, pd.Series or pd.DataFrame
            Sharpe ratio for each asset or strategy.
        """
        if self.window_type == 'rolling':
            sr = (getattr(self.returns, self.window_type)(window=self.window_size).mean() /
                  getattr(self.returns, self.window_type)(window=self.window_size).std()) * np.sqrt(self.ann_factor)

        elif self.window_type == 'expanding':
            sr = (getattr(self.returns, self.window_type)().mean() /
                  getattr(self.returns, self.window_type)().std()) * np.sqrt(self.ann_factor)
        else:
            sr = (self.returns.mean() / self.returns.std()) * np.sqrt(self.ann_factor)

        return sr

    def sortino_ratio(self) -> Union[float, pd.Series, pd.DataFrame]:
        """
        Computes the Sortino ratio of asset or strategy returns.

        Returns
        -------
        sr: float, pd.Series or pd.DataFrame
            Sortino ratio for each asset or strategy.
        """
        if self.window_type == 'rolling':
            sr = (getattr(self.returns, self.window_type)(window=self.window_size).mean() /
                  getattr(self.returns[self.returns < 0], self.window_type)(window=self.window_size).std()) \
                  * np.sqrt(self.ann_factor)
        elif self.window_type == 'expanding':
            sr = (getattr(self.returns, self.window_type)().mean() /
                  getattr(self.returns[self.returns < 0], self.window_type)().std()) * np.sqrt(self.ann_factor)
        else:
            sr = (self.returns.mean() / self.returns[self.returns < 0].std()) * np.sqrt(self.ann_factor)

        return sr

    def calmar_ratio(self) -> Union[float, pd.Series, pd.DataFrame]:
        """
        Computes the Calmar ratio of asset or strategy returns.

        Returns
        -------
        calmar: float, pd.Series or pd.DataFrame
            calmar ratio for each asset or strategy.
        """
        cr = self.annualized_return() / (self.max_drawdown() * -1)

        return cr

    def omega_ratio(self) -> Union[float, pd.Series, pd.DataFrame]:
        """
        Computes Omega ratio of asset or strategy returns.

        Returns
        -------
        omega: float, pd.Series or pd.DataFrame
            Omega ratio.
        """
        if self.window_type == 'rolling':
            omega = getattr(self.returns[self.returns > 0], self.window_type)(window=self.window_size).sum() / \
                    getattr(self.returns[self.returns < 0], self.window_type)(window=self.window_size).sum() * -1
        elif self.window_type == 'expanding':
            omega = getattr(self.returns[self.returns > 0], self.window_type)().sum() / \
                    getattr(self.returns[self.returns < 0], self.window_type)().sum() * -1
        else:
            omega = self.returns[self.returns > 0].sum() / (self.returns[self.returns < 0].sum() * -1)

        return omega

    def profit_factor(self) -> Union[float, pd.Series, pd.DataFrame]:
        """
        Computes the profit factor of asset or strategy returns.

        Returns
        -------
        pf: float, pd.Series or pd.DataFrame
            Profit factor.
        """
        if self.window_type == 'rolling':
            pf = getattr(self.returns[self.returns > 0], self.window_type)(window=self.window_size).mean() / \
                 getattr(self.returns[self.returns < 0], self.window_type)(window=self.window_size).mean() * -1
        elif self.window_type == 'expanding':
            pf = getattr(self.returns[self.returns > 0], self.window_type)().mean() / \
                 getattr(self.returns[self.returns < 0], self.window_type)().mean() * -1
        else:
            pf = self.returns[self.returns > 0].mean() / (self.returns[self.returns < 0].mean() * -1)

        return pf

    def stability(self) -> Union[pd.Series, pd.DataFrame]:
        """
        Computes the stability of a cumulative return.

        Returns
        -------
        stability: pd.Series or pd.DataFrame
            Stability defined as R-squared of cumulative log returns regressed on a constant and time trend.
        """
        # cum ret curve
        cum_log_ret = np.log(self.cumulative_returns(start_val=100))

        # df
        stability_df = pd.DataFrame()

        # compute stability
        if self.window_type == 'rolling':
            for ticker in cum_log_ret.columns:
                df1 = TSA(cum_log_ret[ticker], window_type=self.window_type, window_size=self.window_size, trend='ct').\
                    linear_regression(output='rsquared')
                stability_df = pd.concat([stability_df, df1], axis=1)

            # rename cols
            stability_df.columns = cum_log_ret.columns

        else:
            stability = []
            for ticker in cum_log_ret.columns:
                stability.append(
                    TSA(cum_log_ret[ticker], window_type='fixed', window_size=self.window_size, trend='ct').
                    linear_regression(output='rsquared'))
            # create df
            stability_df = pd.DataFrame(stability, index=cum_log_ret.columns, columns=['stability'])

        return stability_df

    def alpha(self) -> Union[pd.Series, pd.DataFrame]:
        """
        Estimates the alpha coefficient by regressing asset or strategy returns on the market return,
        proxied by the equally weighted average of the cross-section of returns.

        Returns
        -------
        df: pd.Series or pd.DataFrame
            Dataframe with alpha parameter estimate values and p-values.
        """
        # df
        alpha_df, pval_df = pd.DataFrame(), pd.DataFrame()

        if self.window_type == 'rolling':
            for ticker in self.returns.columns:

                # fit OLS regression
                # alpha
                coef = TSA(self.returns[ticker], self.mkt_ret, trend='c', window_type=self.window_type,
                           window_size=self.window_size).linear_regression(output='params')
                alpha = ((coef['const'] + 1) ** self.ann_factor) - 1
                alpha_df = pd.concat([alpha_df, alpha.to_frame()], axis=1)

                # p-values
                p_vals = TSA(self.returns[ticker], self.mkt_ret, trend='c', window_type=self.window_type,
                             window_size=self.window_size).linear_regression(output='pvalues')
                alpha_pval = p_vals['const']
                pval_df = pd.concat([pval_df, alpha_pval.to_frame()], axis=1)

            # rename cols
            alpha_df.columns, pval_df.columns = self.returns.columns, self.returns.columns
            alpha_df.columns.name, pval_df.columns.name = 'ticker', 'ticker'
            alpha_df, pval_df = alpha_df.stack().to_frame('alpha'), pval_df.stack().to_frame('p_val')
            # stack dfs
            df = pd.concat([alpha_df, pval_df], axis=1)

        else:
            alpha_list, alpha_pval_list = [], []
            for ticker in self.returns.columns:
                # fit OLS regression
                # alpha
                coef = TSA(self.returns[ticker], self.mkt_ret, trend='c', window_type='fixed',
                           window_size=self.window_size).linear_regression(output='params')
                alpha = ((coef['const'] + 1) ** self.ann_factor) - 1
                alpha_list.append(alpha)

                # p-values
                p_vals = TSA(self.returns[ticker], self.mkt_ret, trend='c', window_type='fixed',
                             window_size=self.window_size).linear_regression(output='pvalues')
                alpha_pval = p_vals['const']
                alpha_pval_list.append(alpha_pval)

            # create df
            df = pd.DataFrame({'alpha': alpha_list, 'p_val': alpha_pval_list}, index=self.returns.columns)

        return df

    def beta(self) -> Union[pd.Series, pd.DataFrame]:
        """
        Estimates the beta coefficient by regressing asset or strategy returns on the market return,
        proxied by the equally weighted average of the cross-section of returns.

        Returns
        -------
        df: pd.Series or pd.DataFrame
            Dataframe with beta parameter estimate values and p-values.
        """
        # df
        beta_df, pval_df = pd.DataFrame(), pd.DataFrame()

        # compute stability
        if self.window_type == 'rolling':
            for ticker in self.returns.columns:

                # fit OLS regression
                # beta
                coef = TSA(self.returns[ticker], self.mkt_ret, trend='c', window_type=self.window_type,
                           window_size=self.window_size).linear_regression(output='params')
                beta = coef.iloc[:, 1].to_frame()
                beta_df = pd.concat([beta_df, beta], axis=1)

                # p-values
                p_vals = TSA(self.returns[ticker], self.mkt_ret, trend='c', window_type=self.window_type,
                             window_size=self.window_size).linear_regression(output='pvalues')
                beta_pval = p_vals.iloc[:, 1]
                pval_df = pd.concat([pval_df, beta_pval.to_frame()], axis=1)

            # rename cols
            beta_df.columns, pval_df.columns = self.returns.columns, self.returns.columns
            beta_df.columns.name, pval_df.columns.name = 'ticker', 'ticker'
            beta_df, pval_df = beta_df.stack().to_frame('beta'), pval_df.stack().to_frame('p_val')
            # stack dfs
            df = pd.concat([beta_df, pval_df], axis=1)

        else:
            beta_list, beta_pval_list = [], []
            for ticker in self.returns.columns:
                # fit OLS regression
                # beta
                coef = TSA(self.returns[ticker], self.mkt_ret, trend='c', window_type='fixed',
                           window_size=self.window_size).linear_regression(output='params')
                beta = coef.iloc[1]
                beta_list.append(beta)

                # p-values
                p_vals = TSA(self.returns[ticker], self.mkt_ret, trend='c', window_type='fixed',
                             window_size=self.window_size).linear_regression(output='pvalues')
                beta_pval = p_vals.iloc[1]
                beta_pval_list.append(beta_pval)

            # create df
            df = pd.DataFrame({'beta': beta_list, 'p_val': beta_pval_list}, index=self.returns.columns)

        return df
