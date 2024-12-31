import pandas as pd
from typing import Union, Optional, List

from factorlab.strategy_backtesting.metrics import Metrics


class Performance:
    """
    Performance metrics for asset or strategy returns.
    """
    # available metrics
    available_metrics = [
        'Cumulative returns', 'Annual return', 'Winning percentage', 'Annual volatility', 'Skewness', 'Kurtosis',
        'Max drawdown', 'VaR', 'Tail ratio', 'Expected shortfall', 'Conditional drawdown risk', 'Sharpe ratio',
        'Sortino ratio', 'Calmar ratio', 'Omega ratio', 'Profit factor', 'Stability', 'Annual alpha',  'Beta'
    ]

    def __init__(self,
                 returns: Union[pd.Series, pd.DataFrame],
                 risk_free_rate: Optional[Union[pd.DataFrame, pd.Series, float]] = None,
                 as_excess_returns: bool = False,
                 factor_returns: Optional[Union[pd.Series, pd.DataFrame]] = None,
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
            Risk-free rate for computing risk-adjusted returns.
        as_excess_returns: bool, default False
            Whether to compute excess returns.
        factor_returns: pd.Series, pd.DataFrame, default None
            Market returns for computing beta.
        ret_type: str, {'log', 'simple'}, default 'log'
            Type of returns.
        window_type: str, {'fixed', 'expanding', 'rolling'}, default 'fixed'
            Type of window for risk estimation.
        window_size: int, default None
            Window size for risk estimation.
        ann_factor: int, default None
            Annualization factor.
        """
        self.returns = returns
        self.risk_free_rate = risk_free_rate
        self.as_excess_returns = as_excess_returns
        self.factor_returns = factor_returns
        self.ret_type = ret_type
        self.window_type = window_type
        self.window_size = window_size
        self.ann_factor = ann_factor

    # TODO: add testing for series vs df
    def get_metrics(self) -> pd.DataFrame:
        """
        Computes key performance metrics for asset or strategy returns.

        Returns
        -------
        metrics: DataFrame
            DataFrame with computed performance metrics.
        """
        return Metrics(self.returns, risk_free_rate=self.risk_free_rate, as_excess_returns=self.as_excess_returns,
                       factor_returns=self.factor_returns, ret_type=self.ret_type, window_type=self.window_type,
                       window_size=self.window_size, ann_factor=self.ann_factor)

    def get_table(self, metrics: Union[str, List[str]] = 'key_metrics', rank_on: str = None) -> pd.DataFrame:
        """
        Computes key performance metrics for asset or strategy returns.

        Parameters
        ----------
        metrics: str or list, {'returns', 'risks', 'ratios', 'alpha_beta', 'key_metrics', 'all'}, default 'all'
            Performance metrics to compute.
        rank_on: str, default None
            Sorts factors in descending order of performance metric selected. None does not rank factors.

        Returns
        -------
        metrics: DataFrame
            DataFrame with computed performance metrics, ranked by selected metric.
        """
        # create metrics df and add performance metrics
        metrics_df = pd.DataFrame(index=self.returns.columns)

        metrics_dict = {
            'Cumulative returns': 'cumulative_returns',
            'Annual return': 'annualized_return',
            'Winning percentage': 'winning_percentage',
            'Drawdown': 'drawdown',
            'Max drawdown': 'max_drawdown',
            'Conditional drawdown risk': 'conditional_drawdown_risk',
            'Annual volatility': 'annualized_vol',
            'Skewness': 'skewness',
            'Kurtosis': 'kurtosis',
            'VaR': 'value_at_risk',
            'Expected shortfall': 'expected_shortfall',
            'Tail ratio': 'tail_ratio',
            'Sharpe ratio': 'sharpe_ratio',
            'Sortino ratio': 'sortino_ratio',
            'Calmar ratio': 'calmar_ratio',
            'Omega ratio': 'omega_ratio',
            'Profit factor': 'profit_factor',
            'Stability': 'stability',
            'Annual alpha': 'alpha',
            'Beta': 'beta'
        }

        if metrics == 'key_metrics':
            metrics = ['Annual return', 'Annual volatility', 'Max drawdown', 'Sharpe ratio', 'Calmar ratio']
        elif metrics == 'all':
            metrics = self.available_metrics
        elif metrics == 'alpha_beta':
            metrics = ['Annual alpha', 'Beta']
        elif metrics == 'returns':
            metrics = ['Cumulative returns', 'Annual return', 'Winning percentage']
        elif metrics == 'risks':
            metrics = ['Annual volatility', 'Skewness', 'Kurtosis', 'Max drawdown', 'VaR', 'Tail ratio',
                       'Expected shortfall', 'Conditional drawdown risk']
        elif metrics == 'ratios':
            metrics = ['Sharpe ratio', 'Sortino ratio', 'Calmar ratio', 'Omega ratio', 'Stability', 'Profit factor']
        else:
            metrics = metrics

        # compute metrics
        for metric in metrics:
            if metric == 'Cumulative returns':
                metrics_df[metric] = self.get_metrics().cumulative_returns().iloc[-1]
            elif metric == 'Annual alpha':
                metrics_df[metric] = self.get_metrics().alpha()['alpha']
                metrics_df['Alpha p-val'] = self.get_metrics().alpha()['p_val']
            elif metric == 'Beta':
                metrics_df[metric] = self.get_metrics().beta()['beta']
                metrics_df['Beta p-val'] = self.get_metrics().beta()['p_val']
            else:
                metrics_df[metric] = getattr(self.get_metrics(), metrics_dict[metric])()

        # sort by sharpe ratio and round values to 2 decimals
        if rank_on is not None:
            metrics_df = metrics_df.sort_values(by=rank_on, ascending=False)

        return metrics_df.astype(float).round(decimals=4)

    def compute_factor_exposure(self, factor_returns: pd.DataFrame, factor_exposures: pd.DataFrame) -> pd.DataFrame:
        """
        Compute the factor exposure for the strategy returns.

        Parameters
        ----------
        factor_returns: pd.DataFrame
            DataFrame with factor returns.
        factor_exposures: pd.DataFrame
            DataFrame with factor exposures.

        Returns
        -------
        exposure: pd.DataFrame
            DataFrame with factor exposure.
        """
        pass

    def risk_return_attribution(self, factor_returns: pd.DataFrame, factor_exposures: pd.DataFrame) -> pd.DataFrame:
        """
        Compute the factor attribution for the strategy returns.

        Parameters
        ----------
        factor_returns: pd.DataFrame
            DataFrame with factor returns.
        factor_exposures: pd.DataFrame
            DataFrame with factor exposures.

        Returns
        -------
        attribution: pd.DataFrame
            DataFrame with factor attribution.
        """
        pass
