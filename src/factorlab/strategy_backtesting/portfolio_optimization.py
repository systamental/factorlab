import pandas as pd
import numpy as np
from typing import Optional, Union, Any


class PortfolioOptimization:
    """
    Portfolio optimization class.

    This class is used to analyze the transaction cost of a trading strategy.
    """
    def __init__(self,
                 ret: Union[pd.DataFrame, pd.Series],
                 signals: Optional[Union[pd.DataFrame, pd.Series]] = None,
                 method: str = 'ew',
                 target_vol: Optional[float] = None,
                 t_cost: Optional[float] = None,
                 rebal_freq: Optional[Union[str, int]] = None,
                 window_type: str = 'expanding',
                 window_size: int = 365,
                 ann_factor: Optional[int] = None,
                 ):
        """
        Constructor

        Parameters
        ----------
        ret: pd.DataFrame or pd.Series
            The returns of the assets or strategies. If not provided, the returns are computed from the prices.
        signals: pd.DataFrame, default None
            The signals of the assets or strategies.
        method: str, {'equal_weight', 'inverse_volatility', 'max_sharpe_ratio', 'min_variance', 'risk_parity',
                        'max_diversification_ratio', 'min_volatility'}, default 'equal_weight'
            Optimization method to compute weights.
        target_vol: float, default None
            Target volatility for portfolio returns.
        t_cost: float
            Transaction cost.
        rebal_freq: Optional[Union[str, int]]
            Rebalancing frequency.
        window_type: str, {'expanding', 'rolling'}, default 'expanding'
            Type of the window. It can be 'expanding' or 'rolling'.
        window_size: int, default 365
            Size of the window.
        """
        self.ret = ret
        self.signals = signals
        self.method = method
        self.target_vol = target_vol
        self.t_cost = t_cost
        self.rebal_freq = rebal_freq
        self.window_type = window_type
        self.window_size = window_size
        self.ann_factor = ann_factor
        self.freq = None
        self.index = None
        self.signal_rets = None
        self.weights = None
        self.weighted_signals = None
        self.t_costs = None
        self.port_rets = None
        self.preprocess_data()

    def preprocess_data(self) -> None:
        """
        Preprocesses data.
        """
        # convert to df
        if isinstance(self.ret, pd.Series):
            self.ret = self.ret.to_frame()
        if self.signals is not None and isinstance(self.signals, pd.Series):
            self.signals = self.signals.to_frame()

        # concat signals and returns
        data = pd.concat([self.signals, self.ret], axis=1, join='inner').dropna()
        if self.signals is not None:
            self.signals = data.iloc[:, :-1]
            self.ret = data.iloc[:, -1].to_frame()

        # convert to index to datetime
        self.index = data.index
        if isinstance(self.index, pd.MultiIndex):
            if not isinstance(self.index.levels[0], pd.DatetimeIndex):
                self.index = self.index.set_levels(pd.to_datetime(self.index.levels[0]), level=0)
        else:
            self.index = pd.to_datetime(self.index)
        # set index
        if self.signals is not None:
            self.signals.index = self.index
        self.ret.index = self.index

        # freq
        if isinstance(self.index, pd.MultiIndex):
            self.freq = pd.infer_freq(self.index.get_level_values(0).unique())
        else:
            self.freq = pd.infer_freq(self.index)

        # ann_factor
        if self.ann_factor is None:
            if self.freq == 'D':
                self.ann_factor = 365
            elif self.freq == 'W':
                self.ann_factor = 52
            elif self.freq == 'M':
                self.ann_factor = 12
            else:
                self.ann_factor = 252

    def compute_signal_returns(self):
        """
        Compute the signal returns.
        """
        if self.signals is not None:
            # concat signals and returns
            df = pd.concat([self.signals, self.ret], axis=1, join='inner')
            # multiply signals by returns
            self.signal_rets = df.iloc[:, :-1].mul(df.iloc[:, -1].values, axis=0).dropna(how='all')
        else:
            self.signal_rets = self.ret

        return self.signal_rets

    def compute_ew_weights(self):
        """
        Compute equal weights for assets or strategies.
        """
        # compute signal rets
        self.compute_signal_returns()

        if isinstance(self.ret.index, pd.MultiIndex):
            self.weights = 1 / self.signal_rets.groupby(level=0).transform('count')
        else:
            self.weights = 1 / self.signal_rets.count(axis=1)

        # Replace inf and -inf with nan
        self.weights.replace([np.inf, -np.inf], np.nan, inplace=True)

        # sort
        self.weights = self.weights.sort_index()

        return self.weights

    def compute_iv_weights(self,
                           method: str = 'smw'
                           ) -> pd.DataFrame:
        """
        Compute inverse volatility weights for assets or strategies.

        Parameters
        ----------
        method: str, {'smw', 'ewm'}, default 'smw'
            Type of moving window for volatility computation.

        Returns
        -------
        pd.DataFrame
            The inverse volatility weights.
        """
        # compute signal rets
        self.compute_signal_returns()

        # std
        if isinstance(self.signal_rets.index, pd.MultiIndex):
            if method == 'ewm':
                std_df = self.signal_rets.groupby(level=1).ewm(self.window_size, min_periods=self.window_size).\
                    std().droplevel(0)
            else:
                std_df = self.signal_rets.groupby(level=1).rolling(self.window_size).std().droplevel(0)

        else:
            if method == 'ewm':
                std_df = self.signal_rets.ewm(self.window_size, min_periods=self.window_size).std()
            else:
                std_df = self.signal_rets.rolling(self.window_size).std()

        # inv vol factor
        self.weights = self.target_vol / (std_df * np.sqrt(self.ann_factor))

        # replace inf and -inf with nan
        self.weights.replace([np.inf, -np.inf], np.nan, inplace=True)

        # divide by sum of weights
        if isinstance(self.signal_rets.index, pd.MultiIndex):
            self.weights = self.weights.div(self.weights.groupby(level=0).sum(), level=0)
        else:
            self.weights = self.weights.div(self.weights.sum(axis=1), axis=0)

        # sort index
        self.weights = self.weights.sort_index()

        return self.weights

    def compute_mv_weights(self, cov_matrix: pd.DataFrame) -> pd.DataFrame:
        """
        Compute minimum variance optimization weights for assets or strategies.

        Parameters
        ----------
        cov_matrix: pd.DataFrame
            Covariance matrix.

        Returns
        -------
        pd.DataFrame
            The minimum variance optimization weights.
        """
        pass

    def compute_weights(self, **kwargs: Any) -> None:
        """
        Compute the weights of the assets or strategies.
        """
        self.weights = getattr(self, f'compute_{self.method}_weights')(**kwargs)

        return self.weights

    def compute_weighted_signals(self, **kwargs: Any) -> pd.DataFrame:
        """
        Compute the weighted signals.
        """
        # compute weights
        self.compute_weights(**kwargs)

        if self.signals is not None and isinstance(self.signals.index, pd.MultiIndex):
            self.weighted_signals = self.signals.mul(self.weights.values, axis=0)
        elif self.signals is not None:
            self.weighted_signals = self.signals
        else:
            self.weighted_signals = self.weights

        return self.weighted_signals

    def rebalance_portfolio(self, **kwargs: Any) -> pd.DataFrame:
        """
        Rebalance portfolio.

        Returns
        -------
        signals: pd.DataFrame
            Rebalanced factors with DatetimeIndex (level 0), tickers (level 1) and signals (cols).
        """
        # compute weights
        self.compute_weighted_signals(**kwargs)

        # frequency dictionary
        freq_dict = {'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3, 'friday': 4, 'saturday': 5,
                     'sunday': 6, '15th': 15, 'month_end': 'is_month_end', 'month_start': 'is_month_start'}

        # rebalancing
        if self.rebal_freq is not None:

            if isinstance(self.weighted_signals.index, pd.MultiIndex):
                df = self.weighted_signals.unstack().copy()

                # day of the week
                if self.rebal_freq in list(freq_dict.keys())[:7]:
                    rebal_df = df[df.index.dayofweek == freq_dict[self.rebal_freq]]
                # mid-month
                elif self.rebal_freq == '15th':
                    rebal_df = df[df.index.day == 15]
                # fixed period
                elif isinstance(self.rebal_freq, int):
                    rebal_df = df.iloc[::self.rebal_freq, :]
                # month start, month end
                else:
                    rebal_df = df[getattr(df.index, freq_dict[self.rebal_freq])]

                # reindex, forward fill and stack
                self.weighted_signals = rebal_df.reindex(df.index).ffill().stack(future_stack=True).\
                    sort_index().dropna(how='all')

            else:
                df = self.weighted_signals.copy()

                # day of the week
                if self.rebal_freq in list(freq_dict.keys())[:7]:
                    rebal_df = df[df.index.dayofweek == freq_dict[self.rebal_freq]]
                # mid-month
                elif self.rebal_freq == '15th':
                    rebal_df = df[df.index.day == 15]
                # fixed period
                elif isinstance(self.rebal_freq, int):
                    rebal_df = df.iloc[::self.rebal_freq, :]
                # month start, month end
                else:
                    rebal_df = df[getattr(df.index, freq_dict[self.rebal_freq])]

                # reindex and forward fill
                self.weighted_signals = rebal_df.reindex(df.index).ffill().dropna(how='all')

        return self.weighted_signals

    def compute_tcosts(self, **kwargs) -> pd.DataFrame:
        """
        Computes transactions costs from changes in signals.

        Returns
        -------
        t_costs: pd.Series
            Series with DatetimeIndex (level 0), tickers (level 1) and transaction costs (cols).
        """
        # rebalance
        self.rebalance_portfolio(**kwargs)

        # no t-costs
        if self.t_cost is None:
            self.t_costs = pd.DataFrame(data=0, index=self.weighted_signals.index,
                                        columns=self.weighted_signals.columns)
        # t-costs
        else:
            if isinstance(self.weighted_signals.index, pd.MultiIndex):
                self.t_costs = self.weighted_signals.groupby(level=1).diff().abs() * self.t_cost
            else:
                self.t_costs = self.weighted_signals.diff().abs() * self.t_cost

        return self.t_costs

    def compute_portfolio_returns(self, **kwargs: Any) -> pd.DataFrame:
        """
        Computes optimized portfolio returns.
        """
        # t-costs
        self.t_costs = self.compute_tcosts(**kwargs)

        # compute gross returns
        self.port_rets = self.weighted_signals.mul(self.ret.reindex(self.weighted_signals.index).values, axis=0)

        # compute net returns
        self.port_rets = self.port_rets.subtract(self.t_costs, axis=0).dropna(how='all')

        # compute portfolio returns
        if isinstance(self.port_rets.index, pd.MultiIndex):
            self.port_rets = self.port_rets.groupby(level=0).sum()
        else:
            self.port_rets = self.port_rets

        # replace NaNs with 0s
        self.port_rets.iloc[0] = 0

        return self.port_rets
