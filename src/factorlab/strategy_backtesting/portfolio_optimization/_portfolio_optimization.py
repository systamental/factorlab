import pandas as pd
import numpy as np
from typing import Optional, Union, Any, List

from factorlab.strategy_backtesting.portfolio_optimization.naive import NaiveOptimization
from factorlab.strategy_backtesting.portfolio_optimization.mvo import MVO
from factorlab.strategy_backtesting.portfolio_optimization.clustering import HRP, HERC
from factorlab.data_viz.plot import plot_bar, plot_series

from joblib import Parallel, delayed
from tqdm import tqdm


class ProgressParallel(Parallel):
    """Custom Parallel backend that includes a progress bar"""
    def __init__(self, use_tqdm=True, total=None, desc=None, *args, **kwargs):
        self._use_tqdm = use_tqdm
        self._total = total
        self._desc = desc
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        with tqdm(disable=not self._use_tqdm, total=self._total, desc=self._desc) as self._pbar:
            return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        if self._total is None:
            self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()


class PortfolioOptimization:
    """
    Portfolio optimization class.

    This class computes the optimized portfolio weights or returns based on the signals returns of assets or strategies.
    """
    # list of available optimizer names
    available_optimizers = ['equal_weight', 'signal_weight', 'inverse_variance', 'inverse_vol', 'target_vol',
                            'random', 'max_return', 'min_vol', 'max_return_min_vol', 'max_sharpe',
                            'max_diversification', 'efficient_return', 'efficient_risk', 'risk_parity', 'hrp', 'herc']

    def __init__(self,
                 returns: Union[pd.DataFrame, pd.Series],
                 signals: Optional[Union[pd.DataFrame, pd.Series]] = None,
                 as_signal_returns: bool = False,
                 method: str = 'equal_weight',
                 lags: int = 1,
                 risk_free_rate: Optional[float] = 0.0,
                 as_excess_returns: bool = False,
                 max_weight: float = 1.0,
                 min_weight: float = 0.0,
                 fully_invested: bool = False,
                 gross_exposure: float = 1.0,
                 net_exposure: Optional[float] = None,
                 round_weights: bool = False,
                 risk_aversion: float = 1.0,
                 exp_ret_method: Optional[str] = 'historical_mean',
                 cov_matrix_method: Optional[str] = 'covariance',
                 target_return: Optional[float] = 0.15,
                 target_risk: Optional[float] = 0.1,
                 risk_measure: str = 'variance',
                 alpha: float = 0.05,
                 linkage_method: Optional[str] = 'ward',
                 distance_metric: Optional[str] = 'euclidean',
                 side_weights: Optional[pd.Series] = None,
                 t_cost: Optional[float] = None,
                 rebal_freq: Optional[Union[str, int]] = None,
                 window_type: str = 'rolling',
                 window_size: Optional[int] = 180,
                 parallelize: bool = True,
                 n_jobs: Optional[int] = -1,
                 asset_names: Optional[List[str]] = None,
                 ann_factor: Optional[int] = None,
                 solver: Optional[str] = None,
                 **kwargs: Any
                 ):
        """
        Constructor

        Parameters
        ----------
        returns : pd.DataFrame or pd.Series
            Asset or strategy returns.
        signals: pd.DataFrame or pd.Series, default None
            Asset signals.
        as_signal_returns: bool, default False
            Whether to compute signal returns for the optimization.
        method: str, {'equal_weight', 'signal_weight', 'inverse_variance', 'inverse_vol', 'target_vol', 'random',
         'min_vol', 'max_return_min_vol', 'max_sharpe', 'max_diversification', 'efficient_return', 'efficient_risk',
          'risk_parity', 'hrp', 'herc'}, default 'equal_weight'
            Optimization method to compute weights.
        lags: int, default 1
            Number of periods to lag weights.
        risk_free_rate: float, default 0.0
            Risk-free rate.
        as_excess_returns: bool, default False
            Whether to compute excess returns.
        max_weight: float, default 1.0
            Maximum weight of the assets or strategies.
        min_weight: float, default 0.0
            Minimum weight of the assets or strategies.
        fully_invested: bool, default True
            Whether the portfolio is fully invested.
        gross_exposure: float, default 1.0
            Gross exposure of the portfolio.
        net_exposure: float, default None
            Net exposure of the portfolio.
        round_weights: bool, default False
            Whether to round the weights.
        risk_aversion: float, default 1.0
            Risk aversion factor.
        exp_ret_method: str, {'historical_mean', 'historical_median', 'rolling_mean', 'rolling_median', 'ewma',
        'rolling_sharpe', 'rolling_sortino'}, default 'historical_mean'
            Method to compute the expected returns.
        cov_matrix_method: str, {'covariance', 'empirical_covariance', 'shrunk_covariance', 'ledoit_wolf', 'oas',
                       'graphical_lasso', 'graphical_lasso_cv', 'minimum_covariance_determinant', 'semi_covariance',
                          'exponential_covariance', 'denoised_covariance'}, default 'covariance'
            Method to compute the covariance matrix.
        target_return: float, default 0.15
            Target return for the optimization.
        target_risk: float, default 0.1
            Target risk for the optimization.
        risk_measure : str, {'equal_weight', 'variance', 'std', 'expected_shortfall', 'conditional_drawdown_risk'},
        default 'std'
            Risk measure to compute the risk contribution.
        alpha: float, default 0.05
            Significance level for the risk measure.
        linkage_method : str, {'single', 'complete', 'average', 'weighted', 'centroid', 'median', 'ward'},
        default 'single'
            Method to compute the distance matrix.
        distance_metric: str, {‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘cityblock’, ‘correlation’, ‘cosine’, ‘dice’,
        ‘euclidean’, ‘hamming’, ‘jaccard’, ‘jensenshannon’, ‘kulczynski1’, ‘mahalanobis’, ‘matching’, ‘minkowski’,
        ‘rogerstanimoto’, ‘russellrao’, ‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’,
        ‘sqeuclidean’, ‘yule'}, default 'euclidean'
            Metric to compute the distance matrix.
        side_weights: pd.Series, default None
            Side weights for the hierarchical optimization.
        t_cost: float, default None
            Transaction costs.
        rebal_freq: str, int, default None
            Rebalancing frequency.
        window_type: str, {'expanding', 'rolling'}, default 'rolling'
            Window type for the optimization.
        window_size: int, default None
            Window size for the optimization.
        parallelize: bool, default False
            Whether to parallelize the optimization.
        n_jobs: int, default None
            Number of jobs to run in parallel.
        asset_names: list, default None
            Names of the assets or strategies.
        ann_factor: int, default None
            Annualization factor.
        solver: str, default None
            Solver for the optimization.
        kwargs: dict
            Additional keyword arguments.
        """
        self.returns = returns
        self.signals = signals
        self.as_signal_returns = as_signal_returns
        self.method = method
        self.lags = lags
        self.risk_free_rate = risk_free_rate
        self.as_excess_returns = as_excess_returns
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.fully_invested = fully_invested
        self.gross_exposure = gross_exposure
        self.net_exposure = net_exposure
        self.round_weights = round_weights
        self.risk_aversion = risk_aversion
        self.exp_ret_method = exp_ret_method
        self.cov_matrix_method = cov_matrix_method
        self.target_return = target_return
        self.target_risk = target_risk
        self.risk_measure = risk_measure
        self.alpha = alpha
        self.linkage_method = linkage_method
        self.distance_metric = distance_metric
        self.side_weights = side_weights
        self.t_cost = t_cost
        self.rebal_freq = rebal_freq
        self.window_type = window_type
        self.window_size = window_size
        self.parallelize = parallelize
        self.n_jobs = n_jobs
        self.asset_names = asset_names
        self.ann_factor = ann_factor
        self.solver = solver
        self.kwargs = kwargs

        self.optimizer = None
        self.signal_returns = None
        self.opt_returns = None
        self.weights = None
        self.t_costs = None
        self.gross_returns = None
        self.net_returns = None
        self.portfolio_returns = pd.DataFrame()  # sum of net returns

        self._preprocess_data()
        self._check_methods()
        self._compute_signal_returns()
        self._get_optimizer(self.opt_returns)

    def _preprocess_data(self) -> None:
        """
        Preprocess the data for the portfolio optimization.
        """
        def validate_data(data, name):
            if not isinstance(data, (pd.DataFrame, pd.Series)):
                raise ValueError(f"{name} must be a pd.DataFrame or pd.Series")

            if isinstance(data, pd.Series) and isinstance(data.index, pd.MultiIndex):
                data = data.unstack().astype('float64')
            elif isinstance(data, pd.DataFrame) and isinstance(data.index, pd.MultiIndex):
                if data.shape[1] > 1:
                    raise ValueError(
                        f"{name} must be a single index pd.DataFrame or "
                        f"a multi-index pd.DataFrame with a single column")
                else:
                    data = data.squeeze().unstack().astype('float64')

            if not isinstance(data.index, pd.DatetimeIndex):
                data.index = pd.to_datetime(data.index)

            return data

        # validate and preprocess returns
        self.returns = validate_data(self.returns, "Returns")

        # validate and preprocess signals
        if self.signals is not None:
            self.signals = validate_data(self.signals, "Signals")

    @classmethod
    def get_available_optimizers(cls):
        """Returns a list of available optimizer names."""
        return cls.available_optimizers

    def _check_methods(self) -> None:
        """
        Check if the method is valid.
        """
        # method
        if self.method not in self.available_optimizers:
            raise ValueError(f"'{self.method}' is not an available optimizer. Choose from {self.available_optimizers}")

    def _compute_signal_returns(self) -> pd.DataFrame:
        """
        Compute signal returns.

        Returns
        -------
        signal_returns: pd.DataFrame
            Signal returns.
        """
        # signal returns
        if self.signals is not None and self.as_signal_returns:

            lagged_signals = self.signals.shift(self.lags)
            self.signal_returns = lagged_signals.mul(self.returns, axis=0).dropna(how='all')
            self.opt_returns = self.signal_returns.copy()

        else:
            self.opt_returns = self.returns.copy()

        return self.opt_returns

    def _get_optimizer(self, returns: Union[pd.DataFrame, pd.Series]) -> Any:
        """
        Optimization algorithm.
        """
        # naive optimization
        if self.method in ['equal_weight', 'signal_weight', 'inverse_variance', 'inverse_vol', 'target_vol', 'random']:
            self.optimizer = NaiveOptimization(returns, signals=self.signals, method=self.method,
                                               exp_ret_method=self.exp_ret_method,
                                               cov_matrix_method=self.cov_matrix_method,
                                               leverage=self.gross_exposure, target_vol=self.target_risk,
                                               ann_factor=self.ann_factor)

        # mean variance optimization
        elif self.method in ['max_return', 'min_vol', 'max_return_min_vol', 'max_sharpe', 'max_diversification',
                             'efficient_return', 'efficient_risk', 'risk_parity']:
            self.optimizer = MVO(returns, method=self.method, max_weight=self.max_weight, min_weight=self.min_weight,
                                 budget=self.gross_exposure, risk_aversion=self.risk_aversion,
                                 risk_free_rate=self.risk_free_rate, as_excess_returns=self.as_excess_returns,
                                 exp_ret_method=self.exp_ret_method, cov_matrix_method=self.cov_matrix_method,
                                 target_return=self.target_return, target_risk=self.target_risk, solver=self.solver,
                                 window_size=self.window_size, ann_factor=self.ann_factor)

        # hierarchical risk parity
        elif self.method == 'hrp':
            self.optimizer = HRP(returns, cov_matrix_method=self.cov_matrix_method, linkage_method=self.linkage_method,
                                 distance_metric=self.distance_metric, side_weights=self.side_weights,
                                 leverage=self.gross_exposure)

        # hierarchical equal risk contributions
        elif self.method == 'herc':
            self.optimizer = HERC(returns, risk_measure=self.risk_measure, alpha=self.alpha,
                                  cov_matrix_method=self.cov_matrix_method, linkage_method=self.linkage_method,
                                  distance_metric=self.distance_metric, leverage=self.gross_exposure)

        else:
            raise ValueError(f"Method is not supported. Valid methods are: {self.available_optimizers}")

        return self.optimizer

    def _compute_fixed_weights(self) -> pd.DataFrame:
        """
        Compute optimal weights.
        """
        # compute weights
        self.weights = self.optimizer.compute_weights()

        return self.weights.astype('float64').fillna(0)

    def _compute_expanding_window_weights(self) -> pd.DataFrame:
        """
        Compute expanding window weights.

        Returns
        -------
        exp_weights: pd.DataFrame
            Expanding weights.
        """
        # dates
        dates = self.optimizer.returns.index[self.window_size - 1:]
        total_tasks = len(dates)

        def compute_weights_for_date(end_date):
            ret_window = self.opt_returns.loc[:end_date]
            self._get_optimizer(ret_window)
            w = self.optimizer.compute_weights()

            return w

        # parallelize optimization with progress tracking
        if self.parallelize:
            results = ProgressParallel(n_jobs=self.n_jobs,
                                     use_tqdm=True,
                                     total=total_tasks,
                                     desc="Computing expanding window weights")(
                delayed(compute_weights_for_date)(date) for date in dates
            )
        else:
            results = []
            for date in tqdm(dates,
                           desc="Computing expanding window weights",
                           total=total_tasks):
                results.append(compute_weights_for_date(date))

        # weights
        self.weights = pd.concat(results).astype('float64').fillna(0)

        return self.weights

    def _compute_rolling_window_weights(self) -> pd.DataFrame:
        """
        Compute rolling window weights with progress tracking.

        Returns
        -------
        rolling_weights: pd.DataFrame
            Rolling weights.
        """
        # dates
        dates = self.optimizer.returns.index[self.window_size - 1:]
        total_tasks = len(dates)

        def compute_weights_for_date(end_date):
            loc_end = self.opt_returns.index.get_loc(end_date)
            ret_window = self.opt_returns.iloc[loc_end + 1 - self.window_size: loc_end + 1]
            self._get_optimizer(ret_window)
            return self.optimizer.compute_weights()

        # parallelize optimization with progress tracking
        if self.parallelize:
            results = ProgressParallel(n_jobs=self.n_jobs,
                                     use_tqdm=True,
                                     total=total_tasks,
                                     desc="Computing rolling window weights")(
                delayed(compute_weights_for_date)(date) for date in dates
            )
        else:
            results = []
            for date in tqdm(dates,
                           desc="Computing rolling window weights",
                           total=total_tasks):
                results.append(compute_weights_for_date(date))

        # weights
        self.weights = pd.concat(results).astype('float64').fillna(0)

        return self.weights

    def compute_weights(self) -> pd.DataFrame:
        """
        Compute optimal weights.
        """
        if self.method in ['equal_weight', 'signal_weight']:
            self._compute_fixed_weights()
        else:
            if self.window_type == 'expanding':
                self.weights = self._compute_expanding_window_weights()
            elif self.window_type == 'rolling':
                self.weights = self._compute_rolling_window_weights()
            else:
                self._compute_fixed_weights()

        return self.weights

    def compute_weighted_signals(self) -> pd.DataFrame:
        """
        Compute weighted signals.

        Returns
        -------
        weighted_signals: pd.DataFrame
            Weighted signals.
        """
        # adjust weights for signals
        if self.signals is not None:
            if self.method != 'signal_weight':
                if isinstance(self.signals.index, pd.MultiIndex):
                    self.weights = self.weights * self.signals.unstack()
                else:
                    self.weights = self.weights * self.signals

        return self.weights

    def adjust_exposure(self) -> pd.DataFrame:
        """
        Adjust weights to meet gross and net exposure constraints.

        Returns
        -------
        weights: pd.DataFrame
            Adjusted weights.
        """
        # gross exposure adjustment
        current_gross_exposure = self.weights.abs().sum(axis=1)
        if self.fully_invested:
            self.weights = self.weights.div(current_gross_exposure, axis=0) * self.gross_exposure

        # net exposure adjustment
        if self.net_exposure is not None:
            current_net_exposure = self.weights.sum(axis=1)
            adjustment = (self.net_exposure - current_net_exposure) / self.weights.shape[1]
            self.weights = self.weights.add(adjustment, axis=0)

        return self.weights

    def compute_rounded_weights(self, threshold: float = 1e-4) -> pd.DataFrame:
        """
        Check and round weights in a rolling weights DataFrame for long/short strategies.

        Parameters
        ----------
        threshold: float, default 1e-4
            Threshold value to round the weights.

        Returns
        -------
        pd.DataFrame
            DataFrame of rounded weights.
        """
        # round down weights to 4 decimal places
        self.weights = np.floor(self.weights.abs() * (1/threshold)) / (1/threshold) * np.sign(self.weights)

        return self.weights

    def rebalance_portfolio(self) -> pd.DataFrame:
        """
        Rebalance portfolio weights.

        Returns
        -------
        signals: pd.DataFrame
            Rebalanced portfolio weights with DatetimeIndex and weights (cols).
        """
        # frequency dictionary
        freq_dict = {'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3, 'friday': 4, 'saturday': 5,
                     'sunday': 6, '15th': 15, 'month_end': 'is_month_end', 'month_start': 'is_month_start'}

        # rebalancing
        if self.rebal_freq is not None:

            w = self.weights.copy()

            # day of the week
            if self.rebal_freq in list(freq_dict.keys())[:7]:
                rebal_df = w[w.index.dayofweek == freq_dict[self.rebal_freq]]
            # mid-month
            elif self.rebal_freq == '15th':
                rebal_df = w[w.index.day == 15]
            # fixed period
            elif isinstance(self.rebal_freq, int):
                rebal_df = w.iloc[::self.rebal_freq, :]
            # month start, month end
            else:
                rebal_df = w[getattr(w.index, freq_dict[self.rebal_freq])]

            # reindex and forward fill
            self.weights = rebal_df.reindex(w.index).ffill().dropna(how='all')

            # replace forward filled values with last valid observation
            self.weights = self.weights * np.sign(w.abs()).astype('Int64')

            # convert to float
            self.weights = self.weights.astype('float64')

        return self.weights

    def compute_tcosts(self) -> pd.DataFrame:
        """
        Computes transactions costs from changes in weights.

        Returns
        -------
        t_costs: pd.Series
            Series with DatetimeIndex (level 0), tickers (level 1) and transaction costs (cols).
        """
        # no t-costs
        if self.t_cost is None:
            self.t_costs = pd.DataFrame(data=0.0, index=self.weights.index, columns=self.weights.columns)
        # t-costs
        else:
            self.t_costs = self.weights.diff().abs() * self.t_cost

        return self.t_costs

    def compute_gross_returns(self) -> pd.DataFrame:
        """
        Compute gross returns.

        Returns
        -------
        gross_returns: pd.DataFrame
            Gross returns.
        """
        if isinstance(self.returns.index, pd.MultiIndex):
            self.gross_returns = self.weights * self.returns.unstack()
        else:
            self.gross_returns = self.weights * self.returns

        return self.gross_returns

    def compute_net_returns(self) -> pd.DataFrame:
        """
        Compute net returns.

        Returns
        -------
        net_returns: pd.DataFrame
            Net returns.
        """
        self.net_returns = self.gross_returns.subtract(self.t_costs, axis=0).dropna(how='all')

        return self.net_returns

    def compute_portfolio_returns(self, portfolio_name: Optional[str] = None) -> pd.DataFrame:
        """
        Compute portfolio returns for a single index dataframe of asset/strategy returns.

        Parameters
        ----------
        portfolio_name: str, default None
            Portfolio name.

        Returns
        -------
        portfolio_ret: pd.DataFrame
            Portfolio returns.
        """
        # get weights
        self.compute_weights()

        # compute weighted signals
        self.compute_weighted_signals()

        # adjust exposure
        self.adjust_exposure()

        # round weights
        if self.round_weights:
            self.compute_rounded_weights()

        # lag weights
        self.weights = self.weights.shift(self.lags)

        # rebalance portfolio
        self.rebalance_portfolio()

        # t-costs
        self.compute_tcosts()

        # compute gross returns
        self.compute_gross_returns()

        # compute net returns
        self.compute_net_returns()

        # compute portfolio returns
        if portfolio_name is None:
            portfolio_name = 'portfolio'

        self.portfolio_returns = self.net_returns.sum(axis=1).to_frame(portfolio_name)

        return self.portfolio_returns

    def plot_weights(self, plot_type: str = 'bar'):
        """
        Plot the optimized portfolio weights.

        Parameters
        ----------
        plot_type: str, {'series', 'bar'}, default 'bar'
            Type of plot.
        """
        # plot h bar
        if plot_type == 'bar':
            plot_bar(self.weights.T.sort_values(by=[self.weights.index[-1]]), axis='horizontal', x_label='weights')
        # plot series
        else:
            plot_series(self.weights, title='Portfolio Weights', y_label='weights')
