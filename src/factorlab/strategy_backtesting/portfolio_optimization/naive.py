import pandas as pd
import numpy as np
from typing import Optional

from factorlab.strategy_backtesting.portfolio_optimization.return_estimators import ReturnEstimators
from factorlab.strategy_backtesting.portfolio_optimization.risk_estimators import RiskEstimators
from factorlab.data_viz.plot import plot_bar


class NaiveOptimization:
    """
    Naive portfolio optimization class.

    This class computes the optimized portfolio weights based on the returns of the assets or strategies
    using naive optimization techniques.
    """
    def __init__(self,
                 returns: pd.DataFrame,
                 method: str = 'equal_weight',
                 asset_names: Optional[str] = None,
                 leverage: Optional[float] = None,
                 target_vol: Optional[float] = None,
                 ann_factor: Optional[int] = None,
                 ):
        """
        Constructor

        Parameters
        ----------
        returns: pd.DataFrame or pd.Series
            The returns of the assets or strategies. If not provided, the returns are computed from the prices.
        method: str, {'equal_weight', 'inverse_variance', 'inverse_vol', 'target_vol', 'random'},
        default 'equal_weight'
            Optimization method to compute weights.
        asset_names: list, default None
            Names of the assets or strategies.
        leverage: float, default None
            Leverage factor.
        target_vol: float, default None
            Target volatility for portfolio returns.
        ann_factor: int, default None
            Annualization factor.
        """
        self.returns = returns
        self.method = method
        self.asset_names = asset_names
        self.leverage = leverage
        self.target_vol = target_vol
        self.ann_factor = ann_factor
        self.freq = None
        self.n_assets = None
        self.exp_ret = None
        self.cov_matrix = None
        self.weights = None
        self.portfolio_ret = None
        self.portfolio_risk = None
        self.preprocess_data()

    def preprocess_data(self) -> None:
        """
        Preprocess the data for the portfolio optimization.
        """
        # returns
        if not isinstance(self.returns, pd.DataFrame) and not isinstance(self.returns, pd.Series):  # check data type
            raise ValueError('rets must be a pd.DataFrame or pd.Series')
        if isinstance(self.returns, pd.Series):  # convert to df
            self.returns = self.returns.to_frame()
        if isinstance(self.returns.index, pd.MultiIndex):  # convert to single index
            self.returns = self.returns.unstack()
        if not isinstance(self.returns.index, pd.DatetimeIndex):  # convert to single index
            self.returns.index = pd.to_datetime(self.returns.index)  # convert to index to datetime
        # remove missing vals
        last_row = self.returns.iloc[-1]  # select the last row
        columns_to_drop = last_row[last_row.isna()].index  # cols with NaN values
        self.returns = self.returns.drop(columns=columns_to_drop).dropna(how='all')  # drop missing cols and emtpy rows

        # method
        if self.method not in ['equal_weight', 'inverse_variance', 'inverse_vol', 'target_vol', 'random']:
            raise ValueError("Method is not supported. Valid methods are: 'equal_weight', 'inverse_variance', "
                             "'inverse_vol', 'target_vol, 'random'")

        # asset names
        if self.asset_names is None:
            self.asset_names = self.returns.columns.tolist()

        # n_assets
        if self.n_assets is None:
            self.n_assets = self.returns.shape[1]

        # leverage
        if self.leverage is None:
            self.leverage = 1.0

        # target_vol
        if self.target_vol is None:
            self.target_vol = 0.15

        # ann_factor
        if self.ann_factor is None:
            self.ann_factor = self.returns.groupby(self.returns.index.year).count().max().mode()[0]

        # freq
        self.freq = pd.infer_freq(self.returns.index)
        if self.freq is None:
            if self.ann_factor == 1:
                self.freq = 'Y'
            elif self.ann_factor == 4:
                self.freq = 'Q'
            elif self.ann_factor == 12:
                self.freq = 'M'
            elif self.ann_factor == 52:
                self.freq = 'W'
            else:
                self.freq = 'D'

    def compute_estimators(self) -> None:
        """
        Compute estimators.
        """
        # expected returns
        self.exp_ret = ReturnEstimators(self.returns, method='mean').compute_expected_returns().values

        # covariance matrix
        self.cov_matrix = RiskEstimators(self.returns).compute_covariance_matrix(method='covariance')

    def compute_equal_weight(self):
        """
        Compute equal weights for assets or strategies.
        """
        # estimators
        self.compute_estimators()

        # weights
        self.weights = ((np.sign(self.returns.abs()).iloc[-1]) /
                        (np.sign(self.returns.abs().iloc[-1].dropna()).sum())).values
        self.weights = self.weights * self.leverage

        # portfolio risk and return
        self.portfolio_ret = np.dot(self.weights, self.exp_ret)
        self.portfolio_risk = np.dot(self.weights.T, np.dot(self.cov_matrix, self.weights))

        return self.weights

    def compute_inverse_variance(self) -> pd.DataFrame:
        """
        Compute inverse variance weights for assets or strategies.

        Returns
        -------
        pd.DataFrame
            The inverse variance weights.
        """
        # estimators
        self.compute_estimators()

        # weights
        diag = np.diag(self.cov_matrix)
        diag = np.where(diag == 0, np.nan, diag)  # avoid division by zero
        ivp = 1. / diag
        ivp /= np.nansum(ivp)
        self.weights = ivp
        self.weights = self.weights * self.leverage

        # portfolio risk and return
        self.portfolio_ret = np.dot(self.weights, self.exp_ret)
        self.portfolio_risk = np.dot(self.weights.T, np.dot(self.cov_matrix, self.weights))

        return self.weights

    def compute_inverse_vol(self) -> pd.DataFrame:
        """
        Compute inverse volatility weights for assets or strategies.

        Returns
        -------
        pd.DataFrame
            The inverse variance weights.
        """
        # estimators
        self.compute_estimators()

        # weights
        sqrt_diag = np.sqrt(np.diag(self.cov_matrix))
        sqrt_diag = np.where(sqrt_diag == 0, np.nan, sqrt_diag)  # avoid division by zero
        ivp = 1. / sqrt_diag
        ivp /= np.nansum(ivp)
        self.weights = ivp
        self.weights = self.weights * self.leverage

        # portfolio risk and return
        self.portfolio_ret = np.dot(self.weights, self.exp_ret)
        self.portfolio_risk = np.dot(self.weights.T, np.dot(self.cov_matrix, self.weights))

        return self.weights

    def compute_target_vol(self) -> pd.DataFrame:
        """
        Compute target volatility weights for assets or strategies.

        Returns
        -------
        pd.DataFrame
            The target volatility weights.
        """
        # compute target volatility weights
        self.compute_inverse_vol()

        # compute annualized volatility
        ann_vol = np.sqrt(self.portfolio_risk) * np.sqrt(self.ann_factor)

        # adjust weights by target volatility
        self.leverage = self.target_vol / ann_vol
        self.weights = self.weights * self.leverage

        # portfolio risk and return
        self.portfolio_ret = np.dot(self.weights, self.exp_ret)
        self.portfolio_risk = np.dot(self.weights.T, np.dot(self.cov_matrix, self.weights))

        return self.weights

    def compute_random(self) -> pd.DataFrame:
        """
        Compute random weights for assets or strategies using Dirichlet distribution.

        Returns
        -------
        pd.DataFrame
            The random weights.
        """
        # estimators
        self.compute_estimators()

        # random weights
        self.weights = np.random.dirichlet(np.ones(self.n_assets))
        self.weights = self.weights * self.leverage

        # portfolio risk and return
        self.portfolio_ret = np.dot(self.weights, self.exp_ret)
        self.portfolio_risk = np.dot(self.weights.T, np.dot(self.cov_matrix, self.weights))

        return self.weights

    def compute_weights(self) -> pd.DataFrame:
        """
        Compute the portfolio weights based on the optimization method.

        Returns
        -------
        pd.DataFrame
            The portfolio weights.
        """
        # compute weights
        if self.method in ['equal_weight', 'inverse_variance', 'inverse_vol', 'target_vol', 'random']:
            getattr(self, f'compute_{self.method}')()
        else:
            raise ValueError("Method is not supported. Valid methods are: 'equal_weight', 'inverse_variance', "
                             "'inverse_vol', 'target_vol', 'random'")

        self.weights = pd.DataFrame(self.weights, index=self.asset_names, columns=[self.returns.index[-1]]).T

        return self.weights

    def plot_weights(self):
        """
        Plot the optimized portfolio weights.
        """
        plot_bar(self.weights.T.sort_values(by=[self.returns.index[-1]]), axis='horizontal', x_label='weights')
