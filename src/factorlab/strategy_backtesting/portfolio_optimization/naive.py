import pandas as pd
import numpy as np
from typing import Optional


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
                 cov_matrix_method: Optional[str] = None,
                 ann_factor: Optional[int] = None,
                 leverage: Optional[float] = None,
                 target_vol: Optional[float] = None,
                 window_size: Optional[int] = None
                 ):
        """
        Constructor

        Parameters
        ----------
        returns: pd.DataFrame or pd.Series
            The returns of the assets or strategies. If not provided, the returns are computed from the prices.
        method: str, {'equal', 'inverse_volatility'}, default 'equal'
            Optimization method to compute weights.
        asset_names: list, default None
            Names of the assets or strategies.
        cov_matrix_method: str, default None
            Method to compute the covariance matrix.
        ann_factor: int, default None
            Annualization factor.
        leverage: float, default None
            Leverage factor.
        target_vol: float, default None
            Target volatility for portfolio returns.
        window_size: int, default None
            Window size for volatility computation.
        """
        self.returns = returns
        self.method = method
        self.asset_names = asset_names
        self.cov_matrix_method = cov_matrix_method
        self.ann_factor = ann_factor
        self.leverage = leverage
        self.target_vol = target_vol
        self.window_size = window_size
        self.n_assets = None
        self.freq = None
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
        self.returns.index = pd.to_datetime(self.returns.index)  # convert to index to datetime
        self.returns = self.returns.dropna()  # drop na

        # method
        if self.method not in ['equal_weight', 'inverse_vol']:
            raise ValueError("Method is not supported. Valid methods are: 'equal_weight', 'inverse_vol'")

        # asset names
        if self.asset_names is None:
            self.asset_names = self.returns.columns.tolist()

        # n_assets
        if self.n_assets is None:
            self.n_assets = self.returns.shape[1]

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

        # cov matrix
        if self.cov_matrix is None:
            self.cov_matrix = self.returns.cov().values

        # leverage
        if self.leverage is None:
            self.leverage = 1.0

        # target_vol
        if self.target_vol is None:
            self.target_vol = 0.15

        # window_size
        if self.window_size is None:
            self.window_size = self.ann_factor

    def compute_equal_weight(self):
        """
        Compute equal weights for assets or strategies.
        """
        self.weights = np.ones(self.n_assets) * 1 / self.n_assets
        # compute portfolio risk and return
        self.portfolio_ret = np.dot(self.weights, self.returns.mean())
        self.portfolio_risk = np.sqrt(np.dot(self.weights.T, np.dot(self.returns.cov(), self.weights)))
        self.weights = pd.DataFrame(self.weights, index=self.asset_names, columns=['weight']).T

        return self.weights

    def compute_inverse_variance(self) -> pd.DataFrame:
        """
        Compute inverse variance weights for assets or strategies.

        Returns
        -------
        pd.DataFrame
            The inverse variance weights.
        """
        # compute inverse variance weights
        ivp = 1. / np.diag(self.cov_matrix)
        ivp /= ivp.sum()
        self.weights = ivp
        # compute portfolio risk and return
        self.portfolio_risk = np.dot(self.weights, np.dot(self.cov_matrix, self.weights.T))
        self.portfolio_ret = np.dot(self.weights, self.returns.mean())
        self.weights = pd.DataFrame(self.weights, index=self.asset_names, columns=['weight']).T

        return self.weights * self.leverage
