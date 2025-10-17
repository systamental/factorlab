import pandas as pd
import numpy as np
from typing import Optional, Union, Any
from statsmodels.api import OLS
from statsmodels.tsa.tsatools import add_trend

from factorlab.feature_engineering.transformations import Transform
from factorlab.signal_generation.time_series_analysis import rolling_window, expanding_window

from factorlab.signal_generation.supervised_learning import SPCA


class FMP:
    """
    Factor mimicking portfolio.
    """
    def __init__(self,
                 returns: Union[pd.DataFrame, pd.Series, np.array],
                 factors: Union[pd.DataFrame, np.array],
                 ann_factor: Optional[int] = None
                 ):
        """
        Initialize FMP object.

        Parameters
        ----------
        returns: pd.DataFrame, pd.Series or np.ndarray
            Base returns.
        factors: pd.DataFrame or np.ndarray
            Dataframe or numpy array of factors.
        ann_factor: int, default=None
            Annualization factor.
        """
        self.returns = returns
        self.factors = factors
        self.ann_factor = ann_factor
        self.pct_chg = None
        self.factors_pred = pd.DataFrame()
        self.betas = None
        self.weights = None

        self.data = None
        self.index = None
        self.factor_cols = None
        self.return_cols = None

    def orthogonalize_factors(self, window_type: str = 'fixed', min_obs: int = 12, window_size: int = 36) -> None:
        """
        Orthogonalize factors.

        Parameters
        ----------
        window_type: str, {'fixed', 'expanding', 'rolling'}, default='expanding'
            Type of window to use for orthogonalization.
        min_obs: int, default=12
            Minimum number of observations for expanding window.
        window_size: int, default=36
            Window size for rolling window.
        """
        if window_type == 'rolling':
            self.factors = rolling_window(Transform, self.factors, method='orthogonalize', window_size=window_size)
        elif window_type == 'expanding':
            self.factors = expanding_window(Transform, self.factors, method='orthogonalize', min_obs=min_obs)
        else:
            self.factors = Transform(self.factors).orthogonalize()

    def adj_factor_vol(self, ann_vol: int = 0.15) -> pd.DataFrame:
        """
        Adjust factor volatility to target vol.

        Parameters
        ----------
        ann_vol: int, default=0.15
            Annualized volatility target.

        Returns
        -------
        factor_vol_adj: pd.DataFrame
            Factor vol adjusted to target volatility.
        """
        # ann factor
        self.get_ann_factor()
        # adj to target vol`
        self.factors = Transform(self.factors).target_vol(ann_vol=ann_vol, ann_factor=self.ann_factor)

    def preprocess_data(self) -> None:
        """
        Preprocess data.
        """
        if isinstance(self.returns, pd.Series) or isinstance(self.returns, pd.DataFrame) and \
                isinstance(self.factors, pd.DataFrame):
            self.data = pd.concat([self.returns, self.factors], axis=1).dropna()
            self.index = self.data.index
            self.factors = self.data.iloc[:, 1:]
            self.returns = self.data.iloc[:, 0]
        elif isinstance(self.returns, np.ndarray) and isinstance(self.factors, np.ndarray):
            n = min(self.returns.shape[0], self.factors.shape[0])
            self.data = np.concatenate([self.returns[:n].reshape(-1, 1), self.factors[:n].reshape(-1, 1)], axis=1)
            self.factors = self.data[:, 1:]
            self.returns = self.data[:, 0]
        else:
            raise TypeError("Target and features must be a pandas Series, DataFrame or np.array.")

    def get_ann_factor(self) -> str:
        """
        Get annualization factor.

        Returns
        -------
        ann_factor: int
            Annualization factor.
        """
        if self.ann_factor is None and isinstance(self.returns, pd.DataFrame) or isinstance(self.returns, pd.Series):
            # infer freq
            if isinstance(self.returns, pd.MultiIndex):
                freq = pd.infer_freq(self.returns.index.levels[0])
            else:
                freq = pd.infer_freq(self.returns.index)

            # get ann factor
            if freq == 'D':
                self.ann_factor = 252
            elif freq == 'W':
                self.ann_factor = 52
            elif freq == 'M':
                self.ann_factor = 12
            elif freq == 'Q':
                self.ann_factor = 4
            elif freq == 'Y':
                self.ann_factor = 1

    def convert_to_pct_chg(self, periods: Optional[int] = None) -> pd.DataFrame:
        """
        Convert returns to percent changes with overlapping windows.

        Parameters
        ----------
        periods: int, default=None
            Number of periods to shift to convert returns to percent changes with overlapping windows.
        """
        if periods is None:
            periods = self.ann_factor
        self.pct_chg = (1 + self.returns).cumprod().pct_change(periods=periods)

    def add_lags(self, n_lags: int = 24) -> pd.DataFrame:
        """
        Add lags to pct changes.

        Parameters
        ----------
        n_lags: int, default=24
            Number of lags to add to pct changes.
        """
        # convert to pct chg
        self.convert_to_pct_chg()
        self.pct_chg = add_lags(self.pct_chg, n_lags=n_lags)

    def predict_factors(self,
                        window_type: str = 'expanding',
                        ann_vol: int = 0.15,
                        periods: Optional[int] = None,
                        n_lags: int = 24,
                        fwd: int = 0
                        ) -> pd.DataFrame:
        """
        Predict factors.

        Parameters
        ----------
        window_type: str, {'fixed', 'expanding', 'rolling'}, default='expanding'
            Type of window to use for orthogonalization.
        ann_vol: int, default=0.15
            Annualized volatility target.
        periods: int, default=None
            Number of periods to shift to convert returns to percent changes with overlapping windows.
        n_lags: int, default=24
            Number of lags to add to returns.
        fwd: int, default=0
            Number of periods to shift returns forward.
        """
        # orthogonalize factors
        self.orthogonalize_factors(window_type=window_type)
        # adj factors to target vol
        self.adj_factor_vol(ann_vol=ann_vol)
        # convert returns to pct change
        self.convert_to_pct_chg(periods=periods)
        # add lags to pct chg
        self.add_lags(n_lags=n_lags)

        # iterate over factors
        for factor in self.factors.columns:
            # targeted pca
            pcs = SPCA(self.factors[factor].shift(fwd*-1), self.pct_chg, n_feat=30).get_expanding_pcs()
            # predict factor
            pred = linear_reg(self.factors[factor].shift(fwd*-1), pcs, window_type='expanding')
            # add to df
            self.factors_pred = pd.concat([self.factors_pred, pred.rename(columns={'pred': factor})], axis=1)

    def factor_exposures(self) -> pd.DataFrame:
        """
        Estimate factor betas.
        """
        # create df to store betas
        date = self.returns.index[-1]
        idx = pd.MultiIndex.from_product([[date], self.returns.columns], names=['date', 'ticker'])
        factor_cols = self.factors_pred.columns.to_list() + ['const']
        betas_df = pd.DataFrame(index=idx, columns=factor_cols)
        # iterate over ret columns
        for col in self.returns.columns:
            betas = linear_reg(self.returns[col], self.factors_pred, output='coef')
            betas_df.loc[(date, col), :] = betas.values.T
        # set betas
        self.betas = betas_df

    def get_portfolio_weights(self, method: str = 'ml'):
        """

        Parameters
        ----------
        method: str, {'ols-csr', 'wls-csr', 'mcp', 'ml'}, default='ml'

        Returns
        -------

        """
        # get betas
        self.factor_exposures()
        if isinstance(self.betas, pd.DataFrame):
            B = self.betas.to_numpy(dtype=np.float64)[:, :-1]
        else:
            B = self.betas[:, :-1]

        # cov matrix of returns
        cov_mat = np.cov(self.returns, rowvar=False)
        inv_cov = np.linalg.pinv(cov_mat)

        # compute weights
        W = None
        if method == 'ols-csr':
            W = B @ np.linalg.pinv(B.T @ B)
        elif method == 'wls-csr':
            pass
        elif method == 'mcp':
            pass
        elif method == 'ml':
            W = inv_cov @ B @ np.linalg.pinv(B.T @ inv_cov @ B) @ np.identity(B.shape[1])

        self.weights = W

    def get_fmp(self):
        """
        Compute FMP of factors.
        """
        # get weights
        self.get_portfolio_weights()
        # compute FMPs
        rets = self.weights.T @ self.returns.T
        rets = rets.T

        return pd.DataFrame(rets, index=self.index, columns=self.factors.columns)


class PortfolioAnalysis:
    """
    Factor models for the analysis for alpha or risk factors.
    """
    def __init__(self,
                 returns: Union[pd.DataFrame, pd.Series, np.array],
                 factors: Union[pd.DataFrame, np.array],
                 ann_factor: Optional[int] = None
                 ):
        """
        Initialize FactorModel object.

        Parameters
        ----------
        factors: pd.DataFrame or np.array
            Dataframe or numpy array of factors.
        """
        self.returns = returns
        self.factors = factors
        self.ann_factor = ann_factor

    def get_correl_matrix(self) -> pd.DataFrame:
        """
        Compute correlation matrix of factors.

        Returns
        -------
        corr_matrix: pd.DataFrame
            Correlation matrix of factors.
        """
        return pd.DataFrame(self.factors).corr()

    def get_cov_matrix(self) -> pd.DataFrame:
        """
        Compute covariance matrix of factors.

        Returns
        -------
        cov_matrix: pd.DataFrame
            Covariance matrix of factors.
        """
        return pd.DataFrame(self.factors).cov()

    def return_attribution(self) -> pd.DataFrame:
        """
        Compute return attribution.

        Returns
        -------
        return_attribution: pd.DataFrame
            Dataframe of return attribution.
        """
        pass

    def risk_attribution(self) -> pd.DataFrame:
        """
        Compute risk attribution.

        Returns
        -------
        risk_attribution: pd.DataFrame
            Dataframe of risk attribution.
        """
        pass

    def beta_returns(self) -> pd.DataFrame:
        """
        Compute beta returns.

        Returns
        -------
        beta_returns: pd.DataFrame
            Dataframe of beta returns.
        """
        pass

    def alpha_returns(self) -> pd.DataFrame:
        """
        Compute alpha returns.

        Returns
        -------
        residuals: pd.DataFrame
            Dataframe of residuals.
        """
        pass
