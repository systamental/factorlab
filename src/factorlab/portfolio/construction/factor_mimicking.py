import pandas as pd
import numpy as np
from typing import Optional, Union
from sklearn.linear_model import LinearRegression

from factorlab.learning.selectors import FeatureSelector
from factorlab.learning.time_series_analysis import add_lags
from factorlab.learning.unsupervised.unsupervised_learning import PCAWrapper


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

    @staticmethod
    def _orthogonalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """
        Orthogonalize factor columns via sequential residualization.
        """
        out = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)
        cols = list(df.columns)
        for i, col in enumerate(cols):
            y = df[col].astype(float)
            if i == 0:
                out[col] = y
                continue

            X_prev = out.iloc[:, :i]
            valid = X_prev.notna().all(axis=1) & y.notna()
            if int(valid.sum()) <= i:
                out[col] = y
                continue

            beta = np.linalg.lstsq(X_prev.loc[valid].to_numpy(), y.loc[valid].to_numpy(), rcond=None)[0]
            pred = pd.Series(np.nan, index=df.index, dtype=float)
            pred.loc[valid] = X_prev.loc[valid].to_numpy() @ beta
            out[col] = y - pred
        return out

    @staticmethod
    def _target_vol(df: pd.DataFrame, ann_vol: float, ann_factor: int) -> pd.DataFrame:
        """
        Scale each series to the target annualized volatility.
        """
        if ann_factor is None or ann_factor <= 0:
            return df
        vol = df.std(ddof=0) * np.sqrt(float(ann_factor))
        scale = pd.Series(1.0, index=df.columns, dtype=float)
        valid = vol > 0
        scale.loc[valid] = float(ann_vol) / vol.loc[valid]
        return df.mul(scale, axis=1)

    @staticmethod
    def _expanding_linear_prediction(
        target: pd.Series,
        features: pd.DataFrame,
        min_obs: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Generate expanding-window one-step-ahead predictions.
        """
        df = pd.concat([target.rename("target"), features], axis=1).dropna()
        if df.empty:
            return pd.DataFrame(columns=["pred"], index=target.index, dtype=float)

        if min_obs is None:
            min_obs = max(2, features.shape[1] + 1)
        min_obs = max(2, int(min_obs))
        if min_obs >= len(df):
            return pd.DataFrame(columns=["pred"], index=df.index, dtype=float)

        pred = pd.Series(np.nan, index=df.index, dtype=float)
        for row in range(min_obs, len(df)):
            train = df.iloc[:row]
            X_train = train.iloc[:, 1:]
            y_train = train.iloc[:, 0]
            X_next = df.iloc[[row], 1:]
            model = LinearRegression()
            model.fit(X_train, y_train)
            pred.iloc[row] = float(model.predict(X_next)[0])

        return pred.to_frame(name="pred")

    @staticmethod
    def _supervised_expanding_pcs(
        target: pd.Series,
        features: pd.DataFrame,
        n_feat: int = 30,
        method: str = "lasso",
    ) -> pd.DataFrame:
        """
        Select top predictive features and compute expanding PCs.
        """
        target_name = target.name if target.name is not None else "target"
        df = pd.concat([features, target.rename(target_name)], axis=1).dropna()
        if df.empty:
            return pd.DataFrame(index=features.index)

        selector = FeatureSelector(
            method=method,
            feature_cols=list(features.columns),
            target_col=target_name,
            n_features=min(int(n_feat), int(features.shape[1])),
            drop_unselected=True,
        )
        selector.fit(df)
        selected = selector.transform(df).drop(columns=[target_name], errors="ignore")
        if selected.empty:
            return pd.DataFrame(index=df.index)

        n_components = max(1, min(selected.shape[0], selected.shape[1]))
        min_obs = max(2, n_components)
        pcs = PCAWrapper(selected, n_components=n_components).get_expanding_pcs(min_obs=min_obs)
        if isinstance(pcs, np.ndarray):
            pcs = pd.DataFrame(pcs, index=selected.index[-pcs.shape[0]:])
        pcs.columns = [f"PC{i + 1}" for i in range(pcs.shape[1])]
        return pcs

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
        factors_df = self.factors.copy(deep=True) if isinstance(self.factors, pd.DataFrame) else pd.DataFrame(self.factors)

        if window_type == 'rolling':
            if window_size < 1 or window_size > len(factors_df):
                raise ValueError("window_size must be within [1, len(factors)].")
            rows = []
            idx = []
            for row in range(window_size, len(factors_df) + 1):
                window_df = factors_df.iloc[row - window_size: row]
                orth_window = self._orthogonalize_dataframe(window_df)
                rows.append(orth_window.iloc[-1])
                idx.append(window_df.index[-1])
            self.factors = pd.DataFrame(rows, index=idx, columns=factors_df.columns)
        elif window_type == 'expanding':
            if min_obs < 1 or min_obs > len(factors_df):
                raise ValueError("min_obs must be within [1, len(factors)].")
            rows = []
            idx = []
            for row in range(min_obs, len(factors_df) + 1):
                window_df = factors_df.iloc[:row]
                orth_window = self._orthogonalize_dataframe(window_df)
                rows.append(orth_window.iloc[-1])
                idx.append(window_df.index[-1])
            self.factors = pd.DataFrame(rows, index=idx, columns=factors_df.columns)
        else:
            self.factors = self._orthogonalize_dataframe(factors_df)

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
        factors_df = self.factors.copy(deep=True) if isinstance(self.factors, pd.DataFrame) else pd.DataFrame(self.factors)
        self.factors = self._target_vol(factors_df, ann_vol=ann_vol, ann_factor=int(self.ann_factor))

    def preprocess_data(self) -> None:
        """
        Preprocess data.
        """
        if isinstance(self.returns, (pd.Series, pd.DataFrame)) and isinstance(self.factors, pd.DataFrame):
            returns_df = self.returns.to_frame() if isinstance(self.returns, pd.Series) else self.returns.copy(deep=True)
            factors_df = self.factors.copy(deep=True)
            self.return_cols = list(returns_df.columns)
            self.factor_cols = list(factors_df.columns)

            self.data = pd.concat([returns_df, factors_df], axis=1).dropna()
            self.index = self.data.index
            self.returns = self.data.loc[:, self.return_cols]
            self.factors = self.data.loc[:, self.factor_cols]
        elif isinstance(self.returns, np.ndarray) and isinstance(self.factors, np.ndarray):
            n = min(self.returns.shape[0], self.factors.shape[0])
            factors_arr = self.factors[:n]
            if factors_arr.ndim == 1:
                factors_arr = factors_arr.reshape(-1, 1)
            self.data = np.concatenate([self.returns[:n].reshape(-1, 1), factors_arr], axis=1)
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
        if self.ann_factor is None and isinstance(self.returns, (pd.DataFrame, pd.Series)):
            # infer freq
            if isinstance(self.returns.index, pd.MultiIndex):
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
        if self.pct_chg is None:
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
        # align and clean data
        self.preprocess_data()

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
            target = self.factors[factor].shift(-int(fwd)) if int(fwd) > 0 else self.factors[factor]
            pcs = self._supervised_expanding_pcs(
                target=target,
                features=self.pct_chg,
                n_feat=30,
                method="lasso",
            )
            if pcs.empty:
                continue

            pred = self._expanding_linear_prediction(
                target=target.reindex(pcs.index),
                features=pcs,
                min_obs=max(2, pcs.shape[1] + 1),
            )
            self.factors_pred = pd.concat([self.factors_pred, pred.rename(columns={"pred": factor})], axis=1)

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
            df = pd.concat([self.returns[col].rename("ret"), self.factors_pred], axis=1).dropna()
            if df.empty:
                continue
            X = df.iloc[:, 1:]
            y = df.iloc[:, 0]
            model = LinearRegression()
            model.fit(X, y)
            betas_df.loc[(date, col), self.factors_pred.columns] = model.coef_
            betas_df.loc[(date, col), "const"] = model.intercept_
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
    Factor models for the analytics for alpha or risk factors.
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
