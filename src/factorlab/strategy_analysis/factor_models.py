import pandas as pd
import numpy as np
from typing import Optional, Union, Any
from statsmodels.api import OLS
from statsmodels.tsa.tsatools import add_trend

from factorlab.feature_engineering.transformations import Transform
from factorlab.signal_generation.time_series_analysis import rolling_window, expanding_window


class FactorModel:
    """
    Factor model for the analysis for alpha or risk factors.
    """
    def __init__(self,
                 ret: Union[pd.DataFrame, pd.Series, np.array],
                 factors: Union[pd.DataFrame, pd.Series, np.array],
                 strategy: str = 'ts',
                 ann_factor: Optional[int] = None,
                 window_type: str = 'fixed',
                 window_size: int = 365,
                 normalize: bool = False,
                 orthogonalize: bool = False
                 ):
        """
        Initialize FactorModel object.

        Parameters
        ----------
        ret: pd.DataFrame or np.ndarray
            Dataframe or numpy array of asset returns.
        factors: pd.DataFrame or np.ndarray
            Dataframe or numpy array of factors.
        ann_factor: int, default=None
            Annualization factor.
        window_type: str, {'fixed', 'expanding', 'rolling'}, default='fixed'
            Type of window to use for normalization or orthogonalization.
        window_size: int, default=365
            Window size for rolling window.
        normalize: bool, default=False
            Normalize factors and returns.
        orthogonalize: bool, default=False
            Orthogonalize factors.
        """
        self.ret = ret
        self.factors = factors
        self.strategy = strategy
        self.window_type = window_type
        self.window_size = window_size
        self.normalize = normalize
        self.orthogonalize = orthogonalize
        self.freq = None
        self.ann_factor = ann_factor
        self.data = None
        self.index = None
        self.model = None
        self.results = None
        self.preprocess_data()
        self.get_ann_factor()
        self.normalize_data()
        self.orthogonalize_factors()

    def preprocess_data(self) -> None:
        """
        Preprocess data.
        """
        # returns
        if isinstance(self.ret, pd.Series):
            self.ret = self.ret.to_frame().copy()
        # factors
        if isinstance(self.factors, pd.Series):
            self.factors = self.factors.to_frame().copy()

        # concat ret and factors
        self.data = pd.concat([self.ret, self.factors], axis=1).dropna()
        self.ret = self.data.iloc[:, 0].to_frame()
        self.factors = self.data.iloc[:, 1:]

        # index
        self.index = self.data.index
        if isinstance(self.index, pd.MultiIndex):
            if not isinstance(self.index.levels[0], pd.DatetimeIndex):
                self.index = self.index.set_levels(pd.to_datetime(self.index.levels[0]), level=0)
        else:
            self.index = pd.to_datetime(self.index)

        # freq
        if isinstance(self.index, pd.MultiIndex):
            self.freq = pd.infer_freq(self.index.get_level_values(0).unique())
        else:
            self.freq = pd.infer_freq(self.index)

    def get_ann_factor(self) -> str:
        """
        Get annualization factor.

        Returns
        -------
        ann_factor: int
            Annualization factor.
        """
        # get ann factor
        if self.ann_factor is None and self.freq is not None:
            if self.freq == 'D':
                self.ann_factor = 365
            elif self.freq == 'W':
                self.ann_factor = 52
            elif self.freq == 'M':
                self.ann_factor = 12
            elif self.freq == 'Q':
                self.ann_factor = 4
            elif self.freq == 'Y':
                self.ann_factor = 1

    def normalize_data(self,
                       method: str = 'z-score',
                       centering: bool = True,
                       ts_norm: bool = False,
                       winsorize: Optional[int] = None
                       ) -> pd.DataFrame:
        """
        Normalize factors and/or targets.

        Parameters
        ---------
        method: str, {'z-score', 'iqr', 'mod_z', 'min-max', 'percentile'}, default 'z-score'
            z-score: subtracts mean and divides by standard deviation.
            iqr:  subtracts median and divides by interquartile range.
            mod_z: modified z-score using median absolute deviation.
            min-max: rescales to values between 0 and 1 by subtracting the min and dividing by the range.
            percentile: rescales to values between 0 and 1 by computing the percentile rank.
        centering: bool, default True
            Centers values using the appropriate measure of central tendency used for the selected method. Otherwise,
            0 is used.
        ts_norm: bool, default False
            Normalizes factors over the time series before normalizing over the cross-section.
        winsorize: int, default None
            Winsorizes/clips values to between positive and negative values of specified integer.

        Returns
        -------
        factors: pd.DataFrame
            Normalized factors with DatetimeIndex and normalized values (cols).
        """
        if self.normalize:
            if ts_norm:
                self.factors = Transform(self.factors).normalize(method=method, axis='ts', centering=centering,
                                                                 window_type=self.window_type,
                                                                 window_size=self.window_size, winsorize=winsorize)
                self.ret = Transform(self.ret).normalize(method=method, axis='ts', centering=centering,
                                                         window_type=self.window_type,
                                                         window_size=self.window_size, winsorize=winsorize)

            self.factors = Transform(self.factors).normalize(method=method, axis=self.strategy, centering=centering,
                                                             window_type=self.window_type, window_size=self.window_size,
                                                             winsorize=winsorize)
            self.ret = Transform(self.ret).normalize(method=method, axis=self.strategy, centering=centering,
                                                     window_type=self.window_type, window_size=self.window_size,
                                                     winsorize=winsorize)

            return self.factors

    @staticmethod
    def orthogonalize_data(df) -> pd.DataFrame:
        """
        Orthogonalize factors.

        As described by Klein and Chow (2013) in Orthogonalized Factors and Systematic Risk Decompositions:
        https://www.sciencedirect.com/science/article/abs/pii/S1062976913000185

        They propose an optimal simultaneous orthogonal transformation of factors, following the so-called symmetric
        procedure of Schweinler and Wigner (1970) and Löwdin (1970).  The data transformation allows the identification
        of the underlying uncorrelated components of common factors without changing their correlation with the original
        factors. It also facilitates the systematic risk decomposition by disentangling the coefficient of determination
        (R²) based on factor volatilities, which makes it easier to distinguish the marginal risk contribution of each

        Returns
        -------
        orthogonal_factors: pd.DataFrame
            Orthogonalized factors.
        """
        # convert to array
        if isinstance(df, pd.DataFrame):
            # convert to numpy
            arr = df.to_numpy(dtype=np.float64)
        else:
            arr = df.copy()
        # compute cov matrix
        M = np.cov(arr.T)
        # factorize cov matrix M
        u, s, vh = np.linalg.svd(M)
        # solve for symmetric matrix
        S = u @ np.diag(s ** (-0.5)) @ vh
        # rescale symmetric matrix to original variances
        M[M < 0] = np.nan  # remove negative values
        S_rs = S @ (np.diag(np.sqrt(M)) * np.eye(S.shape[0], S.shape[1]))
        # convert to orthogonal matrix
        orthogonal_arr = arr @ S_rs

        return pd.DataFrame(orthogonal_arr, index=df.index, columns=df.columns)

    def orthogonalize_factors(self) -> None:
        """
        Orthogonalize factors.
        """
        if self.orthogonalize:

            # rolling
            if self.window_type == 'rolling':
                if isinstance(self.factors.index, pd.MultiIndex):
                    factors_df = pd.DataFrame()
                    for ticker, ticker_df in self.factors.groupby(level=1):
                        orth_factors = rolling_window(self.orthogonalize_data, self.factors.loc[:, ticker, :],
                                                      window_size=self.window_size)
                        idx = orth_factors.index
                        df = pd.DataFrame(index=pd.MultiIndex.from_product([idx, [ticker]], names=['date', 'ticker']),
                                          data=orth_factors.values, columns=orth_factors.columns)
                        factors_df = pd.concat([factors_df, df])
                    self.factors = factors_df
                else:
                    self.factors = rolling_window(self.orthogonalize_data, self.factors, window_size=self.window_size)

            # expanding
            elif self.window_type == 'expanding':
                if isinstance(self.factors.index, pd.MultiIndex):
                    factors_df = pd.DataFrame()
                    for ticker, ticker_df in self.factors.groupby(level=1):
                        orth_factors = expanding_window(self.orthogonalize_data, self.factors.loc[:, ticker, :],
                                                        min_obs=30)
                        idx = orth_factors.index
                        df = pd.DataFrame(index=pd.MultiIndex.from_product([idx, [ticker]], names=['date', 'ticker']),
                                          data=orth_factors.values, columns=orth_factors.columns)
                        factors_df = pd.concat([factors_df, df])
                    self.factors = factors_df
                else:
                    self.factors = expanding_window(self.orthogonalize_data, self.factors, min_obs=30)

            # fixed
            else:
                if isinstance(self.factors.index, pd.MultiIndex):
                    self.factors = self.factors.groupby(level=1,
                                                        group_keys=False).apply(self.orthogonalize_data).sort_index()
                else:
                    self.factors = self.orthogonalize_data(self.factors)

            return self.factors

    def pooled_regression(self, multivariate: bool = True) -> Any:
        """
        Pooled regression of returns on factors.

        Parameters
        ----------
        multivariate: bool, default False
            Runs a multivariate regression on all factors. Otherwise, runs univariate regressions on each factor.

        Returns
        -------
        results: Any
            Dataframe/table containing pooled regrssion results.
        """
        # add intercept
        self.factors = add_trend(self.factors, trend='c', prepend=True)

        # multivariate regression
        if multivariate:
            self.results = OLS(self.ret, self.factors, missing='drop').fit(cov_type='HAC',
                                                                           cov_kwds={'maxlags': 1}).summary()
        # individual factor regressions
        else:
            self.results = pd.DataFrame(index=self.factors.columns[1:],
                                    columns=['params', 'bse', 'pvalues', 'f_pvalue', 'rsquared'])
            # iterate over factors, stats
            for factor in self.factors.columns[1:]:
                for stat in self.results.columns:
                    # run regression
                    res = OLS(self.ret, self.factors[['const', factor]],
                              missing='drop').fit(cov_type='HAC', cov_kwds={'maxlags': 1})
                    # store results
                    if stat in ['params', 'bse', 'pvalues']:
                        self.results.loc[factor, stat] = getattr(res, stat).iloc[1]
                    else:
                        self.results.loc[factor, stat] = getattr(res, stat)

            # sort by beta, rename columns
            self.results = self.results.sort_values(by='rsquared', ascending=False)
            self.results.columns = ['beta', 'std_error', 'p-val', 'f_p-val', 'R-squared']

        return self.results

    def check_min_obs(self, min_obs: int = 10) -> None:
        """
        Check minimum number of observations in the cross-section and reset start date of dataframe.

        Parameters
        ----------
        min_obs: int, default 10
            Minimum number of observations in the cross-section.
        """
        if self.factors.groupby('date').count()[(self.factors.groupby('date').count() > min_obs)].\
                dropna(how='all').empty:
            raise Exception(f"Cross-section does not meet minimum number of observations. Change min_obs parameter or "
                            f"increase asset universe.")
        else:
            # reset start date
            start_idx = self.data.groupby('date').count()[(self.data.groupby('date').count()
                                                           > min_obs)].dropna().index[0]
            self.data = self.data.loc[start_idx:]
            self.factors = self.factors.loc[start_idx:]
            self.ret = self.ret.loc[start_idx:]

    def fama_macbeth_regression(self, min_obs: int = 10, multivariate: bool = False) -> pd.DataFrame:
        """
        Runs cross-sectional Fama-Macbeth regressions for each time period to compute factor/characteristic betas.

        Parameters
        ----------
        min_obs: int, default 5
            Minimum number of observations in the cross-section to run the Fama Macbeth regression.
        multivariate: bool, default False
            Runs a multivariate regression on all factors. Otherwise, runs univariate regressions on each factor.

        Returns
        -------
        results: pd.DataFrame
            Dataframe with DatetimeIndex and estimated factor betas.
        """
        # check df index
        if not isinstance(self.factors.index, pd.MultiIndex):
            raise Exception("Dataframe index must be a MultiIndex with date and ticker levels.")

        # check n obs for factors in cross-section
        self.check_min_obs(min_obs=min_obs)

        # multivariate regression
        if multivariate:
            # estimate cross-sectional betas
            betas_df = self.data.groupby(level=0).apply(lambda x: OLS(x.iloc[:, 0], x.iloc[:, 1:], missing='drop').
                                                     fit(cov_type='HAC', cov_kwds={'maxlags': 1}).params)

        # individual factor regressions
        else:
            betas_df = pd.DataFrame()
            # iterate over factors, stats
            for i in range(1, self.data.shape[1]):
                # estimate cross-senal betas
                betas = self.data.groupby(level=0).apply(lambda x: OLS(x.iloc[:, 0], x.iloc[:, i], missing='drop').
                                                         fit(cov_type='HAC', cov_kwds={'maxlags': 1}).params)
                betas_df = pd.concat([betas_df, betas], axis=1)

        # get stats
        self.results = betas_df.describe().T
        self.results['std_error'] = self.results['std'] / np.sqrt(self.results['count'])
        self.results['t-stat'] = self.results['mean'] / self.results['std_error']

        self.results = self.results[['mean', 'std_error', 't-stat']]
        self.results.columns = ['beta', 'std_error', 't-stat']

        return self.results.sort_values(by='t-stat', ascending=False)
