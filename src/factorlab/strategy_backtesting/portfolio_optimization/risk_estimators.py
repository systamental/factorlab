import pandas as pd
import numpy as np
from typing import Union, Optional, List
from scipy.optimize import minimize
from sklearn.neighbors import KernelDensity
from sklearn.covariance import MinCovDet, EmpiricalCovariance, ShrunkCovariance, LedoitWolf, OAS, \
    GraphicalLasso, GraphicalLassoCV


class RiskEstimators:
    """
    Risk estimators class.

    This class computes the risk metrics of the assets or strategies
    using different methods.
    """
    def __init__(self,
                 returns: Union[pd.DataFrame, pd.Series],
                 asset_names: Optional[List[str]] = None,
                 window_type: str = 'fixed',
                 window_size: Optional[int] = None,
                 min_obs: Optional[int] = 30
                 ):
        """
        Constructor

        Parameters
        ----------
        returns: pd.DataFrame or pd.Series
            The returns of the assets or strategies. If not provided, the returns are computed from the prices.
        asset_names: list, default None
            Names of the assets or strategies.
        window_type: str, {'fixed', 'expanding', 'rolling'}, default 'fixed'
            Type of window for risk estimation.
        window_size: int, default 252
            Window size for risk estimation.
        min_obs: int, default 30
            Minimum number of observations for risk estimation. If the number of observations is less than this value,
            the risk estimation will not be performed and an error will be raised.
        """
        self.returns = returns
        self.asset_names = asset_names
        self.window_type = window_type
        self.window_size = window_size
        self.min_obs = min_obs
        self.ann_factor = None
        self.cov_matrix = None
        self.portfolio_risk = None
        self.preprocess_data()

    def preprocess_data(self):
        """
        Preprocess the data for the risk estimation.
        """
        # returns
        if not isinstance(self.returns, pd.DataFrame) and not isinstance(self.returns, pd.Series):  # check data type
            raise ValueError('rets must be a pd.DataFrame or pd.Series')
        # convert data type to float64
        if isinstance(self.returns, pd.Series):  # convert to df
            self.returns = self.returns.to_frame().astype('float64')
        elif isinstance(self.returns, pd.DataFrame):
            self.returns = self.returns.astype('float64')
        if isinstance(self.returns.index, pd.MultiIndex):  # convert to single index
            self.returns = self.returns.unstack()
        self.returns.index = pd.to_datetime(self.returns.index)  # convert to index to datetime

        # missing values
        self.returns = self.returns.dropna(how='all')  # drop missing rows

        # asset names
        if self.asset_names is None:
            self.asset_names = self.returns.columns.tolist()

        # annualization factor
        if self.ann_factor is None:
            self.ann_factor = self.returns.groupby(self.returns.index.year).count().max().mode()[0]

        # window size
        if self.window_size is None:
            self.window_size = self.ann_factor

        # cov matrix
        self.cov_matrix = self.covariance()

    def covariance(self) -> pd.DataFrame:
        """
        Compute the covariance matrix.

        Returns
        -------
        pd.DataFrame
            Covariance matrix.
        """
        self.cov_matrix = self.returns.cov().to_numpy('float64')

        return self.cov_matrix

    def empirical_covariance(self, assumed_centered: bool = False) -> pd.DataFrame:
        """
        Compute the empirical covariance matrix.

        Parameters
        ----------
        assumed_centered: bool, default False
            Whether the data is assumed to be centered. If True, data are not centered before computation.
            Useful when working with data whose mean is almost, but not exactly zero.
            If False (default), data are centered before computation.

        Returns
        -------
        pd.DataFrame
            Empirical covariance matrix.
        """
        self.cov_matrix = EmpiricalCovariance(assume_centered=assumed_centered).fit(self.returns).covariance_

        return self.cov_matrix

    def shrunk_covariance(self, assume_centered: bool = False, shrinkage: float = 0.1) -> pd.DataFrame:
        """
        Compute the shrunk covariance matrix.

        Wrapper for sklearn's ShrunkCovariance.

        Transformation to address the covariance matrix problem. Reduces the ratio between the smallest and the largest
        eigenvalues of the covariance matrix.

        Parameters
        ----------
        assume_centered: bool, default False
            Whether the data is assumed to be centered. If True, data are not centered before computation.
            Useful when working with data whose mean is almost, but not exactly zero.
            If False (default), data are centered before computation.
        shrinkage: float, default 0.1
            Coefficient in the convex combination used for the computation of the shrunk estimate. Range is [0, 1].

        Returns
        -------
        pd.DataFrame
            Shrunk covariance matrix.
        """
        self.cov_matrix = ShrunkCovariance(assume_centered=assume_centered, shrinkage=shrinkage).fit(self.returns).\
            covariance_

        return self.cov_matrix

    def ledoit_wolf(self, assume_centered: bool = False) -> pd.DataFrame:
        """
        Compute the Ledoit-Wolf covariance matrix.

        Wrapper for sklearn's LedoitWolf.

        Estimates the shrunk covariance matrix.

        Parameters
        ----------
        assume_centered: bool, default False
            Whether the data is assumed to be centered. If True, data are not centered before computation.
            Useful when working with data whose mean is almost, but not exactly zero.
            If False (default), data are centered before computation.

        Returns
        -------
        pd.DataFrame
            Ledoit-Wolf covariance matrix.
        """
        self.cov_matrix = LedoitWolf(assume_centered=assume_centered).fit(self.returns).covariance_

        return self.cov_matrix

    def oas(self, assume_centered: bool = False) -> pd.DataFrame:
        """
        Compute the OAS covariance matrix.

        Wrapper for sklearn's OAS.

        Estimates the shrunk covariance matrix.

        Parameters
        ----------
        assume_centered: bool, default False
            Whether the data is assumed to be centered. If True, data are not centered before computation.
            Useful when working with data whose mean is almost, but not exactly zero.
            If False (default), data are centered before computation.

        Returns
        -------
        pd.DataFrame
            OAS covariance matrix.
        """
        self.cov_matrix = OAS(assume_centered=assume_centered).fit(self.returns).covariance_

        return self.cov_matrix

    def graphical_lasso(self, alpha: float = 0.01, assume_centered: bool = False) -> pd.DataFrame:
        """
        Compute the Graphical Lasso covariance matrix.

        Wrapper for sklearn's GraphicalLasso.

        Estimates the precision matrix using Graphical Lasso.

        Parameters
        ----------
        alpha: float, default 0.01
            Regularization parameter. The higher alpha, the more regularization, the sparser the inverse covariance.
            Range is (0, inf].
        assume_centered: bool, default False
            Whether the data is assumed to be centered. If True, data are not centered before computation.
            Useful when working with data whose mean is almost, but not exactly zero.
            If False (default), data are centered before computation.

        Returns
        -------
        pd.DataFrame
            Graphical Lasso covariance matrix.
        """
        self.cov_matrix = GraphicalLasso(alpha=alpha, assume_centered=assume_centered).fit(self.returns).covariance_

        return self.cov_matrix

    def graphical_lasso_cv(self, assume_centered: bool = False) -> pd.DataFrame:
        """
        Compute the Graphical Lasso covariance matrix using cross-validation.

        Wrapper for sklearn's GraphicalLassoCV.

        Estimates the precision matrix using Graphical Lasso with cross-validation.

        Parameters
        ----------
        assume_centered: bool, default False
            Whether the data is assumed to be centered. If True, data are not centered before computation.
            Useful when working with data whose mean is almost, but not exactly zero.
            If False (default), data are centered before computation.

        Returns
        -------
        pd.DataFrame
            Graphical Lasso covariance matrix.
        """
        self.cov_matrix = GraphicalLassoCV(assume_centered=assume_centered).fit(self.returns).covariance_

        return self.cov_matrix

    def minimum_covariance_determinant(self,
                                       assume_centered: bool = False,
                                       support_fraction: Optional[float] = None,
                                       random_state: Optional[int] = None
                                       ) -> pd.DataFrame:
        """
        Compute the minimum covariance determinant matrix.

        Wrapper for sklearn's MinCovDet.

        Estimates the minimum covariance determinant matrix.

        Parameters
        ----------
        assume_centered: bool, default False
            Whether the data is assumed to be centered. If True, data are not centered before computation.
            Useful when working with data whose mean is almost, but not exactly zero.
            If False (default), data are centered before computation.
        support_fraction: float, default None
            The proportion of points to be included in the support of the raw MCD estimate.
            Range is (0, 1]. If None, the minimum value of support_fraction will be used within the algorithm.
        random_state: int, default None
            Seed for the random number generator.

        Returns
        -------
        pd.DataFrame
            Minimum covariance determinant matrix.
        """
        self.cov_matrix = MinCovDet(assume_centered=assume_centered, support_fraction=support_fraction,
                                    random_state=random_state).fit(self.returns).covariance_

        return self.cov_matrix

    def semi_covariance(self, return_thresh: float = 0.0) -> pd.DataFrame:
        """
        Compute the semi-covariance matrix.

        Semi-covariance is a measure of the dispersion of returns in the lower tail of the distribution.
        It is used to compute the downside risk of an asset or portfolio.

        Returns
        -------
        pd.DataFrame
            Semi-covariance matrix.
        """
        # cov matrix
        self.cov_matrix = self.returns.cov()

        # filter downside returns
        downside_rets = np.where(self.returns < return_thresh, self.returns, 0)

        self.cov_matrix = (downside_rets.T @ downside_rets) / downside_rets.size

        return self.cov_matrix

    def exponential_covariance(self) -> pd.DataFrame:
        """
        Compute the exponential covariance matrix.

        Returns
        -------
        pd.DataFrame
            Exponential covariance matrix.
        """
        # demean returns
        ret_demeaned = self.returns - self.returns.mean()
        # ewm cov matrix
        self.cov_matrix = ret_demeaned.ewm(span=self.window_size).cov()
        # last date
        last_date = self.cov_matrix.index.get_level_values('date').max()
        self.cov_matrix = self.cov_matrix.loc[last_date].values

        return self.cov_matrix

    @staticmethod
    def mp_pdf(var: float, q: float, pts: int) -> pd.Series:
        """
        Compute the Marcenko-Pastur probability density function.

        Parameters
        ----------
        var: float
            Variance.
        q: float
            Number of observations T over number of variables N.
        pts: int
            Number of points.

        Returns
        -------
        float
            Marcenko-Pastur probability density function.
        """
        # check type
        if isinstance(var, np.ndarray):
            var = var[0]

        # eigenvalues range
        eig_min, eig_max = var * (1 - (1. / q) ** .5) ** 2, var * (1 + (1. / q) ** .5) ** 2
        eigen_val = np.linspace(eig_min, eig_max, pts)

        # Marcenko-Pastur pdf
        pdf = q / (2 * np.pi * var * eigen_val) * ((eig_max - eigen_val) * (eigen_val - eig_min)) ** .5
        pdf = pd.Series(pdf, index=eigen_val)

        return pdf

    @staticmethod
    def get_pca(hermitian_matrix: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Computes the eigenvalues and eigenvectors of a hermitian matrix.

        Parameters
        ----------
        hermitian_matrix: np.ndarray
            Covariance matrix.

        Returns
        -------
        np.ndarray
            Principal components analysis.
        """
        # eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(hermitian_matrix)

        # sort eigenvalues
        idx = eigenvalues.argsort()[::-1]
        eigenvalues, eigenvectors = eigenvalues[idx], eigenvectors[:, idx]

        # diagonal matrix of eigenvalues
        eigenvalues = np.diagflat(eigenvalues)

        return eigenvalues, eigenvectors

    @staticmethod
    def fit_kde(obs: np.ndarray,
                bwidth: float = 0.25,
                kernel: str = 'gaussian',
                eval_pts: np.ndarray = None
                ) -> pd.Series:
        """
        Fit the kernel density estimation to a series of observations
        and derives the log probability of the observations.

        Parameters
        ----------
        obs: np.ndarray
            Observations.
        bwidth: float, default 0.25
            Bandwidth of the kernel.
        kernel: str, default 'gaussian'
            Kernel type.
        eval_pts: np.ndarray, default None
            Evaluation points.

        Returns
        -------
        pdf: pd.Series
            Log probability of the observations.
        """
        # reshape obs
        obs = obs.reshape(-1, 1)

        # kernel density
        kde = KernelDensity(bandwidth=bwidth, kernel=kernel).fit(obs)

        # evaluation points
        if eval_pts is None:
            eval_pts = np.unique(obs).reshape(-1, 1)
        if len(eval_pts.shape) == 1:
            eval_pts = eval_pts.reshape(-1, 1)

        # log probability
        log_prob = kde.score_samples(eval_pts)
        # pdf
        pdf = pd.Series(np.exp(log_prob), index=eval_pts.flatten())

        return pdf

    @staticmethod
    def cov_to_corr(cov_matrix: np.ndarray) -> np.ndarray:
        """
        Compute the correlation matrix from the covariance matrix.

        Parameters
        ----------
        cov_matrix: np.ndarray
            Covariance matrix.

        Returns
        -------
        corr: np.ndarray
            Correlation matrix.
        """
        # std deviation
        std = np.sqrt(np.diag(cov_matrix))
        # correlation matrix
        corr = cov_matrix / np.outer(std, std)
        # numerical errors
        corr[corr < -1], corr[corr > 1] = -1, 1

        return corr

    @staticmethod
    def corr_to_cov(corr_matrix: np.ndarray, std: np.ndarray) -> np.ndarray:
        """
        Compute the covariance matrix from the correlation matrix.

        Parameters
        ----------
        corr_matrix: np.ndarray
            Correlation matrix.
        std: np.ndarray
            Standard deviation.

        Returns
        -------
        np.ndarray
            Covariance matrix.
        """
        cov_matrix = corr_matrix * np.outer(std, std)

        return cov_matrix

    def fit_pdf(self,
                var: float,
                eigenvalues: np.ndarray,
                q: float,
                bwidth: float = 0.25,
                kernel: str = 'gaussian',
                pts: int = 1000
                ) -> pd.Series:
        """
        Fit the Marcenko-Pastur probability density function.

        Parameters
        ----------
        var: float
            Variance.
        eigenvalues: np.ndarray
            Eigenvalues.
        q: float
            Number of observations T over number of variables N.
        bwidth: float, default 0.01
            Bandwidth of the kernel.
        kernel: str, default 'gaussian'
            Kernel type.
        pts: int, default 1000
            Number of points.

        Returns
        -------
        sse: float
            Sum of squared errors.
        """
        # fit theoretical Marcenko-Pastur pdf
        pdf = self.mp_pdf(var, q, pts)
        # fit empirical pdf
        kde_pdf = self.fit_kde(eigenvalues, bwidth=bwidth, kernel=kernel, eval_pts=pdf.index.values)

        # sum of squared errors
        sse = np.sum((kde_pdf - pdf) ** 2)

        return sse

    def find_max_eigenval(self, eigenvalues: np.ndarray, q: float, bwidth: float = 0.01) -> float:
        """
        Find the maximum eigenvalue.

        Parameters
        ----------
        eigenvalues: np.ndarray
            Eigenvalues.
        q: float
            Number of observations T over number of variables N.
        bwidth: float, default 0.01
            Bandwidth of the kernel.

        Returns
        -------
        float
            Maximum eigenvalue.
        """
        # optimization
        res = minimize(self.fit_pdf, 0.5, args=(eigenvalues, q, bwidth),  bounds=((1e-5, 1 - 1e-5),))

        # solution
        if res.success:
            var = res['x'][0]
        else:
            var = 1

        # maximum eigenvalue
        eigen_max = var * (1 + (1. / q) ** .5) ** 2

        return eigen_max, var

    def denoised_corr_constant_resid_eigenval(self,
                                              eigenvals: np.ndarray,
                                              eigenvecs: np.ndarray,
                                              n_factors: int
                                              ) -> np.ndarray:
        """
        Compute the denoised correlation matrix.

        For more details, see:  "Machine Learning for Asset Managers". Elements in Quantitative Finance.
        Lòpez de Prado (2020).

        Parameters
        ----------
        eigenvals: np.ndarray
            Eigenvalues.
        eigenvecs: np.ndarray
            Eigenvectors.
        n_factors: int
            Number of factors.

        Returns
        -------
        corr: np.ndarray
            Denoised correlation matrix.
        """
        # eigenvalues from diagonal
        evals = np.diag(eigenvals).copy()

        # denoised eigenvalues
        evals[n_factors:] = evals[n_factors:].sum() / float(evals.shape[0] - n_factors)
        evals = np.diag(evals)

        # denoised correlation matrix
        corr1 = np.dot(eigenvecs, evals).dot(eigenvecs.T)
        corr = self.cov_to_corr(corr1)

        return corr

    @staticmethod
    def deonoised_corr_targeted_shrinkage(eigenvals: np.ndarray,
                                          eigenvecs: np.ndarray,
                                          n_factors: int,
                                          alpha: float = 0
                                          ) -> np.ndarray:
        """
        Compute the denoised correlation matrix using targeted shrinkage.

        Parameters
        ----------
        eigenvals: np.ndarray
            Eigenvalues.
        eigenvecs: np.ndarray
            Eigenvectors.
        n_factors: int
            Number of factors.
        alpha: float, default 0
            Shrinkage parameter.

        Returns
        -------
        corr: np.ndarray
            Denoised correlation matrix.
        """
        # get eigenvalues and eigenvectors
        eigenvals_left, eigenvecs_left = eigenvals[:n_factors, :n_factors], eigenvecs[:, :n_factors]
        eigevals_right, eigenvecs_right = eigenvals[n_factors:, n_factors:], eigenvecs[:, n_factors:]

        # targeted shrinkage
        corr0 = np.dot(eigenvecs_left, eigenvals_left).dot(eigenvecs_left.T)
        corr1 = np.dot(eigenvecs_right, eigevals_right).dot(eigenvecs_right.T)
        corr2 = corr0 + alpha * corr1 + (1 - alpha) * np.diag(np.diag(corr1))

        return corr2

    def denoise_corr(self,
                     eigenvals: np.ndarray,
                     eigenvecs: np.ndarray,
                     n_factors: int,
                     method: str = 'constant_residual_eigenval',
                     alpha: float = 0
                     ) -> np.ndarray:
        """
        Denoise the covariance matrix.

        Parameters
        ----------

        Returns
        -------
        np.ndarray
            Denoised covariance matrix.
        """
        if method == 'constant_residual_eigenval':
            return self.denoised_corr_constant_resid_eigenval(eigenvals, eigenvecs, n_factors)
        elif method == 'targeted_shrinkage':
            return self.deonoised_corr_targeted_shrinkage(eigenvals, eigenvecs, n_factors, alpha)
        else:
            raise ValueError('Invalid method. Please choose from: constant_residual_eigenval, targeted_shrinkage')

    def detone_corr(self,
                    eigenvals: np.ndarray,
                    eigenvecs: np.ndarray,
                    n_factors: int) -> np.ndarray:
        """
        Compute the detoned correlation matrix.
        
        For more details, see:  "Machine Learning for Asset Managers". Elements in Quantitative Finance.
        Lòpez de Prado (2020).
        
        Parameters
        ----------
        eigenvals: np.ndarray
            Eigenvalues.
        eigenvecs: np.ndarray
            Eigenvectors.
        n_factors: int
            Number of factors.
                
        Returns
        -------
        corr: np.ndarray
            Detoned correlation matrix.
        """
        # denoised correlation matrix
        corr = self.denoise_corr(eigenvals, eigenvecs, n_factors, method='constant_residual_eigenval', alpha=0)

        # market factor
        eigenvals_mkt, eigenvecs_mkt = eigenvals[: 1, : 1], eigenvecs[:, :1]

        # detoned correlation matrix
        corr = corr - np.dot(eigenvecs_mkt, eigenvals_mkt).dot(eigenvecs_mkt.T)

        # rescale correlation matrix
        corr = self.cov_to_corr(corr)

        return corr

    def denoised_covariance(self,
                            method: str = 'constant_residual_eigenval',
                            bwidth: float = 0.01,
                            alpha: float = 0,
                            detone: bool = False,
                            q: Optional[float] = None,
                            ) -> np.ndarray:
        """
        Compute the denoised covariance matrix.

        See "Machine Learning for Asset Managers". Elements in Quantitative Finance. Lòpez de Prado (2020).

        Parameters
        ----------
        q: float
            Number of observations T over number of variables N.
        method: str, {'constant_residual_eigenval', 'targeted_shrinkage'}, default 'constant_residual_eigenval'
            Method to denoise the covariance matrix.
        bwidth: float, default 0.01
            Bandwidth of the kernel.
        alpha: float, default 0
            Shrinkage parameter.
        detone: bool, default False
            Whether to detone the covariance matrix.

        Returns
        -------
        cov_matrix: np.ndarray
            Denoised covariance matrix.
        """
        # q
        if q is None:
            q = self.returns.shape[0] / self.returns.shape[1]

        # correlation matrix
        corr_matrix = self.cov_to_corr(self.cov_matrix)

        # eigenvalues and eigenvectors
        eigenvalues, eigenvectors = self.get_pca(corr_matrix)

        # maximum eigenvalue
        eigen_max, var = self.find_max_eigenval(eigenvalues, q, bwidth)

        # threshold for eigenvalues
        n_factors = np.sum(eigenvalues > eigen_max)

        # denoised correlation matrix
        corr = self.denoise_corr(eigenvalues, eigenvectors, n_factors, method=method, alpha=alpha)

        # detoned correlation matrix
        if detone:
            corr = self.detone_corr(eigenvalues, eigenvectors, n_factors)

        # denoised covariance matrix
        self.cov_matrix = self.corr_to_cov(corr, np.sqrt(np.diag(self.cov_matrix)))

        return self.cov_matrix

    def compute_covariance_matrix(self, method: str = 'covariance', **kwargs) -> Union[pd.DataFrame, np.ndarray]:
        """
        Compute the portfolio variance.

        Parameters
        ----------
        method: str, {'covariance', 'empirical_covariance', 'shrunk_covariance', 'ledoit_wolf', 'oas',
                       'graphical_lasso', 'graphical_lasso_cv', 'minimum_covariance_determinant', 'semi_covariance',
                          'exponential_covariance', 'denoised_covariance'}, default 'covariance'
            Method to compute the covariance matrix.

        Returns
        -------
        np.ndarray
            Covariance matrix.
        """
        # compute covariance matrix
        if method in ['covariance', 'empirical_covariance', 'shrunk_covariance', 'ledoit_wolf', 'oas',
                      'graphical_lasso', 'graphical_lasso_cv', 'minimum_covariance_determinant', 'semi_covariance',
                      'exponential_covariance', 'denoised_covariance']:
            self.cov_matrix = getattr(self, method)(**kwargs)

        else:
            raise ValueError("Method is not supported. Valid methods are: 'covariance', 'empirical_covariance', "
                             "'shrunk_covariance', 'ledoit_wolf', 'oas', 'graphical_lasso', 'graphical_lasso_cv', "
                             "'minimum_covariance_determinant', 'semi_covariance', 'exponential_covariance', "
                             "'denoised_covariance'")

        return self.cov_matrix

    # TODO add KKT index computation:
    # https://globalmarkets.statestreet.com/research/service/public/v1/article/insights/pdf/v2/5b4b47fa-8256-4e4a-8991-afe13469268b/joim_a_new_index_of_the_business_cycle.pdf
    def turbulence_index(self, component: str = 'turbulence') -> pd.Series:
        """
        Compute the turbulence index.

        The turbulence index is a measure of the risk of the assets or strategies. It is computed as the Mahalanobis
        distance of the demeaned returns of the assets or strategies. The Mahalanobis distance is a measure of the
        distance between a point and a distribution. It is used to compute the risk of the assets or strategies.

        Turbulence can be further decomposed into the correlation surprise and the magnitude surprise. The correlation
        surprise is the ratio of the turbulence to the magnitude surprise. The magnitude surprise is the Mahalanobis
        distance of the demeaned returns of the assets or strategies divided by the number of assets or strategies,
        where the covariance matrix is the correlation blind covariance matrix (magnitude).

        For more details, see:

        Kritzman, Mark and Li, Yuanzhen, Skulls, Financial Turbulence, and Risk Management (October 13, 2010).
        Financial Analysts Journal, Vol. 66, No. 5, 2010, Available at SSRN: https://ssrn.com/abstract=1691756

        Kinlaw, William B. and Turkington, David, Correlation Surprise (July 13, 2012).
        Available at SSRN: https://ssrn.com/abstract=2133396 or http://dx.doi.org/10.2139/ssrn.2133396

        Kinlaw, William B. and Kritzman, Mark and Turkington, David, A New Index of the Business Cycle
        (January 15, 2020). MIT Sloan Research Paper No. 5908-20,
        Available at SSRN: https://ssrn.com/abstract=3521300 or http://dx.doi.org/10.2139/ssrn.3521300

        Parameters
        ----------
        component: str, {'turbulence', 'correlation_surprise', 'magnitude_surprise'}, default 'turbulence'
            Component of the turbulence index.

        Returns
        -------
        pd.DataFrame
            Turbulence index, correlation surprise, or magnitude surprise.
        """
        turb, corr_surprise, magnitude_surprise = None, None, None

        # check component
        if component not in ['turbulence', 'correlation_surprise', 'magnitude_surprise']:
            raise ValueError("Invalid component. Please choose from: 'turbulence', 'correlation_surprise', "
                             "'magnitude_surprise'")

        # demean returns
        ret_demeaned = (self.returns - self.returns.mean()).values

        # inv cov matrix
        inv_cov_matrix = np.linalg.pinv(self.cov_matrix)

        # mahalanobis distance
        turb = np.diagonal(ret_demeaned @ inv_cov_matrix @ ret_demeaned.T) / self.returns.shape[1]

        # compute correlation blind cov matrix (magnitude)
        if component != 'turbulence':
            cov_matrix_mag = np.diag(np.diag(self.cov_matrix))
            inv_cov_matrix_mag = np.linalg.pinv(cov_matrix_mag)
            magnitude_surprise = (np.diagonal(ret_demeaned @ inv_cov_matrix_mag @ ret_demeaned.T) /
                                  self.returns.shape[1])
            corr_surprise = turb / magnitude_surprise

        # component
        if component == 'correlation_surprise':
            return pd.DataFrame(corr_surprise, index=self.returns.index, columns=[component])
        elif component == 'magnitude_surprise':
            return pd.DataFrame(magnitude_surprise, index=self.returns.index, columns=[component])
        else:
            return pd.DataFrame(turb, index=self.returns.index, columns=[component])

    def compute_portfolio_risk(self, method: str = 'turbulence_index', **kwargs) -> Union[pd.DataFrame, np.ndarray]:
        """
        Compute the portfolio risk.

        Parameters
        ----------
        method: str, {'turbulence_index'}, default 'turbulence_index'
            Method to compute the portfolio risk.

        Returns
        -------
        np.ndarray or pd.DataFrame
            Portfolio risk.
        """
        # compute portfolio risk
        if method in ['turbulence_index']:

            if self.window_type == 'rolling':

                # loop through rows of returns
                for row in range(self.returns.shape[0] - self.window_size + 1):

                    # data window
                    data_window = self.returns.iloc[row:row + self.window_size].dropna(how='all').dropna(axis=1)

                    # compute risk estimator
                    risk = getattr(RiskEstimators(data_window), method)(**kwargs)
                    if row == 0:
                        self.portfolio_risk = pd.DataFrame(risk.iloc[-1]).T
                    else:
                        self.portfolio_risk = pd.concat([self.portfolio_risk, pd.DataFrame(risk.iloc[-1]).T])

            elif self.window_type == 'expanding':

                # loop through rows of returns
                for row in range(self.min_obs, self.returns.shape[0] + 1):

                    # compute risk estimator
                    risk = getattr(RiskEstimators(self.returns.iloc[:row]), method)(**kwargs)
                    if row == self.min_obs:
                        self.portfolio_risk = pd.DataFrame(risk.iloc[-1]).T
                    else:
                        self.portfolio_risk = pd.concat([self.portfolio_risk, pd.DataFrame(risk.iloc[-1]).T])

            else:
                self.portfolio_risk = getattr(self, method)(**kwargs)

        else:
            raise ValueError("Method is not supported. Valid methods are: 'turbulence_index'")

        return self.portfolio_risk
