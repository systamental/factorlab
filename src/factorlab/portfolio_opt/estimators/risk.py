import pandas as pd
import numpy as np
from typing import Union, Optional
from sklearn.covariance import MinCovDet, EmpiricalCovariance, ShrunkCovariance, LedoitWolf, OAS, GraphicalLasso, \
    GraphicalLassoCV
from scipy.optimize import minimize
from sklearn.neighbors import KernelDensity


class RiskMetrics:
    """
    A stateless utility class for computing various risk and moment estimation
    metrics (first and second moments) from historical returns data.

    This class is designed to be called by PortfolioOptimizerBase implementations
    during their fit() step. It does not handle time-window slicing or state.
    """

    @staticmethod
    def _validate_returns(returns: pd.DataFrame) -> pd.DataFrame:
        """
        Ensures input is a DataFrame of float64 type with a single index and
        a minimum number of assets/observations.
        """
        if not isinstance(returns, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame of returns.")
        if returns.empty:
            raise ValueError("Input DataFrame is empty.")

        # ensure single index
        if isinstance(returns.index, pd.MultiIndex):
            raise ValueError(
                f"Input DataFrame index must be a single index (e.g., DateTimeIndex), not a MultiIndex. "
                f"Please ensure the data is unstacked (T rows x N columns) before calling RiskMetrics.")

        # Ensure correct type (copy to avoid side effects on original data)
        validated_returns = returns.astype('float64', copy=True).dropna(how='all')

        # Simple check for minimum data required for matrix inversion/estimation
        if validated_returns.shape[0] < 5 or validated_returns.shape[1] < 2:
            raise ValueError("Insufficient data (rows < 5 or columns < 2) for moment estimation.")

        return validated_returns

    @staticmethod
    def covariance(returns: pd.DataFrame) -> pd.DataFrame:
        """Compute the standard empirical covariance matrix."""
        returns = RiskMetrics._validate_returns(returns)
        return returns.cov()

    @staticmethod
    def semi_covariance(returns: pd.DataFrame, return_thresh: float = 0.0) -> pd.DataFrame:
        """
        Compute the semi-covariance matrix, considering only downside returns.

        This is typically used in downside risk measures like Sortino Ratio.
        """
        returns = RiskMetrics._validate_returns(returns)

        downside_rets = returns.where(returns < return_thresh, 0)

        semi_cov = (downside_rets.T @ downside_rets) / downside_rets.size

        return semi_cov

    @staticmethod
    def empirical_covariance(returns: pd.DataFrame, assumed_centered: bool = False) -> pd.DataFrame:
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
        returns = RiskMetrics._validate_returns(returns)

        cov_matrix_np = EmpiricalCovariance(assume_centered=assumed_centered).fit(returns).covariance_

        # Convert back to a DataFrame with asset names
        return pd.DataFrame(cov_matrix_np, index=returns.columns, columns=returns.columns)

    @staticmethod
    def shrunk_covariance(returns: pd.DataFrame, shrinkage: float = 0.1, assume_centered: bool = False) -> pd.DataFrame:
        """
        Compute the shrunk covariance matrix.

        Parameters
        ----------
        shrinkage: float, default 0.1
            The shrinkage parameter, between 0 and 1.
            0 corresponds to the empirical covariance matrix,
            1 corresponds to the shrinkage target (the identity matrix).

        assume_centered: bool, default False
            Whether the data is assumed to be centered. If True, data are not centered before computation.
            Useful when working with data whose mean is almost, but not exactly zero.
            If False (default), data are centered before computation.

        Returns
        -------
        pd.DataFrame
            Shrunk covariance matrix.
        """
        returns = RiskMetrics._validate_returns(returns)

        cov_matrix_np = ShrunkCovariance(shrinkage=shrinkage, assume_centered=assume_centered).fit(returns).covariance_

        # Convert back to a DataFrame with asset names
        return pd.DataFrame(cov_matrix_np, index=returns.columns, columns=returns.columns)

    @staticmethod
    def ledoit_wolf(returns: pd.DataFrame, assume_centered: bool = False) -> pd.DataFrame:
        """
        Compute the Ledoit-Wolf shrunk covariance matrix.

        Wrapper for sklearn's LedoitWolf.

        Parameters
        ----------
        assume_centered: bool, default False
            Whether the data is assumed to be centered. If True, data are not centered before computation.
            Useful when working with data whose mean is almost, but not exactly zero.
            If False (default), data are centered before computation.

        Returns
        -------
        pd.DataFrame
            Ledoit-Wolf shrunk covariance matrix.
        """
        returns = RiskMetrics._validate_returns(returns)

        cov_matrix_np = LedoitWolf(assume_centered=assume_centered).fit(returns).covariance_

        # Convert back to a DataFrame with asset names
        return pd.DataFrame(cov_matrix_np, index=returns.columns, columns=returns.columns)

    @staticmethod
    def oas(returns: pd.DataFrame, assume_centered: bool = False) -> pd.DataFrame:
        """
        Compute the OAS (Oracle Approximating Shrinkage) covariance matrix.

        Wrapper for sklearn's OAS.

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
        returns = RiskMetrics._validate_returns(returns)

        cov_matrix_np = OAS(assume_centered=assume_centered).fit(returns).covariance_

        return pd.DataFrame(cov_matrix_np, index=returns.columns, columns=returns.columns)

    @staticmethod
    def graphical_lasso(returns: pd.DataFrame,
                        alpha: float = 0.01,
                        max_iter: int = 100,
                        tol: float = 1e-4,
                        assume_centered: bool = False) -> pd.DataFrame:
        """
        Compute the Graphical Lasso covariance matrix.

        Wrapper for sklearn's GraphicalLasso.

        Parameters
        ----------
        alpha: float, default 0.01
            Regularization parameter.
        max_iter: int, default 100
            Maximum number of iterations.
        tol: float, default 1e-4
            Convergence tolerance.
        assume_centered: bool, default False
            Whether the data is assumed to be centered. If True, data are not centered before computation.
            Useful when working with data whose mean is almost, but not exactly zero.
            If False (default), data are centered before computation.

        Returns
        -------
        pd.DataFrame
            Graphical Lasso covariance matrix.
        """
        returns = RiskMetrics._validate_returns(returns)

        cov_matrix_np = GraphicalLasso(alpha=alpha, max_iter=max_iter, tol=tol,
                                       assume_centered=assume_centered).fit(returns).covariance_

        return pd.DataFrame(cov_matrix_np, index=returns.columns, columns=returns.columns)

    @staticmethod
    def graphical_lasso_cv(returns: pd.DataFrame,
                           assume_centered: bool = False) -> pd.DataFrame:
        """
        Compute the Graphical Lasso with Cross-Validation covariance matrix.

        Wrapper for sklearn's GraphicalLassoCV.

        Parameters
        ----------
        assume_centered: bool, default False
            Whether the data is assumed to be centered. If True, data are not centered before computation.
            Useful when working with data whose mean is almost, but not exactly zero.
            If False (default), data are centered before computation.

        Returns
        -------
        pd.DataFrame
            Graphical Lasso CV covariance matrix.
        """
        returns = RiskMetrics._validate_returns(returns)

        cov_matrix_np = GraphicalLassoCV(assume_centered=assume_centered).fit(returns).covariance_

        return pd.DataFrame(cov_matrix_np, index=returns.columns, columns=returns.columns)

    @staticmethod
    def minimum_covariance_determinant(returns: pd.DataFrame,
                                      support_fraction: Optional[float] = None,
                                      random_state: Optional[int] = None,
                                      assume_centered: bool = False) -> pd.DataFrame:
        """
        Compute the Minimum Covariance Determinant (MCD) robust covariance matrix.

        Wrapper for sklearn's MinCovDet.

        Parameters
        ----------
        support_fraction: float, optional
            The proportion of points to be included in the support of the raw MCD estimate.
            If None, the default value is (n + p + 1) / (2 * n) where n is the number of samples
            and p is the number of features.
        random_state: int, optional
            Seed for random number generator.
        assume_centered: bool, default False
            Whether the data is assumed to be centered. If True, data are not centered before computation.
            Useful when working with data whose mean is almost, but not exactly zero.
            If False (default), data are centered before computation.

        Returns
        -------
        pd.DataFrame
            Minimum Covariance Determinant covariance matrix.
        """
        returns = RiskMetrics._validate_returns(returns)

        cov_matrix_np = MinCovDet(support_fraction=support_fraction,
                                  random_state=random_state,
                                  assume_centered=assume_centered).fit(returns).covariance_

        return pd.DataFrame(cov_matrix_np, index=returns.columns, columns=returns.columns)

    @staticmethod
    def exponential_covariance(returns: pd.DataFrame, span: int = 60) -> pd.DataFrame:
        """
        Compute the exponentially weighted covariance matrix.

        Parameters
        ----------
        span: int, default 60
            The span for the exponential weighting. Higher values give more weight to recent observations.

        Returns
        -------
        pd.DataFrame
            Exponentially weighted covariance matrix.
        """
        returns = RiskMetrics._validate_returns(returns)

        # demean returns
        ret_demeaned = returns - returns.mean()
        # ewm cov matrix
        cov_matrix = ret_demeaned.ewm(span=span).cov()
        # last date
        last_date = cov_matrix.index.get_level_values('date').max()
        cov_matrix = cov_matrix.loc[last_date].values

        return pd.DataFrame(cov_matrix, index=returns.columns, columns=returns.columns)

    @staticmethod
    def cov_to_corr(cov_matrix: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Compute the correlation matrix from the covariance matrix.

        Parameters
        ----------
        cov_matrix: Union[pd.DataFrame, np.ndarray]
            Covariance matrix.

        Returns
        -------
        corr: np.ndarray
            Correlation matrix.
        """
        cov_matrix = np.asarray(cov_matrix)
        std = np.sqrt(np.diag(cov_matrix))
        std[std == 0] = 1e-10  # Avoid division by zero
        corr = cov_matrix / np.outer(std, std)
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
            Standard deviation vector.

        Returns
        -------
        np.ndarray
            Covariance matrix.
        """
        return corr_matrix * np.outer(std, std)

    @staticmethod
    def get_pca(hermitian_matrix: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Computes the eigenvalues and eigenvectors of a hermitian matrix (e.g., correlation matrix).
        """
        eigenvalues, eigenvectors = np.linalg.eigh(hermitian_matrix)

        # Sort eigenvalues in descending order and reorder eigenvectors
        idx = eigenvalues.argsort()[::-1]
        eigenvalues, eigenvectors = eigenvalues[idx], eigenvectors[:, idx]

        # Return eigenvalues as a diagonal matrix
        return np.diagflat(eigenvalues), eigenvectors

    @staticmethod
    def mp_pdf(var: float, q: float, pts: int) -> pd.Series:
        """Compute the Marcenko-Pastur probability density function (PDF)."""
        # Ensure var is float if somehow passed as an array
        if isinstance(var, np.ndarray):
            var = var.item()  # Use .item() for safe extraction of a single value

        # Eigenvalues range
        eig_min, eig_max = var * (1 - (1. / q) ** .5) ** 2, var * (1 + (1. / q) ** .5) ** 2
        eigen_val = np.linspace(eig_min, eig_max, pts)

        # Marcenko-Pastur PDF formula
        pdf = q / (2 * np.pi * var * eigen_val) * ((eig_max - eigen_val) * (eigen_val - eig_min)) ** .5
        pdf = pd.Series(pdf, index=eigen_val)

        return pdf

    @staticmethod
    def fit_kde(obs: np.ndarray,
                bwidth: float = 0.25,
                kernel: str = 'gaussian',
                eval_pts: np.ndarray = None
                ) -> pd.Series:
        """Fit the kernel density estimation to observations."""
        obs = obs.reshape(-1, 1)

        kde = KernelDensity(bandwidth=bwidth, kernel=kernel).fit(obs)

        if eval_pts is None:
            eval_pts = np.unique(obs).reshape(-1, 1)
        if len(eval_pts.shape) == 1:
            eval_pts = eval_pts.reshape(-1, 1)

        log_prob = kde.score_samples(eval_pts)
        pdf = pd.Series(np.exp(log_prob), index=eval_pts.flatten())

        return pdf

    @staticmethod
    def fit_pdf(var: float, eigenvalues: np.ndarray, q: float, bwidth: float = 0.25, kernel: str = 'gaussian',
                pts: int = 1000) -> float:
        """
        Objective function for optimization: computes the Sum of Squared Errors (SSE)
        between the theoretical Marcenko-Pastur PDF and the empirical KDE of eigenvalues.

        Returns
        -------
        sse: float
            Sum of squared errors.
        """
        # Fit theoretical Marcenko-Pastur pdf
        pdf = RiskMetrics.mp_pdf(var, q, pts)

        # Fit empirical pdf using KDE on the same evaluation points
        kde_pdf = RiskMetrics.fit_kde(eigenvalues, bwidth=bwidth, kernel=kernel, eval_pts=pdf.index.values)

        # Sum of squared errors
        sse = np.sum((kde_pdf - pdf) ** 2)

        return sse

    @staticmethod
    def find_max_eigenval(eigenvalues: np.ndarray, q: float, bwidth: float = 0.01) -> (float, float):
        """
        Find the maximum theoretical eigenvalue (Lambda Max) and the estimated
        single-factor variance (var) via optimization.

        Returns
        -------
        (eigen_max, var): (float, float)
        """
        # Optimization: minimize SSE by adjusting the single factor variance (var)
        # Bounds ensures var is between 0 and 1
        res = minimize(RiskMetrics.fit_pdf, 0.5, args=(eigenvalues, q, bwidth), bounds=((1e-5, 1.0 - 1e-5),))

        if res.success:
            var = res.x[0]
        else:
            # Fallback if optimization fails
            var = 1.0

            # Calculate the maximum theoretical eigenvalue based on the optimized variance
        eigen_max = var * (1 + (1. / q) ** .5) ** 2

        return eigen_max, var

    @staticmethod
    def denoised_corr_constant_resid_eigenval(eigenvals: np.ndarray, eigenvecs: np.ndarray,
                                              n_factors: int) -> np.ndarray:
        """
        Denoises correlation by replacing small (residual) eigenvalues with their average.
        """
        evals = np.diag(eigenvals).copy()

        # Replace residual eigenvalues with their average
        evals[n_factors:] = evals[n_factors:].sum() / float(evals.shape[0] - n_factors)

        # Reconstruct the denoised correlation matrix
        evals = np.diag(evals)
        corr1 = np.dot(eigenvecs, evals).dot(eigenvecs.T)

        # Ensure it's a valid correlation matrix
        corr = RiskMetrics.cov_to_corr(corr1)

        return corr

    @staticmethod
    def deonoised_corr_targeted_shrinkage(eigenvals: np.ndarray, eigenvecs: np.ndarray, n_factors: int,
                                          alpha: float = 0) -> np.ndarray:
        """
        Denoises correlation using targeted shrinkage (convex combination).
        """
        # Get signal (left) and noise (right) components
        eigenvals_signal, eigenvecs_signal = eigenvals[:n_factors, :n_factors], eigenvecs[:, :n_factors]
        eigenvals_noise, eigenvecs_noise = eigenvals[n_factors:, n_factors:], eigenvecs[:, n_factors:]

        # Reconstruct correlation components
        corr_signal = np.dot(eigenvecs_signal, eigenvals_signal).dot(eigenvecs_signal.T)
        corr_noise = np.dot(eigenvecs_noise, eigenvals_noise).dot(eigenvecs_noise.T)

        # Targeted shrinkage formula (López de Prado)
        corr_denoised = corr_signal + alpha * corr_noise + (1 - alpha) * np.diag(np.diag(corr_noise))

        return corr_denoised

    @staticmethod
    def denoise_corr(eigenvals: np.ndarray, eigenvecs: np.ndarray, n_factors: int,
                     method: str = 'constant_residual_eigenval', alpha: float = 0) -> np.ndarray:
        """
        Dispatcher for correlation matrix denoising methods.
        """
        if method == 'constant_residual_eigenval':
            return RiskMetrics.denoised_corr_constant_resid_eigenval(eigenvals, eigenvecs, n_factors)
        elif method == 'targeted_shrinkage':
            return RiskMetrics.deonoised_corr_targeted_shrinkage(eigenvals, eigenvecs, n_factors, alpha)
        else:
            raise ValueError('Invalid method. Please choose from: constant_residual_eigenval, targeted_shrinkage')

    @staticmethod
    def detone_corr(eigenvals: np.ndarray, eigenvecs: np.ndarray, n_factors: int) -> np.ndarray:
        """
        Detones (removes the market factor) from the denoised correlation matrix.
        """
        # constant residual denoised correlation matrix
        corr = RiskMetrics.denoise_corr(eigenvals, eigenvecs, n_factors, method='constant_residual_eigenval', alpha=0)

        # extract the market factor (the largest eigenvalue/eigenvector)
        eigenvals_mkt, eigenvecs_mkt = eigenvals[: 1, : 1], eigenvecs[:, :1]

        # subtract the market factor component
        corr_detoned = corr - np.dot(eigenvecs_mkt, eigenvals_mkt).dot(eigenvecs_mkt.T)

        # rescale to ensure it remains a valid correlation matrix (diagonal elements = 1)
        corr_rescaled = RiskMetrics.cov_to_corr(corr_detoned)

        return corr_rescaled

    @staticmethod
    def denoised_covariance(returns: pd.DataFrame,
                            method: str = 'constant_residual_eigenval',
                            bwidth: float = 0.01,
                            alpha: float = 0,
                            detone: bool = False,
                            empirical_cov: Optional[Union[pd.DataFrame, np.ndarray]] = None
                            ) -> np.ndarray:
        """
        Compute the denoised covariance matrix.

        See "Machine Learning for Asset Managers". Elements in Quantitative Finance. Lòpez de Prado (2020).

        Parameters
        ----------
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
        np.ndarray
            Denoised covariance matrix.
        """
        returns = RiskMetrics._validate_returns(returns)

        # calculate the empirical covariance if not provided
        if empirical_cov is None:
            empirical_cov = RiskMetrics.covariance(returns)

        # calculate q (T/N: number of observations / number of variables)
        num_obs, num_vars = returns.shape
        q = num_obs / num_vars

        # convert to correlation matrix
        corr_matrix = RiskMetrics.cov_to_corr(empirical_cov)
        std_devs = np.sqrt(np.diag(empirical_cov))

        # compute PCA
        eigenvalues, eigenvectors = RiskMetrics.get_pca(corr_matrix)

        # find the maximum theoretical eigenvalue (Lambda Max)
        eigen_max, _ = RiskMetrics.find_max_eigenval(np.diag(eigenvalues), q, bwidth)

        # determine the number of signal factors (eigenvalues > Lambda Max)
        n_factors = np.sum(np.diag(eigenvalues) > eigen_max)

        # denoise and/or detone the correlation matrix
        if detone:
            corr_denoised = RiskMetrics.detone_corr(eigenvalues, eigenvectors, n_factors)
        else:
            corr_denoised = RiskMetrics.denoise_corr(eigenvalues, eigenvectors, n_factors, method=method, alpha=alpha)

        # reconstruct the covariance matrix
        cov_matrix_denoised = RiskMetrics.corr_to_cov(corr_denoised, std_devs)

        return cov_matrix_denoised

    @staticmethod
    def turbulence_index(returns: pd.DataFrame, component: str = 'turbulence') -> pd.Series:
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
        returns: pd.DataFrame
            DataFrame of asset or strategy returns.
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
        ret_demeaned = (returns - returns.mean()).values

        # covariance matrix
        cov_matrix = RiskMetrics.covariance(returns).values

        # inv cov matrix
        inv_cov_matrix = np.linalg.pinv(cov_matrix)

        # mahalanobis distance
        turb = np.diagonal(ret_demeaned @ inv_cov_matrix @ ret_demeaned.T) / returns.shape[1]

        # compute correlation blind cov matrix (magnitude)
        if component != 'turbulence':
            cov_matrix_mag = np.diag(np.diag(cov_matrix))
            inv_cov_matrix_mag = np.linalg.pinv(cov_matrix_mag)
            magnitude_surprise = (np.diagonal(ret_demeaned @ inv_cov_matrix_mag @ ret_demeaned.T) /
                                  returns.shape[1])
            corr_surprise = turb / magnitude_surprise

        # component
        if component == 'correlation_surprise':
            return pd.DataFrame(corr_surprise, index=returns.index, columns=[component])
        elif component == 'magnitude_surprise':
            return pd.DataFrame(magnitude_surprise, index=returns.index, columns=[component])
        else:
            return pd.DataFrame(turb, index=returns.index, columns=[component])
