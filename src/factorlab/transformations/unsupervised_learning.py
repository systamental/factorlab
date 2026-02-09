import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from typing import Optional, Union, Any, Tuple
from scipy.optimize import linear_sum_assignment
from factorlab.core.base_transform import BaseTransform


class R2PCA(BaseTransform):
    """
    R2-PCA principal component analysis class.

    See details for R2-PCA: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4400158

    Parameters
    ----------
    n_components: int, default None
        Number of principal components.
    svd_solver: str, {'auto', 'full', 'arpack', 'randomized'}, default='auto'
        SVD solver to use.
        See sklearn.decomposition.PCA for details:
        https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
    **kwargs: Optional keyword arguments, for PCA object. See sklearn.decomposition.PCA for details.

    R2-PCA principal component analysis transform.

    See details for R2-PCA: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4400158
    """

    def __init__(self,
                 n_components: Optional[int] = None,
                 svd_solver: str = 'auto',
                 **kwargs: Any
                 ):
        super().__init__()
        self.n_components = n_components
        self.svd_solver = svd_solver
        self.pca_kwargs = kwargs

        # Internal state
        self._data: Optional[pd.DataFrame] = None
        self.ticker_to_idx: dict[str, int] = {}

        # Results storage
        self.pcs: Optional[pd.DataFrame] = None
        self.eigenvecs: Optional[pd.DataFrame] = None
        self.expl_var_ratio: Optional[Union[pd.DataFrame, np.ndarray]] = None
        self.alignment_scores: Optional[pd.DataFrame] = None

        # For rolling/expanding outputs
        self._window_outputs: dict[str, Any] = {}

    def _get_clean_window_data(self, window_df: pd.DataFrame) -> pd.DataFrame:
        """
        Filters a window to only include assets (columns) that have no
        missing values for the duration of this specific window.

        Parameters
        ----------
        window_df : pd.DataFrame
            DataFrame for the current rolling window.

        Returns
        -------
        clean_df : pd.DataFrame
            Cleaned DataFrame with rows and columns with missing values removed.
        """
        clean_df = window_df.dropna(how='all').dropna(axis=1, how='any')
        if clean_df.empty or clean_df.shape[1] < (self.n_components or 1):
            raise ValueError("Not enough data after cleaning missing values for PCA computation.")

        self.n_components = min(clean_df.shape) if self.n_components is None else self.n_components
        return clean_df

    @staticmethod
    def _normalize_data(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        Normalizes the data to have zero mean and unit variance.

        Parameters:
        -----------
        data : pd.DataFrame
            The data (Observations x Assets).

        Returns:
        --------
        normalized_data : pd.DataFrame
            The normalized data.
        """
        mu = data.mean()
        sigma = data.std().replace(0, 1)
        normalized_data = (data - mu) / sigma
        return normalized_data, mu, sigma

    @staticmethod
    def correct_pc1_sign(eigenvecs: np.ndarray, window_data: pd.DataFrame) -> np.ndarray:
        """
        Ensures PC1 is positively correlated with the cross-sectional market return.

        Parameters:
        -----------
        eigenvecs : np.ndarray
            Loadings matrix (Assets x Components) from PCA.fit().
        window_data : pd.DataFrame
            The returns data (Observations x Assets).

        Returns:
        --------
        eigenvecs : np.ndarray
            Loadings matrix with PC1 sign corrected.
        """
        pc1_loadings = eigenvecs[:, 0]
        pc1_ts = np.dot(window_data.values, pc1_loadings)
        market_proxy = window_data.mean(axis=1).values
        if np.dot(pc1_ts, market_proxy) < 0:
            eigenvecs *= -1
        return eigenvecs

    @staticmethod
    def _pad_vector(vec: np.ndarray, target_k: int) -> np.ndarray:
        """
        Pads a vector with NaNs to reach the target length of target_k.

        Parameters:
        -----------
        vec : np.ndarray
            The input vector to pad.
        target_k : int
            The desired length of the output vector.

        Returns:
        --------
        padded_vec : np.ndarray
            The input vector padded with NaNs to reach target_k length.

        """
        out = np.full(target_k, np.nan)
        out[:vec.shape[0]] = vec
        return out

    def fit(self,
            X: Union[pd.Series, pd.DataFrame],
            y: Optional[Union[pd.Series, pd.DataFrame]] = None
            ) -> 'R2PCA':
        """
        Fit the R2-PCA model to the data in X.

        Parameters
        ----------
        X: Union[pd.Series, pd.DataFrame]
            Input data to fit the model on. Should be a DataFrame of returns with shape (
        y: Optional[Union[pd.Series, pd.DataFrame]], optional
            Optional target data (not used in unsupervised learning, but included for API consistency).

        Returns
        -------
        self : R2PCA
            The fitted R2PCA instance.

        """
        df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        self._data = df
        self.ticker_to_idx = {ticker: i for i, ticker in enumerate(df.columns)}

        clean_df = self._get_clean_window_data(df)
        norm_df, mu, sigma = self._normalize_data(clean_df)

        target_k = self.n_components if self.n_components is not None else df.shape[1]
        k_eff = min(min(norm_df.shape), target_k)

        pca = PCA(n_components=k_eff, svd_solver=self.svd_solver, **self.pca_kwargs)
        pca.fit(norm_df)

        ev_raw = pca.components_.T
        ev_corrected = self.correct_pc1_sign(ev_raw, norm_df)

        # Map to full universe
        n_assets = df.shape[1]
        full_ev = np.zeros((n_assets, target_k))
        clean_indices = df.columns.get_indexer(clean_df.columns)
        full_ev[clean_indices, :k_eff] = ev_corrected

        self._fitted_params['mu'] = mu
        self._fitted_params['sigma'] = sigma
        self._fitted_params['clean_columns'] = clean_df.columns
        self._fitted_params['universe_cols'] = df.columns
        self._fitted_params['target_k'] = target_k
        self._fitted_params['k_eff'] = k_eff
        self._fitted_params['eigenvectors_full'] = full_ev
        self._fitted_params['expl_var_ratio_raw'] = pca.explained_variance_ratio_
        self._fitted_params['expl_var_ratio'] = self._pad_vector(pca.explained_variance_ratio_, target_k)
        self._fitted_params['alignment_scores'] = self._pad_vector(np.ones(k_eff), target_k)

        self._is_fitted = True
        return self

    def align(self, previous_params: dict) -> None:
        """
        Maps current eigenvectors to the full universe and aligns them to the reference eigenvectors
        from previous_params using the Hungarian algorithm for optimal matching,
        corrects signs to ensure positive correlation with reference eigenvectors,
        and pads results to target_k length.

        Parameters
        ----------
        previous_params: dict
            The fitted parameters from the previous window, containing 'eigenvectors_aligned' or 'eigenvectors_full'
            and 'k_eff' for reference alignment.

        """
        curr_ev = self._fitted_params['eigenvectors_full']
        ref_ev = previous_params.get('eigenvectors_aligned', previous_params.get('eigenvectors_full'))

        if ref_ev is None:
            self._fitted_params['eigenvectors_aligned'] = curr_ev
            return

        k_eff = self._fitted_params['k_eff']
        curr_ev_eff = curr_ev[:, :k_eff]
        ref_k_eff = previous_params.get('k_eff', ref_ev.shape[1])
        ref_ev_eff = ref_ev[:, :ref_k_eff]

        corr_matrix = np.dot(curr_ev_eff.T, ref_ev_eff)
        cost_matrix = -np.abs(corr_matrix)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        similarities = np.abs(corr_matrix[row_ind, col_ind])
        reordered_ev = curr_ev_eff[:, col_ind]

        for j in range(reordered_ev.shape[1]):
            if np.dot(reordered_ev[:, j], ref_ev_eff[:, j]) < 0:
                reordered_ev[:, j] *= -1

        target_k = self._fitted_params['target_k']
        aligned_full = np.zeros((curr_ev.shape[0], target_k))
        aligned_full[:, :reordered_ev.shape[1]] = reordered_ev

        self._fitted_params['eigenvectors_aligned'] = aligned_full
        self._fitted_params['col_map'] = col_ind
        self._fitted_params['alignment_scores'] = self._pad_vector(similarities, target_k)

        expl = self._fitted_params['expl_var_ratio_raw']
        mapped = expl[col_ind]
        self._fitted_params['expl_var_ratio'] = self._pad_vector(mapped, target_k)

    def transform(self, X: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
        """
        Transforms the input data X into principal component space using the fitted eigenvectors and
        normalization parameters.

        Parameters
        ----------
        X: Union[pd.Series, pd.DataFrame]
            Input data to transform. Should be a DataFrame of returns with shape (Observations x Assets).
        Returns
        -------
        pcs_df: pd.DataFrame
            DataFrame containing the principal component scores for each observation,
            with columns named 'PC1', 'PC2', ..., 'PCk' where k is the number of components specified by n_components
        """

        if not self._is_fitted:
            raise RuntimeError("Transform called before fit.")

        df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        cols = self._fitted_params['clean_columns']
        mu = self._fitted_params['mu']
        sigma = self._fitted_params['sigma']
        universe_cols = self._fitted_params['universe_cols']
        target_k = self._fitted_params['target_k']
        k_eff = self._fitted_params['k_eff']
        ev_full = self._fitted_params.get('eigenvectors_aligned', self._fitted_params['eigenvectors_full'])

        norm_df = (df[cols] - mu) / sigma
        clean_indices = universe_cols.get_indexer(cols)
        ev_clean = ev_full[clean_indices, :k_eff]
        pcs_k = np.dot(norm_df.values, ev_clean)

        pcs_full = np.full((pcs_k.shape[0], target_k), np.nan)
        pcs_full[:, :k_eff] = pcs_k
        pcs_df = pd.DataFrame(pcs_full, index=df.index, columns=[f"PC{i + 1}" for i in range(target_k)])

        # Expose rolling-compatible outputs for window wrappers
        self._window_outputs = {
            'eigenvectors_full': ev_full,
            'expl_var_ratio': self._fitted_params['expl_var_ratio'],
            'alignment_scores': self._fitted_params['alignment_scores'],
            'target_k': target_k,
            'universe_cols': universe_cols
        }

        return pcs_df
