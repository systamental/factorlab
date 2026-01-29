import pandas as pd
import numpy as np
from scipy.linalg import orth
from scipy.optimize import linear_sum_assignment

from typing import Optional, Union, Any, Tuple
from sklearn.decomposition import PCA


class PCAWrapper:
    """
    Principal component analysis wrapper class.
    """
    def __init__(self,
                 data: Union[np.array, pd.DataFrame],
                 n_components: Optional[int] = None,
                 missing_values: str = 'drop_rows',
                 **kwargs: Any
                 ):
        """
        Initialize PCAWrapper object.

        Parameters
        ----------
        data: np.ndarray or pd.DataFrame
            Data matrix for principal component analysis
        n_components: int or float, default None
            Number of principal components, or percentage of variance explained.
        missing_values: str, {'drop_any_rows', 'drop_all_rows_any_cols'}, default 'drop_any_rows'
            How to handle missing values in the data.
            'drop_any_rows' - drop rows with any missing values,
            'drop_all_rows_any_cols' - drop rows where all columns have missing values and
            columns with any missing values.
        **kwargs: Optional keyword arguments, for PCA object. See sklearn.decomposition.PCA for details:
            https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
        """
        self.raw_data = data
        self.missing_values = missing_values
        self.data = self.remove_missing()
        self.n_components = min(self.data.shape) if n_components is None else n_components
        self.kwargs = kwargs
        self.index = data.dropna().index if isinstance(data, pd.DataFrame) else None
        self.data_window = self.data.copy()
        self.pca = self.create_pca_instance()
        self.eigenvecs = None
        self.expl_var_ratio = None
        self.pcs = None

    def remove_missing(self) -> np.array:
        """
        Remove missing values from data.

        Returns
        -------
        data: np.ndarray
            Data matrix with missing values removed.
        """
        if isinstance(self.raw_data, pd.DataFrame):
            if self.missing_values == 'drop_rows':
                self.data = self.raw_data.dropna().to_numpy(dtype=np.float64)
            elif self.missing_values == 'drop_all_rows_any_cols':
                self.data = self.raw_data.dropna(how='all').dropna(axis=1).to_numpy(dtype=np.float64)

        elif isinstance(self.raw_data, np.ndarray):
            if self.missing_values == 'drop_rows':
                self.data = self.raw_data[~np.isnan(self.raw_data).any(axis=1)].astype(np.float64)

            elif self.missing_values == 'drop_all_rows_any_cols':
                not_all_nan_rows = ~np.isnan(self.raw_data).all(axis=1)
                temp = self.raw_data[not_all_nan_rows]
                not_any_nan_cols = ~np.isnan(temp).any(axis=0)
                self.data = temp[:, not_any_nan_cols].astype(np.float64)

        else:
            raise ValueError(f"Data must be pd.DataFrame or np.array.")

        return self.data

    def create_pca_instance(self) -> PCA:
        """
        Perform PCA.

        Returns
        -------
        pca: PCA
            PCA object.
        """
        self.pca = PCA(n_components=self.n_components, **self.kwargs)

        return self.pca

    def get_eigenvectors(self) -> np.ndarray:
        """
        Get eigenvectors from SVD decomposition.

        Returns
        -------
        eigenvecs: np.ndarray
            Eigenvectors from SVD decomposition.
        """
        # pca
        self.pca.fit(self.data_window)

        # eigenvectors
        self.eigenvecs = self.pca.components_.T

        return self.eigenvecs

    def correct_sign_pc1(self, pcs: np.ndarray) -> np.ndarray:
        """
        Constrain the sign of the first principal component to be consistent with the mean of the cross-section.

        Parameters
        ----------
        pcs: np.ndarray
            Principal components.

        Returns
        -------
        pcs: np.ndarray
            Principal components with first PC sign corrected.
        """
        # check sign of first PC
        if np.dot(pcs[:, 0], np.mean(self.data_window, axis=1)) < 0:
            pcs *= -1

        return pcs

    def get_pcs(self) -> Union[np.ndarray, pd.DataFrame]:
        """
        Get principal components.

        Returns
        -------
        pcs: np.ndarray or pd.DataFrame
            Principal components.
        """
        # pcs
        self.pcs = self.pca.fit_transform(self.data_window)

        # flip first PC
        self.pcs = self.correct_sign_pc1(self.pcs)

        # add index and cols if available
        if self.index is not None:
            self.pcs = pd.DataFrame(self.pcs, index=self.index[-self.pcs.shape[0]:])

        return self.pcs

    def get_expl_var_ratio(self) -> np.ndarray:
        """
        Get explained variance ratio from sklearn PCA implementation.

        Returns
        -------
        explained_var_ratio: np.ndarray
            Explained variance ratio.
        """
        # explained variance ratio
        self.expl_var_ratio = self.pca.fit(self.data_window).explained_variance_ratio_

        return self.expl_var_ratio

    def get_rolling_pcs(self, window_size: Optional[int] = None) -> Union[np.ndarray, pd.DataFrame]:
        """
        Get rolling principal components.

        Parameters
        ----------
        window_size: int, default None
            Size of rolling window (number of observations).

        Returns
        -------
        rolling_pcs: np.ndarray or pd.DataFrame
            Rolling principal components.
        """
        # window size
        if window_size is None:
            window_size = self.n_components
        if window_size < self.n_components:
            self.n_components = window_size

        # get rolling window pcs
        out = None

        # loop through rows of data
        for row in range(self.data.shape[0] - window_size + 1):

            # set rolling window
            self.data_window = self.data[row: row + window_size, :]
            # get pcs
            self.get_pcs()

            # add output to array
            if row == 0:
                if isinstance(self.pcs, pd.DataFrame):
                    out = self.pcs.values[-1]
                else:
                    out = self.pcs[-1]
            else:
                # add output to array
                if isinstance(self.pcs, pd.DataFrame):
                    out = np.vstack([out, self.pcs.values[-1]])
                else:
                    out = np.vstack([out, self.pcs[-1]])

        self.pcs = out

        # add index and cols if available
        if self.index is not None:
            self.pcs = pd.DataFrame(self.pcs, index=self.index[-self.pcs.shape[0]:])

        return self.pcs

    def get_rolling_expl_var_ratio(self, window_size: Optional[int] = None) -> Union[np.ndarray, pd.DataFrame]:
        """
        Get rolling explained variance ratio.

        Parameters
        ----------
        window_size: int, default None
            Size of rolling window (number of observations).

        Returns
        -------
        rolling_expl_var_ratio: np.ndarray or pd.DataFrame
            Rolling explained variance ratio.
        """
        # window size
        if window_size is None:
            window_size = self.n_components
        if window_size < self.n_components:
            raise ValueError(f"Window size {window_size} is less than {self.n_components} n_components.")

        # get rolling window expl var
        out = None

        # loop through rows of data
        for row in range(self.data.shape[0] - window_size + 1):

            # set rolling window
            self.data_window = self.data[row: row + window_size, :]
            # get expl var ratio
            self.get_expl_var_ratio()

            if row == 0:
                out = self.expl_var_ratio
            else:
                out = np.vstack([out, self.expl_var_ratio])

        self.expl_var_ratio = out

        # add index and cols if available
        if self.expl_var_ratio is not None:
            self.expl_var_ratio = pd.DataFrame(self.expl_var_ratio, index=self.index[-self.expl_var_ratio.shape[0]:])

        return self.expl_var_ratio

    def get_expanding_pcs(self, min_obs: Optional[int] = None) -> Union[np.ndarray, pd.DataFrame]:
        """
        Get rolling principal components.

        Parameters
        ----------
        min_obs: int, default 1
            Mininum number of observations for expanding window computation.

        Returns
        -------
        exp_pcs: np.ndarray or pd.DataFrame
            Expanding principal components.
        """
        # min obs
        if min_obs is None:
            min_obs = min(self.data.shape)
        if min_obs < self.n_components:
            raise ValueError(f"Minimum observations {min_obs} is less than {self.n_components} n_components.")

        # get expanding window pcs
        out = None

        # loop through rows of data
        for row in range(min_obs, self.data.shape[0] + 1):

            # set expanding window
            self.data_window = self.data[: row, :]

            # get pcs
            self.get_pcs()

            if row == min_obs:
                if isinstance(self.pcs, pd.DataFrame):
                    out = self.pcs.values[-1]
                else:
                    out = self.pcs[-1]
            else:
                # add output to array
                if isinstance(self.pcs, pd.DataFrame):
                    out = np.vstack([out, self.pcs.values[-1]])
                else:
                    out = np.vstack([out, self.pcs[-1]])

        self.pcs = out

        # add index and cols if available
        if self.index is not None:
            self.pcs = pd.DataFrame(self.pcs, index=self.index[-self.pcs.shape[0]:], columns=range(self.pcs.shape[1]))

        return self.pcs

    def get_expanding_expl_var_ratio(self, min_obs: Optional[int] = None) -> Union[np.ndarray, pd.DataFrame]:
        """
        Get rolling explained variance ratio.

        Parameters
        ----------
        min_obs: int, default 1
            Mininum number of observations for expanding window computation.

        Returns
        -------
        exp_expl_var_ratio: np.ndarray or pd.DataFrame
            Expanding explained variance ratio.
        """
        # min obs
        if min_obs is None:
            min_obs = min(self.data.shape)
        if min_obs < self.n_components:
            raise ValueError(f"Minimum observations {min_obs} is less than {self.n_components} n_components.")

        # get expanding window expl var
        out = None

        # loop through rows of data
        for row in range(min_obs, self.data.shape[0] + 1):

            # set expanding window
            self.data_window = self.data[: row, :]
            # get expl var ratio
            self.get_expl_var_ratio()

            if row == min_obs:
                out = self.expl_var_ratio
            else:
                out = np.vstack([out, self.expl_var_ratio])

        self.expl_var_ratio = out

        # add index and cols if available
        if self.index is not None:
            self.expl_var_ratio = pd.DataFrame(self.expl_var_ratio, index=self.index[-self.expl_var_ratio.shape[0]:])

        return self.expl_var_ratio


class R2PCA:
    """
    R2-PCA principal component analysis class.

    See details for R2-PCA: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4400158

    Parameters
    ----------
    data: np.ndarray or pd.DataFrame
        Data matrix for principal component analysis
    n_components: int, default None
        Number of principal components.
    svd_solver: str, {'auto', 'full', 'arpack', 'randomized'}, default='auto'
        SVD solver to use.
        See sklearn.decomposition.PCA for details:
        https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
    **kwargs: Optional keyword arguments, for PCA object. See sklearn.decomposition.PCA for details.

    """

    def __init__(self,
                 data: Union[np.ndarray, pd.DataFrame],
                 n_components: Optional[int] = None,
                 svd_solver: str = 'auto',
                 **kwargs: Any
                 ):
        """
        Initialize R2-PCA object.
        """
        self.raw_data = data
        self.df = pd.DataFrame(data) if not isinstance(data, pd.DataFrame) else data
        self.n_components = n_components
        self.svd_solver = svd_solver
        self.pca_kwargs = kwargs

        # Internal state
        self.eigenvecs = None
        self.expl_var_ratio = None
        self.pcs = None

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
        # drop missing values
        clean_df = window_df.dropna(how='all').dropna(axis=1, how='any')
        self.n_components = min(clean_df.shape) if self.n_components is None else self.n_components

        # check if enough data
        if clean_df.empty or clean_df.shape[1] < (self.n_components or 1):
            raise ValueError("Not enough data after cleaning missing values for PCA computation.")

        return clean_df

    def correct_sign_pc1(self, pcs: np.ndarray, data_window: np.ndarray) -> np.ndarray:
        """Corrects sign of first PC based on cross-sectional mean to align with market portfolio.

        Parameters
        ----------
        pcs : np.ndarray
            Principal components.
        data_window : np.ndarray
            Data matrix for the current window.

        Returns
        -------
        pcs : np.ndarray
            Principal components with first PC sign corrected to match market factor convention.

        """
        if pcs.shape[1] > 0:
            if np.dot(pcs[:, 0], np.mean(data_window, axis=1)) < 0:
                pcs *= -1
        return pcs

    def get_fixed_results(self) -> None:
        """
        Get PCA results for fixed data window.
        """
        # clean data
        clean_df = self._get_clean_window_data(self.df)

        # Fit PCA
        pca = PCA(n_components=self.n_components, svd_solver=self.svd_solver, **self.pca_kwargs)
        pca_transformed = pca.fit_transform(clean_df)

        # eigenvectors
        self.eigenvecs = pca.components_.T
        # pcs, correct sign
        self.pcs = self.correct_sign_pc1(pca_transformed, clean_df.values)
        if isinstance(clean_df, pd.DataFrame):
            self.pcs = pd.DataFrame(self.pcs, index=clean_df.index, columns=range(self.pcs.shape[1]))
        # explained variance ratio
        self.expl_var_ratio = pca.explained_variance_ratio_

    def get_pcs(self) -> Union[np.array, pd.DataFrame]:
        """
        Get principal components.

        Returns
        -------
        pcs: np.array or pd.DataFrame
            Principal components.
        """
        self.get_fixed_results()

        return self.pcs

    def get_eigenvectors(self) -> np.ndarray:
        """
        Get eigenvectors from SVD decomposition.

        Returns
        -------
        eigenvecs: np.ndarray
            Eigenvectors from SVD decomposition.
        """
        if self.eigenvecs is None:
            self.get_fixed_results()

        return self.eigenvecs

    def get_expl_var_ratio(self) -> np.ndarray:
        """
        Get explained variance ratio from sklearn PCA implementation.

        Returns
        -------
        explained_var_ratio: np.ndarray
            Explained variance ratio.
        """
        if self.expl_var_ratio is None:
            self.get_fixed_results()

        return self.expl_var_ratio

    # def align_and_correct(self,
    #                       current_ev: np.ndarray,
    #                       clean_columns: pd.Index,
    #                       ref_ev: Optional[np.ndarray]) -> np.ndarray:
    #     """
    #     Maps current eigenvectors to the full universe and corrects for
    #     ordering and sign flips to ensure factor continuity.
    #
    #     Parameters
    #     ----------
    #     current_ev : np.ndarray
    #         Eigenvectors from the current window (N_window x K)
    #     clean_columns : pd.Index
    #         The tickers present in the current window.
    #     ref_ev : np.ndarray, optional
    #         Reference eigenvectors from the previous window for alignment (N_total x K).
    #
    #     Returns
    #     -------
    #     aligned_ev : np.ndarray
    #         Aligned and sign-corrected eigenvectors mapped to the full universe.
    #     """
    #     # map to full universe: create (N_total x K) matrix
    #     full_ev = pd.DataFrame(0.0, index=self.df.columns, columns=range(current_ev.shape[1]))
    #     current_ev_df = pd.DataFrame(current_ev, index=clean_columns)
    #     full_ev.update(current_ev_df)
    #     full_ev_np = full_ev.values
    #
    #     if ref_ev is None:
    #         return full_ev_np
    #
    #     # ordering consistency (Hungarian Algorithm)
    #     # compute a cost matrix based on absolute correlation to
    #     # find the best match between current and reference vectors
    #     corr_matrix = np.dot(full_ev_np.T, ref_ev)
    #     cost_matrix = -np.abs(corr_matrix)
    #     row_ind, col_ind = linear_sum_assignment(cost_matrix)
    #
    #     # reorder current eigenvectors to match reference
    #     reordered_ev = full_ev_np[:, col_ind]
    #
    #     # sign consistency
    #     for j in range(reordered_ev.shape[1]):
    #         dot_prod = np.dot(reordered_ev[:, j], ref_ev[:, j])
    #         if dot_prod < 0:
    #             reordered_ev[:, j] *= -1
    #
    #     return reordered_ev

    def align_and_correct(self,
                          current_ev: np.ndarray,
                          clean_columns: pd.Index,
                          ref_ev: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Maps current eigenvectors to the full universe and corrects for
        ordering and sign flips to ensure factor continuity.
        """
        # map to full universe: create (N_total x K) matrix
        full_ev = pd.DataFrame(0.0, index=self.df.columns, columns=range(current_ev.shape[1]))
        current_ev_df = pd.DataFrame(current_ev, index=clean_columns)
        full_ev.update(current_ev_df)
        full_ev_np = full_ev.values

        if ref_ev is None:
            # Modified: Return identity mapping (0, 1, 2...) if no reference exists
            return full_ev_np, np.arange(current_ev.shape[1])

        # ordering consistency (Hungarian Algorithm)
        corr_matrix = np.dot(full_ev_np.T, ref_ev)
        cost_matrix = -np.abs(corr_matrix)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # reorder current eigenvectors to match reference
        # Modified: Using col_ind to ensure we track which raw PC went where
        reordered_ev = full_ev_np[:, col_ind]

        # sign consistency
        for j in range(reordered_ev.shape[1]):
            dot_prod = np.dot(reordered_ev[:, j], ref_ev[:, j])
            if dot_prod < 0:
                reordered_ev[:, j] *= -1

        # Modified: Return the reordered eigenvectors AND the column indices for the variance ratio
        return reordered_ev, col_ind

    def get_rolling_results(self, window_size: int):
        """
        Computes rolling PCA results with cross-window consistency.
        """
        if window_size > len(self.df):
            raise ValueError("Window size exceeds data length.")

        all_pcs = []
        all_ev_ratios = []
        all_eigenvecs = []

        ref_ev = None
        dates = self.df.index[window_size - 1:]

        for i in range(len(self.df) - window_size + 1):
            window_raw = self.df.iloc[i: i + window_size]
            clean_df = self._get_clean_window_data(window_raw)

            if clean_df.empty:
                continue

            # Determine components for this window
            n_comps = min(min(clean_df.shape), self.n_components)

            # Fit PCA
            pca = PCA(n_components=n_comps, svd_solver=self.svd_solver, **self.pca_kwargs)
            pca.fit(clean_df)

            # Align to full universe and correct ordering/sign
            current_raw_ev = pca.components_.T  # (N_window x K)
            corrected_ev, col_map = self.align_and_correct(current_raw_ev, clean_df.columns, ref_ev)

            # Market factor alignment (PC1 correction)
            if np.sum(corrected_ev[:, 0]) < 0:  # if sum of loadings negative, flip sign
                corrected_ev[:, 0] *= -1

            # Update reference eigenvectors for next window to ensure continuity
            ref_ev = corrected_ev

            # Compute PCs
            last_row_full = clean_df.iloc[-1].reindex(self.df.columns, fill_value=0).values  # reindex to full universe
            aligned_pcs_last_row = np.dot(last_row_full, corrected_ev)  # project last row

            # Reorder Explained Variance Ratio
            raw_ev_ratios = pca.explained_variance_ratio_
            aligned_ratios = raw_ev_ratios[col_map]

            # Store results
            all_pcs.append(aligned_pcs_last_row.flatten())  # last row only

            # Eigenvectors (Storing as long-format for the full universe)
            ev_df = pd.DataFrame(corrected_ev,
                                 index=self.df.columns,
                                 columns=[f"EV{j + 1}" for j in range(corrected_ev.shape[1])])
            ev_df['date'] = clean_df.index[-1]
            all_eigenvecs.append(ev_df.reset_index().rename(columns={'index': 'ticker'}))

            # EV ratio (padded to target dimension)
            ev_ratio = np.full(self.n_components, np.nan)
            ev_ratio[:len(aligned_ratios)] = aligned_ratios
            all_ev_ratios.append(ev_ratio)

        # format outputs
        pc_cols = [f"PC{j + 1}" for j in range(self.n_components)]
        self.pcs = pd.DataFrame(all_pcs, index=dates, columns=pc_cols[:len(all_pcs[0])])
        self.expl_var_ratio = pd.DataFrame(all_ev_ratios, index=dates, columns=pc_cols)
        self.eigenvecs = pd.concat(all_eigenvecs).set_index(['date', 'ticker']).sort_index()

    def get_expanding_results(self, min_obs: int):
        """
        Computes expanding PCA results with cross-window consistency.
        """
        if min_obs > len(self.df):
            raise ValueError("Minimum observations exceed data length.")

        all_pcs = []
        all_ev_ratios = []
        all_eigenvecs = []

        ref_ev = None
        dates = self.df.index[min_obs - 1:]

        for i in range(min_obs, len(self.df) + 1):
            window_raw = self.df.iloc[:i]
            clean_df = self._get_clean_window_data(window_raw)

            # Determine components for this window
            n_comps = min(min(clean_df.shape), self.n_components)

            # Fit PCA
            pca = PCA(n_components=n_comps, svd_solver=self.svd_solver, **self.pca_kwargs)
            pca.fit(clean_df)

            # Align to full universe and correct ordering/sign
            current_raw_ev = pca.components_.T  # (N_window x K)
            corrected_ev, col_map = self.align_and_correct(current_raw_ev, clean_df.columns, ref_ev)

            # Market factor alignment (PC1 correction)
            if np.sum(corrected_ev[:, 0]) < 0:  # Changed: Standardized PC1 sign check
                corrected_ev[:, 0] *= -1

            # Update reference for next window to ensure continuity
            ref_ev = corrected_ev

            # Compute PCs
            last_row_full = clean_df.iloc[-1].reindex(self.df.columns, fill_value=0).values  # Reindex to full universe
            aligned_pcs_last_row = np.dot(last_row_full, corrected_ev)  # Project last row

            # Reorder Explained Variance Ratio
            raw_ev_ratios = pca.explained_variance_ratio_
            aligned_ratios = raw_ev_ratios[col_map]

            # Store results
            all_pcs.append(aligned_pcs_last_row.flatten())

            # EV Ratio (padded to target dimension)
            ev_ratio = np.full(self.n_components, np.nan)
            ev_ratio[:len(aligned_ratios)] = aligned_ratios
            all_ev_ratios.append(ev_ratio)

            # Eigenvectors (Storing as long-format for the full universe)
            ev_df = pd.DataFrame(corrected_ev,
                                 index=self.df.columns,
                                 columns=[f"EV{j + 1}" for j in range(corrected_ev.shape[1])])
            ev_df['date'] = clean_df.index[-1]
            all_eigenvecs.append(ev_df.reset_index().rename(columns={'index': 'ticker'}))

        # Format outputs
        pc_cols = [f"PC{j + 1}" for j in range(self.n_components)]
        self.pcs = pd.DataFrame(all_pcs, index=dates, columns=pc_cols[:len(all_pcs[0])])
        self.expl_var_ratio = pd.DataFrame(all_ev_ratios, index=dates, columns=pc_cols)
        self.eigenvecs = pd.concat(all_eigenvecs).set_index(['date', 'ticker']).sort_index()

    def get_rolling_pcs(self, window_size: int) -> pd.DataFrame:
        """Returns the rolling principal component time series."""
        if self.pcs is None:
            self.get_rolling_results(window_size)
        return self.pcs

    def get_rolling_eigenvectors(self, window_size: int) -> pd.DataFrame:
        """Returns the rolling eigenvectors mapped to the full universe."""
        if self.eigenvecs is None:
            self.get_rolling_results(window_size)
        return self.eigenvecs

    def get_rolling_expl_var_ratio(self, window_size: int) -> pd.DataFrame:
        """Returns the rolling explained variance ratios."""
        if self.expl_var_ratio is None:
            self.get_rolling_results(window_size)
        return self.expl_var_ratio

    def get_expanding_pcs(self, min_obs: int) -> pd.DataFrame:
        """Returns the expanding principal component time series."""
        if self.pcs is None:
            self.get_expanding_results(min_obs)
        return self.pcs

    def get_expanding_eigenvectors(self, min_obs: int) -> pd.DataFrame:
        """Returns the expanding eigenvectors mapped to the full universe."""
        if self.eigenvecs is None:
            self.get_expanding_results(min_obs)
        return self.eigenvecs

    def get_expanding_expl_var_ratio(self, min_obs: int) -> pd.DataFrame:
        """Returns the expanding explained variance ratios."""
        if self.expl_var_ratio is None:
            self.get_expanding_results(min_obs)
        return self.expl_var_ratio


class PPCA:
    """
    Probabilistic PCA class.
    """
    def __init__(self,
                 data: Union[np.array, pd.DataFrame],
                 min_obs: int = 10,
                 min_feat: int = 5,
                 n_components: Optional[int] = None,
                 thresh: float = 1E-4
                 ):
        """
        Initialize PPCA object.

        Parameters
        ----------
        data: np.array or pd.DataFrame
            Data matrix for principal component analysis
        min_obs: int, default 10
            Minimum number of observations for each feature (col).
        min_feat: int, default 5
            Minimum number of features for each observation (row).
        n_components: int, default None
            Number of principal components.
        thresh: float, default 1E-4
            Threshold for convergence.
        """
        self.n_cols = None
        self.n_rows = None
        self.feature_means = None
        self.tot_missing_vals = None
        self.missing_vals = None
        self.index = None
        self.raw_data = data
        self.min_obs = min_obs
        self.min_feat = min_feat
        self.data = self.preprocess_data()
        self.n_components = min(self.data.shape) if n_components is None else n_components
        self.thresh = thresh
        self.data_window = self.data.copy()
        self.eigenvecs = None
        self.eigenvals = None
        self.expl_var_ratio = None
        self.pcs = None

    def preprocess_data(self) -> np.array:
        """
        Preprocess data.

        Returns
        -------
        data: np.array
            Expected complete data
        """
        # convert to np.array
        if isinstance(self.raw_data, pd.DataFrame):
            self.data = self.raw_data.to_numpy(dtype=np.float64)
            self.index = self.raw_data.dropna(thresh=self.min_obs, axis=1).dropna(thresh=self.min_feat).index
        elif isinstance(self.raw_data, np.ndarray):
            self.data = self.raw_data
            self.index = None
        else:
            raise ValueError(f"Data must be pd.DataFrame or np.array.")

        # remove cols with less than min obs
        valid_cols = np.sum(~np.isnan(self.data), axis=0) >= self.min_obs
        self.data = self.data[:, valid_cols]
        # remove rows with less than min features
        valid_rows = np.sum(~np.isnan(self.data), axis=1) >= self.min_feat
        self.data = self.data[valid_rows]

        # compute missing vals
        self.missing_vals = np.isnan(self.data)
        self.tot_missing_vals = np.sum(self.missing_vals)
        # impute mean val for missing vals in features
        self.feature_means = np.nanmean(self.data, axis=0)

        # set n_rows, n_cols
        self.n_rows, self.n_cols = self.data.shape

        # reconstructed data
        self.data = self.data - np.tile(self.feature_means, (self.n_rows, 1))
        if self.tot_missing_vals > 0:
            self.data[self.missing_vals] = 0

        return self.data

    def em_algo(self) -> np.array:
        """
        EM algorithm.

        Returns
        -------
        C: np.array
            Data matrix.
        """
        # initialize
        C = np.random.normal(loc=0.0, scale=1.0, size=(self.n_cols, self.n_components))
        CtC = C.T @ C
        X = self.data @ C @ np.linalg.inv(CtC)
        recon_data = X @ C.T
        recon_data[self.missing_vals] = 0
        ss = np.sum((recon_data - self.data) ** 2) / (self.n_rows * self.n_cols - self.tot_missing_vals)

        # EM Iterations
        count = 1
        old = np.inf
        threshold = self.thresh  # min relative change in obj. fcn to continue
        while count:
            Sx = np.linalg.inv(np.identity(self.n_components) + CtC / ss)  # E-step, covariances
            ss_old = ss
            if self.tot_missing_vals > 0:
                proj = X @ C.T
                self.data[self.missing_vals] = proj[self.missing_vals]
            # E-step: expected values
            X = self.data @ C @ Sx / ss
            # M-step: update parameters
            SumXtX = X.T @ X
            C = self.data.T @ X @ (SumXtX + self.n_rows * Sx).T @ \
                np.linalg.inv(((SumXtX + self.n_rows * Sx) @ (SumXtX + self.n_rows * Sx).T))
            CtC = C.T @ C
            ss = (np.sum((X @ C.T - self.data) ** 2) + self.n_rows * np.sum(CtC * Sx) + self.tot_missing_vals
                  * ss_old) / (self.n_rows * self.n_cols)

            # transform Sx det into np longdouble to deal with high dimensionality
            Sx_det = np.min(Sx).astype(np.longdouble) ** np.shape(Sx)[0] * np.linalg.det(Sx / np.min(Sx))
            objective = (
                    self.n_rows * self.n_cols
                    + self.n_rows * (
                            self.n_cols * np.log(ss, where=ss > 0)
                            + np.trace(Sx) - np.log(Sx_det, where=Sx_det > 0)
                    )
                    + np.trace(SumXtX) - self.tot_missing_vals
                    * np.log(ss_old, where=ss_old > 0)
            )
            rel_ch = np.abs(1 - objective / old)
            old = objective

            # check convergence
            count += 1
            if rel_ch < threshold and count > 5:
                count = 0

        return C

    def decompose(self) -> np.array:
        """
        Decompose data matrix using EM algorithm.

        Returns
        -------
        C: np.array
            Decomposed data matrix.
        """
        # initialize EM algo
        C = self.em_algo()
        # cov matrix
        C = orth(C)
        cov_matrix = np.cov((self.data @ C).T)
        # eigen decomp
        eigenvals, eigenvecs = np.linalg.eig(cov_matrix)
        # sort
        idx = np.argsort(eigenvals)[::-1]
        self.eigenvecs = eigenvecs[:, idx]
        self.eigenvals = eigenvals[idx]
        # add data mean to expected complete data
        self.data = self.data + np.tile(self.feature_means, (self.n_rows, 1))

        return C

    def get_eigenvectors(self) -> np.array:
        """
        Get eigenvectors.

        Returns
        -------
        eig: np.array
            Eigenvectors.
        """
        # decompose
        C = self.decompose()
        # flip sign
        C = np.dot(C, self.eigenvecs)

        # normalize the eigenvectors
        norm_eig = C.T / np.linalg.norm(C.T, axis=1, keepdims=True)
        # identify indices to flip based on the largest absolute values
        flip_indices = np.argmax(np.abs(norm_eig), axis=1)
        # sign
        sign = np.sign(C.T[np.arange(len(norm_eig)), flip_indices].reshape(-1, 1))
        # flip the sign of the corresponding eigenvectors
        eig = (C.T * sign).T

        return eig

    def get_pcs(self) -> Union[np.array, pd.DataFrame]:
        """
        Get principal components.

        Returns
        -------
        pcs: np.array or pd.DataFrame
            Principal components.
        """
        # get eig
        eig = self.get_eigenvectors()
        # transform pcs
        pcs = np.dot(self.data, eig)
        pcs = pcs[:, :self.n_components]

        # add index and cols if available
        if self.index is not None:
            pcs = pd.DataFrame(pcs, index=self.index[-pcs.shape[0]:], columns=range(pcs.shape[1]))

        return pcs

    def get_expl_var_ratio(self) -> np.array:
        """
        Get explained variance ratio.

        Returns
        -------
        var_exp: np.array
            Explained variance ratio.
        """
        # decompose
        self.decompose()
        # compute expl var
        self.expl_var_ratio = self.eigenvals / np.sum(self.eigenvals)

        return self.expl_var_ratio[: self.n_components]
