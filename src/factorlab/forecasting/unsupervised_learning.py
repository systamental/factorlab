import pandas as pd
import numpy as np
from scipy.linalg import orth
from scipy.optimize import linear_sum_assignment

from typing import Optional, Union, Any, Tuple
from sklearn.decomposition import PCA
from tqdm import tqdm


class PCAWrapper:
    """
    Principal component analytics wrapper class.
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
            Data matrix for principal component analytics
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
    R2-PCA principal component analytics class.

    See details for R2-PCA: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4400158

    Parameters
    ----------
    data: np.ndarray or pd.DataFrame
        Data matrix for principal component analytics
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

        # ticker mapping
        self.ticker_to_idx = {ticker: i for i, ticker in enumerate(self.df.columns)}

        # Internal state
        self.eigenvecs = None
        self.expl_var_ratio = None
        self.pcs = None
        self.alignment_scores = None

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
        # check if enough data
        if clean_df.empty or clean_df.shape[1] < (self.n_components or 1):
            raise ValueError("Not enough data after cleaning missing values for PCA computation.")

        # n components
        self.n_components = min(clean_df.shape) if self.n_components is None else self.n_components

        return clean_df

    def _normalize_data(self, data: pd.DataFrame) -> pd.DataFrame:
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
        # mu, sigma
        mu = data.mean()
        sigma = data.std()

        normalized_data = (data - mu) / sigma.replace(0, 1)  # avoid division by zero

        return normalized_data

    def correct_pc1_sign(self, eigenvecs: np.ndarray, window_data: pd.DataFrame) -> np.ndarray:
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
        # compute PC1 time series
        pc1_loadings = eigenvecs[:, 0]
        pc1_ts = np.dot(window_data.values, pc1_loadings)

        # compute market proxy (cross-sectional mean at each time t)
        market_proxy = window_data.mean(axis=1).values

        # Direct alignment check
        if np.dot(pc1_ts, market_proxy) < 0:  # if dot product negative, flip sign
            eigenvecs *= -1

        return eigenvecs

    def get_fixed_results(self) -> None:
        """
        Get PCA results for fixed data window.
        """
        # clean data
        clean_df = self._get_clean_window_data(self.df)

        # normalize data
        norm_df = self._normalize_data(clean_df)

        # fit PCA
        pca = PCA(n_components=self.n_components, svd_solver=self.svd_solver, **self.pca_kwargs)
        pca.fit(norm_df)

        # raw eigenvectors from fit
        ev_raw = pca.components_.T  # (N_assets x K)
        # correct pc1 sign
        ev_corrected = self.correct_pc1_sign(ev_raw, norm_df)

        # calculate scores (PCs) using corrected loadings
        pcs_raw = np.dot(norm_df.values, ev_corrected)  # (N_obs x K)
        self.pcs = pd.DataFrame(pcs_raw, index=norm_df.index, columns=[f"PC{i + 1}" for i in range(pcs_raw.shape[1])])
        self.eigenvecs = pd.DataFrame(ev_corrected,
                                      index=norm_df.columns,
                                      columns=[f"EV{i + 1}" for i in range(ev_corrected.shape[1])])
        self.expl_var_ratio = pca.explained_variance_ratio_

    def get_pcs(self) -> pd.DataFrame:
        """
        Get principal components.

        Returns
        -------
        pcs: pd.DataFrame
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
        self.get_fixed_results()

        return self.expl_var_ratio

    def align_and_correct(self,
                          current_ev: np.ndarray,
                          clean_columns: pd.Index,
                          ref_ev: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Maps current eigenvectors to the full universe and corrects for ordering and
        sign flips using fast NumPy-based mapping.

        Parameters
        ----------
        current_ev: np.ndarray
            Current eigenvectors (N_clean x K)
        clean_columns: pd.Index
            Columns corresponding to current eigenvectors.
        ref_ev: Optional[np.ndarray]
            Reference eigenvectors for alignment (N_total x K)

        Returns
        -------
        aligned_ev: np.ndarray
            Aligned and sign-corrected eigenvectors (N_total x K)
        """
        # create mapping indices
        clean_indices = [self.ticker_to_idx[col] for col in clean_columns]

        # map to full universe: (N_total x K) matrix using NumPy indexing
        n_total = self.df.shape[1]
        n_comps = current_ev.shape[1]
        full_ev_np = np.zeros((n_total, n_comps))
        full_ev_np[clean_indices, :] = current_ev

        if ref_ev is None:
            # If no reference, similarity is 1.0 (perfect match with self)
            return full_ev_np, np.arange(n_comps), np.ones(n_comps)

        # project current vectors onto reference vectors to find the best match
        corr_matrix = np.dot(full_ev_np.T, ref_ev)
        cost_matrix = -np.abs(corr_matrix)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Extract the absolute correlation for the best matches
        similarities = np.abs(corr_matrix[row_ind, col_ind])

        # reorder current eigenvectors to match reference
        reordered_ev = full_ev_np[:, col_ind]

        # sign consistency
        # vectorized dot product for each column pair to check for sign flips
        for j in range(reordered_ev.shape[1]):
            # use dot product of the column against the corresponding reference column
            if np.dot(reordered_ev[:, j], ref_ev[:, j]) < 0:
                reordered_ev[:, j] *= -1

        return reordered_ev, col_ind, similarities

    def get_rolling_results(self, window_size: int, step: int = 1, show_progress: bool = True) -> None:
        """
        Computes rolling PCA results.

        Parameters
        ----------
        window_size: int
            Size of the rolling window (number of observations).
        step: int, default 1
            Step size for filling results. For example, step=5 will compute PCA every 5 days and
            forward-fill results for the intermediate days.
        """
        # check window size
        if window_size > len(self.df):
            raise ValueError("Window size exceeds data length.")

        # pre-compute window and asset counts, target components, and dates for indexing
        n_windows = len(self.df) - window_size + 1
        n_assets = self.df.shape[1]
        target_k = self.n_components if self.n_components is not None else n_assets
        dates = self.df.index[window_size - 1:]

        # Pre-allocate arrays for results to ensure consistent shapes and efficient memory usage
        all_pcs = np.full((n_windows, target_k), np.nan)
        all_ev_ratios = np.full((n_windows, target_k), np.nan)
        all_similarities = np.full((n_windows, target_k), np.nan)
        all_eigenvecs = np.full((n_windows * n_assets, target_k), np.nan)

        # reference eigenvectors
        ref_ev = None
        universe_cols = self.df.columns

        # Progress bar with step
        loop_iterator = tqdm(range(0, n_windows, step),
                             desc=f"Rolling PCA (Step={step})",
                             disable=not show_progress)

        # Main loop through windows with step
        for i in loop_iterator:
            # window data and normalization
            window_raw = self.df.iloc[i: i + window_size]
            clean_df = self._get_clean_window_data(window_raw)
            norm_df = self._normalize_data(clean_df)

            # pre-compute stats for intermediate day standardization
            window_mean = clean_df.mean()
            window_std = clean_df.std().replace(0, 1)

            # Fit PCA
            k_eff = min(min(norm_df.shape), target_k)
            pca = PCA(n_components=k_eff, svd_solver=self.svd_solver, **self.pca_kwargs)
            pca.fit(norm_df)

            # Sign correction and alignment
            current_raw_ev = pca.components_.T
            current_raw_ev = self.correct_pc1_sign(current_raw_ev, norm_df)
            corrected_ev, col_map, sims = self.align_and_correct(current_raw_ev, norm_df.columns, ref_ev)
            ref_ev = corrected_ev.copy()

            # Indexing setup
            clean_indices = universe_cols.get_indexer(norm_df.columns)
            n_comp_found = corrected_ev.shape[1]
            aligned_ev_clean = corrected_ev[clean_indices, :n_comp_found]

            # sub-loop for step filling
            fill_until = min(i + step, n_windows)
            for j in range(i, fill_until):
                if j == i:
                    norm_row_values = norm_df.iloc[-1].values
                else:
                    day_idx = window_size - 1 + j
                    raw_row = self.df.iloc[day_idx][norm_df.columns]
                    norm_row_values = ((raw_row - window_mean) / window_std).values

                # project to get PCs
                proj_pc = np.dot(norm_row_values, aligned_ev_clean)

                # storage with dynamic broadcasting
                all_pcs[j, :proj_pc.shape[0]] = proj_pc
                mapped_ratios = pca.explained_variance_ratio_[col_map]
                all_ev_ratios[j, :mapped_ratios.shape[0]] = mapped_ratios
                all_similarities[j, :sims.shape[0]] = sims
                # Store eigenvectors for the current window (same for all days in the step)
                start_idx = j * n_assets
                all_eigenvecs[start_idx: start_idx + n_assets, :n_comp_found] = corrected_ev

        # format outputs
        pc_cols = [f"PC{k + 1}" for k in range(target_k)]
        self.pcs = pd.DataFrame(all_pcs, index=dates, columns=pc_cols)
        self.expl_var_ratio = pd.DataFrame(all_ev_ratios, index=dates, columns=pc_cols)
        self.alignment_scores = pd.DataFrame(all_similarities, index=dates, columns=pc_cols)
        self.eigenvecs = pd.DataFrame(
            all_eigenvecs,
            index=pd.MultiIndex.from_product([dates, universe_cols], names=['date', 'ticker']),
            columns=[f"EV{k + 1}" for k in range(target_k)]
        ).sort_index()

    def get_expanding_results(self, min_obs: int, step: int = 1, show_progress: bool = True) -> None:
        """
        Computes expanding PCA results.

        Parameters
        ----------
        min_obs: int
            Minimum number of observations to start the expanding window.
        step: int, default 1
            Step size for filling results. For example, step=5 will compute PCA every 5 days and
            forward-fill results for the intermediate days.
        """
        # check min_periods
        if min_obs > len(self.df):
            raise ValueError("min_periods exceeds data length.")

        # pre-compute window and asset counts, target components, and dates for indexing
        n_windows = len(self.df) - min_obs + 1
        n_assets = self.df.shape[1]
        target_k = self.n_components if self.n_components is not None else n_assets
        dates = self.df.index[min_obs - 1:]

        # Pre-allocate arrays
        all_pcs = np.full((n_windows, target_k), np.nan)
        all_ev_ratios = np.full((n_windows, target_k), np.nan)
        all_similarities = np.full((n_windows, target_k), np.nan)
        all_eigenvecs = np.full((n_windows * n_assets, target_k), np.nan)

        # reference eigenvectors
        ref_ev = None
        universe_cols = self.df.columns

        # Progress bar with step
        loop_iterator = tqdm(range(0, n_windows, step),
                             desc=f"Expanding PCA (Step={step})",
                             disable=not show_progress)

        # Main loop through windows with step
        for i in loop_iterator:
            # --- EXPANDING WINDOW CHANGE ---
            # Fixed start at 0, expanding end at i + min_periods
            window_raw = self.df.iloc[: i + min_obs]
            clean_df = self._get_clean_window_data(window_raw)
            norm_df = self._normalize_data(clean_df)

            # pre-compute stats for intermediate day standardization
            window_mean = clean_df.mean()
            window_std = clean_df.std().replace(0, 1)

            # Fit PCA
            k_eff = min(min(norm_df.shape), target_k)
            pca = PCA(n_components=k_eff, svd_solver=self.svd_solver, **self.pca_kwargs)
            pca.fit(norm_df)

            # Sign correction and alignment
            current_raw_ev = pca.components_.T
            current_raw_ev = self.correct_pc1_sign(current_raw_ev, norm_df)
            corrected_ev, col_map, sims = self.align_and_correct(current_raw_ev, norm_df.columns, ref_ev)
            ref_ev = corrected_ev.copy()

            # Indexing setup
            clean_indices = universe_cols.get_indexer(norm_df.columns)
            n_comp_found = corrected_ev.shape[1]
            aligned_ev_clean = corrected_ev[clean_indices, :n_comp_found]

            # sub-loop for step filling
            fill_until = min(i + step, n_windows)
            for j in range(i, fill_until):
                if j == i:
                    norm_row_values = norm_df.iloc[-1].values
                else:
                    # --- EXPANDING WINDOW CHANGE ---
                    # The current day is always at index min_periods - 1 + j
                    day_idx = min_obs - 1 + j
                    raw_row = self.df.iloc[day_idx][norm_df.columns]
                    norm_row_values = ((raw_row - window_mean) / window_std).values

                # project to get PCs
                proj_pc = np.dot(norm_row_values, aligned_ev_clean)

                # storage with dynamic broadcasting
                all_pcs[j, :proj_pc.shape[0]] = proj_pc
                mapped_ratios = pca.explained_variance_ratio_[col_map]
                all_ev_ratios[j, :mapped_ratios.shape[0]] = mapped_ratios
                all_similarities[j, :sims.shape[0]] = sims

                # Store eigenvectors
                start_idx = j * n_assets
                all_eigenvecs[start_idx: start_idx + n_assets, :n_comp_found] = corrected_ev

        # format outputs
        pc_cols = [f"PC{k + 1}" for k in range(target_k)]
        self.pcs = pd.DataFrame(all_pcs, index=dates, columns=pc_cols)
        self.expl_var_ratio = pd.DataFrame(all_ev_ratios, index=dates, columns=pc_cols)
        self.alignment_scores = pd.DataFrame(all_similarities, index=dates, columns=pc_cols)
        self.eigenvecs = pd.DataFrame(
            all_eigenvecs,
            index=pd.MultiIndex.from_product([dates, universe_cols], names=['date', 'ticker']),
            columns=[f"EV{k + 1}" for k in range(target_k)]
        ).sort_index()

    def get_rolling_pcs(self, window_size: int) -> pd.DataFrame:
        """
        Returns the rolling principal component time series.

        Parameters
        ----------
        window_size: int
            Size of the rolling window (number of observations).

        Returns
        -------
        pd.DataFrame
            Rolling principal component time series.

        """
        self.get_rolling_results(window_size)
        return self.pcs

    def get_rolling_eigenvectors(self, window_size: int) -> pd.DataFrame:
        """
        Returns the rolling eigenvectors mapped to the full universe.

        Parameters
        ----------
        window_size: int
            Size of the rolling window (number of observations).

        Returns
        -------
        pd.DataFrame
            Rolling eigenvectors mapped to the full universe.
        """
        self.get_rolling_results(window_size)
        return self.eigenvecs

    def get_rolling_expl_var_ratio(self, window_size: int) -> pd.DataFrame:
        """
        Returns the rolling explained variance ratios.

        Parameters
        ----------
        window_size: int
            Size of the rolling window (number of observations).

        Returns
        -------
        pd.DataFrame
            Rolling explained variance ratios.

        """
        self.get_rolling_results(window_size)
        return self.expl_var_ratio

    def get_expanding_pcs(self, min_obs: int) -> pd.DataFrame:
        """
        Returns the expanding principal component time series.

        Parameters
        ----------
        min_obs: int
            Minimum number of observations to start expanding window.

        Returns
        -------
        pd.DataFrame
            Expanding principal component time series.

        """
        self.get_expanding_results(min_obs)
        return self.pcs

    def get_expanding_eigenvectors(self, min_obs: int) -> pd.DataFrame:
        """
        Returns the expanding eigenvectors mapped to the full universe.

        Parameters
        ----------
        min_obs: int
            Minimum number of observations to start expanding window.

        Returns
        -------
        pd.DataFrame
            Expanding eigenvectors mapped to the full universe.

        """
        self.get_expanding_results(min_obs)
        return self.eigenvecs

    def get_expanding_expl_var_ratio(self, min_obs: int) -> pd.DataFrame:
        """
        Returns the expanding explained variance ratios.

        Parameters
        ----------
        min_obs: int
            Minimum number of observations to start expanding window.

        Returns
        -------
        pd.DataFrame
            Expanding explained variance ratios.

        """
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
            Data matrix for principal component analytics
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
