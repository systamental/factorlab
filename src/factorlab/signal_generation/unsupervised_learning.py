import pandas as pd
import numpy as np
import scipy

from typing import Optional, Union, Any
from sklearn.decomposition import PCA


class PCAWrapper:
    """
    Principal component analysis wrapper class.
    """
    def __init__(self,
                 data: Union[np.array, pd.DataFrame],
                 n_components: Optional[int] = None,
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
        **kwargs: Optional keyword arguments, for PCA object. See sklearn.decomposition.PCA for details:
            https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
        """
        self.raw_data = data
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
            self.data = self.raw_data.dropna().to_numpy(dtype=np.float64)
        elif isinstance(self.raw_data, np.ndarray):
            self.data = self.raw_data[~np.isnan(self.raw_data).any(axis=1)]
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
    """

    def __init__(self,
                 data: Union[np.array, pd.DataFrame],
                 n_components: Optional[int] = None,
                 svd_solver: str = 'auto',
                 **kwargs: Any
                 ):
        """
        Initialize R2-PCA object.

        See details for R2-PCA:
        https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4400158

        Parameters
        ----------
        data: np.array or pd.DataFrame
            Data matrix for principal component analysis
        n_components: int or float, default None
            Number of principal components, or percentage of variance explained.
        svd_solver: str, {'auto', 'full', 'arpack', 'randomized'}, default='auto'
            SVD solver to use.
            See sklearn.decomposition.PCA for details:
            https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
        **kwargs: Optional keyword arguments, for PCA object. See sklearn.decomposition.PCA for details.
        """
        self.raw_data = data
        self.data = self.remove_missing()
        self.n_components = min(self.data.shape) if n_components is None else n_components
        self.svd_solver = svd_solver
        self.index = data.dropna().index if isinstance(data, pd.DataFrame) else None
        self.data_window = self.data.copy()
        self.pca = self.create_pca_instance(**kwargs)
        self.eigenvecs = None
        self.expl_var_ratio = None
        self.pcs = None

    def remove_missing(self) -> np.array:
        """
        Remove missing values from data.

        Returns
        -------
        data: np.array
            Data matrix with missing values removed.
        """
        if isinstance(self.raw_data, pd.DataFrame):
            self.data = self.raw_data.dropna().to_numpy(dtype=np.float64)
        elif isinstance(self.raw_data, np.ndarray):
            self.data = self.raw_data[~np.isnan(self.raw_data).any(axis=1)]
        else:
            raise ValueError(f"Data must be pd.DataFrame or np.array.")

        return self.data

    def create_pca_instance(self, **kwargs: Any) -> PCA:
        """
        Perform PCA.

        Parameters
        ----------
        **kwargs: Optional keyword arguments, for PCA object. See sklearn.decomposition.PCA for details.

        Returns
        -------
        pca: PCA
            PCA object.
        """
        self.pca = PCA(n_components=self.n_components, **kwargs)

        return self.pca

    def get_eigenvectors(self) -> np.array:
        """
        Get eigenvectors from SVD decomposition.

        Returns
        -------
        eigenvecs: np.array
            Eigenvectors from SVD decomposition.
        """
        # eigenvecs
        self.eigenvecs = self.pca.fit(self.data_window).components_.T

        return self.eigenvecs

    def correct_sign_pc1(self, pcs: np.array) -> np.array:
        """
        Constrain the sign of the first principal component to be consistent with the mean of the cross-section.

        Parameters
        ----------
        pcs: np.array
            Principal components.

        Returns
        -------
        pcs: np.array
            Principal components with first PC sign corrected.
        """
        # correct sign of first pc
        if np.dot(pcs[:, 0], np.mean(self.data_window, axis=1)) < 0:
            pcs *= -1

        return pcs

    def get_pcs(self) -> Union[np.array, pd.DataFrame]:
        """
        Get principal components.

        Returns
        -------
        pcs: np.array or pd.DataFrame
            Principal components.
        """
        # pca
        self.pcs = self.pca.fit_transform(self.data_window)

        # correct pc1 sign
        self.pcs = self.correct_sign_pc1(self.pcs)

        # add index and cols if available
        if self.index is not None:
            self.pcs = pd.DataFrame(self.pcs, index=self.index[-self.pcs.shape[0]:], columns=range(self.pcs.shape[1]))

        return self.pcs

    def get_expl_var_ratio(self) -> np.array:
        """
        Get explained variance ratio from sklearn PCA implementation.

        Returns
        -------
        expl_var_ratio: np.array
            Explained variance ratio.
        """
        # explained variance ratio
        self.expl_var_ratio = self.pca.fit(self.data_window).explained_variance_ratio_

        return self.expl_var_ratio

    @staticmethod
    def correct_eigenvectors(eigenvectors: np.array, ref_eigenvectors: np.array):
        """
        Correct eigenvectors for consistent ordering and signs.

        Parameters
        ----------
        eigenvectors: np.array
            Eigenvectors from SVD decomposition.
        ref_eigenvectors: np.array
            Reference eigenvectors for reordering and sign correction.

        Returns
        -------
        eigenvectors: np.array
            Corrected eigenvectors.
        """
        # eigenvector idxs
        eig_idxs = []
        # loop through eigenvectors
        for eig in range(eigenvectors.shape[1]):

            # similarity scores
            sim_scores = np.dot(eigenvectors[:, eig], ref_eigenvectors)
            # find eigenvector with the highest absolute similarity score for ordering
            max_abs_eig_idx = np.argmax(np.abs(sim_scores))
            # store eig idx
            eig_idxs.append(max_abs_eig_idx)
            # correct sign flip
            if sim_scores[max_abs_eig_idx] < 0:
                eigenvectors[:, eig] *= -1

        # reorder eigenvectors
        eigenvectors = eigenvectors[:, eig_idxs]

        return eigenvectors

    def get_rolling_eigenvectors(self, window_size: int) -> pd.DataFrame:
        """
        Compute rolling eigenvectors with consistent ordering and sign correction.

        Parameters
        ----------
        window_size : int
            Size of the rolling window.

        Returns
        -------
        pd.DataFrame
            Stacked DataFrame of rolling eigenvectors.
            Index = [date, tickers], Columns = ['EV1', 'EV2', ...]
        """
        if window_size is None:
            raise ValueError("Window size must be specified.")
        if window_size >= self.data.shape[0]:
            raise ValueError(f"Window size {window_size} is too large for data length.")
        if window_size < self.n_components:
            self.n_components = window_size
            self.create_pca_instance()

        rolling_eigenvecs = []

        # Initial reference eigenvectors
        self.data_window = self.data[:window_size, :]
        ref_eigenvectors = self.get_eigenvectors()
        rolling_eigenvecs.append(ref_eigenvectors)

        # Rolling loop
        for row in range(1, self.data.shape[0] - window_size + 1):
            self.data_window = self.data[row: row + window_size, :]
            eigvec = self.get_eigenvectors()
            eigvec = self.correct_eigenvectors(eigvec, ref_eigenvectors)
            ref_eigenvectors = eigvec
            rolling_eigenvecs.append(eigvec)

        # Build stacked tidy DataFrame
        dates = self.index[window_size - 1:]
        assets = self.raw_data.columns
        component_names = [f"EV{i + 1}" for i in range(self.n_components)]
        records = []

        for eigvec, date in zip(rolling_eigenvecs, dates):
            df = pd.DataFrame(eigvec, index=assets, columns=component_names)
            df['date'] = date
            df['ticker'] = df.index
            records.append(df.reset_index(drop=True))

        df_long = pd.concat(records)
        df_long = df_long.set_index(['date', 'ticker']).sort_index()

        return df_long

    def get_rolling_pcs(self, window_size: int) -> Union[np.array, pd.DataFrame]:
        """
        Get rolling principal components using R2-PCA.

        See details for R2-PCA: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4400158

        Parameters
        ----------
        window_size: int
            Size of rolling window (number of observations).

        Returns
        -------
        rolling_pcs: np.array or pd.DataFrame
            Rolling principal components.
        """
        # window size
        if window_size is None:
            raise ValueError(f"Window size must be specified.")
        if window_size >= self.data.shape[0]:
            raise ValueError(f"Window size {window_size} is greater than the number of observations.")
        if window_size < self.n_components:
            self.n_components = window_size
            self.create_pca_instance()

        # set rolling window for first window
        self.data_window = self.data[:window_size, :].copy()
        # eigenvectors
        eigenvectors = np.copy(self.get_eigenvectors())
        ref_eigenvectors = np.copy(eigenvectors)
        # pcs
        self.get_pcs()
        if isinstance(self.pcs, pd.DataFrame):
            self.pcs = self.pcs.values[-1]
        else:
            self.pcs = self.pcs[-1]

        # apply rolling pca with consistent ordering and signs
        for row in range(1, self.data.shape[0] - window_size + 1):
            # set rolling window
            self.data_window = self.data[row: row + window_size, :]
            # get eigenvectors
            self.get_eigenvectors()

            # correct eigenvectors
            self.eigenvecs = self.correct_eigenvectors(self.eigenvecs, ref_eigenvectors)

            # update ref eigenvectors
            ref_eigenvectors = self.eigenvecs
            # add rolling eigenvectors, expl var to array
            eigenvectors = np.vstack([eigenvectors, self.eigenvecs[-1]])

            # project data
            rolling_pcs = np.dot(self.data_window, self.eigenvecs)
            # correct pc1
            rolling_pcs = self.correct_sign_pc1(rolling_pcs)
            self.pcs = np.vstack([self.pcs, rolling_pcs[-1]])

        # add index and cols if available
        if self.index is not None:
            self.pcs = pd.DataFrame(self.pcs, index=self.index[-self.pcs.shape[0]:])

        return self.pcs

    def get_rolling_expl_var_ratio(self, window_size: int) -> Union[np.array, pd.DataFrame]:
        """
        Get rolling explained variance ratio.

        Parameters
        ----------
        window_size: int
            Size of rolling window (number of observations).

        Returns
        -------
        rolling_expl_var_ratio: np.array or pd.DataFrame
            Rolling principal components.
        """
        # window size
        if window_size is None:
            raise ValueError(f"Window size must be specified.")
        if window_size >= self.data.shape[0]:
            raise ValueError(f"Window size {window_size} is greater than the number of observations.")
        if window_size < self.n_components:
            self.n_components = window_size
            self.create_pca_instance()

        # output
        out = None

        # get rolling window expl var
        for row in range(self.data.shape[0] - window_size + 1):

            # set rolling window
            self.data_window = self.data[row: row + window_size, :]

            # get expl var ratio
            self.get_expl_var_ratio()

            if row == 0:
                if len(self.expl_var_ratio.shape) == 1:
                    out = self.expl_var_ratio.reshape(1, -1)
                else:
                    out = self.expl_var_ratio[-1]
            else:
                # add output to array
                if len(self.expl_var_ratio.shape) == 1:
                    self.expl_var_ratio = self.expl_var_ratio.reshape(1, -1)
                out = np.vstack([out, self.expl_var_ratio[-1]])

        self.expl_var_ratio = out

        # add index and cols if available
        if self.expl_var_ratio is not None:
            self.expl_var_ratio = pd.DataFrame(self.expl_var_ratio, index=self.index[-self.expl_var_ratio.shape[0]:])

        return self.expl_var_ratio

    def get_expanding_eigenvectors(self, min_obs: int) -> pd.DataFrame:
        """
        Compute expanding eigenvectors with consistent ordering and sign correction.

        Parameters
        ----------
        min_obs : int
            Minimum number of observations to start computing eigenvectors.

        Returns
        -------
        pd.DataFrame
            Stacked DataFrame of expanding eigenvectors.
            Index = [date, asset], Columns = ['EV1', 'EV2', ...]
        """
        if min_obs is None:
            raise ValueError("Minimum observations must be specified.")
        if min_obs >= self.data.shape[0]:
            raise ValueError(f"Minimum observations {min_obs} is too large for data length.")
        if min_obs < self.n_components:
            self.n_components = min_obs
            self.create_pca_instance()

        expanding_eigenvecs = []

        # Initial expanding window
        self.data_window = self.data[:min_obs, :]
        ref_eigenvectors = self.get_eigenvectors()
        expanding_eigenvecs.append(ref_eigenvectors)

        for row in range(min_obs + 1, self.data.shape[0] + 1):
            self.data_window = self.data[:row, :]
            eigvec = self.get_eigenvectors()
            eigvec = self.correct_eigenvectors(eigvec, ref_eigenvectors)
            ref_eigenvectors = eigvec
            expanding_eigenvecs.append(eigvec)

        # Build stacked tidy DataFrame
        dates = self.index[min_obs - 1:]  # align to end of each window
        assets = self.raw_data.columns
        component_names = [f"EV{i + 1}" for i in range(self.n_components)]
        records = []

        for eigvec, date in zip(expanding_eigenvecs, dates):
            df = pd.DataFrame(eigvec, index=assets, columns=component_names)
            df['date'] = date
            df['asset'] = df.index
            records.append(df.reset_index(drop=True))

        df_long = pd.concat(records)
        df_long = df_long.set_index(['date', 'asset']).sort_index()

        return df_long

    def get_expanding_pcs(self, min_obs: int) -> Union[np.array, pd.DataFrame]:
        """
        Get rolling principal components.

        Parameters
        ----------
        min_obs: int
            Minimum number of observations for expanding window computation.

        Returns
        -------
        exp_pcs: np.array or pd.DataFrame
            Rolling principal components.
        """
        # min obs
        if min_obs is None:
            raise ValueError(f"Minimum observations must be specified.")
        if min_obs >= self.data.shape[0]:
            raise ValueError(f"Minimum observations {min_obs} is greater than the total number of observations.")
        if min_obs < self.n_components:
            self.n_components = min_obs
            self.create_pca_instance()

        # set rolling window for first window
        self.data_window = self.data[:min_obs, :].copy()
        # eigenvectors
        eigenvectors = np.copy(self.get_eigenvectors())
        ref_eigenvectors = np.copy(eigenvectors)
        # pcs
        self.get_pcs()
        if isinstance(self.pcs, pd.DataFrame):
            self.pcs = self.pcs.values[-1]
        else:
            self.pcs = self.pcs[-1]

        # apply expanding pca with consistent ordering and signs
        for row in range(min_obs + 1, self.data.shape[0] + 1):
            # set expanding window
            self.data_window = self.data[: row]
            # get eigenvectors
            self.get_eigenvectors()
            # correct eigenvectors
            self.eigenvecs = self.correct_eigenvectors(self.eigenvecs, ref_eigenvectors)
            # update ref eigenvectors
            ref_eigenvectors = self.eigenvecs
            # add expanding eigenvectors to array
            eigenvectors = np.vstack([eigenvectors, self.eigenvecs[-1]])

            # project data
            exp_pcs = np.dot(self.data_window, self.eigenvecs)
            # correct pc1
            exp_pcs = self.correct_sign_pc1(exp_pcs)
            self.pcs = np.vstack([self.pcs, exp_pcs[-1]])

        # add index and cols if available
        if self.index is not None:
            self.pcs = pd.DataFrame(self.pcs, index=self.index[-self.pcs.shape[0]:])

        return self.pcs

    def get_expanding_expl_var_ratio(self, min_obs: Optional[int] = None) -> Union[np.array, pd.DataFrame]:
        """
        Get expanding explained variance ratio.

        Parameters
        ----------
        min_obs: int, default 1
            Minimum number of observations for expanding window computation.

        Returns
        -------
        exp_expl_var_ratio: np.array or pd.DataFrame
            Expanding explained variance ratio.
        """
        # min obs
        if min_obs is None:
            raise ValueError(f"Minimum observations must be specified.")
        if min_obs >= self.data.shape[0]:
            raise ValueError(f"Minimum observations {min_obs} is greater than the total number of observations.")
        if min_obs < self.n_components:
            self.n_components = min_obs
            self.create_pca_instance()

        # min obs
        if min_obs is None:
            min_obs = min(self.data.shape)
        if min_obs < self.n_components:
            self.n_components = min_obs
            self.create_pca_instance(n_components=self.n_components)

        # output
        out = None

        # get expanding window expl var
        for row in range(min_obs, self.data.shape[0] + 1):

            # set expanding window
            self.data_window = self.data[: row]

            # get expl var ratio
            self.get_expl_var_ratio()

            if row == min_obs:
                if len(self.expl_var_ratio.shape) == 1:
                    out = self.expl_var_ratio.reshape(1, -1)
                else:
                    out = self.expl_var_ratio[-1]
            else:
                # add output to array
                if len(self.expl_var_ratio.shape) == 1:  # reshape to 2d arr
                    self.expl_var_ratio = self.expl_var_ratio.reshape(1, -1)
                out = np.vstack([out, self.expl_var_ratio[-1]])

        self.expl_var_ratio = out

        # add index and cols if available
        if self.index is not None:
            self.expl_var_ratio = pd.DataFrame(self.expl_var_ratio, index=self.index[-self.expl_var_ratio.shape[0]:])

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
        C = scipy.linalg.orth(C)
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
