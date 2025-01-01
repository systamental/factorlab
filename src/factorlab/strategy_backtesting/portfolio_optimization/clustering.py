import pandas as pd
import numpy as np
from typing import Optional, Union, List
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import linkage, fcluster, cut_tree, dendrogram
from scipy.spatial.distance import squareform

from factorlab.strategy_backtesting.portfolio_optimization.risk_estimators import RiskEstimators
from factorlab.strategy_backtesting.metrics import Metrics
from factorlab.data_viz.plot import plot_bar


class HRP:
    """
    Hierarchical risk parity optimization class.

    This class computes the optimized portfolio weights based on the returns of the assets or strategies
    using hierarchical risk parity optimization techniques.

    Hierarchical Risk Parity uses a hierarchical clustering algorithm to create a tree of assets
    (i.e. hierarchical tree-clustering) and then allocates risk based on the inverse of the distance between assets in
     the tree.

     The algorithm is based on the following paper:
    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2708678

    """
    def __init__(self,
                 returns: Union[pd.DataFrame, pd.Series],
                 cov_matrix_method: str = 'covariance',
                 linkage_method: str = 'single',
                 distance_metric: str = 'euclidean',
                 side_weights: Optional[pd.Series] = None,
                 leverage: float = 1.0,
                 asset_names: Optional[List[str]] = None
                 ):
        """
        Initialize the HRP portfolio optimization class.

        Parameters
        ----------
        returns : pd.DataFrame or pd.Series
            Returns of the assets or strategies.
        cov_matrix_method : str, {'covariance', 'empirical_covariance', 'shrunk_covariance', 'ledoit_wolf', 'oas',
                       'graphical_lasso', 'graphical_lasso_cv', 'minimum_covariance_determinant', 'semi_covariance',
                          'exponential_covariance', 'denoised_covariance'}, default 'covariance'
            Method to compute the covariance matrix.
        linkage_method : str, {'single', 'complete', 'average', 'weighted', 'centroid', 'median', 'ward'},
        default 'single'
            Method to compute the distance matrix.
        distance_metric: str, {‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘cityblock’, ‘correlation’, ‘cosine’, ‘dice’,
        ‘euclidean’, ‘hamming’, ‘jaccard’, ‘jensenshannon’, ‘kulczynski1’, ‘mahalanobis’, ‘matching’, ‘minkowski’,
        ‘rogerstanimoto’, ‘russellrao’, ‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’,
        ‘sqeuclidean’, ‘yule'}, default 'euclidean'
            Metric to compute the distance matrix.
        side_weights : pd.Series, default None
            Side/direction weights for the assets or strategies (e.g. 1 for long, -1 for short).
        leverage : float, default 1.0
            Leverage factor.
        asset_names : list, default None
            Names of the assets or strategies.
        """
        self.returns = returns
        self.cov_matrix_method = cov_matrix_method
        self.linkage_method = linkage_method
        self.distance_metric = distance_metric
        self.side_weights = side_weights
        self.leverage = leverage
        self.asset_names = asset_names
        self.n_assets = None
        self.cov_matrix = None
        self.corr_matrix = None
        self.distance_matrix = None
        self.clusters = None
        self.idxs = None
        self.weights = None
        self.preprocess_data()

    # TODO: check for symmetry in the distance matrix
    def preprocess_data(self):
        """
        Preprocess the data for the portfolio optimization.
        """
        # returns
        if not isinstance(self.returns, pd.DataFrame) and not isinstance(self.returns, pd.Series):  # check data type
            raise ValueError('returns must be a pd.DataFrame or pd.Series')
        # convert data type to float64
        if isinstance(self.returns, pd.Series):  # convert to df
            self.returns = self.returns.to_frame().astype('float64')
        elif isinstance(self.returns, pd.DataFrame):
            self.returns = self.returns.astype('float64')
        if isinstance(self.returns.index, pd.MultiIndex):  # convert to single index
            self.returns = self.returns.unstack()
        self.returns.index = pd.to_datetime(self.returns.index)  # convert to index to datetime

        # remove missing vals
        self.returns = self.returns.dropna(how='all').dropna(how='any', axis=1)

        # method
        if self.linkage_method not in ['single', 'complete', 'average', 'weighted', 'centroid', 'median', 'ward']:
            raise ValueError('method must be one of the following: single, complete, average, weighted, centroid, '
                             'median, ward')

        # side weights
        if self.side_weights is None:
            self.side_weights = pd.Series(1, index=self.returns.columns, name='weights')

        # asset names
        if self.asset_names is None:
            self.asset_names = self.returns.columns.tolist()

        # n_assets
        if self.n_assets is None:
            self.n_assets = self.returns.shape[1]

    def compute_estimators(self):
        """
        Compute the covariance matrix, correlation matrix, and distance matrix.
        """
        # covariance matrix
        if self.cov_matrix is None:
            self.cov_matrix = RiskEstimators(self.returns).compute_covariance_matrix(method=self.cov_matrix_method)

        # correlation matrix
        if self.corr_matrix is None:
            self.corr_matrix = self.returns.corr().to_numpy('float64')

        # distance matrix
        if self.distance_matrix is None:
            self.distance_matrix = np.sqrt((1 - self.corr_matrix) / 2)

    def tree_clustering(self):
        """
        Perform hierarchical tree clustering.
        """
        # clusters
        self.clusters = linkage(squareform(self.distance_matrix), metric=self.distance_metric,
                                method=self.linkage_method)

        return self.clusters

    def quasi_diagonalization(self):
        """
        Reorders the rows and columns of the covariance matrix so that the largest values lie along the diagonal.

        Returns
        -------
        list
            The list of indexes from the quasi-diagonalization.
        """
        # convert to int
        clusters = self.clusters.astype(int).copy()
        # Sort clustered items by distance
        sorted_idx = pd.Series([clusters[-1, 0], clusters[-1, 1]])
        n_items = clusters[-1, 3]  # number of original items
        # create index
        while sorted_idx.max() >= n_items:
            # make space
            sorted_idx.index = range(0, sorted_idx.shape[0] * 2, 2)
            # find clusters
            df0 = sorted_idx[sorted_idx >= n_items]
            i = df0.index
            j = df0.values - n_items
            # item 1
            sorted_idx[i] = clusters[j, 0]
            df0 = pd.Series(clusters[j, 1], index=i + 1)
            sorted_idx = pd.concat([sorted_idx, df0])
            # re-sort
            sorted_idx = sorted_idx.sort_index()
            # re-index
            sorted_idx.index = range(sorted_idx.shape[0])

        return sorted_idx.tolist()

    @staticmethod
    def compute_inverse_variance_weights(cov_matrix):
        """
        Compute the inverse variance weights.
        """
        iv = 1 / np.diag(cov_matrix)
        iv /= iv.sum()
        return iv

    def compute_cluster_variance(self, cluster_idxs):
        """
        Get the cluster variance.

        Parameters
        ----------
        cluster_idxs : list
            The list of indexes for the cluster.
        """
        cluster_cov = self.cov_matrix[np.ix_(cluster_idxs, cluster_idxs)]
        w = self.compute_inverse_variance_weights(cluster_cov)
        cluster_var = np.dot(w, np.dot(cluster_cov, w))

        return cluster_var

    def recursive_bisection(self):
        """
        Perform recursive bisection.
        """
        # initialize start weights
        self.weights = pd.Series(1.0, index=self.idxs, name='weights')
        cluster_items = [self.idxs]

        # recursively bisect the clusters
        while cluster_items:

            # bisect each cluster
            cluster_items = [cluster[start:end]
                             for cluster in cluster_items
                             for start, end in ((0, len(cluster) // 2), (len(cluster) // 2, len(cluster)))
                             if len(cluster) > 1]

            # parse in pairs
            for subcluster in range(0, len(cluster_items), 2):
                cluster_1 = cluster_items[subcluster]
                cluster_2 = cluster_items[subcluster + 1]

                # Get left and right cluster variances and calculate allocation factor
                cluster_1_variance = self.compute_cluster_variance(cluster_1)
                cluster_2_variance = self.compute_cluster_variance(cluster_2)
                alpha = 1 - cluster_1_variance / (cluster_1_variance + cluster_2_variance)

                # Assign weights to each sub-cluster
                self.weights[cluster_1] *= alpha
                self.weights[cluster_2] *= 1 - alpha

        # reorder asset names
        self.asset_names = [self.asset_names[i] for i in self.idxs]

        self.weights = pd.DataFrame(self.weights.values, index=self.asset_names, columns=[self.returns.index[-1]])

    def create_portfolio(self):
        """
        Create the portfolio.
        """
        # long short portfolios
        short_port = self.side_weights[self.side_weights == -1].index
        long_port = self.side_weights[self.side_weights == 1].index

        # create portfolio
        if len(short_port) > 0:
            # Short half size
            self.weights.loc[short_port] /= self.weights.loc[short_port].sum().values[0]
            self.weights.loc[short_port] *= self.leverage * -1

            # Buy other half
            self.weights.loc[long_port] /= self.weights.loc[long_port].sum().values[0]
            self.weights.loc[long_port] *= self.leverage

        else:
            self.weights *= self.leverage

        self.weights = self.weights.T

    def compute_weights(self):
        """
        Compute the optimized portfolio weights.
        """
        # compute estimators
        self.compute_estimators()

        # hierarchical tree clustering
        self.tree_clustering()

        # quasi diagonalization
        self.idxs = self.quasi_diagonalization()

        # recursive bisection
        self.recursive_bisection()

        # create portfolio
        self.create_portfolio()

        return self.weights

    def plot_clusters(self):
        """
        Plot the clusters.
        """
        plt.figure(figsize=(15, 7))
        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('Assets')
        plt.ylabel('Distance')
        dendrogram(
            self.clusters,
            labels=self.asset_names,
            orientation='left',
            leaf_rotation=0.,  # rotates the x axis labels
            leaf_font_size=8.,  # font size for the x axis labels
        )
        plt.tight_layout()
        plt.show()

    def plot_weights(self):
        """
        Plot the optimized portfolio weights.
        """
        plot_bar(self.weights.T.sort_values(by=[self.returns.index[-1]]), axis='horizontal', x_label='weights')


class HERC:
    """
    Hierarchical equal risk contribution optimization class.

    This class computes the optimized portfolio weights based on the returns of the assets or strategies
    using hierarchical equal risk contribution optimization techniques.

    See the following papers for more details:

    https://www.pm-research.com/content/iijpormgmt/44/2/89

    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3237540

    """
    def __init__(self,
                 returns: Union[pd.DataFrame, pd.Series],
                 risk_measure: str = 'equal_weight',
                 alpha: float = 0.05,
                 cov_matrix_method: str = 'covariance',
                 n_clusters: Optional[int] = None,
                 linkage_method: str = 'ward',
                 distance_metric: str = 'euclidean',
                 leverage: float = 1.0,
                 asset_names: Optional[List[str]] = None
                 ):
        """
        Initialize the HERC portfolio optimization class.

        Parameters
        ----------
        returns : pd.DataFrame or pd.Series
            Returns of the assets or strategies.
        risk_measure : str, {'equal_weight', 'variance', 'std', 'expected_shortfall', 'conditional_drawdown_risk'},
        default 'equal_weight'
            Risk measure to compute the risk contribution.
        alpha : float, default 0.05
            Confidence level/threshold for the tails of the distribution of returns for risk measure.
        cov_matrix_method : str, {'covariance', 'empirical_covariance', 'shrunk_covariance', 'ledoit_wolf', 'oas',
                        'graphical_lasso', 'graphical_lasso_cv', 'minimum_covariance_determinant', 'semi_covariance',
                        'exponential_covariance', 'denoised_covariance'}, default 'covariance'
            Method to compute the covariance matrix.
        n_clusters : int, default None
            Number of clusters.
        linkage_method : str, {'single', 'complete', 'average', 'weighted', 'centroid', 'median', 'ward'},
        default 'ward'
            Method to compute the linkage matrix.
        distance_metric: str, {‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘cityblock’, ‘correlation’, ‘cosine’, ‘dice’,
        ‘euclidean’, ‘hamming’, ‘jaccard’, ‘jensenshannon’, ‘kulczynski1’, ‘mahalanobis’, ‘matching’, ‘minkowski’,
        ‘rogerstanimoto’, ‘russellrao’, ‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’,
        ‘sqeuclidean’, ‘yule'}, default 'euclidean'
            Metric to compute the distance matrix.
        leverage : float, default 1.0
            Leverage factor.
        asset_names : list, default None
            Names of the assets or strategies.
        """
        self.returns = returns
        self.risk_measure = risk_measure
        self.alpha = alpha
        self.cov_matrix_method = cov_matrix_method
        self.linkage_method = linkage_method
        self.distance_metric = distance_metric
        self.leverage = leverage
        self.asset_names = asset_names

        self.n_assets = None
        self.cov_matrix = None
        self.corr_matrix = None
        self.distance_matrix = None
        self.clusters = None
        self.cluster_children = None
        self.n_clusters = n_clusters
        self.clusters_contribution = None
        self.idxs = None
        self.weights = None
        self.preprocess_data()

    def preprocess_data(self) -> None:
        """
        Preprocess the data for the portfolio optimization.
        """
        # returns
        if not isinstance(self.returns, pd.DataFrame) and not isinstance(self.returns, pd.Series):  # check data type
            raise ValueError('returns must be a pd.DataFrame or pd.Series')
        # convert data type to float64
        if isinstance(self.returns, pd.Series):  # convert to df
            self.returns = self.returns.to_frame().astype('float64')
        elif isinstance(self.returns, pd.DataFrame):
            self.returns = self.returns.astype('float64')
        if isinstance(self.returns.index, pd.MultiIndex):  # convert to single index
            self.returns = self.returns.unstack()
        self.returns.index = pd.to_datetime(self.returns.index)  # convert to index to datetime

        # remove missing vals
        self.returns = self.returns.dropna(how='all').dropna(how='any', axis=1)

        # risk measure
        if self.risk_measure not in ['equal_weight', 'variance', 'std', 'expected_shortfall',
                                     'conditional_drawdown_risk']:
            raise ValueError("Unknown allocation metric specified. Supported metrics are - equal_weight, "
                             "variance, std, expected_shortfall, conditional_drawdown_risk")

        # method
        if self.linkage_method not in ['single', 'complete', 'average', 'weighted', 'centroid', 'median', 'ward']:
            raise ValueError('method must be one of the following: single, complete, average, weighted, centroid, '
                             'median, ward')

        # distance metric
        if self.distance_metric not in ['braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine',
                                        'dice', 'euclidean', 'hamming', 'jaccard', 'jensenshannon', 'kulczynski1',
                                        'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao',
                                        'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule']:
            raise ValueError('distance_metric must be one of the following: braycurtis, canberra, chebyshev, cityblock, '
                             'correlation, cosine, dice, euclidean, hamming, jaccard, jensenshannon, kulczynski1, '
                             'mahalanobis, matching, minkowski, rogerstanimoto, russellrao, seuclidean, sokalmichener, '
                             'sokalsneath, sqeuclidean, yule')

        # asset names
        if self.asset_names is None:
            self.asset_names = self.returns.columns.tolist()

        # n_assets
        if self.n_assets is None:
            self.n_assets = self.returns.shape[1]

    def compute_estimators(self) -> None:
        """
        Compute the covariance matrix, correlation matrix, and distance matrix.
        """
        # covariance matrix
        if self.cov_matrix is None:
            self.cov_matrix = RiskEstimators(self.returns).compute_covariance_matrix(method=self.cov_matrix_method)

        # correlation matrix
        if self.corr_matrix is None:
            self.corr_matrix = self.returns.corr().to_numpy('float64')

        # distance matrix
        if self.distance_matrix is None:
            self.distance_matrix = np.sqrt(2 * (1 - self.corr_matrix))

    def get_clusters(self) -> np.ndarray:
        """
        Get the clusters based on the linkage method and distance metric.

        Returns
        -------
        clusters: np.ndarray
            The clusters.
        """
        self.clusters = linkage(squareform(self.distance_matrix), metric=self.distance_metric,
                                method=self.linkage_method)

        return self.clusters

    def compute_optimal_n_clusters(self) -> int:
        """
        Computes the optimal number of clusters based on Two-Order Difference to Gap Statistic

        See the following paper for more details:
        https://link.springer.com/article/10.1007/s12209-008-0039-1

        Returns
        -------
        n_clusers: int
            The optimal number of clusters.
        """
        if self.n_clusters is None:

            c_tree = cut_tree(self.clusters)
            n = c_tree.shape[1]
            max_clusters = min(n, max(8, round(np.sqrt(n))))
            dispersion = []

            for k in range(max_clusters):
                level = c_tree[:, n - k - 1]
                cluster_density = []
                for i in range(np.max(level) + 1):
                    cluster_idx = np.argwhere(level == i).flatten()
                    cluster_dists = squareform(
                        self.distance_matrix[cluster_idx, :][:, cluster_idx], checks=False
                    )
                    if cluster_dists.shape[0] != 0:
                        cluster_density.append(np.nan_to_num(cluster_dists.mean()))
                dispersion.append(np.sum(cluster_density))

            dispersion = np.array(dispersion)
            gaps = np.roll(dispersion, -2) + dispersion - 2 * np.roll(dispersion, -1)
            gaps = gaps[:-2]

            self.n_clusters = np.argmax(gaps) + 2

    def get_cluster_children(self) -> None:
        """
        Get the cluster children.
        """
        clustering_idxs = fcluster(self.clusters, self.n_clusters, criterion='maxclust')
        self.cluster_children = {index - 1: [] for index in range(min(clustering_idxs), max(clustering_idxs) + 1)}
        for index, cluster_index in enumerate(clustering_idxs):
            self.cluster_children[cluster_index - 1].append(index)

    def quasi_diagonalization(self, idx: int) -> list:
        """
        Reorders the rows and columns of the matrix so that the largest values lie along the diagonal.

        Returns
        -------
        list
            The list of indexes from the quasi-diagonalization.
        """
        if idx < self.n_assets:
            return [idx]

        left = int(self.clusters[idx - self.n_assets, 0])
        right = int(self.clusters[idx - self.n_assets, 1])

        return self.quasi_diagonalization(left) + self.quasi_diagonalization(right)

    @staticmethod
    def get_intersection(list1, list2) -> list:
        """
        Get the intersection of two lists.
        """
        return list(set(list1) & set(list2))

    def get_children_cluster_idxs(self, parent_cluster_idx) -> tuple:
        """
        Get the children cluster indexes.

        Parameters
        ----------
        parent_cluster_idx : int
            The parent cluster index.

        Returns
        -------
        left_cluster_idxs : list
            The left cluster indexes.
        right_cluster_idxs : list
            The right cluster indexes.
        """
        left = int(self.clusters[self.n_assets - 2 - parent_cluster_idx, 0])
        right = int(self.clusters[self.n_assets - 2 - parent_cluster_idx, 1])
        left_cluster = self.quasi_diagonalization(left)
        right_cluster = self.quasi_diagonalization(right)

        left_cluster_idxs = []
        right_cluster_idxs = []
        for id_cluster, cluster in self.cluster_children.items():
            if sorted(self.get_intersection(left_cluster, cluster)) == sorted(cluster):
                left_cluster_idxs.append(id_cluster)
            if sorted(self.get_intersection(right_cluster, cluster)) == sorted(cluster):
                right_cluster_idxs.append(id_cluster)

        return left_cluster_idxs, right_cluster_idxs

    @staticmethod
    def compute_inverse_variance_weights(cov_matrix) -> np.ndarray:
        """
        Compute the inverse variance weights.

        Parameters
        ----------
        cov_matrix : np.ndarray
            The covariance matrix.

        Returns
        -------
        w: np.ndarray
            The inverse variance weights.
        """
        w = 1 / np.diag(cov_matrix)
        w /= w.sum()
        return w

    @staticmethod
    def compute_inverse_volatility_weights(cov_matrix) -> np.ndarray:
        """
        Compute the inverse volatility weights.

        Parameters
        ----------
        cov_matrix : np.ndarray
            The covariance matrix.

        Returns
        -------
        w: np.ndarray
            The inverse volatility weights.
        """
        w = 1 / np.sqrt(np.diag(cov_matrix))
        w /= w.sum()
        return w

    def compute_inverse_cvar_weights(self, returns) -> np.ndarray:
        """
        Compute the inverse CVaR weights.

        Parameters
        ----------
        returns : pd.DataFrame
            The returns of the assets or strategies.

        Returns
        -------
        w: np.ndarray
            The inverse CVaR weights.
        """
        w = 1 / Metrics(returns).expected_shortfall(alpha=self.alpha)
        w /= w.sum()

        return w.values

    def compute_inverse_cdar_weights(self, returns) -> np.ndarray:
        """
        Compute the inverse CDaR weights.

        Parameters
        ----------
        returns : pd.DataFrame
            The returns of the assets or strategies.

        Returns
        -------
        w: np.ndarray
            The inverse CDaR weights.
        """
        w = 1 / Metrics(returns).conditional_drawdown_risk(alpha=self.alpha)
        w /= w.sum()

        return w.values

    def compute_cluster_variance(self, cluster_idxs) -> float:
        """
        Get the cluster risk.

        Parameters
        ----------
        cluster_idxs : list
            The list of indexes for the cluster.

        Returns
        -------
        cluster_var: float
            The cluster variance.
        """
        cluster_cov = self.cov_matrix[np.ix_(cluster_idxs, cluster_idxs)]
        w = self.compute_inverse_variance_weights(cluster_cov)
        cluster_var = np.dot(w, np.dot(cluster_cov, w))

        return cluster_var

    def compute_cluster_expected_shortfall(self, cluster_idxs) -> float:
        """
        Get the cluster expected shortfall.

        Parameters
        ----------
        cluster_idxs : list
            The list of indexes for the cluster.

        Returns
        -------
        cluster_cvar: float
            The cluster expected shortfall.
        """
        cluster_asset_returns = self.returns.iloc[:, cluster_idxs]
        w = self.compute_inverse_cvar_weights(cluster_asset_returns)
        portfolio_returns = cluster_asset_returns @ w
        cluster_cvar = Metrics(portfolio_returns).expected_shortfall(alpha=self.alpha)

        return cluster_cvar

    def compute_cluster_conditional_drawdown_risk(self, cluster_idxs) -> float:
        """
        Get the cluster conditional drawdown at risk.

        Parameters
        ----------
        cluster_idxs : list
            The list of indexes for the cluster.

        Returns
        -------
        cluster_cdar: float
            The cluster conditional drawdown at risk.
        """
        cluster_asset_returns = self.returns.iloc[:, cluster_idxs]
        w = self.compute_inverse_cdar_weights(cluster_asset_returns)
        portfolio_returns = cluster_asset_returns @ w
        cluster_cdar = Metrics(portfolio_returns).conditional_drawdown_risk(alpha=self.alpha)

        return cluster_cdar

    def compute_cluster_risk_contribution(self) -> None:
        """
        Compute the cluster risk contribution.
        """
        # initialize cluster contribution
        self.clusters_contribution = np.ones(shape=self.n_clusters)

        # compute cluster risk contribution
        for cluster_idx in range(self.n_clusters):
            cluster_asset_idxs = self.cluster_children[cluster_idx]

            if self.risk_measure == 'variance':
                self.clusters_contribution[cluster_idx] = self.compute_cluster_variance(cluster_asset_idxs)

            elif self.risk_measure == 'std':
                self.clusters_contribution[cluster_idx] = np.sqrt(
                    self.compute_cluster_variance(cluster_asset_idxs))

            elif self.risk_measure == 'expected_shortfall':
                self.clusters_contribution[cluster_idx] = self.compute_cluster_expected_shortfall(cluster_asset_idxs)[0]

            elif self.risk_measure == 'conditional_drawdown_risk':
                self.clusters_contribution[cluster_idx] = \
                    self.compute_cluster_conditional_drawdown_risk(cluster_asset_idxs)[0]

    def compute_naive_risk_parity_weights(self,
                                          returns: pd.DataFrame,
                                          cov_matrix: pd.DataFrame,
                                          cluster_idx: int
                                          ) -> np.ndarray:
        """
        Compute the naive risk parity weights.

        Parameters
        ----------
        returns : pd.DataFrame
            The returns of the assets or strategies.
        cov_matrix : pd.DataFrame
            The covariance matrix.
        cluster_idx : int
            The cluster index.

        Returns
        -------
        parity_weights: np.ndarray
            The naive risk parity weights.
        """
        if self.risk_measure == 'equal_weight':
            n_assets_in_cluster = len(self.cluster_children[cluster_idx])
            parity_weights = np.ones(n_assets_in_cluster) / n_assets_in_cluster
        elif self.risk_measure == 'variance':
            parity_weights = self.compute_inverse_variance_weights(cov_matrix)
        elif self.risk_measure == 'std':
            parity_weights = self.compute_inverse_volatility_weights(cov_matrix)
        elif self.risk_measure == 'expected_shortfall':
            parity_weights = self.compute_inverse_cvar_weights(returns)
        else:
            parity_weights = self.compute_inverse_cdar_weights(returns)

        return parity_weights

    def get_cluster_weights(self, clusters_weights) -> None:
        """
        Get the cluster weights.

        Parameters
        ----------
        clusters_weights : np.ndarray
            The weights of the clusters.
        """
        for cluster_index in range(self.n_clusters):
            asset_indices = self.cluster_children[cluster_index]

            # assets
            cluster_returns = self.returns.iloc[:, asset_indices]
            # cov
            cluster_cov = self.cov_matrix[np.ix_(asset_indices, asset_indices)]
            # risk parity weights
            parity_weights = self.compute_naive_risk_parity_weights(cluster_returns, cluster_cov, cluster_index)

            self.weights[asset_indices] = parity_weights * clusters_weights[cluster_index]

    def recursive_bisection(self):
        """
        Perform recursive bisection.
        """
        # initialize weights
        self.weights = np.ones(self.n_assets)
        clusters_contribution = np.ones(self.n_clusters)
        clusters_weights = np.ones(self.n_clusters)

        # compute cluster risk
        self.compute_cluster_risk_contribution()

        # recursive bisection
        for cluster_index in range(self.n_clusters - 1):

            # get children cluster idxs
            left_cluster_idxs, right_cluster_idxs = self.get_children_cluster_idxs(cluster_index)

            # compute alpha
            left_cluster_contribution = np.sum(clusters_contribution[left_cluster_idxs])
            right_cluster_contribution = np.sum(clusters_contribution[right_cluster_idxs])
            if self.risk_measure == 'equal_weight':
                allocation_factor = 0.5
            else:
                allocation_factor = 1 - left_cluster_contribution / (left_cluster_contribution +
                                                                     right_cluster_contribution)

            # assign weights to each sub-cluster
            clusters_weights[left_cluster_idxs] *= allocation_factor
            clusters_weights[right_cluster_idxs] *= 1 - allocation_factor

        # get weights
        self.get_cluster_weights(clusters_weights)

    def compute_weights(self):
        """
        Compute the optimized portfolio weights.
        """
        # compute estimators
        self.compute_estimators()

        # get clusters
        self.get_clusters()

        # compute optimal number of clusters
        self.compute_optimal_n_clusters()

        # get cluster children
        self.get_cluster_children()

        # quasi diagonalization
        self.idxs = self.quasi_diagonalization(self.n_assets * 2 - 2)

        # recursive bisection
        self.recursive_bisection()

        # weights
        self.weights = pd.DataFrame(self.weights * self.leverage, index=self.asset_names,
                                    columns=[self.returns.index[-1]]).iloc[self.idxs].T

        return self.weights

    def plot_clusters(self):
        """
        Plot the clusters.
        """
        plt.figure(figsize=(15, 7))
        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('Assets')
        plt.ylabel('Distance')
        dendrogram(
            self.clusters,
            labels=self.asset_names,
            orientation='left',
            leaf_rotation=0.,  # rotates the x axis labels
            leaf_font_size=8.,  # font size for the x axis labels
        )
        plt.tight_layout()
        plt.show()

    def plot_weights(self):
        """
        Plot the optimized portfolio weights.
        """
        plot_bar(self.weights.T.sort_values(by=[self.returns.index[-1]]), axis='horizontal', x_label='weights')


class NCO:
    """
    Nested clustering optimization class.

    This class computes the optimized portfolio weights based on the returns of the assets or strategies
    using nested clustering optimization techniques.
    """
    pass
