import pytest
import pandas as pd
import numpy as np

from factorlab.strategy_backtesting.portfolio_optimization.clustering import HRP, HERC


@pytest.fixture
def asset_returns():
    """
    Fixture for asset returns.
    """
    df = pd.read_csv("../src/factorlab/datasets/data/asset_excess_returns_daily.csv", index_col=0)
    df.index = pd.to_datetime(df.index)
    df.index.name = 'date'

    # drop tickers with nobs < ts_obs
    obs = df.count()
    drop_tickers_list = obs[obs < 260].index.to_list()
    df = df.drop(columns=drop_tickers_list)

    # stack
    df = df.stack(future_stack=True).to_frame('ret')
    # replace no chg
    df = df.replace(0, np.nan)
    # start date
    rets = df.dropna().unstack().ret

    return rets


class TestHRP:
    """
    Test class for Hierarchical Risk Parity.
    """
    @pytest.fixture(autouse=True)
    def hrp_default_instance(self, asset_returns):
        self.default_hrp_instance = HRP(asset_returns)

    @pytest.fixture(autouse=True)
    def hrp_nomissing_instance(self, asset_returns):
        self.nomissing_hrp_instance = HRP(asset_returns.dropna())

    def test_initialization(self) -> None:
        """
        Test initialization.
        """
        # types
        assert isinstance(self.default_hrp_instance, HRP)
        assert isinstance(self.default_hrp_instance.returns, pd.DataFrame)
        assert (self.default_hrp_instance.returns.dtypes == np.float64).all()
        assert isinstance(self.default_hrp_instance.linkage_method, str)
        assert isinstance(self.default_hrp_instance.distance_metric, str)
        assert isinstance(self.default_hrp_instance.leverage, float)
        assert isinstance(self.default_hrp_instance.asset_names, list)
        assert isinstance(self.default_hrp_instance.n_assets, int)

        # vals
        assert self.default_hrp_instance.asset_names == self.default_hrp_instance.returns.columns.tolist()
        assert self.default_hrp_instance.n_assets == self.default_hrp_instance.returns.shape[1]

    def test_compute_estimators(self) -> None:
        """
        Test compute_estimators.
        """
        self.default_hrp_instance.compute_estimators()

        # dtypes
        assert isinstance(self.default_hrp_instance.cov_matrix, np.ndarray)
        assert isinstance(self.default_hrp_instance.corr_matrix, np.ndarray)
        assert isinstance(self.default_hrp_instance.distance_matrix, np.ndarray)
        assert self.default_hrp_instance.cov_matrix.dtype == np.float64
        assert self.default_hrp_instance.corr_matrix.dtype == np.float64
        assert self.default_hrp_instance.distance_matrix.dtype == np.float64

        # shapes
        assert self.default_hrp_instance.cov_matrix.shape == (self.default_hrp_instance.n_assets,
                                                              self.default_hrp_instance.n_assets)
        assert self.default_hrp_instance.corr_matrix.shape == (self.default_hrp_instance.n_assets,
                                                               self.default_hrp_instance.n_assets)
        assert self.default_hrp_instance.distance_matrix.shape == (self.default_hrp_instance.n_assets,
                                                                   self.default_hrp_instance.n_assets)

        # vals
        assert np.isnan(self.default_hrp_instance.cov_matrix).sum() == 0
        assert np.isnan(self.default_hrp_instance.corr_matrix).sum() == 0
        assert np.isnan(self.default_hrp_instance.distance_matrix).sum() == 0

    @pytest.mark.parametrize("linkage_method", ['single', 'complete', 'average', 'weighted', 'centroid', 'median',
                                                'ward'])
    def test_tree_clustering(self, linkage_method) -> None:
        """
        Test tree_clustering.
        """
        self.default_hrp_instance.compute_estimators()
        self.default_hrp_instance.linkage_method = linkage_method
        self.default_hrp_instance.tree_clustering()

        # dtypes
        assert isinstance(self.default_hrp_instance.clusters, np.ndarray)
        assert self.default_hrp_instance.clusters.dtype == np.float64

    def test_quasi_diagnalization(self) -> None:
        """
        Test quasi_diagnalization.
        """
        self.default_hrp_instance.compute_estimators()
        self.default_hrp_instance.tree_clustering()
        sorted_idx = self.default_hrp_instance.quasi_diagonalization()

        # dtypes
        assert isinstance(sorted_idx, list)

    def test_compute_inverse_variance_weights(self) -> None:
        """
        Test compute_inverse_variance_weights.
        """
        self.default_hrp_instance.compute_estimators()
        w = self.default_hrp_instance.compute_inverse_variance_weights(self.default_hrp_instance.cov_matrix)

        # dtypes
        assert isinstance(w, np.ndarray)
        assert w.dtype == np.float64

        # shape
        assert w.shape == (self.default_hrp_instance.n_assets,)

        # vals
        assert np.allclose(np.sum(w), 1.0)
        assert np.all(w >= 0)

    def test_compute_cluster_variance(self) -> None:
        """
        Test compute_cluster_variance.
        """
        self.default_hrp_instance.compute_estimators()
        self.default_hrp_instance.tree_clustering()
        sorted_idxs = self.default_hrp_instance.quasi_diagonalization()
        cv = self.default_hrp_instance.compute_cluster_variance(sorted_idxs)

        # dtypes
        assert isinstance(cv, float)

    def test_recursive_bisection(self) -> None:
        """
        Test recursive_bisection.
        """
        # compute estimators
        self.default_hrp_instance.compute_estimators()
        # tree clustering
        self.default_hrp_instance.tree_clustering()
        # quasi diagonalization
        self.default_hrp_instance.idxs = self.default_hrp_instance.quasi_diagonalization()
        # recursive bisection
        self.default_hrp_instance.recursive_bisection()

        # dtypes
        assert isinstance(self.default_hrp_instance.weights, pd.DataFrame)
        assert isinstance(self.default_hrp_instance.asset_names, list)
        assert (self.default_hrp_instance.weights.dtypes == np.float64).all()

        # shape
        assert self.default_hrp_instance.weights.shape == (self.default_hrp_instance.n_assets, 1)

        # vals
        assert (self.default_hrp_instance.weights >= 0).all().all()

    def test_create_portfolio(self) -> None:
        """
        Test create_portfolio.
        """
        weights = np.ones(self.default_hrp_instance.n_assets)
        weights[-1] = -1
        self.default_hrp_instance.side_weights = pd.Series(index=self.default_hrp_instance.asset_names, data=weights)
        # compute estimators
        self.default_hrp_instance.compute_estimators()
        # tree clustering
        self.default_hrp_instance.tree_clustering()
        # quasi diagonalization
        self.default_hrp_instance.idxs = self.default_hrp_instance.quasi_diagonalization()
        # recursive bisection
        self.default_hrp_instance.recursive_bisection()
        self.default_hrp_instance.create_portfolio()

        # dtypes
        assert isinstance(self.default_hrp_instance.weights, pd.DataFrame)
        assert (self.default_hrp_instance.weights.dtypes == np.float64).all()

        # shape
        assert self.default_hrp_instance.weights.T.shape == (self.default_hrp_instance.n_assets, 1)

        # vals
        assert (self.default_hrp_instance.weights.abs() >= 0).all().all()
        assert (self.default_hrp_instance.weights.abs() <= 1).all().all()

    @pytest.mark.parametrize("cov_variance_method", ['covariance', 'empirical_covariance', 'shrunk_covariance',
                                                     'ledoit_wolf', 'oas', 'graphical_lasso', 'graphical_lasso_cv',
                                                     'minimum_covariance_determinant', 'semi_covariance',
                                                     'exponential_covariance', 'denoised_covariance'])
    def test_compute_weights(self, cov_variance_method) -> None:
        """
        Test compute_weights.
        """
        self.default_hrp_instance.cov_matrix_method = cov_variance_method
        self.default_hrp_instance.compute_weights()

        # dtypes
        assert isinstance(self.default_hrp_instance.weights, pd.DataFrame)
        assert isinstance(self.default_hrp_instance.asset_names, list)
        assert (self.default_hrp_instance.weights.dtypes == np.float64).all()

        # shape
        assert self.default_hrp_instance.weights.shape == (1, self.default_hrp_instance.n_assets)

        # vals
        assert (self.default_hrp_instance.weights >= 0).all().all()
        assert (self.default_hrp_instance.weights <= 1).all().all()
        assert np.allclose(self.default_hrp_instance.weights.sum().sum(), 1.0)

        # col
        assert set(self.default_hrp_instance.weights.columns.to_list()) == set(self.default_hrp_instance.asset_names)

        # index
        assert self.default_hrp_instance.weights.index == [self.default_hrp_instance.returns.index[-1]]


class TestHERC:
    """
    Test class for Hierarchical Equal Risk Contribution.
    """
    @pytest.fixture(autouse=True)
    def herc_default_instance(self, asset_returns):
        self.default_herc_instance = HERC(asset_returns.dropna())

    def test_initialization(self) -> None:
        """
        Test initialization.
        """
        # types
        assert isinstance(self.default_herc_instance, HERC)
        assert isinstance(self.default_herc_instance.returns, pd.DataFrame)
        assert (self.default_herc_instance.returns.dtypes == np.float64).all()
        assert isinstance(self.default_herc_instance.alpha, float)
        assert isinstance(self.default_herc_instance.risk_measure, str)
        assert isinstance(self.default_herc_instance.linkage_method, str)
        assert isinstance(self.default_herc_instance.distance_metric, str)
        assert isinstance(self.default_herc_instance.asset_names, list)
        assert isinstance(self.default_herc_instance.n_assets, int)

    def test_compute_estimators(self) -> None:
        """
        Test compute_estimators.
        """
        self.default_herc_instance.compute_estimators()

        # dtypes
        assert isinstance(self.default_herc_instance.cov_matrix, np.ndarray)
        assert isinstance(self.default_herc_instance.corr_matrix, np.ndarray)
        assert isinstance(self.default_herc_instance.distance_matrix, np.ndarray)
        assert self.default_herc_instance.cov_matrix.dtype == np.float64
        assert self.default_herc_instance.corr_matrix.dtype == np.float64
        assert self.default_herc_instance.distance_matrix.dtype == np.float64

        # shapes
        assert self.default_herc_instance.cov_matrix.shape == (self.default_herc_instance.n_assets,
                                                               self.default_herc_instance.n_assets)
        assert self.default_herc_instance.corr_matrix.shape == (self.default_herc_instance.n_assets,
                                                                self.default_herc_instance.n_assets)
        assert self.default_herc_instance.distance_matrix.shape == (self.default_herc_instance.n_assets,
                                                                    self.default_herc_instance.n_assets)

        # vals
        assert np.isnan(self.default_herc_instance.cov_matrix).sum() == 0
        assert np.isnan(self.default_herc_instance.corr_matrix).sum() == 0
        assert np.isnan(self.default_herc_instance.distance_matrix).sum() == 0

    def test_get_clusters(self) -> None:
        """
        Test get_clusters.
        """
        self.default_herc_instance.compute_estimators()
        self.default_herc_instance.get_clusters()

        # dtypes
        assert isinstance(self.default_herc_instance.clusters, np.ndarray)
        assert self.default_herc_instance.clusters.dtype == np.float64

    def test_compute_optimal_n_clusters(self) -> None:
        """
        Test compute_optimal_n_clusters.
        """
        self.default_herc_instance.compute_estimators()
        self.default_herc_instance.get_clusters()
        self.default_herc_instance.compute_optimal_n_clusters()

        # dtypes
        assert isinstance(self.default_herc_instance.n_clusters, np.int64)

        # vals
        assert self.default_herc_instance.n_clusters == 6

    def test_get_cluster_children(self) -> None:
        """
        Test get_cluster_children.
        """
        self.default_herc_instance.compute_estimators()
        self.default_herc_instance.get_clusters()
        self.default_herc_instance.compute_optimal_n_clusters()
        self.default_herc_instance.get_cluster_children()

        # dtypes
        assert isinstance(self.default_herc_instance.cluster_children, dict)

    def test_quasi_diagnalization(self) -> None:
        """
        Test quasi_diagnalization.
        """
        self.default_herc_instance.compute_estimators()
        self.default_herc_instance.get_clusters()
        self.default_herc_instance.compute_optimal_n_clusters()
        self.default_herc_instance.get_cluster_children()
        idxs = self.default_herc_instance.quasi_diagonalization(self.default_herc_instance.n_assets * 2 - 2)

        # dtypes
        assert isinstance(idxs, list)

    def test_get_intersection(self) -> None:
        """
        Test get_intersection.
        """
        list1 = [1, 2, 3, 4, 5]
        list2 = [4, 5, 6, 7, 8]
        intersection = self.default_herc_instance.get_intersection(list1, list2)

        # dtypes
        assert isinstance(intersection, list)

    def test_get_children_cluster_idxs(self) -> None:
        """
        Test get_children_cluster_idxs.
        """
        self.default_herc_instance.compute_estimators()
        self.default_herc_instance.get_clusters()
        self.default_herc_instance.compute_optimal_n_clusters()
        self.default_herc_instance.get_cluster_children()
        children_idxs = self.default_herc_instance.get_children_cluster_idxs(0)

        # dtypes
        assert isinstance(children_idxs, tuple)

    def test_compute_inverse_variance_weights(self) -> None:
        """
        Test compute_inverse_variance_weights.
        """
        self.default_herc_instance.compute_estimators()
        w = self.default_herc_instance.compute_inverse_variance_weights(self.default_herc_instance.cov_matrix)

        # dtypes
        assert isinstance(w, np.ndarray)
        assert w.dtype == np.float64

        # shape
        assert w.shape == (self.default_herc_instance.n_assets,)

        # vals
        assert np.allclose(np.sum(w), 1.0)
        assert np.all(w >= 0)

    def test_compute_inverse_cvar_weights(self) -> None:
        """
        Test compute_inverse_cvar_weights.
        """
        self.default_herc_instance.compute_estimators()
        w = self.default_herc_instance.compute_inverse_cvar_weights(self.default_herc_instance.returns)

        # dtypes
        assert isinstance(w, np.ndarray)
        assert w.dtype == np.float64

        # shape
        assert w.shape == (self.default_herc_instance.n_assets,)

        # vals
        assert np.allclose(np.sum(w), 1.0)
        assert np.all(w >= 0)

    def test_compute_inverse_cdar_weights(self) -> None:
        """
        Test compute_inverse_cvar_weights.
        """
        self.default_herc_instance.compute_estimators()
        w = self.default_herc_instance.compute_inverse_cdar_weights(self.default_herc_instance.returns)

        # dtypes
        assert isinstance(w, np.ndarray)
        assert w.dtype == np.float64

        # shape
        assert w.shape == (self.default_herc_instance.n_assets,)

        # vals
        assert np.allclose(np.sum(w), 1.0)
        assert np.all(w >= 0)

    def test_compute_cluster_variance(self) -> None:
        """
        Test compute_cluster_variance.
        """
        self.default_herc_instance.compute_estimators()
        cv = self.default_herc_instance.compute_cluster_variance([1, 2, 3])

        # dtypes
        assert isinstance(cv, float)

    def test_compute_cluster_expected_shortfall(self) -> None:
        """
        Test compute_cluster_expected_shortfall.
        """
        self.default_herc_instance.compute_estimators()
        ces = self.default_herc_instance.compute_cluster_expected_shortfall([1, 2, 3])[0]

        # dtypes
        assert isinstance(ces, float)

    def test_compute_cluster_conditional_drawdown_risk(self) -> None:
        """
        Test compute_cluster_conditional_drawdown_risk.
        """
        self.default_herc_instance.compute_estimators()
        cddr = self.default_herc_instance.compute_cluster_conditional_drawdown_risk([1, 2, 3])[0]

        # dtypes
        assert isinstance(cddr, float)

    @pytest.mark.parametrize("risk_measure", ['equal_weight', 'variance', 'std', 'expected_shortfall',
                                              'conditional_drawdown_risk'])
    def test_compute_cluster_risk_contribution(self, risk_measure) -> None:
        """
        Test compute_cluster_risk_contribution.
        """
        self.default_herc_instance.compute_estimators()
        self.default_herc_instance.get_clusters()
        self.default_herc_instance.compute_optimal_n_clusters()
        self.default_herc_instance.get_cluster_children()
        self.default_herc_instance.risk_measure = risk_measure
        self.default_herc_instance.compute_cluster_risk_contribution()

        # dtypes
        assert isinstance(self.default_herc_instance.clusters_contribution, np.ndarray)
        assert self.default_herc_instance.clusters_contribution.dtype == np.float64

        # shape
        assert self.default_herc_instance.clusters_contribution.shape == (6,)

    @pytest.mark.parametrize("risk_measure", ['equal_weight', 'variance', 'std', 'expected_shortfall',
                                              'conditional_drawdown_risk'])
    def test_compute_naive_risk_parity_weights(self, risk_measure) -> None:
        """
        Test compute_naive_risk_parity_weights.
        """
        self.default_herc_instance.compute_estimators()
        self.default_herc_instance.get_clusters()
        self.default_herc_instance.compute_optimal_n_clusters()
        self.default_herc_instance.get_cluster_children()
        self.default_herc_instance.risk_measure = risk_measure
        w = self.default_herc_instance.compute_naive_risk_parity_weights(self.default_herc_instance.returns,
                                                                         self.default_herc_instance.cov_matrix, 0)

        # dtypes
        assert isinstance(w, np.ndarray)
        assert w.dtype == np.float64

        # shape
        if risk_measure == 'equal_weight':
            assert w.shape == (8,)
        else:
            assert w.shape == (self.default_herc_instance.n_assets,)

        # vals
        assert np.allclose(np.sum(w), 1.0)
        assert np.all(w >= 0)

    def test_recursive_bisection(self) -> None:
        """
        Test recursive_bisection.
        """
        self.default_herc_instance.compute_estimators()
        self.default_herc_instance.get_clusters()
        self.default_herc_instance.compute_optimal_n_clusters()
        self.default_herc_instance.get_cluster_children()
        self.default_herc_instance.quasi_diagonalization(self.default_herc_instance.n_assets * 2 - 2)
        self.default_herc_instance.recursive_bisection()

        # dtypes
        assert isinstance(self.default_herc_instance.weights, np.ndarray)
        assert isinstance(self.default_herc_instance.asset_names, list)
        assert self.default_herc_instance.weights.dtype == np.float64

        # shape
        assert self.default_herc_instance.weights.shape == (self.default_herc_instance.n_assets, )

        # vals
        assert (self.default_herc_instance.weights >= 0).all().all()

    @pytest.mark.parametrize("risk_measure", ['equal_weight', 'variance', 'std', 'expected_shortfall',
                                              'conditional_drawdown_risk'])
    def test_compute_weights(self, risk_measure) -> None:
        """
        Test compute_weights.
        """
        self.default_herc_instance.risk_measure = risk_measure
        self.default_herc_instance.compute_weights()

        # dtypes
        assert isinstance(self.default_herc_instance.weights, pd.DataFrame)
        assert isinstance(self.default_herc_instance.asset_names, list)
        assert (self.default_herc_instance.weights.dtypes == np.float64).all()

        # shape
        assert self.default_herc_instance.weights.shape == (1, self.default_herc_instance.n_assets)

        # vals
        assert (self.default_herc_instance.weights >= 0).all().all()
        assert (self.default_herc_instance.weights <= 1).all().all()
        assert np.allclose(self.default_herc_instance.weights.sum().sum(), 1.0)

        # col
        assert set(self.default_herc_instance.weights.columns.to_list()) == set(self.default_herc_instance.asset_names)

        # index
        assert self.default_herc_instance.weights.index == [self.default_herc_instance.returns.index[-1]]
