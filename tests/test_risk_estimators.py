import pytest
import pandas as pd
import numpy as np

from factorlab.strategy_backtesting.portfolio_optimization.risk_estimators import RiskEstimators


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


class TestRiskEstimators:
    """
    Test class for RiskEstimators.
    """
    @pytest.fixture(autouse=True)
    def re_default_instance(self, asset_returns):
        self.default_risk_est_instance = RiskEstimators(asset_returns)

    @pytest.fixture(autouse=True)
    def re_nomissing_instance(self, asset_returns):
        self.nomissing_risk_est_instance = RiskEstimators(asset_returns.dropna())

    def test_initialization(self) -> None:
        """
        Test initialization.
        """
        # types
        assert isinstance(self.default_risk_est_instance, RiskEstimators)
        assert isinstance(self.default_risk_est_instance.returns, pd.DataFrame)
        assert isinstance(self.default_risk_est_instance.window_size, np.integer)
        assert isinstance(self.default_risk_est_instance.window_type, str)
        assert (self.default_risk_est_instance.returns.dtypes == np.float64).all()

    def test_covariance(self) -> None:
        """
        Test covariance computation.
        """
        self.default_risk_est_instance.covariance()
        # type
        assert isinstance(self.default_risk_est_instance.cov_matrix, np.ndarray)
        # shape
        assert self.default_risk_est_instance.cov_matrix.shape == (self.default_risk_est_instance.returns.shape[1],
                                         self.default_risk_est_instance.returns.shape[1])
        # vals
        assert np.isnan(self.default_risk_est_instance.cov_matrix).sum() == 0

    def test_empirical_covariance(self) -> None:
        """
        Test empirical covariance computation.
        """
        self.nomissing_risk_est_instance.empirical_covariance()
        # type
        assert isinstance(self.nomissing_risk_est_instance.cov_matrix, np.ndarray)
        assert self.nomissing_risk_est_instance.cov_matrix.dtype == np.float64
        # shape
        assert self.nomissing_risk_est_instance.cov_matrix.shape == (self.nomissing_risk_est_instance.returns.shape[1],
                                         self.nomissing_risk_est_instance.returns.shape[1])
        # vals
        assert np.isnan(self.nomissing_risk_est_instance.cov_matrix).sum() == 0

    def test_shrunk_covariance(self) -> None:
        """
        Test shrunk covariance computation.
        """
        self.nomissing_risk_est_instance.shrunk_covariance()
        # type
        assert isinstance(self.nomissing_risk_est_instance.cov_matrix, np.ndarray)
        assert self.nomissing_risk_est_instance.cov_matrix.dtype == np.float64
        # shape
        assert self.nomissing_risk_est_instance.cov_matrix.shape == (self.nomissing_risk_est_instance.returns.shape[1],
                                         self.nomissing_risk_est_instance.returns.shape[1])
        # vals
        assert np.isnan(self.nomissing_risk_est_instance.cov_matrix).sum() == 0

    def test_ledoit_wolf_covariance(self) -> None:
        """
        Test Ledoit-Wolf covariance computation.
        """
        self.nomissing_risk_est_instance.ledoit_wolf()
        # type
        assert isinstance(self.nomissing_risk_est_instance.cov_matrix, np.ndarray)
        assert self.nomissing_risk_est_instance.cov_matrix.dtype == np.float64
        # shape
        assert self.nomissing_risk_est_instance.cov_matrix.shape == (self.nomissing_risk_est_instance.returns.shape[1],
                                         self.nomissing_risk_est_instance.returns.shape[1])
        # vals
        assert np.isnan(self.nomissing_risk_est_instance.cov_matrix).sum() == 0

    def test_oas_covariance(self) -> None:
        """
        Test OAS covariance computation.
        """
        self.nomissing_risk_est_instance.oas()
        # type
        assert isinstance(self.nomissing_risk_est_instance.cov_matrix, np.ndarray)
        assert self.nomissing_risk_est_instance.cov_matrix.dtype == np.float64
        # shape
        assert self.nomissing_risk_est_instance.cov_matrix.shape == (self.nomissing_risk_est_instance.returns.shape[1],
                                         self.nomissing_risk_est_instance.returns.shape[1])
        # vals
        assert np.isnan(self.nomissing_risk_est_instance.cov_matrix).sum() == 0

    def test_graphical_lasso_covariance(self) -> None:
        """
        Test Graphical Lasso covariance computation.
        """
        self.nomissing_risk_est_instance.graphical_lasso()
        # type
        assert isinstance(self.nomissing_risk_est_instance.cov_matrix, np.ndarray)
        assert self.nomissing_risk_est_instance.cov_matrix.dtype == np.float64
        # shape
        assert self.nomissing_risk_est_instance.cov_matrix.shape == (self.nomissing_risk_est_instance.returns.shape[1],
                                         self.nomissing_risk_est_instance.returns.shape[1])
        # vals
        assert np.isnan(self.nomissing_risk_est_instance.cov_matrix).sum() == 0

    def test_graphical_lassocv_covariance(self) -> None:
        """
        Test Graphical Lasso CV covariance computation.
        """
        self.nomissing_risk_est_instance.graphical_lasso_cv()
        # type
        assert isinstance(self.nomissing_risk_est_instance.cov_matrix, np.ndarray)
        assert self.nomissing_risk_est_instance.cov_matrix.dtype == np.float64
        # shape
        assert self.nomissing_risk_est_instance.cov_matrix.shape == (self.nomissing_risk_est_instance.returns.shape[1],
                                         self.nomissing_risk_est_instance.returns.shape[1])
        # vals
        assert np.isnan(self.nomissing_risk_est_instance.cov_matrix).sum() == 0

    def test_minimum_covariance_determinant(self) -> None:
        """
        Test Minimum Covariance Determinant covariance computation.
        """
        self.nomissing_risk_est_instance.minimum_covariance_determinant()
        # type
        assert isinstance(self.nomissing_risk_est_instance.cov_matrix, np.ndarray)
        assert self.nomissing_risk_est_instance.cov_matrix.dtype == np.float64
        # shape
        assert self.nomissing_risk_est_instance.cov_matrix.shape == (self.nomissing_risk_est_instance.returns.shape[1],
                                         self.nomissing_risk_est_instance.returns.shape[1])
        # vals
        assert np.isnan(self.nomissing_risk_est_instance.cov_matrix).sum() == 0

    def test_semi_covariance(self) -> None:
        """
        Test semi-covariance computation.
        """
        self.default_risk_est_instance.semi_covariance()
        # type
        assert isinstance(self.default_risk_est_instance.cov_matrix, np.ndarray)
        # shape
        assert self.default_risk_est_instance.cov_matrix.shape == (self.default_risk_est_instance.returns.shape[1],
                                         self.default_risk_est_instance.returns.shape[1])
        # vals
        assert np.isnan(self.default_risk_est_instance.cov_matrix).sum() == 0

    def test_exponential_covariance(self) -> None:
        """
        Test exponential covariance computation.
        """
        self.default_risk_est_instance.exponential_covariance()
        # type
        assert isinstance(self.default_risk_est_instance.cov_matrix, np.ndarray)
        # shape
        assert self.default_risk_est_instance.cov_matrix.shape == (self.default_risk_est_instance.returns.shape[1],
                                         self.default_risk_est_instance.returns.shape[1])
        # vals
        assert np.isnan(self.default_risk_est_instance.cov_matrix).sum() == 0

    @pytest.mark.parametrize("method, detone", [('constant_residual_eigenval', True),
                                                ('constant_residual_eigenval', False), ('targeted_shrinkage', True),
                                                ('targeted_shrinkage', False)])
    def test_denoised_covariance(self, method, detone) -> None:
        """
        Test denoised covariance computation.
        """
        self.default_risk_est_instance.denoised_covariance(method=method, detone=detone)
        # type
        assert isinstance(self.default_risk_est_instance.cov_matrix, np.ndarray)
        # shape
        assert self.default_risk_est_instance.cov_matrix.shape == (self.default_risk_est_instance.returns.shape[1],
                                         self.default_risk_est_instance.returns.shape[1])
        # vals
        assert np.isnan(self.default_risk_est_instance.cov_matrix).sum() == 0

    @pytest.mark.parametrize("method", ['empirical_covariance', 'shrunk_covariance', 'ledoit_wolf', 'oas',
                                        'graphical_lasso', 'graphical_lasso_cv', 'minimum_covariance_determinant',
                                        'semi_covariance', 'exponential_covariance', 'denoised_covariance'])
    def test_compute_covariance_matrix(self, method) -> None:
        """
        Test compute covariance matrix.
        """
        self.nomissing_risk_est_instance.compute_covariance_matrix(method=method)
        # type
        assert isinstance(self.nomissing_risk_est_instance.cov_matrix, np.ndarray)
        # shape
        assert self.nomissing_risk_est_instance.cov_matrix.shape == (self.nomissing_risk_est_instance.returns.shape[1],
                                         self.nomissing_risk_est_instance.returns.shape[1])
        # vals
        assert np.isnan(self.nomissing_risk_est_instance.cov_matrix).sum() == 0

    def test_compute_covariance_matrix_errors(self):
        """
        Test compute covariance matrix errors.
        """
        # invalid method
        try:
            self.nomissing_risk_est_instance.compute_covariance_matrix(method='invalid')
        except ValueError as e:
            assert str(e) == "Method is not supported. Valid methods are: 'covariance', 'empirical_covariance', " \
                             "'shrunk_covariance', 'ledoit_wolf', 'oas', 'graphical_lasso', 'graphical_lasso_cv', " \
                             "'minimum_covariance_determinant', 'semi_covariance', 'exponential_covariance', " \
                             "'denoised_covariance'"

    def test_compute_turbulence_index(self) -> None:
        """
        Test compute turbulence index.
        """
        turb = self.nomissing_risk_est_instance.compute_turbulence_index()
        # type
        assert isinstance(turb, pd.DataFrame)
        # shape
        assert turb.shape[0] == self.nomissing_risk_est_instance.returns.shape[0]
        # vals
        assert turb.isna().sum().sum() == 0
