import pytest
import pandas as pd
import numpy as np

from factorlab.strategy_backtesting.portfolio_optimization.return_estimators import ReturnEstimators


@pytest.fixture
def btc_spot_ret():
    """
    Fixture for BTC OHLCV prices.
    """
    # read csv from datasets/data
    df = pd.read_csv("../src/factorlab/datasets/data/cc_spot_prices.csv", index_col=['date', 'ticker'],
                     parse_dates=['date'])
    return df.loc[:, 'BTC', :]['close'].pct_change()


@pytest.fixture
def risk_free_rates():
    """
    Fixture for US real rates data.
    """
    # read csv from datasets/data
    return pd.read_csv("../src/factorlab/datasets/data/us_real_rates_10y_monthly.csv", index_col=['date'],
                       parse_dates=['date']).loc[:, 'US_Rates_3M'] / 100


class TestReturnEstimators:
    """
    Test class for ReturnEstimators.
    """

    @pytest.fixture(autouse=True)
    def transform_setup_default(self, btc_spot_ret, risk_free_rates):
        self.default_ret_est_instance = ReturnEstimators(btc_spot_ret, risk_free_rate=risk_free_rates,
                                                         as_excess_returns=True)

    def test_initialization(self) -> None:
        """
        Test initialization.
        """
        # types
        assert isinstance(self.default_ret_est_instance, ReturnEstimators)
        assert isinstance(self.default_ret_est_instance.returns, pd.DataFrame)
        assert isinstance(self.default_ret_est_instance.method, str)
        assert isinstance(self.default_ret_est_instance.as_excess_returns, bool)
        assert isinstance(self.default_ret_est_instance.risk_free_rate, pd.Series)
        assert isinstance(self.default_ret_est_instance.as_ann_returns, bool)
        assert isinstance(self.default_ret_est_instance.freq, str)
        assert isinstance(self.default_ret_est_instance.ann_factor, np.integer)
        assert isinstance(self.default_ret_est_instance.window_size, np.integer)
        assert (self.default_ret_est_instance.returns.dtypes == np.float64).all()

    def test_deannualize_rf_rate(self) -> None:
        """
        Test deannualize risk-free rate.
        """
        rf_rate = self.default_ret_est_instance.risk_free_rate.copy()
        # de-annualize
        self.default_ret_est_instance.deannualize_rf_rate()
        assert rf_rate.dropna().abs().ge(self.default_ret_est_instance.risk_free_rate.dropna().abs()).all()

    def test_compute_excess_returns(self) -> None:
        """
        Test compute expected returns.
        """
        rets = self.default_ret_est_instance.returns.copy()
        # excess returns
        self.default_ret_est_instance.compute_excess_returns()
        assert ((rets > self.default_ret_est_instance.returns).sum() /
                self.default_ret_est_instance.returns.shape[0] > 0.99).all()

    def test_historical_mean_returns(self) -> None:
        """
        Test compute historical mean returns.
        """
        self.default_ret_est_instance.compute_historical_mean_returns()
        assert isinstance(self.default_ret_est_instance.exp_returns.iloc[0],  np.float64)
        assert (self.default_ret_est_instance.exp_returns == self.default_ret_est_instance.returns.mean()).all()

    def test_historical_median_returns(self) -> None:
        """
        Test compute historical median returns.
        """
        self.default_ret_est_instance.compute_historical_median_returns()
        assert isinstance(self.default_ret_est_instance.exp_returns.iloc[0],  np.float64)
        assert (self.default_ret_est_instance.exp_returns == self.default_ret_est_instance.returns.median()).all()

    def test_ewma_returns(self) -> None:
        """
        Test compute EWMA returns.
        """
        self.default_ret_est_instance.compute_ewma_returns()
        assert isinstance(self.default_ret_est_instance.exp_returns.iloc[0],  np.float64)
        assert (self.default_ret_est_instance.exp_returns == self.default_ret_est_instance.returns.ewm(
            span=self.default_ret_est_instance.window_size).mean().iloc[-1]).all()

    @pytest.mark.parametrize("method", ['median', 'mean', 'ewma'])
    def test_compute_expected_returns(self, method) -> None:
        """
        Test compute expected returns.
        """
        self.default_ret_est_instance.method = method
        self.default_ret_est_instance.compute_expected_returns()
        assert isinstance(self.default_ret_est_instance.exp_returns.iloc[0],  np.float64)
