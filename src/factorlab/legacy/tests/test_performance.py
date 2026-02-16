import pytest
import pandas as pd
import numpy as np

from factorlab.feature_engineering.transformations import Transform
from factorlab.analysis.metrics import Metrics
from factorlab.analysis.performance import Performance


@pytest.fixture
def binance_spot_prices():
    """
    Fixture for crypto OHLCV prices.
    """
    # read csv from datasets/data
    df = pd.read_csv("datasets/data/binance_spot_prices.csv", index_col=['date', 'ticker'],
                     parse_dates=['date']).loc[:, : 'close']

    # drop tickers with nobs < ts_obs
    obs = df.groupby(level=1).count().min(axis=1)
    drop_tickers_list = obs[obs < 365].index.to_list()
    df = df.drop(drop_tickers_list, level=1, axis=0)

    return df


@pytest.fixture
def crypto_log_returns(binance_spot_prices):
    """
    Fixture for crypto OHLCV prices.
    """
    ret = Transform(binance_spot_prices).returns().close.unstack()

    return ret


@pytest.fixture
def crypto_simple_returns(binance_spot_prices):
    """
    Fixture for crypto OHLCV prices.
    """
    ret = Transform(binance_spot_prices).returns(method='simple').close.unstack()

    return ret


@pytest.fixture
def risk_free_rates():
    """
    Fixture for US real rates data.
    """
    # read csv from datasets/data
    return pd.read_csv("datasets/data/us_real_rates_10y_monthly.csv", index_col=['date'],
                       parse_dates=['date']).loc[:, 'US_Rates_3M'] / 100


class TestPerformance:
    """
    Test class for Metrics.
    """
    @pytest.fixture(autouse=True)
    def perf_setup_log_ret(self, crypto_log_returns, risk_free_rates):
        self.perf_log_ret_instance = Performance(crypto_log_returns, risk_free_rate=risk_free_rates, ret_type='log')

    @pytest.fixture(autouse=True)
    def perf_setup_simple_ret(self, crypto_simple_returns, risk_free_rates):
        self.perf_simple_ret_instance = Performance(crypto_simple_returns, risk_free_rate=risk_free_rates,
                                                   ret_type='simple')

    def test_initialization(self) -> None:
        """
        Test initialization.
        """
        # dtypes
        assert isinstance(self.perf_log_ret_instance, Performance)
        assert isinstance(self.perf_log_ret_instance.returns, pd.DataFrame)
        # index
        assert isinstance(self.perf_log_ret_instance.returns.index, pd.DatetimeIndex)

    def test_get_metrics(self) -> None:
        """
        Test get_metrics method.
        """
        metrics = self.perf_log_ret_instance.get_metrics()
        assert isinstance(metrics, Metrics)

    @pytest.mark.parametrize("metrics", ['returns', 'risks', 'ratios', 'alpha_beta', 'key_metrics', 'all'])
    def test_get_table(self, metrics) -> None:
        """
        Test get_table method.
        """
        metrics = self.perf_log_ret_instance.get_table(metrics)
        # dtypes
        assert isinstance(metrics, pd.DataFrame)
        assert (metrics.dtypes == np.float64).all()
        # shape
        assert metrics.shape[0] == self.perf_log_ret_instance.returns.shape[1]
        # index
        assert (metrics.index == self.perf_log_ret_instance.returns.columns).all()
