import pytest
import pandas as pd
import numpy as np

from factorlab.feature_engineering.factors.carry import Carry


@pytest.fixture
def binance_perp_fut():
    """
    Fixture for daily cryptoasset returns.
    """
    # read csv from datasets/data
    df = pd.read_csv('datasets/data/binance_perp_fut_prices.csv', index_col=['date', 'ticker'],
                     parse_dates=True)

    # drop tickers with nobs < ts_obs
    obs = df.iloc[:, :-1].groupby(level=1).count().min(axis=1)
    drop_tickers_list = obs[obs < 30].index.to_list()
    df = df.drop(drop_tickers_list, level=1, axis=0)

    # drop tickers with nobs < cs_obs
    obs = df.iloc[:, :-1].groupby(level=0).count().min(axis=1)
    idx_start = obs[obs > 3].index[0]
    df = df.unstack()[df.unstack().index > idx_start].stack()

    # sort cols
    df = df[['open', 'high', 'low', 'close', 'volume', 'funding_rate']]
    # rename close
    df = df.rename(columns={'close': 'spot', 'funding_rate': 'rate'})

    return df


@pytest.fixture()
def carry_instance_default(binance_perp_fut):
    return Carry(binance_perp_fut)


class TestCarry:
    """
    Test Carry class.
    """
    @pytest.fixture(autouse=True)
    def carry_setup_default(self, carry_instance_default):
        self.default_carry_instance = carry_instance_default

    def test_initialization(self):
        """
        Test initialization.
        """
        assert isinstance(self.default_carry_instance, Carry)
        assert isinstance(self.default_carry_instance.df, pd.DataFrame)
        assert isinstance(self.default_carry_instance.df.index, pd.MultiIndex)
        assert 'spot' in self.default_carry_instance.df.columns
        assert 'rate' in self.default_carry_instance.df.columns

    def test_compute_carry(self, binance_perp_fut):
        """
        Test compute_carry method.
        """
        # get actual and expected
        actual = self.default_carry_instance.compute_carry()
        expected = binance_perp_fut.rate.unstack().dropna(how='all').stack().to_frame('carry').sort_index() * -1

        # check inf
        assert not actual.isin([np.inf, -np.inf]).any().any()

        # test values
        assert all(actual == expected)

        # test shape
        assert actual.shape[1] == 1
        assert actual.unstack().shape[1] == expected.unstack().shape[1]
        assert actual.unstack().shape[0] == expected.unstack().shape[0]

        # test data type
        assert isinstance(actual, pd.DataFrame)
        assert all(actual.dtypes == np.float64)

        # test name
        assert actual.columns[0] == 'carry'

    def test_smooth(self, binance_perp_fut):
        """
        Test smooth method.
        """
        # get actual and expected
        actual = self.default_carry_instance.smooth()

        # check inf
        assert not actual.isin([np.inf, -np.inf]).any().any()

        # test shape
        assert actual.shape[1] == 1

        # test data type
        assert isinstance(actual, pd.DataFrame)
        assert all(actual.dtypes == np.float64)

        # test name
        assert actual.columns[0] == 'carry_mean_3'

    def test_carry_risk_ratio(self, binance_perp_fut):
        """
        Test smooth method.
        """
        # get actual and expected
        actual = self.default_carry_instance.carry_risk_ratio()

        # check inf
        assert not actual.isin([np.inf, -np.inf]).any().any()

        # test shape
        assert actual.shape[1] == 1

        # test data type
        assert isinstance(actual, pd.DataFrame)
        assert all(actual.dtypes == np.float64)

        # test name
        assert actual.columns[0] == 'carry_to_std_30'
