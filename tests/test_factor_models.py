import pytest
import pandas as pd
import statsmodels

from factorlab.feature_engineering.transformations import Transform
from factorlab.feature_engineering.factors.trend import Trend
from factorlab.strategy_analysis.factor_models import FactorModel


@pytest.fixture
def spot_prices():
    """
    Fixture for crypto spot prices.
    """
    # read csv from datasets/data
    df = pd.read_csv("datasets/data/binance_spot_prices.csv", index_col=['date', 'ticker'],
                     parse_dates=True).loc[:, : 'close']

    # drop tickers with nobs < ts_obs
    obs = df.groupby(level=1).count().min(axis=1)
    drop_tickers_list = obs[obs < 365].index.to_list()
    df = df.drop(drop_tickers_list, level=1, axis=0)

    # drop tickers with nobs < cs_obs
    obs = df.groupby(level=0).count().min(axis=1)
    idx_start = obs[obs > 5].index[0]
    df = df.unstack()[df.unstack().index > idx_start].stack()

    return df


@pytest.fixture
def spot_ret(spot_prices):
    """
    Fixture for spot returns.
    """
    # compute returns
    spot_ret = Transform(spot_prices).returns()

    return spot_ret


@pytest.fixture
def btc_spot_ret(spot_ret):
    """
    Fixture for spot returns.
    """
    # get btc returns
    btc_spot_ret = spot_ret.loc[pd.IndexSlice[:, 'BTC'], :].droplevel(1)

    return btc_spot_ret


@pytest.fixture
def price_mom(spot_prices):
    """
    Fixture for crypto price momentum.
    """
    # compute price mom
    price_mom = Trend(spot_prices, lookback=5).price_mom()
    price_mom['price_mom_10'] = Trend(spot_prices, lookback=10).price_mom()
    price_mom['price_mom_15'] = Trend(spot_prices, lookback=15).price_mom()
    price_mom['price_mom_30'] = Trend(spot_prices, lookback=30).price_mom()
    price_mom['price_mom_45'] = Trend(spot_prices, vwap=True, log=True, lookback=45).price_mom()

    return price_mom


@pytest.fixture
def btc_price_mom(price_mom):
    """
    Fixture for BTC price momentum.
    """
    # compute btc price mom
    btc_price_mom = price_mom.loc[pd.IndexSlice[:, 'BTC'], :].droplevel(1)

    return btc_price_mom


# noinspection PyUnresolvedReferences
class TestFactorModel:
    """
    Test Filter class.
    """
    @pytest.fixture(autouse=True)
    def fm_setup(self, spot_ret, price_mom):
        self.fm_instance = FactorModel(spot_ret.close, price_mom)

    @pytest.fixture(autouse=True)
    def fm_setup_btc(self, btc_spot_ret, btc_price_mom):
        self.fm_btc_instance = FactorModel(btc_spot_ret.close, btc_price_mom)

    def test_initialization(self) -> None:
        """
        Test initialization.
        """
        assert isinstance(self.fm_instance, FactorModel)
        assert isinstance(self.fm_btc_instance, FactorModel)
        assert isinstance(self.fm_instance.data, pd.DataFrame)
        assert isinstance(self.fm_btc_instance.data, pd.DataFrame)
        assert isinstance(self.fm_instance.ret, pd.DataFrame)
        assert isinstance(self.fm_btc_instance.ret, pd.DataFrame)
        assert isinstance(self.fm_instance.factors, pd.DataFrame)
        assert isinstance(self.fm_btc_instance.factors, pd.DataFrame)
        assert all(self.fm_instance.index == self.fm_instance.data.index)
        assert all(self.fm_btc_instance.index == self.fm_btc_instance.data.index)
        assert self.fm_instance.freq == 'D'
        assert self.fm_btc_instance.freq == 'D'

    def test_preprocess_data(self) -> None:
        """
        Test preprocess_data method.
        """
        self.fm_instance.preprocess_data()
        self.fm_btc_instance.preprocess_data()

        # test type
        assert (self.fm_instance.ret.dtypes == 'float64').all()
        assert (self.fm_btc_instance.ret.dtypes == 'float64').all()
        assert (self.fm_instance.factors.dtypes == 'float64').all()
        assert (self.fm_btc_instance.factors.dtypes == 'float64').all()
        assert isinstance(self.fm_instance.freq, str)
        assert isinstance(self.fm_btc_instance.freq, str)

        # test values
        assert self.fm_instance.ret.isin(self.fm_instance.data).all().all()
        assert self.fm_btc_instance.ret.isin(self.fm_btc_instance.data).all().all()
        assert self.fm_instance.factors.isin(self.fm_instance.data).all().all()
        assert self.fm_btc_instance.factors.isin(self.fm_btc_instance.data).all().all()
        assert self.fm_instance.ret.isna().sum().sum() == 0
        assert self.fm_btc_instance.ret.isna().sum().sum() == 0

        # test shape
        assert self.fm_instance.ret.shape[0] == self.fm_instance.data.shape[0]
        assert self.fm_btc_instance.ret.shape[0] == self.fm_btc_instance.data.shape[0]
        assert self.fm_instance.factors.shape[0] == self.fm_instance.data.shape[0]
        assert self.fm_btc_instance.factors.shape[0] == self.fm_btc_instance.data.shape[0]

        # test index
        assert all(self.fm_instance.ret.index == self.fm_instance.data.index)
        assert all(self.fm_btc_instance.ret.index == self.fm_btc_instance.data.index)
        assert all(self.fm_instance.factors.index == self.fm_instance.data.index)
        assert all(self.fm_btc_instance.factors.index == self.fm_btc_instance.data.index)
        assert isinstance(self.fm_instance.index.droplevel(1), pd.DatetimeIndex)
        assert isinstance(self.fm_btc_instance.index, pd.DatetimeIndex)

        # test columns
        assert all(self.fm_instance.ret.columns == self.fm_instance.data.columns[0])
        assert all(self.fm_btc_instance.ret.columns == self.fm_btc_instance.data.columns[0])
        assert all(self.fm_instance.factors.columns == self.fm_instance.data.columns[1:])
        assert all(self.fm_btc_instance.factors.columns == self.fm_btc_instance.data.columns[1:])

    def test_get_ann_factor(self) -> None:
        """
        Test get_ann_factor method.
        """
        self.fm_instance.get_ann_factor()
        self.fm_btc_instance.get_ann_factor()

        # test type
        assert isinstance(self.fm_instance.ann_factor, int)
        assert isinstance(self.fm_btc_instance.ann_factor, int)

        # test values
        assert self.fm_instance.ann_factor in [365, 252, 52, 12, 4, 1]
        assert self.fm_btc_instance.ann_factor in [365, 252, 52, 12, 4, 1]

    def test_normalize_data(self) -> None:
        """
        Test normalize_data method.
        """
        self.fm_instance.normalize_data()
        self.fm_btc_instance.normalize_data()

        # test type
        assert (self.fm_instance.ret.dtypes == 'float64').all()
        assert (self.fm_btc_instance.ret.dtypes == 'float64').all()
        assert (self.fm_instance.factors.dtypes == 'float64').all()
        assert (self.fm_btc_instance.factors.dtypes == 'float64').all()

        # test values
        assert self.fm_instance.ret.isin(self.fm_instance.data).all().all()
        assert self.fm_btc_instance.ret.isin(self.fm_btc_instance.data).all().all()
        assert self.fm_instance.factors.isin(self.fm_instance.data).all().all()
        assert self.fm_btc_instance.factors.isin(self.fm_btc_instance.data).all().all()
        assert self.fm_instance.ret.isna().sum().sum() == 0
        assert self.fm_btc_instance.ret.isna().sum().sum() == 0

        # test shape
        assert self.fm_instance.ret.shape[0] == self.fm_instance.data.shape[0]
        assert self.fm_btc_instance.ret.shape[0] == self.fm_btc_instance.data.shape[0]
        assert self.fm_instance.factors.shape[0] == self.fm_instance.data.shape[0]
        assert self.fm_btc_instance.factors.shape[0] == self.fm_btc_instance.data.shape[0]

        # test index
        assert all(self.fm_instance.ret.index == self.fm_instance.data.index)
        assert all(self.fm_btc_instance.ret.index == self.fm_btc_instance.data.index)
        assert all(self.fm_instance.factors.index == self.fm_instance.data.index)
        assert all(self.fm_btc_instance.factors.index == self.fm_btc_instance.data.index)

        # test columns
        assert all(self.fm_instance.ret.columns == self.fm_instance.data.columns[0])
        assert all(self.fm_btc_instance.ret.columns == self.fm_btc_instance.data.columns[0])
        assert all(self.fm_instance.factors.columns == self.fm_instance.data.columns[1:])
        assert all(self.fm_btc_instance.factors.columns == self.fm_btc_instance.data.columns[1:])

    @pytest.mark.parametrize("window_type", ['fixed', 'expanding', 'rolling'])
    def test_orthongalize_factors(self, spot_ret, price_mom, btc_spot_ret, btc_price_mom, window_type) -> None:
        """
        Test orthongalize_factors method.
        """
        actual = FactorModel(spot_ret.close, price_mom, window_type=window_type, window_size=30,
                             orthogonalize=True)
        actual_btc = FactorModel(btc_spot_ret.close, btc_price_mom, window_type=window_type, window_size=30,
                                 orthogonalize=True)

        # test shape
        if window_type == 'fixed':
            assert actual.factors.shape == self.fm_instance.factors.shape
            assert actual_btc.factors.shape == self.fm_btc_instance.factors.shape
        else:
            assert actual.factors.shape[1] == self.fm_instance.factors.shape[1]
            assert actual_btc.factors.shape[1] == self.fm_btc_instance.factors.shape[1]

        # test type
        assert isinstance(actual.factors, pd.DataFrame)
        assert isinstance(actual_btc.factors, pd.DataFrame)
        assert (actual.factors.dtypes == 'float64').all()
        assert (actual_btc.factors.dtypes == 'float64').all()

        # test index
        if window_type == 'fixed':
            assert all(actual.factors.index == self.fm_instance.data.index)
            assert all(actual_btc.factors.index == self.fm_btc_instance.data.index)

        # test columns
        assert (actual.factors.columns == self.fm_instance.factors.columns).all()
        assert (actual_btc.factors.columns == self.fm_btc_instance.factors.columns).all()

    @pytest.mark.parametrize("multivariate", [True, False])
    def test_pooled_regression(self, multivariate) -> None:
        """
        Test pooled_regression method.
        """
        self.fm_instance.pooled_regression(multivariate=multivariate)
        self.fm_btc_instance.pooled_regression(multivariate=multivariate)

        # test type
        if multivariate:
            assert isinstance(self.fm_instance.results, statsmodels.iolib.summary.Summary)
            assert isinstance(self.fm_btc_instance.results, statsmodels.iolib.summary.Summary)
        else:
            assert isinstance(self.fm_instance.results, pd.DataFrame)
            assert isinstance(self.fm_btc_instance.results, pd.DataFrame)

        # test shape
        if not multivariate:
            assert self.fm_instance.results.shape[0] == self.fm_instance.factors.shape[1] - 1
            assert self.fm_btc_instance.results.shape[0] == self.fm_btc_instance.factors.shape[1] - 1

        # test index
        if not multivariate:
            assert set(self.fm_instance.results.index) == set(self.fm_instance.factors.columns[1:])
            assert set(self.fm_btc_instance.results.index) == set(self.fm_btc_instance.factors.columns[1:])

        # test columns
        if not multivariate:
            assert (self.fm_instance.results.columns == ['beta', 'std_error', 'p-val', 'f_p-val', 'R-squared']).all()
            assert (self.fm_btc_instance.results.columns ==
                    ['beta', 'std_error', 'p-val', 'f_p-val', 'R-squared']).all()

    @pytest.mark.parametrize("min_obs", [10, 20, 30])
    def test_check_min_obs(self, min_obs) -> None:
        """
        Test check_min_obs method.
        """
        self.fm_instance.check_min_obs(min_obs=min_obs)

        assert (self.fm_instance.factors.groupby('date').count().iloc[0] >= min_obs).all()

    def test_check_min_obs_param_errors(self) -> None:
        """
        Test check_min_obs method parameter errors.
        """
        with pytest.raises(Exception):
            self.fm_instance.check_min_obs(1000)

    @pytest.mark.parametrize("multivariate", [True, False])
    def test_fama_macbeth_regression(self, multivariate) -> None:
        """
        Test fama_macbeth_regression method.
        """
        self.fm_instance.fama_macbeth_regression(multivariate=multivariate)

        # test type
        assert isinstance(self.fm_instance.results, pd.DataFrame)
        # test shape
        assert self.fm_instance.results.shape[0] == self.fm_instance.factors.shape[1]
        # test index
        assert (self.fm_instance.results.index == self.fm_instance.factors.columns).all()
        # test columns
        assert (self.fm_instance.results.columns == ['beta', 'std_error', 't-stat']).all()
