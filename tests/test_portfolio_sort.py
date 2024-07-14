import pytest
import pandas as pd
import numpy as np

from factorlab.feature_engineering.transformations import Transform
from factorlab.feature_engineering.factors.trend import Trend
from factorlab.strategy_analysis.portfolio_sort import PortfolioSort


@pytest.fixture
def spot_prices():
    """
    Fixture for crypto spot prices.
    """
    # read csv from datasets/data
    df = pd.read_csv("../src/factorlab/datasets/data/binance_spot_prices.csv", index_col=['date', 'ticker'],
                     parse_dates=True).loc[:, : 'close']

    # drop tickers with nobs < ts_obs
    obs = df.groupby(level=1).count().min(axis=1)
    drop_tickers_list = obs[obs < 365].index.to_list()
    df = df.drop(drop_tickers_list, level=1, axis=0)

    # drop tickers with nobs < cs_obs
    obs = df.groupby(level=0).count().min(axis=1)
    idx_start = obs[obs > 3].index[0]
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
    btc_spot_ret = spot_ret.loc[:, 'BTC', :]

    return btc_spot_ret


@pytest.fixture
def price_mom(spot_prices):
    """
    Fixture for crypto price momentum.
    """
    # compute price mom
    price_mom_5 = Trend(spot_prices, lookback=5).price_mom()
    price_mom_20 = Trend(spot_prices, lookback=20).price_mom()
    price_mom_60 = Trend(spot_prices, lookback=60).price_mom()
    price_mom = pd.concat([price_mom_5, price_mom_20, price_mom_60], axis=1)

    return price_mom


@pytest.fixture
def btc_price_mom(price_mom):
    """
    Fixture for BTC price momentum.
    """
    # compute btc price mom
    btc_price_mom = price_mom.loc[:, 'BTC', :]

    return btc_price_mom


class TestPortfolioSort:
    """
    Test class for PortfolioSort.
    """

    @pytest.fixture(autouse=True)
    def portfolio_sort_single(self, price_mom, spot_ret):
        self.sort_single_instance = PortfolioSort(spot_ret.close, price_mom[['price_mom_5']],
                                                  factor_bins={'price_mom_5': ('ts', 3)})

    @pytest.fixture(autouse=True)
    def portfolio_sort_double(self, price_mom, spot_ret):
        self.sort_double_instance = PortfolioSort(spot_ret.close, price_mom[['price_mom_5', 'price_mom_20']],
                                                  factor_bins={'price_mom_5': ('ts', 3),
                                                               'price_mom_20': ('cs', 3)})

    @pytest.fixture(autouse=True)
    def portfolio_sort_tripple(self, price_mom, spot_ret):
        self.sort_tripple_instance = PortfolioSort(spot_ret.close, price_mom,
                                                   factor_bins={'price_mom_5': ('cs', 3),
                                                                'price_mom_20': ('cs', 3),
                                                                'price_mom_60': ('cs', 3)})

    @pytest.fixture(autouse=True)
    def portfolio_sort_conditional(self, price_mom, spot_ret):
        self.sort_conditional_instance = PortfolioSort(spot_ret.close, price_mom[['price_mom_5', 'price_mom_20']],
                                                       factor_bins={'price_mom_5': ('ts', 3),
                                                                    'price_mom_20': ('cs', 3)},
                                                       conditional=True)

    @pytest.fixture(autouse=True)
    def portfolio_sort_btc_single(self, btc_price_mom, btc_spot_ret):
        self.btc_sort_instance = PortfolioSort(btc_spot_ret.close, btc_price_mom[['price_mom_5']],
                                               factor_bins={'price_mom_5': ('ts', 3)})

    @pytest.fixture(autouse=True)
    def portfolio_sort_btc_double(self, btc_price_mom, btc_spot_ret):
        self.btc_sort_double_instance = PortfolioSort(btc_spot_ret.close,
                                                      btc_price_mom[['price_mom_5', 'price_mom_20']],
                                                      factor_bins={'price_mom_5': ('ts', 3),
                                                                   'price_mom_20': ('ts', 3)})

    @pytest.fixture(autouse=True)
    def portfolio_sort_btc_tripple(self, btc_price_mom, btc_spot_ret):
        self.btc_sort_tripple_instance = PortfolioSort(btc_spot_ret.close, btc_price_mom,
                                                       factor_bins={'price_mom_5': ('ts', 3),
                                                                    'price_mom_20': ('ts', 3),
                                                                    'price_mom_60': ('ts', 3)})

    @pytest.fixture(autouse=True)
    def portfolio_sort_btc_conditional(self, btc_price_mom, btc_spot_ret):
        self.btc_sort_conditional_instance = PortfolioSort(btc_spot_ret.close,
                                                           btc_price_mom[['price_mom_5', 'price_mom_20']],
                                                           factor_bins={'price_mom_5': ('ts', 3),
                                                                        'price_mom_20': ('ts', 3)},
                                                           conditional=True)

    def test_initialization(self) -> None:
        """
        Test initialization.
        """
        # data type
        assert isinstance(self.sort_single_instance, PortfolioSort)
        assert isinstance(self.btc_sort_instance, PortfolioSort)
        assert isinstance(self.sort_single_instance.factors, pd.DataFrame)
        assert isinstance(self.btc_sort_instance.factors, pd.DataFrame)
        assert isinstance(self.sort_single_instance.ret, pd.DataFrame)
        assert isinstance(self.btc_sort_instance.ret, pd.DataFrame)
        assert (self.sort_single_instance.factors.dtypes == np.float64).all()
        assert (self.btc_sort_instance.factors.dtypes == np.float64).all()
        assert (self.sort_single_instance.ret.dtypes == np.float64).all()
        assert (self.btc_sort_instance.ret.dtypes == np.float64).all()

    def test_check_factor_bins_value_errors(self, price_mom, btc_price_mom, spot_ret, btc_spot_ret) -> None:
        """
        Test check_factor_bins.
        """
        # check factor bins
        with pytest.raises(ValueError):
            PortfolioSort(spot_ret,  price_mom, factor_bins={'price_mom_5': ('ts', 3)}).check_factor_bins()
        with pytest.raises(ValueError):
            PortfolioSort(btc_spot_ret.close, btc_price_mom, factor_bins=[5, 6]).check_factor_bins()

    def test_preprocess_data(self) -> None:
        """
        Test preprocess_data.
        """
        # preprocess data
        self.sort_single_instance.preprocess_data()
        self.btc_sort_instance.preprocess_data()

        # shape
        assert self.sort_single_instance.factors.shape[0] == self.sort_single_instance.ret.shape[0]
        assert self.btc_sort_instance.factors.shape[0] == self.btc_sort_instance.ret.shape[0]
        assert self.sort_single_instance.factors.shape[1] == 1
        assert self.btc_sort_instance.factors.shape[1] == 1
        assert self.sort_single_instance.ret.shape[1] == 1
        assert self.btc_sort_instance.ret.shape[1] == 1
        # dtypes
        assert isinstance(self.sort_single_instance.factors, pd.DataFrame)
        assert isinstance(self.btc_sort_instance.factors, pd.DataFrame)
        assert isinstance(self.sort_single_instance.ret, pd.DataFrame)
        assert isinstance(self.btc_sort_instance.ret, pd.DataFrame)
        assert (self.sort_single_instance.factors.dtypes == np.float64).all()
        assert (self.btc_sort_instance.factors.dtypes == np.float64).all()
        assert (self.sort_single_instance.ret.dtypes == np.float64).all()
        assert (self.btc_sort_instance.ret.dtypes == np.float64).all()
        # index
        assert isinstance(self.sort_single_instance.factors.index, pd.MultiIndex)
        assert isinstance(self.sort_single_instance.ret.index, pd.MultiIndex)
        assert isinstance(self.sort_single_instance.factors.index.get_level_values(0), pd.DatetimeIndex)
        assert isinstance(self.sort_single_instance.ret.index.get_level_values(0), pd.DatetimeIndex)
        assert isinstance(self.btc_sort_instance.factors.index, pd.DatetimeIndex)
        assert isinstance(self.btc_sort_instance.ret.index, pd.DatetimeIndex)
        # freq
        assert self.sort_single_instance.freq == 'D'
        assert self.btc_sort_instance.freq == 'D'

    @pytest.mark.parametrize("factor, strategy, bins", [('price_mom_5', 'ts', 3), ('price_mom_5', 'cs', 3)])
    def test_quantize_factor(self, factor, strategy, bins):
        """
        Test quantize_factor.
        """
        # get actual
        actual = self.sort_single_instance.quantize_factor(self.sort_single_instance.factors[factor], strategy, bins)

        # shape
        assert actual.shape[0] == self.sort_single_instance.factors.shape[0]
        assert actual.shape[1] == 1
        # dtypes
        assert isinstance(actual, pd.DataFrame)
        assert (actual.dtypes == np.float64).all()
        # values
        assert (actual.nunique() == bins).all()
        assert ((actual.dropna() >= 1.0) & (actual.dropna() <= bins)).all().all()
        # index
        assert isinstance(actual.index, pd.MultiIndex)
        # cols
        assert actual.columns == [factor]

        if strategy == 'ts':
            actual_btc = self.btc_sort_instance.quantize_factor(self.btc_sort_instance.factors[factor], strategy, bins)
            assert actual_btc.shape[0] == self.btc_sort_instance.factors.shape[0]
            assert actual_btc.shape[1] == 1
            assert (actual_btc.nunique() == bins).all()
            assert ((actual_btc.dropna() >= 1.0) & (actual_btc.dropna() <= bins)).all().all()
            assert isinstance(actual_btc, pd.DataFrame)
            assert (self.btc_sort_instance.factors[factor].dtypes == np.float64)
            assert isinstance(self.btc_sort_instance.factors.index, pd.DatetimeIndex)
            assert isinstance(self.btc_sort_instance.ret.index, pd.DatetimeIndex)
            assert actual_btc.columns == [factor]

    def test_quantize_factor_value_errors(self):
        """
        Test quantize_factor.
        """
        # value errors
        with pytest.raises(ValueError):
            self.btc_sort_instance.quantize_factor(self.btc_sort_instance.factors['price_mom_5'], 'cs', 1)

    def test_unconditional_quantization(self):
        """
        Test unconditional_quantization.
        """
        # unconditional quantization
        self.sort_double_instance.unconditional_quantization()
        self.btc_sort_double_instance.unconditional_quantization()

        # shape
        assert self.sort_double_instance.factor_quantiles.shape[0] == self.sort_double_instance.factors.shape[0]
        assert self.sort_double_instance.factor_quantiles.shape[1] == self.sort_double_instance.factors.shape[1]
        assert self.btc_sort_double_instance.factor_quantiles.shape[0] == self.btc_sort_double_instance.factors.shape[0]
        assert self.btc_sort_double_instance.factor_quantiles.shape[1] == self.btc_sort_double_instance.factors.shape[1]
        # dtypes
        assert isinstance(self.sort_double_instance.factors, pd.DataFrame)
        assert isinstance(self.btc_sort_double_instance.factors, pd.DataFrame)
        assert (self.sort_double_instance.factors.dtypes == np.float64).all()
        assert (self.btc_sort_instance.factors.dtypes == np.float64).all()
        # values
        assert (self.sort_double_instance.factor_quantiles.nunique() == 3).all()
        assert ((self.sort_double_instance.factor_quantiles.dropna() >= 1.0) &
                (self.sort_double_instance.factor_quantiles.dropna() <= 3)).all().all()
        # index
        assert (self.sort_double_instance.factor_quantiles.index == self.sort_double_instance.factors.index).all()
        assert (self.btc_sort_double_instance.factor_quantiles.index ==
                self.btc_sort_double_instance.factors.index).all()
        # cols
        assert (self.sort_double_instance.factor_quantiles.columns == self.sort_double_instance.factors.columns).all()
        assert (self.btc_sort_double_instance.factor_quantiles.columns ==
                self.btc_sort_double_instance.factors.columns).all()

    def test_conditional_quantization(self):
        """
        Test conditional_quantization.
        """
        # conditional quantization
        self.sort_conditional_instance.conditional_quantization()
        self.btc_sort_conditional_instance.conditional_quantization()

        # shape
        assert self.sort_conditional_instance.factor_quantiles.shape[0] == \
               self.sort_conditional_instance.factors.shape[0]
        assert self.sort_conditional_instance.factor_quantiles.shape[1] == \
               self.sort_conditional_instance.factors.shape[1]
        assert self.btc_sort_conditional_instance.factor_quantiles.shape[0] == \
               self.btc_sort_conditional_instance.factors.shape[0]
        assert self.btc_sort_conditional_instance.factor_quantiles.shape[1] == \
               self.btc_sort_conditional_instance.factors.shape[1]
        # dtypes
        assert isinstance(self.sort_conditional_instance.factors, pd.DataFrame)
        assert isinstance(self.btc_sort_conditional_instance.factors, pd.DataFrame)
        assert (self.sort_conditional_instance.factors.dtypes == np.float64).all()
        assert (self.btc_sort_conditional_instance.factors.dtypes == np.float64).all()
        # values
        assert (self.sort_conditional_instance.factor_quantiles.nunique() == 3).all()
        assert ((self.sort_conditional_instance.factor_quantiles.dropna() >= 1.0) &
                (self.sort_conditional_instance.factor_quantiles.dropna() <= 3)).all().all()
        assert (self.btc_sort_conditional_instance.factor_quantiles.nunique() == 3).all()
        assert ((self.btc_sort_conditional_instance.factor_quantiles.dropna() >= 1.0) &
                (self.btc_sort_conditional_instance.factor_quantiles.dropna() <= 3)).all().all()
        # index
        assert (self.sort_conditional_instance.factor_quantiles.index ==
                self.sort_conditional_instance.factors.index).all()
        assert (self.btc_sort_conditional_instance.factor_quantiles.index ==
                self.btc_sort_conditional_instance.factors.index).all()
        # cols
        assert (self.sort_conditional_instance.factor_quantiles.columns ==
                self.sort_conditional_instance.factors.columns).all()
        assert (self.btc_sort_conditional_instance.factor_quantiles.columns ==
                self.btc_sort_conditional_instance.factors.columns).all()

    def test_quantize_factors(self):
        """
        Test quantize_factors.
        """
        # quantize factors
        cond = self.sort_conditional_instance.quantize_factors()
        cond_btc = self.btc_sort_conditional_instance.quantize_factors()
        uncond = self.sort_double_instance.quantize_factors()
        uncond_btc = self.btc_sort_double_instance.quantize_factors()

        # values
        assert np.allclose(cond.iloc[:, 0], uncond.iloc[:, 0], equal_nan=True)
        assert np.allclose(cond_btc.iloc[:, 0], uncond_btc.iloc[:, 0], equal_nan=True)
        assert np.allclose(cond.iloc[:, -1], uncond.iloc[:, -1], equal_nan=True) is False
        assert np.allclose(cond_btc.iloc[:, -1], uncond_btc.iloc[:, -1], equal_nan=True) is False

    def test_join_quantile_rets(self):
        """
        Test join_quantile_rets.
        """
        # join quantile rets
        actual = self.sort_double_instance.join_quantile_rets()
        actual_btc = self.btc_sort_double_instance.join_quantile_rets()

        # shape
        assert actual.shape[0] == self.sort_double_instance.ret.shape[0]
        assert actual.shape[1] == self.sort_double_instance.factors.shape[1] + self.sort_double_instance.ret.shape[1]
        assert actual_btc.shape[0] == self.btc_sort_double_instance.ret.shape[0]
        assert actual_btc.shape[1] == self.btc_sort_double_instance.factors.shape[1] + \
               self.btc_sort_double_instance.ret.shape[1]
        # dtypes
        assert isinstance(actual, pd.DataFrame)
        assert isinstance(actual_btc, pd.DataFrame)
        assert (actual.dtypes == np.float64).all()
        assert (actual_btc.dtypes == np.float64).all()
        # index
        assert (actual.index == self.sort_double_instance.ret.index).all()
        assert (actual_btc.index == self.btc_sort_double_instance.ret.index).all()
        # cols
        assert (actual.columns == self.sort_double_instance.factors.columns.tolist() +
                self.sort_double_instance.ret.columns.tolist()).all()
        assert (actual_btc.columns == self.btc_sort_double_instance.factors.columns.tolist() +
                self.btc_sort_double_instance.ret.columns.tolist()).all()

    def test_sort(self):
        """
        Test sort.
        """
        # sort
        self.sort_double_instance.sort()
        self.btc_sort_double_instance.sort()

        # shape
        assert self.sort_double_instance.quantile_rets.shape[0] == self.sort_double_instance.factors.shape[0] - 1
        assert self.sort_double_instance.quantile_rets.shape[1] == \
               self.sort_double_instance.factor_bins['price_mom_5'][1] + \
               self.sort_double_instance.factor_bins['price_mom_20'][1]
        assert self.btc_sort_double_instance.quantile_rets.shape[0] == \
               self.btc_sort_double_instance.factors.shape[0] - 2
        assert self.btc_sort_double_instance.quantile_rets.shape[1] == \
               self.btc_sort_double_instance.factor_bins['price_mom_5'][1] + \
               self.btc_sort_double_instance.factor_bins['price_mom_20'][1]
        # dtypes
        assert isinstance(self.sort_double_instance.quantile_rets, pd.DataFrame)
        assert (self.sort_double_instance.quantile_rets.dtypes == np.float64).all()
        assert isinstance(self.btc_sort_double_instance.quantile_rets, pd.DataFrame)
        assert (self.btc_sort_double_instance.quantile_rets.dtypes == np.float64).all()
        # index
        assert set(self.sort_double_instance.index.droplevel(0).unique()) == \
               set(self.sort_double_instance.quantile_rets.index.droplevel(0).unique())
        # cols
        cols = pd.MultiIndex.from_product([['price_mom_5', 'price_mom_20'], ['1', '2', '3']])
        assert (self.sort_double_instance.quantile_rets.columns == cols).all()
        assert (self.btc_sort_double_instance.quantile_rets.columns == cols).all()

    def test_compute_quantile_portfolios(self):
        """
        Test compute_quantile_portfolios.
        """
        # compute quantile portfolios
        self.sort_double_instance.compute_quantile_portfolios()
        self.btc_sort_double_instance.compute_quantile_portfolios()

        # shape
        assert self.sort_double_instance.portfolio_rets.shape[1] == self.sort_double_instance.factors.shape[1]
        assert self.btc_sort_double_instance.portfolio_rets.shape[1] == self.btc_sort_double_instance.factors.shape[1]
        # dtypes
        assert isinstance(self.sort_double_instance.portfolio_rets, pd.DataFrame)
        assert (self.sort_double_instance.portfolio_rets.dtypes == np.float64).all()
        assert isinstance(self.btc_sort_double_instance.portfolio_rets, pd.DataFrame)
        assert (self.btc_sort_double_instance.portfolio_rets.dtypes == np.float64).all()
        # index
        assert self.sort_double_instance.portfolio_rets.index.names == ['date', 'quantile']
        assert self.btc_sort_double_instance.portfolio_rets.index.names == ['date', 'quantile']
        # cols
        cols = pd.MultiIndex.from_product([['price_mom_5', 'price_mom_20'], ['1', '2', '3']])
        assert (self.sort_double_instance.quantile_rets.columns == cols).all()
        assert (self.btc_sort_double_instance.quantile_rets.columns == cols).all()

    def test_performance(self):
        """
        Test performance.
        """
        # compute quantile portfolios
        perf_df = self.sort_double_instance.performance()
        perf_btc_df = self.btc_sort_double_instance.performance()

        # shape
        assert perf_df.shape == (3, 3)
        assert perf_btc_df.shape == (3, 3)
        # dtypes
        assert isinstance(perf_df, pd.DataFrame)
        assert (perf_df.dtypes == np.float64).all()
        assert isinstance(perf_btc_df, pd.DataFrame)
        assert (perf_btc_df.dtypes == np.float64).all()
        # index
        idx = pd.MultiIndex.from_product([['price_mom_5'], ['1', '2', '3']])
        assert (perf_df.index == idx).all()
        assert (perf_btc_df.index == idx).all()
        # cols
        cols = pd.MultiIndex.from_product([['price_mom_20'], ['1', '2', '3']])
        assert (perf_df.columns == cols).all()
        assert (perf_btc_df.columns == cols).all()
