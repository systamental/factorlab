import pytest
import pandas as pd
import numpy as np

from factorlab.strategy_analysis.feature_selection import FeatureSelection
from factorlab.feature_engineering.transformations import Transform
from factorlab.feature_engineering.factors.trend import Trend


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
    price_mom = Trend(spot_prices, lookback=5).price_mom()
    price_mom['price_mom_10'] = Trend(spot_prices, lookback=10).price_mom()
    price_mom['price_mom_15'] = Trend(spot_prices, lookback=15).price_mom()
    price_mom['price_mom_30'] = Trend(spot_prices, lookback=30).price_mom()
    price_mom['price_mom_45'] = Trend(spot_prices, lookback=45).price_mom()

    return price_mom


@pytest.fixture
def btc_price_mom(price_mom):
    """
    Fixture for BTC price momentum.
    """
    # compute btc price mom
    btc_price_mom = price_mom.loc[:, 'BTC', :]

    return btc_price_mom


class TestFeatureSelection:
    """
    Test Filter class.
    """
    @pytest.fixture(autouse=True)
    def fs_setup(self, spot_ret, price_mom):
        self.fs_instance = FeatureSelection(spot_ret.close, price_mom)

    @pytest.fixture(autouse=True)
    def fs_norm_setup(self, spot_ret, price_mom):
        self.fs_norm_instance = FeatureSelection(spot_ret.close, price_mom, normalize=True, quantize=False)

    @pytest.fixture(autouse=True)
    def fs_quantiles_setup(self, spot_ret, price_mom):
        self.fs_quantiles_instance = FeatureSelection(spot_ret.close, price_mom, normalize=True, quantize=True)

    @pytest.fixture(autouse=True)
    def fs_setup_btc(self, btc_price_mom, btc_spot_ret):
        self.fs_btc_instance = FeatureSelection(btc_spot_ret.close, btc_price_mom)

    @pytest.fixture(autouse=True)
    def fs_norm_setup_btc(self, btc_price_mom, btc_spot_ret):
        self.fs_btc_norm_instance = FeatureSelection(btc_spot_ret.close, btc_price_mom, normalize=True, quantize=False)

    @pytest.fixture(autouse=True)
    def fs_quantiles_setup_btc(self, btc_spot_ret, btc_price_mom):
        self.fs_btc_quantiles_instance = FeatureSelection(btc_spot_ret.close, btc_price_mom, normalize=True,
                                                    quantize=True)

    def test_initialization(self) -> None:
        """
        Test initialization.
        """
        assert isinstance(self.fs_instance, FeatureSelection)
        assert isinstance(self.fs_btc_instance, FeatureSelection)
        assert isinstance(self.fs_instance.data, pd.DataFrame)
        assert isinstance(self.fs_btc_instance.data, pd.DataFrame)
        assert isinstance(self.fs_instance.target, pd.DataFrame)
        assert isinstance(self.fs_btc_instance.target, pd.DataFrame)
        assert isinstance(self.fs_instance.features, pd.DataFrame)
        assert isinstance(self.fs_btc_instance.features, pd.DataFrame)
        assert all(self.fs_instance.index == self.fs_instance.data.index)
        assert all(self.fs_btc_instance.index == self.fs_btc_instance.data.index)
        assert self.fs_instance.freq == 'D'
        assert self.fs_btc_instance.freq == 'D'

    def test_check_n_feat(self) -> None:
        """
        Test check_n_feat method.
        """
        # test value
        assert self.fs_instance.n_feat == self.fs_instance.features.shape[1]
        assert self.fs_btc_instance.n_feat == self.fs_btc_instance.features.shape[1]

    def test_preprocess_data(self) -> None:
        """
        Test preprocess_data method.
        """
        # test type
        assert (self.fs_instance.target.dtypes == 'float64').all()
        assert (self.fs_btc_instance.target.dtypes == 'float64').all()
        assert (self.fs_instance.features.dtypes == 'float64').all()
        assert (self.fs_btc_instance.features.dtypes == 'float64').all()

        # test values
        assert self.fs_instance.target.isin(self.fs_instance.data).all().all()
        assert self.fs_btc_instance.target.isin(self.fs_btc_instance.data).all().all()
        assert self.fs_instance.features.isin(self.fs_instance.data).all().all()
        assert self.fs_btc_instance.features.isin(self.fs_btc_instance.data).all().all()
        assert self.fs_instance.target.isna().sum().sum() == 0
        assert self.fs_btc_instance.target.isna().sum().sum() == 0

        # test shape
        assert self.fs_instance.target.shape[0] == self.fs_instance.data.shape[0]
        assert self.fs_btc_instance.target.shape[0] == self.fs_btc_instance.data.shape[0]
        assert self.fs_instance.features.shape[0] == self.fs_instance.data.shape[0]
        assert self.fs_btc_instance.features.shape[0] == self.fs_btc_instance.data.shape[0]

        # test index
        assert all(self.fs_instance.target.index == self.fs_instance.data.index)
        assert all(self.fs_btc_instance.target.index == self.fs_btc_instance.data.index)
        assert all(self.fs_instance.features.index == self.fs_instance.data.index)
        assert all(self.fs_btc_instance.features.index == self.fs_btc_instance.data.index)

        # test columns
        assert all(self.fs_instance.target.columns == self.fs_instance.data.columns[-1])
        assert all(self.fs_btc_instance.target.columns == self.fs_btc_instance.data.columns[-1])
        assert all(self.fs_instance.features.columns == self.fs_instance.data.columns[:-1])
        assert all(self.fs_btc_instance.features.columns == self.fs_btc_instance.data.columns[:-1])

    def test_normalize_data(self) -> None:
        """
        Test normalize_data method.
        """
        # test type
        assert isinstance(self.fs_norm_instance.features, pd.DataFrame)
        assert isinstance(self.fs_btc_norm_instance.features, pd.DataFrame)
        assert isinstance(self.fs_btc_norm_instance.target, pd.DataFrame)
        assert isinstance(self.fs_btc_norm_instance.target, pd.DataFrame)
        assert (self.fs_norm_instance.features.dtypes == 'float64').all()
        assert (self.fs_btc_norm_instance.features.dtypes == 'float64').all()
        assert (self.fs_norm_instance.target.dtypes == 'float64').all()
        assert (self.fs_btc_norm_instance.target.dtypes == 'float64').all()

        # test values
        assert self.fs_norm_instance.features.describe().loc['mean'].mean() < 0.0001
        assert self.fs_btc_norm_instance.features.describe().loc['mean'].mean() < 0.0001
        assert self.fs_norm_instance.target.describe().loc['mean'].mean() < 0.0001
        assert self.fs_btc_norm_instance.target.describe().loc['mean'].mean() < 0.0001
        assert ((self.fs_norm_instance.features.describe().loc['std'] - 1).abs() < 0.0015).all()
        assert ((self.fs_btc_norm_instance.features.describe().loc['std'] - 1).abs() < 0.0015).all()
        assert ((self.fs_norm_instance.target.describe().loc['std'] - 1).abs() < 0.0015).all()
        assert ((self.fs_btc_norm_instance.target.describe().loc['std'] - 1).abs() < 0.0015).all()

        # test shape
        assert self.fs_norm_instance.features.shape[0] == self.fs_norm_instance.data.shape[0]
        assert self.fs_btc_norm_instance.features.shape[0] == self.fs_btc_norm_instance.data.shape[0]
        assert self.fs_norm_instance.target.shape[0] == self.fs_norm_instance.data.shape[0]
        assert self.fs_btc_norm_instance.target.shape[0] == self.fs_btc_norm_instance.data.shape[0]

        # test index
        assert all(self.fs_norm_instance.features.index == self.fs_norm_instance.data.index)
        assert all(self.fs_btc_norm_instance.features.index == self.fs_btc_norm_instance.data.index)
        assert all(self.fs_norm_instance.target.index == self.fs_norm_instance.data.index)
        assert all(self.fs_btc_norm_instance.target.index == self.fs_btc_norm_instance.data.index)

        # test columns
        assert all(self.fs_norm_instance.features.columns == self.fs_norm_instance.data.columns[:-1])
        assert all(self.fs_btc_norm_instance.features.columns == self.fs_btc_norm_instance.data.columns[:-1])
        assert all(self.fs_norm_instance.target.columns == self.fs_norm_instance.data.columns[-1])
        assert all(self.fs_btc_norm_instance.target.columns == self.fs_btc_norm_instance.data.columns[-1])

    def test_quantize_data(self) -> None:
        """
        Test quantize method.
        """
        # test type
        assert isinstance(self.fs_quantiles_instance.features, pd.DataFrame)
        assert isinstance(self.fs_btc_quantiles_instance.features, pd.DataFrame)
        assert isinstance(self.fs_quantiles_instance.target, pd.DataFrame)
        assert isinstance(self.fs_btc_quantiles_instance.target, pd.DataFrame)
        assert (self.fs_quantiles_instance.features.dtypes == 'float64').all()
        assert (self.fs_btc_quantiles_instance.features.dtypes == 'float64').all()
        assert (self.fs_quantiles_instance.target.dtypes == 'float64').all()
        assert (self.fs_btc_quantiles_instance.target.dtypes == 'float64').all()

        # test values
        assert (self.fs_quantiles_instance.features.nunique() == self.fs_quantiles_instance.feature_bins).all()
        assert (self.fs_btc_quantiles_instance.features.nunique() ==
                self.fs_btc_quantiles_instance.feature_bins).all()
        assert (self.fs_quantiles_instance.target.nunique() == self.fs_quantiles_instance.target_bins).all()
        assert (self.fs_btc_quantiles_instance.target.nunique() ==
                self.fs_btc_quantiles_instance.target_bins).all().all()
        assert (np.sort(self.fs_quantiles_instance.features.iloc[:, 0].unique()) ==
                np.arange(1, self.fs_quantiles_instance.feature_bins + 1).astype(float)).all()
        assert (np.sort(self.fs_btc_quantiles_instance.features.iloc[:, 0].unique()) ==
                np.arange(1, self.fs_btc_quantiles_instance.feature_bins + 1).astype(float)).all()
        assert (np.sort(self.fs_quantiles_instance.target.iloc[:, 0].unique()) ==
                np.arange(1, self.fs_quantiles_instance.target_bins + 1).astype(float)).all()
        assert (np.sort(self.fs_btc_quantiles_instance.target.iloc[:, 0].unique()) ==
                np.arange(1, self.fs_btc_quantiles_instance.target_bins + 1).astype(float)).all()

        # test shape
        assert self.fs_quantiles_instance.features.shape[0] == self.fs_quantiles_instance.data.shape[0]
        assert self.fs_btc_quantiles_instance.features.shape[0] == self.fs_btc_quantiles_instance.data.shape[0]
        assert self.fs_quantiles_instance.target.shape[0] == self.fs_quantiles_instance.data.shape[0]
        assert self.fs_btc_quantiles_instance.target.shape[0] == self.fs_btc_quantiles_instance.data.shape[0]

        # test index
        assert all(self.fs_quantiles_instance.features.index == self.fs_quantiles_instance.data.index)
        assert all(self.fs_btc_quantiles_instance.features.index == self.fs_btc_quantiles_instance.data.index)
        assert all(self.fs_quantiles_instance.target.index == self.fs_quantiles_instance.data.index)
        assert all(self.fs_btc_quantiles_instance.target.index == self.fs_btc_quantiles_instance.data.index)

        # test columns
        assert all(self.fs_quantiles_instance.features.columns == self.fs_quantiles_instance.data.columns[:-1])
        assert all(self.fs_btc_quantiles_instance.features.columns ==
                   self.fs_btc_quantiles_instance.data.columns[:-1])
        assert all(self.fs_quantiles_instance.target.columns == self.fs_quantiles_instance.data.columns[-1])
        assert all(self.fs_btc_quantiles_instance.target.columns ==
                   self.fs_btc_quantiles_instance.data.columns[-1])

    def test_spearman_rank(self) -> None:
        """
        Test spearman_rank method
        """
        actual = self.fs_instance.spearman_rank()
        actual_btc = self.fs_btc_instance.spearman_rank()

        # test type
        assert isinstance(actual, pd.DataFrame)
        assert isinstance(actual_btc, pd.DataFrame)
        assert (actual.dtypes == 'float64').all()
        assert (actual_btc.dtypes == 'float64').all()

        # test values
        assert ((actual.abs() <= 1) & (actual.abs() >= 0)).all().all()
        assert ((actual_btc.abs() <= 1) & (actual_btc.abs() >= 0)).all().all()

        # test shape
        assert actual.shape[0] == self.fs_instance.features.shape[1]
        assert actual_btc.shape[0] == self.fs_btc_quantiles_instance.features.shape[1]
        assert actual.shape[1] == 2
        assert actual_btc.shape[1] == 2

        # test index
        assert set(actual.index) == set(self.fs_instance.features.columns)
        assert set(actual_btc.index) == set(self.fs_btc_instance.features.columns)

        # test columns
        assert all(actual.columns == ['spearman_rank', 'p-val'])
        assert all(actual_btc.columns == ['spearman_rank', 'p-val'])

    def test_kendall_tau(self) -> None:
        """
        Test kendall_tau method
        """
        actual = self.fs_instance.kendall_tau()
        actual_btc = self.fs_btc_instance.kendall_tau()

        # test type
        assert isinstance(actual, pd.DataFrame)
        assert isinstance(actual_btc, pd.DataFrame)
        assert (actual.dtypes == 'float64').all()
        assert (actual_btc.dtypes == 'float64').all()

        # test values
        assert ((actual.abs() <= 1) & (actual.abs() >= 0)).all().all()
        assert ((actual_btc.abs() <= 1) & (actual_btc.abs() >= 0)).all().all()

        # test shape
        assert actual.shape[0] == self.fs_instance.features.shape[1]
        assert actual_btc.shape[0] == self.fs_btc_instance.features.shape[1]
        assert actual.shape[1] == 2
        assert actual_btc.shape[1] == 2

        # test index
        assert set(actual.index) == set(self.fs_instance.features.columns)
        assert set(actual_btc.index) == set(self.fs_btc_instance.features.columns)

        # test columns
        assert all(actual.columns == ['kendall_tau', 'p-val'])
        assert all(actual_btc.columns == ['kendall_tau', 'p-val'])

    def test_cramer_v(self) -> None:
        """
        Test cramer_v method
        """
        actual = self.fs_quantiles_instance.cramer_v()
        actual_btc = self.fs_btc_quantiles_instance.cramer_v()

        # test type
        assert isinstance(actual, pd.DataFrame)
        assert isinstance(actual_btc, pd.DataFrame)
        assert (actual.dtypes == 'float64').all()
        assert (actual_btc.dtypes == 'float64').all()

        # test values
        assert ((actual.abs() <= 1) & (actual.abs() >= 0)).all().all()
        assert ((actual_btc.abs() <= 1) & (actual_btc.abs() >= 0)).all().all()

        # test shape
        assert actual.shape[0] == self.fs_quantiles_instance.features.shape[1]
        assert actual_btc.shape[0] == self.fs_btc_quantiles_instance.features.shape[1]
        assert actual.shape[1] == 1
        assert actual_btc.shape[1] == 1

        # test index
        assert set(actual.index) == set(self.fs_quantiles_instance.features.columns)
        assert set(actual_btc.index) == set(self.fs_btc_quantiles_instance.features.columns)

        # test columns
        assert all(actual.columns == ['cramer_v'])
        assert all(actual_btc.columns == ['cramer_v'])

    def test_tschuprow(self) -> None:
        """
        Test tschuprow method
        """
        actual = self.fs_quantiles_instance.tschuprow()
        actual_btc = self.fs_btc_quantiles_instance.tschuprow()

        # test type
        assert isinstance(actual, pd.DataFrame)
        assert isinstance(actual_btc, pd.DataFrame)
        assert (actual.dtypes == 'float64').all()
        assert (actual_btc.dtypes == 'float64').all()

        # test values
        assert ((actual.abs() <= 1) & (actual.abs() >= 0)).all().all()
        assert ((actual_btc.abs() <= 1) & (actual_btc.abs() >= 0)).all().all()

        # test shape
        assert actual.shape[0] == self.fs_quantiles_instance.features.shape[1]
        assert actual_btc.shape[0] == self.fs_btc_quantiles_instance.features.shape[1]
        assert actual.shape[1] == 1
        assert actual_btc.shape[1] == 1

        # test index
        assert set(actual.index) == set(self.fs_quantiles_instance.features.columns)
        assert set(actual_btc.index) == set(self.fs_btc_quantiles_instance.features.columns)

        # test columns
        assert all(actual.columns == ['tschuprow'])
        assert all(actual_btc.columns == ['tschuprow'])

    def test_pearson_cc(self) -> None:
        """
        Test pearson_cc method
        """
        actual = self.fs_quantiles_instance.pearson_cc()
        actual_btc = self.fs_btc_quantiles_instance.pearson_cc()

        # test type
        assert isinstance(actual, pd.DataFrame)
        assert isinstance(actual_btc, pd.DataFrame)
        assert (actual.dtypes == 'float64').all()
        assert (actual_btc.dtypes == 'float64').all()

        # test values
        assert ((actual.abs() <= 1) & (actual.abs() >= 0)).all().all()
        assert ((actual_btc.abs() <= 1) & (actual_btc.abs() >= 0)).all().all()

        # test shape
        assert actual.shape[0] == self.fs_quantiles_instance.features.shape[1]
        assert actual_btc.shape[0] == self.fs_btc_quantiles_instance.features.shape[1]
        assert actual.shape[1] == 1
        assert actual_btc.shape[1] == 1

        # test index
        assert set(actual.index) == set(self.fs_quantiles_instance.features.columns)
        assert set(actual_btc.index) == set(self.fs_btc_quantiles_instance.features.columns)

        # test columns
        assert all(actual.columns == ['pearson_cc'])
        assert all(actual_btc.columns == ['pearson_cc'])

    def test_chi2(self) -> None:
        """
        Test chi2 method
        """
        actual = self.fs_quantiles_instance.chi2()
        actual_btc = self.fs_btc_quantiles_instance.chi2()

        # test type
        assert isinstance(actual, pd.DataFrame)
        assert isinstance(actual_btc, pd.DataFrame)
        assert (actual.dtypes == 'float64').all()
        assert (actual_btc.dtypes == 'float64').all()

        # test values
        assert (actual.abs() >= 0).all().all()
        assert (actual_btc.abs() >= 0).all().all()

        # test shape
        assert actual.shape[0] == self.fs_quantiles_instance.features.shape[1]
        assert actual_btc.shape[0] == self.fs_btc_quantiles_instance.features.shape[1]
        assert actual.shape[1] == 1
        assert actual_btc.shape[1] == 1

        # test index
        assert set(actual.index) == set(self.fs_quantiles_instance.features.columns)
        assert set(actual_btc.index) == set(self.fs_btc_quantiles_instance.features.columns)

        # test columns
        assert all(actual.columns == ['chi2'])
        assert all(actual_btc.columns == ['chi2'])

    def test_mutual_info(self) -> None:
        """
        Test mutual information method.
        """
        actual = self.fs_quantiles_instance.mutual_info()
        actual_btc = self.fs_btc_quantiles_instance.mutual_info()

        # test type
        assert isinstance(actual, pd.DataFrame)
        assert isinstance(actual_btc, pd.DataFrame)
        assert (actual.dtypes == 'float64').all()
        assert (actual_btc.dtypes == 'float64').all()

        # test values
        assert (actual.abs() >= 0).all().all()
        assert (actual_btc.abs() >= 0).all().all()

        # test shape
        assert actual.shape[0] == self.fs_quantiles_instance.features.shape[1]
        assert actual_btc.shape[0] == self.fs_btc_quantiles_instance.features.shape[1]
        assert actual.shape[1] == 1
        assert actual_btc.shape[1] == 1

        # test index
        assert set(actual.index) == set(self.fs_quantiles_instance.features.columns)
        assert set(actual_btc.index) == set(self.fs_btc_quantiles_instance.features.columns)

        # test columns
        assert all(actual.columns == ['mutual_info'])
        assert all(actual_btc.columns == ['mutual_info'])

    @pytest.mark.parametrize("method", ['spearman_rank', 'kendall_tau', 'cramer_v', 'tschuprow', 'pearson_cc',
                                       'chi2', 'mutual_info'])
    def test_filter(self, method) -> None:
        """
        Test filter methods.
        """
        if method in ['spearman_rank', 'kendall_tau']:
            actual = self.fs_instance.filter(method=method)
            actual_btc = self.fs_btc_instance.filter(method=method)
        else:
            actual = self.fs_instance.filter(method=method)
            actual_btc = self.fs_btc_instance.filter(method=method)

        # test type
        assert isinstance(actual, pd.DataFrame)
        assert isinstance(actual_btc, pd.DataFrame)
        assert (actual.dtypes == 'float64').all()
        assert (actual_btc.dtypes == 'float64').all()

        # test values
        assert (actual.abs() >= 0).all().all()
        assert (actual_btc.abs() >= 0).all().all()

        # test shape
        assert actual.shape[0] == self.fs_instance.features.shape[1]
        assert actual_btc.shape[0] == self.fs_btc_instance.features.shape[1]
        if method in ['spearman_rank', 'kendall_tau']:
            assert actual.shape[1] == 2
            assert actual_btc.shape[1] == 2
        else:
            assert actual.shape[1] == 1
            assert actual_btc.shape[1] == 1

        # test index
        assert set(actual.index) == set(self.fs_instance.features.columns)
        assert set(actual_btc.index) == set(self.fs_btc_instance.features.columns)

        # test columns
        if method in ['spearman_rank', 'kendall_tau']:
            assert all(actual.columns == [method, 'p-val'])
            assert all(actual_btc.columns == [method, 'p-val'])
        else:
            assert all(actual.columns == [method])
            assert all(actual_btc.columns == [method])

    def test_filter_param_error(self) -> None:
        """
        Test filter method parameter error.
        """
        with pytest.raises(ValueError):
            self.fs_instance.filter(method='error')
        with pytest.raises(ValueError):
            self.fs_btc_instance.filter(method='error')

    @pytest.mark.parametrize("feature", ['price_mom_10', 'price_mom_15', 'price_mom_30'])
    def test_ic(self, feature) -> None:
        """
        Test ic method.
        """
        actual = self.fs_instance.ic(feature=feature)
        actual_btc = self.fs_btc_instance.ic(feature=feature)

        # test type
        assert isinstance(actual, pd.DataFrame)
        assert isinstance(actual_btc, pd.DataFrame)
        assert (actual.dtypes == 'float64').all()
        assert (actual_btc.dtypes == 'float64').all()

        # test values
        assert ((actual.dropna().abs() <= 1) & (actual.dropna().abs() >= 0)).all().all()
        assert ((actual_btc.dropna().abs() <= 1) & (actual_btc.dropna().abs() >= 0)).all().all()

        # test shape
        assert actual.shape[1] == 1
        assert actual_btc.shape[1] == 1

        # test columns
        assert all(actual.columns == [feature])
        assert all(actual_btc.columns == [feature])

    def test_lars(self) -> None:
        """
        Test LARS feature selection method.
        """
        # select features
        self.fs_instance.lars()
        self.fs_btc_instance.lars()

        # test type
        assert isinstance(self.fs_instance.feature_importance, pd.DataFrame)
        assert isinstance(self.fs_btc_instance.feature_importance, pd.DataFrame)
        assert isinstance(self.fs_instance.ranked_features, pd.DataFrame)
        assert isinstance(self.fs_btc_instance.ranked_features, pd.DataFrame)
        assert isinstance(self.fs_instance.ranked_features_list, list)
        assert isinstance(self.fs_btc_instance.ranked_features_list, list)
        assert (self.fs_instance.feature_importance.dtypes == 'float64').all()
        assert (self.fs_btc_instance.feature_importance.dtypes == 'float64').all()

        # test values
        assert self.fs_instance.ranked_features.isin(self.fs_instance.features).all().all()
        assert self.fs_btc_instance.ranked_features.isin(self.fs_btc_instance.features).all().all()
        assert set(self.fs_instance.ranked_features_list) == set(self.fs_instance.features.columns)
        assert set(self.fs_btc_instance.ranked_features_list) == set(self.fs_btc_instance.features.columns)

        # test shape
        assert self.fs_instance.feature_importance.shape[0] <= self.fs_instance.n_feat
        assert self.fs_btc_instance.feature_importance.shape[0] <= self.fs_btc_instance.n_feat
        assert self.fs_instance.feature_importance.shape[1] == 1
        assert self.fs_btc_instance.feature_importance.shape[1] == 1
        assert self.fs_instance.ranked_features.shape[0] == self.fs_instance.features.shape[0]
        assert self.fs_btc_instance.ranked_features.shape[0] == self.fs_btc_instance.features.shape[0]
        assert self.fs_instance.ranked_features.shape[1] == self.fs_instance.features.shape[1]
        assert self.fs_btc_instance.ranked_features.shape[1] == self.fs_btc_instance.features.shape[1]
        assert len(self.fs_instance.ranked_features_list) == self.fs_instance.features.shape[1]
        assert len(self.fs_btc_instance.ranked_features_list) == self.fs_btc_instance.features.shape[1]

    def test_lasso(self) -> None:
        """
        Test LASSO feature selection method.
        """
        # select features
        self.fs_instance.lasso()
        self.fs_btc_instance.lasso()

        # test type
        assert isinstance(self.fs_instance.feature_importance, pd.DataFrame)
        assert isinstance(self.fs_btc_instance.feature_importance, pd.DataFrame)
        assert isinstance(self.fs_instance.ranked_features, pd.DataFrame)
        assert isinstance(self.fs_btc_instance.ranked_features, pd.DataFrame)
        assert isinstance(self.fs_instance.ranked_features_list, list)
        assert isinstance(self.fs_btc_instance.ranked_features_list, list)
        assert (self.fs_instance.feature_importance.dtypes == 'float64').all()
        assert (self.fs_btc_instance.feature_importance.dtypes == 'float64').all()

        # test values
        assert self.fs_instance.ranked_features.isin(self.fs_instance.features).all().all()
        assert self.fs_btc_instance.ranked_features.isin(self.fs_btc_instance.features).all().all()
        assert set(self.fs_instance.ranked_features_list) == set(self.fs_instance.features.columns)
        assert set(self.fs_btc_instance.ranked_features_list) == set(self.fs_btc_instance.features.columns)

        # test shape
        assert self.fs_instance.feature_importance.shape[0] <= self.fs_instance.n_feat
        assert self.fs_btc_instance.feature_importance.shape[0] <= self.fs_btc_instance.n_feat
        assert self.fs_instance.feature_importance.shape[1] == 1
        assert self.fs_btc_instance.feature_importance.shape[1] == 1
        assert self.fs_instance.ranked_features.shape[0] == self.fs_instance.features.shape[0]
        assert self.fs_btc_instance.ranked_features.shape[0] == self.fs_btc_instance.features.shape[0]
        assert self.fs_instance.ranked_features.shape[1] == self.fs_instance.features.shape[1]
        assert self.fs_btc_instance.ranked_features.shape[1] == self.fs_btc_instance.features.shape[1]
        assert len(self.fs_instance.ranked_features_list) == self.fs_instance.features.shape[1]
        assert len(self.fs_btc_instance.ranked_features_list) == self.fs_btc_instance.features.shape[1]

    def test_ridge(self) -> None:
        """
        Test Ridge feature selection method.
        """
        # select features
        self.fs_instance.ridge()
        self.fs_btc_instance.ridge()

        # test type
        assert isinstance(self.fs_instance.feature_importance, pd.DataFrame)
        assert isinstance(self.fs_btc_instance.feature_importance, pd.DataFrame)
        assert isinstance(self.fs_instance.ranked_features, pd.DataFrame)
        assert isinstance(self.fs_btc_instance.ranked_features, pd.DataFrame)
        assert isinstance(self.fs_instance.ranked_features_list, list)
        assert isinstance(self.fs_btc_instance.ranked_features_list, list)
        assert (self.fs_instance.feature_importance.dtypes == 'float64').all()
        assert (self.fs_btc_instance.feature_importance.dtypes == 'float64').all()

        # test values
        assert self.fs_instance.ranked_features.isin(self.fs_instance.features).all().all()
        assert self.fs_btc_instance.ranked_features.isin(self.fs_btc_instance.features).all().all()
        assert set(self.fs_instance.ranked_features_list) == set(self.fs_instance.features.columns)
        assert set(self.fs_btc_instance.ranked_features_list) == set(self.fs_btc_instance.features.columns)

    def test_elastic_net(self) -> None:
        """
        Test Elastic Net feature selection method.
        """
        # feature selection
        self.fs_instance.elastic_net()
        self.fs_btc_instance.elastic_net()

        # test type
        assert isinstance(self.fs_instance.feature_importance, pd.DataFrame)
        assert isinstance(self.fs_btc_instance.feature_importance, pd.DataFrame)
        assert isinstance(self.fs_instance.ranked_features, pd.DataFrame)
        assert isinstance(self.fs_btc_instance.ranked_features, pd.DataFrame)
        assert isinstance(self.fs_instance.ranked_features_list, list)
        assert isinstance(self.fs_btc_instance.ranked_features_list, list)
        assert (self.fs_instance.feature_importance.dtypes == 'float64').all()
        assert (self.fs_btc_instance.feature_importance.dtypes == 'float64').all()

        # test values
        assert self.fs_instance.ranked_features.isin(self.fs_instance.features).all().all()
        assert self.fs_btc_instance.ranked_features.isin(self.fs_btc_instance.features).all().all()
        assert set(self.fs_instance.ranked_features_list) == set(self.fs_instance.features.columns)
        assert set(self.fs_btc_instance.ranked_features_list) == set(self.fs_btc_instance.features.columns)

        # test shape
        assert self.fs_instance.feature_importance.shape[0] <= self.fs_instance.n_feat
        assert self.fs_btc_instance.feature_importance.shape[0] <= self.fs_btc_instance.n_feat
        assert self.fs_instance.feature_importance.shape[1] == 1
        assert self.fs_btc_instance.feature_importance.shape[1] == 1
        assert self.fs_instance.ranked_features.shape[0] == self.fs_instance.features.shape[0]
        assert self.fs_btc_instance.ranked_features.shape[0] == self.fs_btc_instance.features.shape[0]
        assert self.fs_instance.ranked_features.shape[1] == self.fs_instance.features.shape[1]
        assert self.fs_btc_instance.ranked_features.shape[1] == self.fs_btc_instance.features.shape[1]
        assert len(self.fs_instance.ranked_features_list) == self.fs_instance.features.shape[1]
        assert len(self.fs_btc_instance.ranked_features_list) == self.fs_btc_instance.features.shape[1]

    def test_random_forest(self) -> None:
        """
        Test Random Forest feature selection method.
        """
        # feature selection
        self.fs_instance.random_forest()
        self.fs_btc_instance.random_forest()

        # test type
        assert isinstance(self.fs_instance.feature_importance, pd.DataFrame)
        assert isinstance(self.fs_btc_instance.feature_importance, pd.DataFrame)
        assert isinstance(self.fs_instance.ranked_features, pd.DataFrame)
        assert isinstance(self.fs_btc_instance.ranked_features, pd.DataFrame)
        assert isinstance(self.fs_instance.ranked_features_list, list)
        assert isinstance(self.fs_btc_instance.ranked_features_list, list)
        assert (self.fs_instance.feature_importance.dtypes == 'float64').all()
        assert (self.fs_btc_instance.feature_importance.dtypes == 'float64').all()

        # test values
        assert self.fs_instance.ranked_features.isin(self.fs_instance.features).all().all()
        assert self.fs_btc_instance.ranked_features.isin(self.fs_btc_instance.features).all().all()
        assert set(self.fs_instance.ranked_features_list) == set(self.fs_instance.features.columns)
        assert set(self.fs_btc_instance.ranked_features_list) == set(self.fs_btc_instance.features.columns)

        # test shape
        assert self.fs_instance.feature_importance.shape[0] <= self.fs_instance.n_feat
        assert self.fs_btc_instance.feature_importance.shape[0] <= self.fs_btc_instance.n_feat
        assert self.fs_instance.feature_importance.shape[1] == 1
        assert self.fs_btc_instance.feature_importance.shape[1] == 1
        assert self.fs_instance.ranked_features.shape[0] == self.fs_instance.features.shape[0]
        assert self.fs_btc_instance.ranked_features.shape[0] == self.fs_btc_instance.features.shape[0]
        assert self.fs_instance.ranked_features.shape[1] == self.fs_instance.features.shape[1]
        assert self.fs_btc_instance.ranked_features.shape[1] == self.fs_btc_instance.features.shape[1]
        assert len(self.fs_instance.ranked_features_list) == self.fs_instance.features.shape[1]
        assert len(self.fs_btc_instance.ranked_features_list) == self.fs_btc_instance.features.shape[1]

    def test_xgboost(self) -> None:
        """
        Test XGBoost feature selection method.
        """
        # feature selection
        self.fs_instance.xgboost()
        self.fs_btc_instance.xgboost()

        # test type
        assert isinstance(self.fs_instance.feature_importance, pd.DataFrame)
        assert isinstance(self.fs_btc_instance.feature_importance, pd.DataFrame)
        assert isinstance(self.fs_instance.ranked_features, pd.DataFrame)
        assert isinstance(self.fs_btc_instance.ranked_features, pd.DataFrame)
        assert isinstance(self.fs_instance.ranked_features_list, list)
        assert isinstance(self.fs_btc_instance.ranked_features_list, list)
        assert (self.fs_instance.feature_importance.dtypes == 'float64').all()
        assert (self.fs_btc_instance.feature_importance.dtypes == 'float64').all()

        # test values
        assert self.fs_instance.ranked_features.isin(self.fs_instance.features).all().all()
        assert self.fs_btc_instance.ranked_features.isin(self.fs_btc_instance.features).all().all()
        assert set(self.fs_instance.ranked_features_list) == set(self.fs_instance.features.columns)
        assert set(self.fs_btc_instance.ranked_features_list) == set(self.fs_btc_instance.features.columns)

        # test shape
        assert self.fs_instance.feature_importance.shape[0] <= self.fs_instance.n_feat
        assert self.fs_btc_instance.feature_importance.shape[0] <= self.fs_btc_instance.n_feat
        assert self.fs_instance.feature_importance.shape[1] == 1
        assert self.fs_btc_instance.feature_importance.shape[1] == 1
        assert self.fs_instance.ranked_features.shape[0] == self.fs_instance.features.shape[0]
        assert self.fs_btc_instance.ranked_features.shape[0] == self.fs_btc_instance.features.shape[0]
        assert self.fs_instance.ranked_features.shape[1] == self.fs_instance.features.shape[1]
        assert self.fs_btc_instance.ranked_features.shape[1] == self.fs_btc_instance.features.shape[1]
        assert len(self.fs_instance.ranked_features_list) == self.fs_instance.features.shape[1]
        assert len(self.fs_btc_instance.ranked_features_list) == self.fs_btc_instance.features.shape[1]
