import pytest
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

from factorlab.feature_engineering.transformations import Transform
from factorlab.signal_generation.unsupervised_learning import PCAWrapper, R2PCA, PPCA


@pytest.fixture
def fx_returns_z_monthly():
    """
    Fixture for standardized (z-score) monthly FX returns.
    """
    # read csv from datasets/data
    df = pd.read_csv("../src/factorlab/datasets/data/fx_returns_monthly.csv", index_col='date', parse_dates=True)
    # compute standardized returns and return dataframe with single index
    return Transform(df).normalize(axis='ts', window_type='expanding').dropna()


@pytest.fixture
def us_eqty_returns_z_daily():
    """
    Fixture for standardized (z-score) daily US equity returns.
    """
    # read csv from datasets/data
    df = pd.read_csv("../src/factorlab/datasets/data/us_equities_top3000_close_adj_daily.csv",
                     index_col=['date', 'ticker'], parse_dates=True)
    # drop tickers with nobs < ts_obs
    obs = df.unstack().iloc[-2500:].close_adj.count()
    drop_tickers_list = obs[obs < 2500].index.to_list()
    df1 = df.drop(drop_tickers_list, level=1, axis=0)  # drop tickers with nobs < ts_obs
    # compute standardized returns and return dataframe with single index
    eqty_ret_df = Transform(df1.unstack().close_adj).returns()
    return Transform(eqty_ret_df).normalize(axis='ts', window_type='expanding').dropna()


@pytest.fixture
def us_eqty_returns_monthly_yoy():
    """
    Fixture for monthly US equity returns, yoy % chg.
    """
    # read csv from datasets/data
    df = pd.read_csv("../src/factorlab/datasets/data/us_equities_top3000_close_adj_daily.csv",
                     index_col=['date', 'ticker'], parse_dates=True)
    # drop tickers with nobs < ts_obs
    obs = df.unstack().iloc[-2500:].close_adj.count()
    drop_tickers_list = obs[obs < 2500].index.to_list()
    df1 = df.drop(drop_tickers_list, level=1, axis=0)  # drop tickers with nobs < ts_obs
    # resample to monthly
    eqty_ret_df = df1.unstack().close_adj.resample('M').last().pct_change(12).dropna()

    return eqty_ret_df


@pytest.fixture
def crypto_returns_z_daily():
    """
    Fixture for daily cryptoasset returns.
    """
    # read csv from datasets/data
    df = pd.read_csv('../src/factorlab/datasets/data/crypto_market_data.csv', index_col=['date', 'ticker'],
                     parse_dates=True)
    ret = Transform(df).returns()
    # compute standardized returns and return dataframe with single index
    return Transform(ret).normalize(axis='ts', window_type='expanding', min_periods=30).close.unstack()


@pytest.fixture
def macro_z_monthly():
    """
    Fixture for global growth (PMI) and inflation data.
    """
    # read csv from datasets/data
    pmi_df = pd.read_csv('../src/factorlab/datasets/data/wld_pmi_monthly.csv', index_col=['date', 'ticker'],
                     parse_dates=True)
    infl_df = pd.read_csv('../src/factorlab/datasets/data/wld_infl_cpi_yoy_monthly.csv', index_col=['date'],
                          parse_dates=True)
    # macro df
    macro_df = pd.concat([pmi_df.unstack().actual.WL_Manuf_PMI, infl_df.actual], axis=1)
    macro_df.columns = ['growth', 'inflation']
    macro_z_df = Transform(macro_df).normalize(axis='ts', window_type='expanding').dropna()

    return macro_z_df


class TestPCAWrapper:
    """
    Test PCAWrapper class.
    """
    @pytest.fixture(autouse=True)
    def pca_setup_default(self, fx_returns_z_monthly):
        self.default_pca_instance = PCAWrapper(fx_returns_z_monthly)

    def test_initialization(self, fx_returns_z_monthly) -> None:
        """
        Test initialization.
        """
        assert isinstance(self.default_pca_instance, PCAWrapper)
        assert isinstance(self.default_pca_instance.data, np.ndarray)
        assert (pd.DataFrame(self.default_pca_instance.data).isna().sum().sum()) == 0
        assert self.default_pca_instance.n_components == min(fx_returns_z_monthly.shape)
        assert all(self.default_pca_instance.index == fx_returns_z_monthly.index)
        assert np.allclose(self.default_pca_instance.data_window, self.default_pca_instance.data)
        assert isinstance(self.default_pca_instance.pca, PCA)

    def test_remove_missing(self) -> None:
        """
        Test remove_missing method.
        """
        # remove missing values
        data = self.default_pca_instance.remove_missing()
        assert len(data[np.isnan(data)]) == 0

    def test_sign_pc1(self, fx_returns_z_monthly) -> None:
        """
        Test sign of PC1.
        """
        # get actual and expected
        actual = self.default_pca_instance.get_pcs()
        cs_mean = fx_returns_z_monthly.mean(axis=1)

        # test sign of pc1
        assert pd.concat([actual.iloc[:, 0], cs_mean], axis=1).corr().iloc[0, 1] > 0

        # test shape
        assert actual.shape == fx_returns_z_monthly.shape

        # index
        assert all(actual.index == fx_returns_z_monthly.index)

        # cols
        assert all(actual.columns == range(fx_returns_z_monthly.shape[1]))

    def test_get_rolling_pcs(self, fx_returns_z_monthly, window_size=60) -> None:
        """
        Test get_rolling_pcs method.
        """
        # get rolling and fixed pcs
        rolling = self.default_pca_instance.get_rolling_pcs(window_size=window_size)
        fixed = self.default_pca_instance.get_pcs()

        # test pc1 correl
        assert pd.concat([fixed[0], rolling[0]], axis=1).corr().iloc[0, 1] > 0.9

        # test shape
        assert (fixed.shape[0] - rolling.shape[0]) < window_size

        # index
        assert all(rolling.index == fixed.iloc[-rolling.shape[0]:].index)

        # cols
        assert all(rolling.columns == range(fx_returns_z_monthly.shape[1]))

    def test_get_rolling_expl_var(self, fx_returns_z_monthly, window_size=60) -> None:
        """
        Test get_rolling_expl_var method.
        """
        # get rolling and fixed pcs
        rolling = self.default_pca_instance.get_rolling_expl_var_ratio(window_size=window_size)
        fixed = self.default_pca_instance.get_expl_var_ratio()

        # test values
        assert all(np.abs(fixed - rolling.mean()) < 0.15)

        # test shape
        assert (fixed.shape[0] - rolling.shape[0]) < 60

        # index
        assert all(rolling.index == fx_returns_z_monthly.iloc[-rolling.shape[0]:].index)

        # cols
        assert all(rolling.columns == range(fx_returns_z_monthly.shape[1]))

    def test_get_expanding_pcs(self, fx_returns_z_monthly, min_obs=100) -> None:
        """
        Test get_expanding_pcs method.
        """
        # get exp and fixed pcs
        exp = self.default_pca_instance.get_expanding_pcs(min_obs=min_obs)
        fixed = self.default_pca_instance.get_pcs()

        # test pc1 correl
        assert pd.concat([fixed[0], exp[0]], axis=1).corr().iloc[0, 1] > 0.9
        # test values
        assert np.allclose(fixed.iloc[-1], exp.iloc[-1])

        # test shape
        assert (fixed.shape[0] - exp.shape[0]) < min_obs

        # index
        assert all(exp.index == fixed.iloc[-exp.shape[0]:].index)

        # cols
        assert all(exp.columns == range(fx_returns_z_monthly.shape[1]))

    def test_get_expanding_expl_var(self, fx_returns_z_monthly, min_obs=100) -> None:
        """
        Test get_expanding_expl_var method.
        """
        # get exp and fixed pcs
        exp = self.default_pca_instance.get_expanding_expl_var_ratio(min_obs=min_obs)
        fixed = self.default_pca_instance.get_expl_var_ratio()

        # test values
        assert all(np.abs(fixed - exp.mean()) < 0.15)
        assert np.allclose(exp.iloc[-1].values, fixed)

        # test shape
        assert (exp.shape[0] - fx_returns_z_monthly.shape[0]) < min_obs

        # index
        assert all(exp.index == fx_returns_z_monthly.iloc[-exp.shape[0]:].index)

        # cols
        assert all(exp.columns == range(fx_returns_z_monthly.shape[1]))


class TestR2PCA:
    """
    Test R2PCA class.
    """
    @pytest.fixture(autouse=True)
    def pca_setup_default(self, fx_returns_z_monthly):
        self.default_pca_instance = R2PCA(fx_returns_z_monthly)

    def test_initialization(self, fx_returns_z_monthly) -> None:
        """
        Test initialization.
        """
        assert isinstance(self.default_pca_instance, R2PCA)
        assert isinstance(self.default_pca_instance.pca, PCA)
        assert isinstance(self.default_pca_instance.data, np.ndarray)
        assert self.default_pca_instance.n_components == min(fx_returns_z_monthly.shape)
        assert all(self.default_pca_instance.index == fx_returns_z_monthly.index)
        assert all(self.default_pca_instance.data_window == fx_returns_z_monthly)

    def test_remove_missing(self) -> None:
        """
        Test remove_missing method.
        """
        # remove missing values
        data = self.default_pca_instance.remove_missing()
        assert len(data[np.isnan(data)]) == 0

    def test_sign_pc1(self, fx_returns_z_monthly) -> None:
        """
        Test sign of PC1.
        """
        # get actual and expected
        actual = self.default_pca_instance.get_pcs()
        cs_mean = fx_returns_z_monthly.mean(axis=1)

        # test sign of pc1
        assert pd.concat([actual.iloc[:, 0], cs_mean], axis=1).corr().iloc[0, 1] > 0

        # test shape
        assert actual.shape == fx_returns_z_monthly.shape

        # index
        assert all(actual.index == fx_returns_z_monthly.index)

        # cols
        assert all(actual.columns == range(self.default_pca_instance.n_components))

    def test_get_rolling_pcs(self, fx_returns_z_monthly, window_size=60) -> None:
        """
        Test get_rolling_pcs method.
        """
        # get fixed and rolling pcs
        fixed = self.default_pca_instance.get_pcs()
        rolling = self.default_pca_instance.get_rolling_pcs(window_size=window_size)

        # test pc1 correl
        assert pd.concat([fixed[0], rolling[0]], axis=1).corr().iloc[0, 1] > 0.9

        # test shape
        assert (fx_returns_z_monthly.shape[0] - rolling.shape[0]) < window_size
        assert rolling.shape[1] == self.default_pca_instance.n_components

        # index
        assert all(rolling.index == fx_returns_z_monthly.iloc[-rolling.shape[0]:].index)

        # cols
        assert all(rolling.columns == range(self.default_pca_instance.n_components))

    def test_get_rolling_expl_var(self, fx_returns_z_monthly, window_size=60) -> None:
        """
        Test get_rolling_expl_var method.
        """
        # get fixed and rolling pcs
        fixed = self.default_pca_instance.get_expl_var_ratio()
        rolling = self.default_pca_instance.get_rolling_expl_var_ratio(window_size=window_size)

        # test values
        assert all(np.abs(fixed - rolling.mean()) < 0.15)

        # test shape
        assert (fx_returns_z_monthly.shape[0] - rolling.shape[0]) < 60
        assert rolling.shape[1] == self.default_pca_instance.n_components

        # index
        assert all(rolling.index == fx_returns_z_monthly.iloc[-rolling.shape[0]:].index)

        # cols
        assert all(rolling.columns == range(self.default_pca_instance.n_components))

    def test_get_expanding_pcs(self, fx_returns_z_monthly, min_obs=100) -> None:
        """
        Test get_expanding_pcs method.
        """
        # get exp and fixed pcs
        exp = self.default_pca_instance.get_expanding_pcs(min_obs=min_obs)
        fixed = self.default_pca_instance.get_pcs()

        # test pc1 correl
        assert pd.concat([fixed[0], exp[0]], axis=1).corr().iloc[0, 1] > 0.9

        # test shape
        assert (fx_returns_z_monthly.shape[0] - exp.shape[0]) < min_obs
        assert exp.shape[1] == self.default_pca_instance.n_components

        # index
        assert all(exp.index == fx_returns_z_monthly.iloc[-exp.shape[0]:].index)

        # cols
        assert all(exp.columns == range(self.default_pca_instance.n_components))

    def test_get_expanding_expl_var(self, fx_returns_z_monthly, min_obs=100) -> None:
        """
        Test get_expanding_expl_var method.
        """
        # get exp and fixed pcs
        exp = self.default_pca_instance.get_expanding_expl_var_ratio(min_obs=min_obs)
        fixed = self.default_pca_instance.get_expl_var_ratio()

        # test values
        assert all(np.abs(fixed - exp.mean()) < 0.15)
        assert np.allclose(exp.iloc[-1].values, fixed)

        # test shape
        assert (exp.shape[0] - fx_returns_z_monthly.shape[0]) < min_obs
        assert exp.shape[1] == self.default_pca_instance.n_components

        # index
        assert all(exp.index == fx_returns_z_monthly.iloc[-exp.shape[0]:].index)

        # cols
        assert all(exp.columns == range(self.default_pca_instance.n_components))


class TestPPCA:
    """
    Test PPCA class.
    """
    @pytest.fixture(autouse=True)
    def ppca_setup_default(self, crypto_returns_z_daily):
        self.default_ppca_instance = PPCA(crypto_returns_z_daily)

    @pytest.fixture(autouse=True)
    def ppca_fx_setup_default(self, fx_returns_z_monthly):
        self.default_fx_ppca_instance = PPCA(fx_returns_z_monthly)

    def test_initialization(self, crypto_returns_z_daily) -> None:
        """
        Test initialization.
        """
        assert isinstance(self.default_ppca_instance, PPCA)
        assert isinstance(self.default_ppca_instance.data, np.ndarray)
        assert self.default_ppca_instance.n_components == min(crypto_returns_z_daily.shape)
        assert all(self.default_ppca_instance.index == crypto_returns_z_daily.index[30:])

    @pytest.mark.parametrize("min_obs, min_feat, expected",
                             [
                                 (10, 5, (2870, 142)),
                                 (10, 8, (2866, 142)),
                                 (365, 5, (2870, 131))
                             ]
                             )
    def test_preprocess_data(self, crypto_returns_z_daily, min_obs, min_feat, expected) -> None:
        """
        Test preprocess_data method.
        """
        # actual
        actual = PPCA(crypto_returns_z_daily, min_obs=min_obs, min_feat=min_feat).preprocess_data().shape

        # test shape
        assert actual == expected

    def test_em_algo(self) -> None:
        """
        Test EM algorithm.
        """
        # actual
        actual = self.default_ppca_instance.em_algo()
        actual_fx = self.default_fx_ppca_instance.em_algo()

        # test shape
        assert actual.shape == (142, 142)
        assert actual_fx.shape == (27, 27)

    def test_decompose(self) -> None:
        """
        Test decompose method.
        """
        # actual
        actual = self.default_ppca_instance.decompose()
        actual_fx = self.default_fx_ppca_instance.decompose()

        # test shape
        assert actual.shape[0] == 142
        assert actual_fx.shape == (27, 27)

    def test_get_eig(self, fx_returns_z_monthly) -> None:
        """
        Test get_eig method.
        """
        actual_ppca_fx = self.default_fx_ppca_instance.get_eigenvectors()
        actual_pca_fx = PCAWrapper(fx_returns_z_monthly).get_eigenvectors()

        # test values
        assert np.allclose(np.abs(actual_ppca_fx), np.abs(actual_pca_fx), rtol=0.25, atol=0.25)

        # test shape
        assert actual_ppca_fx.shape == actual_pca_fx.shape

        # test type
        assert isinstance(actual_ppca_fx, np.ndarray)

    def test_get_pcs(self, fx_returns_z_monthly) -> None:
        """
        Test get_pcs method.
        """
        # actual
        actual_ppca_fx = self.default_fx_ppca_instance.get_pcs()
        actual_pca_fx = PCAWrapper(fx_returns_z_monthly).get_pcs()

        # test values
        assert np.allclose(np.abs(actual_ppca_fx), np.abs(actual_pca_fx), rtol=0.25, atol=0.25)

        # test shape
        assert actual_ppca_fx.shape == actual_pca_fx.shape

        # test type
        assert isinstance(actual_ppca_fx, np.ndarray) or isinstance(actual_pca_fx, pd.DataFrame)

    def test_get_expl_var(self, fx_returns_z_monthly) -> None:
        """
        Test get_expl_var method.
        """
        # actual
        actual_ppca_fx = self.default_fx_ppca_instance.get_expl_var_ratio()
        actual_pca_fx = PCAWrapper(fx_returns_z_monthly).get_expl_var_ratio()

        # test values
        assert np.allclose(actual_ppca_fx, actual_pca_fx)

        # test shape
        assert actual_ppca_fx.shape == actual_pca_fx.shape

        # test type
        assert isinstance(actual_ppca_fx, np.ndarray)
