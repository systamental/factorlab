import pytest
import pandas as pd
import numpy as np

from factorlab.feature_engineering.transformations import Transform
from factorlab.signal_generation.supervised_learning import LinearModel, SPCA


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


@pytest.fixture
def asset_rets_yoy_z_monthly():
    """
    Fixture for asset returns data.
    """
    # agg asset er
    asset_class_er_df = pd.read_csv('../src/factorlab/datasets/data/asset_excess_returns_monthly.csv',
                                    index_col=0,
                                    parse_dates=True)
    asset_class_er_df = asset_class_er_df.resample('M').sum()
    asset_class_er_df.index.name = 'date'
    # remove short series
    asset_class_er_df = asset_class_er_df.drop(
        columns=['Global_ilb', 'US_hy_credit', 'Global_real_estate', 'US_equity_volatility', 'US_breakeven_inflation',
                 'US_real_estate'])
    # price
    asset_class_pr_df = (1 + asset_class_er_df).cumprod()
    # yoy
    asset_class_yoy_ret = (asset_class_pr_df / asset_class_pr_df.shift(12)) - 1
    # zscore
    asset_class_yoy_ret_z = Transform(asset_class_yoy_ret).normalize(axis='ts', window_type='expanding').dropna()

    return asset_class_yoy_ret_z


class TestLinearModel:
    """
    Test class for LinearModel.
    """
    @pytest.fixture(autouse=True)
    def lm_setup_default(self, macro_z_monthly, asset_rets_yoy_z_monthly):
        self.default_lm_instance = LinearModel(macro_z_monthly.growth, asset_rets_yoy_z_monthly, method='lasso',
                                               alpha=0.002, t_lags=6, h_lookahead=1)

    def test_initialization(self, macro_z_monthly, asset_rets_yoy_z_monthly, t_lags=6, h_lookahead=1) -> None:
        """
        Test initialization.
        """
        # data type
        assert isinstance(self.default_lm_instance, LinearModel)
        assert isinstance(self.default_lm_instance.target_lags, pd.DataFrame) or \
               isinstance(self.default_lm_instance.target_lags, np.ndarray)
        assert isinstance(self.default_lm_instance.feature_lags, pd.DataFrame) or \
               isinstance(self.default_lm_instance.feature_lags, np.ndarray)
        assert isinstance(self.default_lm_instance.features, pd.DataFrame) or \
               isinstance(self.default_lm_instance.features, np.ndarray)
        assert isinstance(self.default_lm_instance.data, np.ndarray) or \
               isinstance(self.default_lm_instance.data, pd.DataFrame)
        assert isinstance(self.default_lm_instance.target_fcst, np.ndarray) or \
               isinstance(self.default_lm_instance.target_fcst, pd.Series)
        assert isinstance(self.default_lm_instance.predictors, np.ndarray) or \
               isinstance(self.default_lm_instance.predictors, pd.DataFrame)
        assert isinstance(self.default_lm_instance.features_window, np.ndarray) or \
               isinstance(self.default_lm_instance.features_window, pd.DataFrame)
        # shape
        assert self.default_lm_instance.target_lags.shape[0] == self.default_lm_instance.target.shape[0]
        assert self.default_lm_instance.target_lags.shape[1] == 1 + t_lags
        assert self.default_lm_instance.features.shape[1] == self.default_lm_instance.target_lags.shape[1] + \
               self.default_lm_instance.feature_lags.shape[1]
        assert self.default_lm_instance.target_fcst.shape[0] <= self.default_lm_instance.target.shape[0] - h_lookahead
        assert self.default_lm_instance.predictors.shape[0] == self.default_lm_instance.target_fcst.shape[0]
        assert self.default_lm_instance.predictors.shape[1] == self.default_lm_instance.features.shape[1]
        assert self.default_lm_instance.features_window.shape[0] == self.default_lm_instance.features.shape[0]
        assert self.default_lm_instance.features_window.shape[1] == self.default_lm_instance.features.shape[1]
        # values
        assert self.default_lm_instance.data.isna().sum().sum() == 0
        assert self.default_lm_instance.target_fcst.isna().sum() == 0
        assert self.default_lm_instance.predictors.isna().sum().sum() == 0
        assert self.default_lm_instance.features_window.isna().sum().sum() == 0
        assert all(self.default_lm_instance.features == self.default_lm_instance.features_window)
        # index
        assert all(self.default_lm_instance.index == self.default_lm_instance.features.index)
        # cols
        assert all(self.default_lm_instance.predictors.columns == self.default_lm_instance.features_window.columns)
        assert all([col in self.default_lm_instance.predictors.columns
                    for col in self.default_lm_instance.target_lags.columns])
        assert all([col in self.default_lm_instance.predictors.columns
                    for col in self.default_lm_instance.feature_lags.columns])

    def test_fit(self):
        """
        Test fit method.
        """
        self.default_lm_instance.fit()
        # method
        assert hasattr(self.default_lm_instance, 'model')
        assert self.default_lm_instance.method == 'lasso'
        # type
        assert isinstance(self.default_lm_instance.model, object)

    def test_predict(self):
        """
        Test predict method.
        """
        self.default_lm_instance.predict()
        # type
        assert isinstance(self.default_lm_instance.yhat, pd.DataFrame)
        # shape
        assert self.default_lm_instance.yhat.shape[0] == self.default_lm_instance.features_window.shape[0]
        assert self.default_lm_instance.yhat.shape[1] == 1
        # values
        assert all(self.default_lm_instance.yhat.isna().sum() == 0)
        # index
        assert all(self.default_lm_instance.yhat.index == self.default_lm_instance.features_window.index)
        # cols
        assert self.default_lm_instance.yhat.columns[0] == f"{self.default_lm_instance.target.name}_fcst_t+" \
                                                           f"{str(self.default_lm_instance.h_lookahead)}"

    @pytest.mark.parametrize("metric", ['mse', 'rmse', 'r2', 'adj_r2', 'chg_accuracy'])
    def test_compute_score(self, metric) -> None:
        """
        Test compute_score method.
        """
        # actual
        actual = self.default_lm_instance.compute_score(metric)
        # type
        assert isinstance(actual, float)
        # values
        assert 0.0 <= actual <= 1.0

    def test_get_selected_features(self):
        """
        Test get_feature_importance method.
        """
        self.default_lm_instance.get_selected_features()
        # type
        assert isinstance(self.default_lm_instance.selected_features, pd.DataFrame)
        assert isinstance(self.default_lm_instance.feature_importance, pd.DataFrame)
        # shape
        assert self.default_lm_instance.feature_importance.shape[1] <= self.default_lm_instance.features.shape[1]
        assert self.default_lm_instance.selected_features.shape[1] <= self.default_lm_instance.features.shape[1]
        assert self.default_lm_instance.feature_importance.shape[1] == 1
        # cols
        assert all([feat in self.default_lm_instance.features.columns
                    for feat in self.default_lm_instance.selected_features.columns])
        assert all([feat in self.default_lm_instance.features.columns
                    for feat in self.default_lm_instance.feature_importance.index])

    def test_expanding_window_data(self, row=120):
        """
        Test expanding_window_data method.
        """
        # data
        self.default_lm_instance.expanding_window_data(row=row)
        # index
        assert all(self.default_lm_instance.predictors.index == self.default_lm_instance.target_fcst.index)
        # cols
        assert all(self.default_lm_instance.predictors.columns == self.default_lm_instance.features_window.columns)
        # shape
        assert self.default_lm_instance.predictors.shape[0] == row
        assert self.default_lm_instance.features_window.shape[0] == row

    def test_rolling_window_data(self, row=0, window_size=120):
        """
        Test rolling_window_data method.
        """
        # data
        self.default_lm_instance.rolling_window_data(row, window_size)
        # index
        assert all(self.default_lm_instance.predictors.index == self.default_lm_instance.target_fcst.index)
        # cols
        assert all(self.default_lm_instance.predictors.columns == self.default_lm_instance.features_window.columns)
        # shape
        assert self.default_lm_instance.predictors.shape[0] == self.default_lm_instance.target_fcst.shape[0]
        assert self.default_lm_instance.predictors.shape[0] == window_size
        assert self.default_lm_instance.target_fcst.shape[0] == window_size
        assert self.default_lm_instance.features_window.shape[0] == window_size

    def test_expanding_predict(self, min_obs=120):
        """
        Test expanding_predict method.
        """
        self.default_lm_instance.expanding_predict(min_obs=min_obs)
        n_yhat = self.default_lm_instance.features.shape[0] - \
            (self.default_lm_instance.data.iloc[:min_obs].shape[0] - 1)
        # type
        assert isinstance(self.default_lm_instance.yhat, pd.DataFrame)
        # shape
        assert self.default_lm_instance.yhat.shape[0] == n_yhat
        assert self.default_lm_instance.yhat.shape[1] == 1
        # values
        assert all(self.default_lm_instance.yhat.isna().sum() == 0)
        # index
        assert all(self.default_lm_instance.yhat.index == self.default_lm_instance.features_window.index[-n_yhat:])
        # cols
        assert self.default_lm_instance.yhat.columns[0] == f"{self.default_lm_instance.target.name}_fcst_t+" \
                                                           f"{str(self.default_lm_instance.h_lookahead)}"

    def test_rolling_predict(self, window_size=120):
        """
        Test rolling_predict method.
        """
        self.default_lm_instance.rolling_predict(window_size)
        n_yhat = self.default_lm_instance.features.shape[0] - (window_size - 1)
        # type
        assert isinstance(self.default_lm_instance.yhat, pd.DataFrame)
        # shape
        assert self.default_lm_instance.yhat.shape[0] == n_yhat
        assert self.default_lm_instance.yhat.shape[1] == 1
        # values
        assert all(self.default_lm_instance.yhat.isna().sum() == 0)
        # index
        assert all(self.default_lm_instance.yhat.index == self.default_lm_instance.features.index[-n_yhat:])
        # cols
        assert self.default_lm_instance.yhat.columns[0] == f"{self.default_lm_instance.target.name}_fcst_t+" \
                                                           f"{str(self.default_lm_instance.h_lookahead)}"


class TestSPCA:
    """
    Test class for SPC (Supervised Principal Component Analysis).
    """
    @pytest.fixture(autouse=True)
    def spca_setup_default(self, macro_z_monthly, asset_rets_yoy_z_monthly):
        self.default_spca_instance = SPCA(macro_z_monthly.growth, asset_rets_yoy_z_monthly,  method='lasso',
                                          oos=True, n_feat=30, h_lookahead=6, t_lags=12, alpha=0.0002)

    def test_initialization(self) -> None:
        """
        Test initialization.
        """
        # data type
        assert isinstance(self.default_spca_instance, SPCA)
        assert isinstance(self.default_spca_instance.target, pd.Series) or \
               isinstance(self.default_spca_instance.target, np.ndarray)
        assert isinstance(self.default_spca_instance.features, pd.DataFrame) or \
               isinstance(self.default_spca_instance.features, np.ndarray)
        assert isinstance(self.default_spca_instance.features_window, np.ndarray) or \
               isinstance(self.default_spca_instance.features_window, pd.DataFrame)
        assert isinstance(self.default_spca_instance.method, str)
        assert isinstance(self.default_spca_instance.oos, bool)
        assert isinstance(self.default_spca_instance.n_feat, int)
        assert isinstance(self.default_spca_instance.n_components, int) or \
               self.default_spca_instance.n_components is None
        assert isinstance(self.default_spca_instance.h_lookahead, int)
        assert isinstance(self.default_spca_instance.t_lags, int)
        # values
        assert all(self.default_spca_instance.features == self.default_spca_instance.features_window)
        # index
        assert all(self.default_spca_instance.index == self.default_spca_instance.features.index)
        # cols
        assert all(self.default_spca_instance.features.columns == self.default_spca_instance.features_window.columns)

    def test_get_selected_features(self):
        """
        Test get_feature_importance method.
        """
        self.default_spca_instance.get_selected_features()
        # type
        assert isinstance(self.default_spca_instance.selected_features, pd.DataFrame)
        assert isinstance(self.default_spca_instance.feature_importance, pd.DataFrame)
        # shape
        assert self.default_spca_instance.selected_features.shape[1] <= self.default_spca_instance.features.shape[1]
        assert self.default_spca_instance.feature_importance.shape[1] == 1
        assert self.default_spca_instance.feature_importance.shape[0] == \
               self.default_spca_instance.selected_features.shape[1]
        # cols
        assert all([feat in self.default_spca_instance.features.columns
                    for feat in self.default_spca_instance.selected_features.columns])
        assert all([feat in self.default_spca_instance.features.columns
                    for feat in self.default_spca_instance.feature_importance.index])

    def test_get_pcs(self):
        """
        Test get_pcs method.
        """
        self.default_spca_instance.get_pcs()
        # type
        assert isinstance(self.default_spca_instance.pcs, pd.DataFrame)
        # shape
        assert self.default_spca_instance.pcs.shape[0] == self.default_spca_instance.selected_features.shape[0]
        assert self.default_spca_instance.pcs.shape[1] == self.default_spca_instance.n_components
        # values
        assert all(self.default_spca_instance.pcs.isna().sum() == 0)
        # cols
        assert all([col in self.default_spca_instance.pcs.columns
                    for col in [f"PC{i}" for i in range(1, self.default_spca_instance.n_components + 1)]])

    def test_predict(self):
        """
        Test predict method.
        """
        self.default_spca_instance.predict()
        # type
        assert isinstance(self.default_spca_instance.yhat, pd.DataFrame)
        # shape
        assert self.default_spca_instance.pcs.shape[0] - self.default_spca_instance.yhat.shape[0] == \
               self.default_spca_instance.t_lags
        assert self.default_spca_instance.yhat.shape[1] == 1
        # values
        assert all(self.default_spca_instance.yhat.isna().sum() == 0)
        # index
        assert all(self.default_spca_instance.yhat.index ==
                   self.default_spca_instance.pcs.index[self.default_spca_instance.t_lags:])
        # cols
        assert self.default_spca_instance.yhat.columns[0] == f"{self.default_spca_instance.target.name}_fcst_t+" \
                                                             f"{str(self.default_spca_instance.h_lookahead)}"

    @pytest.mark.parametrize("metric", ['mse', 'rmse', 'r2', 'adj_r2', 'chg_accuracy'])
    def test_compute_score(self, metric) -> None:
        """
        Test compute_score method.
        """
        # actual
        actual = self.default_spca_instance.compute_score(metric)
        # type
        assert isinstance(actual, float)
        # values
        assert 0.0 <= actual <= 1.0

    def test_expanding_window_data(self, row=120):
        """
        Test expanding_window_data method.
        """
        # data
        self.default_spca_instance.expanding_window_data(row=row)
        # shape
        assert self.default_spca_instance.features_window.shape[0] == row

    def test_rolling_window_data(self, row=0, window_size=120):
        """
        Test rolling_window_data method.
        """
        # data
        self.default_spca_instance.rolling_window_data(row, window_size)
        # shape
        assert self.default_spca_instance.features_window.shape[0] == window_size

    def test_expanding_predict(self, min_obs=120):
        """
        Test expanding_predict method.
        """
        self.default_spca_instance.expanding_predict(min_obs=min_obs, method='lasso', alpha=0.0002)
        n_yhat = self.default_spca_instance.features.iloc[min_obs:].shape[0] + 1
        # type
        assert isinstance(self.default_spca_instance.yhat, pd.DataFrame)
        # shape
        assert self.default_spca_instance.yhat.shape[0] == n_yhat
        assert self.default_spca_instance.yhat.shape[1] == 1
        # values
        assert all(self.default_spca_instance.yhat.isna().sum() == 0)
        # index
        assert all(self.default_spca_instance.yhat.index == self.default_spca_instance.features.index[-n_yhat:])
        # cols
        assert self.default_spca_instance.yhat.columns[0] == f"{self.default_spca_instance.target.name}_fcst_t+" \
                                                             f"{str(self.default_spca_instance.h_lookahead)}"

    def test_rolling_predict(self, window_size=120):
        """
        Test rolling_predict method.
        """
        self.default_spca_instance.rolling_predict(window_size=window_size, method='lasso', alpha=0.0002)
        n_yhat = self.default_spca_instance.features.iloc[window_size:].shape[0] + 1
        # type
        assert isinstance(self.default_spca_instance.yhat, pd.DataFrame)
        # shape
        assert self.default_spca_instance.yhat.shape[0] == n_yhat
        assert self.default_spca_instance.yhat.shape[1] == 1
        # values
        assert all(self.default_spca_instance.yhat.isna().sum() == 0)
        # index
        assert all(self.default_spca_instance.yhat.index == self.default_spca_instance.features.index[-n_yhat:])
        # cols
        assert self.default_spca_instance.yhat.columns[0] == f"{self.default_spca_instance.target.name}_fcst_t+" \
                                                             f"{str(self.default_spca_instance.h_lookahead)}"
