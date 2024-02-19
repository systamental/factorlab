import pytest
import pandas as pd

from factorlab.feature_selection.feature_selection import FeatureSelection


@pytest.fixture
def target():
    """
    Fixture for world PMI data.
    """
    # read csv from datasets/data
    pmi_df = pd.read_csv("../src/factorlab/datasets/data/wld_pmi_monthly.csv", index_col=['date', 'ticker'])
    return pmi_df.unstack().actual.WL_Manuf_PMI


@pytest.fixture
def features():
    """
    Fixture for asset class excess returns.
    """
    # read csv from datasets/data
    df = pd.read_csv("../src/factorlab/datasets/data/asset_excess_returns_monthly.csv", index_col=0)
    df = df.drop(columns=['Global_ilb', 'US_hy_credit', 'Global_real_estate', 'US_equity_volatility',
                          'US_breakeven_inflation', 'US_real_estate'])
    pr_df = (1 + df).cumprod()
    yoy_ret = (pr_df / pr_df.shift(12)) - 1
    return yoy_ret


class TestFeatureSelection:
    """
    Test FeatureSelection class.
    """
    @pytest.fixture(autouse=True)
    def fs_setup_default(self, features, target):
        self.default_fs_instance = FeatureSelection(target, features)

    @pytest.fixture(autouse=True)
    def fs_setup_5feat(self, features, target):
        self.fs_5feat_instance = FeatureSelection(target, features, n_feat=5)

    def test_initialization(self, target) -> None:
        """
        Test initialization.
        """
        assert isinstance(self.default_fs_instance, FeatureSelection)
        assert isinstance(self.fs_5feat_instance, FeatureSelection)
        assert isinstance(self.default_fs_instance.data, pd.DataFrame)
        assert isinstance(self.fs_5feat_instance.data, pd.DataFrame)
        assert isinstance(self.default_fs_instance.target, pd.Series)
        assert isinstance(self.fs_5feat_instance.target, pd.Series)
        assert isinstance(self.default_fs_instance.features, pd.DataFrame)
        assert isinstance(self.fs_5feat_instance.features, pd.DataFrame)
        assert all(self.default_fs_instance.index == self.default_fs_instance.data.index)
        assert all(self.fs_5feat_instance.index == self.fs_5feat_instance.data.index)
        assert self.default_fs_instance.lagged_target.shape[1] == 5
        assert all(self.default_fs_instance.target.iloc[-12:].values == target.shift(-1).dropna().iloc[-12:].values)

    def test_check_n_feat(self) -> None:
        """
        Test check_n_feat method.
        """
        # get fs instances
        actual = self.default_fs_instance.n_feat
        actual_5feat = self.fs_5feat_instance.n_feat

        # test type
        assert isinstance(actual, int)
        assert isinstance(actual_5feat, int)

        # test value
        assert actual == self.default_fs_instance.features.shape[1]
        assert actual_5feat == 5

    def test_lars(self) -> None:
        """
        Test LARS feature selection method.
        """
        # select features
        self.default_fs_instance.lars()
        self.fs_5feat_instance.lars()

        # test type
        assert isinstance(self.default_fs_instance.selected_features, pd.DataFrame)
        assert isinstance(self.fs_5feat_instance.selected_features, pd.DataFrame)
        assert isinstance(self.default_fs_instance.ranked_features, pd.DataFrame)
        assert isinstance(self.fs_5feat_instance.ranked_features, pd.DataFrame)
        assert isinstance(self.default_fs_instance.ranked_features_list, list)
        assert isinstance(self.fs_5feat_instance.ranked_features_list, list)
        assert isinstance(self.default_fs_instance.feature_importance, pd.DataFrame)
        assert isinstance(self.fs_5feat_instance.feature_importance, pd.DataFrame)

        # test values
        assert self.default_fs_instance.selected_features.isin(self.default_fs_instance.data).all().all()
        assert self.fs_5feat_instance.selected_features.isin(self.fs_5feat_instance.data).all().all()
        assert self.default_fs_instance.ranked_features.isin(self.default_fs_instance.data).all().all()
        assert self.fs_5feat_instance.ranked_features.isin(self.fs_5feat_instance.data).all().all()
        assert set(self.default_fs_instance.ranked_features_list) == set(self.default_fs_instance.features.columns)
        assert set(self.fs_5feat_instance.ranked_features_list) == set(self.fs_5feat_instance.features.columns)

        # test shape
        assert self.default_fs_instance.selected_features.shape[1] <= self.default_fs_instance.n_feat
        assert self.fs_5feat_instance.selected_features.shape[1] <= self.fs_5feat_instance.n_feat
        assert self.default_fs_instance.selected_features.shape[0] == self.default_fs_instance.data.shape[0]
        assert self.fs_5feat_instance.selected_features.shape[0] == self.fs_5feat_instance.data.shape[0]
        assert self.default_fs_instance.ranked_features.shape[1] == self.default_fs_instance.features.shape[1] - 5
        assert self.fs_5feat_instance.ranked_features.shape[1] == self.fs_5feat_instance.features.shape[1] - 5
        assert self.default_fs_instance.ranked_features.shape[0] == self.default_fs_instance.data.shape[0]
        assert self.fs_5feat_instance.ranked_features.shape[0] == self.fs_5feat_instance.data.shape[0]
        assert len(self.default_fs_instance.ranked_features_list) == self.default_fs_instance.features.shape[1]
        assert len(self.fs_5feat_instance.ranked_features_list) == self.fs_5feat_instance.features.shape[1]

    def test_lasso(self) -> None:
        """
        Test LASSO feature selection method.
        """
        # select features
        self.default_fs_instance.lasso()
        self.fs_5feat_instance.lasso(alpha=0.01)

        # test type
        assert isinstance(self.default_fs_instance.selected_features, pd.DataFrame)
        assert isinstance(self.fs_5feat_instance.selected_features, pd.DataFrame)
        assert isinstance(self.default_fs_instance.ranked_features, pd.DataFrame)
        assert isinstance(self.fs_5feat_instance.ranked_features, pd.DataFrame)
        assert isinstance(self.default_fs_instance.ranked_features_list, list)
        assert isinstance(self.fs_5feat_instance.ranked_features_list, list)
        assert isinstance(self.default_fs_instance.feature_importance, pd.DataFrame)
        assert isinstance(self.fs_5feat_instance.feature_importance, pd.DataFrame)

        # test values
        assert self.default_fs_instance.selected_features.isin(self.default_fs_instance.data).all().all()
        assert self.fs_5feat_instance.selected_features.isin(self.fs_5feat_instance.data).all().all()
        assert self.default_fs_instance.ranked_features.isin(self.default_fs_instance.data).all().all()
        assert self.fs_5feat_instance.ranked_features.isin(self.fs_5feat_instance.data).all().all()
        assert set(self.default_fs_instance.ranked_features_list) == set(self.default_fs_instance.features.columns)
        assert set(self.fs_5feat_instance.ranked_features_list) == set(self.fs_5feat_instance.features.columns)

        # test shape
        assert self.default_fs_instance.selected_features.shape[1] <= self.default_fs_instance.n_feat
        assert self.fs_5feat_instance.selected_features.shape[1] <= self.fs_5feat_instance.n_feat
        assert self.default_fs_instance.selected_features.shape[0] == self.default_fs_instance.data.shape[0]
        assert self.fs_5feat_instance.selected_features.shape[0] == self.fs_5feat_instance.data.shape[0]
        assert self.default_fs_instance.ranked_features.shape[1] == self.default_fs_instance.features.shape[1] - 5
        assert self.fs_5feat_instance.ranked_features.shape[1] == self.fs_5feat_instance.features.shape[1] - 5
        assert self.default_fs_instance.ranked_features.shape[0] == self.default_fs_instance.data.shape[0]
        assert self.fs_5feat_instance.ranked_features.shape[0] == self.fs_5feat_instance.data.shape[0]
        assert len(self.default_fs_instance.ranked_features_list) == self.default_fs_instance.features.shape[1]
        assert len(self.fs_5feat_instance.ranked_features_list) == self.fs_5feat_instance.features.shape[1]

    def test_ridge(self) -> None:
        """
        Test ridge feature selection method.
        """
        # select features
        self.default_fs_instance.ridge()
        self.fs_5feat_instance.ridge()

        # test type
        assert isinstance(self.default_fs_instance.selected_features, pd.DataFrame)
        assert isinstance(self.fs_5feat_instance.selected_features, pd.DataFrame)
        assert isinstance(self.default_fs_instance.ranked_features, pd.DataFrame)
        assert isinstance(self.fs_5feat_instance.ranked_features, pd.DataFrame)
        assert isinstance(self.default_fs_instance.ranked_features_list, list)
        assert isinstance(self.fs_5feat_instance.ranked_features_list, list)
        assert isinstance(self.default_fs_instance.feature_importance, pd.DataFrame)
        assert isinstance(self.fs_5feat_instance.feature_importance, pd.DataFrame)

        # test values
        assert self.default_fs_instance.selected_features.isin(self.default_fs_instance.data).all().all()
        assert self.fs_5feat_instance.selected_features.isin(self.fs_5feat_instance.data).all().all()
        assert self.default_fs_instance.ranked_features.isin(self.default_fs_instance.data).all().all()
        assert self.fs_5feat_instance.ranked_features.isin(self.fs_5feat_instance.data).all().all()
        assert set(self.default_fs_instance.ranked_features_list) == set(self.default_fs_instance.features.columns)
        assert set(self.fs_5feat_instance.ranked_features_list) == set(self.fs_5feat_instance.features.columns)

        # test shape
        assert self.default_fs_instance.selected_features.shape[1] <= self.default_fs_instance.n_feat
        assert self.fs_5feat_instance.selected_features.shape[1] <= self.fs_5feat_instance.n_feat
        assert self.default_fs_instance.selected_features.shape[0] == self.default_fs_instance.data.shape[0]
        assert self.fs_5feat_instance.selected_features.shape[0] == self.fs_5feat_instance.data.shape[0]
        assert self.default_fs_instance.ranked_features.shape[1] == self.default_fs_instance.features.shape[1] - 5
        assert self.fs_5feat_instance.ranked_features.shape[1] == self.fs_5feat_instance.features.shape[1] - 5
        assert self.default_fs_instance.ranked_features.shape[0] == self.default_fs_instance.data.shape[0]
        assert self.fs_5feat_instance.ranked_features.shape[0] == self.fs_5feat_instance.data.shape[0]
        assert len(self.default_fs_instance.ranked_features_list) == self.default_fs_instance.features.shape[1]
        assert len(self.fs_5feat_instance.ranked_features_list) == self.fs_5feat_instance.features.shape[1]

    def test_elastic_net(self) -> None:
        """
        Test elastic net feature selection method.
        """
        # select features
        self.default_fs_instance.elastic_net()
        self.fs_5feat_instance.elastic_net()

        # test type
        assert isinstance(self.default_fs_instance.selected_features, pd.DataFrame)
        assert isinstance(self.fs_5feat_instance.selected_features, pd.DataFrame)
        assert isinstance(self.default_fs_instance.ranked_features, pd.DataFrame)
        assert isinstance(self.fs_5feat_instance.ranked_features, pd.DataFrame)
        assert isinstance(self.default_fs_instance.ranked_features_list, list)
        assert isinstance(self.fs_5feat_instance.ranked_features_list, list)
        assert isinstance(self.default_fs_instance.feature_importance, pd.DataFrame)
        assert isinstance(self.fs_5feat_instance.feature_importance, pd.DataFrame)

        # test values
        assert self.default_fs_instance.selected_features.isin(self.default_fs_instance.data).all().all()
        assert self.fs_5feat_instance.selected_features.isin(self.fs_5feat_instance.data).all().all()
        assert self.default_fs_instance.ranked_features.isin(self.default_fs_instance.data).all().all()
        assert self.fs_5feat_instance.ranked_features.isin(self.fs_5feat_instance.data).all().all()
        assert set(self.default_fs_instance.ranked_features_list) == set(self.default_fs_instance.features.columns)
        assert set(self.fs_5feat_instance.ranked_features_list) == set(self.fs_5feat_instance.features.columns)

        # test shape
        assert self.default_fs_instance.selected_features.shape[1] <= self.default_fs_instance.n_feat
        assert self.fs_5feat_instance.selected_features.shape[1] <= self.fs_5feat_instance.n_feat
        assert self.default_fs_instance.selected_features.shape[0] == self.default_fs_instance.data.shape[0]
        assert self.fs_5feat_instance.selected_features.shape[0] == self.fs_5feat_instance.data.shape[0]
        assert self.default_fs_instance.ranked_features.shape[1] == self.default_fs_instance.features.shape[1] - 5
        assert self.fs_5feat_instance.ranked_features.shape[1] == self.fs_5feat_instance.features.shape[1] - 5
        assert self.default_fs_instance.ranked_features.shape[0] == self.default_fs_instance.data.shape[0]
        assert self.fs_5feat_instance.ranked_features.shape[0] == self.fs_5feat_instance.data.shape[0]
        assert len(self.default_fs_instance.ranked_features_list) == self.default_fs_instance.features.shape[1]
        assert len(self.fs_5feat_instance.ranked_features_list) == self.fs_5feat_instance.features.shape[1]

    def test_random_forest(self) -> None:
        """
        Test random forest feature selection method.
        """
        # select features
        self.default_fs_instance.random_forest()
        self.fs_5feat_instance.random_forest()

        # test type
        assert isinstance(self.default_fs_instance.selected_features, pd.DataFrame)
        assert isinstance(self.fs_5feat_instance.selected_features, pd.DataFrame)
        assert isinstance(self.default_fs_instance.ranked_features, pd.DataFrame)
        assert isinstance(self.fs_5feat_instance.ranked_features, pd.DataFrame)
        assert isinstance(self.default_fs_instance.ranked_features_list, list)
        assert isinstance(self.fs_5feat_instance.ranked_features_list, list)
        assert isinstance(self.default_fs_instance.feature_importance, pd.DataFrame)
        assert isinstance(self.fs_5feat_instance.feature_importance, pd.DataFrame)

        # test values
        assert self.default_fs_instance.selected_features.isin(self.default_fs_instance.data).all().all()
        assert self.fs_5feat_instance.selected_features.isin(self.fs_5feat_instance.data).all().all()
        assert self.default_fs_instance.ranked_features.isin(self.default_fs_instance.data).all().all()
        assert self.fs_5feat_instance.ranked_features.isin(self.fs_5feat_instance.data).all().all()
        assert set(self.default_fs_instance.ranked_features_list) == set(self.default_fs_instance.features.columns)
        assert set(self.fs_5feat_instance.ranked_features_list) == set(self.fs_5feat_instance.features.columns)

        # test shape
        assert self.default_fs_instance.selected_features.shape[1] <= self.default_fs_instance.n_feat
        assert self.fs_5feat_instance.selected_features.shape[1] <= self.fs_5feat_instance.n_feat
        assert self.default_fs_instance.selected_features.shape[0] == self.default_fs_instance.data.shape[0]
        assert self.fs_5feat_instance.selected_features.shape[0] == self.fs_5feat_instance.data.shape[0]
        assert self.default_fs_instance.ranked_features.shape[1] == self.default_fs_instance.features.shape[1] - 5
        assert self.fs_5feat_instance.ranked_features.shape[1] == self.fs_5feat_instance.features.shape[1] - 5
        assert self.default_fs_instance.ranked_features.shape[0] == self.default_fs_instance.data.shape[0]
        assert self.fs_5feat_instance.ranked_features.shape[0] == self.fs_5feat_instance.data.shape[0]
        assert len(self.default_fs_instance.ranked_features_list) == self.default_fs_instance.features.shape[1]
        assert len(self.fs_5feat_instance.ranked_features_list) == self.fs_5feat_instance.features.shape[1]

    def test_xgboost(self) -> None:
        """
        Test XGBoost feature selection method.
        """
        # select features
        self.default_fs_instance.xgboost()
        self.fs_5feat_instance.xgboost()

        # test type
        assert isinstance(self.default_fs_instance.selected_features, pd.DataFrame)
        assert isinstance(self.fs_5feat_instance.selected_features, pd.DataFrame)
        assert isinstance(self.default_fs_instance.ranked_features, pd.DataFrame)
        assert isinstance(self.fs_5feat_instance.ranked_features, pd.DataFrame)
        assert isinstance(self.default_fs_instance.ranked_features_list, list)
        assert isinstance(self.fs_5feat_instance.ranked_features_list, list)
        assert isinstance(self.default_fs_instance.feature_importance, pd.DataFrame)
        assert isinstance(self.fs_5feat_instance.feature_importance, pd.DataFrame)

        # test values
        assert self.default_fs_instance.selected_features.isin(self.default_fs_instance.data).all().all()
        assert self.fs_5feat_instance.selected_features.isin(self.fs_5feat_instance.data).all().all()
        assert self.default_fs_instance.ranked_features.isin(self.default_fs_instance.data).all().all()
        assert self.fs_5feat_instance.ranked_features.isin(self.fs_5feat_instance.data).all().all()
        assert set(self.default_fs_instance.ranked_features_list) == set(self.default_fs_instance.features.columns)
        assert set(self.fs_5feat_instance.ranked_features_list) == set(self.fs_5feat_instance.features.columns)

        # test shape
        assert self.default_fs_instance.selected_features.shape[1] <= self.default_fs_instance.n_feat
        assert self.fs_5feat_instance.selected_features.shape[1] <= self.fs_5feat_instance.n_feat
        assert self.default_fs_instance.selected_features.shape[0] == self.default_fs_instance.data.shape[0]
        assert self.fs_5feat_instance.selected_features.shape[0] == self.fs_5feat_instance.data.shape[0]
        assert self.default_fs_instance.ranked_features.shape[1] == self.default_fs_instance.features.shape[1] - 5
        assert self.fs_5feat_instance.ranked_features.shape[1] == self.fs_5feat_instance.features.shape[1] - 5
        assert self.default_fs_instance.ranked_features.shape[0] == self.default_fs_instance.data.shape[0]
        assert self.fs_5feat_instance.ranked_features.shape[0] == self.fs_5feat_instance.data.shape[0]
        assert len(self.default_fs_instance.ranked_features_list) == self.default_fs_instance.features.shape[1]
        assert len(self.fs_5feat_instance.ranked_features_list) == self.fs_5feat_instance.features.shape[1]
