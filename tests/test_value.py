import pytest
import pandas as pd
import numpy as np

from factorlab.feature_engineering.factors.value import Value


@pytest.fixture
def onchain_data():
    """
    Fixture for daily cryptoasset on-chain data.
    """
    # read csv from datasets/data
    mktcap_df = pd.read_csv('../src/factorlab/datasets/data/crypto_mkt_cap.csv', index_col=[0, 1], parse_dates=True)
    onchain_df = pd.read_csv('../src/factorlab/datasets/data/crypto_onchain_data.csv', index_col=[0, 1],
                             parse_dates=True)

    # concat dfs
    oc_df = pd.concat([mktcap_df, onchain_df], axis=1).sort_index()
    # mvrv ratio
    oc_df['mvrv'] = oc_df.mkt_cap / oc_df.mkt_cap_real
    # adj transaction value
    oc_df['tfr_val_usd_adj'] = oc_df.tx_count * oc_df.tfr_val_mean_usd
    # nvt ratio
    oc_df['nvt'] = oc_df.mkt_cap / oc_df.tfr_val_usd
    oc_df['nvt_adj'] = oc_df.mkt_cap / oc_df.tfr_val_usd_adj

    return oc_df


class TestValue:
    """
    Test Value class.
    """
    @pytest.fixture(autouse=True)
    def value_setup_default(self, onchain_data):
        self.default_value_instance = Value(onchain_data)

    def test_initialization(self):
        """
        Test initialization.
        """
        assert isinstance(self.default_value_instance, Value)
        assert isinstance(self.default_value_instance.df, pd.DataFrame)
        assert isinstance(self.default_value_instance.df.index, pd.MultiIndex)
        assert any([col not in self.default_value_instance.df.columns for col in ['ratio', 'spot', 'mkt_cap']])

    def test_remove_empty_cols(self):
        """
        Test remove_empty_cols.
        """
        self.default_value_instance.remove_empty_cols()
        assert all((self.default_value_instance.df.unstack().dropna(how='all').isna().sum() /
                    self.default_value_instance.df.unstack().shape[0]) < 1)

    def test_compute_ratio(self):
        """
        Test compute_ratio method.
        """
        # get actual and expected
        actual = self.default_value_instance.compute_ratio(price='mkt_cap', value_metric='add_act')

        # check inf
        assert not actual.isin([np.inf, -np.inf]).any().any()
        # test data types
        assert isinstance(actual, pd.Series)
        assert actual.dtypes == np.float64

    def test_compute_resid(self):
        """
        Test compute_resid method.
        """
        # get actual and expected
        actual = self.default_value_instance.compute_residual(price='mkt_cap', value_metric='add_act')

        # check inf
        assert not actual.isin([np.inf, -np.inf]).any().any()
        # test data types
        assert isinstance(actual, pd.DataFrame)
        assert all(actual.dtypes == np.float64)
        # test name
        assert actual.columns[0] == 'resid'

    def test_compute_value_factor(self):
        """
        Test compute_value_factor method.
        """
        # get actual and expected
        actual = self.default_value_instance.compute_value_factor(price='mkt_cap', value_metric='add_act', name='nvm')

        # check inf
        assert not actual.isin([np.inf, -np.inf]).any().any()
        # test shape
        assert actual.shape[1] == 1
        # test data types
        assert isinstance(actual, pd.DataFrame)
        assert all(actual.dtypes == np.float64)
        # test name
        assert actual.columns[0].split('_')[0] == 'nvm'
        assert actual.columns[0].split('_')[1] == self.default_value_instance.method

    def test_nvt(self):
        """
        Test nvt method.
        """
        # get actual and expected
        actual = self.default_value_instance.nvt(price='mkt_cap', value_metric='tfr_val_usd', name='nvt')

        # check inf
        assert not actual.isin([np.inf, -np.inf]).any().any()
        # test shape
        assert actual.shape[1] == 1
        # test data types
        assert isinstance(actual, pd.DataFrame)
        assert all(actual.dtypes == np.float64)
        # test name
        assert actual.columns[0].split('_')[0] == 'nvt'
        assert actual.columns[0].split('_')[1] == self.default_value_instance.method

    def test_nvm(self):
        """
        Test nvm method.
        """
        # get actual and expected
        actual = self.default_value_instance.nvt(price='mkt_cap', value_metric='add_act', name='nvm')

        # check inf
        assert not actual.isin([np.inf, -np.inf]).any().any()
        # test shape
        assert actual.shape[1] == 1
        # test data types
        assert isinstance(actual, pd.DataFrame)
        assert all(actual.dtypes == np.float64)
        # test name
        assert actual.columns[0].split('_')[0] == 'nvm'
        assert actual.columns[0].split('_')[1] == self.default_value_instance.method

    def test_nvsf(self):
        """
        Test nvsf method.
        """
        # get actual and expected
        actual = self.default_value_instance.nvt(price='mkt_cap', value_metric='supply_circ', name='nvsf')

        # check inf
        assert not actual.isin([np.inf, -np.inf]).any().any()
        # test shape
        assert actual.shape[1] == 1
        # test data types
        assert isinstance(actual, pd.DataFrame)
        assert all(actual.dtypes == np.float64)
        # test name
        assert actual.columns[0].split('_')[0] == 'nvsf'
        assert actual.columns[0].split('_')[1] == self.default_value_instance.method

    def test_nvc(self):
        """
        Test nvc method.
        """
        # get actual and expected
        actual = self.default_value_instance.nvt(price='mkt_cap', value_metric='hashrate', name='nvc')

        # check inf
        assert not actual.isin([np.inf, -np.inf]).any().any()
        # test shape
        assert actual.shape[1] == 1
        # test data types
        assert isinstance(actual, pd.DataFrame)
        assert all(actual.dtypes == np.float64)
        # test name
        assert actual.columns[0].split('_')[0] == 'nvc'
        assert actual.columns[0].split('_')[1] == self.default_value_instance.method

    def test_npm(self):
        """
        Test npm method.
        """
        # get actual and expected
        actual = self.default_value_instance.npm(price='mkt_cap', lookback=200, name='npm')

        # check inf
        assert not actual.isin([np.inf, -np.inf]).any().any()
        # test shape
        assert actual.shape[1] == 1
        # test data types
        assert isinstance(actual, pd.DataFrame)
        assert all(actual.dtypes == np.float64)
        # test name
        assert actual.columns[0].split('_')[0] == 'npm'
        assert actual.columns[0].split('_')[1] == '200'
