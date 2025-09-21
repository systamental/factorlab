import pytest
import pandas as pd
import numpy as np

from factorlab.feature_engineering.factors.size import Size


@pytest.fixture
def mkt_cap():
    """
    Fixture for daily cryptoasset returns.
    """
    # read csv from datasets/data
    df = pd.read_csv('datasets/data/crypto_mkt_cap.csv', index_col=['date', 'ticker'],
                     parse_dates=True)

    return df


class TestSize:
    """
    Test Size class.
    """
    @pytest.fixture(autouse=True)
    def size_setup_default(self, mkt_cap):
        self.size_instance = Size(mkt_cap, size_metric='mkt_cap')

    def test_initialization(self):
        """
        Test initialization.
        """
        assert isinstance(self.size_instance, Size)
        assert isinstance(self.size_instance.df, pd.DataFrame)
        assert isinstance(self.size_instance.df.index, pd.MultiIndex)
        assert 'mkt_cap' in self.size_instance.df.columns
        assert self.size_instance.size_metric == 'mkt_cap'
        assert self.size_instance.log is True

    def test_convert_to_multiindex(self):
        """
        Test convert_to_multiindex method.
        """
        df = self.size_instance.convert_to_multiindex()
        assert isinstance(df.index, pd.MultiIndex)

    def test_compute_size_factor(self):
        """
        Test compute_size_factor method.
        """
        size = self.size_instance.compute_size_factor()

        # shape
        assert size.shape[0] == self.size_instance.df.shape[0]
        assert size.shape[1] == 1
        # data types
        assert isinstance(size, pd.DataFrame)
        assert isinstance(size.index, pd.MultiIndex)
        assert (size.dtypes == np.float64).all()
        # values
        assert (size.values >= 0).all()
        # index
        assert (size.index == self.size_instance.df.index).all()
        # cols
        assert 'size' in size.columns
