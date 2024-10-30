import pytest
import pandas as pd
import numpy as np

from factorlab.feature_engineering.target import Target


@pytest.fixture
def binance_spot():
    """
    Fixture for crypto OHLCV prices.
    """
    # read csv from datasets/data
    return pd.read_csv("../src/factorlab/datasets/data/binance_spot_prices.csv",
                       index_col=['date', 'ticker'],
                       parse_dates=True).loc[:, : 'close']


@pytest.fixture
def btc_spot_prices(binance_spot):
    """
    Fixture for BTC OHLCV prices.
    """
    # read csv from datasets/data
    return binance_spot.loc[pd.IndexSlice[:, 'BTC'], : 'close'].droplevel(1)


@pytest.fixture
def btc_spot_close_price(binance_spot):
    """
    Fixture for BTC close price.
    """
    # read csv from datasets/data
    return binance_spot.loc[pd.IndexSlice[:, 'BTC'], 'close'].droplevel(1)


class TestTarget:
    """
    Test class for Target class.
    """
    @pytest.fixture(autouse=True)
    def target_setup_default(self, binance_spot):
        self.default_target_instance = Target(binance_spot)

    @pytest.fixture(autouse=True)
    def target_setup_no_missing(self, binance_spot):
        df = binance_spot.loc[pd.IndexSlice[:, ['BTC', 'ETH', 'LTC', 'DOGE']], :].unstack().dropna().stack()
        self.no_missing_target_instance = Target(df)

    def test_initialization(self) -> None:
        """
        Test initialization.
        """
        # dtypes
        assert isinstance(self.default_target_instance, Target)
        assert isinstance(self.default_target_instance.df, pd.DataFrame)
        assert isinstance(self.default_target_instance.strategy, str)
        assert isinstance(self.default_target_instance.bins, int)
        assert isinstance(self.default_target_instance.lead, int)
        assert isinstance(self.default_target_instance.vwap, bool)
        assert isinstance(self.default_target_instance.normalize, bool)
        assert isinstance(self.default_target_instance.power_transform, bool)
        assert isinstance(self.default_target_instance.quantize, bool)
        assert isinstance(self.default_target_instance.rank, bool)
        assert isinstance(self.default_target_instance.window_type, str)
        assert isinstance(self.default_target_instance.window_size, int)
        assert (self.default_target_instance.df.dtypes == np.float64).all()

    @pytest.mark.parametrize("method", ['simple', 'log', 'diff'])
    def test_compute_price_chg(self, method) -> None:
        """
        Test compute_price_chg.
        """
        self.default_target_instance.compute_price_chg(method=method)

        # dtypes
        assert (self.default_target_instance.target, pd.DataFrame)
        assert (self.default_target_instance.target.dtypes == np.float64).all()
        # shape
        assert (self.default_target_instance.target.unstack().shape[0] ==
                self.default_target_instance.df.unstack().shape[0] - self.default_target_instance.lead)
        assert self.default_target_instance.target.shape[1] == self.default_target_instance.df.shape[1]
        # index
        assert (self.default_target_instance.target.unstack().index
                .equals(self.default_target_instance.df.unstack().index[:-self.default_target_instance.lead]))
        # columns
        assert (self.default_target_instance.target.columns
                .equals(self.default_target_instance.df.columns))

    @pytest.mark.parametrize("method, centering, strategy",
                             [
                                 ('z-score', True, 'ts'), ('z-score', False, 'ts'),
                                 ('z-score', True, 'cs'), ('z-score', False, 'cs'),
                                 ('min-max', True, 'ts'), ('min-max', False, 'ts'),
                                 ('min-max', True, 'cs'), ('min-max', False, 'cs'),
                                 ('iqr', True, 'ts'), ('iqr', False, 'ts'),
                                 ('iqr', True, 'cs'), ('iqr', False, 'cs'),
                                 ('atr', True, 'ts'), ('atr', False, 'ts'),
                                 ('percentile', True, 'ts'), ('percentile', False, 'ts'),
                                 ('percentile', True, 'cs'), ('percentile', False, 'cs')
                             ]
                             )
    def test_normalize_price_chg(self, method, centering, strategy) -> None:
        """
        Test normalize_price_chg.
        """
        self.default_target_instance.strategy = strategy
        self.default_target_instance.compute_price_chg(method='simple')
        self.default_target_instance.normalize_price_chg(method=method, centering=centering)

        # dtypes
        assert (self.default_target_instance.target, pd.DataFrame)
        assert (self.default_target_instance.target.dtypes == np.float64).all()
        # shape
        if strategy == 'ts':
            assert (self.default_target_instance.target.unstack().shape[0] ==
                    self.default_target_instance.df.unstack().shape[0] - self.default_target_instance.lead - 1)
        else:
            assert (self.default_target_instance.target.unstack().shape[0] ==
                    self.default_target_instance.df.unstack().shape[0] - self.default_target_instance.lead)
        # index
        if strategy == 'ts':
            assert (self.default_target_instance.target.unstack().index
                    .equals(self.default_target_instance.df.unstack()
                            .index[1:-self.default_target_instance.lead]))
        else:
            assert (self.default_target_instance.target.unstack().index
                    .equals(self.default_target_instance.df.unstack().index[:-self.default_target_instance.lead]))
        # columns
        assert (self.default_target_instance.target.columns
                .equals(self.default_target_instance.df.columns))

    @pytest.mark.parametrize("strategy", ['ts', 'cs'])
    def test_transform(self, strategy) -> None:
        """
        Test transform.
        """
        self.no_missing_target_instance.strategy = strategy

        self.no_missing_target_instance.compute_price_chg(method='simple')
        self.no_missing_target_instance.normalize_price_chg(method='z-score', centering=True)
        self.no_missing_target_instance.transform()

        # dtypes
        assert (self.no_missing_target_instance.target, pd.DataFrame)
        assert (self.no_missing_target_instance.target.dtypes == np.float64).all()
        # shape
        if strategy == 'ts':
            assert (self.no_missing_target_instance.target.unstack().shape[0] ==
                    self.no_missing_target_instance.df.unstack().shape[0] - self.no_missing_target_instance.lead -
                    self.no_missing_target_instance.window_size)
        else:
            assert (self.no_missing_target_instance.target.unstack().shape[0] ==
                    self.no_missing_target_instance.df.unstack().shape[0] - 1)
        assert self.no_missing_target_instance.target.shape[1] == self.no_missing_target_instance.df.shape[1]
        # index
        if strategy == 'ts':
            assert (self.no_missing_target_instance.target.unstack().index
                    .equals(self.no_missing_target_instance.df.unstack()
                            .index[self.no_missing_target_instance.window_size:-self.no_missing_target_instance.lead]))
        else:
            assert (self.no_missing_target_instance.target.unstack().index
                    .equals(self.no_missing_target_instance.df.unstack().index[:-self.no_missing_target_instance.lead]))
        # columns
        assert (self.no_missing_target_instance.target.columns
                .equals(self.no_missing_target_instance.df.columns))

    @pytest.mark.parametrize("strategy", ['ts', 'cs'])
    def test_quantize_target(self, strategy) -> None:
        """
        Test quantize_target.
        """
        self.default_target_instance.strategy = strategy

        self.default_target_instance.compute_price_chg(method='simple')
        self.default_target_instance.quantize_target()

        # dtypes
        assert (self.default_target_instance.target, pd.DataFrame)
        assert (self.default_target_instance.target.dtypes == np.float64).all()
        # shape
        if strategy == 'ts':
            assert (self.default_target_instance.target.unstack().shape[0] ==
                    self.default_target_instance.df.unstack().shape[0] - self.default_target_instance.lead - 1)
        else:
            assert (self.default_target_instance.target.unstack().shape[0] ==
                    self.default_target_instance.df.unstack().shape[0] - 1)
        # values
        assert (np.sort(self.default_target_instance.target.close.dropna().unique()) ==
                np.array((range(1, self.default_target_instance.bins + 1))).astype(float)).all()
        # index
        if strategy == 'ts':
            assert (self.default_target_instance.target.unstack().index
                    .equals(self.default_target_instance.df.unstack()
                            .index[1:-self.default_target_instance.lead]))
        # else:
            assert (self.default_target_instance.target.unstack().index
                    .equals(self.default_target_instance.df.unstack().index[1:-self.default_target_instance.lead]))
        # columns
        assert (self.default_target_instance.target.columns
                .equals(self.default_target_instance.df.columns))

    @pytest.mark.parametrize("strategy", ['ts', 'cs'])
    def test_rank_target(self, strategy) -> None:
        """
        Test rank_target.
        """
        self.default_target_instance.strategy = strategy

        self.default_target_instance.compute_price_chg(method='simple')
        self.default_target_instance.rank_target()

        # dtypes
        assert (self.default_target_instance.target, pd.DataFrame)
        assert (self.default_target_instance.target.dtypes == np.float64).all()
        # shape
        if strategy == 'ts':
            assert (self.default_target_instance.target.unstack().shape[0] ==
                    self.default_target_instance.df.unstack().shape[0] - self.default_target_instance.lead - 1)
        else:
            assert (self.default_target_instance.target.unstack().shape[0] ==
                    self.default_target_instance.df.unstack().shape[0] - 1)
        assert self.default_target_instance.target.shape[1] == self.default_target_instance.df.shape[1]
        # index
        if strategy == 'ts':
            assert (self.default_target_instance.target.unstack().index
                    .equals(self.default_target_instance.df.unstack()
                            .index[1:-self.default_target_instance.lead]))
        else:
            assert (self.default_target_instance.target.unstack().index
                    .equals(self.default_target_instance.df.unstack().index[:-self.default_target_instance.lead]))
        # columns
        assert (self.default_target_instance.target.columns
                .equals(self.default_target_instance.df.columns))

    @pytest.mark.parametrize("normalize, power_transform, quantize, rank, strategy",
                             [
                                 (False, False, False, False, 'ts'), (False, False, False, False, 'cs'),
                                 (True, False, False, False, 'ts'), (True, False, False, False, 'cs'),
                                 (True, True, False, False, 'ts'), (True, True, False, False, 'cs'),
                                 (False, False, True, False, 'ts'), (False, False, True, False, 'cs'),
                                 (True, False, False, True, 'ts'), (True, False, False, True, 'cs')
                             ])
    def test_compute_target(self, normalize, power_transform, quantize, rank, strategy) -> None:
        """
        Test compute_target.
        """
        self.no_missing_target_instance.normalize = normalize
        self.no_missing_target_instance.power_transform = power_transform
        self.no_missing_target_instance.quantize = quantize
        self.no_missing_target_instance.rank = rank
        self.no_missing_target_instance.strategy = strategy
        self.no_missing_target_instance.compute_target()

        # dtypes
        assert (self.no_missing_target_instance.target, pd.DataFrame)
        assert (self.no_missing_target_instance.target.dtypes == np.float64).all()
        # columns
        assert (self.no_missing_target_instance.target.columns
                .equals(self.no_missing_target_instance.df.columns))
