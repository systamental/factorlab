import pytest
import pandas as pd
import numpy as np

from factorlab.feature_engineering.transformations import Transform
from factorlab.feature_engineering.factors.trend import Trend
from factorlab.signal_generation.signal import Signal


@pytest.fixture
def spot_prices():
    """
    Fixture for crypto spot prices.
    """
    # read csv from datasets/data
    df = pd.read_csv("../src/factorlab/datasets/data/binance_spot_prices.csv",
                     index_col=['date', 'ticker'],
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
    spot_ret = Transform(spot_prices).returns().unstack().dropna().stack(future_stack=True)

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
    price_mom_10 = Trend(spot_prices, window_size=10).price_mom()
    price_mom = Trend(spot_prices).price_mom()

    # concat
    price_mom = pd.concat([price_mom_10, price_mom], axis=1, keys=['price_mom_10', 'price_mom_20'])
    price_mom = price_mom.unstack().dropna().stack(future_stack=True)

    return price_mom


@pytest.fixture
def btc_price_mom(price_mom):
    """
    Fixture for BTC price momentum.
    """
    # compute btc price mom
    btc_price_mom = price_mom.loc[:, 'BTC', :]

    return btc_price_mom


class TestSignal:
    """
    Test class for Transform.
    """
    @pytest.fixture(autouse=True)
    def ts_signal_default(self, spot_ret, price_mom):
        self.ts_signal = Signal(price_mom, returns=spot_ret.close, signal_thresh=0.6)

    @pytest.fixture(autouse=True)
    def cs_signal_default(self, spot_ret, price_mom):
        self.cs_signal = Signal(price_mom, returns=spot_ret.close, strategy='cross_sectional', signal_thresh=0.6)

    @pytest.fixture(autouse=True)
    def dual_signal_default(self, spot_ret, price_mom):
        self.dual_signal = Signal(price_mom, returns=spot_ret.close, strategy='dual', signal_thresh=0.6)

    @pytest.fixture(autouse=True)
    def btc_ts_signal_default(self, btc_spot_ret, btc_price_mom):
        self.btc_signal = Signal(btc_price_mom, returns=btc_spot_ret.close, signal_thresh=0.6)

    def test_initialization(self) -> None:
        """
        Test initialization.
        """
        # dtypes
        assert isinstance(self.ts_signal, Signal)
        assert isinstance(self.cs_signal, Signal)
        assert isinstance(self.dual_signal, Signal)
        assert isinstance(self.btc_signal, Signal)
        assert isinstance(self.ts_signal.factors, pd.DataFrame)
        assert isinstance(self.cs_signal.factors, pd.DataFrame)
        assert isinstance(self.dual_signal.factors, pd.DataFrame)
        assert isinstance(self.btc_signal.factors, pd.DataFrame)
        assert isinstance(self.ts_signal.returns, pd.DataFrame)
        assert isinstance(self.cs_signal.returns, pd.DataFrame)
        assert isinstance(self.dual_signal.returns, pd.DataFrame)
        assert isinstance(self.btc_signal.returns, pd.DataFrame)
        assert (self.ts_signal.factors.dtypes == np.float64).all()
        assert (self.cs_signal.factors.dtypes == np.float64).all()
        assert (self.dual_signal.factors.dtypes == np.float64).all()
        assert (self.btc_signal.factors.dtypes == np.float64).all()

    def test_check_params_errors(self, price_mom, spot_ret) -> None:
        """
        Test check_params method errors.
        """
        # check strategy
        with pytest.raises(ValueError):
            Signal(price_mom, returns=spot_ret.close, strategy='ts')
        with pytest.raises(ValueError):
            Signal(price_mom, returns=spot_ret.close, strategy='cs')

        # check direction
        with pytest.raises(ValueError):
            Signal(price_mom, returns=spot_ret.close, direction='long_only')
        with pytest.raises(ValueError):
            Signal(price_mom, returns=spot_ret.close, strategy='cross_sectional', direction='short_only')

        # signal type
        with pytest.raises(ValueError):
            Signal(price_mom, returns=spot_ret.close, signal='signal')

        # check signal_thresh
        with pytest.raises(ValueError):
            Signal(price_mom, returns=spot_ret.close, signal_thresh=1.5)

        # check bins
        with pytest.raises(ValueError):
            Signal(price_mom, returns=spot_ret.close, bins=1)

        # check window type
        with pytest.raises(ValueError):
            Signal(price_mom, returns=spot_ret.close, window_type='ts')

        # check window size
        with pytest.raises(ValueError):
            Signal(price_mom, returns=spot_ret.close, window_size=1)

    @pytest.mark.parametrize("method", ['z-score', 'iqr', 'mod_z', 'min-max', 'percentile'])
    def test_normalize_factors(self, method):
        """
        Test normalize method.
        """
        # get actual
        actual_ts = self.ts_signal.normalize_factors(method=method, centering=True)
        actual_cs = self.cs_signal.normalize_factors(method=method, centering=True)
        actual_btc = self.btc_signal.normalize_factors(method=method, centering=True)

        # shape
        assert self.ts_signal.factors.shape == actual_ts.shape
        assert self.cs_signal.factors.shape == actual_cs.shape
        assert self.btc_signal.factors.shape == actual_btc.shape

        # values
        assert (actual_ts.dropna().std() <= 1.5).all()
        assert (actual_cs.dropna().std() <= 1.5).all()
        assert (actual_btc.dropna().std() < 1.5).all()
        assert np.allclose(actual_ts.loc[pd.IndexSlice[:, 'BTC'], :], actual_btc, equal_nan=True)

        # dtypes
        assert isinstance(actual_ts, pd.DataFrame)
        assert isinstance(actual_cs, pd.DataFrame)
        assert isinstance(actual_btc, pd.DataFrame)
        assert (actual_ts.dtypes == np.float64).all()
        assert (actual_cs.dtypes == np.float64).all()
        assert (actual_btc.dtypes == np.float64).all()

        # index
        assert (actual_ts.index == self.ts_signal.factors.index).all()
        assert (actual_cs.index == self.cs_signal.factors.index).all()
        assert (actual_btc.index == self.btc_signal.factors.index).all()

        # cols
        assert (actual_ts.columns == self.ts_signal.factors.columns).all()
        assert (actual_cs.columns == self.cs_signal.factors.columns).all()
        assert (actual_btc.columns == self.btc_signal.factors.columns).all()

    @pytest.mark.parametrize("method", ['box-cox', 'yeo-johnson'])
    def test_transform_factors(self, method):
        """
        Test transform method.
        """
        # normalize factors
        self.ts_signal.normalize_factors(method='z-score', centering=True)
        self.cs_signal.normalize_factors(method='z-score', centering=True)
        self.btc_signal.normalize_factors(method='z-score', centering=True)

        # transform factors
        self.ts_signal.transform, self.ts_signal.window_type = True, 'fixed'
        self.cs_signal.transform, self.cs_signal.window_type = True, 'fixed'
        self.btc_signal.transform, self.btc_signal.window_type = True, 'fixed'

        # actual
        actual_ts = self.ts_signal.transform_factors(method=method)
        actual_cs = self.cs_signal.transform_factors(method=method)
        actual_btc = self.btc_signal.transform_factors(method=method)

        # shape
        assert self.ts_signal.factors.shape == actual_ts.shape
        assert self.cs_signal.factors.shape == actual_cs.shape
        assert self.btc_signal.factors.shape == actual_btc.shape

        # values
        assert (actual_ts.dropna().std() <= 1.5).all()
        assert (actual_cs.dropna().std() <= 1.5).all()
        assert (actual_btc.dropna().std() < 1.5).all()
        assert np.allclose(actual_ts.loc[pd.IndexSlice[:, 'BTC'], :], actual_btc, equal_nan=True)

        # dtypes
        assert isinstance(actual_ts, pd.DataFrame)
        assert isinstance(actual_cs, pd.DataFrame)
        assert isinstance(actual_btc, pd.DataFrame)
        assert (actual_ts.dtypes == np.float64).all()
        assert (actual_cs.dtypes == np.float64).all()
        assert (actual_btc.dtypes == np.float64).all()

        # index
        assert (actual_ts.index == self.ts_signal.factors.index).all()
        assert (actual_cs.index == self.cs_signal.factors.index).all()
        assert (actual_btc.index == self.btc_signal.factors.index).all()

        # cols
        assert (actual_ts.columns == self.ts_signal.factors.columns).all()
        assert (actual_cs.columns == self.cs_signal.factors.columns).all()
        assert (actual_btc.columns == self.btc_signal.factors.columns).all()

    @pytest.mark.parametrize("strategy, bins",
                             [
                                 ('time_series', 5),
                                 ('cross_sectional', 5),
                                 ('time_series', 10),
                                 ('cross_sectional', 10),
                             ]
                             )
    def test_quantize(self, price_mom, btc_price_mom, spot_ret, btc_spot_ret, strategy, bins):
        """
        Test quantize method.
        """
        # get actual multiindex
        signal = Signal(price_mom, returns=spot_ret.close, strategy=strategy, bins=bins)
        signal.quantize = True
        actual = signal.quantize_factors()
        # get actual single index
        signal_btc = Signal(btc_price_mom, returns=btc_spot_ret.close, strategy=strategy, bins=bins)
        signal_btc.quantize = True
        actual_btc = signal_btc.quantize_factors()

        # shape
        assert signal.factors.shape == actual.shape
        assert self.btc_signal.factors.shape == actual_btc.shape

        # values
        assert (actual.nunique() == bins).all()
        if strategy == 'time_series':
            assert np.allclose(actual.loc[pd.IndexSlice[:, 'BTC'], :], actual_btc, equal_nan=True)
            assert (actual_btc.nunique() == bins).all()

        # dtypes
        assert isinstance(actual, pd.DataFrame)
        assert isinstance(actual_btc, pd.DataFrame)
        assert (actual.dtypes == np.float64).all()
        assert (actual_btc.dtypes == np.float64).all()

        # index
        assert (actual.index == signal.factors.index).all()
        assert (actual_btc.index == signal_btc.factors.index).all()

        # cols
        assert (actual.columns == signal.factors.columns).all()
        assert (actual_btc.columns == signal_btc.factors.columns).all()

    @pytest.mark.parametrize("strategy", ['time_series', 'cross_sectional'])
    def test_rank_factors(self, strategy) -> None:
        """
        Test rank_factors method.
        """
        self.ts_signal.rank = True
        self.btc_signal.rank = True

        # get actual ts
        actual_ts = self.ts_signal.rank_factors()
        actual_btc = self.btc_signal.rank_factors()

        # shape
        assert self.ts_signal.factors.shape == actual_ts.shape
        assert self.btc_signal.factors.shape == actual_btc.shape

        # values
        assert ((actual_ts.dropna() <= 1.0) & (actual_ts.dropna() >= 0.0)).all().all()
        assert ((actual_btc.dropna() <= 1.0) & (actual_btc.dropna() >= 0.0)).all().all()
        assert np.allclose(actual_ts.loc[pd.IndexSlice[:, 'BTC'], :], actual_btc, equal_nan=True)

        # dtypes
        assert isinstance(actual_ts, pd.DataFrame)
        assert isinstance(actual_btc, pd.DataFrame)
        assert (actual_ts.dtypes == np.float64).all()
        assert (actual_btc.dtypes == np.float64).all()

        # index
        assert (actual_ts.index == self.ts_signal.factors.index).all()
        assert (actual_btc.index == self.btc_signal.factors.index).all()

        # cols
        assert (actual_ts.columns == self.ts_signal.factors.columns).all()
        assert (actual_btc.columns == self.btc_signal.factors.columns).all()

        # get actual cs
        self.cs_signal.rank = True
        actual_cs = self.cs_signal.rank_factors()

        # shape
        assert self.cs_signal.factors.shape == actual_cs.shape

        # values
        assert (actual_cs.groupby(level=0).max().fillna(0) == actual_cs.groupby(level=0).count()).all().all()

        # dtypes
        assert isinstance(actual_cs, pd.DataFrame)
        assert (actual_cs.dtypes == np.float64).all()

        # cols
        assert (actual_ts.columns == self.ts_signal.factors.columns).all()
        assert (actual_cs.columns == self.cs_signal.factors.columns).all()
        assert (actual_btc.columns == self.btc_signal.factors.columns).all()

    @pytest.mark.parametrize("method", ['mean', 'median', 'min', 'max', 'sum', 'prod', 'value-weighted'])
    def test_combine_factors(self, method):
        """
        Test combine_factors method.
        """
        # get actual
        self.ts_signal.combine = True
        actual_ts = self.ts_signal.combine_factors(method=method)
        self.cs_signal.combine = True
        actual_cs = self.cs_signal.combine_factors(method=method)
        self.btc_signal.combine = True
        actual_btc = self.btc_signal.combine_factors(method=method)

        # shape
        assert actual_ts.shape[1] == 1
        assert actual_cs.shape[1] == 1
        assert actual_btc.shape[1] == 1

        # values
        assert np.allclose(actual_ts.loc[pd.IndexSlice[:, 'BTC'], :], actual_btc, equal_nan=True)

        # dtypes
        assert isinstance(actual_ts, pd.DataFrame)
        assert isinstance(actual_cs, pd.DataFrame)
        assert isinstance(actual_btc, pd.DataFrame)
        assert (actual_ts.dtypes == np.float64).all()
        assert (actual_cs.dtypes == np.float64).all()
        assert (actual_btc.dtypes == np.float64).all()

        # index
        assert (actual_ts.index == self.ts_signal.factors.index).all()
        assert (actual_cs.index == self.cs_signal.factors.index).all()
        assert (actual_btc.index == self.btc_signal.factors.index).all()

        # cols
        assert actual_ts.columns == ['combined_factor']
        assert actual_cs.columns == ['combined_factor']
        assert actual_btc.columns == ['combined_factor']

    @pytest.mark.parametrize("transformation", ['norm', 'logistic', 'adj_norm', 'percentile', 'min-max'])
    def test_factors_to_signals(self, transformation):
        """
        Test convert_to_signals method.
        """
        if transformation in ['min-max', 'percentile']:
            self.ts_signal.normalize_factors(method=transformation, centering=True)
            self.cs_signal.normalize_factors(method=transformation, centering=True)
            self.btc_signal.normalize_factors(method=transformation, centering=True)
        else:
            self.ts_signal.normalize_factors(method='z-score', centering=True)
            self.cs_signal.normalize_factors(method='z-score', centering=True)
            self.btc_signal.normalize_factors(method='z-score', centering=True)

        # get actual
        actual_ts = self.ts_signal.factors_to_signals(transformation=transformation)
        actual_cs = self.cs_signal.factors_to_signals(transformation=transformation)
        actual_btc = self.btc_signal.factors_to_signals(transformation=transformation)

        # shape
        assert self.ts_signal.factors.shape == actual_ts.shape
        assert self.cs_signal.factors.shape == actual_cs.shape
        assert self.btc_signal.factors.shape == actual_btc.shape

        # values
        assert ((actual_ts.dropna() >= -1) & (actual_ts.dropna() <= 1)).all().all()
        assert ((actual_cs.dropna() >= -1) & (actual_cs.dropna() <= 1)).all().all()
        assert ((actual_btc.dropna() >= -1) & (actual_btc.dropna() <= 1)).all().all()
        assert np.allclose(actual_ts.loc[pd.IndexSlice[:, 'BTC'], :], actual_btc, equal_nan=True)

        # dtypes
        assert isinstance(actual_ts, pd.DataFrame)
        assert isinstance(actual_cs, pd.DataFrame)
        assert isinstance(actual_btc, pd.DataFrame)
        assert (actual_ts.dtypes == np.float64).all()
        assert (actual_cs.dtypes == np.float64).all()
        assert (actual_btc.dtypes == np.float64).all()

        # index
        assert (actual_ts.index == self.ts_signal.factors.index).all()
        assert (actual_cs.index == self.cs_signal.factors.index).all()
        assert (actual_btc.index == self.btc_signal.factors.index).all()

        # cols
        assert (actual_ts.columns == self.ts_signal.factors.columns).all()
        assert (actual_cs.columns == self.cs_signal.factors.columns).all()
        assert (actual_btc.columns == self.btc_signal.factors.columns).all()

    @pytest.mark.parametrize("bins", [3, 5, 10])
    def test_quantiles_to_signals(self, bins) -> None:
        """
        Test quantiles_to_signals method.
        """
        # bins
        self.ts_signal.bins, self.ts_signal.quantize = bins, True
        self.cs_signal.bins, self.cs_signal.quantize = bins, True
        self.btc_signal.bins, self.btc_signal.quantize = bins, True

        # quantiles
        self.ts_signal.quantize_factors()
        self.cs_signal.quantize_factors()
        self.btc_signal.quantize_factors()

        # get actual
        actual_ts = self.ts_signal.quantiles_to_signals()
        actual_cs = self.cs_signal.quantiles_to_signals()
        actual_btc = self.btc_signal.quantiles_to_signals()

        # shape
        assert self.ts_signal.factors.shape == actual_ts.shape
        assert self.cs_signal.factors.shape == actual_cs.shape
        assert self.btc_signal.factors.shape == actual_btc.shape

        # values
        assert ((actual_ts.dropna() >= -1) & (actual_ts.dropna() <= 1)).all().all()
        assert ((actual_cs.dropna() >= -1) & (actual_cs.dropna() <= 1)).all().all()
        assert ((actual_btc.dropna() >= -1) & (actual_btc.dropna() <= 1)).all().all()
        assert np.allclose(actual_ts.loc[pd.IndexSlice[:, 'BTC'], :], actual_btc, equal_nan=True)

        # dtypes
        assert isinstance(actual_ts, pd.DataFrame)
        assert isinstance(actual_cs, pd.DataFrame)
        assert isinstance(actual_btc, pd.DataFrame)
        assert (actual_ts.dtypes == np.float64).all()
        assert (actual_cs.dtypes == np.float64).all()
        assert (actual_btc.dtypes == np.float64).all()

        # index
        assert (actual_ts.index == self.ts_signal.factors.index).all()
        assert (actual_cs.index == self.cs_signal.factors.index).all()
        assert (actual_btc.index == self.btc_signal.factors.index).all()

        # cols
        assert (actual_ts.columns == self.ts_signal.factors.columns).all()
        assert (actual_cs.columns == self.cs_signal.factors.columns).all()
        assert (actual_btc.columns == self.btc_signal.factors.columns).all()

    def test_ranks_to_signals(self) -> None:
        """
        Test ranks_to_signals method.
        """
        # rank
        self.ts_signal.rank = True
        self.cs_signal.rank = True
        self.btc_signal.rank = True

        # ranks
        self.ts_signal.rank_factors()
        self.cs_signal.rank_factors()
        self.btc_signal.rank_factors()

        # get actual
        actual_ts = self.ts_signal.ranks_to_signals()
        actual_cs = self.cs_signal.ranks_to_signals()
        actual_btc = self.btc_signal.ranks_to_signals()

        # shape
        assert self.ts_signal.factors.shape == actual_ts.shape
        assert self.cs_signal.factors.shape == actual_cs.shape
        assert self.btc_signal.factors.shape == actual_btc.shape

        # values
        assert ((actual_ts.dropna() >= -1) & (actual_ts.dropna() <= 1)).all().all()
        assert ((actual_cs.dropna() >= -1) & (actual_cs.dropna() <= 1)).all().all()
        assert ((actual_btc.dropna() >= -1) & (actual_btc.dropna() <= 1)).all().all()
        assert np.allclose(actual_ts.loc[pd.IndexSlice[:, 'BTC'], :], actual_btc, equal_nan=True)

        # dtypes
        assert isinstance(actual_ts, pd.DataFrame)
        assert isinstance(actual_cs, pd.DataFrame)
        assert isinstance(actual_btc, pd.DataFrame)
        assert (actual_ts.dtypes == np.float64).all()
        assert (actual_cs.dtypes == np.float64).all()
        assert (actual_btc.dtypes == np.float64).all()

        # index
        assert (actual_ts.index == self.ts_signal.factors.index).all()
        assert (actual_cs.index == self.cs_signal.factors.index).all()
        assert (actual_btc.index == self.btc_signal.factors.index).all()

        # cols
        assert (actual_ts.columns == self.ts_signal.factors.columns).all()
        assert (actual_cs.columns == self.cs_signal.factors.columns).all()
        assert (actual_btc.columns == self.btc_signal.factors.columns).all()

    @pytest.mark.parametrize("signal_thresh, n_factor",
                             [
                                 (0.6, None),
                                 (0.6, 5),
                                 (0, 10)
                             ]
                             )
    def test_discretize_signals(self, signal_thresh, n_factor):
        """
        Test discretize_signals method.
        """
        self.ts_signal.signal_thresh, self.ts_signal.n_factors = signal_thresh, n_factor
        self.cs_signal.signal_thresh, self.cs_signal.n_factors = signal_thresh, n_factor
        self.btc_signal.signal_thresh, self.btc_signal.n_factors = signal_thresh, n_factor

        # normalize
        self.ts_signal.normalize_factors(method='z-score', centering=True)
        self.cs_signal.normalize_factors(method='z-score', centering=True)
        self.btc_signal.normalize_factors(method='z-score', centering=True)

        # signals
        self.ts_signal.factors_to_signals()
        self.cs_signal.factors_to_signals()
        self.btc_signal.factors_to_signals()

        # get actual
        actual_ts = self.ts_signal.discretize_signals()
        actual_cs = self.cs_signal.discretize_signals()
        actual_btc = self.btc_signal.discretize_signals()

        # shape
        assert self.ts_signal.factors.shape == actual_ts.shape
        assert self.cs_signal.factors.shape == actual_cs.shape
        assert self.btc_signal.factors.shape == actual_btc.shape

        # values
        assert actual_ts.isin([-1, 0, 1]).all().all()
        assert actual_cs.isin([-1, 0, 1]).all().all()
        assert actual_btc.isin([-1, 0, 1]).all().all()

        # dtypes
        assert isinstance(actual_ts, pd.DataFrame)
        assert isinstance(actual_cs, pd.DataFrame)
        assert isinstance(actual_btc, pd.DataFrame)
        assert (actual_ts.dtypes == 'float64').all()
        assert (actual_cs.dtypes == 'float64').all()
        assert (actual_btc.dtypes == 'float64').all()

        # index
        assert (actual_ts.index == self.ts_signal.factors.index).all()
        assert (actual_cs.index == self.cs_signal.factors.index).all()
        assert (actual_btc.index == self.btc_signal.factors.index).all()

        # cols
        assert (actual_ts.columns == self.ts_signal.factors.columns).all()
        assert (actual_cs.columns == self.cs_signal.factors.columns).all()
        assert (actual_btc.columns == self.btc_signal.factors.columns).all()

    # TODO change direction params
    @pytest.mark.parametrize("direction", ['long', 'short', 'long-short'])
    def test_filter_direction(self, direction) -> None:
        """
        Test filter_direction method.
        """
        self.ts_signal.direction = direction
        self.cs_signal.direction = direction
        self.btc_signal.direction = direction

        # normalize
        self.ts_signal.normalize_factors(method='z-score', centering=True)
        self.cs_signal.normalize_factors(method='z-score', centering=True)
        self.btc_signal.normalize_factors(method='z-score', centering=True)

        # signals
        self.ts_signal.factors_to_signals()
        self.cs_signal.factors_to_signals()
        self.btc_signal.factors_to_signals()

        # get actual
        actual_ts = self.ts_signal.filter_direction()
        actual_cs = self.cs_signal.filter_direction()
        actual_btc = self.btc_signal.filter_direction()

        # shape
        assert self.ts_signal.factors.shape == actual_ts.shape
        assert self.cs_signal.factors.shape == actual_cs.shape
        assert self.btc_signal.factors.shape == actual_btc.shape

        # values
        if direction == 'long':
            assert (actual_ts.dropna() >= 0).all().all()
            assert (actual_cs.dropna() >= 0).all().all()
            assert (actual_btc.dropna() >= 0).all().all()
        elif direction == 'short':
            assert (actual_ts.dropna() <= 0).all().all()
            assert (actual_cs.dropna() <= 0).all().all()
            assert (actual_btc.dropna() <= 0).all().all()

        # dtypes
        assert isinstance(actual_ts, pd.DataFrame)
        assert isinstance(actual_cs, pd.DataFrame)
        assert isinstance(actual_btc, pd.DataFrame)
        assert (actual_ts.dtypes == 'float64').all()
        assert (actual_cs.dtypes == 'float64').all()
        assert (actual_btc.dtypes == 'float64').all()

        # index
        assert (actual_ts.index == self.ts_signal.factors.index).all()
        assert (actual_cs.index == self.cs_signal.factors.index).all()
        assert (actual_btc.index == self.btc_signal.factors.index).all()

        # cols
        assert (actual_ts.columns == self.ts_signal.factors.columns).all()
        assert (actual_cs.columns == self.cs_signal.factors.columns).all()
        assert (actual_btc.columns == self.btc_signal.factors.columns).all()

    @pytest.mark.parametrize("signal_type, lags",
                             [
                                 (None, None),
                                 (None, 1),
                                 ('signal', None),
                                 ('signal', 0),
                                 ('signal_quantiles', None),
                                 ('signal_quantiles', 0),
                                 ('signal_quantiles', 1),
                                 ('signal_ranks', None),
                                 ('signal_ranks', 0),
                                 ('signal_ranks', 2),
                             ]
                             )
    def test_compute_signals(self, signal_type, lags):
        """
        Test compute_signals method.
        """
        if signal_type == 'signal':
            self.ts_signal.normalize_factors(method='z-score', centering=True)
            self.cs_signal.normalize_factors(method='z-score', centering=True)
            self.btc_signal.normalize_factors(method='z-score', centering=True)

        elif signal_type == 'signal_quantiles':
            self.ts_signal.bins, self.ts_signal.quantize = 3, True
            self.cs_signal.bins, self.cs_signal.quantize = 3, True
            self.btc_signal.bins, self.btc_signal.quantize = 3, True
            self.ts_signal.quantize_factors()
            self.cs_signal.quantize_factors()
            self.btc_signal.quantize_factors()

        elif signal_type == 'signal_ranks':
            self.ts_signal.rank = True
            self.cs_signal.rank = True
            self.btc_signal.rank = True
            self.ts_signal.rank_factors()
            self.cs_signal.rank_factors()
            self.btc_signal.rank_factors()

        # actual
        actual_ts = self.ts_signal.compute_signals(signal_type=signal_type, lags=lags)
        actual_cs = self.cs_signal.compute_signals(signal_type=signal_type, lags=lags)
        actual_btc = self.btc_signal.compute_signals(signal_type=signal_type, lags=lags)

        # shape
        assert self.ts_signal.factors.shape == actual_ts.shape
        assert self.cs_signal.factors.shape == actual_cs.shape
        assert self.btc_signal.factors.shape == actual_btc.shape

        # values
        if signal_type is None:

            if lags is None:
                assert self.ts_signal.factors.equals(self.ts_signal.signals)
                assert self.cs_signal.factors.equals(self.cs_signal.signals)
                assert self.btc_signal.factors.equals(self.btc_signal.signals)
            else:
                assert self.ts_signal.factors.groupby(level=1).shift(lags).equals(self.ts_signal.signals)
                assert self.cs_signal.factors.groupby(level=1).shift(lags).equals(self.cs_signal.signals)
                assert self.btc_signal.factors.shift(lags).equals(self.btc_signal.signals)

        elif signal_type == 'signal':

            # signal bounds check
            assert ((actual_ts.dropna() >= -1) & (actual_ts.dropna() <= 1)).all().all()
            assert ((actual_cs.dropna() >= -1) & (actual_cs.dropna() <= 1)).all().all()
            assert ((actual_btc.dropna() >= -1) & (actual_btc.dropna() <= 1)).all().all()
            assert np.allclose(actual_ts.loc[pd.IndexSlice[:, 'BTC'], :], actual_btc, equal_nan=True)

            # signal correlations check
            factor_mean = self.ts_signal.factors.unstack().mean(axis=1)
            signal_mean = self.ts_signal.signals.unstack().mean(axis=1)
            factor_tickers = self.cs_signal.factors.unstack().iloc[-1].sort_values().index.to_list()
            cs_tickers = self.cs_signal.signals.unstack().iloc[-1].sort_values().index.to_list()

            if lags is None or lags == 0:
                assert pd.concat([factor_mean, signal_mean], axis=1).corr(method='spearman').iloc[0, 1] > 0.5
                assert factor_tickers == cs_tickers
            else:
                assert pd.concat([factor_mean.shift(lags), signal_mean],
                                 axis=1).corr(method='spearman').iloc[0, 1] > 0.5
                assert factor_tickers == cs_tickers

        elif signal_type == 'signal_quantiles':

            # signal bounds check
            assert actual_ts.dropna().isin([-1.0, 0.0, 1.0]).all().all()
            assert actual_cs.dropna().isin([-1.0, 0.0, 1.0]).all().all()
            assert actual_btc.dropna().isin([-1.0, 0.0, 1.0]).all().all()
            assert np.allclose(actual_ts.loc[pd.IndexSlice[:, 'BTC'], :], actual_btc, equal_nan=True)

        elif signal_type == 'signal_ranks':

            # signal bounds check
            assert ((actual_ts.dropna() >= -1) & (actual_ts.dropna() <= 1)).all().all()
            assert ((actual_cs.dropna() >= -1) & (actual_cs.dropna() <= 1)).all().all()
            assert ((actual_btc.dropna() >= -1) & (actual_btc.dropna() <= 1)).all().all()
            assert np.allclose(actual_ts.loc[pd.IndexSlice[:, 'BTC'], :], actual_btc, equal_nan=True)

        # dtypes
        assert isinstance(actual_ts, pd.DataFrame)
        assert isinstance(actual_cs, pd.DataFrame)
        assert isinstance(actual_btc, pd.DataFrame)
        assert (actual_ts.dtypes == np.float64).all()
        assert (actual_cs.dtypes == np.float64).all()
        assert (actual_btc.dtypes == np.float64).all()

        # index
        assert (actual_ts.index == self.ts_signal.factors.index).all()
        assert (actual_cs.index == self.cs_signal.factors.index).all()
        assert (actual_btc.index == self.btc_signal.factors.index).all()

        # cols
        assert (actual_ts.columns == self.ts_signal.factors.columns).all()
        assert (actual_cs.columns == self.cs_signal.factors.columns).all()
        assert (actual_btc.columns == self.btc_signal.factors.columns).all()

    def test_compute_dual_signals(self):
        """
        Test compute_dual_signals method.
        """
        # get actual

        actual = self.dual_signal.compute_dual_signals()

        # shape
        assert self.dual_signal.factors.shape == actual.shape

        # values
        assert ((actual.dropna() >= -1) & (actual.dropna() <= 1)).all().all()

        # dtypes
        assert isinstance(actual, pd.DataFrame)
        assert (actual.dtypes == np.float64).all()

        # index
        assert (actual.index == self.dual_signal.factors.index).all()

        # cols
        assert (actual.columns == self.dual_signal.factors.columns).all()

    @pytest.mark.parametrize('rebal_freq',
                             [None, 5, 7, 'monday', 'friday']
                             )
    def test_rebalance_signals(self, rebal_freq):
        """
        Test rebalance_signals method.
        """
        # get actual
        signals = self.ts_signal.compute_signals().unstack()
        rebal_signals = self.ts_signal.rebalance_signals(rebal_freq=rebal_freq).unstack()

        # shape
        assert signals.shape == rebal_signals.shape

        # values
        assert ((rebal_signals.dropna() >= -1) &
                (rebal_signals.dropna() <= 1)).all().all()

        if isinstance(rebal_freq, int):
            assert np.allclose((rebal_signals.diff().abs().dropna() == 0).sum() /
                               rebal_signals.dropna().shape[0], 1 - 1 / rebal_freq, rtol=0.05)
        elif rebal_freq is None:
            assert signals.equals(rebal_signals)
        else:
            assert np.allclose((rebal_signals.diff().abs().dropna() == 0).sum() /
                              rebal_signals.dropna().shape[0], 1 - 1 / 7, rtol=0.05)

        # dtypes
        assert isinstance(rebal_signals, pd.DataFrame)
        assert (rebal_signals.dtypes == np.float64).all()

        # index
        assert (signals.index == rebal_signals.index).all()

        # cols
        assert (signals.columns == rebal_signals.columns).all()

    @pytest.mark.parametrize('t_cost', [None, 0.001])
    def test_compute_tcosts(self, t_cost):
        """
        Test compute_tcosts method.
        """
        # get actual
        self.ts_signal.compute_signals()
        self.ts_signal.rebalance_signals()
        actual = self.ts_signal.compute_tcosts(t_cost=t_cost)

        # shape
        assert self.ts_signal.signals.shape == actual.shape

        # values
        if t_cost is not None:
            assert (actual.dropna() <= t_cost * 2).all().all()
        else:
            assert (actual == 0).all().all()

        # dtypes
        assert isinstance(actual, pd.DataFrame)
        assert (actual.dtypes == np.float64).all()

        # index
        assert (actual.index == self.ts_signal.signals.index).all()

        # cols
        assert (actual.columns == self.ts_signal.signals.columns).all()

    def test_compute_gross_returns(self):
        """
        Test compute_gross_returns method.
        """
        # get actual
        self.ts_signal.compute_signals()
        self.cs_signal.compute_signals()
        self.btc_signal.compute_signals()

        self.ts_signal.compute_gross_returns()
        self.cs_signal.compute_gross_returns()
        self.btc_signal.compute_gross_returns()

        # shape
        assert self.ts_signal.signal_rets.shape[1] == self.ts_signal.signals.shape[1]
        assert self.cs_signal.signal_rets.shape[1] == self.cs_signal.signals.shape[1]
        assert self.btc_signal.signal_rets.shape[1] == self.btc_signal.signals.shape[1]

        # dtypes
        assert isinstance(self.ts_signal.signal_rets, pd.DataFrame)
        assert isinstance(self.cs_signal.signal_rets, pd.DataFrame)
        assert isinstance(self.btc_signal.signal_rets, pd.DataFrame)
        assert (self.ts_signal.signal_rets.dtypes == np.float64).all()
        assert (self.cs_signal.signal_rets.dtypes == np.float64).all()
        assert (self.btc_signal.signal_rets.dtypes == np.float64).all()

        # values
        assert np.allclose(self.ts_signal.signal_rets.loc[pd.IndexSlice[:, 'BTC'], :].dropna(),
                           self.btc_signal.signal_rets.dropna())

        # index
        assert (set(self.ts_signal.signal_rets.index.droplevel(0).unique()) ==
                set(self.ts_signal.signals.index.droplevel(0).unique()))
        assert (set(self.cs_signal.signal_rets.index.droplevel(0).unique()) ==
                set(self.cs_signal.signals.index.droplevel(0).unique()))

        # cols
        assert (self.ts_signal.signal_rets.columns == self.ts_signal.signals.columns).all()
        assert (self.cs_signal.signal_rets.columns == self.cs_signal.signals.columns).all()
        assert (self.btc_signal.signal_rets.columns == self.btc_signal.signals.columns).all()

    @pytest.mark.parametrize('t_cost', [None, 0.001])
    def test_compute_net_returns(self, t_cost):
        """
        Test compute_net_returns method.
        """
        # get actual
        self.ts_signal.compute_signals()
        self.cs_signal.compute_signals()
        self.btc_signal.compute_signals()

        ts_signal_t_cost_df = self.ts_signal.compute_tcosts(t_cost=t_cost)
        cs_signal_t_cost_df = self.cs_signal.compute_tcosts(t_cost=t_cost)
        btc_signal_t_cost_df = self.btc_signal.compute_tcosts(t_cost=t_cost)

        ts_signal_gross_rets = self.ts_signal.compute_gross_returns()
        cs_signal_gross_rets = self.cs_signal.compute_gross_returns()
        btc_signal_gross_rets = self.btc_signal.compute_gross_returns()

        self.ts_signal.compute_net_returns(ts_signal_t_cost_df).sort_index()
        self.cs_signal.compute_net_returns(cs_signal_t_cost_df).sort_index()
        self.btc_signal.compute_net_returns(btc_signal_t_cost_df).sort_index()

        # shape
        assert self.ts_signal.signal_rets.shape[1] == self.ts_signal.signals.shape[1]
        assert self.cs_signal.signal_rets.shape[1] == self.cs_signal.signals.shape[1]
        assert self.btc_signal.signal_rets.shape[1] == self.btc_signal.signals.shape[1]

        # dtypes
        assert isinstance(self.ts_signal.signal_rets, pd.DataFrame)
        assert isinstance(self.cs_signal.signal_rets, pd.DataFrame)
        assert isinstance(self.btc_signal.signal_rets, pd.DataFrame)
        assert (self.ts_signal.signal_rets.dtypes == np.float64).all()
        assert (self.cs_signal.signal_rets.dtypes == np.float64).all()
        assert (self.btc_signal.signal_rets.dtypes == np.float64).all()

        # values
        assert np.allclose(self.ts_signal.signal_rets.loc[pd.IndexSlice[:, 'BTC'], :].dropna(),
                           self.btc_signal.signal_rets.dropna())
        if t_cost is None:
            assert (self.ts_signal.signal_rets.unstack().dropna().iloc[-300:].
                    equals(ts_signal_gross_rets.unstack().dropna().iloc[-300:]))
            assert (self.cs_signal.signal_rets.unstack().dropna().iloc[-300:].
                    equals(cs_signal_gross_rets.unstack().dropna().iloc[-300:]))
            assert (self.btc_signal.signal_rets.unstack().dropna().iloc[-300:].
                    equals(btc_signal_gross_rets.unstack().dropna().iloc[-300:]))

        # index
        assert (set(self.ts_signal.signal_rets.index.droplevel(0).unique()) ==
                set(self.ts_signal.signals.index.droplevel(0).unique()))
        assert (set(self.cs_signal.signal_rets.index.droplevel(0).unique()) ==
                set(self.cs_signal.signals.index.droplevel(0).unique()))

        # cols
        assert (self.ts_signal.signal_rets.columns == self.ts_signal.signals.columns).all()
        assert (self.cs_signal.signal_rets.columns == self.cs_signal.signals.columns).all()
        assert (self.btc_signal.signal_rets.columns == self.btc_signal.signals.columns).all()

    def test_compute_signal_returns(self):
        """
        Test compute_signal_returns method.
        """
        # get actual
        self.ts_signal.compute_signal_returns()
        self.cs_signal.compute_signal_returns()
        self.btc_signal.compute_signal_returns()

        # shape
        assert self.ts_signal.signal_rets.shape[1] == self.ts_signal.signals.shape[1]
        assert self.cs_signal.signal_rets.shape[1] == self.cs_signal.signals.shape[1]
        assert self.btc_signal.signal_rets.shape[1] == self.btc_signal.signals.shape[1]

        # dtypes
        assert isinstance(self.ts_signal.signal_rets, pd.DataFrame)
        assert isinstance(self.cs_signal.signal_rets, pd.DataFrame)
        assert isinstance(self.btc_signal.signal_rets, pd.DataFrame)
        assert (self.ts_signal.signal_rets.dtypes == np.float64).all()
        assert (self.cs_signal.signal_rets.dtypes == np.float64).all()
        assert (self.btc_signal.signal_rets.dtypes == np.float64).all()

        # values
        assert np.allclose(self.ts_signal.signal_rets.loc[pd.IndexSlice[:, 'BTC'], :].dropna(),
                           self.btc_signal.signal_rets.dropna())

        # index
        assert (set(self.ts_signal.signal_rets.index.droplevel(0).unique()) ==
                set(self.ts_signal.signals.index.droplevel(0).unique()))
        assert (set(self.cs_signal.signal_rets.index.droplevel(0).unique()) ==
                set(self.cs_signal.signals.index.droplevel(0).unique()))

        # cols
        assert (self.ts_signal.signal_rets.columns == self.ts_signal.signals.columns).all()
        assert (self.cs_signal.signal_rets.columns == self.cs_signal.signals.columns).all()
        assert (self.btc_signal.signal_rets.columns == self.btc_signal.signals.columns).all()

    @pytest.mark.parametrize("method", ['sign', 'std', 'skew', 'range'])
    def test_signal_dispersion(self, method):
        """
        Test signal_dispersion method.
        """
        # get actual
        self.ts_signal.compute_signals()
        self.ts_signal.signal_dispersion(method)

        # shape
        assert self.ts_signal.signals.unstack().shape[0] == self.ts_signal.signal_disp.shape[0]

        # dtypes
        assert isinstance(self.ts_signal.signal_disp, pd.DataFrame)
        assert (self.ts_signal.signal_disp.dtypes == np.float64).all()

        # index
        assert (self.ts_signal.signal_disp.index == self.ts_signal.signals.unstack().index).all()

        # cols
        assert (self.ts_signal.signal_disp.columns == self.ts_signal.signals.columns).all()

    def test_signal_correlation(self) -> None:
        """
        Test signal_correlation method.
        """
        # get actual
        self.ts_signal.compute_signals()
        self.ts_signal.signal_correlation()

        # shape
        assert self.ts_signal.signal_corr.shape == (self.ts_signal.signals.shape[1], self.ts_signal.signals.shape[1])

        # values
        assert ((self.ts_signal.signal_corr <= 1) & (self.ts_signal.signal_corr >= 0)).all().all()

        # dtypes
        assert isinstance(self.ts_signal.signal_corr, pd.DataFrame)
        assert (self.ts_signal.signal_corr.dtypes == np.float64).all()

        # index
        assert (self.ts_signal.signal_corr.index == self.ts_signal.signals.columns).all()

        # cols
        assert (self.ts_signal.signal_corr.columns == self.ts_signal.signals.columns).all()
