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
    btc_spot_ret = spot_ret.loc[pd.IndexSlice[:, 'BTC'], :].droplevel(1)

    return btc_spot_ret


@pytest.fixture
def price_mom(spot_prices):
    """
    Fixture for crypto price momentum.
    """
    # compute price mom
    price_mom = Trend(spot_prices).price_mom()

    return price_mom


@pytest.fixture
def btc_price_mom(price_mom):
    """
    Fixture for BTC price momentum.
    """
    # compute btc price mom
    btc_price_mom = price_mom.loc[pd.IndexSlice[:, 'BTC'], :].droplevel(1)

    return btc_price_mom


class TestSignal:
    """
    Test class for Transform.
    """
    @pytest.fixture(autouse=True)
    def signal_setup_default(self, price_mom, spot_ret):
        self.signal_instance = Signal(spot_ret.close, price_mom)

    @pytest.fixture(autouse=True)
    def dual_signal_default(self, price_mom, spot_ret):
        self.dual_signal_instance = Signal(spot_ret.close, price_mom, strategy='dual_ls')

    @pytest.fixture(autouse=True)
    def transform_setup_btc(self, btc_price_mom, btc_spot_ret):
        self.btc_signal_instance = Signal(btc_spot_ret.close, btc_price_mom)

    def test_initialization(self) -> None:
        """
        Test initialization.
        """
        # data type
        assert isinstance(self.signal_instance, Signal)
        assert isinstance(self.btc_signal_instance, Signal)
        assert isinstance(self.signal_instance.factors, pd.DataFrame)
        assert isinstance(self.btc_signal_instance.factors, pd.DataFrame)
        assert isinstance(self.signal_instance.ret, pd.DataFrame)
        assert isinstance(self.btc_signal_instance.ret, pd.DataFrame)

    def test_normalize(self):
        """
        Test normalize method.
        """
        # get actual
        actual = self.signal_instance.normalize()
        actual_btc = self.btc_signal_instance.normalize()

        # shape
        assert self.signal_instance.factors.shape == actual.shape
        assert self.btc_signal_instance.factors.shape == actual_btc.shape
        assert self.signal_instance.ret.shape == self.signal_instance.norm_ret.shape
        # values
        assert (actual.dropna().std() <= 1.5).all()
        assert (actual_btc.dropna().std() < 1.5).all()
        assert np.allclose(actual.loc[pd.IndexSlice[:, 'BTC'], :], actual_btc, equal_nan=True)
        # dtypes
        assert isinstance(actual, pd.DataFrame)
        assert isinstance(actual_btc, pd.DataFrame)
        assert (actual.dtypes == np.float64).all()
        assert (actual_btc.dtypes == np.float64).all()
        # index
        assert (actual.index == self.signal_instance.factors.index).all()
        assert (actual_btc.index == self.btc_signal_instance.factors.index).all()
        assert (self.signal_instance.ret.index == self.signal_instance.norm_ret.index).all()
        assert (self.btc_signal_instance.ret.index == self.btc_signal_instance.norm_ret.index).all()
        # cols
        assert (actual.columns == self.signal_instance.factors.columns).all()
        assert (actual_btc.columns == self.btc_signal_instance.factors.columns).all()

    @pytest.mark.parametrize("strategy, factor_bins, ret_bins",
                             [
                                 ('ts_ls', 5, 3),
                                 ('cs_ls', 5, 3),
                                 ('ts_ls', 10, 5),
                                 ('cs_ls', 10, 5),
                             ]
                             )
    def test_quantize(self, price_mom, btc_price_mom, spot_ret, btc_spot_ret, strategy, factor_bins, ret_bins):
        """
        Test quantize method.
        """
        # get actual multiindex
        signal = Signal(spot_ret.close, price_mom, strategy=strategy, factor_bins=factor_bins, ret_bins=ret_bins)
        actual = signal.quantize()

        # shape
        assert self.signal_instance.factors.shape == actual.shape
        assert self.signal_instance.ret.shape == signal.ret_quantiles.shape
        # values
        assert (actual.nunique() == factor_bins).all()
        assert (signal.ret_quantiles.nunique() == ret_bins).all()
        # dtypes
        assert isinstance(actual, pd.DataFrame)
        assert (actual.dtypes == np.float64).all()
        # index
        assert (actual.index == signal.factor_quantiles.index).all()
        # cols
        assert (actual.columns == signal.factor_quantiles.columns).all()

        # get actual single index
        signal_btc = Signal(btc_spot_ret.close, btc_price_mom, strategy=strategy, factor_bins=factor_bins,
                            ret_bins=ret_bins)
        actual_btc = signal_btc.quantize()
        # shape
        assert self.btc_signal_instance.factors.shape == actual_btc.shape
        assert self.btc_signal_instance.ret.shape == signal_btc.ret_quantiles.shape
        # values
        if strategy == 'ts_ls':
            assert np.allclose(actual.loc[pd.IndexSlice[:, 'BTC'], :], actual_btc, equal_nan=True)
            assert (actual_btc.nunique() == factor_bins).all()
            assert (signal_btc.ret_quantiles.nunique() == ret_bins).all()
        # dtypes
        assert isinstance(actual_btc, pd.DataFrame)
        assert (actual_btc.dtypes == np.float64).all()
        # index
        assert (actual_btc.index == signal_btc.factor_quantiles.index).all()
        # cols
        assert (actual_btc.columns == signal_btc.factor_quantiles.columns).all()

    @pytest.mark.parametrize("pdf", ['norm', 'percentile', 'min-max', 'logistic', 'adj_norm'])
    def test_convert_to_signals(self, pdf):
        """
        Test convert_to_signals method.
        """
        # get actual
        actual = self.signal_instance.convert_to_signals(pdf=pdf)
        actual_btc = self.btc_signal_instance.convert_to_signals(pdf=pdf)

        # shape
        assert self.signal_instance.factors.shape == actual.shape
        assert self.btc_signal_instance.factors.shape == actual_btc.shape
        # values
        assert ((actual.dropna() >= -1) & (actual.dropna() <= 1)).all().all()
        assert ((actual_btc.dropna() >= -1) & (actual_btc.dropna() <= 1)).all().all()
        # dtypes
        assert isinstance(actual, pd.DataFrame)
        assert isinstance(actual_btc, pd.DataFrame)
        assert (actual.dtypes == np.float64).all()
        assert (actual_btc.dtypes == np.float64).all()
        # index
        assert (actual.index == self.signal_instance.factors.index).all()
        assert (actual_btc.index == self.btc_signal_instance.factors.index).all()
        # cols
        assert (actual.columns == self.signal_instance.factors.columns).all()
        assert (actual_btc.columns == self.btc_signal_instance.factors.columns).all()

    def test_discretize_signals(self):
        """
        Test discretize_signals method.
        """
        # get actual
        actual = self.signal_instance.discretize_signals()
        actual_btc = self.btc_signal_instance.discretize_signals()

        # shape
        assert self.signal_instance.factors.shape == actual.shape
        assert self.btc_signal_instance.factors.shape == actual_btc.shape
        # dtypes
        assert (actual.dtypes == np.float64).all()
        assert (actual_btc.dtypes == np.float64).all()
        # index
        assert isinstance(actual, pd.DataFrame)
        assert isinstance(actual_btc, pd.DataFrame)
        assert (actual.index == self.signal_instance.factors.index).all()
        assert (actual_btc.index == self.btc_signal_instance.factors.index).all()
        # cols
        assert (actual.columns == self.signal_instance.factors.columns).all()
        assert (actual_btc.columns == self.btc_signal_instance.factors.columns).all()
        # values
        assert actual.isin([-1, 0, 1]).all().all()
        assert actual_btc.isin([-1, 0, 1]).all().all()
        signals = self.signal_instance.convert_to_signals()
        signals_btc = self.btc_signal_instance.convert_to_signals()
        assert (actual.corrwith(signals) > 0.5).all()
        assert (actual_btc.corrwith(signals_btc) > 0.5).all()

    @pytest.mark.parametrize("strategy, factor_bins",
                             [
                                 ('ts_ls', 5),
                                 ('cs_ls', 5),
                                 ('ts_ls', 10),
                                 ('cs_ls', 10),
                             ]
                             )
    def test_signal_to_quantiles(self, price_mom, btc_price_mom, spot_ret, btc_spot_ret, strategy, factor_bins):
        """
        Test quantize method.
        """
        # get actual multiindex
        actual = Signal(spot_ret.close, price_mom, strategy=strategy, factor_bins=factor_bins).signals_to_quantiles()

        # shape
        assert self.signal_instance.factors.shape == actual.shape
        # values
        assert (actual.nunique() == factor_bins).all()
        assert ((actual.dropna() >= -1) & (actual.dropna() <= 1)).all().all()
        assert actual.max().max() == 1.0
        assert actual.min().min() == -1.0
        # dtypes
        assert isinstance(actual, pd.DataFrame)
        assert (actual.dtypes == np.float64).all()
        # index
        assert (actual.index == self.signal_instance.factors.index).all()
        # cols
        assert (actual.columns == self.signal_instance.factors.columns).all()

        # get actual single index
        if strategy == 'ts_ls':
            actual_btc = Signal(btc_spot_ret.close, btc_price_mom, strategy='ts_ls',
                                factor_bins=factor_bins).signals_to_quantiles()
            # shape
            assert self.btc_signal_instance.factors.shape == actual_btc.shape
            # values
            assert (actual_btc.nunique() == factor_bins).all()
            assert ((actual_btc.dropna() >= -1) & (actual_btc.dropna() <= 1)).all().all()
            assert actual_btc.max().max() == 1.0
            assert actual_btc.min().min() == -1.0
            assert np.allclose(actual.loc[pd.IndexSlice[:, 'BTC'], :], actual_btc, equal_nan=True)
            # dtypes
            assert isinstance(actual_btc, pd.DataFrame)
            assert (actual_btc.dtypes == np.float64).all()
            # index
            assert (actual_btc.index == self.btc_signal_instance.factors.index).all()
            # cols
            assert (actual_btc.columns == self.btc_signal_instance.factors.columns).all()

    @pytest.mark.parametrize("n_factors", [5, 10, 20])
    def test_signal_rank(self, price_mom, spot_ret, n_factors):
        """
        Test signals_rank method.
        """
        # get actual multiindex
        actual = Signal(spot_ret.close, price_mom, strategy='cs_ls', n_factors=n_factors).signals_to_rank()

        # shape
        assert self.signal_instance.factors.shape[1] == actual.shape[1]
        # values
        assert ((actual[actual != 0].groupby(level=0).count() == n_factors * 2).sum() /
                actual.unstack().shape[0] > 0.95).all()  # n of signals
        assert ((actual[actual != 0].groupby(level=0).sum() == 0).sum() /
                actual.unstack().shape[0] > 0.95).all()  # sum of signals
        assert ((actual[actual != 0].groupby(level=0).max() == 1).sum() /
                actual.unstack().shape[0] > 0.95).all()
        assert ((actual[actual != 0].groupby(level=0).min() == -1).sum() /
        actual.unstack().shape[0] > 0.95).all()  # min of signals
        df = pd.concat([price_mom, actual], axis=1)
        assert df[df != 0].dropna().corr('spearman').iloc[0, 1] > 0.5
        # dtypes
        assert isinstance(actual, pd.DataFrame)
        assert (actual.dtypes == np.int64).all()
        # cols
        assert (actual.columns == self.signal_instance.factors.columns).all()

    @pytest.mark.parametrize("signal_type, lags",
                             [
                                ('signal', None),
                                ('signal', 1),
                                ('disc_signal', 2),
                                ('signal_quantiles', 0),
                                ('signal_quantiles', 1)
                             ]
                             )
    def test_compute_signals(self, signal_type, lags):
        """
        Test compute_signals method.
        """
        # get actual
        actual = self.signal_instance.compute_signals(signal_type=signal_type, lags=lags)
        actual_long = Signal(self.signal_instance.ret, self.signal_instance.factors, strategy='ts_l').\
            compute_signals(signal_type=signal_type, lags=lags)
        actual_short = Signal(self.signal_instance.ret, self.signal_instance.factors, strategy='ts_s').\
            compute_signals(signal_type=signal_type, lags=lags)

        # shape
        assert self.signal_instance.factors.shape == actual.shape
        assert self.signal_instance.factors.shape == actual_long.shape
        assert self.signal_instance.factors.shape == actual_short.shape

        # values
        assert ((actual.dropna() >= -1) & (actual.dropna() <= 1)).all().all()
        assert ((actual_long.dropna() >= 0) & (actual_long.dropna() <= 1)).all().all()
        assert ((actual_short.dropna() >= -1) & (actual_short.dropna() <= 0)).all().all()

        # dtypes
        assert isinstance(actual, pd.DataFrame)
        assert isinstance(actual_long, pd.DataFrame)
        assert isinstance(actual_short, pd.DataFrame)
        assert (actual.dtypes == np.float64).all()
        assert (actual_long.dtypes == np.float64).all()
        assert (actual_short.dtypes == np.float64).all()
        # index
        assert (actual.index == self.signal_instance.factors.index).all()
        assert (actual_long.index == self.signal_instance.factors.index).all()
        assert (actual_short.index == self.signal_instance.factors.index).all()
        # cols
        assert (actual.columns == self.signal_instance.factors.columns).all()
        assert (actual_long.columns == self.signal_instance.factors.columns).all()
        assert (actual_short.columns == self.signal_instance.factors.columns).all()

        # get actual single index
        actual_btc = self.btc_signal_instance.compute_signals(signal_type=signal_type, lags=lags)
        actual_btc_long = Signal(self.btc_signal_instance.ret, self.btc_signal_instance.factors, strategy='ts_l').\
            compute_signals(signal_type=signal_type, lags=lags)
        actual_btc_short = Signal(self.btc_signal_instance.ret, self.btc_signal_instance.factors, strategy='ts_s').\
            compute_signals(signal_type=signal_type, lags=lags)

        # shape
        assert self.btc_signal_instance.factors.shape == actual_btc.shape
        assert self.btc_signal_instance.factors.shape == actual_btc_long.shape
        assert self.btc_signal_instance.factors.shape == actual_btc_short.shape
        # values
        assert ((actual_btc.dropna() >= -1) & (actual_btc.dropna() <= 1)).all().all()
        assert ((actual_btc_long.dropna() >= 0) & (actual_btc_long.dropna() <= 1)).all().all()
        assert ((actual_btc_short.dropna() >= -1) & (actual_btc_short.dropna() <= 0)).all().all()
        assert np.allclose(actual.loc[pd.IndexSlice[:, 'BTC'], :].dropna(), actual_btc.dropna())
        assert np.allclose(actual_long.loc[pd.IndexSlice[:, 'BTC'], :].dropna(), actual_btc_long.dropna())
        assert np.allclose(actual_short.loc[pd.IndexSlice[:, 'BTC'], :].dropna(), actual_btc_short.dropna())
        # dtypes
        assert isinstance(actual_btc, pd.DataFrame)
        assert isinstance(actual_btc_long, pd.DataFrame)
        assert isinstance(actual_btc_short, pd.DataFrame)
        assert (actual_btc.dtypes == np.float64).all()
        assert (actual_btc_long.dtypes == np.float64).all()
        assert (actual_btc_short.dtypes == np.float64).all()
        # index
        assert (actual_btc.index == self.btc_signal_instance.factors.index).all()
        assert (actual_btc_long.index == self.btc_signal_instance.factors.index).all()
        assert (actual_btc_short.index == self.btc_signal_instance.factors.index).all()
        # cols
        assert (actual_btc.columns == self.btc_signal_instance.factors.columns).all()
        assert (actual_btc_long.columns == self.btc_signal_instance.factors.columns).all()
        assert (actual_btc_short.columns == self.btc_signal_instance.factors.columns).all()

    def test_compute_dual_signals(self):
        """
        Test compute_dual_signals method.
        """
        # get actual
        actual = self.dual_signal_instance.compute_dual_signals()

        # shape
        assert self.dual_signal_instance.factors.shape == actual.shape
        # values
        assert ((actual.dropna() >= -1) & (actual.dropna() <= 1)).all().all()
        # dtypes
        assert isinstance(actual, pd.DataFrame)
        assert (actual.dtypes == np.float64).all()
        # index
        assert (actual.index == self.dual_signal_instance.factors.index).all()
        # cols
        assert (actual.columns == self.dual_signal_instance.factors.columns).all()

    def test_compute_signal_returns(self):
        """
        Test compute_signal_returns method.
        """
        # get actual
        self.signal_instance.compute_signal_returns()
        self.btc_signal_instance.compute_signal_returns()

        # shape
        assert self.signal_instance.signal_rets.shape == self.signal_instance.signals.dropna().shape
        assert self.btc_signal_instance.signal_rets.shape == self.btc_signal_instance.signals.dropna().shape
        # dtypes
        assert isinstance(self.signal_instance.signal_rets, pd.DataFrame)
        assert isinstance(self.btc_signal_instance.signal_rets, pd.DataFrame)
        assert (self.signal_instance.signal_rets.dtypes == np.float64).all()
        assert (self.btc_signal_instance.signal_rets.dtypes == np.float64).all()
        # values
        assert np.allclose(self.signal_instance.signal_rets.loc[pd.IndexSlice[:, 'BTC'], :].dropna(),
                           self.btc_signal_instance.signal_rets.dropna())
        # index
        assert (self.signal_instance.signal_rets.index == self.signal_instance.signals.dropna().index).all()
        assert (self.btc_signal_instance.signal_rets.index == self.btc_signal_instance.signals.dropna().index).all()
        # cols
        assert (self.signal_instance.signal_rets.columns == self.signal_instance.signals.dropna().columns).all()
        assert (self.btc_signal_instance.signal_rets.columns == self.btc_signal_instance.signals.dropna().columns).all()

    @pytest.mark.parametrize("method", ['sign', 'std', 'skew', 'range'])
    def test_signal_dispersion(self, method):
        """
        Test signal_dispersion method.
        """
        # get actual
        self.signal_instance.compute_signals()
        actual = self.signal_instance.signal_dispersion(method)

        # shape
        assert self.signal_instance.factors.unstack().shape[0] == actual.shape[0]
        assert actual.shape[1] == 1
        # dtypes
        assert isinstance(actual, pd.DataFrame)
        assert (actual.dtypes == np.float64).all()
        # index
        assert (actual.index == self.signal_instance.factors.unstack().index).all()
        # cols
        assert (actual.columns == self.signal_instance.factors.columns).all()
