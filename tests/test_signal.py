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
    btc_spot_ret = spot_ret.loc[:, 'BTC', :]

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
    btc_price_mom = price_mom.loc[:, 'BTC', :]

    return btc_price_mom


class TestSignal:
    """
    Test class for Transform.
    """
    @pytest.fixture(autouse=True)
    def ts_signal_default(self, spot_ret, price_mom):
        self.ts_signal = Signal(spot_ret.close, price_mom, disc_thresh=0.5)

    @pytest.fixture(autouse=True)
    def cs_signal_default(self, spot_ret, price_mom):
        self.cs_signal = Signal(spot_ret.close, price_mom, strategy='cs_ls', disc_thresh=0.8)

    @pytest.fixture(autouse=True)
    def dual_signal_default(self, spot_ret, price_mom):
        self.dual_signal = Signal(spot_ret.close, price_mom, strategy='dual_ls')

    @pytest.fixture(autouse=True)
    def btc_ts_signal_default(self, btc_spot_ret, btc_price_mom):
        self.btc_signal = Signal(btc_spot_ret.close, btc_price_mom, disc_thresh=0.5)

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

        # shape
        assert self.ts_signal.factors.shape[1] == self.ts_signal.returns.shape[1]
        assert self.cs_signal.factors.shape[1] == self.cs_signal.returns.shape[1]
        assert self.dual_signal.factors.shape[1] == self.dual_signal.returns.shape[1]
        assert self.btc_signal.factors.shape[1] == self.btc_signal.returns.shape[1]

    def test_normalize(self):
        """
        Test normalize method.
        """
        # get actual
        actual_ts = self.ts_signal.normalize()
        actual_cs = self.cs_signal.normalize()
        actual_btc = self.btc_signal.normalize()

        # shape
        assert self.ts_signal.factors.shape == actual_ts.shape
        assert self.cs_signal.factors.shape == actual_cs.shape
        assert self.btc_signal.factors.shape == actual_btc.shape
        assert self.ts_signal.returns.shape == self.ts_signal.norm_ret.shape
        assert self.cs_signal.returns.shape == self.cs_signal.norm_ret.shape
        assert self.btc_signal.returns.shape == self.btc_signal.norm_ret.shape

        # values
        assert (actual_ts.dropna().std() <= 1.5).all()
        assert (actual_cs.dropna().std() <= 1.5).all()
        assert (actual_btc.dropna().std() < 1.5).all()
        assert np.allclose(actual_ts.loc[pd.IndexSlice[:, 'BTC'], :], actual_btc, equal_nan=True)
        assert (self.ts_signal.norm_ret.std() <= 1.5).all()
        assert (self.cs_signal.norm_ret.std() <= 1.5).all()
        assert (self.btc_signal.norm_ret.std() < 1.5).all()
        assert np.allclose(self.ts_signal.norm_ret.loc[pd.IndexSlice[:, 'BTC'], :], self.btc_signal.norm_ret,
                           equal_nan=True)
        # dtypes
        assert isinstance(actual_ts, pd.DataFrame)
        assert isinstance(actual_cs, pd.DataFrame)
        assert isinstance(actual_btc, pd.DataFrame)
        assert isinstance(self.ts_signal.norm_ret, pd.DataFrame)
        assert isinstance(self.cs_signal.norm_ret, pd.DataFrame)
        assert isinstance(self.btc_signal.norm_ret, pd.DataFrame)
        assert (actual_ts.dtypes == np.float64).all()
        assert (actual_cs.dtypes == np.float64).all()
        assert (actual_btc.dtypes == np.float64).all()
        assert (self.ts_signal.norm_ret.dtypes == np.float64).all()
        assert (self.cs_signal.norm_ret.dtypes == np.float64).all()
        assert (self.btc_signal.norm_ret.dtypes == np.float64).all()

        # index
        assert (actual_ts.index == self.ts_signal.factors.index).all()
        assert (actual_cs.index == self.cs_signal.factors.index).all()
        assert (actual_btc.index == self.btc_signal.factors.index).all()
        assert (self.ts_signal.returns.index == self.ts_signal.norm_ret.index).all()
        assert (self.cs_signal.returns.index == self.cs_signal.norm_ret.index).all()
        assert (self.btc_signal.returns.index == self.btc_signal.norm_ret.index).all()

        # cols
        assert (actual_ts.columns == self.ts_signal.factors.columns).all()
        assert (actual_cs.columns == self.cs_signal.factors.columns).all()
        assert (actual_btc.columns == self.btc_signal.factors.columns).all()
        assert (self.ts_signal.returns.columns == self.ts_signal.norm_ret.columns).all()
        assert (self.cs_signal.returns.columns == self.cs_signal.norm_ret.columns).all()
        assert (self.btc_signal.returns.columns == self.btc_signal.norm_ret.columns).all()

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
        signal = Signal(spot_ret.close, price_mom, strategy=strategy, factor_bins=factor_bins, return_bins=ret_bins)
        actual = signal.quantize()

        # shape
        assert signal.factors.shape == actual.shape
        assert signal.returns.shape == signal.ret_quantiles.shape

        # values
        assert (actual.nunique() == factor_bins).all()
        assert (signal.ret_quantiles.nunique() == ret_bins).all()

        # dtypes
        assert isinstance(actual, pd.DataFrame)
        assert isinstance(signal.ret_quantiles, pd.DataFrame)
        assert (actual.dtypes == np.float64).all()
        assert (signal.ret_quantiles.dtypes == np.float64).all()

        # index
        assert (actual.index == signal.factors.index).all()
        assert (signal.ret_quantiles.index == signal.returns.index).all()

        # cols
        assert (actual.columns == signal.factors.columns).all()
        assert (signal.ret_quantiles.columns == signal.returns.columns).all()

        # get actual single index
        signal_btc = Signal(btc_spot_ret.close, btc_price_mom, strategy=strategy, factor_bins=factor_bins,
                            return_bins=ret_bins)
        actual_btc = signal_btc.quantize()

        # shape
        assert self.btc_signal.factors.shape == actual_btc.shape
        assert self.btc_signal.returns.shape == signal_btc.ret_quantiles.shape

        # values
        if strategy == 'ts_ls':
            assert np.allclose(actual.loc[pd.IndexSlice[:, 'BTC'], :], actual_btc, equal_nan=True)
            assert (actual_btc.nunique() == factor_bins).all()
            assert (signal_btc.ret_quantiles.nunique() == ret_bins).all()

        # dtypes
        assert isinstance(actual_btc, pd.DataFrame)
        assert (actual_btc.dtypes == np.float64).all()

        # index
        assert (actual_btc.index == signal_btc.factors.index).all()

        # cols
        assert (actual_btc.columns == signal_btc.factors.columns).all()

    @pytest.mark.parametrize("pdf", ['norm', 'percentile', 'min-max', 'logistic', 'adj_norm'])
    def test_convert_to_signals(self, pdf):
        """
        Test convert_to_signals method.
        """
        # get actual
        actual_ts = self.ts_signal.convert_to_signals(pdf=pdf)
        actual_cs = self.cs_signal.convert_to_signals(pdf=pdf)
        actual_btc = self.btc_signal.convert_to_signals(pdf=pdf)

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

    def test_discretize_signals(self):
        """
        Test discretize_signals method.
        """
        # get actual
        actual_ts = self.ts_signal.discretize_signals()
        actual_cs = self.cs_signal.discretize_signals()
        actual_btc = self.btc_signal.discretize_signals()

        # shape
        assert self.ts_signal.factors.shape == actual_ts.shape
        assert self.cs_signal.factors.shape == actual_cs.shape
        assert self.btc_signal.factors.shape == actual_btc.shape

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

        # values
        assert actual_ts.isin([-1, 0, 1]).all().all()
        assert actual_cs.isin([-1, 0, 1]).all().all()
        assert actual_btc.isin([-1, 0, 1]).all().all()
        signals = self.ts_signal.convert_to_signals()
        signals_cs = self.cs_signal.convert_to_signals()
        signals_btc = self.btc_signal.convert_to_signals()
        assert (actual_ts.corrwith(signals) > 0.5).all()
        assert (actual_cs.corrwith(signals_cs) > 0.5).all()
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
        signal = Signal(spot_ret.close, price_mom, strategy=strategy, factor_bins=factor_bins)
        actual = signal.signals_to_quantiles()

        # shape
        assert signal.factors.shape == actual.shape

        # values
        assert (actual.nunique() == factor_bins).all()
        assert ((actual.dropna() >= -1) & (actual.dropna() <= 1)).all().all()
        assert actual.max().max() == 1.0
        assert actual.min().min() == -1.0

        # dtypes
        assert isinstance(actual, pd.DataFrame)
        assert (actual.dtypes == np.float64).all()

        # index
        assert (actual.index == signal.factors.index).all()

        # cols
        assert (actual.columns == signal.factors.columns).all()

        # get actual single index
        if strategy == 'ts_ls':
            actual_btc = Signal(btc_spot_ret.close, btc_price_mom, strategy='ts_ls',
                                factor_bins=factor_bins).signals_to_quantiles()
            # shape
            assert self.btc_signal.factors.shape == actual_btc.shape

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
            assert (actual_btc.index == self.btc_signal.factors.index).all()

            # cols
            assert (actual_btc.columns == self.btc_signal.factors.columns).all()

    @pytest.mark.parametrize("n_factors", [5, 10, 20])
    def test_signal_rank(self, price_mom, spot_ret, n_factors):
        """
        Test signals_rank method.
        """
        # get actual multiindex
        actual = Signal(spot_ret.close, price_mom, strategy='cs_ls', n_factors=n_factors).signals_to_rank()

        # dtypes
        assert isinstance(actual, pd.DataFrame)
        assert (actual.dtypes == np.int64).all()

        # shape
        assert self.cs_signal.factors.shape[1] == actual.shape[1]

        # values
        (actual[actual != 0].groupby(level=0).count() == n_factors * 2).all()  # n of signals
        (actual[actual != 0].groupby(level=0).sum() == 0).all()  # sum of signals/market neutral
        assert (actual[actual != 0].groupby(level=0).max() == 1).all().all()  # long signals
        assert (actual[actual != 0].groupby(level=0).min() == -1).all().all()  # short signals
        df = pd.concat([price_mom, actual], axis=1)
        assert df[df != 0].dropna().corr('spearman').iloc[0, 1] > 0.5  # correlation with price momentum

        # cols
        assert (actual.columns == self.cs_signal.factors.columns).all()

    @pytest.mark.parametrize("signal_type, lags",
                             [
                                 ('signal', None),
                                 ('signal', 1),
                                 ('signal_quantiles', 0),
                                 ('signal_quantiles', 1),
                                 ('disc_signal', None),
                                 ('disc_signal', 1),
                                 ('signal_rank', 1),
                                 ('signal_rank', 2)
                             ]
                             )
    def test_compute_signals(self, signal_type, lags, price_mom):
        """
        Test compute_signals method.
        """
        # get actual
        if signal_type != 'signal_rank':
            actual_ts = self.ts_signal.compute_signals(signal_type=signal_type, lags=lags)
            actual_cs = self.cs_signal.compute_signals(signal_type=signal_type, pdf='min-max', lags=lags)
            actual_btc = self.btc_signal.compute_signals(signal_type=signal_type, lags=lags)

            # shape
            assert self.ts_signal.factors.shape == actual_ts.shape
            assert self.cs_signal.factors.shape == actual_cs.shape
            assert self.btc_signal.factors.shape == actual_btc.shape

            # values
            assert ((actual_ts.dropna() >= -1) & (actual_ts.dropna() <= 1)).all().all()
            assert ((actual_cs.dropna() >= -1) & (actual_cs.dropna() <= 1)).all().all()
            assert ((actual_btc.dropna() >= -1) & (actual_btc.dropna() <= 1)).all().all()
            assert np.allclose(actual_ts.loc[pd.IndexSlice[:, 'BTC'], :].dropna(), actual_btc.dropna())

            # check factors vs. signals correlation
            if signal_type == 'signal':
                if lags is None or lags == 0:
                    trend_factor_mean = price_mom.price_mom_20.unstack().mean(axis=1)
                    signal_mean = actual_ts.price_mom_20.unstack().mean(axis=1)
                    trend_tickers_rank = price_mom.price_mom_20.unstack().iloc[-1].sort_values().index.to_list()
                    cs_tickers_rank = actual_cs.price_mom_20.unstack().iloc[-1].sort_values().index.to_list()
                elif lags == 1:
                    trend_factor_mean = price_mom.price_mom_20.unstack().mean(axis=1).shift(1)
                    signal_mean = actual_ts.price_mom_20.unstack().mean(axis=1)
                    trend_tickers_rank = price_mom.price_mom_20.unstack().iloc[-2].sort_values().index.to_list()
                    cs_tickers_rank = actual_cs.price_mom_20.unstack().iloc[-1].sort_values().index.to_list()
                elif lags == 2:
                    trend_factor_mean = price_mom.price_mom_20.unstack().mean(axis=1).shift(2)
                    signal_mean = actual_ts.price_mom_20.unstack().mean(axis=1)
                    trend_tickers_rank = price_mom.price_mom_20.unstack().iloc[-3].sort_values().index.to_list()
                    cs_tickers_rank = actual_cs.price_mom_20.unstack().iloc[-1].sort_values().index.to_list()
                assert pd.concat([trend_factor_mean, signal_mean], axis=1).corr(method='spearman').iloc[0, 1] > 0.5
                assert trend_tickers_rank == cs_tickers_rank

            elif signal_type == 'signal_quantiles':
                if lags is None or lags == 0:
                    trend_factor_mean = price_mom.price_mom_20.unstack().mean(axis=1)
                    signal_mean = actual_ts.price_mom_20.unstack().mean(axis=1)
                    trend_tickers_rank = price_mom.price_mom_20.unstack().iloc[-1].sort_values()
                    cs_tickers_rank = actual_cs.price_mom_20.unstack().iloc[-1].sort_values()
                elif lags == 1:
                    trend_factor_mean = price_mom.price_mom_20.unstack().mean(axis=1).shift(1)
                    signal_mean = actual_ts.price_mom_20.unstack().mean(axis=1)
                    trend_tickers_rank = price_mom.price_mom_20.unstack().iloc[-2].sort_values()
                    cs_tickers_rank = actual_cs.price_mom_20.unstack().iloc[-1].sort_values()
                elif lags == 2:
                    trend_factor_mean = price_mom.price_mom_20.unstack().mean(axis=1).shift(2)
                    signal_mean = actual_ts.price_mom_20.unstack().mean(axis=1)
                    trend_tickers_rank = price_mom.price_mom_20.unstack().iloc[-3].sort_values()
                    cs_tickers_rank = actual_cs.price_mom_20.unstack().iloc[-1].sort_values()
                assert pd.concat([trend_factor_mean, signal_mean], axis=1).corr(method='spearman').iloc[0, 1] > 0.5
                assert pd.concat([trend_tickers_rank, cs_tickers_rank], axis=1).corr(method='spearman').iloc[0, 1] > 0.8

            elif signal_type == 'disc_signal':
                if lags is None or lags == 0:
                    trend_factor_mean = price_mom.price_mom_20.unstack().mean(axis=1)
                    signal_mean = actual_ts.price_mom_20.unstack().mean(axis=1)
                    trend_tickers_rank = price_mom.price_mom_20.unstack().iloc[-1].sort_values()
                    cs_tickers_rank = actual_cs.price_mom_20.unstack().iloc[-1].sort_values()
                elif lags == 1:
                    trend_factor_mean = price_mom.price_mom_20.unstack().mean(axis=1).shift(1)
                    signal_mean = actual_ts.price_mom_20.unstack().mean(axis=1)
                    trend_tickers_rank = price_mom.price_mom_20.unstack().iloc[-2].sort_values()
                    cs_tickers_rank = actual_cs.price_mom_20.unstack().iloc[-1].sort_values()
                elif lags == 2:
                    trend_factor_mean = price_mom.price_mom_20.unstack().mean(axis=1).shift(2)
                    signal_mean = actual_ts.price_mom_20.unstack().mean(axis=1)
                    trend_tickers_rank = price_mom.price_mom_20.unstack().iloc[-3].sort_values()
                    cs_tickers_rank = actual_cs.price_mom_20.unstack().iloc[-1].sort_values()

                assert pd.concat([trend_factor_mean, signal_mean], axis=1).corr(method='spearman').iloc[0, 1] > 0.5
                assert pd.concat([trend_tickers_rank, cs_tickers_rank], axis=1).corr(method='spearman').iloc[0, 1] > 0.6

            elif signal_type == 'signal_rank':
                if lags == 1:
                    trend_tickers_rank = price_mom.price_mom_20.unstack().iloc[-2].sort_values()
                    cs_tickers_rank = actual_cs.price_mom_20.unstack().iloc[-1].sort_values()
                elif lags == 2:
                    trend_tickers_rank = price_mom.price_mom_20.unstack().iloc[-3].sort_values()
                    cs_tickers_rank = actual_cs.price_mom_20.unstack().iloc[-1].sort_values()
                assert pd.concat([trend_tickers_rank, cs_tickers_rank], axis=1).corr(method='spearman').iloc[0, 1] > 0.5

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

        # signal rank
        else:
            actual_cs = self.cs_signal.compute_signals(signal_type=signal_type, lags=lags)
            assert ((actual_cs.dropna() >= -1) & (actual_cs.dropna() <= 1)).all().all()
            assert isinstance(actual_cs, pd.DataFrame)
            assert (actual_cs.dtypes == 'float64').all()
            assert (actual_cs.columns == self.cs_signal.factors.columns).all()

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

    def test_compute_signal_returns(self):
        """
        Test compute_signal_returns method.
        """
        # get actual
        self.ts_signal.compute_signal_returns()
        self.btc_signal.compute_signal_returns()

        # shape
        assert self.ts_signal.signal_rets.shape == self.ts_signal.signals.dropna().shape
        assert self.btc_signal.signal_rets.shape == self.btc_signal.signals.dropna().shape

        # dtypes
        assert isinstance(self.ts_signal.signal_rets, pd.DataFrame)
        assert isinstance(self.btc_signal.signal_rets, pd.DataFrame)
        assert (self.ts_signal.signal_rets.dtypes == np.float64).all()
        assert (self.btc_signal.signal_rets.dtypes == np.float64).all()

        # values
        assert np.allclose(self.ts_signal.signal_rets.loc[pd.IndexSlice[:, 'BTC'], :].dropna(),
                           self.btc_signal.signal_rets.dropna())

        # index
        assert (self.ts_signal.signal_rets.index == self.ts_signal.signals.dropna().index).all()
        assert (self.btc_signal.signal_rets.index == self.btc_signal.signals.dropna().index).all()

        # cols
        assert (self.ts_signal.signal_rets.columns == self.ts_signal.signals.dropna().columns).all()
        assert (self.btc_signal.signal_rets.columns == self.btc_signal.signals.dropna().columns).all()

    @pytest.mark.parametrize("method", ['sign', 'std', 'skew', 'range'])
    def test_signal_dispersion(self, method):
        """
        Test signal_dispersion method.
        """
        # get actual
        self.ts_signal.compute_signals()
        actual = self.ts_signal.signal_dispersion(method)

        # shape
        assert self.ts_signal.factors.unstack().shape[0] == actual.shape[0]
        assert actual.shape[1] == 1

        # dtypes
        assert isinstance(actual, pd.DataFrame)
        assert (actual.dtypes == np.float64).all()

        # index
        assert (actual.index == self.ts_signal.factors.unstack().index).all()

        # cols
        assert (actual.columns == self.ts_signal.factors.columns).all()
