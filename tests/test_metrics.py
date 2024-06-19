import pytest
import pandas as pd
import numpy as np

from factorlab.feature_engineering.transformations import Transform
from factorlab.strategy_backtesting.metrics import Metrics


@pytest.fixture
def crypto_log_returns():
    """
    Fixture for crypto OHLCV prices.
    """
    # read csv from datasets/data
    df = pd.read_csv("../src/factorlab/datasets/data/cc_spot_prices.csv", index_col=['date', 'ticker'],
                     parse_dates=['date']).loc[:, : 'close']

    # drop tickers with nobs < ts_obs
    obs = df.groupby(level=1).count().min(axis=1)
    drop_tickers_list = obs[obs < 365].index.to_list()
    df = df.drop(drop_tickers_list, level=1, axis=0)

    ret = Transform(df).returns().close.unstack()

    return ret


@pytest.fixture
def crypto_simple_returns():
    """
    Fixture for crypto OHLCV prices.
    """
    # read csv from datasets/data
    df = pd.read_csv("../src/factorlab/datasets/data/cc_spot_prices.csv", index_col=['date', 'ticker'],
                     parse_dates=['date']).loc[:, : 'close']

    # drop tickers with nobs < ts_obs
    # drop tickers with nobs < ts_obs
    obs = df.groupby(level=1).count().min(axis=1)
    drop_tickers_list = obs[obs < 365].index.to_list()
    df = df.drop(drop_tickers_list, level=1, axis=0)

    ret = Transform(df).returns(method='simple').close.unstack()

    return ret


@pytest.fixture
def risk_free_rates():
    """
    Fixture for US real rates data.
    """
    # read csv from datasets/data
    return pd.read_csv("../src/factorlab/datasets/data/us_real_rates_10y_monthly.csv", index_col=['date'],
                       parse_dates=['date']).loc[:, 'US_Rates_3M'] / 100


class TestMetrics:
    """
    Test class for Metrics.
    """
    @pytest.fixture(autouse=True)
    def metrics_setup_log_ret(self, crypto_log_returns, risk_free_rates):
        self.metrics_log_ret_instance = Metrics(crypto_log_returns, risk_free_rate=risk_free_rates, ret_type='log')

    @pytest.fixture(autouse=True)
    def metrics_setup_simple_ret(self, crypto_simple_returns, risk_free_rates):
        self.metrics_simple_ret_instance = Metrics(crypto_simple_returns, risk_free_rate=risk_free_rates,
                                                   ret_type='simple')

    def test_initialization(self) -> None:
        """
        Test initialization.
        """
        # dtypes
        assert isinstance(self.metrics_log_ret_instance, Metrics)
        assert isinstance(self.metrics_log_ret_instance.returns, pd.DataFrame)
        assert isinstance(self.metrics_log_ret_instance.risk_free_rate, (pd.Series, pd.DataFrame, float))
        assert isinstance(self.metrics_log_ret_instance.mkt_ret, (pd.Series, pd.DataFrame))
        assert isinstance(self.metrics_log_ret_instance.freq, str)
        assert type(self.metrics_log_ret_instance.ann_factor) == np.int64
        assert type(self.metrics_log_ret_instance.window_size) == np.int64
        assert (self.metrics_log_ret_instance.returns.dtypes == np.float64).all()
        # shape
        assert self.metrics_log_ret_instance.returns.shape[0] == \
               self.metrics_simple_ret_instance.risk_free_rate.shape[0]
        # index
        assert isinstance(self.metrics_log_ret_instance.returns.index, pd.DatetimeIndex)

    def test_excess_returns(self) -> None:
        """
        Test excess returns computation.
        """
        rets = self.metrics_log_ret_instance.returns
        self.metrics_log_ret_instance.excess_returns()
        # dtypes
        assert isinstance(self.metrics_log_ret_instance.returns, pd.DataFrame)
        assert (self.metrics_log_ret_instance.returns.dtypes == np.float64).all()
        # shape
        assert rets.shape[0] == self.metrics_log_ret_instance.returns.shape[0]
        assert rets.shape[1] == self.metrics_log_ret_instance.returns.shape[1]

    @pytest.mark.parametrize("ret_type", ['simple', 'log'])
    def test_cumulative_returns(self, ret_type) -> None:
        """
        Test cumulative returns computation.
        """
        cum_ret_log = self.metrics_log_ret_instance.cumulative_returns()
        cum_ret_simple = self.metrics_simple_ret_instance.cumulative_returns()
        # dtypes
        assert isinstance(cum_ret_log, pd.DataFrame)
        assert isinstance(cum_ret_simple, pd.DataFrame)
        assert (cum_ret_log.dtypes == np.float64).all()
        assert (cum_ret_simple.dtypes == np.float64).all()
        # shape
        assert cum_ret_log.shape[0] == self.metrics_log_ret_instance.returns.shape[0]
        assert cum_ret_simple.shape[0] == self.metrics_simple_ret_instance.returns.shape[0]
        assert cum_ret_log.shape[1] == self.metrics_log_ret_instance.returns.shape[1]
        assert cum_ret_simple.shape[1] == self.metrics_simple_ret_instance.returns.shape[1]
        # values
        assert ((cum_ret_log - cum_ret_simple).sum() < 0.01).all()

    def test_annualized_return(self) -> None:
        """
        Test annualized return computation.
        """
        ann_ret = self.metrics_log_ret_instance.annualized_return()
        # dtypes
        assert isinstance(ann_ret, pd.Series)
        assert ann_ret.dtypes == np.float64
        # shape
        assert ann_ret.shape[0] == self.metrics_log_ret_instance.returns.shape[1]

    @pytest.mark.parametrize("window_type", ['rolling', 'expanding', 'fixed'])
    def test_winning_percentage(self, window_type) -> None:
        """
        Test winning percentage computation.
        """
        self.metrics_log_ret_instance.window_type = window_type
        win_perc = self.metrics_log_ret_instance.winning_percentage()
        # dtypes
        assert isinstance(win_perc, (pd.Series, pd.DataFrame))
        if window_type == 'fixed':
            assert win_perc.dtypes == np.float64
        else:
            assert (win_perc.dtypes == np.float64).all()
        # shape
        if window_type == 'fixed':
            assert win_perc.shape[0] == self.metrics_log_ret_instance.returns.shape[1]
        else:
            assert win_perc.shape[0] == self.metrics_log_ret_instance.returns.shape[0]
            assert win_perc.shape[1] == self.metrics_log_ret_instance.returns.shape[1]

    def test_drawdown(self) -> None:
        """
        Test drawdown computation.
        """
        drawdown = self.metrics_log_ret_instance.drawdown()
        # dtypes
        assert isinstance(drawdown, (pd.Series, pd.DataFrame))
        assert (drawdown.dtypes == np.float64).all()
        # shape
        assert drawdown.shape[0] == self.metrics_log_ret_instance.returns.shape[0]
        assert drawdown.shape[1] == self.metrics_log_ret_instance.returns.shape[1]

    @pytest.mark.parametrize("window_type", ['rolling', 'expanding', 'fixed'])
    def test_max_drawdown(self, window_type) -> None:
        """
        Test max drawdown computation.
        """
        self.metrics_log_ret_instance.window_type = window_type
        drawdown = self.metrics_log_ret_instance.max_drawdown()
        # dtypes
        assert isinstance(drawdown, (pd.Series, pd.DataFrame))
        if window_type == 'fixed':
            assert drawdown.dtypes == np.float64
        else:
            assert (drawdown.dtypes == np.float64).all()
        # shape
        if window_type == 'fixed':
            assert drawdown.shape[0] == self.metrics_log_ret_instance.returns.shape[1]
        else:
            assert drawdown.shape[0] == self.metrics_log_ret_instance.returns.shape[0]
            assert drawdown.shape[1] == self.metrics_log_ret_instance.returns.shape[1]

    def test_conditional_drawdown_risk(self) -> None:
        """
        Test conditional drawdown at risk computation.
        """
        cvar = self.metrics_log_ret_instance.conditional_drawdown_risk()
        # dtypes
        assert isinstance(cvar, (pd.Series, pd.DataFrame))
        assert cvar.dtypes == np.float64
        # shape
        assert cvar.shape[0] == self.metrics_log_ret_instance.returns.shape[1]

    @pytest.mark.parametrize("window_type", ['rolling', 'expanding', 'fixed'])
    def test_annualized_vol(self, window_type) -> None:
        """
        Test annualized volatility computation.
        """
        self.metrics_log_ret_instance.window_type = window_type
        ann_vol = self.metrics_log_ret_instance.annualized_vol()
        # dtypes
        assert isinstance(ann_vol, (pd.Series, pd.DataFrame))
        if window_type == 'fixed':
            assert ann_vol.dtypes == np.float64
        else:
            assert (ann_vol.dtypes == np.float64).all()
        # shape
        if window_type == 'fixed':
            assert ann_vol.shape[0] == self.metrics_log_ret_instance.returns.shape[1]
        else:
            assert ann_vol.shape[0] == self.metrics_log_ret_instance.returns.shape[0]
            assert ann_vol.shape[1] == self.metrics_log_ret_instance.returns.shape[1]

    @pytest.mark.parametrize("window_type", ['rolling', 'expanding', 'fixed'])
    def test_skewness(self, window_type) -> None:
        """
        Test skewness computation.
        """
        self.metrics_log_ret_instance.window_type = window_type
        skew = self.metrics_log_ret_instance.skewness()
        # dtypes
        assert isinstance(skew, (pd.Series, pd.DataFrame))
        if window_type == 'fixed':
            assert skew.dtypes == np.float64
        else:
            assert (skew.dtypes == np.float64).all()
        # shape
        if window_type == 'fixed':
            assert skew.shape[0] == self.metrics_log_ret_instance.returns.shape[1]
        else:
            assert skew.shape[0] == self.metrics_log_ret_instance.returns.shape[0]
            assert skew.shape[1] == self.metrics_log_ret_instance.returns.shape[1]

    @pytest.mark.parametrize("window_type", ['rolling', 'expanding', 'fixed'])
    def test_kurtosis(self, window_type) -> None:
        """
        Test skewness computation.
        """
        self.metrics_log_ret_instance.window_type = window_type
        kurt = self.metrics_log_ret_instance.skewness()
        # dtypes
        assert isinstance(kurt, (pd.Series, pd.DataFrame))
        if window_type == 'fixed':
            assert kurt.dtypes == np.float64
        else:
            assert (kurt.dtypes == np.float64).all()
        # shape
        if window_type == 'fixed':
            assert kurt.shape[0] == self.metrics_log_ret_instance.returns.shape[1]
        else:
            assert kurt.shape[0] == self.metrics_log_ret_instance.returns.shape[0]
            assert kurt.shape[1] == self.metrics_log_ret_instance.returns.shape[1]

    @pytest.mark.parametrize("window_type", ['rolling', 'expanding', 'fixed'])
    def test_var(self, window_type) -> None:
        """
        Test VaR computation.
        """
        self.metrics_log_ret_instance.window_type = window_type
        var = self.metrics_log_ret_instance.value_at_risk()
        # dtypes
        assert isinstance(var, (pd.Series, pd.DataFrame))
        if window_type == 'fixed':
            assert var.dtypes == np.float64
        else:
            assert (var.dtypes == np.float64).all()
        # shape
        if window_type == 'fixed':
            assert var.shape[0] == self.metrics_log_ret_instance.returns.shape[1]
        else:
            assert var.shape[0] == self.metrics_log_ret_instance.returns.shape[0]
            assert var.shape[1] == self.metrics_log_ret_instance.returns.shape[1]

    @pytest.mark.parametrize("window_type", ['rolling', 'expanding', 'fixed'])
    def test_expected_shortfall(self, window_type) -> None:
        """
        Test expected shortfall computation.
        """
        self.metrics_log_ret_instance.window_type = window_type
        es = self.metrics_log_ret_instance.expected_shortfall()
        # dtypes
        assert isinstance(es, (pd.Series, pd.DataFrame))
        if window_type == 'fixed':
            assert es.dtypes == np.float64
        else:
            assert (es.dtypes == np.float64).all()
        # shape
        if window_type == 'fixed':
            assert es.shape[0] == self.metrics_log_ret_instance.returns.shape[1]
        else:
            assert es.shape[0] == self.metrics_log_ret_instance.returns.shape[0]
            assert es.shape[1] == self.metrics_log_ret_instance.returns.shape[1]

    @pytest.mark.parametrize("window_type", ['rolling', 'expanding', 'fixed'])
    def test_tail_ratio(self, window_type) -> None:
        """
        Test tail ratio computation.
        """
        self.metrics_log_ret_instance.window_type = window_type
        tr = self.metrics_log_ret_instance.tail_ratio()
        # dtypes
        assert isinstance(tr, (pd.Series, pd.DataFrame))
        if window_type == 'fixed':
            assert tr.dtypes == np.float64
        else:
            assert (tr.dtypes == np.float64).all()
        # shape
        if window_type == 'fixed':
            assert tr.shape[0] == self.metrics_log_ret_instance.returns.shape[1]
        else:
            assert tr.shape[0] == self.metrics_log_ret_instance.returns.shape[0]
            assert tr.shape[1] == self.metrics_log_ret_instance.returns.shape[1]

    @pytest.mark.parametrize("window_type", ['rolling', 'expanding', 'fixed'])
    def test_sharpe_ratio(self, window_type) -> None:
        """
        Test sharpe ratio computation.
        """
        self.metrics_log_ret_instance.window_type = window_type
        sr = self.metrics_log_ret_instance.sharpe_ratio()
        # dtypes
        assert isinstance(sr, (pd.Series, pd.DataFrame))
        if window_type == 'fixed':
            assert sr.dtypes == np.float64
        else:
            assert (sr.dtypes == np.float64).all()
        # shape
        if window_type == 'fixed':
            assert sr.shape[0] == self.metrics_log_ret_instance.returns.shape[1]
        else:
            assert sr.shape[0] == self.metrics_log_ret_instance.returns.shape[0]
            assert sr.shape[1] == self.metrics_log_ret_instance.returns.shape[1]

    @pytest.mark.parametrize("window_type", ['rolling', 'expanding', 'fixed'])
    def test_profit_factor(self, window_type) -> None:
        """
        Test sharpe ratio computation.
        """
        self.metrics_log_ret_instance.window_type = window_type
        pf = self.metrics_log_ret_instance.profit_factor()
        # dtypes
        assert isinstance(pf, (pd.Series, pd.DataFrame))
        if window_type == 'fixed':
            assert pf.dtypes == np.float64
        else:
            assert (pf.dtypes == np.float64).all()
        # shape
        if window_type == 'fixed':
            assert pf.shape[0] == self.metrics_log_ret_instance.returns.shape[1]
        else:
            assert pf.shape[0] == self.metrics_log_ret_instance.returns.shape[0]
            assert pf.shape[1] == self.metrics_log_ret_instance.returns.shape[1]

    @pytest.mark.parametrize("window_type", ['rolling', 'expanding', 'fixed'])
    def test_sortino_ratio(self, window_type) -> None:
        """
        Test sortino ratio computation.
        """
        self.metrics_log_ret_instance.window_type = window_type
        sr = self.metrics_log_ret_instance.sortino_ratio()
        # dtypes
        assert isinstance(sr, (pd.Series, pd.DataFrame))
        if window_type == 'fixed':
            assert sr.dtypes == np.float64
        else:
            assert (sr.dtypes == np.float64).all()
        # shape
        if window_type == 'fixed':
            assert sr.shape[0] == self.metrics_log_ret_instance.returns.shape[1]
        else:
            assert sr.shape[0] == self.metrics_log_ret_instance.returns.shape[0]
            assert sr.shape[1] == self.metrics_log_ret_instance.returns.shape[1]

    def test_calmar_ratio(self) -> None:
        """
        Test calmar ratio computation.
        """
        cr = self.metrics_log_ret_instance.calmar_ratio()
        # dtypes
        assert isinstance(cr, (pd.Series, pd.DataFrame))
        assert cr.dtypes == np.float64
        # shape
        assert cr.shape[0] == self.metrics_log_ret_instance.returns.shape[1]

    @pytest.mark.parametrize("window_type", ['rolling', 'expanding', 'fixed'])
    def test_omega_ratio(self, window_type) -> None:
        """
        Test omega ratio computation.
        """
        self.metrics_log_ret_instance.window_type = window_type
        omr = self.metrics_log_ret_instance.omega_ratio()
        # dtypes
        assert isinstance(omr, (pd.Series, pd.DataFrame))
        if window_type == 'fixed':
            assert omr.dtypes == np.float64
        else:
            assert (omr.dtypes == np.float64).all()
        # shape
        if window_type == 'fixed':
            assert omr.shape[0] == self.metrics_log_ret_instance.returns.shape[1]
        else:
            assert omr.shape[0] == self.metrics_log_ret_instance.returns.shape[0]
            assert omr.shape[1] == self.metrics_log_ret_instance.returns.shape[1]

    @pytest.mark.parametrize("window_type", ['rolling', 'expanding', 'fixed'])
    def test_stability(self, window_type) -> None:
        """
        Test stability computation.
        """
        self.metrics_log_ret_instance.window_type = window_type
        stab = self.metrics_log_ret_instance.stability()
        # dtypes
        assert isinstance(stab, (pd.Series, pd.DataFrame))
        assert (stab.dtypes == np.float64).all()
        # shape
        if window_type == 'rolling':
            assert stab.shape[0] == self.metrics_log_ret_instance.returns.shape[0]
            assert stab.shape[1] == self.metrics_log_ret_instance.returns.shape[1]
        else:
            assert stab.shape[0] == self.metrics_log_ret_instance.returns.shape[1]
        # index
        if window_type == 'rolling':
            assert isinstance(stab.index, pd.DatetimeIndex)
            assert stab.index.equals(self.metrics_log_ret_instance.returns.index)
        else:
            assert stab.index.equals(self.metrics_log_ret_instance.returns.columns)
        # cols
        if window_type == 'rolling':
            assert (stab.columns == self.metrics_log_ret_instance.returns.columns).all()
        else:
            assert stab.columns == ['stability']

    @pytest.mark.parametrize("window_type", ['rolling', 'expanding', 'fixed'])
    def test_alpha(self, window_type) -> None:
        """
        Test alpha computation.
        """
        self.metrics_log_ret_instance.window_type = window_type
        a = self.metrics_log_ret_instance.alpha()
        # dtypes
        assert isinstance(a, (pd.Series, pd.DataFrame))
        assert (a.dtypes == np.float64).all()
        # shape
        assert a.shape[1] == 2
        if window_type == 'fixed' or window_type == 'expanding':
            assert a.shape[0] == self.metrics_log_ret_instance.returns.shape[1]
        # index
        if window_type == 'rolling':
            assert isinstance(a.index, pd.MultiIndex)
        else:
            assert a.index.equals(self.metrics_log_ret_instance.returns.columns)
        # cols
        assert (a.columns == ['alpha', 'p_val']).all()

    @pytest.mark.parametrize("window_type", ['rolling', 'expanding', 'fixed'])
    def test_beta(self, window_type) -> None:
        """
        Test beta computation.
        """
        self.metrics_log_ret_instance.window_type = window_type
        b = self.metrics_log_ret_instance.beta()
        # dtypes
        assert isinstance(b, (pd.Series, pd.DataFrame))
        assert (b.dtypes == np.float64).all()
        # shape
        assert b.shape[1] == 2
        if window_type == 'fixed' or window_type == 'expanding':
            assert b.shape[0] == self.metrics_log_ret_instance.returns.shape[1]
        # index
        if window_type == 'rolling':
            assert isinstance(b.index, pd.MultiIndex)
        else:
            assert b.index.equals(self.metrics_log_ret_instance.returns.columns)
        # cols
        assert (b.columns == ['beta', 'p_val']).all()
