import pytest
import pandas as pd
import numpy as np

import statsmodels

from factorlab.signal_generation.time_series_analysis import add_lags, rolling_window, expanding_window, \
    TimeSeriesAnalysis as TSA
from factorlab.feature_engineering.transformations import Transform
from factorlab.feature_engineering.factors.trend import Trend


@pytest.fixture
def fx_returns_monthly():
    """
    Fixture for monthly FX returns.
    """
    # read csv from datasets/data
    return pd.read_csv("../src/factorlab/datasets/data/fx_returns_monthly.csv", index_col='date')


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


def compute_mean(data):
    """
    Function to compute mean.
    """
    return np.mean(data, axis=0)


def test_rolling_window(fx_returns_monthly) -> None:
    """
    Test rolling window function.
    """
    # get actual and expected
    rolling_mean = rolling_window(compute_mean, fx_returns_monthly, 60)
    actual = pd.DataFrame(rolling_mean, index=fx_returns_monthly.index[-rolling_mean.shape[0]:],
                          columns=fx_returns_monthly.columns)
    expected = fx_returns_monthly.rolling(60).mean().dropna()

    # test shape
    assert actual.shape == expected.shape

    # test values
    assert np.allclose(actual, expected)
    assert all(actual.iloc[-1].values == fx_returns_monthly.iloc[-60:].mean().values)

    # test index
    assert all(actual.index == expected.index)


def test_rolling_window_size_param_errors(fx_returns_monthly) -> None:
    """
    Test rolling window function window size errors.
    """

    # test if window is greater than length of data
    with pytest.raises(ValueError):
        rolling_window(compute_mean, fx_returns_monthly, fx_returns_monthly.shape[0]+1)

    # test if window is less than 1
    with pytest.raises(ValueError):
        rolling_window(compute_mean, fx_returns_monthly,  0)

    # test if window is not an integer
    with pytest.raises(TypeError):
        rolling_window(compute_mean, fx_returns_monthly, 1.5)


def test_expanding_window(fx_returns_monthly) -> None:
    """
    Test expanding window function.
    """
    # get actual and expected
    expanding_mean = expanding_window(compute_mean, fx_returns_monthly, 60)
    actual = pd.DataFrame(expanding_mean, index=fx_returns_monthly.index[-expanding_mean.shape[0]:],
                          columns=fx_returns_monthly.columns)
    expected = fx_returns_monthly.expanding(60).mean().dropna()

    # test shape
    assert actual.shape == expected.shape

    # test values
    assert np.allclose(actual, expected)
    assert all(actual.iloc[-1].values == fx_returns_monthly.mean().values)

    # test index
    assert all(actual.index == expected.index)


def test_expanding_window_size_param_errors(fx_returns_monthly) -> None:
    """
    Test rolling window function window size errors.
    """

    # test if window is greater than length of data
    with pytest.raises(ValueError):
        expanding_window(compute_mean, fx_returns_monthly, fx_returns_monthly.shape[0]+1)

    # test if window is less than 1
    with pytest.raises(ValueError):
        expanding_window(compute_mean, fx_returns_monthly,  0)

    # test if window is not an integer
    with pytest.raises(TypeError):
        expanding_window(compute_mean, fx_returns_monthly, 1.5)


@pytest.mark.parametrize("n_lags", [1, 2, 3])
def test_create_lags(spot_ret, btc_spot_ret, n_lags) -> None:
    """
    Test create_lags method.
    """
    # get actual and expected
    actual = add_lags(spot_ret.close, n_lags)
    actual_btc = add_lags(btc_spot_ret.close, n_lags)

    # test shape
    assert actual.shape[1] == 1 + (n_lags * 1)
    assert actual_btc.shape[1] == 1 + (n_lags * 1)
    # test dtypes
    assert isinstance(actual, pd.DataFrame)
    assert isinstance(actual_btc, pd.DataFrame)
    assert (actual.dtypes == np.float64).all()
    assert (actual_btc.dtypes == np.float64).all()

    # test values
    np.allclose(actual.loc[pd.IndexSlice[:, 'BTC'], :].iloc[:, 0].values, actual_btc.iloc[:, 0].values)
    np.allclose(actual.loc[pd.IndexSlice[:, 'BTC'], :].iloc[:-1, 0].values,
                actual.loc[pd.IndexSlice[:, 'BTC'], :].iloc[1:, 1].values)
    np.allclose(actual_btc.iloc[:-1, 0].values, actual_btc.iloc[1:, 1].values)
    if n_lags > 1:
        np.allclose(actual.loc[pd.IndexSlice[:, 'BTC'], :].iloc[1:, 1].values, actual_btc.iloc[1:, 1].values)
        np.allclose(actual.loc[pd.IndexSlice[:, 'BTC'], :].iloc[:-2, 0].values,
                    actual.loc[pd.IndexSlice[:, 'BTC'], :].iloc[2:, 2].values)
    if n_lags > 2:
        np.allclose(actual.loc[pd.IndexSlice[:, 'BTC'], :].iloc[2:, 2].values,
                    actual_btc.iloc[2:, 2].values)
        np.allclose(actual.loc[pd.IndexSlice[:, 'BTC'], :].iloc[:-3, 0].values,
                    actual.loc[pd.IndexSlice[:, 'BTC'], :].iloc[3:, 3].values)

    # test index
    assert (actual.index[-100:] == spot_ret.index[-100:]).all()
    assert (actual_btc.index[-100:] == btc_spot_ret.index[-100:]).all()

    # test col names
    assert actual.columns[0] == 'close'
    assert actual.columns[1] == 'close_L1'
    if n_lags > 1:
        assert actual.columns[2] == 'close_L2'
    if n_lags > 2:
        assert actual.columns[3] == 'close_L3'


def test_create_lags_param_errors(spot_ret, n_lags=0) -> None:
    """
    Test create_lags method parameter errors.
    """
    # test if lags is less than 1
    with pytest.raises(ValueError):
        add_lags(spot_ret.close, n_lags=n_lags)


# noinspection PyUnresolvedReferences
class TestTimeSeriesAnalysis:
    """
    Test class for time series analysis methods.
    """
    @pytest.fixture(autouse=True)
    def tsa_setup_default(self, price_mom, spot_ret):
        self.tsa_instance = TSA(spot_ret.close, price_mom, window_size=30)

    @pytest.fixture(autouse=True)
    def tsa_setup_btc(self, btc_price_mom, btc_spot_ret):
        self.btc_tsa_instance = TSA(btc_spot_ret.close, btc_price_mom, window_size=30)

    def test_initialization(self) -> None:
        """
        Test initialization.
        """
        # data type
        assert isinstance(self.tsa_instance, TSA)
        assert isinstance(self.btc_tsa_instance, TSA)
        assert isinstance(self.tsa_instance.features, pd.DataFrame)
        assert isinstance(self.btc_tsa_instance.features, pd.DataFrame)
        assert isinstance(self.tsa_instance.target, pd.Series)
        assert isinstance(self.btc_tsa_instance.target, pd.Series)

    @pytest.mark.parametrize("n_lags", [1, 2, 3])
    def test_create_lags(self, n_lags) -> None:
        """
        Test create_lags method.
        """
        # get actual and expected
        actual = TSA(self.tsa_instance.target, self.tsa_instance.features, n_lags=n_lags)
        actual_btc = TSA(self.btc_tsa_instance.target, self.btc_tsa_instance.features, n_lags=n_lags)

        # test shape
        assert actual.data.shape[1] == self.tsa_instance.data.shape[1] + (n_lags * self.tsa_instance.data.shape[1])
        assert actual_btc.data.shape[1] == self.btc_tsa_instance.data.shape[1] + \
               (n_lags * self.btc_tsa_instance.data.shape[1])
        # test dtypes
        assert isinstance(actual.data, pd.DataFrame)
        assert isinstance(actual_btc.data, pd.DataFrame)
        assert (actual.data.dtypes == np.float64).all()
        assert (actual_btc.data.dtypes == np.float64).all()

        # test values
        np.allclose(actual.data.loc[pd.IndexSlice[:, 'BTC'], :].iloc[:, 0].values,
                    actual_btc.data.iloc[:, 0].values)
        np.allclose(actual.data.loc[pd.IndexSlice[:, 'BTC'], :].iloc[:-1, 0].values,
                    actual.data.loc[pd.IndexSlice[:, 'BTC'], :].iloc[1:, 1].values)
        np.allclose(actual_btc.data.iloc[:-1, 0].values, actual_btc.data.iloc[1:, 1].values)
        if n_lags > 1:
            np.allclose(actual.data.loc[pd.IndexSlice[:, 'BTC'], :].iloc[1:, 1].values,
                        actual_btc.data.iloc[1:, 1].values)
            np.allclose(actual.data.loc[pd.IndexSlice[:, 'BTC'], :].iloc[:-2, 0].values,
                        actual.data.loc[pd.IndexSlice[:, 'BTC'], :].iloc[2:, 2].values)
        if n_lags > 2:
            np.allclose(actual.data.loc[pd.IndexSlice[:, 'BTC'], :].iloc[2:, 2].values,
                        actual_btc.data.iloc[2:, 2].values)
            np.allclose(actual.data.loc[pd.IndexSlice[:, 'BTC'], :].iloc[:-3, 0].values,
                        actual.data.loc[pd.IndexSlice[:, 'BTC'], :].iloc[3:, 3].values)

        # test index
        assert (actual.data.index[-100:] == self.tsa_instance.target.index[-100:]).all()
        assert (actual_btc.data.index[-100:] == self.btc_tsa_instance.target.index[-100:]).all()

        # test col names
        assert actual.data.columns[0] == 'close'
        assert actual.data.columns[1] == 'close_L1'
        if n_lags > 1:
            assert actual.data.columns[2] == 'close_L2'
        if n_lags > 2:
            assert actual.data.columns[3] == 'close_L3'

    def test_create_lags_param_errors(self, price_mom, spot_ret, n_lags=0) -> None:
        """
        Test create_lags method parameter errors.
        """
        # test if lags is less than 1
        with pytest.raises(ValueError):
            TSA(spot_ret.close, price_mom, n_lags=n_lags).create_lags()

    @pytest.mark.parametrize("trend", [None, 'n', 'c', 't', 'ct', 'ctt'])
    def test_add_trend(self, spot_ret, price_mom, btc_spot_ret, btc_price_mom, trend) -> None:
        """
        Test add trend function.
        """
        # get actual and expected
        actual = TSA(spot_ret.close, price_mom, trend=trend)
        actual_btc = TSA(btc_spot_ret.close, btc_price_mom, trend=trend)

        # test shape
        assert actual.features.shape[0] == self.tsa_instance.features.shape[0]
        assert actual_btc.features.shape[0] == self.btc_tsa_instance.features.shape[0]
        if trend is None or trend == 'n':
            assert actual.features.shape[1] == self.tsa_instance.features.shape[1]
            assert actual_btc.features.shape[1] == self.btc_tsa_instance.features.shape[1]
        else:
            assert actual.features.shape[1] == self.tsa_instance.features.shape[1] + len(trend)
            assert actual_btc.features.shape[1] == self.btc_tsa_instance.features.shape[1] + len(trend)

        # test dtypes
        assert isinstance(actual.features, pd.DataFrame)
        assert isinstance(actual_btc.features, pd.DataFrame)
        assert (actual.features.dtypes == np.float64).all()
        assert (actual_btc.features.dtypes == np.float64).all()

        # test values
        if trend == 'n':
            assert (actual.features == self.tsa_instance.features).all().all()
            assert (actual_btc.features == self.btc_tsa_instance.features).all().all()
        elif trend == 'c':
            assert (actual.features.const == 1).all().all()
            assert (actual_btc.features.const == 1).all().all()
        elif trend == 't':
            assert (actual.features.trend.groupby(level=1).first() == 1).all()
            assert (actual.features.trend.groupby(level=1).last() ==
                    actual.features.groupby(level=1).count().median(axis=1)).all()
            assert (actual_btc.features.trend == np.arange(1, actual_btc.features.shape[0] + 1)).all().all()
            assert np.allclose(actual.features.loc[pd.IndexSlice[:, 'BTC'], 'trend'].values,
                               actual_btc.features.trend.values)
        elif trend == 'ct':
            assert (actual.features.const == 1).all().all()
            assert (actual.features.trend.groupby(level=1).first() == 1).all()
            assert (actual.features.trend.groupby(level=1).last() ==
                    actual.features.groupby(level=1).count().median(axis=1)).all()
            assert (actual_btc.features.const == 1).all().all()
            assert (actual_btc.features.trend == np.arange(1, actual_btc.features.shape[0] + 1)).all().all()
            assert np.allclose(actual.features.loc[pd.IndexSlice[:, 'BTC'], 'const'].values,
                               actual_btc.features.const.values)
            assert np.allclose(actual.features.loc[pd.IndexSlice[:, 'BTC'], 'trend'].values,
                               actual_btc.features.trend.values)
        elif trend == 'ctt':
            assert (actual.features.const == 1).all().all()
            assert (actual.features.trend.groupby(level=1).first() == 1).all()
            assert (actual.features.trend.groupby(level=1).last() ==
                    actual.features.groupby(level=1).count().median(axis=1)).all()
            assert (actual.features.trend_squared.groupby(level=1).first() == 1).all()
            assert (actual.features.trend_squared.groupby(level=1).last() ==
                    actual.features.groupby(level=1).count().median(axis=1)**2).all()
            assert (actual_btc.features.const == 1).all().all()
            assert (actual_btc.features.trend == np.arange(1, actual_btc.features.shape[0] + 1)).all().all()
            assert (actual_btc.features.trend_squared == actual_btc.features.trend ** 2).all().all()
            assert np.allclose(actual.features.loc[pd.IndexSlice[:, 'BTC'], 'const'].values,
                               actual_btc.features.const.values)
            assert np.allclose(actual.features.loc[pd.IndexSlice[:, 'BTC'], 'trend'].values,
                               actual_btc.features.trend.values)
            assert np.allclose(actual.features.loc[pd.IndexSlice[:, 'BTC'], 'trend_squared'].values,
                           actual_btc.features.trend_squared.values)

        # test index
        assert (actual.features.index == self.tsa_instance.features.index).all()
        assert (actual_btc.features.index == self.btc_tsa_instance.features.index).all()

        # test col names
        if trend == 'n':
            assert actual.features.columns[0] == self.tsa_instance.features.columns[0]
            assert actual_btc.features.columns[0] == self.btc_tsa_instance.features.columns[0]
        elif trend == 'c':
            assert actual.features.columns[0] == 'const'
            assert actual_btc.features.columns[0] == 'const'
        elif trend == 't':
            assert actual.features.columns[0] == 'trend'
            assert actual_btc.features.columns[0] == 'trend'
        elif trend == 'ct':
            assert actual.features.columns[0] == 'const'
            assert actual.features.columns[1] == 'trend'
            assert actual_btc.features.columns[0] == 'const'
            assert actual_btc.features.columns[1] == 'trend'
        elif trend == 'ctt':
            assert actual.features.columns[0] == 'const'
            assert actual.features.columns[1] == 'trend'
            assert actual.features.columns[2] == 'trend_squared'
            assert actual_btc.features.columns[0] == 'const'
            assert actual_btc.features.columns[1] == 'trend'
            assert actual_btc.features.columns[2] == 'trend_squared'

    @pytest.mark.parametrize("trend", [1, 'x', 'cttt'])
    def test_add_trend_param_errors(self, price_mom, spot_ret, trend) -> None:
        """
        Test add trend function parameter errors.
        """
        # test if trend is not a string
        with pytest.raises(ValueError):
            TSA(spot_ret.close, price_mom, trend=trend).add_trend()

    @pytest.mark.parametrize("cov_type", ['nonrobust', 'HC0', 'HC1', 'HC2', 'HC3', 'HAC'])
    def test_fit_ols(self, cov_type):
        """
        Test fit_ols function.
        """
        # get actual and expected
        if cov_type == 'HAC':
            actual = self.tsa_instance.fit_ols(cov_type=cov_type, cov_kwds={'maxlags': 1})
            actual_btc = self.btc_tsa_instance.fit_ols(cov_type=cov_type, cov_kwds={'maxlags': 1})
        else:
            actual = self.tsa_instance.fit_ols(cov_type=cov_type)
            actual_btc = self.btc_tsa_instance.fit_ols(cov_type=cov_type)

        # dtypes
        assert isinstance(actual, dict)
        assert isinstance(actual_btc, statsmodels.regression.linear_model.RegressionResultsWrapper)

        # test shape
        assert len(actual) == self.tsa_instance.index.droplevel(0).unique().shape[0]

    def test_fit_recursive_ols(self):
        """
        Test fit_recursive_ols function.
        """
        # get actual and expected
        actual = self.tsa_instance.fit_recursive_ols()
        actual_btc = self.btc_tsa_instance.fit_recursive_ols()

        # dtypes
        assert isinstance(actual, dict)
        assert isinstance(actual_btc, statsmodels.regression.recursive_ls.RecursiveLSResultsWrapper)

        # test shape
        assert len(actual) == self.tsa_instance.index.droplevel(0).unique().shape[0]

    def test_fit_rolling_ols(self):
        """
        Test fit_rolling_ols function.
        """
        # get actual and expected
        actual = self.tsa_instance.fit_rolling_ols()
        actual_btc = self.btc_tsa_instance.fit_rolling_ols()

        # dtypes
        assert isinstance(actual, dict)
        assert isinstance(actual_btc, statsmodels.regression.rolling.RollingRegressionResults)

        # test shape
        assert len(actual) == self.tsa_instance.index.droplevel(0).unique().shape[0]

    @pytest.mark.parametrize("output", ['params', 'pvalues', 'predict', 'resid', 'rsquared', 'f_pvalue', 'summary'])
    def test_ols(self, output) -> None:
        """
        Test ols function.
        """
        # get actual and expected
        actual = self.tsa_instance.ols(output=output)
        actual_btc = self.btc_tsa_instance.ols(output=output)

        # test shape
        if output == 'params' or output == 'pvalues':
            assert actual.shape[0] == self.tsa_instance.index.droplevel(0).unique().shape[0]
            assert actual.shape[1] == self.tsa_instance.features.shape[1]
            assert actual_btc.shape[0] == self.btc_tsa_instance.features.shape[1]
        elif output == 'predict' or output == 'resid':
            assert actual.shape == self.tsa_instance.target.to_frame().shape
            assert actual_btc.shape == self.btc_tsa_instance.target.to_frame().shape
        elif output == 'rsquared' or output == 'f_pvalue' or output == 'summary':
            assert actual.shape[0] == self.tsa_instance.index.droplevel(0).unique().shape[0]

        # test dtypes
        if output == 'params' or output == 'pvalues':
            assert isinstance(actual, pd.DataFrame)
            assert isinstance(actual_btc, pd.Series)
            assert (actual.dtypes == np.float64).all()
            assert actual_btc.dtypes == np.float64
        elif output in ['predict', 'resid']:
            assert isinstance(actual, pd.DataFrame)
            assert isinstance(actual_btc, pd.DataFrame)
            assert (actual.dtypes == np.float64).all()
            assert (actual_btc.dtypes == np.float64).all()
        elif output == 'rsquared':
            assert isinstance(actual, pd.DataFrame)
            assert isinstance(actual_btc, float)
        elif output == 'f_pvalue':
            assert actual_btc.dtype == np.float64
        elif output == 'summary':
            assert isinstance(actual, pd.DataFrame)
            assert isinstance(actual_btc, statsmodels.iolib.summary.Summary)

        # test values
        if output == 'pvalues':
            assert ((actual.abs() >= 0) & (actual.abs() <= 1)).all().all()
            assert actual_btc.abs().between(0, 1).all()
        elif output == 'rsquared':
            assert ((actual.abs() >= 0) & (actual.abs() <= 1)).all().all()
            assert 0 <= abs(actual_btc) <= 1

    @pytest.mark.parametrize("output", ['params', 'pvalues', 'predict', 'resid', 'rsquared', 'summary'])
    def test_expanding_ols(self, output) -> None:
        """
        Test expanding_ols function.
        """
        # get actual and expected
        actual = self.tsa_instance.expanding_ols(output=output)
        actual_btc = self.btc_tsa_instance.expanding_ols(output=output)

        # test shape
        if output == 'params':
            assert actual.shape == self.tsa_instance.features.shape
            assert actual_btc.shape == self.btc_tsa_instance.features.shape
        elif output == 'pvalues':
            assert actual.shape[0] == self.tsa_instance.index.droplevel(0).unique().shape[0]
            assert actual.shape[1] == self.tsa_instance.features.shape[1]
            assert actual_btc.shape[0] == self.btc_tsa_instance.features.shape[1]
        elif output == 'resid' or output == 'predict':
            assert actual.shape == self.tsa_instance.target.to_frame().shape
            assert actual_btc.shape == self.btc_tsa_instance.target.to_frame().shape
        elif output == 'rsquared' or output == 'summary':
            assert actual.shape[0] == self.tsa_instance.index.droplevel(0).unique().shape[0]

        # test dtypes
        if output == 'params':
            assert isinstance(actual, pd.DataFrame)
            assert isinstance(actual_btc, pd.DataFrame)
            assert (actual.dtypes == np.float64).all()
            assert (actual_btc.dtypes == np.float64).all()
        elif output == 'pvalues':
            assert isinstance(actual, pd.DataFrame)
            assert isinstance(actual_btc, pd.Series)
            assert (actual.dtypes == np.float64).all()
            assert actual_btc.dtypes == np.float64
        elif output in ['predict', 'resid']:
            assert isinstance(actual, pd.DataFrame)
            assert isinstance(actual_btc, pd.DataFrame)
            assert (actual.dtypes == np.float64).all()
            assert (actual_btc.dtypes == np.float64).all()
        elif output == 'rsquared':
            assert isinstance(actual, pd.DataFrame)
            assert isinstance(actual_btc, float)
        elif output == 'summary':
            assert isinstance(actual, pd.DataFrame)
            assert isinstance(actual_btc, statsmodels.iolib.summary.Summary)

        # test values
        if output == 'params':
            assert actual.isna().sum().sum() == 0
            assert actual_btc.isna().sum().sum() == 0
        elif output == 'pvalues':
            assert ((actual.dropna() >= 0) & (actual.dropna() <= 1)).all().all()
            assert actual_btc.between(0, 1).all()
        elif output == 'predict' or output == 'resid':
            assert actual.isna().sum().sum() == 0
            assert actual_btc.isna().sum().sum() == 0
            assert np.allclose(actual.loc[:, 'BTC', :], actual_btc, equal_nan=True)
        elif output == 'rsquared':
            assert 0 <= abs(actual_btc) <= 1

    @pytest.mark.parametrize("output", ['params', 'pvalues', 'rsquared', 'predict', 'resid'])
    def test_rolling_ols(self, output) -> None:
        """
        Test rolling_ols function.
        """
        # get actual and expected
        actual = self.tsa_instance.rolling_ols(output=output)
        actual_btc = self.btc_tsa_instance.rolling_ols(output=output)

        # test shape
        if output in ['params', 'pvalues']:
            assert actual.shape == self.tsa_instance.features.shape
            assert actual_btc.shape == self.btc_tsa_instance.features.shape
        elif output in ['rsquared', 'predict', 'resid']:
            assert actual.shape[0] == self.tsa_instance.target.shape[0]
            assert actual.shape[1] == 1
            assert actual_btc.shape[0] == self.btc_tsa_instance.target.shape[0]
            assert actual_btc.shape[1] == 1

        # test dtypes
        assert isinstance(actual, pd.DataFrame)
        assert (actual.dtypes == np.float64).all()
        assert isinstance(actual_btc, pd.DataFrame)
        assert (actual_btc.dtypes == np.float64).all()

        # test values
        if output in ['params', 'pvalues', 'rsquared', 'predict', 'resid']:
            assert np.allclose(actual.loc[:, 'BTC', :], actual_btc, equal_nan=True)
        if output in ['pvalues', 'rsquared']:
            assert ((actual.dropna() >= 0) & (actual.dropna() <= 1)).all().all()
            assert ((actual_btc.dropna() >= 0) & (actual_btc.dropna() <= 1)).all().all()

    def test_adf_test(self) -> None:
        """
        Test adf_test function.
        """
        # get actual and expected
        actual = self.tsa_instance.adf_test()
        actual_btc = self.btc_tsa_instance.adf_test()

        # test shape
        assert actual.shape[0] == self.tsa_instance.index.droplevel(0).unique().shape[0] * \
               (self.tsa_instance.features.shape[1] + 1)
        assert actual_btc.shape[0] == self.btc_tsa_instance.features.shape[1] + 1
        assert actual.shape[1] == 7
        assert actual_btc.shape[1] == 7

        # test dtypes
        assert isinstance(actual, pd.DataFrame)
        assert isinstance(actual_btc, pd.DataFrame)

        # test index
        assert set(actual.index.droplevel(1).unique()) == set(self.tsa_instance.index.droplevel(0).unique())
        assert set(actual.index.droplevel(0).unique()) == set(self.tsa_instance.data.columns)
        assert set(actual_btc.index) == set(self.btc_tsa_instance.data.columns)

        # test col names
        assert (actual.columns == ['adf', 'p-val', 'lags', 'nobs', '1%', '5%', '10%']).all()
        assert (actual_btc.columns == ['adf', 'p-val', 'lags', 'nobs', '1%', '5%', '10%']).all()

    def test_granger_causality(self) -> None:
        """
        Test granger_causality function.
        """
        # get actual and expected
        actual_btc = self.btc_tsa_instance.granger_causality()

        # test shape
        assert actual_btc.shape[0] == self.btc_tsa_instance.features.shape[1]
        assert actual_btc.shape[1] == 2

        # test dtypes
        assert isinstance(actual_btc, pd.DataFrame)

        # test index
        assert set(actual_btc.index) == set(self.btc_tsa_instance.features.columns)

        # test col names
        assert (actual_btc.columns == ['ssr_ftest', 'p-val']).all()

    def test_granger_causality_param_errors(self) -> None:
        """
        Test granger_causality function parameter errors.
        """
        # test if target is not a pd.Series
        with pytest.raises(TypeError):
            self.tsa_instance.granger_causality()
