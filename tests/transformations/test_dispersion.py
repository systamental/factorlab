import pytest
import pandas as pd
import numpy as np

from factorlab.transformations.returns import Returns
from factorlab.transformations.dispersion import (
    StandardDeviation,
    Quantile,
    InterquartileRange as IQR,
    MedianAbsoluteDeviation,
    MinMax,
    Variance,
    AverageTrueRange
)


@pytest.fixture
def binance_spot():
    """
    Fixture for crypto OHLCV prices.
    """
    # read csv from datasets/data
    df = pd.read_csv("../datasets/data/binance_spot_prices.csv",
                     index_col=['date', 'ticker'],
                     parse_dates=['date']).loc[:, : 'close']

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
def binance_spot_returns(binance_spot):
    """
    Fixture for crypto OHLCV returns.
    """
    # compute returns
    return binance_spot.groupby(level=1).pct_change().dropna().sort_index()


@pytest.fixture
def btc_spot_prices(binance_spot):
    """
    Fixture for BTC OHLCV prices.
    """
    # read csv from datasets/data
    return binance_spot.loc[:, 'BTC', :]


@pytest.fixture
def btc_spot_returns(btc_spot_prices):
    """
    Fixture for BTC OHLCV returns.
    """
    # compute returns
    return btc_spot_prices.pct_change().dropna()


@pytest.mark.parametrize("axis, window_type",
                         [('ts', 'fixed'), ('ts', 'expanding'), ('ts', 'rolling'), ('cs', 'fixed')])
def test_standard_deviation(binance_spot, btc_spot_prices, axis, window_type) -> None:
    """
    Test standard deviation computation.
    """
    # get actual and expected
    actual = StandardDeviation(target_col='close', axis=axis, window_type=window_type).compute(binance_spot)
    if axis != 'cs':
        actual_btc = StandardDeviation(target_col='close', axis=axis, window_type=window_type).compute(btc_spot_prices)

    # window type
    if axis == 'ts' and window_type == 'fixed':
        assert actual.shape[0] == binance_spot.shape[0]  # shape
        assert actual_btc.shape[1] == btc_spot_prices.shape[1] + 1
        assert actual.close_std.equals(binance_spot.close.groupby(level=1).transform('std'))  # values
        assert np.allclose(actual_btc.close_std.mean(), btc_spot_prices.close.std())
        assert np.allclose(actual.loc[:, 'BTC', :].close_std, actual_btc.close_std, equal_nan=True)
        assert isinstance(actual, pd.DataFrame)  # dtypes
        assert isinstance(actual_btc, pd.DataFrame)
        assert all(actual.dtypes == np.float64)
        assert all(actual_btc.dtypes == np.float64)
        assert actual.index.equals(binance_spot.index)  # index
        assert all(actual_btc.index == btc_spot_prices.index)

    elif axis == 'ts' and window_type == 'expanding':
        assert actual.shape[0] == binance_spot.shape[0]  # shape
        assert actual.shape[1] == binance_spot.shape[1] + 1  # shape
        assert actual_btc.shape[0] == btc_spot_prices.shape[0]
        assert np.allclose(actual.loc[:, 'BTC', :].close_std, actual_btc.close_std, equal_nan=True)  # values
        assert isinstance(actual, pd.DataFrame)  # dtypes
        assert isinstance(actual_btc, pd.DataFrame)
        assert all(actual.dtypes == np.float64)
        assert all(actual_btc.dtypes == np.float64)
        assert all(actual.index == binance_spot.index)  # index
        assert all(actual_btc.index == btc_spot_prices.index)
        assert all(actual.columns[:-1] == binance_spot.columns)  # cols
        assert all(actual_btc.columns[:-1] == btc_spot_prices.columns)

    elif axis == 'ts' and window_type == 'rolling':
        assert actual.shape[0] == binance_spot.shape[0]  # shape
        assert actual.shape[1] == binance_spot.shape[1] + 1  # shape
        assert actual_btc.shape[0] == btc_spot_prices.shape[0]
        assert np.allclose(actual.loc[:, 'BTC', :].close_std, actual_btc.close_std, equal_nan=True)  # values
        assert isinstance(actual, pd.DataFrame)  # dtypes
        assert isinstance(actual_btc, pd.DataFrame)
        assert all(actual.dtypes == np.float64)
        assert all(actual_btc.dtypes == np.float64)
        assert all(actual.index == binance_spot.index)  # index
        assert all(actual_btc.index == btc_spot_prices.index)
        assert all(actual.columns[:-1] == binance_spot.columns)  # cols
        assert all(actual_btc.columns[:-1] == btc_spot_prices.columns)

    if axis == 'cs':
        assert actual.shape[0] == binance_spot.shape[0]  # shape
        assert actual.shape[1] == binance_spot.shape[1] + 1  # shape
        assert isinstance(actual, pd.DataFrame)  # dtypes
        assert all(actual.dtypes == np.float64)
        assert all(actual.index == binance_spot.index)  # index
        assert all(actual.columns[:-1] == binance_spot.columns)  # cols


@pytest.mark.parametrize("q, axis, window_type",
                         [(0.05, 'ts', 'fixed'), (0.05, 'ts', 'expanding'), (0.05, 'ts', 'rolling'),
                          (0.05, 'cs', 'fixed')]
                         )
def test_quantile(binance_spot, btc_spot_prices, q, axis, window_type) -> None:
    """
    Test quantile computation.
    """
    # get actual and expected
    actual = Quantile(target_col='close', q=q, axis=axis, window_type=window_type).compute(binance_spot)
    if axis != 'cs':
        actual_btc = Quantile(target_col='close', q=q, axis=axis, window_type=window_type).compute(btc_spot_prices)

    # window type
    if axis == 'ts' and window_type == 'fixed':
        assert actual.shape[0] == binance_spot.shape[0]  # shape
        assert actual_btc.shape[1] == btc_spot_prices.shape[1] + 1
        assert np.allclose(actual['quantile'], binance_spot.close.groupby(level=1).transform('quantile', 0.05))  # values
        assert np.allclose(actual_btc['quantile'].mean(), btc_spot_prices.close.quantile(0.05))
        assert np.allclose(actual.loc[:, 'BTC', :]['quantile'], actual_btc['quantile'], equal_nan=True)
        assert isinstance(actual, pd.DataFrame)  # dtypes
        assert isinstance(actual_btc, pd.DataFrame)
        assert all(actual.dtypes == np.float64)
        assert all(actual_btc.dtypes == np.float64)
        assert actual.index.equals(binance_spot.index)  # index
        assert all(actual_btc.index == btc_spot_prices.index)

    elif axis == 'ts' and window_type == 'expanding':
        assert actual.shape[0] == binance_spot.shape[0]  # shape
        assert actual.shape[1] == binance_spot.shape[1] + 1  # shape
        assert actual_btc.shape[0] == btc_spot_prices.shape[0]
        assert np.allclose(actual.loc[:, 'BTC', :]['quantile'], actual_btc['quantile'], equal_nan=True)  # values
        assert isinstance(actual, pd.DataFrame)  # dtypes
        assert isinstance(actual_btc, pd.DataFrame)
        assert all(actual.dtypes == np.float64)
        assert all(actual_btc.dtypes == np.float64)
        assert all(actual.index == binance_spot.index)  # index
        assert all(actual_btc.index == btc_spot_prices.index)
        assert all(actual.columns[:-1] == binance_spot.columns)  # cols
        assert all(actual_btc.columns[:-1] == btc_spot_prices.columns)

    elif axis == 'ts' and window_type == 'rolling':
        assert actual.shape[0] == binance_spot.shape[0]  # shape
        assert actual.shape[1] == binance_spot.shape[1] + 1  # shape
        assert actual_btc.shape[0] == btc_spot_prices.shape[0]
        assert np.allclose(actual.loc[:, 'BTC', :]['quantile'], actual_btc['quantile'], equal_nan=True)  # values
        assert isinstance(actual, pd.DataFrame)  # dtypes
        assert isinstance(actual_btc, pd.DataFrame)
        assert all(actual.dtypes == np.float64)
        assert all(actual_btc.dtypes == np.float64)
        assert all(actual.index == binance_spot.index)  # index
        assert all(actual_btc.index == btc_spot_prices.index)
        assert all(actual.columns[:-1] == binance_spot.columns)  # cols
        assert all(actual_btc.columns[:-1] == btc_spot_prices.columns)

    if axis == 'cs':
        assert actual.shape[0] == binance_spot.shape[0]  # shape
        assert actual.shape[1] == binance_spot.shape[1] + 1  # shape
        assert isinstance(actual, pd.DataFrame)  # dtypes
        assert all(actual.dtypes == np.float64)
        assert all(actual.index == binance_spot.index)  # index
        assert all(actual.columns[:-1] == binance_spot.columns)  # cols


@pytest.mark.parametrize("axis, window_type",
                         [('ts', 'fixed'), ('ts', 'expanding'), ('ts', 'rolling'), ('cs', 'fixed')]
                         )
def test_iqr(binance_spot, btc_spot_prices, axis, window_type) -> None:
    """
    Test interquartile range (IQR) computation.
    """
    # get actual and expected
    actual = IQR(target_col='close', axis=axis, window_type=window_type).compute(binance_spot)
    if axis != 'cs':
        actual_btc = IQR(target_col='close', axis=axis, window_type=window_type).compute(btc_spot_prices)

    # window type
    if axis == 'ts' and window_type == 'fixed':
        assert actual.shape[0] == binance_spot.shape[0]  # shape
        assert actual_btc.shape[1] == btc_spot_prices.shape[1] + 1
        assert np.allclose(actual.loc[:, 'BTC', :].iqr, actual_btc.iqr, equal_nan=True)
        assert isinstance(actual, pd.DataFrame)  # dtypes
        assert isinstance(actual_btc, pd.DataFrame)
        assert all(actual.dtypes == np.float64)
        assert all(actual_btc.dtypes == np.float64)
        assert actual.index.equals(binance_spot.index)  # index
        assert all(actual_btc.index == btc_spot_prices.index)

    elif axis == 'ts' and window_type == 'expanding':
        assert actual.shape[0] == binance_spot.shape[0]  # shape
        assert actual.shape[1] == binance_spot.shape[1] + 1  # shape
        assert actual_btc.shape[0] == btc_spot_prices.shape[0]
        assert np.allclose(actual.loc[:, 'BTC', :].iqr, actual_btc.iqr, equal_nan=True)  # values
        assert isinstance(actual, pd.DataFrame)  # dtypes
        assert isinstance(actual_btc, pd.DataFrame)
        assert all(actual.dtypes == np.float64)
        assert all(actual_btc.dtypes == np.float64)
        assert all(actual.index == binance_spot.index)  # index
        assert all(actual_btc.index == btc_spot_prices.index)
        assert all(actual.columns[:-1] == binance_spot.columns)  # cols
        assert all(actual_btc.columns[:-1] == btc_spot_prices.columns)

    elif axis == 'ts' and window_type == 'rolling':
        assert actual.shape[0] == binance_spot.shape[0]  # shape
        assert actual.shape[1] == binance_spot.shape[1] + 1  # shape
        assert actual_btc.shape[0] == btc_spot_prices.shape[0]
        assert np.allclose(actual.loc[:, 'BTC', :].iqr, actual_btc.iqr, equal_nan=True)  # values
        assert isinstance(actual, pd.DataFrame)  # dtypes
        assert isinstance(actual_btc, pd.DataFrame)
        assert all(actual.dtypes == np.float64)
        assert all(actual_btc.dtypes == np.float64)
        assert all(actual.index == binance_spot.index)  # index
        assert all(actual_btc.index == btc_spot_prices.index)
        assert all(actual.columns[:-1] == binance_spot.columns)  # cols
        assert all(actual_btc.columns[:-1] == btc_spot_prices.columns)

    if axis == 'cs':
        assert actual.shape[0] == binance_spot.shape[0]  # shape
        assert actual.shape[1] == binance_spot.shape[1] + 1  # shape
        assert isinstance(actual, pd.DataFrame)  # dtypes
        assert all(actual.dtypes == np.float64)
        assert all(actual.index == binance_spot.index)  # index
        assert all(actual.columns[:-1] == binance_spot.columns)  # cols


@pytest.mark.parametrize("axis, window_type",
                         [('ts', 'fixed'), ('ts', 'expanding'), ('ts', 'rolling'), ('cs', 'fixed')]
                         )
def test_mad(binance_spot, btc_spot_prices, axis, window_type) -> None:
    """
    Test mean absolute deviation (MAD) computation.
    """
    # get actual and expected
    actual = MedianAbsoluteDeviation(target_col='close', axis=axis, window_type=window_type).compute(binance_spot)
    if axis != 'cs':
        actual_btc = MedianAbsoluteDeviation(target_col='close', axis=axis, window_type=window_type).compute(btc_spot_prices)

    # window type
    if axis == 'ts' and window_type == 'fixed':
        assert actual.shape[0] == binance_spot.shape[0]  # shape
        assert actual_btc.shape[1] == btc_spot_prices.shape[1] + 1
        assert np.allclose(actual.loc[:, 'BTC', :].mad, actual_btc.mad, equal_nan=True)
        assert isinstance(actual, pd.DataFrame)  # dtypes
        assert isinstance(actual_btc, pd.DataFrame)
        assert all(actual.dtypes == np.float64)
        assert all(actual_btc.dtypes == np.float64)
        assert actual.index.equals(binance_spot.index)  # index
        assert all(actual_btc.index == btc_spot_prices.index)

    elif axis == 'ts' and window_type == 'expanding':
        assert actual.shape[0] == binance_spot.shape[0]  # shape
        assert actual.shape[1] == binance_spot.shape[1] + 1  # shape
        assert actual_btc.shape[0] == btc_spot_prices.shape[0]
        assert np.allclose(actual.loc[:, 'BTC', :].mad, actual_btc.mad, equal_nan=True)  # values
        assert isinstance(actual, pd.DataFrame)  # dtypes
        assert isinstance(actual_btc, pd.DataFrame)
        assert all(actual.dtypes == np.float64)
        assert all(actual_btc.dtypes == np.float64)
        assert all(actual.index == binance_spot.index)  # index
        assert all(actual_btc.index == btc_spot_prices.index)
        assert all(actual.columns[:-1] == binance_spot.columns)  # cols
        assert all(actual_btc.columns[:-1] == btc_spot_prices.columns)

    elif axis == 'ts' and window_type == 'rolling':
        assert actual.shape[0] == binance_spot.shape[0]  # shape
        assert actual.shape[1] == binance_spot.shape[1] + 1  # shape
        assert actual_btc.shape[0] == btc_spot_prices.shape[0]
        assert np.allclose(actual.loc[:, 'BTC', :].mad, actual_btc.mad, equal_nan=True)  # values
        assert isinstance(actual, pd.DataFrame)  # dtypes
        assert isinstance(actual_btc, pd.DataFrame)
        assert all(actual.dtypes == np.float64)
        assert all(actual_btc.dtypes == np.float64)
        assert all(actual.index == binance_spot.index)  # index
        assert all(actual_btc.index == btc_spot_prices.index)
        assert all(actual.columns[:-1] == binance_spot.columns)  # cols
        assert all(actual_btc.columns[:-1] == btc_spot_prices.columns)

    if axis == 'cs':
        assert actual.shape[0] == binance_spot.shape[0]  # shape
        assert actual.shape[1] == binance_spot.shape[1] + 1  # shape
        assert isinstance(actual, pd.DataFrame)  # dtypes
        assert all(actual.dtypes == np.float64)
        assert all(actual.index == binance_spot.index)  # index
        assert all(actual.columns[:-1] == binance_spot.columns)  # cols


@pytest.mark.parametrize("axis, window_type",
                         [('ts', 'fixed'), ('ts', 'expanding'), ('ts', 'rolling'), ('cs', 'fixed')]
                         )
def test_variance(binance_spot, btc_spot_prices, axis, window_type) -> None:
    """
    Test variance computation.
    """
    # get actual and expected
    actual = Variance(target_col='close', axis=axis, window_type=window_type).compute(binance_spot)
    if axis != 'cs':
        actual_btc = Variance(target_col='close', axis=axis, window_type=window_type).compute(btc_spot_prices)

    # window type
    if axis == 'ts' and window_type == 'fixed':
        assert actual.shape[0] == binance_spot.shape[0]  # shape
        assert actual_btc.shape[1] == btc_spot_prices.shape[1] + 1
        assert actual['var'].equals(binance_spot.groupby(level=1).transform('var').close)  # values
        assert np.allclose(actual_btc['var'].mean(), btc_spot_prices.close.var())
        assert np.allclose(actual.loc[:, 'BTC', :]['var'], actual_btc['var'], equal_nan=True)
        assert isinstance(actual, pd.DataFrame)  # dtypes
        assert isinstance(actual_btc, pd.DataFrame)
        assert all(actual.dtypes == np.float64)
        assert all(actual_btc.dtypes == np.float64)
        assert actual.index.equals(binance_spot.index)  # index
        assert all(actual_btc.index == btc_spot_prices.index)

    elif axis == 'ts' and window_type == 'expanding':
        assert actual.shape[0] == binance_spot.shape[0]  # shape
        assert actual.shape[1] == binance_spot.shape[1] + 1  # shape
        assert actual_btc.shape[0] == btc_spot_prices.shape[0]
        assert np.allclose(actual.loc[:, 'BTC', :]['var'], actual_btc['var'], equal_nan=True)  # values
        assert isinstance(actual, pd.DataFrame)  # dtypes
        assert isinstance(actual_btc, pd.DataFrame)
        assert all(actual.dtypes == np.float64)
        assert all(actual_btc.dtypes == np.float64)
        assert all(actual.index == binance_spot.index)  # index
        assert all(actual_btc.index == btc_spot_prices.index)
        assert all(actual.columns[:-1] == binance_spot.columns)  # cols
        assert all(actual_btc.columns[:-1] == btc_spot_prices.columns)

    elif axis == 'ts' and window_type == 'rolling':
        assert actual.shape[0] == binance_spot.shape[0]  # shape
        assert actual.shape[1] == binance_spot.shape[1] + 1  # shape
        assert actual_btc.shape[0] == btc_spot_prices.shape[0]
        assert np.allclose(actual.loc[:, 'BTC', :]['var'], actual_btc['var'], equal_nan=True)  # values
        assert isinstance(actual, pd.DataFrame)  # dtypes
        assert isinstance(actual_btc, pd.DataFrame)
        assert all(actual.dtypes == np.float64)
        assert all(actual_btc.dtypes == np.float64)
        assert all(actual.index == binance_spot.index)  # index
        assert all(actual_btc.index == btc_spot_prices.index)
        assert all(actual.columns[:-1] == binance_spot.columns)  # cols
        assert all(actual_btc.columns[:-1] == btc_spot_prices.columns)

    if axis == 'cs':
        assert actual.shape[0] == binance_spot.shape[0]  # shape
        assert actual.shape[1] == binance_spot.shape[1] + 1  # shape
        assert isinstance(actual, pd.DataFrame)  # dtypes
        assert all(actual.dtypes == np.float64)
        assert all(actual.index == binance_spot.index)  # index
        assert all(actual.columns[:-1] == binance_spot.columns)  # cols


@pytest.mark.parametrize("window_type",
                         ['fixed', 'expanding', 'rolling'])
def test_atr(binance_spot, btc_spot_prices, window_type) -> None:
    """
    Test compute_atr method.
    """
    # actual
    actual = AverageTrueRange(window_type=window_type).compute(binance_spot)
    actual_btc = AverageTrueRange(window_type=window_type).compute(btc_spot_prices)

    # window type
    if window_type == 'fixed':
        assert actual.shape[0] == binance_spot.shape[0]  # shape
        assert np.allclose(actual.loc[:, 'BTC', :], actual_btc, equal_nan=True)  # values
        assert isinstance(actual, pd.DataFrame)  # dtypes
        assert isinstance(actual_btc, pd.DataFrame)
        assert all(actual.dtypes == np.float64)
        assert all(actual_btc.dtypes == np.float64)
        assert actual.index.equals(binance_spot.index)  # index
        assert all(actual_btc.index == btc_spot_prices.index)

    elif window_type == 'expanding':
        assert actual.shape[0] == binance_spot.shape[0]  # shape
        assert actual_btc.shape[0] == btc_spot_prices.shape[0]
        assert actual.shape[1] == binance_spot.shape[1] + 1  # shape
        assert actual_btc.shape[1] == btc_spot_prices.shape[1] + 1
        assert np.allclose(actual.loc[:, 'BTC', :], actual_btc, equal_nan=True)  # values
        assert isinstance(actual, pd.DataFrame)  # dtypes
        assert isinstance(actual_btc, pd.DataFrame)
        assert all(actual.dtypes == np.float64)
        assert all(actual_btc.dtypes == np.float64)
        assert all(actual.index == binance_spot.index)  # index
        assert all(actual_btc.index == btc_spot_prices.index)

    elif window_type == 'rolling':
        assert actual.shape[0] == binance_spot.shape[0]  # shape
        assert actual_btc.shape[0] == btc_spot_prices.shape[0]
        assert actual.shape[1] == binance_spot.shape[1] + 1
        assert actual_btc.shape[1] == btc_spot_prices.shape[1] + 1
        assert np.allclose(actual.loc[pd.IndexSlice[:, 'BTC'], :], actual_btc, equal_nan=True)  # values
        assert isinstance(actual, pd.DataFrame)
        assert isinstance(actual_btc, pd.DataFrame)
        assert all(actual.dtypes == np.float64)
        assert all(actual_btc.dtypes == np.float64)
        assert all(actual.index == binance_spot.index)  # index
        assert all(actual_btc.index == btc_spot_prices.index)

