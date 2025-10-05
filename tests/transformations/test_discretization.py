import pytest
import pandas as pd
import numpy as np

from factorlab.transformations.discretization import (
    Quantize,
    Discretize
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
    return binance_spot.groupby(level=1).pct_change().unstack().dropna().stack().sort_index()


@pytest.fixture
def btc_spot_prices(binance_spot):
    """
    Fixture for BTC OHLCV prices.
    """
    # read csv from datasets/data
    return binance_spot.loc[:, 'BTC', :]


@pytest.fixture
def btc_spot_returns(binance_spot_returns):
    """
    Fixture for BTC OHLCV returns.
    """
    # compute returns
    return binance_spot_returns.loc[:, 'BTC', :]


@pytest.mark.parametrize("bins, axis, window_type",
                         [
                             (5, 'ts', 'fixed'),
                             (5, 'ts', 'expanding'),
                             (5, 'ts', 'rolling'),
                             (5, 'cs', 'fixed'),
                             (10, 'ts', 'fixed'),
                             (10, 'ts', 'expanding'),
                             (10, 'ts', 'rolling'),
                             (10, 'cs', 'fixed'),
                         ]
                         )
def test_quantize(bins, axis, window_type, binance_spot_returns, btc_spot_returns):
    """
    Test the Quantize transformation with various parameters.
    """
    actual = Quantize(
        input_col='close',
        bins=bins,
        axis=axis,
        window_type=window_type,
        window_size=30,
        min_periods=2
    ).fit(binance_spot_returns).transform(binance_spot_returns)

    # shape
    assert actual.shape[0] == binance_spot_returns.shape[0]
    assert actual.shape[1] == binance_spot_returns.shape[1] + 1

    if axis == 'ts':
        actual_btc = Quantize(
            input_col='close',
            bins=bins,
            axis=axis,
            window_type=window_type,
            window_size=30,
            min_periods=2
        ).fit(btc_spot_returns).transform(btc_spot_returns)
        assert actual_btc.shape[0] == btc_spot_returns.shape[0]
        assert isinstance(actual_btc, pd.DataFrame)
        assert all(actual_btc.dtypes == np.float64)
        assert actual_btc.index.equals(btc_spot_returns.index)
        assert all(actual['quantile'].unstack().nunique() == bins)  # values
        assert actual_btc['quantile'].nunique() == bins
        assert np.allclose(actual.loc[:, 'BTC', :]['quantile'], actual_btc['quantile'], equal_nan=True)
        if window_type == 'fixed':
            # percentage of values in each bin
            counts_per_bin = actual['quantile'].groupby(level=1).value_counts()  # n of values per bin
            counts_per_ticker = actual['quantile'].groupby(level=1).value_counts().groupby('ticker').sum()  # ticker vals
            bin_perc = counts_per_bin / counts_per_ticker  # percentage of values per bin per ticker
            assert ((bin_perc - (1 / bins)).abs() < 0.05).sum() / \
                   (len(actual['quantile'].index.get_level_values(1).unique()) * bins) >= 0.99
            assert (((actual_btc['quantile'].dropna().value_counts() / actual_btc['quantile'].dropna().shape[0]) - (1 / bins)).abs()
                < 0.05).all()
    else:
        assert all(actual.groupby(level=0).nunique() == bins)

    assert isinstance(actual, pd.DataFrame)  # dtypes
    assert all(actual.dtypes == np.float64)
    assert actual.index.equals(binance_spot_returns.index)  # index


@pytest.mark.parametrize("bins, axis, method, window_type",
                         [
                             (4, 'ts', 'quantile', 'fixed'),
                             (4, 'ts', 'quantile',  'expanding'),
                             (4, 'ts', 'quantile', 'rolling'),
                             (4, 'cs', 'quantile', 'fixed'),
                             (2, 'ts', 'uniform', 'fixed'),
                             (2, 'ts', 'uniform',  'expanding'),
                             (2, 'ts', 'uniform', 'rolling'),
                             (2, 'cs', 'uniform', 'fixed'),
                         ]
                         )
def test_discretize(bins, axis, method, window_type, binance_spot_returns, btc_spot_returns):
    """
    Test the Quantize transformation with various parameters.
    """
    actual = Discretize(
        input_col='close',
        output_col='quantile',
        bins=bins,
        axis=axis,
        method=method,
        window_type=window_type,
        window_size=30,
    ).fit(binance_spot_returns).transform(binance_spot_returns)

    # shape
    assert actual.shape[0] == binance_spot_returns.shape[0]
    assert actual.shape[1] == binance_spot_returns.shape[1] + 1

    if axis == 'ts':
        actual_btc = Discretize(
            input_col='close',
            output_col='quantile',
            bins=bins,
            axis=axis,
            method=method,
            window_type=window_type,
            window_size=30,
        ).fit(btc_spot_returns).transform(btc_spot_returns)

        assert actual_btc.shape[0] == btc_spot_returns.shape[0]
        assert isinstance(actual_btc, pd.DataFrame)
        assert all(actual_btc.dtypes == np.float64)
        assert actual_btc.index.equals(btc_spot_returns.index)
        assert all(actual['quantile'].unstack().nunique() == bins)  # values
        assert actual_btc['quantile'].nunique() == bins
        assert np.allclose(actual.loc[:, 'BTC', :]['quantile'], actual_btc['quantile'], equal_nan=True)
        if window_type == 'fixed':
            # percentage of values in each bin
            if method == 'quantile':
                counts_per_bin = actual['quantile'].groupby(level=1).value_counts()  # n of values per bin
                counts_per_ticker = actual['quantile'].groupby(level=1).value_counts().groupby('ticker').sum()  # ticker vals
                bin_perc = counts_per_bin / counts_per_ticker  # percentage of values per bin per ticker
                assert ((bin_perc - (1 / bins)).abs() < 0.05).sum() / \
                       (len(actual['quantile'].index.get_level_values(1).unique()) * bins) >= 0.95
                assert (((actual_btc['quantile'].dropna().value_counts() / actual_btc['quantile'].dropna().shape[0]) -
                         (1 / bins)).abs() < 0.05).all()
    else:
        assert all(actual.groupby(level=0).nunique() == bins)

    assert isinstance(actual, pd.DataFrame)  # dtypes
    assert all(actual.dtypes == np.float64)
    assert actual.index.equals(binance_spot_returns.index)  # index
