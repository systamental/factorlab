import pytest
import pandas as pd
import numpy as np

from factorlab.transformations.normalization import (
    Center,
    ZScore,
    RobustZScore,
    ModZScore,
    MinMaxScaler,
    Percentile,
    ATRScaler,
    PowerTransform
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


@pytest.fixture
def norm_returns(binance_spot_returns):
    """
    Fixture for normalized returns.
    """
    # normalize returns
    z_score = ZScore(axis='ts', centering=True, window_type='fixed').compute(binance_spot_returns)
    z_score = z_score.unstack().dropna().stack()
    return z_score


@pytest.fixture
def norm_returns_btc(btc_spot_returns):
    """
    Fixture for normalized BTC returns.
    """
    # normalize BTC returns
    z_score = ZScore(axis='ts', centering=True, window_type='fixed').compute(btc_spot_returns)
    z_score = z_score.dropna()
    return z_score


@pytest.mark.parametrize("axis, method, window_type",
                         [('ts', 'mean', 'fixed'), ('ts', 'mean', 'expanding'), ('ts', 'mean', 'rolling'),
                          ('cs', 'mean', 'fixed'),
                          ('ts', 'median', 'fixed'), ('ts', 'median', 'expanding'),
                          ('ts', 'median', 'rolling'), ('cs', 'median', 'fixed')
                          ]
                         )
def test_center(axis, method, window_type, binance_spot_returns, btc_spot_returns):
    """
    Test the Center transformation with various axis and window_type parameters.
    """
    # get actual and expected
    actual = Center(axis=axis,
                    method=method,
                    window_type=window_type).fit(binance_spot_returns).transform(binance_spot_returns)
    actual_btc = Center(axis=axis,
                        method=method,
                        window_type=window_type).fit(btc_spot_returns).transform(btc_spot_returns)

    if axis == 'ts':
        assert actual.shape[0] == binance_spot_returns.shape[0]  # shape
        assert actual.shape[1] == binance_spot_returns.shape[1] + 1
        assert actual_btc.shape[0] == btc_spot_returns.shape[0]
        assert actual_btc.shape[1] == btc_spot_returns.shape[1] + 1
        assert np.allclose(actual.loc[:, 'BTC', :], actual_btc, equal_nan=True)
        assert isinstance(actual, pd.DataFrame)  # dtypes
        assert isinstance(actual_btc, pd.DataFrame)
        assert all(actual.dtypes == np.float64)
        assert all(actual_btc.dtypes == np.float64)
        assert all(actual.index == binance_spot_returns.index)  # index
        assert all(actual_btc.index == btc_spot_returns.index)

    elif axis == 'cs':
        assert actual.shape[0] == binance_spot_returns.shape[0]  # shape
        assert actual.shape[1] == binance_spot_returns.shape[1] + 1
        assert isinstance(actual, pd.DataFrame)  # dtypes
        assert isinstance(actual_btc, pd.DataFrame)
        assert all(actual.dtypes == np.float64)
        assert all(actual_btc.dtypes == np.float64)
        assert all(actual.index == binance_spot_returns.index)  # index
        assert all(actual_btc.index == btc_spot_returns.index)


@pytest.mark.parametrize("axis, centering, window_type",
                         [
                             ('ts', True, 'fixed'),
                             ('ts', True, 'expanding'),
                             ('ts', True, 'rolling'),
                             ('cs', True, 'fixed')
                            ]
                         )
def test_zscore(axis, centering, window_type, binance_spot_returns, btc_spot_returns):
    """
    Test the ZScore transformation with various axis and window_type parameters.
    """
    # get actual and expected
    actual = ZScore(axis=axis,
                    centering=centering,
                    window_type=window_type).fit(binance_spot_returns).transform(binance_spot_returns)

    assert actual.shape[0] == binance_spot_returns.shape[0]  # shape
    assert actual.shape[1] == binance_spot_returns.shape[1] + 3
    # values
    if axis == 'ts' and window_type == 'fixed':
        assert np.allclose(actual.zscore.groupby(level=1).std(), 1.0)

    assert isinstance(actual, pd.DataFrame)  # dtypes
    assert all(actual.dtypes == np.float64)
    assert all(actual.index == binance_spot_returns.index)  # index

    # single index
    if axis != 'cs':
        actual_btc = ZScore(axis=axis,
                            centering=centering,
                            window_type=window_type).fit(btc_spot_returns).transform(btc_spot_returns)
        assert actual_btc.shape[0] == btc_spot_returns.shape[0]
        assert actual_btc.shape[1] == btc_spot_returns.shape[1] + 3
        # values
        if axis == 'ts' and window_type == 'fixed':
            assert np.allclose(actual_btc.std().zscore, 1.0)
            if axis == 'ts':
                assert np.allclose(actual.loc[pd.IndexSlice[:, 'BTC'], :], actual_btc, equal_nan=True)
            assert isinstance(actual, pd.DataFrame)  # dtypes
            assert isinstance(actual_btc, pd.DataFrame)
            assert all(actual.dtypes == np.float64)
            assert all(actual_btc.dtypes == np.float64)
            assert all(actual_btc.index == btc_spot_returns.index)


@pytest.mark.parametrize("axis, centering, window_type",
                         [
                             ('ts', True, 'fixed'),
                             ('ts', True, 'expanding'),
                             ('ts', True, 'rolling'),
                             ('cs', True, 'fixed')
                            ]
                         )
def test_robust_zscore(axis, centering, window_type, binance_spot_returns, btc_spot_returns):
    """
    Test the RobustZScore transformation with various axis and window_type parameters.
    """
    # get actual and expected
    actual = RobustZScore(axis=axis,
                          centering=centering,
                          window_type=window_type).fit(binance_spot_returns).transform(binance_spot_returns)

    assert actual.shape[0] == binance_spot_returns.shape[0]  # shape
    assert actual.shape[1] == binance_spot_returns.shape[1] + 3
    assert isinstance(actual, pd.DataFrame)  # dtypes
    assert all(actual.dtypes == np.float64)
    assert all(actual.index == binance_spot_returns.index)  # index

    # single index
    if axis != 'cs':
        actual_btc = RobustZScore(axis=axis,
                                  centering=centering,
                                  window_type=window_type).fit(btc_spot_returns).transform(btc_spot_returns)
        assert actual_btc.shape[0] == btc_spot_returns.shape[0]
        assert actual_btc.shape[1] == btc_spot_returns.shape[1] + 3
        # values
        if axis == 'ts':
            assert np.allclose(actual.loc[pd.IndexSlice[:, 'BTC'], :], actual_btc, equal_nan=True)
            assert isinstance(actual, pd.DataFrame)  # dtypes
            assert isinstance(actual_btc, pd.DataFrame)
            assert all(actual.dtypes == np.float64)
            assert all(actual_btc.dtypes == np.float64)
            assert all(actual_btc.index == btc_spot_returns.index)


@pytest.mark.parametrize("axis, centering, window_type",
                         [
                             ('ts', True, 'fixed'),
                             ('ts', True, 'expanding'),
                             ('ts', True, 'rolling'),
                             ('cs', True, 'fixed')
                            ]
                         )
def test_modzscore(axis, centering, window_type, binance_spot_returns, btc_spot_returns):
    """
    Test the MinMaxScaler transformation with various axis and window_type parameters.
    """
    actual = ModZScore(axis=axis,
                       centering=centering,
                       window_type=window_type).fit(binance_spot_returns).transform(binance_spot_returns)

    assert actual.shape[0] == binance_spot_returns.shape[0]  # shape
    assert actual.shape[1] == binance_spot_returns.shape[1] + 3
    assert isinstance(actual, pd.DataFrame)  # dtypes
    assert all(actual.dtypes == np.float64)
    assert all(actual.index == binance_spot_returns.index)  # index

    # single index
    if axis != 'cs':
        actual_btc = ModZScore(axis=axis,
                               centering=centering,
                               window_type=window_type).fit(btc_spot_returns).transform(btc_spot_returns)
        assert actual_btc.shape[0] == btc_spot_returns.shape[0]
        assert actual_btc.shape[1] == btc_spot_returns.shape[1] + 3
        # values
        if axis == 'ts':
            assert np.allclose(actual.loc[pd.IndexSlice[:, 'BTC'], :], actual_btc, equal_nan=True)
            assert isinstance(actual, pd.DataFrame)  # dtypes
            assert isinstance(actual_btc, pd.DataFrame)
            assert all(actual.dtypes == np.float64)
            assert all(actual_btc.dtypes == np.float64)
            assert all(actual_btc.index == btc_spot_returns.index)


@pytest.mark.parametrize("axis, window_type",
                         [('ts', 'fixed'), ('ts', 'expanding'), ('ts', 'rolling'), ('cs', 'fixed')]
                         )
def test_percentile(binance_spot_returns, btc_spot_returns, axis, window_type) -> None:
    """
    Test percentile computation.
    """
    # get actual and expected
    actual = Percentile(axis=axis,
                        window_type=window_type).fit(binance_spot_returns).compute(binance_spot_returns)
    if axis == 'ts':
        actual_btc = Percentile(axis=axis,
                                window_type=window_type).fit(btc_spot_returns).compute(btc_spot_returns)
        assert actual.shape[0] == binance_spot_returns.shape[0]  # shape
        assert actual_btc.shape[0] == btc_spot_returns.shape[0]
        assert actual.shape[1] == binance_spot_returns.shape[1] + 1
        assert actual_btc.shape[1] == btc_spot_returns.shape[1] + 1
        assert np.allclose(actual.loc[:, 'BTC', :], actual_btc, equal_nan=True)  # values
        assert ((actual['percentile'] <= 1.0) & (actual['percentile'] >= 0.0)).all().all()
        assert ((actual_btc['percentile'] <= 1.0) & (actual_btc['percentile'] >= 0.0)).all().all()
        assert isinstance(actual, pd.DataFrame)  # dtypes
        assert isinstance(actual_btc, pd.DataFrame)
        assert all(actual.dtypes == np.float64)
        assert all(actual_btc.dtypes == np.float64)
        assert all(actual.index == binance_spot_returns.index)  # index
        assert all(actual_btc.index == btc_spot_returns.index)

    else:
        assert actual.shape[0] == binance_spot_returns.shape[0]  # shape
        assert actual.shape[1] == binance_spot_returns.shape[1] + 1
        assert ((actual['percentile'] <= 1.0) & (actual['percentile'] >= 0.0)).all().all()
        assert isinstance(actual, pd.DataFrame)  # dtypes
        assert all(actual.dtypes == np.float64)
        assert all(actual.index == binance_spot_returns.index)  # index


@pytest.mark.parametrize("axis, centering, window_type",
                         [
                             ('ts', True, 'fixed'),
                             ('ts', True, 'expanding'),
                             ('ts', True, 'rolling'),
                             ('cs', True, 'fixed')
                            ]
                         )
def test_minmax_scaler(axis, centering, window_type, binance_spot_returns, btc_spot_returns):
    """
    Test the MinMaxScaler transformation with various axis and window_type parameters.
    """
    # get actual and expected
    actual = MinMaxScaler(axis=axis,
                          centering=centering,
                          window_type=window_type).fit(binance_spot_returns).transform(binance_spot_returns)
    if axis == 'ts':
        actual_btc = MinMaxScaler(axis=axis,
                                  window_type=window_type).fit(btc_spot_returns).compute(btc_spot_returns)
        assert actual.shape[0] == binance_spot_returns.shape[0]  # shape
        assert actual_btc.shape[0] == btc_spot_returns.shape[0]
        assert actual.shape[1] == binance_spot_returns.shape[1] + 3
        assert actual_btc.shape[1] == btc_spot_returns.shape[1] + 3
        assert np.allclose(actual.loc[:, 'BTC', :], actual_btc, equal_nan=True)  # values
        assert ((actual['min_max_scaled'].dropna() <= 1.0) & (actual['min_max_scaled'].dropna() >= 0.0)).all().all()
        assert ((actual_btc['min_max_scaled'].dropna() <= 1.0) &
                (actual_btc['min_max_scaled'].dropna() >= 0.0)).all().all()
        assert isinstance(actual, pd.DataFrame)  # dtypes
        assert isinstance(actual_btc, pd.DataFrame)
        assert all(actual.dtypes == np.float64)
        assert all(actual_btc.dtypes == np.float64)
        assert all(actual.index == binance_spot_returns.index)  # index
        assert all(actual_btc.index == btc_spot_returns.index)

    else:
        assert actual.shape[0] == binance_spot_returns.shape[0]  # shape
        assert actual.shape[1] == binance_spot_returns.shape[1] + 3
        assert ((actual['min_max_scaled'].dropna() <= 1.0) & (actual['min_max_scaled'].dropna() >= 0.0)).all().all()
        assert isinstance(actual, pd.DataFrame)  # dtypes
        assert all(actual.dtypes == np.float64)
        assert all(actual.index == binance_spot_returns.index)  # index


@pytest.mark.parametrize("centering, window_type",
                         [
                             (True, 'fixed'),
                             (True, 'expanding'),
                             (True, 'rolling'),
                             (True, 'fixed')
                            ]
                         )
def test_atrscaler(centering, window_type, binance_spot_returns, btc_spot_returns):
    """
    Test the ATRScaler transformation with various centering and window_type parameters.
    """
    # get actual and expected
    actual = ATRScaler(centering=centering,
                       window_type=window_type).fit(binance_spot_returns).transform(binance_spot_returns)
    actual_btc = ATRScaler(window_type=window_type).fit(btc_spot_returns).compute(btc_spot_returns)

    assert actual.shape[0] == binance_spot_returns.shape[0]  # shape
    assert actual_btc.shape[0] == btc_spot_returns.shape[0]
    assert actual.shape[1] == binance_spot_returns.shape[1] + 4
    assert actual_btc.shape[1] == btc_spot_returns.shape[1] + 4
    assert np.allclose(actual.loc[:, 'BTC', :], actual_btc, equal_nan=True)  # values
    assert isinstance(actual, pd.DataFrame)  # dtypes
    assert isinstance(actual_btc, pd.DataFrame)
    assert all(actual.dtypes == np.float64)
    assert all(actual_btc.dtypes == np.float64)
    assert all(actual.index == binance_spot_returns.index)  # index
    assert all(actual_btc.index == btc_spot_returns.index)


@pytest.mark.parametrize("method, axis, window_type",
                         [
                             ('yeo-johnson', 'ts', 'fixed'),
                             ('box-cox', 'ts', 'fixed'),
                             ('yeo-johnson', 'cs', 'fixed'),
                             ('box-cox', 'cs', 'fixed'),
                             ('yeo-johnson', 'cs', 'expanding'),
                             ('box-cox', 'cs', 'expanding'),
                             ('yeo-johnson', 'ts', 'rolling'),
                             ('box-cox', 'ts', 'rolling')
                          ]
                         )
def test_transformations(method, axis, window_type, norm_returns, norm_returns_btc):
    """
    Test the transformation methods with various parameters.
    """
    norm_returns = norm_returns.unstack().dropna().stack()
    norm_returns_btc = norm_returns_btc.dropna()

    # get actual and expected
    actual = PowerTransform(method=method,
                            axis=axis,
                            window_type=window_type).fit(norm_returns).transform(norm_returns)
    if axis == 'ts':
        actual_btc = PowerTransform(method=method,
                                    axis=axis,
                                    window_type=window_type).fit(norm_returns_btc).transform(norm_returns_btc)
        assert actual_btc.shape[0] == norm_returns_btc.shape[0]
        assert all(actual_btc.dtypes == np.float64)
        assert isinstance(actual_btc, pd.DataFrame)
        assert actual_btc.index.equals(norm_returns_btc.index)

    assert isinstance(actual, pd.DataFrame)  # dtypes
    assert all(actual.dtypes == np.float64)
    assert actual.index.equals(norm_returns.index)  # index
    assert actual.shape[0] == norm_returns.shape[0]  # shape





