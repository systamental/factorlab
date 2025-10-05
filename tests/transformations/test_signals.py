import pytest
import pandas as pd
import numpy as np
from factorlab.transformations.returns import PctChange
from factorlab.transformations.discretization import Quantize
from factorlab.transformations.ranking import Rank
from factorlab.transformations.normalization import Percentile, ZScore, MinMaxScaler

from factorlab.transformations.signals import (
    ScoresToSignals,
    QuantilesToSignals,
    RanksToSignals
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
def btc_spot_prices(binance_spot):
    """
    Fixture for BTC OHLCV prices.
    """
    # read csv from datasets/data
    return binance_spot.loc[:, 'BTC', :]


@pytest.fixture
def binance_spot_returns(binance_spot):
    """
    Fixture for crypto OHLCV returns.
    """
    return PctChange(lags=30).compute(binance_spot)


@pytest.fixture
def btc_spot_returns(btc_spot_prices):
    """
    Fixture for BTC OHLCV returns.
    """
    return PctChange(lags=30).compute(btc_spot_prices)


@pytest.mark.parametrize("method", ['norm',  'logistic', 'adj_norm', 'tanh', 'min_max', 'percentile'])
def test_scores_to_signals(binance_spot_returns, btc_spot_returns, method):
    """
    Test ScoreToSignal transformation.
    """
    # compute scores
    if method == 'percentile':
        scores = Percentile(output_col='scores').compute(binance_spot_returns)
        scores_btc = Percentile(output_col='scores').compute(btc_spot_returns)

    elif method == 'min_max':
        scores = MinMaxScaler(output_col='scores').compute(binance_spot_returns)
        scores_btc = MinMaxScaler(output_col='scores').compute(btc_spot_returns)

    else:
        scores = ZScore(output_col='scores').compute(binance_spot_returns)
        scores_btc = ZScore(output_col='scores').compute(btc_spot_returns)

    # get actual and expected
    actual = ScoresToSignals(input_col='scores', method=method).fit(scores).transform(scores)
    actual_btc = ScoresToSignals(input_col='scores', method=method).fit(scores_btc).transform(scores_btc)

    assert actual.shape[0] == scores.shape[0]  # shape
    assert actual_btc.shape[0] == scores_btc.shape[0]
    assert actual.shape[1] == scores.shape[1] + 1
    assert actual_btc.shape[1] == scores_btc.shape[1] + 1
    assert all((actual['signal'].dropna().abs() >= 0) & (actual['signal'].dropna().abs() <= 1))  # values
    assert all((actual_btc['signal'].dropna().abs() >= 0) & (actual_btc['signal'].dropna().abs() <= 1))
    assert isinstance(actual, pd.DataFrame)  # dtypes
    assert isinstance(actual_btc, pd.DataFrame)
    assert all(actual.dtypes == np.float64)
    assert all(actual.dtypes == np.float64)
    assert all(actual.index == scores.index)
    assert all(actual_btc.index == scores_btc.index)


@pytest.mark.parametrize("axis, bins",
                        [
                             ('ts', 4),
                             ('cs', 4),
                             ('ts', 5),
                             ('cs', 5),
                             ('ts', 10),
                             ('cs', 10),
                             ('ts', None),
                             ('cs', None)
                             ]
                         )
def test_quantiles_to_signals(binance_spot_returns, btc_spot_returns, axis, bins):
    """
    Test QuantileToSignal transformation.
    """
    quantiles = Quantize(input_col='ret', axis=axis, window_type='fixed').compute(binance_spot_returns)
    actual = QuantilesToSignals(input_col='quantile').compute(quantiles)

    assert actual.shape[0] == quantiles.shape[0]  # shape
    assert actual.shape[1] == quantiles.shape[1] + 1
    assert all((actual['signal'].dropna().abs() >= 0) & (actual['signal'].dropna().abs() <= 1))  # values
    assert isinstance(actual, pd.DataFrame)  # dtypes
    assert all(actual.dtypes == np.float64)
    assert all(actual.index == quantiles.index)

    if axis == 'ts':

        quantiles_btc = Quantize(input_col='ret', axis=axis, window_type='fixed').compute(btc_spot_returns)
        actual_btc = QuantilesToSignals(input_col='quantile').compute(quantiles_btc)

        assert actual_btc.shape[0] == quantiles_btc.shape[0]
        assert actual_btc.shape[1] == quantiles_btc.shape[1] + 1
        assert all((actual_btc['signal'].dropna().abs() >= 0) & (actual_btc['signal'].dropna().abs() <= 1))
        assert isinstance(actual_btc, pd.DataFrame)
        assert all(actual_btc.dtypes == np.float64)
        assert all(actual_btc.index == quantiles_btc.index)


@pytest.mark.parametrize("axis", ['ts', 'cs'])
def test_ranks_to_signals(binance_spot_returns, btc_spot_returns, axis):
    """
    Test RankToSignal transformation.
    """
    rank = Rank(input_col='ret', axis=axis, window_type='fixed').compute(binance_spot_returns)
    actual = RanksToSignals(input_col='rank').compute(rank)

    assert actual.shape[0] == rank.shape[0]  # shape
    assert actual.shape[1] == rank.shape[1] + 1
    assert all((actual['signal'].dropna().abs() >= 0) & (actual['signal'].dropna().abs() <= 1))  # values
    assert isinstance(actual, pd.DataFrame)  # dtypes
    assert all(actual.dtypes == np.float64)
    assert all(actual.index == rank.index)

    if axis == 'ts':
        rank_btc = Rank(input_col='ret', axis=axis, window_type='fixed').compute(btc_spot_returns)
        actual_btc = RanksToSignals(input_col='rank').compute(rank_btc)

        assert actual_btc.shape[0] == rank_btc.shape[0]
        assert actual_btc.shape[1] == rank_btc.shape[1] + 1
        assert all((actual_btc['signal'].dropna().abs() >= 0) & (actual_btc['signal'].dropna().abs() <= 1))
        assert isinstance(actual_btc, pd.DataFrame)
        assert all(actual_btc.dtypes == np.float64)
        assert all(actual_btc.index == rank_btc.index)