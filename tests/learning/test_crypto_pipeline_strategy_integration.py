from pathlib import Path

import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression

from factorlab.analytics.performance import Performance
from factorlab.backtesting.engine import BacktestEngine
from factorlab.core.pipeline import Pipeline
from factorlab.factors.library.liquidity.liquidity import Liquidity
from factorlab.factors.library.trend.trend import Trend
from factorlab.factors.library.volatility.volatility import Vol
from factorlab.features.transforms.returns import Returns
from factorlab.learning import (
    ExpandingIncrementPanelSplit,
    FeatureSelector,
    RegressionLearner,
    WalkForwardLearner,
)
from factorlab.learning.sklearn_wrapper import SKLearnWrapper
from factorlab.portfolio.cost_models.fixed import FixedCommissionModel
from factorlab.portfolio.optimization.signal_weighted import SignalWeighted
from factorlab.signals.generator import SignalGenerator
from factorlab.strategy.strategy import LearningSpec, StrategySpec
from factorlab.targets import ForwardReturnTarget, ForwardTargetSpec


def _find_real_crypto_ohlc_path() -> Path | None:
    repo_root = Path(__file__).resolve().parents[2]
    candidates = [
        repo_root.parent / "data" / "systamental" / "crypto" / "markets" / "perpetual_futures" / "daily" / "ohlcv" / "clean" / "binance.parquet",
        repo_root.parent / "data" / "systamental" / "crypto" / "markets" / "spot" / "daily" / "ohlcv" / "clean" / "binance.parquet",
        repo_root / "data" / "systamental" / "crypto" / "markets" / "perpetual_futures" / "daily" / "ohlcv" / "clean" / "binance.parquet",
        repo_root / "data" / "systamental" / "crypto" / "markets" / "spot" / "daily" / "ohlcv" / "clean" / "binance.parquet",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _load_real_crypto_panel(
    n_days: int = 320,
    preferred_tickers: tuple[str, ...] = ("BTC", "ETH", "XRP", "ADA", "LINK"),
) -> pd.DataFrame | None:
    path = _find_real_crypto_ohlc_path()
    if path is None:
        return None

    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.MultiIndex):
        return None
    if "close" not in df.columns or "volume" not in df.columns:
        return None

    if "funding_rate" not in df.columns:
        df = df.copy()
        df["funding_rate"] = 0.0

    df = df[["close", "volume", "funding_rate"]].sort_index()
    if df.index.names != ["date", "ticker"]:
        df.index = df.index.set_names(["date", "ticker"])

    available_tickers = set(df.index.get_level_values("ticker").unique())
    selected = [ticker for ticker in preferred_tickers if ticker in available_tickers]
    if len(selected) < 3:
        counts = df.groupby(level="ticker").size().sort_values(ascending=False)
        selected = list(counts.head(5).index)
    if len(selected) < 3:
        return None

    df = df.loc[(slice(None), selected), :].sort_index()
    end_date = df.index.get_level_values("date").max()
    start_date = end_date - pd.Timedelta(days=n_days - 1)
    date_idx = df.index.get_level_values("date")
    df = df[(date_idx >= start_date) & (date_idx <= end_date)].sort_index()

    df = df[~df.index.duplicated(keep="last")]
    df = df.dropna(subset=["close", "volume"])
    if df.empty or df.index.get_level_values("ticker").nunique() < 3:
        return None
    return df


@pytest.fixture(scope="module")
def crypto_panel() -> pd.DataFrame:
    real_data = _load_real_crypto_panel()
    if real_data is None:
        pytest.skip("Real crypto OHLCV parquet data not available under ../data or ./data.")
    return real_data


def test_crypto_factor_to_learning_to_strategy_evaluation(crypto_panel: pd.DataFrame):
    data = crypto_panel.copy()

    pipeline = Pipeline(
        steps=[
            (
                "returns",
                Returns(method="pct", input_col="close", output_col="ret", lags=1),
            ),
            (
                "trend",
                Trend(
                    method="price_momentum",
                    input_col="close",
                    window_size=5,
                    scale=False,
                ),
            ),
            (
                "volatility",
                Vol(
                    method="std",
                    input_col="ret",
                    window_type="rolling",
                    window_size=20,
                    annualize=False,
                ),
            ),
            (
                "liquidity",
                Liquidity(
                    method="amihud",
                    return_col="ret",
                    price_col="close",
                    volume_col="volume",
                ),
            ),
            (
                "learner",
                WalkForwardLearner(
                    model=Ridge(alpha=1.0),
                    feature_cols=["PriceMomentum_5", "STD_rolling_20", "AmihudIlliquidity"],
                    target_transform=ForwardReturnTarget(
                        input_col="close",
                        output_col="fwd_ret",
                        horizon=1,
                    ),
                    prediction_col="forecast",
                    window_type="expanding",
                    min_train_periods=45,
                    retrain_interval=1,
                    min_train_samples=150,
                ),
            ),
            (
                "signal",
                SignalGenerator(
                    signal_type="sign",
                    input_col="forecast",
                    output_col="signal",
                ),
            ),
        ]
    )

    strategy = StrategySpec(
        name="CryptoLearningIntegration",
        data_pipeline=pipeline,
        optimizer=SignalWeighted(window_size=20),
        cost_model=FixedCommissionModel(rate=0.0005),
        rebal_freq="d",
    )

    engine = BacktestEngine(
        config=strategy,
        data=data,
        initial_capital=1_000_000.0,
        verbose=False,
        ann_factor=365,
    )

    results = engine.run_backtest()
    assert results
    assert "account_value" in results
    assert "weights" in results
    assert len(results["account_value"]) > 10
    assert results["account_value"].iloc[-1] > 0

    strategy_returns = engine.portfolio_return.dropna().to_frame("strategy")
    perf_table = Performance(strategy_returns, ret_type="simple", ann_factor=365).get_table(metrics="key_metrics")
    assert not perf_table.empty
    assert "Sharpe ratio" in perf_table.columns


def test_crypto_strategy_with_walk_forward_runner_learning_spec(crypto_panel: pd.DataFrame):
    data = crypto_panel.copy()

    pipeline = Pipeline(
        steps=[
            (
                "returns",
                Returns(method="pct", input_col="close", output_col="ret", lags=1),
            ),
            (
                "target",
                ForwardReturnTarget(input_col="close", output_col="target", horizon=1),
            ),
            (
                "learner",
                SKLearnWrapper(
                    model=LinearRegression(),
                    feature_cols=["ret"],
                    target_col="target",
                    output_col="forecast",
                ),
            ),
            (
                "signal",
                SignalGenerator(
                    signal_type="sign",
                    input_col="forecast",
                    output_col="signal",
                ),
            ),
        ]
    )

    learning_spec = LearningSpec(
        splitter=ExpandingIncrementPanelSplit(
            train_intervals=1,
            test_size=1,
            min_train_periods=45,
            lookahead=1,
            embargo=0,
        ),
        date_level=0,
        show_progress=False,
    )

    strategy = StrategySpec(
        name="CryptoLearningSpecIntegration",
        data_pipeline=pipeline,
        optimizer=SignalWeighted(window_size=20),
        cost_model=FixedCommissionModel(rate=0.0005),
        learning_spec=learning_spec,
        rebal_freq="d",
    )

    engine = BacktestEngine(
        config=strategy,
        data=data,
        initial_capital=1_000_000.0,
        verbose=False,
        ann_factor=365,
    )

    results = engine.run_backtest()
    assert results
    assert "account_value" in results
    assert engine.walk_forward_runner is not None
    assert len(engine.walk_forward_runner.fold_info) > 0


def test_crypto_strategy_learning_spec_with_target_spec(crypto_panel: pd.DataFrame):
    data = crypto_panel.copy()

    pipeline = Pipeline(
        steps=[
            (
                "returns",
                Returns(method="pct", input_col="close", output_col="ret", lags=1),
            ),
            (
                "learner",
                SKLearnWrapper(
                    model=LinearRegression(),
                    feature_cols=["ret"],
                    target_col=None,
                    output_col="forecast",
                ),
            ),
            (
                "signal",
                SignalGenerator(
                    signal_type="sign",
                    input_col="forecast",
                    output_col="signal",
                ),
            ),
        ]
    )

    learning_spec = LearningSpec(
        splitter=ExpandingIncrementPanelSplit(
            train_intervals=1,
            test_size=1,
            min_train_periods=45,
            lookahead=1,
            embargo=0,
        ),
        target_spec=ForwardTargetSpec(
            input_col="close",
            output_col="target",
            horizon=1,
            kind="return",
            method="pct",
            group_level=1,
        ),
        date_level=0,
        show_progress=False,
    )

    strategy = StrategySpec(
        name="CryptoLearningSpecTargetSpecIntegration",
        data_pipeline=pipeline,
        optimizer=SignalWeighted(window_size=20),
        cost_model=FixedCommissionModel(rate=0.0005),
        learning_spec=learning_spec,
        rebal_freq="d",
    )

    engine = BacktestEngine(
        config=strategy,
        data=data,
        initial_capital=1_000_000.0,
        verbose=False,
        ann_factor=365,
    )

    results = engine.run_backtest()
    assert results
    assert "account_value" in results
    assert engine.walk_forward_runner is not None
    assert len(engine.walk_forward_runner.fold_info) > 0
    assert any(
        (info.n_trainable is not None) and (info.n_trainable > 0)
        for info in engine.walk_forward_runner.fold_info
    )


def test_crypto_strategy_pipeline_native_selector_and_learner(crypto_panel: pd.DataFrame):
    data = crypto_panel.copy()

    pipeline = Pipeline(
        steps=[
            (
                "returns",
                Returns(method="pct", input_col="close", output_col="ret", lags=1),
            ),
            (
                "trend",
                Trend(
                    method="price_momentum",
                    input_col="close",
                    window_size=5,
                    scale=False,
                ),
            ),
            (
                "volatility",
                Vol(
                    method="std",
                    input_col="ret",
                    window_type="rolling",
                    window_size=20,
                    annualize=False,
                ),
            ),
            (
                "liquidity",
                Liquidity(
                    method="amihud",
                    return_col="ret",
                    price_col="close",
                    volume_col="volume",
                ),
            ),
            (
                "selector",
                FeatureSelector(
                    method="spearman",
                    feature_cols=["PriceMomentum_5", "STD_rolling_20", "AmihudIlliquidity"],
                    n_features=2,
                    drop_unselected=True,
                ),
            ),
            (
                "learner",
                RegressionLearner(
                    method="ridge",
                    feature_cols=None,
                    target_col=None,
                    output_col="forecast",
                    exclude_cols=["close", "volume", "funding_rate", "ret"],
                    alpha=1.0,
                ),
            ),
            (
                "signal",
                SignalGenerator(
                    signal_type="sign",
                    input_col="forecast",
                    output_col="signal",
                ),
            ),
        ]
    )

    learning_spec = LearningSpec(
        splitter=ExpandingIncrementPanelSplit(
            train_intervals=1,
            test_size=1,
            min_train_periods=45,
            lookahead=1,
            embargo=0,
        ),
        target_spec=ForwardTargetSpec(
            input_col="close",
            output_col="target",
            horizon=1,
            kind="return",
            method="pct",
            group_level=1,
        ),
        date_level=0,
        show_progress=False,
    )

    strategy = StrategySpec(
        name="CryptoPipelineNativeSelectorLearner",
        data_pipeline=pipeline,
        optimizer=SignalWeighted(window_size=20),
        cost_model=FixedCommissionModel(rate=0.0005),
        learning_spec=learning_spec,
        rebal_freq="d",
    )

    engine = BacktestEngine(
        config=strategy,
        data=data,
        initial_capital=1_000_000.0,
        verbose=False,
        ann_factor=365,
    )

    results = engine.run_backtest()
    assert results
    assert "account_value" in results
    assert "weights" in results
    assert engine.walk_forward_runner is not None
    assert len(engine.walk_forward_runner.fold_info) > 0
    assert results["account_value"].iloc[-1] > 0

    strategy_returns = engine.portfolio_return.dropna().to_frame("strategy")
    perf_table = Performance(strategy_returns, ret_type="simple", ann_factor=365).get_table(metrics="key_metrics")
    assert not perf_table.empty
    assert "Sharpe ratio" in perf_table.columns
