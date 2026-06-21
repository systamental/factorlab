import pandas as pd
import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from factorlab.core.pipeline import Pipeline
from factorlab.core.walk_forward_runner import WalkForwardRunner
from factorlab.features.transforms.returns import Returns
from factorlab.learning import ExpandingIncrementPanelSplit
from factorlab.learning.sklearn_wrapper import SKLearnWrapper
from factorlab.signals.generator import SignalGenerator
from factorlab.targets import ForwardReturnTarget, ForwardTargetSpec


def _make_panel(n_days: int = 220, tickers: tuple[str, ...] = ("BTC", "ETH", "SOL"), seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2021-01-01", periods=n_days, freq="D")
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    df = pd.DataFrame(index=idx, columns=["close"], dtype=float)
    for i, t in enumerate(tickers):
        m = df.index.get_level_values("ticker") == t
        ret = rng.normal(loc=0.0002 + i * 0.00005, scale=0.02, size=n_days)
        df.loc[m, "close"] = 100.0 * np.exp(np.cumsum(ret))
    return df.sort_index()


def _make_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("ret", Returns(method="pct", input_col="close", output_col="ret", lags=1)),
            ("target", ForwardReturnTarget(input_col="close", output_col="target", horizon=1)),
            (
                "learner",
                SKLearnWrapper(
                    model=LinearRegression(),
                    feature_cols=["ret"],
                    target_col="target",
                    output_col="forecast",
                    prediction_method="predict",
                ),
            ),
            ("signal", SignalGenerator(signal_type="sign", input_col="forecast", output_col="signal")),
        ]
    )


def _make_pipeline_external_target() -> Pipeline:
    return Pipeline(
        steps=[
            ("ret", Returns(method="pct", input_col="close", output_col="ret", lags=1)),
            (
                "learner",
                SKLearnWrapper(
                    model=LinearRegression(),
                    feature_cols=["ret"],
                    target_col=None,
                    output_col="forecast",
                    prediction_method="predict",
                ),
            ),
            ("signal", SignalGenerator(signal_type="sign", input_col="forecast", output_col="signal")),
        ]
    )


def test_walk_forward_runner_pipeline_chain():
    X = _make_panel()
    splitter = ExpandingIncrementPanelSplit(
        train_intervals=1,
        test_size=1,
        min_train_periods=45,
        lookahead=1,
        embargo=0,
    )
    runner = WalkForwardRunner(pipeline=_make_pipeline(), splitter=splitter, date_level=0)
    out = runner.run(X)

    assert "forecast" in out.columns
    assert "signal" in out.columns
    assert out["forecast"].notna().sum() > 0
    assert len(runner.fold_info) > 0


def test_walk_forward_runner_future_perturbation_invariance():
    X = _make_panel()
    splitter = ExpandingIncrementPanelSplit(
        train_intervals=1,
        test_size=1,
        min_train_periods=50,
        lookahead=1,
        embargo=0,
    )

    runner_1 = WalkForwardRunner(pipeline=_make_pipeline(), splitter=splitter, date_level=0)
    out_1 = runner_1.run(X)

    shock_date = X.index.get_level_values("date").unique()[170]
    X_shock = X.copy(deep=True)
    m = X_shock.index.get_level_values("date") >= shock_date
    X_shock.loc[m, "close"] = X_shock.loc[m, "close"] * 5.0

    runner_2 = WalkForwardRunner(pipeline=_make_pipeline(), splitter=splitter, date_level=0)
    out_2 = runner_2.run(X_shock)

    pre = out_1.index.get_level_values("date") < shock_date
    pd.testing.assert_series_equal(
        out_1.loc[pre, "forecast"],
        out_2.loc[pre, "forecast"],
        check_names=False,
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )


def test_walk_forward_runner_with_fold_local_target_spec():
    X = _make_panel()
    splitter = ExpandingIncrementPanelSplit(
        train_intervals=1,
        test_size=1,
        min_train_periods=50,
        lookahead=1,
        embargo=0,
    )
    target_spec = ForwardTargetSpec(
        input_col="close",
        output_col="target",
        horizon=1,
        method="pct",
        group_level=1,
    )

    runner = WalkForwardRunner(
        pipeline=_make_pipeline_external_target(),
        splitter=splitter,
        target_spec=target_spec,
        date_level=0,
    )
    out = runner.run(X)

    assert "forecast" in out.columns
    assert out["forecast"].notna().sum() > 0
    assert len(runner.fold_info) > 0
    assert all((fi.n_trainable is None) or (fi.n_trainable > 0) for fi in runner.fold_info)


def test_walk_forward_runner_raises_when_lookahead_shorter_than_horizon():
    X = _make_panel()
    splitter = ExpandingIncrementPanelSplit(
        train_intervals=1,
        test_size=1,
        min_train_periods=50,
        lookahead=0,
        embargo=0,
    )
    target_spec = ForwardTargetSpec(input_col="close", output_col="target", horizon=1)
    runner = WalkForwardRunner(
        pipeline=_make_pipeline_external_target(),
        splitter=splitter,
        target_spec=target_spec,
        date_level=0,
    )

    with pytest.raises(ValueError, match="lookahead is shorter than target_spec.horizon"):
        runner.run(X)
