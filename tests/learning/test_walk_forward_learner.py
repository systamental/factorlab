import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from factorlab.core.pipeline import Pipeline
from factorlab.factors.library.trend.trend import Trend
from factorlab.learning import WalkForwardLearner
from factorlab.signals.generator import SignalGenerator
from factorlab.targets import ForwardReturnTarget


def _make_panel(
    n_days: int = 220,
    tickers: tuple[str, ...] = ("BTC", "ETH", "SOL", "XRP"),
    seed: int = 7,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2021-01-01", periods=n_days, freq="D")
    index = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    df = pd.DataFrame(index=index, columns=["close"], dtype=float)

    for i, ticker in enumerate(tickers):
        ticker_mask = df.index.get_level_values("ticker") == ticker
        drift = 0.0003 + (i * 0.0001)
        ret = rng.normal(loc=drift, scale=0.02, size=n_days)
        close = 100.0 * np.exp(np.cumsum(ret))
        df.loc[ticker_mask, "close"] = close

    return df.sort_index()


def _make_pipeline(include_signal: bool = True) -> Pipeline:
    steps = [
        (
            "trend_factor",
            Trend(
                method="price_momentum",
                input_col="close",
                window_size=5,
                scale=False,
            ),
        ),
        (
            "forecaster",
            WalkForwardLearner(
                model=LinearRegression(),
                feature_cols=["PriceMomentum_5"],
                target_transform=ForwardReturnTarget(
                    input_col="close",
                    output_col="fwd_ret",
                    horizon=1,
                ),
                prediction_col="forecast",
                window_type="expanding",
                min_train_periods=40,
                retrain_interval=1,
                min_train_samples=100,
            ),
        ),
    ]

    if include_signal:
        steps.append(
            (
                "signal",
                SignalGenerator(
                    signal_type="sign",
                    input_col="forecast",
                    output_col="signal",
                ),
            )
        )

    return Pipeline(steps=steps)


def test_pipeline_factor_to_forecast_to_signal_chain():
    df = _make_panel()
    pipeline = _make_pipeline(include_signal=True)

    out = pipeline.fit_transform(df)

    assert "PriceMomentum_5" in out.columns
    assert "forecast" in out.columns
    assert "signal" in out.columns

    assert out["forecast"].notna().sum() > 0
    signal_values = out["signal"].dropna().unique().tolist()
    assert set(signal_values).issubset({-1.0, 0.0, 1.0})


def test_walk_forward_forecast_is_invariant_to_future_data_changes():
    df = _make_panel()
    pipe_1 = _make_pipeline(include_signal=False)
    out_1 = pipe_1.fit_transform(df)

    shock_date = df.index.get_level_values("date").unique()[160]
    df_shocked = df.copy(deep=True)
    future_mask = df_shocked.index.get_level_values("date") >= shock_date
    df_shocked.loc[future_mask, "close"] = df_shocked.loc[future_mask, "close"] * 4.0

    pipe_2 = _make_pipeline(include_signal=False)
    out_2 = pipe_2.fit_transform(df_shocked)

    pre_shock_mask = out_1.index.get_level_values("date") < shock_date
    s1 = out_1.loc[pre_shock_mask, "forecast"]
    s2 = out_2.loc[pre_shock_mask, "forecast"]

    assert s1.notna().sum() > 0
    pd.testing.assert_series_equal(
        s1,
        s2,
        check_names=False,
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )
