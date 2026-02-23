import numpy as np
import pandas as pd

from factorlab.learning import (
    LagFeatures,
    StatsmodelsOLSLearner,
    TimeSeriesAnalysis,
    TimeSeriesDiagnostics,
    add_lags,
    expanding_window,
    rolling_window,
)


def _make_ts_df(n: int = 180, seed: int = 17) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    target = 0.6 * x1 - 0.3 * x2 + rng.normal(scale=0.1, size=n)
    return pd.DataFrame({"x1": x1, "x2": x2, "target": target}, index=idx)


def test_add_lags_single_index():
    df = _make_ts_df(n=20)
    out = add_lags(df[["x1"]], n_lags=2, include_original=True)
    assert "x1" in out.columns
    assert "x1_L1" in out.columns
    assert "x1_L2" in out.columns


def test_lag_features_transform():
    df = _make_ts_df(n=20)
    tr = LagFeatures(input_cols=["x1", "x2"], n_lags=2)
    tr.fit(df)
    out = tr.transform(df)
    assert "x1_L1" in out.columns
    assert "x2_L2" in out.columns


def test_statsmodels_ols_learner_fit_transform():
    df = _make_ts_df()
    learner = StatsmodelsOLSLearner(
        feature_cols=["x1", "x2"],
        target_col="target",
        output_col="forecast",
    )
    learner.fit(df)
    out = learner.transform(df)
    assert "forecast" in out.columns
    assert out["forecast"].notna().sum() == len(df)
    assert learner.model_result is not None


def test_time_series_diagnostics_outputs():
    df = _make_ts_df()
    adf = TimeSeriesDiagnostics.adf_test(df[["x1", "x2"]])
    assert {"adf_stat", "p_value", "used_lag", "n_obs"}.issubset(adf.columns)
    assert "x1" in adf.index

    gc = TimeSeriesDiagnostics.granger_causality(
        target=df["target"],
        features=df[["x1", "x2"]],
        max_lag=3,
        verbose=False,
    )
    assert {"min_p_value", "best_lag"}.issubset(gc.columns)
    assert "x1" in gc.index


def test_time_series_analysis_linear_regression_params_and_resid():
    df = _make_ts_df(n=120)
    tsa = TimeSeriesAnalysis(
        target=df["target"],
        features=df[["x1", "x2"]],
        trend="c",
        window_type="rolling",
        window_size=30,
    )
    params = tsa.linear_regression(output="params")
    resid = tsa.linear_regression(output="resid")

    assert {"const", "x1", "x2"}.issubset(params.columns)
    assert "resid" in resid.columns
    assert resid["resid"].notna().sum() > 0


def test_rolling_and_expanding_window_helpers():
    s = pd.Series(np.arange(10, dtype=float), index=pd.date_range("2021-01-01", periods=10, freq="D"))

    def win_mean(x):
        return pd.Series({"mean": float(x.mean())})

    roll = rolling_window(win_mean, s, window_size=4)
    exp = expanding_window(win_mean, s, min_obs=4)

    assert isinstance(roll, pd.DataFrame)
    assert isinstance(exp, pd.DataFrame)
    assert roll.shape[0] == 7
    assert exp.shape[0] == 7
