import numpy as np
import pandas as pd

from factorlab.core.pipeline import Pipeline
from factorlab.learning import (
    ExpandingKFoldPanelSplit,
    LinearRegressionLearner,
    WalkForwardGridSearch,
)


def _make_panel(seed: int = 7) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2021-01-01", periods=80, freq="D")
    tickers = ["BTC", "ETH", "SOL"]
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])

    x1 = rng.normal(size=len(idx))
    x2 = rng.normal(size=len(idx))
    noise = rng.normal(scale=0.05, size=len(idx))
    y = 0.25 * x1 + 1.1 * x2 + noise

    X = pd.DataFrame({"x1": x1, "x2": x2}, index=idx)
    y_series = pd.Series(y, index=idx, name="target")
    return X, y_series


def test_walk_forward_grid_search_runs_and_ranks():
    X, y = _make_panel()

    pipeline = Pipeline(
        [
            (
                "lr",
                LinearRegressionLearner(
                    feature_cols=["x1", "x2"],
                    output_col="forecast",
                ),
            )
        ]
    )
    splitter = ExpandingKFoldPanelSplit(n_splits=3, lookahead=0, embargo=0)

    def scorer(out: pd.DataFrame) -> float:
        joined = pd.concat([out["forecast"], y], axis=1).dropna()
        mse = ((joined.iloc[:, 0] - joined.iloc[:, 1]) ** 2).mean()
        return -float(mse)

    search = WalkForwardGridSearch(
        pipeline=pipeline,
        splitter=splitter,
        y=y,
        param_grid={"lr__exclude_cols": [[], ["x2"]]},
        scorer=scorer,
    )
    res = search.run(X)

    assert not res.empty
    assert {"score", "n_folds", "lr__exclude_cols"}.issubset(res.columns)
    assert res.iloc[0]["score"] >= res.iloc[-1]["score"]
