import numpy as np
import pandas as pd

from factorlab.portfolio.construction.factor_mimicking import FMP


def _make_inputs(n: int = 120, seed: int = 17):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2018-01-31", periods=n, freq="ME")

    factors = pd.DataFrame(
        {
            "factor_a": rng.normal(scale=0.04, size=n),
            "factor_b": rng.normal(scale=0.03, size=n),
            "factor_c": rng.normal(scale=0.05, size=n),
        },
        index=idx,
    )

    returns = pd.DataFrame(
        {
            "asset_1": 0.3 * factors["factor_a"] + rng.normal(scale=0.02, size=n),
            "asset_2": 0.4 * factors["factor_b"] + rng.normal(scale=0.02, size=n),
            "asset_3": 0.2 * factors["factor_c"] + rng.normal(scale=0.02, size=n),
        },
        index=idx,
    )

    return returns, factors


def test_factor_mimicking_predict_and_exposure_runs():
    returns, factors = _make_inputs()
    fmp = FMP(returns=returns, factors=factors, ann_factor=12)

    fmp.predict_factors(
        window_type="fixed",
        ann_vol=0.15,
        periods=12,
        n_lags=6,
        fwd=0,
    )
    assert not fmp.factors_pred.empty
    assert set(fmp.factors_pred.columns).issubset(set(factors.columns))

    fmp.factor_exposures()
    assert fmp.betas is not None
    assert "const" in fmp.betas.columns
    assert set(factors.columns).issubset(set(fmp.betas.columns))
