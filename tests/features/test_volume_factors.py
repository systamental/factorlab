import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from factorlab.factors.volume import Volume


FACTOR_SPECS = [
    ("volume_momentum", {"hist_length": 20, "multiplier": 4}),
    ("delta_volume_momentum", {"hist_length": 20, "multiplier": 4, "delta_len": 100}),
    ("volume_weighted_ma_over_ma", {"hist_length": 50}),
    ("diff_volume_weighted_ma_over_ma", {"short_dist": 20, "long_dist": 100}),
    ("price_volume_fit", {"hist_length": 50}),
    ("diff_price_volume_fit", {"short_dist": 20, "long_dist": 100}),
    ("delta_price_volume_fit", {"hist_length": 20, "delta_dist": 30}),
    ("on_balance_volume", {"hist_length": 50}),
    ("delta_on_balance_volume", {"hist_length": 50, "delta_dist": 45}),
    ("positive_volume_indicator", {"hist_length": 40}),
    ("delta_positive_volume_indicator", {"hist_length": 40, "delta_dist": 35}),
    ("negative_volume_indicator", {"hist_length": 40}),
    ("delta_negative_volume_indicator", {"hist_length": 40, "delta_dist": 35}),
    ("product_price_volume", {"hist_length": 25}),
    ("sum_price_volume", {"hist_length": 25}),
    ("delta_product_price_volume", {"hist_length": 40, "delta_dist": 35}),
    ("delta_sum_price_volume", {"hist_length": 40, "delta_dist": 35}),
]


@pytest.fixture(scope="module")
def crypto_universe() -> pd.DataFrame:
    data_path = Path(__file__).resolve().parents[1] / "datasets" / "data" / "binance_spot_prices.csv"
    df = pd.read_csv(
        data_path,
        index_col=["date", "ticker"],
        parse_dates=["date"],
    )
    df = df.sort_index()

    # keep symbols with at least 300 daily bars
    counts = df.groupby(level=1).size()
    keep = counts[counts >= 300].index
    df = df[df.index.get_level_values(1).isin(keep)]

    # keep a liquid subset to keep tests fast and stable
    avg_notional = (df["close"] * df["volume"]).groupby(level=1).mean()
    top_symbols = avg_notional.nlargest(60).index
    df = df[df.index.get_level_values(1).isin(top_symbols)]

    return df[["open", "high", "low", "close", "volume"]]


@pytest.mark.parametrize("method,kwargs", FACTOR_SPECS)
def test_volume_factor_methods_smoke(crypto_universe: pd.DataFrame, method: str, kwargs: dict) -> None:
    factor = Volume(method=method, **kwargs)
    out = factor.compute(crypto_universe)

    created_cols = [col for col in out.columns if col not in crypto_universe.columns]
    assert len(created_cols) == 1

    factor_col = created_cols[0]
    values = out[factor_col].dropna()

    assert len(values) > 0
    assert (values <= 50).all()
    assert (values >= -50).all()

    pd.testing.assert_frame_equal(out[crypto_universe.columns], crypto_universe)
    assert out.index.equals(crypto_universe.index)


def test_volume_factor_crypto_rank_ic_smoke(crypto_universe: pd.DataFrame) -> None:
    close = crypto_universe["close"]
    volume = crypto_universe["volume"]

    fwd_ret = close.groupby(level=1).shift(-1).div(close) - 1.0

    # daily tradable universe proxy: top 40 by 20-day average notional
    notional = close * volume
    liquidity = (
        notional.groupby(level=1)
        .rolling(window=20, min_periods=20)
        .mean()
        .droplevel(0)
        .sort_index()
    )
    eligible = liquidity.groupby(level=0).rank(ascending=False, method="first") <= 40

    rows = []
    for method, kwargs in FACTOR_SPECS:
        factor = Volume(method=method, **kwargs)
        out = factor.compute(crypto_universe)
        factor_col = [col for col in out.columns if col not in crypto_universe.columns][0]

        panel = pd.concat(
            [
                out[factor_col].rename("factor"),
                fwd_ret.rename("fwd_ret"),
                eligible.rename("eligible"),
            ],
            axis=1,
        )
        panel = panel[panel["eligible"]].dropna()

        daily_ic = panel.groupby(level=0).apply(
            lambda g: g["factor"].corr(g["fwd_ret"], method="spearman") if g.shape[0] >= 12 else np.nan
        )

        n_obs = int(daily_ic.notna().sum())
        mean_ic = float(daily_ic.mean()) if n_obs > 0 else np.nan
        std_ic = float(daily_ic.std()) if n_obs > 1 else np.nan

        rows.append(
            {
                "method": method,
                "n_obs": n_obs,
                "mean_ic": mean_ic,
                "std_ic": std_ic,
            }
        )

    summary = pd.DataFrame(rows).set_index("method")

    assert summary.shape[0] == len(FACTOR_SPECS)
    assert (summary["n_obs"] >= 30).sum() >= 12
    assert np.isfinite(summary["mean_ic"].dropna()).all()
    assert (summary["mean_ic"].dropna().abs() <= 1).all()
