from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from factorlab.factors.volume import Volume  # noqa: E402


FACTOR_SPECS: List[Tuple[str, Dict[str, int]]] = [
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


def load_crypto_ohlcv(data_dir: Path, max_symbols: int) -> pd.DataFrame:
    files = sorted(data_dir.glob("*.csv"))[:max_symbols]
    if not files:
        raise FileNotFoundError(f"No CSV files found in: {data_dir}")

    parts = []
    for path in files:
        try:
            tmp = pd.read_csv(
                path,
                usecols=["open_time", "open", "high", "low", "close", "volume", "ticker"],
            )
        except Exception:
            continue

        if tmp.empty:
            continue

        tmp["date"] = pd.to_datetime(tmp["open_time"], errors="coerce")
        tmp = tmp.dropna(subset=["date", "ticker"])
        tmp = tmp.set_index(["date", "ticker"]).sort_index()
        parts.append(tmp[["open", "high", "low", "close", "volume"]])

    if not parts:
        raise RuntimeError(f"Could not load usable OHLCV data from: {data_dir}")

    df = pd.concat(parts, axis=0).sort_index()
    counts = df.groupby(level=1).size()
    keep = counts[counts >= 365].index
    df = df[df.index.get_level_values(1).isin(keep)]
    return df


def evaluate(df: pd.DataFrame, top_n: int, ann_factor: int = 365) -> pd.DataFrame:
    close = df["close"]
    volume = df["volume"]
    fwd_ret = close.groupby(level=1).shift(-1).div(close) - 1.0

    notional = close * volume
    liquidity = (
        notional.groupby(level=1)
        .rolling(window=20, min_periods=20)
        .mean()
        .droplevel(0)
        .sort_index()
    )
    eligible = liquidity.groupby(level=0).rank(ascending=False, method="first") <= top_n

    rows = []
    for method, kwargs in FACTOR_SPECS:
        factor = Volume(method=method, **kwargs)
        out = factor.compute(df)
        new_col = [c for c in out.columns if c not in df.columns][0]

        panel = pd.concat(
            [
                out[new_col].rename("factor"),
                fwd_ret.rename("fwd_ret"),
                eligible.rename("eligible"),
            ],
            axis=1,
        )
        panel = panel[panel["eligible"]].dropna()
        if panel.empty:
            continue

        daily_ic = panel.groupby(level=0).apply(
            lambda g: g["factor"].corr(g["fwd_ret"], method="spearman") if g.shape[0] >= 10 else np.nan
        )

        ranked = panel.copy()
        ranked["weight"] = ranked.groupby(level=0)["factor"].rank(pct=True) - 0.5
        gross = ranked.groupby(level=0)["weight"].transform(lambda s: s.abs().sum())
        ranked["weight"] = ranked["weight"] / gross.replace(0, np.nan)
        ls_ret = (ranked["weight"] * ranked["fwd_ret"]).groupby(level=0).sum()

        mean_ic = float(daily_ic.mean()) if daily_ic.notna().any() else np.nan
        std_ic = float(daily_ic.std()) if daily_ic.notna().sum() > 1 else np.nan
        ic_ir = mean_ic / std_ic if std_ic and np.isfinite(std_ic) else np.nan

        ann_ret = float(ls_ret.mean() * ann_factor) if ls_ret.notna().any() else np.nan
        ann_vol = float(ls_ret.std() * np.sqrt(ann_factor)) if ls_ret.notna().sum() > 1 else np.nan
        sharpe_365 = ann_ret / ann_vol if ann_vol and np.isfinite(ann_vol) else np.nan

        rows.append(
            {
                "method": method,
                "n_ic_obs": int(daily_ic.notna().sum()),
                "mean_ic": mean_ic,
                "ic_ir": ic_ir,
                "ann_ret_365": ann_ret,
                "ann_vol_365": ann_vol,
                "sharpe_365": sharpe_365,
            }
        )

    if not rows:
        raise RuntimeError("No factors produced valid evaluation rows.")

    return pd.DataFrame(rows).set_index("method").sort_values("mean_ic", ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate volume factors on crypto universe.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/Users/mikuts/astrofactor/astroblade/data/systamental/crypto/survivorship/binance_klines_history/daily/futures"),
        help="Directory containing per-symbol daily OHLCV CSV files.",
    )
    parser.add_argument("--max-symbols", type=int, default=120, help="Maximum number of symbol CSVs to load.")
    parser.add_argument("--top-n", type=int, default=60, help="Top-N liquid assets used each day.")
    parser.add_argument("--output", type=Path, default=None, help="Optional output CSV for summary.")
    args = parser.parse_args()

    df = load_crypto_ohlcv(args.data_dir, args.max_symbols)
    summary = evaluate(df, top_n=args.top_n, ann_factor=365)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print(summary.round(4))

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.output)
        print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()

