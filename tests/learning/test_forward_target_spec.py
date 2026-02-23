import numpy as np
import pandas as pd

from factorlab.targets import ForwardTargetSpec


def _make_panel(n_days: int = 15, tickers: tuple[str, ...] = ("BTC", "ETH"), seed: int = 9) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-01", periods=n_days, freq="D")
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    close = 100.0 + np.cumsum(rng.normal(0.0, 1.0, size=len(idx)))
    return pd.DataFrame({"close": close}, index=idx).sort_index()


def test_forward_target_spec_trainable_mask_excludes_labels_past_train_end():
    X = _make_panel()
    spec = ForwardTargetSpec(input_col="close", output_col="target", horizon=2, group_level=1)

    unique_dates = X.index.get_level_values("date").unique().sort_values()
    train_dates = unique_dates[:10]
    train_index = X.loc[(slice(train_dates.min(), train_dates.max()), slice(None)), :].index

    mask = spec.trainable_mask(index=X.index, train_index=train_index, date_level=0)
    label_end = spec.label_end_dates(index=X.index, date_level=0)
    train_end = pd.Timestamp(train_dates.max())

    assert mask.dtype == bool
    assert int(mask.sum()) > 0
    assert (pd.DatetimeIndex(label_end[mask].values) <= train_end).all()

    leaky_rows = (
        pd.Index(X.index).isin(train_index)
        & label_end.notna().to_numpy()
        & (pd.DatetimeIndex(label_end.values) > train_end)
    )
    assert not mask[leaky_rows].any()
