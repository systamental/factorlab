import numpy as np
import pandas as pd

from factorlab.learning import PCATransform, PPCATransform, R2PCATransform


def _make_df(n: int = 120, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    x1 = rng.normal(size=n)
    x2 = 0.5 * x1 + rng.normal(scale=0.5, size=n)
    x3 = -0.25 * x1 + 0.75 * x2 + rng.normal(scale=0.5, size=n)
    df = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3}, index=idx)
    df.loc[idx[5], "x2"] = np.nan
    df.loc[idx[11], "x3"] = np.nan
    return df


def test_pca_transform_appends_components():
    df = _make_df()
    tr = PCATransform(input_cols=["x1", "x2", "x3"], n_components=2, output_prefix="PC")
    tr.fit(df)
    out = tr.transform(df)

    assert {"PC1", "PC2"}.issubset(out.columns)
    assert out[["PC1", "PC2"]].notna().sum().min() > 0
    assert tr.eigenvectors_ is not None
    assert tr.explained_variance_ratio_ is not None


def test_ppca_transform_append_and_shape():
    df = _make_df(n=150, seed=17)
    tr = PPCATransform(
        input_cols=["x1", "x2", "x3"],
        n_components=2,
        min_obs=20,
        min_feat=2,
        output_prefix="PPCA",
    )
    tr.fit(df)
    out = tr.transform(df)

    assert {"PPCA1", "PPCA2"}.issubset(out.columns)
    assert out[["PPCA1", "PPCA2"]].shape[0] == len(df)
    assert tr.eigenvectors_ is not None
    assert tr.explained_variance_ratio_ is not None


def test_r2pca_transform_can_return_components_only():
    df = _make_df(n=90, seed=23).fillna(0.0)
    tr = R2PCATransform(
        input_cols=["x1", "x2", "x3"],
        n_components=2,
        output_prefix="R2PC",
        append=False,
        random_state=0,
    )
    tr.fit(df)
    pcs = tr.transform(df)

    assert list(pcs.columns) == ["R2PC1", "R2PC2"]
    assert pcs.shape == (len(df), 2)
