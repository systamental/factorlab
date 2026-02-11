import numpy as np
import pandas as pd

from factorlab.transformations.unsupervised_learning import R2PCA
from factorlab.core.wrappers import RollingTransform, ExpandingTransform


def _make_returns(n_obs=40, n_assets=8, seed=42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    data = rng.standard_normal((n_obs, n_assets))
    df = pd.DataFrame(
        data,
        index=pd.date_range("2020-01-01", periods=n_obs, freq="B"),
        columns=[f"A{i}" for i in range(n_assets)]
    )
    # Inject missing data to exercise cleaning
    df.iloc[5, 2] = np.nan
    df.iloc[10, 4] = np.nan
    return df


def test_r2pca_fixed_outputs_shapes_and_invariants():
    df = _make_returns()
    r2 = R2PCA(n_components=3, random_state=0)
    pcs = r2.fit_transform(df)

    assert isinstance(pcs, pd.DataFrame)
    assert isinstance(r2.pcs, pd.DataFrame)
    assert isinstance(r2.output, pd.DataFrame)
    assert isinstance(r2.eigenvecs, pd.DataFrame)
    assert isinstance(r2.expl_var_ratio, pd.DataFrame)
    assert isinstance(r2.alignment_scores, pd.DataFrame)

    assert r2.pcs.shape[1] == 3
    assert r2.eigenvecs.shape[1] == 3
    assert r2.expl_var_ratio.shape[1] == 3
    assert r2.alignment_scores.shape[1] == 3

    # Invariants for explained variance ratio
    evr = r2.expl_var_ratio.iloc[0].values
    assert np.all(evr >= 0.0)
    assert np.all(evr <= 1.0 + 1e-8)
    assert np.nansum(evr) <= 1.0 + 1e-6

    # Explained variance ratio should be non-increasing
    evr_no_nan = evr[~np.isnan(evr)]
    assert np.all(np.diff(evr_no_nan) <= 1e-8)

    # PC1 sign should align with cross-sectional mean (market proxy)
    clean_df = df.dropna(how="all").dropna(axis=1, how="any")
    market_proxy = clean_df.mean(axis=1).values
    pc1 = r2.pcs.loc[clean_df.index, "PC1"].values
    assert np.dot(pc1, market_proxy) >= -1e-8


def test_r2pca_determinism_with_random_state():
    df = _make_returns()
    r2a = R2PCA(n_components=3, random_state=123)
    r2b = R2PCA(n_components=3, random_state=123)

    pcs_a = r2a.fit_transform(df)
    pcs_b = r2b.fit_transform(df)

    assert np.allclose(pcs_a.values, pcs_b.values, equal_nan=True)
    assert np.allclose(r2a.eigenvecs.values, r2b.eigenvecs.values, equal_nan=True)
    assert np.allclose(r2a.expl_var_ratio.values, r2b.expl_var_ratio.values, equal_nan=True)


def test_r2pca_rolling_output_integrity():
    df = _make_returns(n_obs=50)
    r2 = R2PCA(n_components=3, random_state=0)
    rolling = RollingTransform(r2, window_size=10, step=1, show_progress=False)
    pcs = rolling.transform(df)

    assert isinstance(pcs, pd.DataFrame)
    assert pcs.index[0] == df.index[9]
    assert pcs.shape[1] == 3

    assert isinstance(rolling.output, pd.DataFrame)
    assert isinstance(rolling.eigenvecs, pd.DataFrame)
    assert isinstance(rolling.expl_var_ratio, pd.DataFrame)
    assert isinstance(rolling.alignment_scores, pd.DataFrame)

    assert rolling.output.index.equals(pcs.index)
    assert rolling.expl_var_ratio.index.equals(pcs.index)
    assert rolling.alignment_scores.index.equals(pcs.index)

    assert isinstance(rolling.eigenvecs.index, pd.MultiIndex)
    assert rolling.eigenvecs.index.names == ['date', 'ticker']


def test_r2pca_expanding_output_integrity():
    df = _make_returns(n_obs=50)
    r2 = R2PCA(n_components=3, random_state=0)
    expanding = ExpandingTransform(r2, min_obs=10, step=1, show_progress=False)
    pcs = expanding.transform(df)

    assert isinstance(pcs, pd.DataFrame)
    assert pcs.index[0] == df.index[9]
    assert pcs.shape[1] == 3

    assert isinstance(expanding.output, pd.DataFrame)
    assert isinstance(expanding.expl_var_ratio, pd.DataFrame)
    assert isinstance(expanding.alignment_scores, pd.DataFrame)
    assert expanding.expl_var_ratio.index.equals(pcs.index)
    assert expanding.alignment_scores.index.equals(pcs.index)


def test_r2pca_step_size_forward_fill():
    df = _make_returns(n_obs=50)
    r2 = R2PCA(n_components=3, random_state=0)
    rolling = RollingTransform(r2, window_size=10, step=5, show_progress=False)
    pcs = rolling.transform(df)

    expected_len = len(df) - 10 + 1
    assert len(pcs) == expected_len


def test_r2pca_window_bounds_validation():
    df = _make_returns(n_obs=20)
    r2 = R2PCA(n_components=3, random_state=0)

    rolling = RollingTransform(r2, window_size=30, step=1, show_progress=False)
    try:
        rolling.transform(df)
        assert False, "Expected ValueError for window_size > len(X)"
    except ValueError:
        pass

    expanding = ExpandingTransform(r2, min_obs=30, step=1, show_progress=False)
    try:
        expanding.transform(df)
        assert False, "Expected ValueError for min_obs > len(X)"
    except ValueError:
        pass


def test_r2pca_missing_data_robustness():
    df = _make_returns(n_obs=40, n_assets=6)
    # Drop a full column of data for part of the window to force cleaning
    df.iloc[:20, 3] = np.nan

    r2 = R2PCA(n_components=3, random_state=0)
    pcs = r2.fit_transform(df)
    assert isinstance(pcs, pd.DataFrame)


def test_r2pca_reproducibility_rolling_expanding():
    df = _make_returns(n_obs=50, n_assets=6, seed=7)

    r2a = R2PCA(n_components=3, random_state=123)
    r2b = R2PCA(n_components=3, random_state=123)

    roll_a = RollingTransform(r2a, window_size=10, step=1, show_progress=False)
    roll_b = RollingTransform(r2b, window_size=10, step=1, show_progress=False)
    pcs_a = roll_a.transform(df)
    pcs_b = roll_b.transform(df)
    assert np.allclose(pcs_a.values, pcs_b.values, equal_nan=True)

    r2c = R2PCA(n_components=3, random_state=123)
    r2d = R2PCA(n_components=3, random_state=123)

    exp_a = ExpandingTransform(r2c, min_obs=10, step=1, show_progress=False)
    exp_b = ExpandingTransform(r2d, min_obs=10, step=1, show_progress=False)
    pcs_c = exp_a.transform(df)
    pcs_d = exp_b.transform(df)
    assert np.allclose(pcs_c.values, pcs_d.values, equal_nan=True)
