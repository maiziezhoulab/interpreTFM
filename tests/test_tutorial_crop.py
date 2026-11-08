"""The tutorial's CosMx crop is built locally from the source file (``interpretfm.hub.tutorial_crop``); offline."""
import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from interpretfm import hub


def _source(path, n=60, n_genes=7, seed=0, origin=(0.0, 0.0), extent=100.0):
    """A small h5ad in the layout of the CosMx source: two sections (``library_key`` 0 and 7), coordinates in pixels."""
    rng = np.random.default_rng(seed)
    lib = np.array([0] * 25 + [7] * (n - 25))
    xy = np.asarray(origin) + rng.uniform(0, extent, size=(n, 2))
    types = np.array(["B-cell", "macrophage", "tumor 13", "tumor 5"])
    ct = np.where(lib == 0, "tumor 5", types[rng.integers(0, 3, n)])  # "tumor 5" occurs only outside section 7
    X = sp.random(n, n_genes, density=0.4, format="csr", random_state=seed, dtype=np.float32)
    X.data = np.ceil(X.data * 5)
    obs = pd.DataFrame({"library_key": pd.Categorical(lib), "author_cell_type": pd.Categorical(ct, categories=types),
                        "Width": rng.integers(20, 90, n), "Height": rng.integers(20, 90, n), "cell_ID": np.arange(n),
                        "niche": pd.Categorical(np.where(lib == 0, "lymphoid structure", "stroma")),
                        "other": rng.normal(size=n)},
                       index=[f"{i}_{1 + i // 20}" for i in range(n)])
    var = pd.DataFrame({"feature_name": pd.Categorical([f"G{j}" for j in range(n_genes)])}, index=[f"ENSG{j:011d}" for j in range(n_genes)])
    a = ad.AnnData(X=X, obs=obs, var=var)
    a.obsm["spatial"] = xy
    a.obsm["X_pca"] = rng.normal(size=(n, 3)).astype(np.float32)
    a.layers["raw"] = X.copy()
    a.write_h5ad(path)
    return a


def test_crop_section_reads_one_window_of_one_section(tmp_path):
    src = tmp_path / "source.h5ad"
    a = _source(src)
    window = (20.0, 10.0, 50.0)
    crop = hub.crop_section(src, library_key=7, window_px=window)
    xy = a.obsm["spatial"]
    want = np.flatnonzero((a.obs["library_key"].astype(int).values == 7) & (xy[:, 0] >= 20) & (xy[:, 0] < 70)
                          & (xy[:, 1] >= 10) & (xy[:, 1] < 60))
    assert len(want) > 3
    assert list(crop.obs_names) == list(a.obs_names[want])
    assert np.array_equal(crop.X.toarray(), a.X[want].toarray()) and crop.X.dtype == np.float32 and crop.X.format == "csr"
    assert np.array_equal(crop.obsm["spatial"], xy[want]) and crop.obsm["spatial"].dtype == np.float64
    assert list(crop.obs.columns) == list(hub.TUTORIAL_OBS_COLUMNS)
    for c in ("Width", "Height", "cell_ID"):
        assert np.array_equal(crop.obs[c].values, a.obs[c].values[want])
    # categories that occur only in the other section are dropped, the section's own ones kept (anndata's subsetting)
    assert list(crop.obs["author_cell_type"].cat.categories) == ["B-cell", "macrophage", "tumor 13"]
    assert list(crop.obs["niche"].cat.categories) == ["stroma"]
    assert list(crop.var_names) == list(a.var_names) and list(crop.var["feature_name"]) == list(a.var["feature_name"])
    assert list(crop.obsm.keys()) == ["spatial"] and not list(crop.layers.keys())
    meta = crop.uns["interpretfm_tutorial"]
    assert meta["um_per_unit"] == 0.18 and list(meta["window_px"]) == list(window)
    with pytest.raises(ValueError):
        hub.crop_section(src, library_key=3, window_px=window)  # no such section
    with pytest.raises(ValueError):
        hub.crop_section(src, library_key=7, window_px=(500.0, 500.0, 10.0))  # empty window


def test_tutorial_crop_caches_and_checks(tmp_path, monkeypatch):
    src = tmp_path / "source.h5ad"
    _source(src)
    out = tmp_path / "cache" / "crop.h5ad"
    window = (0.0, 0.0, 60.0)
    monkeypatch.setenv(hub.COSMX_SOURCE_ENV, str(src))
    assert hub.cosmx_source() == src
    first = hub.tutorial_crop(path=out, library_key=7, window_px=window)  # source from the environment variable
    assert out.exists() and first.n_obs > 0
    assert not list(out.parent.glob("*.partial.h5ad"))
    # a second call reads the cached crop and never opens the source
    monkeypatch.setenv(hub.COSMX_SOURCE_ENV, str(tmp_path / "missing.h5ad"))
    again = hub.tutorial_crop(path=out, library_key=7, window_px=window)
    assert list(again.obs_names) == list(first.obs_names) and hub.crop_digest(again) == hub.crop_digest(first)
    # another window rebuilds it (and needs the source)
    with pytest.raises(FileNotFoundError):
        hub.tutorial_crop(path=out, library_key=7, window_px=(0.0, 0.0, 30.0))
    other = hub.tutorial_crop(path=out, source=src, library_key=7, window_px=(0.0, 0.0, 30.0))
    assert other.n_obs < first.n_obs
    # the default window is checked against the digest of the published crop: a different source is refused
    other_src = tmp_path / "other_source.h5ad"
    _source(other_src, origin=hub.TUTORIAL_WINDOW_PX[:2], extent=hub.TUTORIAL_WINDOW_PX[2])
    with pytest.raises(ValueError, match="differs from the tutorial crop"):
        hub.tutorial_crop(path=tmp_path / "default.h5ad", source=other_src)
    assert not (tmp_path / "default.h5ad").exists()


def test_tutorial_bundle_does_not_need_the_crop(tmp_path):
    with pytest.raises(FileNotFoundError):
        hub.tutorial_bundle("local", path=tmp_path)
    (tmp_path / "tutorial").mkdir()
    (tmp_path / hub.TUTORIAL_MARKER).write_bytes(b"")
    assert hub.tutorial_bundle("local", path=tmp_path) == tmp_path
    assert not any("cosmx_crop" in p for p in hub.TUTORIAL_FILES)
    assert hub.TUTORIAL_WINDOW_PX[2] == 4500.0 and hub.COSMX_SOURCE["repo_type"] == "dataset"
