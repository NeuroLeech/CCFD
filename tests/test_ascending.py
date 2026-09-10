import os
import numpy as np
import pytest

from paths import DATA
import ascending


def _cortex():
    from mesh_cache import load_cortex
    return load_cortex("fsaverage5", verbose=False)


def _bfs_dist(c, seed):
    import scipy.sparse as sp
    from scipy.sparse.csgraph import dijkstra
    E = c.edges
    A = sp.coo_matrix((np.ones(len(E)), (E[:, 0], E[:, 1])),
                      shape=(c.nV, c.nV))
    return dijkstra((A + A.T).tocsr(), indices=[seed])[0]


def _synthetic_fields(c, n=3, scale=8.0, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for s in rng.choice(c.nV, n, replace=False):
        d = _bfs_dist(c, s)
        out.append(np.exp(-0.5 * (d / scale) ** 2))
    return np.stack(out)


# ---- Task 2: manifest + masks ------------------------------------------------

@pytest.mark.skipif(not os.path.exists(ascending.MANIFEST), reason="atlas manifest missing")
def test_manifest_entries_exist():
    entries = ascending.load_manifest()
    assert entries, "empty manifest"
    names = [e["name"] for e in entries]
    assert len(names) == len(set(names)), "duplicate nucleus names"


@pytest.mark.skipif(not os.path.exists(ascending.FIB_PATH), reason="FIB missing")
def test_masks_load_on_fib_grid():
    entries = ascending.load_manifest()
    names, masks = ascending.nucleus_masks(entries, verbose=False)
    assert len(names) == len(entries)
    assert masks.shape[0] == len(entries)
    assert masks.ndim == 4
    for nm, m in zip(names, masks):
        assert m.max() > 0, f"{nm} mask is empty after resampling"
        assert m.min() >= -1e-6, f"{nm} mask has negative values"
        assert m.max() <= 1.0 + 1e-6, f"{nm} mask exceeds 1"


@pytest.mark.skipif(not os.path.exists(ascending.FIB_PATH), reason="FIB missing")
def test_grid_matches_dsi_studio():
    shape, affine = ascending._grid()
    assert shape == (78, 94, 68), f"unexpected FIB grid {shape}"
    assert np.allclose(np.diag(affine)[:3], [-2.0, -2.0, 2.0]), \
        f"unexpected voxel size {np.diag(affine)}"


# ---- Task 3: endpoint density (no tracking needed) ---------------------------

def test_endpoint_density_rasterizes_endpoints():
    shape, affine = ascending._grid()
    trk_path = os.path.join(DATA, "cache", "_test_endpoints.trk")
    import nibabel as nib
    vox = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])
    ras = nib.affines.apply_affine(affine, vox)
    # streamline 0 endpoints = vox0, vox1; streamline 1 is a single point, so BOTH its
    # endpoints land on vox1
    streamlines = [np.stack([ras[0], ras[1]]), np.stack([ras[1]])]
    tfile = nib.streamlines.Tractogram(streamlines, affine_to_rasmm=np.eye(4))
    nib.streamlines.save(tfile, trk_path)
    V = ascending._endpoint_density(trk_path)
    assert V[tuple(vox[0].astype(int))] == 1
    assert V[tuple(vox[1].astype(int))] == 3
    assert float(V.sum()) == 4
    os.remove(trk_path)


# ---- Task 4: modes and coverage ----------------------------------------------

def test_mode0_is_plain_field():
    c = _cortex()
    fields = _synthetic_fields(c)
    names = [f"nuc{i}" for i in range(len(fields))]
    P, tags, frac, meta = ascending.modal_channels(c, names, fields, tau=0.3,
                                                   budget=12, verbose=False)
    for nm in names:
        rows = [k for k, t in enumerate(tags) if t.startswith(nm + ":")]
        i = names.index(nm)
        f = fields[i]
        on = f > 0.3
        r = np.corrcoef(P[rows[0]][on], f[on])[0, 1]
        assert r > 0.95, f"{nm} mode 0 does not match the plain field (r={r:.3f})"


def test_field_modes_orthonormal():
    c = _cortex()
    f = _synthetic_fields(c, n=1)[0]
    verts = ascending.support(c, f, 0.3)
    V = ascending._field_modes(c, f, verts, 3)
    err = np.abs(V.T @ V - np.eye(V.shape[1])).max()
    assert err < 1e-8, f"modes not orthonormal ({err:.2e})"


def test_budget_and_tags():
    c = _cortex()
    fields = _synthetic_fields(c, n=4)
    names = [f"nuc{i}" for i in range(len(fields))]
    P, tags, frac, meta = ascending.modal_channels(c, names, fields, tau=0.3,
                                                   budget=12, verbose=False)
    K = len(tags)
    assert K <= 12 + len(names) * 4, f"channel budget blown: {K}"
    assert K >= 3, "too few channels"
    for t in tags:
        nm, j = t.split(":")
        assert nm in names and j.isdigit()
    assert P.shape == (K, c.nV)
    assert P.dtype == np.float32
    assert np.isfinite(P).all()
    assert ((np.abs(P) @ c.A) > 0).all(), "a channel has zero area-weighted mass"
    for nm in names:
        rows = [k for k, t in enumerate(tags) if t == f"{nm}:0"]
        assert rows and (P[rows[0]] >= -1e-6).all(), f"{nm} mode 0 has negative lobes"
        assert meta["k_n"][nm] <= 4


def test_coverage_monotonic_and_tau_bisection():
    c = _cortex()
    fields = _synthetic_fields(c, n=8, scale=40.0)
    names = [f"nuc{i}" for i in range(len(fields))]
    c1 = ascending.coverage(c, fields, 0.1)[0]
    c2 = ascending.coverage(c, fields, 0.5)[0]
    assert c1 >= c2, "coverage must shrink as tau rises"
    maxcov = ascending.coverage(c, fields, 0.001)[0]
    assert maxcov > 0.15, f"fields too small for the bisection test ({maxcov:.3f})"
    tau = ascending.tau_for_coverage(c, names, fields, 0.5 * maxcov)
    got = ascending.coverage(c, fields, tau)[0]
    assert got <= 0.5 * maxcov + 1e-6, "bisection overshot the target"
    assert abs(got - 0.5 * maxcov) < 0.02, f"tau_for_coverage hit {got:.3f}, " \
        f"wanted {0.5 * maxcov:.3f}"


# ---- Task 5: profile cache tag -----------------------------------------------

def test_profile_tag_distinguishes_profiles():
    import xspec
    rng = np.random.default_rng(0)
    P1 = rng.standard_normal((5, 200)).astype(np.float32)
    P2 = P1.copy()
    P2[0, 0] += 1e-3
    assert xspec.profile_tag(P1) == xspec.profile_tag(P1)
    assert xspec.profile_tag(P1) != xspec.profile_tag(P2)
    assert xspec.profile_tag(P1) == xspec.profile_tag(np.ascontiguousarray(P1, np.float32))
