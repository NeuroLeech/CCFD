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
    for e in entries:
        assert e["label"] in ascending.THOMAS_LEFT, e["name"]


def test_nucleus_masks_are_thomas_onehots():
    entries = ascending.load_manifest()
    names, masks, affs = ascending.nucleus_masks(entries, verbose=False)
    assert len(names) == len(entries) == len(masks) == len(affs)
    for nm, m, af in zip(names, masks, affs):
        assert m.max() > 0, f"{nm} mask is empty"
        assert np.all(np.isin(m, (0.0, 1.0))), f"{nm} mask is not a one-hot"
        assert af.shape == (4, 4)
    assert masks[0].shape == (394, 466, 378), "not the THOMAS native grid"


def _subfields_cached():
    import glob
    return bool(glob.glob(os.path.join(ascending.CACHE, "thalamus_subfields_*.npz")))


@pytest.mark.skipif(not _subfields_cached(), reason="subfields cache missing")
def test_relay_fields_anatomy():
    c = _cortex()
    names, F = ascending.load_fields(c, verbose=False)
    lab = np.asarray(c.lab)
    assert len(F) == 12 and F.shape[1] == c.nV
    for nm, f in zip(names, F):
        assert f.max() > 0, f"{nm} field is empty"
    vpl = F[names.index("VPL")]
    s1 = np.isin(lab, [9, 51, 52])                     # 3b, 1, 2
    assert vpl[s1].sum() / vpl.sum() > 0.3, \
        f"VPL field does not concentrate on S1 ({vpl[s1].sum()/vpl.sum():.3f})"
    cm = F[names.index("CM")]
    m1 = np.isin(lab, [8, 55])                         # 4, 6mp
    assert cm[m1].sum() / cm.sum() > 0.5, \
        f"CM field does not concentrate on motor cortex ({cm[m1].sum()/cm.sum():.3f})"


@pytest.mark.skipif(not os.path.exists(ascending.FIB_PATH), reason="FIB missing")
def test_grid_matches_dsi_studio():
    shape, affine = ascending._grid()
    assert shape == (78, 94, 68), f"unexpected FIB grid {shape}"
    assert np.allclose(np.diag(affine)[:3], [-2.0, -2.0, 2.0]), \
        f"unexpected voxel size {np.diag(affine)}"


# ---- Task 3: .tt.gz parsing and endpoint density -----------------------------

def _enc_entry(typ, name, data, rows, cols):
    import struct
    nb = len(name.encode()) + 1
    hdr = struct.pack("<IIIII", typ, rows, cols, 0, nb) + name.encode() + b"\x00"
    return hdr + data


def _build_tt(streamlines, shape=(78, 94, 68)):
    import struct, gzip, tempfile, os
    body = b""
    body += _enc_entry(20, "dimension", struct.pack("<III", *shape), 1, 3)
    body += _enc_entry(10, "voxel_size", struct.pack("<fff", 2.0, 2.0, 2.0), 1, 3)
    body += _enc_entry(10, "trans_to_mni", struct.pack("<16f", *np.eye(4).ravel()), 4, 4)
    buf = b""
    for s in streamlines:
        s32 = np.round(np.asarray(s, np.float32) * 32.0).astype(np.int64).ravel()
        count = s32.size
        buf += struct.pack("<I", count)
        buf += struct.pack("<iii", int(s32[0]), int(s32[1]), int(s32[2]))
        for j in range(3, count):
            buf += struct.pack("<b", int(s32[j] - s32[j - 3]))
    body += _enc_entry(50, "track", buf, len(buf), 1)
    fd, path = tempfile.mkstemp(suffix=".tt.gz")
    os.close(fd)
    with gzip.open(path, "wb") as f:
        f.write(body)
    return path


def test_parse_tt_decodes_streamlines():
    s1 = np.array([[10.0, 20.0, 30.0], [10.5, 20.0, 30.0], [11.0, 20.5, 30.0]])
    s2 = np.array([[40.0, 50.0, 60.0]])
    path = _build_tt([s1, s2])
    streams = ascending._parse_tt(path)
    assert len(streams) == 2
    assert np.allclose(streams[0], s1, atol=1 / 32.0)
    assert np.allclose(streams[1], s2, atol=1 / 32.0)
    os.remove(path)


def test_endpoint_density_extends_past_the_last_point():
    """The terminal segment walks FORWARD along its own direction.

    Tracking stops at the WM/GM interface; the cortical target is a few mm further along
    the same heading. So density must appear beyond the streamline's last point, in the
    direction it was travelling, and not behind it."""
    vs = float(np.abs(np.diag(ascending._grid()[1])).min())
    # densely sampled: the .tt container delta-encodes as int8 of 32x coordinates, so
    # consecutive points must sit within ~3.9 voxels of each other
    s1 = np.c_[np.arange(10.0, 12.01, 0.5), np.full(5, 20.0), np.full(5, 30.0)]
    path = _build_tt([s1])
    shape, _ = ascending._grid()
    seed = np.zeros(shape, bool)
    seed[10, 20, 30] = True                                   # isolate the far end
    V = ascending._endpoint_density(path, seed_v=seed, tail_mm=0.0, extend_mm=4.0,
                                    step_mm=0.5)
    beyond = V[13:, 20, 30].sum()
    behind = V[:10, 20, 30].sum()
    assert beyond > 0, "no density past the terminal point"
    assert behind == 0, "density deposited behind the streamline"
    # the extension reaches about extend_mm past the end, in voxels
    reach = np.nonzero(V[:, 20, 30])[0].max() - 12
    assert reach <= round(4.0 / vs) + 1
    os.remove(path)


def test_endpoint_density_counts_only_the_far_end():
    """With a seed mask, the end sitting in the seed does not deposit.

    A thalamus-seeded streamline otherwise puts as much weight back in the thalamus as at
    its cortical target, and one that never escaped deposits twice in the seed."""
    shape, _ = ascending._grid()
    seed = np.zeros(shape, bool)
    seed[10, 20, 30] = True                                   # the seed end of s1
    s1 = np.c_[np.arange(10.0, 16.01, 0.5), np.full(13, 20.0), np.full(13, 30.0)]
    s2 = np.array([[10.0, 20.0, 30.0], [10.0, 20.0, 30.0]])   # never left the seed
    path = _build_tt([s1, s2])
    V = ascending._endpoint_density(path, seed_v=seed, tail_mm=0.0, extend_mm=3.0)
    assert V[:13, 20, 30].sum() == 0, "deposited at the seed end"
    assert V[16:, 20, 30].sum() > 0, "nothing at the far end"
    os.remove(path)


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
