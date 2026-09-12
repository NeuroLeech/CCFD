"""Ascending-pathway input basis: nucleus masks -> tractography fields -> modal channels.

The input regions are defined by the ascending systems into cortex rather than by the
Glasser atlas: each nucleus mask (thalamic nuclei, brainstem arousal nuclei, basal
forebrain, hypothalamus, amygdala) is tracked through a group dMRI template in MNI space
and the streamline endpoints become a graded projection field on fsaverage5. Each field is
then decomposed into weighted-Laplacian eigenmodes on its support; channels are
(nucleus, mode) pairs whose amplitudes and phases the cross-spectrum solve sets.

  python fit/ascending.py --fields                    # run tracking + mapping (once)
  python fit/ascending.py --channels --coverage 0.085 # show the channel basis
"""
import _path  # noqa: F401  - puts the sibling code folders on sys.path
import os, re, json, hashlib, subprocess, tempfile, argparse
import numpy as np

from paths import DATA, CACHE

ATLAS_DIR = os.path.join(DATA, "atlases")
FIB_PATH = os.path.join(DATA, "hcp1021.fib.gz")
# The 1 mm HCP1065 template (Zenodo 6324701): the basis fields are tracked through it.
THAL_FIB = os.path.join(DATA, "hcp1065_1mm.fib.gz")
MANIFEST = os.path.join(ATLAS_DIR, "ascending_manifest.json")

WB_COMMAND = os.environ.get(
    "WB_COMMAND", os.path.expanduser("~/Downloads/workbench/bin_macosxub/wb_command"))
DSI_STUDIO = os.environ.get(
    "DSI_STUDIO", "/Applications/DSI Studio.app/Contents/MacOS/dsi_studio")

# THOMAS (Najdenovska 2018) thalamic nuclei, DSI Studio's LUT: odd labels are left.
THOMAS = os.path.join(ATLAS_DIR, "dsistudio", "THOMAS.nii.gz")
THOMAS_LEFT = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]
# HCP-MMP is cortex-only and ships with DSI Studio in the FIB's ICBM152 space.
HCP_MMP = os.path.join(os.path.dirname(DSI_STUDIO), "atlas", "human",
                       "HCP-MMP.nii.gz")


def load_manifest(variant="ascending_v1"):
    with open(MANIFEST) as f:
        return json.load(f)[variant]


def wb(*args):
    """Run wb_command; raises on failure."""
    subprocess.run([WB_COMMAND] + [str(a) for a in args], check=True,
                   capture_output=True)


def _grid_from_dsi(fib=FIB_PATH):
    """-> (shape, affine) by asking DSI Studio; cached. The FIB header is not nifti-
    readable, so dsi_studio --action=ana is the source of truth for dim and trans."""
    key = hashlib.sha1(os.path.basename(fib).encode()).hexdigest()[:12]
    cache = os.path.join(CACHE, f"fib_grid_{key}.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        return tuple(int(v) for v in z["shape"]), z["affine"]
    if not os.path.exists(DSI_STUDIO):
        raise SystemExit(f"  dsi_studio not found at {DSI_STUDIO}; export DSI_STUDIO")
    out = subprocess.run([DSI_STUDIO, "--action=ana", f"--source={fib}"],
                         capture_output=True, text=True).stdout
    out = re.sub(r"\x1b\[[0-9;]*m", "", out)          # strip ANSI colour
    m = re.search(r"dim:\s*([0-9]+)\s+([0-9]+)\s+([0-9]+)", out)
    t = re.search(r"trans:\s*([^\n]*)", out)
    if not m or not t:
        raise SystemExit(f"  could not parse the FIB grid from dsi_studio --action=ana:\n{out}")
    shape = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
    nums = [float(x) for x in re.findall(r"[-0-9.]+", t.group(1))]
    if len(nums) < 16:
        raise SystemExit(f"  could not parse 16 affine numbers from trans line: {t.group(1)!r}")
    affine = np.array(nums[:16], float).reshape(4, 4)
    np.savez(cache, shape=np.array(shape, int), affine=affine)
    return shape, affine


def _grid(fib=FIB_PATH):
    """-> (shape, affine) of the FIB volume. Nifti-readable sources load directly;
    the DSI Studio FIB (custom format) falls back to parsing `dsi_studio --action=ana`."""
    import nibabel as nib
    try:
        img = nib.load(fib)
        return img.shape[:3], np.asarray(img.affine, float)
    except Exception:
        return _grid_from_dsi(fib)


def nucleus_masks(entries, verbose=True):
    """-> (names, masks, affines): each nucleus's mask as a THOMAS one-hot in the
    atlas's native 0.5 mm space, normalised to max 1. The native affine is carried so
    dsi_studio can register each seed onto the FIB via its srow matrix."""
    import nibabel as nib
    img = nib.load(THOMAS)
    d = np.asarray(img.dataobj)
    names, masks, affs = [], [], []
    for e in entries:
        m = (d == int(e["label"])).astype(np.float32)
        names.append(e["name"])
        masks.append(m)
        affs.append(np.asarray(img.affine, float))
        if verbose:
            print(f"  {e['name']:<12s} {int((m > 0).sum())} voxels in native space")
    return names, masks, affs


def _run_dsi(seed_nii, output_dir, seed_count, fib=FIB_PATH, threads=8,
             method=0, turn=45.0, otsu=None):
    """Deterministic tracking seeded from one nucleus mask through the group FIB.

    Returns the written .tt.gz path. `output_dir` is a directory: DSI Studio names the
    tract file `<source_basename>.tt.gz` inside it."""
    if not os.path.exists(DSI_STUDIO):
        raise SystemExit(f"  dsi_studio not found at {DSI_STUDIO}; export DSI_STUDIO")
    os.makedirs(output_dir, exist_ok=True)
    cmd = [DSI_STUDIO, "--action=trk", f"--source={fib}", f"--seed={seed_nii}",
           f"--seed_count={seed_count}", f"--method={method}",
           f"--turning_angle={turn}", "--min_length=10", "--max_length=300",
           f"--thread_count={threads}", f"--output={output_dir}"]
    if otsu is not None:
        cmd.append(f"--otsu_threshold={otsu}")
    subprocess.run(cmd, check=True, capture_output=True)
    out = os.path.join(output_dir, os.path.basename(fib) + ".tt.gz")
    if not os.path.exists(out):
        raise SystemExit(f"  dsi_studio wrote no tract file under {output_dir}")
    return out


# The on-disk .tt.gz container: a gzipped sequence of entries, each
#   uint32 type (0=double 10=float 20=uint32 30=int16 40=uint16 50=uint8 60=uint64)
#   uint32 rows, uint32 cols, uint32 n_subdata, uint32 name_len
#   name bytes, then rows*cols values of the type's size.
# Tract blocks ("track", "track1", ...) are type 50 (uint8) with the streamlines
# delta-encoded: per tract, uint32 count then int32 x,y,z of the first point (all x32),
# then count-3 int8 coordinate deltas. Coordinates are FIB VOXEL indices.

_TT_TYPE_SIZE = {0: 8, 10: 4, 20: 4, 30: 2, 40: 2, 50: 1, 60: 8}


def _parse_tt(tt_path):
    """-> list of (n, 3) float32 streamline arrays, in FIB voxel coordinates."""
    import gzip, struct
    with gzip.open(tt_path, "rb") as f:
        raw = f.read()
    entries = {}
    pos, n = 0, len(raw)
    while pos < n:
        typ, rows, cols, nsub, namelen = struct.unpack_from("<IIIII", raw, pos)
        pos += 20
        name = raw[pos:pos + namelen].rstrip(b"\x00").decode()
        pos += namelen
        elsize = _TT_TYPE_SIZE[typ]
        nbytes = rows * cols * elsize
        entries[name] = raw[pos:pos + nbytes]
        pos += nbytes
    streams = []
    for k, buf in entries.items():
        if k != "track" and not k.startswith("track"):
            continue
        p = 0
        L = len(buf)
        while p + 16 <= L:
            count = struct.unpack_from("<I", buf, p)[0]
            x, y, z = struct.unpack_from("<iii", buf, p + 4)
            n_pts = count // 3
            if n_pts < 1 or p + 16 + max(0, count - 3) > L:
                break
            flat = np.empty(count, np.float32)
            flat[0], flat[1], flat[2] = x, y, z
            d = np.frombuffer(buf, np.int8, count - 3, offset=p + 16)
            for j in range(3, count):
                flat[j] = flat[j - 3] + int(d[j - 3])
            streams.append((flat.reshape(n_pts, 3)) / 32.0)
            p += 16 + (count - 3)
    return streams


def _splat(V, xyz):
    """Trilinear deposit of fractional voxel coordinates into V, in place.

    Nearest-voxel rounding quantises a 1 mm terminal segment into a staircase and puts
    all of a streamline's terminal weight in one voxel; the ribbon mapping then sees a
    spike beside the ribbon rather than a gradient across it."""
    shape = V.shape
    b = np.floor(xyz).astype(np.int64)
    f = xyz - b
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = ((f[:, 0] if dx else 1 - f[:, 0]) *
                     (f[:, 1] if dy else 1 - f[:, 1]) *
                     (f[:, 2] if dz else 1 - f[:, 2]))
                ix, iy, iz = b[:, 0] + dx, b[:, 1] + dy, b[:, 2] + dz
                ok = ((ix >= 0) & (ix < shape[0]) & (iy >= 0) & (iy < shape[1]) &
                      (iz >= 0) & (iz < shape[2]) & (w > 0))
                if ok.any():
                    np.add.at(V, (ix[ok], iy[ok], iz[ok]), w[ok])


def _terminal_samples(s, vs, tail_mm, extend_mm, step_mm):
    """-> (m, 3) voxel coords sampling one streamline's terminal segment, plus a
    directed extension beyond its last point.

    Tracking stops at the WM/GM interface where the fibre orientation distribution gives
    out. The cortical target is a few mm FURTHER ALONG the same direction, through the
    ribbon - and for a sulcal target that direction is lateral, not radially outward, so
    an isotropic dilation reaches the wrong bank. Walking the terminal tangent forward
    puts the weight where the streamline was heading.

    The tail is sampled as well as the extension so that a streamline running along a
    gyral blade deposits over its length rather than a single point."""
    d = np.linalg.norm(np.diff(s, axis=0), axis=1) * vs
    if len(d) == 0:
        return s[-1:].copy()
    arc = np.concatenate([[0.0], np.cumsum(d)])
    tail = s[arc >= arc[-1] - tail_mm]
    if len(tail) < 2:
        tail = s[-2:]
    u = tail[-1] - tail[max(0, len(tail) - 3)]
    n = np.linalg.norm(u) * vs
    pts = [tail]
    if n > 1e-6 and extend_mm > 0:
        u = u / (n / vs)                                   # unit vector, voxel units
        k = np.arange(step_mm, extend_mm + 1e-9, step_mm) / vs
        pts.append(tail[-1][None, :] + k[:, None] * u[None, :])
    return np.concatenate(pts, axis=0)


def _endpoint_density(tt_path, fib=FIB_PATH, seed_v=None, tail_mm=4.0,
                      extend_mm=4.0, step_mm=0.5):
    """Rasterise streamline TERMINAL SEGMENTS (voxel coords) onto the FIB grid.

    `seed_v` is the seed mask on the FIB grid. When given, only the end NOT in the seed
    is counted: a thalamus-seeded streamline otherwise deposits as much weight back in
    the thalamus as it does at its cortical target, and a streamline that never left the
    seed deposits twice there and nowhere else."""
    shape, _ = _grid(fib)
    vs = float(np.abs(np.diag(_grid(fib)[1])).min())
    V = np.zeros(shape, np.float32)

    def in_seed(pt):
        if seed_v is None:
            return False
        e = np.round(pt).astype(int)
        return (all(0 <= e[i] < shape[i] for i in range(3)) and bool(seed_v[tuple(e)]))

    for s in _parse_tt(tt_path):
        if len(s) < 2:
            continue
        a_in, b_in = in_seed(s[0]), in_seed(s[-1])
        if seed_v is not None and a_in and b_in:
            continue                                   # never escaped the seed
        ends = []
        if seed_v is None:
            ends = [s, s[::-1]]
        elif a_in and not b_in:
            ends = [s]
        elif b_in and not a_in:
            ends = [s[::-1]]
        else:
            ends = [s, s[::-1]]                        # neither end in seed: count both
        for e in ends:
            _splat(V, _terminal_samples(e, vs, tail_mm, extend_mm, step_mm))
    return V


def _vol_to_surface(V, affine, name, tmpdir):
    """MNI-mm volume -> metric on full fsaverage5 (10242 vertices).

    neuromaps ships fsaverage5 white and pial surfaces in MNI152 space, vertex-identical
    to the FreeSurfer fsaverage5 the rest of the project uses (fc_vertexwise checks this),
    so ribbon-constrained volume-to-surface maps the MNI-mm density straight onto the
    model's own vertices. midthickness is the mean of white and pial."""
    import nibabel as nib
    from neuromaps.datasets import fetch_atlas, get_atlas_dir
    dens = os.path.join(tmpdir, f"{name}_density.nii.gz")
    nib.save(nib.Nifti1Image(np.asarray(V, np.float32), affine), dens)
    fetch_atlas("fsaverage", "10k")
    d = get_atlas_dir("fsaverage")
    white = f"{d}/tpl-fsaverage_den-10k_hemi-L_white.surf.gii"
    pial = f"{d}/tpl-fsaverage_den-10k_hemi-L_pial.surf.gii"
    mid = os.path.join(tmpdir, f"{name}_midthickness.surf.gii")
    w = nib.load(white)
    p = nib.load(pial)
    w.darrays[0].data = ((w.darrays[0].data + p.darrays[0].data) / 2).astype(np.float32)
    nib.save(w, mid)
    out = os.path.join(tmpdir, f"{name}.func.gii")
    wb("-volume-to-surface-mapping", dens, mid, out, "-ribbon-constrained", white, pial)
    return np.asarray(nib.load(out).darrays[0].data, np.float32)


def _to_surface(name, V, tmpdir, fib=FIB_PATH, dilate_mm=0.0):
    """Endpoint-density volume (FIB grid) -> metric on full fsaverage5.

    `dilate_mm` is an ISOTROPIC maximum filter, and it was how terminal density used to
    be pushed into the white-pial ribbon. It reaches the nearest ribbon in every
    direction at once, so a streamline ending in a gyral blade lights both banks and the
    dorsal convexity accumulates from everything beneath it. `_terminal_samples` now
    walks the terminal tangent instead, which is directed, so the default is off. Kept as
    a knob for comparison against the old fields."""
    if dilate_mm > 0:
        from scipy.ndimage import maximum_filter
        vs = float(np.abs(np.diag(_grid(fib)[1])).min())
        V = maximum_filter(np.asarray(V, np.float32),
                           size=int(round(dilate_mm / vs) * 2 + 1))
    return _vol_to_surface(V, _grid(fib)[1], name, tmpdir)


def projection_fields(entries, seed_count=20000, cache=True, verbose=True):
    """-> (names, fields): endpoint-density fields on the cortex submesh (c.nV)."""
    from mesh_cache import load_cortex
    import nibabel as nib
    c = load_cortex("fsaverage5", verbose=False)
    key = (hashlib.sha1(
        json.dumps([e["name"] for e in entries]).encode()).hexdigest()[:12])
    cache_f = os.path.join(CACHE, f"ascending_fields_{key}_{seed_count}.npz")
    if cache and os.path.exists(cache_f):
        z = np.load(cache_f, allow_pickle=True)
        if verbose:
            print(f"  loaded {cache_f}")
        return list(z["names"]), np.asarray(z["fields"])
    names, masks, affs = nucleus_masks(entries, verbose=verbose)
    fields = []
    for nm, mk, af in zip(names, masks, affs):
        with tempfile.TemporaryDirectory() as td:
            seed = os.path.join(td, f"{nm}_seed_mni.nii.gz")
            outdir = os.path.join(td, "out")
            nib.save(nib.Nifti1Image(mk, af), seed)
            tt = _run_dsi(seed, outdir, seed_count)
            V = _endpoint_density(tt)
            f_full = _to_surface(nm, V, td)
            fields.append(f_full[c.old].astype(np.float32))
            if verbose:
                print(f"  {nm:<12s} {int((f_full > 0).sum())} vertices reached, "
                      f"field sum {f_full.sum():.0f}")
    F = np.stack(fields)
    np.savez(cache_f, names=np.array(names, dtype=object), fields=F)
    if verbose:
        print(f"  wrote {cache_f}")
    return names, F


def load_fields(c, verbose=True):
    """Per-nucleus cortical projection fields from the 1 mm FIB, on the given cortex.

    -> (names, fields). Built by seeding the whole left thalamus (200k seeds,
    probabilistic, turn 60, otsu 0.35) and attributing each streamline to the THOMAS
    nucleus at its seed; cached in data/cache. Each field is normalised to unit max so
    the coverage threshold tau is comparable across nuclei; the solve sets the
    per-nucleus amplitude."""
    names, F = thalamus_subfields(fib=THAL_FIB, verbose=verbose)
    F = np.asarray(F[:, :c.nV], np.float32)
    for i in range(F.shape[0]):
        mx = float(F[i].max())
        if mx > 0:
            F[i] = F[i] / mx
    return names, F




def support(c, f, tau):
    """Vertices of the largest connected component of f > tau, in mesh-adjacency order."""
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components
    m = np.asarray(f) > tau
    if m.sum() == 0:
        return np.zeros(0, np.int64)
    pos = -np.ones(c.nV, np.int64)
    pos[m] = np.arange(m.sum())
    E = c.edges
    keep = (pos[E[:, 0]] >= 0) & (pos[E[:, 1]] >= 0)
    n, lab = connected_components(
        sp.coo_matrix((np.ones(keep.sum()), (pos[E[keep, 0]], pos[E[keep, 1]])),
                      shape=(m.sum(),) * 2), directed=False)
    big = np.argmax(np.bincount(lab))
    return np.flatnonzero(m)[lab == big]


def _field_modes(c, f, verts, k):
    """The k smallest eigenmodes of the graph Laplacian on `verts`.

    Edges are weighted by 1 - 0.5*(f_i + f_j), floored at 1e-3 so the graph cannot
    disconnect; the periphery (low f) is therefore cheap to cut. Mode 0 is near-constant
    on the support, so channel (nucleus, 0) is approximately the plain projection field."""
    import scipy.sparse as sp
    from scipy.sparse.linalg import eigsh
    pos = -np.ones(c.nV, np.int64)
    pos[verts] = np.arange(len(verts))
    E = c.edges
    keep = (pos[E[:, 0]] >= 0) & (pos[E[:, 1]] >= 0)
    a, b = pos[E[keep, 0]], pos[E[keep, 1]]
    wgt = np.maximum(1.0 - 0.5 * (np.asarray(f)[E[keep, 0]] + np.asarray(f)[E[keep, 1]]),
                     1e-3)
    A = sp.coo_matrix((wgt, (a, b)), shape=(len(verts),) * 2)
    A = A + A.T
    L = sp.diags(np.asarray(A.sum(1)).ravel()) - A
    _, V = eigsh(L.astype(float), k=min(k, len(verts)), sigma=-1e-6, which="LM")
    # fix the sign ambiguity of the mode-0 (near-constant) vector: positive mean
    V = V * np.sign(V.sum(0))[None, :]
    return V


def coverage(c, fields, tau):
    """-> (fraction, mm2) of cortex where max_n f_n > tau."""
    m = np.max(np.stack([np.asarray(f) for f in fields]), axis=0)
    on = m > tau
    area = np.asarray(c.A, float)
    return float(area[on].sum() / area.sum()), float(area[on].sum())


def tau_for_coverage(c, names, fields, frac):
    """The smallest tau whose coverage is at most `frac` (bisection over densities)."""
    m = np.max(np.stack([np.asarray(f) for f in fields]), axis=0)
    area = np.asarray(c.A, float)
    tot = float(area.sum())
    vals = np.sort(np.unique(m[m > 0]))
    lo, hi = 0, len(vals) - 1

    def cov(i):
        return float(area[m > vals[i]].sum() / tot)

    while lo < hi:
        mid = (lo + hi) // 2
        if cov(mid) <= frac:
            hi = mid
        else:
            lo = mid + 1
    return float(vals[lo])


def modal_channels(c, names, fields, tau, budget=80, k_max=4, verbose=True):
    """-> (P, tags, (coverage_frac, coverage_mm2), meta).

    k_n = clamp(round(area_n / target), 1, k_max) with target = total support area /
    budget - the same area-proportional logic split_parcels uses. Profiles are
    envelope x mode: P[nuc, j] = f_n * psi_j on the support, 0 elsewhere."""
    area = np.asarray(c.A, float)
    sups = [support(c, f, tau) for f in fields]
    areas = np.array([area[v].sum() for v in sups])
    target = float(areas.sum() / float(budget))
    P, tags, k_n = [], [], {}
    for i, (nm, f, v) in enumerate(zip(names, fields, sups)):
        if len(v) == 0:
            k_n[nm] = 0
            continue
        k = int(np.clip(round(areas[i] / target), 1, k_max))
        V = _field_modes(c, f, v, k)
        k_n[nm] = V.shape[1]
        for j in range(V.shape[1]):
            row = np.zeros(c.nV, np.float32)
            row[v] = np.asarray(f)[v] * V[:, j]
            P.append(row)
            tags.append(f"{nm}:{j}")
    frac = coverage(c, fields, tau)
    if verbose:
        print(f"  tau {tau:g}: coverage {frac[0]:.2%} ({frac[1]:.0f} mm2), "
              f"{len(tags)} channels from {len(names)} nuclei")
        for i, nm in enumerate(names):
            print(f"    {nm:<14s} {k_n[nm]} modes ({areas[i]:.0f} mm2)")
    return np.asarray(P, np.float32), tags, frac, dict(k_n=k_n)


def _left_thalamus_seed(tmpdir):
    """Union of the LEFT THOMAS labels, in THOMAS native space -> seed nifti path."""
    import nibabel as nib
    img = nib.load(THOMAS)
    m = np.isin(np.asarray(img.dataobj), THOMAS_LEFT).astype(np.float32)
    p = os.path.join(tmpdir, "left_thalamus_mni.nii.gz")
    nib.save(nib.Nifti1Image(m, np.asarray(img.affine, float), img.header), p)
    return p


def _cortical_mask(fib=FIB_PATH):
    """HCP-MMP parcels (cortex only) resampled onto the FIB grid -> bool volume."""
    import nibabel as nib
    from scipy.ndimage import affine_transform
    key = hashlib.sha1((fib + HCP_MMP).encode()).hexdigest()[:12]
    cache = os.path.join(CACHE, f"cortical_mask_{key}.npz")
    if os.path.exists(cache):
        return np.load(cache)["m"] > 0
    img = nib.load(HCP_MMP)
    d = (np.asarray(img.dataobj) > 0).astype(np.float32)
    shape, aff = _grid(fib)
    T = np.linalg.inv(np.asarray(img.affine, float)) @ aff
    m = affine_transform(d, T, output_shape=shape, order=0)
    np.savez(cache, m=m)
    return m > 0.5


def _seed_on_grid(seed_nii, fib=FIB_PATH):
    """Seed volume resampled onto the FIB grid -> bool volume."""
    import nibabel as nib
    from scipy.ndimage import affine_transform
    img = nib.load(seed_nii)
    shape, aff = _grid(fib)
    T = np.linalg.inv(np.asarray(img.affine, float)) @ aff
    m = affine_transform((np.asarray(img.dataobj) > 0).astype(np.float32),
                         T, output_shape=shape, order=0)
    return m > 0.5


def thalamus_gate(seed_count=150000, method=0, turn=45.0, otsu=None, threads=None,
                  fib=FIB_PATH, verbose=True):
    """Seed the whole LEFT thalamus and measure whether tracks reach cortex.

    -> dict with the total streamlines, the length distribution (mm, 2 mm FIB voxels),
    the fraction of streamlines whose far endpoint lands in the HCP-MMP cortical ribbon,
    and the fraction that never left the seed."""
    threads = threads or os.cpu_count()
    cort = _cortical_mask(fib)
    shape, _ = _grid(fib)
    with tempfile.TemporaryDirectory() as td:
        seed = _left_thalamus_seed(td)
        if verbose:
            print(f"  seed: left thalamus ({int(np.asarray(__import__('nibabel').load(seed).dataobj).sum())} 0.5-mm voxels), "
                  f"{seed_count} seeds, method {method}, turn {turn:g}, otsu {otsu}")
        tt = _run_dsi(seed, os.path.join(td, "out"), seed_count, fib=fib,
                      threads=threads, method=method, turn=turn, otsu=otsu)
        streams = _parse_tt(tt)
        seed_v = _seed_on_grid(seed, fib)
        n = len(streams)
        if verbose:
            print(f"  parsed {n} streamlines")
        if n == 0:
            return dict(total=0, far_cortex_frac=np.nan, both_in_seed_frac=np.nan,
                        len_median_mm=np.nan, len_p90_mm=np.nan, len_max_mm=np.nan)
        lens = []
        cort_hit = 0
        dead = 0
        vs = float(np.abs(np.diag(_grid(fib)[1])).min())
        for s in streams:
            lens.append(float(np.linalg.norm(np.diff(s, axis=0), axis=1).sum() * vs))
            e0 = np.round(s[0]).astype(int)
            e1 = np.round(s[-1]).astype(int)
            def ok(e):
                return all(0 <= e[i] < shape[i] for i in range(3))
            in0 = ok(e0) and seed_v[tuple(e0)]
            in1 = ok(e1) and seed_v[tuple(e1)]
            if in0 and in1:
                dead += 1
                continue
            hit = ((ok(e0) and cort[tuple(e0)]) or (ok(e1) and cort[tuple(e1)]))
            cort_hit += int(hit)
        lens = np.asarray(lens)
        r = dict(
            total=n, cort_hit_frac=float(cort_hit / n),
            both_in_seed_frac=float(dead / n),
            len_median_mm=float(np.median(lens)), len_p90_mm=float(np.percentile(lens, 90)),
            len_max_mm=float(lens.max()))
        if verbose:
            print(f"  streamlines with an end in the cortical ribbon: "
                  f"{r['cort_hit_frac']:.2%} ({cort_hit}/{n})")
            print(f"  both endpoints in seed (never escaped): {r['both_in_seed_frac']:.2%}")
            print(f"  length mm: median {r['len_median_mm']:.0f}, "
                  f"p90 {r['len_p90_mm']:.0f}, max {r['len_max_mm']:.0f}")
        return r


def gate_ladder(seed_count=50000, grid=None, fib=FIB_PATH, threads=None, verbose=True):
    """Run `thalamus_gate` over a small tracking grid and report where tracks end.

    The plan's step-1 ladder, run as a table rather than one setting at a time: grey
    matter needs a lower QA threshold than the default and a wider turning angle than a
    deterministic tracker's, and which of the two is binding is not predictable from the
    FIB. Seed count is deliberately low - this ranks settings, it does not build the
    field."""
    grid = grid or [(m, t, o) for m in (0, 1) for t in (45.0, 60.0, 75.0)
                    for o in (0.6, 0.45, 0.35, 0.25)]
    rows = []
    if verbose:
        print(f"  {seed_count} seeds per setting, {len(grid)} settings\n")
        print(f"  {'method':>6s} {'turn':>5s} {'otsu':>5s} {'tracks':>8s} "
              f"{'end in cortex':>14s} {'stuck in seed':>14s} {'med mm':>7s}")
    for m, t, o in grid:
        try:
            r = thalamus_gate(seed_count, method=m, turn=t, otsu=o, fib=fib,
                              threads=threads, verbose=False)
        except SystemExit as e:
            if verbose:
                print(f"  {m:>6d} {t:>5g} {o:>5g}   failed: {e}")
            continue
        r.update(method=m, turn=t, otsu=o)
        rows.append(r)
        if verbose:
            print(f"  {m:>6d} {t:>5g} {o:>5g} {r['total']:>8d} "
                  f"{r['cort_hit_frac']:>13.1%} {r['both_in_seed_frac']:>13.1%} "
                  f"{r['len_median_mm']:>7.0f}", flush=True)
    if rows and verbose:
        b = max(rows, key=lambda r: r["cort_hit_frac"])
        print(f"\n  best: method {b['method']}, turn {b['turn']:g}, otsu {b['otsu']:g}"
              f" -> {b['cort_hit_frac']:.1%} of streamlines end in the cortical ribbon")
    return rows


def thalamus_field(seed_count=200000, method=1, turn=60.0, otsu=0.35, threads=None,
                   cache=True, verbose=True, fib=FIB_PATH, tail_mm=4.0, extend_mm=4.0,
                   dilate_mm=0.0):
    """The ONE vertex-level thalamic cortical projection field.

    Seeded from the whole left thalamus (THOMAS odd labels) through the group FIB with
    the gate-winning settings (probabilistic, turn 60, otsu 0.35 -> ~36% of streamlines
    end in the cortical ribbon); streamline endpoints rasterise into a density volume and
    map to fsaverage5 through the ribbon. Returns the (10242,) full-hemisphere field."""
    threads = threads or os.cpu_count()
    # the endpoint-extraction parameters belong in the key: the same tracking settings
    # with a different terminal walk are a DIFFERENT field, and without them a rebuild
    # silently returns the previous one
    key = hashlib.sha1(
        f"{seed_count},{method},{turn},{otsu},{os.path.basename(fib)},"
        f"{tail_mm},{extend_mm},{dilate_mm}".encode()
    ).hexdigest()[:12]
    cache_f = os.path.join(CACHE, f"thalamus_field_{key}.npz")
    if cache and os.path.exists(cache_f):
        f = np.load(cache_f)["field"]
        if verbose:
            print(f"  loaded {cache_f}")
        return f
    with tempfile.TemporaryDirectory() as td:
        seed = _left_thalamus_seed(td)
        tt = _run_dsi(seed, os.path.join(td, "out"), seed_count, fib=fib,
                      threads=threads, method=method, turn=turn, otsu=otsu)
        V = _endpoint_density(tt, fib=fib, seed_v=_seed_on_grid(seed, fib),
                              tail_mm=tail_mm, extend_mm=extend_mm)
        f = _to_surface("thalamus", V, td, fib=fib, dilate_mm=dilate_mm)
    np.savez(cache_f, field=f)
    if verbose:
        print(f"  wrote {cache_f}")
    return f


def _first_in_seed_label(s, th_d, th_T, names):
    """THOMAS label of the first stored point inside the left-thalamus mask.

    DSI Studio walks each seed in both directions and stores the full path, so the
    stream's first point is a walk END (usually brainstem), not the seed. Walking from
    the start, the first point with a THOMAS label is the seed itself."""
    import nibabel as nib
    for p in s:
        v = np.round(nib.affines.apply_affine(th_T, p)).astype(int)
        if all(0 <= v[i] < th_d.shape[i] for i in range(3)):
            l = int(th_d[tuple(v)])
            if l in names:
                return l
    return -1


def thalamus_subfields(seed_count=200000, method=1, turn=60.0, otsu=0.35, threads=None,
                       verbose=True, fib=FIB_PATH, tail_mm=4.0, extend_mm=4.0,
                       dilate_mm=0.0):
    """-> (names, fields (M, c.nV)): per-nucleus cortical endpoint fields.

    Each streamline is attributed to the THOMAS nucleus at its SEED (the first stored
    point inside the left-thalamus mask); only the FAR terminal segment rasterises, so a
    nucleus's field is where its streamlines reach the cortical ribbon and not also where
    they started."""
    import nibabel as nib
    threads = threads or os.cpu_count()
    key = hashlib.sha1(
        f"sub{seed_count},{method},{turn},{otsu},{os.path.basename(fib)},"
        f"{tail_mm},{extend_mm},{dilate_mm}".encode()
    ).hexdigest()[:12]
    cache_f = os.path.join(CACHE, f"thalamus_subfields_{key}.npz")
    th = nib.load(THOMAS)
    th_d = np.asarray(th.dataobj)
    th_T = np.linalg.inv(np.asarray(th.affine, float)) @ _grid(fib)[1]
    with tempfile.TemporaryDirectory() as td:
        seed = _left_thalamus_seed(td)
        tt = _run_dsi(seed, os.path.join(td, "out"), seed_count, fib=fib,
                      threads=threads, method=method, turn=turn, otsu=otsu)
        streams = _parse_tt(tt)
        shape = _grid(fib)[0]
        vs = float(np.abs(np.diag(_grid(fib)[1])).min())
        seed_v = _seed_on_grid(seed, fib)
        V = np.zeros((len(THOMAS_LEFT),) + shape, np.float32)
        n_att = np.zeros(len(THOMAS_LEFT), int)

        def _in_seed(pt):
            e = np.round(pt).astype(int)
            return (all(0 <= e[j] < shape[j] for j in range(3))
                    and bool(seed_v[tuple(e)]))

        for st in streams:
            if len(st) < 2:
                continue
            l = _first_in_seed_label(st, th_d, th_T, THOMAS_LEFT)
            if l < 0:
                continue
            a_in, b_in = _in_seed(st[0]), _in_seed(st[-1])
            if a_in and b_in:
                continue                               # never escaped the thalamus
            i = THOMAS_LEFT.index(l)
            n_att[i] += 1
            walks = [st] if (a_in and not b_in) else \
                    [st[::-1]] if (b_in and not a_in) else [st, st[::-1]]
            for e in walks:
                _splat(V[i], _terminal_samples(e, vs, tail_mm, extend_mm, 0.5))
        from mesh_cache import load_cortex
        c = load_cortex("fsaverage5", verbose=False)
        F = np.stack([_to_surface(f"nuc{l}", V[i], td, fib=fib,
                                  dilate_mm=dilate_mm)[c.old]
                      for i, l in enumerate(THOMAS_LEFT)])
        lut = {1: "AV", 3: "VA", 5: "VLa", 7: "VLp", 9: "VPL", 11: "Pul",
               13: "LGN", 15: "MGN", 17: "CM", 19: "MD", 21: "Hb", 23: "MTT"}
        names = [lut[l] for l in THOMAS_LEFT]
        if verbose:
            for nm, na in zip(names, n_att):
                print(f"  {nm:<5s} {na:>7d} streamlines attributed, "
                      f"field mass {F[names.index(nm)].sum():.0f}")
    np.savez(cache_f, names=np.array(names, dtype=object), fields=F,
             n_attributed=n_att)
    if verbose:
        print(f"  wrote {cache_f}")
    return names, F


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fields", action="store_true",
                    help="track every nucleus through the FIB and map endpoints to "
                         "fsaverage5 (cached)")
    ap.add_argument("--channels", action="store_true",
                    help="print the modal channel basis for the cached fields")
    ap.add_argument("--seed-count", type=int, default=20000, dest="seed_count")
    ap.add_argument("--tau", type=float, default=0.2)
    ap.add_argument("--budget", type=int, default=80)
    ap.add_argument("--coverage", type=float, default=0.0,
                    help="pick tau for this driven-area fraction (overrides --tau)")
    ap.add_argument("--quality", action="store_true",
                    help="quality gate: seed the whole left thalamus and report where "
                         "the streamlines end")
    ap.add_argument("--fib", default=FIB_PATH,
                    help="FIB file to track through (default: the 2 mm template)")
    ap.add_argument("--method", type=int, default=0, choices=(0, 1),
                    help="tracking method: 0 deterministic, 1 probabilistic (gate only)")
    ap.add_argument("--turn", type=float, default=45.0,
                    help="turning angle in degrees (gate only)")
    ap.add_argument("--otsu", type=float, default=None,
                    help="otsu threshold override (gate only)")
    ap.add_argument("--thalamus", action="store_true",
                    help="build the whole-thalamus cortical projection field (cached)")
    ap.add_argument("--ladder", action="store_true",
                    help="quality gate over a grid of tracking settings, ranked by the "
                         "fraction of streamlines ending in the cortical ribbon")
    a = ap.parse_args()

    entries = load_manifest()
    from mesh_cache import load_cortex
    c = load_cortex("fsaverage5", verbose=False)
    if a.ladder:
        gate_ladder(a.seed_count, fib=a.fib)
        return
    if a.quality:
        thalamus_gate(a.seed_count, method=a.method, turn=a.turn, otsu=a.otsu,
                      fib=a.fib)
        return
    if a.thalamus:
        f = thalamus_field(a.seed_count, method=a.method, turn=a.turn,
                           otsu=a.otsu if a.otsu is not None else 0.35, fib=a.fib)
        on = f[c.old]
        print(f"  field over {int((on > 0).sum())} cortex vertices, "
              f"peak {on.max():.0f}, sum {on.sum():.0f}")
        return
    if a.fields:
        names, fields = projection_fields(entries, seed_count=a.seed_count)
        print(f"  {len(names)} projection fields on {fields.shape[1]} vertices")
        return
    if a.channels:
        names, fields = load_fields(c)
        tau = a.tau
        if a.coverage > 0:
            tau = tau_for_coverage(c, names, fields, a.coverage)
        P, tags, frac, meta = modal_channels(c, names, fields, tau, a.budget)
        print(f"  tau {tau:g}: {frac[0]:.2%} of cortex ({frac[1]:.0f} mm2), "
              f"{len(tags)} channels")
        for nm, k in meta["k_n"].items():
            print(f"    {nm:<12s} {k} modes")
        return
    ap.print_help()


if __name__ == "__main__":
    main()
