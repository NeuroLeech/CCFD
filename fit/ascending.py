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
MANIFEST = os.path.join(ATLAS_DIR, "ascending_manifest.json")

WB_COMMAND = os.environ.get(
    "WB_COMMAND", os.path.expanduser("~/Downloads/workbench/bin_macosxub/wb_command"))
DSI_STUDIO = os.environ.get(
    "DSI_STUDIO", "/Applications/DSI Studio.app/Contents/MacOS/dsi_studio")


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
    """-> (names, masks, affines): each nucleus's mask in its NATIVE space (not resampled),
    normalised to max 1. `label` entries are one-hot from a labelled volume; `label: null`
    entries are already probability maps. The native (MNI) affine is carried so dsi_studio
    can register each seed onto the FIB via its srow matrix."""
    import nibabel as nib
    names, masks, affs = [], [], []
    for e in entries:
        img = nib.load(os.path.join(ATLAS_DIR, e["file"]))
        d = np.asarray(img.dataobj).astype(np.float32)
        if e.get("label") is not None:
            m = (d == int(e["label"])).astype(np.float32)
        else:
            m = d
        mx = float(m.max())
        if mx > 0:
            m = m / mx
        names.append(e["name"])
        masks.append(m.astype(np.float32))
        affs.append(np.asarray(img.affine, float))
        if verbose:
            print(f"  {e['name']:<12s} {int((m > 0).sum())} voxels in native space")
    return names, masks, affs


def _run_dsi(seed_nii, output_dir, seed_count, fib=FIB_PATH, threads=8):
    """Deterministic tracking seeded from one nucleus mask through the group FIB.

    Returns the written .tt.gz path. `output_dir` is a directory: DSI Studio names the
    tract file `<source_basename>.tt.gz` inside it."""
    if not os.path.exists(DSI_STUDIO):
        raise SystemExit(f"  dsi_studio not found at {DSI_STUDIO}; export DSI_STUDIO")
    os.makedirs(output_dir, exist_ok=True)
    cmd = [DSI_STUDIO, "--action=trk", f"--source={fib}", f"--seed={seed_nii}",
           f"--seed_count={seed_count}", "--method=0", "--turning_angle=45",
           "--min_length=10", "--max_length=300", f"--thread_count={threads}",
           f"--output={output_dir}"]
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


def _endpoint_density(tt_path, fib=FIB_PATH):
    """Rasterise streamline ENDPOINTS (voxel coords) onto the FIB grid. Both ends count."""
    shape, _ = _grid(fib)
    pts = [p for s in _parse_tt(tt_path) for p in (s[0], s[-1])]
    xyz = np.round(np.asarray(pts)).astype(np.int64)
    ok = ((xyz[:, 0] >= 0) & (xyz[:, 0] < shape[0]) &
          (xyz[:, 1] >= 0) & (xyz[:, 1] < shape[1]) &
          (xyz[:, 2] >= 0) & (xyz[:, 2] < shape[2]))
    xyz = xyz[ok]
    V = np.zeros(shape, np.float32)
    if len(xyz):
        np.add.at(V, (xyz[:, 0], xyz[:, 1], xyz[:, 2]), 1.0)
    return V


def _to_surface(name, V, tmpdir):
    """Endpoint-density volume (FIB/ICBM152 grid) -> metric on full fsaverage5 (10242).

    neuromaps ships fsaverage5 white and pial surfaces in MNI152 space, vertex-identical
    to the FreeSurfer fsaverage5 the rest of the project uses (fc_vertexwise checks this),
    so ribbon-constrained volume-to-surface maps the MNI-mm density straight onto the
    model's own vertices. midthickness is the mean of white and pial."""
    import nibabel as nib
    from neuromaps.datasets import fetch_atlas, get_atlas_dir
    dens = os.path.join(tmpdir, f"{name}_density.nii.gz")
    nib.save(nib.Nifti1Image(V, _grid()[1]), dens)
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
    """Cached fields for the manifest, on the given cortex. -> (names, fields)."""
    entries = load_manifest()
    names, F = projection_fields(entries, verbose=verbose)
    return names, np.asarray(F[:, :c.nV], np.float32)


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
    a = ap.parse_args()

    entries = load_manifest()
    from mesh_cache import load_cortex
    c = load_cortex("fsaverage5", verbose=False)
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
