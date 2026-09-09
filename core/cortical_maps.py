"""Group-average cortical maps on the simulation mesh, for grading the medium.

Wave speed and damping should vary with cortical structure rather than with the regions
we happen to drive, so the fields need maps: myelin (T1w/T2w), cortical thickness, sulcal
depth, and the first three Margulies functional gradients. Everything except sulc comes
from neuromaps at fsLR 32k and is resampled with the same wb_command call the fMRI took
(fc_vertexwise.resample); sulc is read straight from FreeSurfer's fsaverage5, which is
already the right mesh.

Maps are returned on the Cortex submesh, z-scored, with the few vertices outside the fsLR
medial wall filled from their neighbours so no map has holes in it.

  python core/cortical_maps.py            # fetch, resample, cache, and draw them
"""
import _path  # noqa: F401  - puts the sibling code folders on sys.path
import os
import numpy as np
import nibabel as nib

from paths import CACHE

SOURCES = {
    "myelin":    ("hcps1200", "myelinmap"),
    "thickness": ("hcps1200", "thickness"),
    "grad1":     ("margulies2016", "fcgradient01"),
    "grad2":     ("margulies2016", "fcgradient02"),
    "grad3":     ("margulies2016", "fcgradient03"),
}
FS_SULC = "/Applications/freesurfer/dev/subjects/{mesh}/surf/lh.sulc"
NAMES = ("myelin", "thickness", "sulc", "grad1", "grad2", "grad3")


def _fill_holes(v, cortex, passes=8):
    """Replace zeros (vertices the fsLR ROI does not cover) with a neighbour mean."""
    E = cortex.edges
    v = v.astype(np.float64).copy()
    bad = v == 0
    for _ in range(passes):
        if not bad.any():
            break
        acc = np.zeros(cortex.nV); cnt = np.zeros(cortex.nV)
        good = ~bad
        np.add.at(acc, E[:, 0], np.where(good[E[:, 1]], v[E[:, 1]], 0.0))
        np.add.at(cnt, E[:, 0], good[E[:, 1]].astype(float))
        np.add.at(acc, E[:, 1], np.where(good[E[:, 0]], v[E[:, 0]], 0.0))
        np.add.at(cnt, E[:, 1], good[E[:, 0]].astype(float))
        fix = bad & (cnt > 0)
        v[fix] = acc[fix] / cnt[fix]
        bad = bad & ~fix
    return v


def lb_modes(cortex, n=32, verbose=True):
    """First `n` non-constant Laplace-Beltrami eigenmodes, z-scored. -> (n, nV).

    Built from the SAME discrete operator the medium integrates, not a generic mesh
    Laplacian: swe_rot's `grad` is (h[Ej]-h[Ei])/d and its `divergence` accumulates
    +-l*ue then divides by A, so div(grad(h))_i = (1/A_i) sum_j w_ij (h_j - h_i) with
    w = l/d. The modes are therefore the generalised eigenvectors of K phi = lam A phi,
    K being the cotangent stiffness and A the vertex-area mass matrix, and a coefficient
    on mode j grades exactly the operator the wave propagates through.

    Used as a map BASIS for the speed/damping grading, where the drive stays localised and
    anatomical. That is the opposite of driving eigenmodes directly, which would return a
    global pattern for a global input and say nothing."""
    import scipy.sparse as sp
    from scipy.sparse.linalg import eigsh
    cache = os.path.join(CACHE, f"lb_modes_{cortex.mesh}_{n}.npy")
    if os.path.exists(cache):
        return np.load(cache)
    E = np.asarray(cortex.edges, int)
    w = np.asarray(cortex.l, float) / np.maximum(np.asarray(cortex.d, float), 1e-30)
    nV = cortex.nV
    i = np.concatenate([E[:, 0], E[:, 1], E[:, 0], E[:, 1]])
    j = np.concatenate([E[:, 1], E[:, 0], E[:, 0], E[:, 1]])
    v = np.concatenate([-w, -w, w, w])
    K = sp.csr_matrix((v, (i, j)), shape=(nV, nV))
    M = sp.diags(np.maximum(np.asarray(cortex.A, float), 1e-30))
    ev, V = eigsh(K, k=n + 1, M=M, sigma=-1e-8, which="LM")
    order = np.argsort(ev)
    ev, V = ev[order][1:], V[:, order][:, 1:]          # drop the constant mode
    P = V.T
    P = (P - P.mean(1, keepdims=True)) / np.maximum(P.std(1, keepdims=True), 1e-30)
    if verbose:
        sl = 2 * np.pi / np.sqrt(np.maximum(ev, 1e-30))
        print(f"  {n} LB modes: eigenvalue {ev[0]:.3e}-{ev[-1]:.3e}, "
              f"wavelength 2pi/sqrt(lam) {sl[0]:.0f} -> {sl[-1]:.1f} mm")
    np.save(cache, P)
    return P


def load_maps(cortex, names=NAMES, verbose=True):
    """-> dict name -> (nV,) z-scored map on the Cortex submesh."""
    # "lb<k>" resolves to the k-th Laplace-Beltrami mode, so a map-graded medium can be
    # given a geometric basis instead of an anatomical one with no change downstream:
    # fluid.map_fields only ever asks load_maps for names and stacks what it gets back.
    lb = sorted({int(nm[2:]) for nm in names if nm.startswith("lb") and nm[2:].isdigit()})
    if lb:
        P = lb_modes(cortex, max(max(lb), 8), verbose=verbose)
        out = {f"lb{k}": P[k - 1] for k in lb}
        rest = [nm for nm in names if not nm.startswith("lb")]
        if rest:
            out.update(load_maps(cortex, tuple(rest), verbose=verbose))
        return out

    cache = os.path.join(CACHE, f"cortical_maps_{cortex.mesh}.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        if all(n in z.files for n in names):
            return {n: z[n] for n in names}

    import tempfile
    from neuromaps.datasets import fetch_annotation
    from fc_vertexwise import atlas_files, resample

    files = atlas_files(cortex.mesh)
    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        for name in names:
            if name == "sulc":
                v, _ = nib.freesurfer.read_geometry(
                    FS_SULC.format(mesh=cortex.mesh)) if False else (None, None)
                v = nib.freesurfer.read_morph_data(FS_SULC.format(mesh=cortex.mesh))
                full = np.asarray(v, np.float64)
            else:
                src, desc = SOURCES[name]
                p = fetch_annotation(source=src, desc=desc, space="fsLR", den="32k",
                                     hemi="L", verbose=0)
                p = p[0] if isinstance(p, (list, tuple)) else p
                dense = np.asarray(nib.load(p).darrays[0].data, np.float32)[:, None]
                full = resample(dense, cortex.mesh, files, tmp)[:, 0].astype(np.float64)
            sub = full[np.asarray(cortex.old, int)]
            sub = _fill_holes(sub, cortex)
            sub = (sub - sub.mean()) / max(sub.std(), 1e-12)
            out[name] = sub.astype(np.float32)
            if verbose:
                print(f"  {name:10s} range {sub.min():+.2f} to {sub.max():+.2f}")
    np.savez(cache, **out)
    if verbose:
        print(f"  cached {cache}")
    return out


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.tri import Triangulation
    from mesh_cache import load_cortex
    from render_regimes import _proj
    from paths import RESULTS

    c = load_cortex("fsaverage5", verbose=False)
    maps = load_maps(c)
    print("\n  correlations between maps:")
    ks = list(maps)
    M = np.stack([maps[k] for k in ks])
    R = np.corrcoef(M)
    print("      " + "".join(f"{k:>10s}" for k in ks))
    for i, k in enumerate(ks):
        print(f"  {k:10s}" + "".join(f"{R[i,j]:+10.2f}" for j in range(len(ks))))

    proj = _proj(c.V, c.F)
    fig = plt.figure(figsize=(3.6 * len(proj), 2.4 * len(ks)))
    gs = fig.add_gridspec(len(ks), len(proj), hspace=0.04, wspace=0.02)
    for r, k in enumerate(ks):
        v = maps[k]
        lo, hi = np.percentile(v, [2, 98])
        for j, (xy, vis, nm) in enumerate(proj):
            ax = fig.add_subplot(gs[r, j])
            keep = vis[c.F].all(1)
            ax.tripcolor(Triangulation(xy[:, 0], xy[:, 1], c.F[keep]), v,
                         shading="gouraud", cmap="viridis", vmin=lo, vmax=hi,
                         rasterized=True)
            ax.set_xlim(xy[:, 0].min(), xy[:, 0].max())
            ax.set_ylim(xy[:, 1].min(), xy[:, 1].max())
            ax.set_aspect("equal"); ax.axis("off")
            if r == 0:
                ax.set_title(nm, fontsize=10)
            if j == 0:
                ax.text(-0.03, 0.5, k, rotation=90, transform=ax.transAxes,
                        ha="center", va="center", fontsize=9, family="monospace")
    path = os.path.join(RESULTS, "cortical_maps.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    print(f"\n  wrote {path}")
