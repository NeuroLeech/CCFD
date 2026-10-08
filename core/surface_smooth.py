"""Geodesic Gaussian smoothing on the fsaverage5 cortex, the counterpart of the target's 6 mm.

The empirical target is XCP-D's `denoisedSmoothed` output: 6 mm FWHM on the fsLR midthickness,
then resampled to fsaverage5. The model was scored unsmoothed, so its observable carried spatial
detail the measurement cannot. This builds the matching operator for the model's own mesh.

DISTANCES are on the MIDTHICKNESS, (white + pial) / 2 from nilearn's fsaverage5 - the surface
wb_command smooths on - not the inflated surface the medium is built on, whose distances are
not cortical millimetres. They are shortest paths along mesh edges within the cortex submesh
(so nothing crosses the medial wall, as XCP-D's -roi prevents), which run a few percent longer
than true geodesics; the self-check reports the kernel's effective width so that is visible.

    K[i, j] = exp(-d_ij^2 / (2 sigma^2)), d_ij <= 3 sigma, rows normalised to 1

applied to frames (T, nV) as frames @ K.T. Spatial and temporal filters commute, so it can go
before or after the bandpass; for an envelope observable it must come after the square.

  python core/surface_smooth.py --fwhm 6
"""
import _path  # noqa: F401
import os, argparse
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
from scipy.sparse.csgraph import dijkstra

from paths import CACHE


def midthickness(mesh="fsaverage5"):
    import nibabel as nib
    from nilearn import datasets
    f = datasets.fetch_surf_fsaverage(mesh)
    w = nib.load(f["white_left"]).darrays[0].data.astype(np.float64)
    p = nib.load(f["pial_left"]).darrays[0].data.astype(np.float64)
    F = nib.load(f["white_left"]).darrays[1].data.astype(np.int64)
    return 0.5 * (w + p), F


def smoothing_matrix(cortex, fwhm=6.0, mesh="fsaverage5", cache=True):
    """-> (nV, nV) row-stochastic CSR on the cortex submesh's vertices (cortex.old order)."""
    path = os.path.join(CACHE, f"surfsmooth_{mesh}_fwhm{fwhm:g}.npz")
    if cache and os.path.exists(path):
        z = np.load(path)
        return csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
    V, F = midthickness(mesh)
    old = np.asarray(cortex.old)
    remap = -np.ones(len(V), np.int64); remap[old] = np.arange(len(old))
    Fc = remap[F]; Fc = Fc[(Fc >= 0).all(1)]                     # faces inside the cortex
    e = np.concatenate([Fc[:, [0, 1]], Fc[:, [1, 2]], Fc[:, [2, 0]]])
    e = np.unique(np.sort(e, 1), axis=0)
    Vc = V[old]
    L = np.linalg.norm(Vc[e[:, 0]] - Vc[e[:, 1]], axis=1)
    n = len(old)
    Gr = coo_matrix((np.r_[L, L], (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])),
                    shape=(n, n)).tocsr()
    # Edge paths run longer than geodesics, so a kernel built at sigma comes out narrower than
    # asked: 5.2 mm effective for 6 requested. Built twice - once to measure, once with sigma
    # scaled by requested / effective - so the effective width is the one asked for.
    sig = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    D = dijkstra(Gr, directed=False, limit=4.0 * sig)            # inf beyond the cutoff
    rows, cols = np.nonzero(np.isfinite(D))
    d = D[rows, cols]
    for _ in range(2):
        m = d <= 3.0 * sig
        w = np.exp(-0.5 * (d[m] / sig) ** 2)
        K = coo_matrix((w, (rows[m], cols[m])), shape=(n, n)).tocsr()
        K = csr_matrix(K.multiply(1.0 / np.asarray(K.sum(1)).ravel()[:, None]))
        sig = sig * fwhm / effective_fwhm(K, Vc)
    if cache:
        os.makedirs(CACHE, exist_ok=True)
        np.savez(path, data=K.data, indices=K.indices, indptr=K.indptr, shape=K.shape)
    return K


def effective_fwhm(K, Vc, n=500, seed=0):
    """Median over rows of the kernel's width from its second moment about the weighted
    centroid in straight-line distance: for a 2-D gaussian E[r^2] = 2 sigma^2."""
    rng = np.random.default_rng(seed)
    out = []
    for i in rng.choice(K.shape[0], min(n, K.shape[0]), replace=False):
        a, b = K.indptr[i], K.indptr[i + 1]
        j, w = K.indices[a:b], K.data[a:b]
        mu = (w[:, None] * Vc[j]).sum(0)
        out.append(np.sqrt((w * ((Vc[j] - mu) ** 2).sum(1)).sum() / 2.0))
    return 2.0 * np.sqrt(2.0 * np.log(2.0)) * float(np.median(out))


def smooth(frames, K):
    """(T, nV) -> (T, nV), each frame smoothed over the surface."""
    return np.asarray(K @ np.asarray(frames, np.float64).T).T


def _selfcheck(fwhm):
    from mesh_cache import load_cortex
    c = load_cortex("fsaverage5", verbose=False)
    K = smoothing_matrix(c, fwhm)
    V, _ = midthickness()
    Vc = V[np.asarray(c.old)]
    nnz = np.diff(K.indptr)
    fe = effective_fwhm(K, Vc, seed=1)
    rng = np.random.default_rng(0)
    print(f"  fwhm {fwhm:g} mm on the midthickness: {c.nV} vertices, neighbours per row median "
          f"{int(np.median(nnz))} (5-95% {int(np.percentile(nnz, 5))}-{int(np.percentile(nnz, 95))}), "
          f"effective FWHM {fe:.2f} mm (straight-line second moment)")
    # what it does to white noise: neighbouring-vertex correlation before and after
    X = rng.standard_normal((200, c.nV))
    Y = smooth(X, K)
    I, J = K.nonzero(); m = I < J
    I, J = I[m][:5000], J[m][:5000]
    cr = lambda Z: np.mean([np.corrcoef(Z[:, i], Z[:, j])[0, 1] for i, j in zip(I[:500], J[:500])])
    print(f"  white noise, mean correlation between vertices inside the kernel: before "
          f"{cr(X):+.3f}, after {cr(Y):+.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fwhm", type=float, default=6.0)
    a = ap.parse_args()
    _selfcheck(a.fwhm)
