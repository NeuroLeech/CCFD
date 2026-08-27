"""Moran's I of FC maps on the cortical mesh, empirical against model.

Every row of an FC matrix is a map over the surface, so an FC matrix has a whole
distribution of Moran's I - one per seed - and the mean of that distribution is a
one-number summary of how spatially smooth the connectivity structure is.

A single lag is not enough to separate the two failure modes we have seen: a field can
be smooth because it is genuinely organised at a network scale, or smooth because it is
nearly static and low rank. Moran's I is therefore computed over rings of the mesh graph
(1-ring neighbours, 2-ring, ...), giving a correlogram whose decay is the spatial scale.

  python fc_moran.py                                   # empirical + every model run found
  python fc_moran.py --rings 1 2 4 8 --n-seeds 2000
"""
import os, glob, argparse
import numpy as np
import scipy.sparse as sp

from mesh_cache import load_cortex
from fc_score import FCTarget

DEFAULT_RINGS = (1, 2, 3, 5, 8, 12)


def ring_weights(cortex, cols, rings=DEFAULT_RINGS):
    """-> {k: sparse (V,V) 0/1 matrix of vertex pairs exactly k graph hops apart}.

    Built on the simulation submesh and then restricted to the vertices the FC covers,
    so the weights follow the cortical sheet rather than Euclidean proximity - two banks
    of a sulcus are far apart here even where they nearly touch in space."""
    E = cortex.edges
    n = cortex.nV
    A = sp.coo_matrix((np.ones(len(E)), (E[:, 0], E[:, 1])), shape=(n, n))
    A = ((A + A.T) > 0).astype(np.int8).tocsr()

    out, reached, cur = {}, sp.identity(n, dtype=np.int8, format="csr"), \
        sp.identity(n, dtype=np.int8, format="csr")
    reached = reached.astype(bool)
    kmax = max(rings)
    for k in range(1, kmax + 1):
        cur = ((cur @ A) > 0)
        ring = (cur.astype(bool) > reached.astype(bool)).astype(np.float32)   # newly reached
        reached = (reached + cur) > 0
        if k in rings:
            R = ring.tocsr()[cols][:, cols]
            R.setdiag(0); R.eliminate_zeros()
            out[k] = R.astype(np.float32)
    return out


def moran_maps(FC, W, n_seeds=2000, seed=0):
    """Moran's I of each FC row under weights W. -> (mean, sd, per-seed array).

    I = (n / sum(W)) * (z' W z) / (z' z) for each centred map z."""
    n = FC.shape[0]
    idx = (np.arange(n) if n_seeds is None or n_seeds >= n
           else np.random.default_rng(seed).choice(n, n_seeds, replace=False))
    X = np.asarray(FC[idx], np.float32)
    X = X - X.mean(1, keepdims=True)
    num = (X * (X @ W)).sum(1)             # z' W z per map
    den = (X * X).sum(1)
    I = (n / W.sum()) * num / np.maximum(den, 1e-30)
    return float(I.mean()), float(I.std()), I


class MoranMatch:
    """Empirical Moran's I per ring, plus the same quantity for a candidate run.

    Only a fixed subset of seed maps is used (300 by default). A model FC row is
    Z[i] @ Z.T / T, so scoring 300 rows costs a (300 x V) matmul rather than the full
    V x V matrix - the whole penalty adds well under a second to an evaluation. The seed
    subset is fixed, so empirical and model I are always computed on the same maps.

    gap(Z) is the mean absolute difference in I across rings; attach it to an FCTarget
    with FCTarget.attach_moran(matcher, lam)."""

    def __init__(self, cortex, target, rings=DEFAULT_RINGS, n_seeds=300, seed=0):
        self.rings = tuple(rings)
        self.W = ring_weights(cortex, target.cols, self.rings)
        self.n = target.nV
        self.idx = np.random.default_rng(seed).choice(self.n, min(n_seeds, self.n),
                                                      replace=False)
        self.scale = {k: self.n / self.W[k].sum() for k in self.rings}
        self.emp = self._I(np.asarray(target.target_fc()[self.idx], np.float32))

    def _I(self, X):
        """Moran's I per ring for a stack of maps (rows), averaged over the maps."""
        X = X - X.mean(1, keepdims=True)
        den = np.maximum((X * X).sum(1), 1e-30)
        return np.array([float((self.scale[k] * (X * (X @ self.W[k])).sum(1)
                                / den).mean()) for k in self.rings])

    def model_I(self, Z):
        T = Z.shape[1]
        return self._I((Z[self.idx] @ Z.T) / T)

    def gap(self, Z):
        return float(np.abs(self.model_I(Z) - self.emp).mean())


def model_fc(cortex, target, frames_path):
    return target.model_fc(np.load(frames_path))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--space", default="fsaverage5")
    ap.add_argument("--fc", default=None, help="empirical FC (default: newest)")
    ap.add_argument("--frames", nargs="*", default=None,
                    help="model frame files (default: every results/frames*.npy)")
    ap.add_argument("--rings", nargs="*", type=int, default=list(DEFAULT_RINGS))
    ap.add_argument("--n-seeds", type=int, default=2000, dest="n_seeds")
    ap.add_argument("--burn", type=int, default=50)
    a = ap.parse_args()

    cortex = load_cortex(a.space, verbose=False)
    target = FCTarget(cortex, fc_path=a.fc, burn=a.burn, verbose=True)
    W = ring_weights(cortex, target.cols, a.rings)
    print(f"  ring sizes (mean neighbours per vertex): " +
          ", ".join(f"{k}:{W[k].sum()/W[k].shape[0]:.1f}" for k in a.rings))

    frames = a.frames if a.frames is not None else sorted(glob.glob("results/frames*.npy"))
    rows = [(os.path.basename(target.fc_path).split("_")[0] + " (empirical)",
             target.target_fc())]
    for f in frames:
        rows.append((os.path.basename(f).replace("frames_", "").replace(".npy", ""),
                     model_fc(cortex, target, f)))

    head = "  " + f"{'':34s}" + "".join(f"  ring {k:<2d}" for k in a.rings)
    print("\nMoran's I of FC maps (mean over "
          f"{min(a.n_seeds, rows[0][1].shape[0])} seed maps)\n{head}")
    for name, FC in rows:
        vals = [moran_maps(FC, W[k], a.n_seeds)[0] for k in a.rings]
        print(f"  {name:34s}" + "".join(f"  {v:+7.3f}" for v in vals))


if __name__ == "__main__":
    main()
