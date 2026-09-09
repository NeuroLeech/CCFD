"""FC gradients (Margulies 2016), empirical against simulated.

The score is one correlation over edges; it says nothing about whether the model
reproduces the ORGANISATION of the connectome. The gradient decomposition is the standard
low-dimensional description of that: threshold each row of the FC, take cosine similarity
between rows, and diffusion-map embed. G1 is the unimodal-to-transmodal axis, G2 the
visual-to-somatomotor one.

This computes them the same way for the empirical and the simulated FC, on the same
vertices, and reports how well each gradient corresponds. Sign and order are not
meaningful in an eigendecomposition, so gradients are matched by absolute correlation and
the sign is aligned before anything is plotted.

  python analysis/gradients.py --tags sc2_sen,sc2_sub47
"""
import _path  # noqa: F401  - puts the sibling code folders on sys.path
import os, argparse
import numpy as np

from mesh_cache import load_cortex
from paths import RESULTS
import fc_score


def gradients(FC, n_comp=6, thresh=0.90, alpha=0.5, sparse=True):
    """Diffusion-map embedding of an FC matrix. -> (vertices, n_comp), eigenvalues.

    Row-wise thresholding keeps the top (1-thresh) of each row and zeroes the rest, which
    is what makes the cosine-similarity affinity sparse and positive; that is the recipe
    in Margulies 2016 rather than an arbitrary choice."""
    R = np.array(FC, np.float64, copy=True)
    n = R.shape[0]
    cut = np.quantile(R, thresh, axis=1, keepdims=True)
    R[R < cut] = 0.0
    R[R < 0] = 0.0
    nrm = np.linalg.norm(R, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    A = (R / nrm) @ (R / nrm).T                 # cosine similarity between rows
    np.clip(A, 0, None, out=A)
    A = 0.5 * (A + A.T)

    d = A.sum(1)
    d[d == 0] = 1e-12
    # anisotropic diffusion: L = D^-a A D^-a, then row-normalise
    Da = d ** (-alpha)
    L = A * Da[:, None] * Da[None, :]
    dl = L.sum(1); dl[dl == 0] = 1e-12
    # symmetric conjugate of the row-stochastic operator, so eigh applies
    s = dl ** -0.5
    M = L * s[:, None] * s[None, :]
    M = 0.5 * (M + M.T)
    if sparse:
        from scipy.sparse.linalg import eigsh
        ev, V = eigsh(M, k=n_comp + 1, which="LA")
    else:
        ev, V = np.linalg.eigh(M)
    o = np.argsort(ev)[::-1]
    ev, V = ev[o], V[:, o]
    V = V * s[:, None]                          # back to the diffusion-map basis
    V /= np.maximum(np.linalg.norm(V, axis=0, keepdims=True), 1e-30)
    return V[:, 1:n_comp + 1], ev[1:n_comp + 1]


def affinity(FC, thresh=0.90, dtype=np.float32, copy=True):
    """Cosine affinity of the row-thresholded FC. -> (n, n), same recipe as `gradients`.

    Keep the top (1-thresh) of each row, zero the rest and any negatives, row-normalise,
    then cosine similarity between rows. The ROW-PERCENTILE threshold is what makes this
    invariant to a per-vertex multiplicative scale on the seed: with R_obs = D R D, row i
    is a_i (R D)_i and the a_i cancels in the cosine, so measurement attenuation of the
    seed vertex drops out. An absolute threshold would put it back.

    Applied to whatever matrix it is handed, so both sides of a comparison must be given
    the same treatment - here that is the double-centred FC on both sides, which departs
    from the Margulies convention of using the raw correlation matrix but is consistent."""
    R = np.array(FC, dtype, copy=copy)
    # np.quantile promotes to float64 and allocates a second full copy; np.partition
    # gives the same cut in the working dtype. On 9310^2 that is ~700 MB not allocated,
    # per scoring process, which matters because draws are scored in a pool.
    k = int(round(float(thresh) * (R.shape[1] - 1)))
    cut = np.partition(R, k, axis=1)[:, k:k + 1]
    R[R < cut] = 0.0
    R[R < 0] = 0.0
    R /= np.maximum(np.linalg.norm(R, axis=1, keepdims=True), 1e-30)
    return R @ R.T


def affinity_rows(FC, thresh=0.90, dtype=np.float32, copy=True):
    """The row-thresholded, row-normalised FC - the factor W with affinity = W W^T.

    Returned instead of the affinity itself because a score only reads the affinity at a
    sampled set of edges, and `W W^T` is a 9310^3 matmul and another 347 MB array to get
    values that `einsum('ij,ij->i', W[i], W[j])` produces directly at 2M edges for a
    fortieth of the work."""
    R = np.array(FC, dtype, copy=copy)
    k = int(round(float(thresh) * (R.shape[1] - 1)))
    cut = np.partition(R, k, axis=1)[:, k:k + 1]
    R[R < cut] = 0.0
    R[R < 0] = 0.0
    R /= np.maximum(np.linalg.norm(R, axis=1, keepdims=True), 1e-30)
    return R


def affinity_edges(W, i, j, budget=512e6):
    """Affinity at the given edge pairs, without ever forming the matrix. -> (n_edges,)

    `W[i[b]]` is a fancy-index COPY of shape (chunk, W.shape[1]), so the chunk has to be
    sized from the row width or the temporary dwarfs everything else: at chunk 250,000 and
    9,310 columns that copy is 9.3 GB, twice over. `budget` is bytes per operand."""
    chunk = max(256, int(budget / (W.shape[1] * W.itemsize)))
    out = np.empty(len(i), np.float32)
    for a in range(0, len(i), chunk):
        b = slice(a, a + chunk)
        out[b] = np.einsum("ij,ij->i", W[i[b]], W[j[b]])
    return out


def affinity_lowrank(FC, k, thresh=0.90, sub=None):
    """Rank-k reconstruction of the cosine affinity of FC, as a target to FIT.

    Same recipe as `gradients` up to the affinity: keep the top (1-thresh) of each row,
    zero the rest and any negatives, row-normalise, cosine similarity. Then take the
    leading k eigenpairs and reconstruct.

    Why the affinity rather than FC itself. Measurement noise attenuates a correlation
    multiplicatively - R_obs = D R D with D = diag(a_i), a_i = sqrt(SNR/(1+SNR)) - and a
    low-rank truncation of R_obs does NOT remove D, since D R D has the same rank as R.
    The cosine does: row i is a_i (R D)_i, so in cos(row_i, row_j) the a_i and a_j cancel
    exactly, and what survives is D acting INSIDE the rows, a reweighting of the summation
    dimension rather than a per-vertex scale. The ROW-PERCENTILE threshold is what buys
    that; an absolute threshold reintroduces the scale it was meant to remove.

    `sub` returns only that sub-block, which is all the solve needs. The full
    reconstruction is never materialised: the top-k eigenpairs come from a LinearOperator
    on A = W W^T, so nothing 9310^2 is formed beyond W itself.

    denoise.py's warning applies and is why this is a knob and not a default: PCA keeps the
    highest-VARIANCE directions, which are not the most reliable ones - global signal and
    motion are high variance. Fit to this, score against the raw FC, and let the two
    numbers say whether the discarded components were noise."""
    from scipy.sparse.linalg import LinearOperator, eigsh
    R = np.array(FC, np.float64, copy=True)
    cut = np.quantile(R, thresh, axis=1, keepdims=True)
    R[R < cut] = 0.0
    R[R < 0] = 0.0
    W = R / np.maximum(np.linalg.norm(R, axis=1, keepdims=True), 1e-30)
    n = W.shape[0]
    op = LinearOperator((n, n), matvec=lambda v: W @ (W.T @ v), dtype=np.float64)
    ev, V = eigsh(op, k=int(k), which="LA")
    order = np.argsort(ev)[::-1]
    ev, V = ev[order], V[:, order]
    Vs = V if sub is None else V[np.asarray(sub, int)]
    return (Vs * ev[None, :]) @ Vs.T

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tags", default="sc2_sen,sc2_sub47")
    ap.add_argument("--labels", default="sensory,subcortical")
    ap.add_argument("--nvert", type=int, default=0, help="0 = all vertices")
    ap.add_argument("--ncomp", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    tags = a.tags.split(","); labs = a.labels.split(",")

    c = load_cortex("fsaverage5", verbose=False)
    # RAW FC, not double-centred. Double-centring removes the global component, which is
    # exactly what makes G1 the dominant unimodal-transmodal axis in Margulies 2016 - with
    # it removed the leading eigenvalues come out nearly equal (24.9% and 24.6%) and G1 is
    # not the canonical gradient. Both sides are treated identically, so the comparison
    # stands either way; this way it is comparable to the published decomposition.
    t = fc_score.FCTarget(c, fc_path=fc_score.raw_fc(c.mesh), centre="none",
                          metric="pearson", verbose=False)
    v = (np.arange(t.nV) if a.nvert <= 0 or a.nvert >= t.nV else
         np.sort(np.random.default_rng(a.seed).choice(t.nV, a.nvert, replace=False)))
    G = np.asarray(t.target_fc()[np.ix_(v, v)], np.float64)
    print(f"  {len(v)} vertices, RAW (un-centred) FC; embedding the empirical")
    Ge, ee = gradients(G, a.ncomp)
    print(f"  empirical eigenvalues: " + ", ".join(f"{x:.4f}" for x in ee))
    var = ee / ee.sum()
    print(f"  variance share: " + ", ".join(f"{x:.1%}" for x in var))

    out = {"empirical": (Ge, ee)}
    for tag, lab in zip(tags, labs):
        F = np.asarray(np.load(os.path.join(RESULTS, f"frames_{tag}.npy"), mmap_mode="r"))
        Z, _ = t.model_z(F); del F
        Zs = Z[v].astype(np.float64)
        Zs -= Zs.mean(1, keepdims=True)
        Zs /= np.maximum(Zs.std(1, keepdims=True), 1e-12)
        M = (Zs @ Zs.T) / Zs.shape[1]        # raw, matching the empirical side
        Gm, em = gradients(M, a.ncomp)
        out[lab] = (Gm, em)
        print(f"\n  {lab}: eigenvalues " + ", ".join(f"{x:.4f}" for x in em))
        # match each empirical gradient to its best model counterpart
        C = np.abs(Ge.T @ Gm) / (np.linalg.norm(Ge, axis=0)[:, None]
                                 * np.linalg.norm(Gm, axis=0)[None, :])
        print(f"    {'empirical':<12s} {'best model':>11s} {'|r|':>7s} "
              f"{'same order?':>12s}")
        for i in range(a.ncomp):
            j = int(np.argmax(C[i]))
            print(f"    G{i+1:<11d} {'G'+str(j+1):>11s} {C[i, j]:>7.3f} "
                  f"{'yes' if j == i else 'NO':>12s}")
        del Z, Zs, M

    # picture: the leading gradients on the surface
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from render_regimes import _proj
    from plot_fc_map import surface_row
    proj = _proj(c.V, c.F)
    ncols = 3
    rows = []
    for nm in ["empirical"] + labs:
        Gx, _ = out[nm]
        for i in range(ncols):
            g = Gx[:, i]
            ref = out["empirical"][0][:, i]
            if float(g @ ref) < 0:
                g = -g                              # align sign to the empirical
            full = np.full(t.nV, np.nan)
            full[v] = g
            rows.append((f"{nm}\nG{i+1}", full))
    fig = plt.figure(figsize=(3.4 * len(proj), 2.3 * len(rows)))
    gs = fig.add_gridspec(len(rows), len(proj), hspace=0.05, wspace=0.02)
    for r, (label, vals) in enumerate(rows):
        vv = np.nan_to_num(vals, nan=0.0)
        s = float(np.percentile(np.abs(vv[np.isfinite(vals)]), 98))
        surface_row(fig, gs, r, proj, vv, c, t.cols, "coolwarm", (-s, s), label)
    p = os.path.join(RESULTS, "gradients.png")
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"\n  wrote {p}")


if __name__ == "__main__":
    main()
