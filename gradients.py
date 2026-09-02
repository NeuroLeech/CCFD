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

  python gradients.py --tags sc2_sen,sc2_sub47
"""
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
