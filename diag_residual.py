"""What is the model missing, and is it reachable?

The regime experiment established that the residual is ONE object: seven differently
graded media all fail on the same edges, pairwise residual correlations 0.80-0.996. So it
is worth characterising once rather than sampling more media.

Two questions, in order of importance.

REACHABLE? C = sum_f H S H^H forces range(C) inside the real span of H's columns. If the
residual lies mostly inside that span, the structure exists in the medium and the solve is
simply not finding it - a conditioning problem, fixable in the solver. If it lies outside,
no input to this medium from these pieces can ever produce it, and the architecture has to
change. ||P R P|| / ||R|| over the span's top m directions decides it.

WHAT SHAPE? The residual is symmetric, so it eigendecomposes. Its rank says whether the
miss is a few large structures or diffuse; its leading eigenvectors say where on the sheet
they live, and how they relate to distance from the drive.

  python diag_residual.py --tag sym_sensory
"""
import argparse, os
import numpy as np

from mesh_cache import load_cortex
from paths import RESULTS
import fc_score, xspec, bo_step, subparcels


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", default="damp2e4", help="a saved run, for its realised FC")
    ap.add_argument("--damp", type=float, default=2e-4)
    ap.add_argument("--window", type=int, default=560)
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--nvert", type=int, default=1000)
    ap.add_argument("--split", type=int, default=50)
    a = ap.parse_args()

    c = load_cortex("fsaverage5", verbose=False)
    t = fc_score.default_target(c, verbose=False)
    labels, tg = subparcels.split_parcels(c, subparcels.SENSORY, a.split, verbose=False)
    P = subparcels.taper_profiles(c, labels, len(tg))
    x = bo_step.np.array(__import__("best_fit").BEST_X, copy=True)
    x[0] = np.log10(a.damp)
    p, save, _ = bo_step.unpack(x, c)
    sub = xspec.medoid_subset(t, a.nvert)
    n = len(sub); iu = np.triu_indices(n, 1)

    resp = xspec.impulse_responses(c, list(range(len(P))), p, a.window * save, save,
                                   profiles=P, verbose=False)
    R_ = np.pad(resp, ((0, 0), (0, max(0, 1120 - resp.shape[1])), (0, 0)))
    H, w, idx = xspec.transfer(R_, t.cols[sub], 192)

    raw = np.asarray(t.target_fc()[np.ix_(sub, sub)], np.float64)
    raw = raw - raw.mean(0, keepdims=True) - raw.mean(1, keepdims=True) + raw.mean()
    Tgt = xspec.normal_scores(raw)
    Tgt[np.eye(n, dtype=bool)] = 0.0
    S, C = xspec.solve(H, w, Tgt, iters=a.iters, verbose=False)

    # residual: the target minus its best scalar multiple of the model
    tv, cv = Tgt[iu], C[iu]
    tv = (tv - tv.mean()) / tv.std(); cv = (cv - cv.mean()) / cv.std()
    beta = float(tv @ cv) / len(tv)
    print(f"  model explains {beta:+.4f} of the target (solve objective)")
    Res = np.zeros((n, n)); Res[iu] = tv - beta * cv
    Res = Res + Res.T
    rv = Res[iu]
    print(f"  residual is {np.linalg.norm(rv)/np.linalg.norm(tv):.3f} of the target norm")

    # --- is it reachable?
    A = np.concatenate([np.sqrt(w[f]) * H[f] for f in range(len(w))], axis=1)
    Ar = np.concatenate([A.real, A.imag], axis=1)
    U, sv, _ = np.linalg.svd(Ar, full_matrices=False)
    Tn = np.linalg.norm(Tgt); Rn = np.linalg.norm(Res)
    print(f"\n  {'span dims':>10s} {'target captured':>16s} {'RESIDUAL captured':>18s}")
    for m in (25, 50, 100, 200, 400, 700, n):
        Pm = U[:, :m]
        pt = Pm @ (Pm.T @ Tgt @ Pm) @ Pm.T
        pr = Pm @ (Pm.T @ Res @ Pm) @ Pm.T
        print(f"  {m:10d} {np.linalg.norm(pt)/Tn:16.3f} {np.linalg.norm(pr)/Rn:18.3f}")

    # --- what shape is it?
    ev, V = np.linalg.eigh(Res)
    o = np.argsort(np.abs(ev))[::-1]; ev, V = ev[o], V[:, o]
    e = ev ** 2
    print(f"\n  residual participation rank {float(e.sum()**2/(e**2).sum()):.1f} of {n}")
    print(f"  leading eigenvalues: " + ", ".join(f"{v:+.1f}" for v in ev[:8]))
    print(f"  top 10 modes hold {e[:10].sum()/e.sum():.1%} of residual energy; "
          f"top 50 hold {e[:50].sum()/e.sum():.1%}")

    # where do the leading modes live?
    import units
    driven = (labels >= 0)[t.cols][sub]
    d = units.vertex_geodesic(c, np.flatnonzero((labels >= 0)))[:, t.cols[sub]].min(0)
    print(f"\n  {'mode':>5s} {'eigval':>8s} {'|loading| driven':>17s} {'undriven':>10s} "
          f"{'corr with distance':>19s}")
    for i in range(6):
        v = np.abs(V[:, i])
        print(f"  {i+1:5d} {ev[i]:+8.1f} {v[driven].mean():17.4f} "
              f"{v[~driven].mean():10.4f} {np.corrcoef(v, d)[0,1]:+19.3f}")


if __name__ == "__main__":
    main()
