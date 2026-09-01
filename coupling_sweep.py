"""Does structural connectivity help, and does a lag help more? The cheap screen.

The standing verdict - "coupling is monotone worse in lam" - was taken at a fixed 150
solver iterations, before the centring fix, before --select-iters and before whitening.
Adding coupling changes H, which changes the conditioning, which moves the optimal
stopping point; comparing configurations at a matched iteration count therefore measures
convergence, which is exactly the confound the coverage table was corrected for. So that
verdict does not count and this starts again.

Two screens, and the ORDER matters because only one of them is safe.

REACHABLE SPAN, which is the primary one. C = sum_f w_f 2 Re(H S H^H) forces range(C)
inside the real span of H's columns, so || P Ct P || / ||Ct|| over its top m directions
bounds what ANY input to this medium could achieve. It is a property of H alone - no
solve, no iteration count, no stopping rule - so it cannot be confounded the way the
objective can. If coupling does not enlarge the span it cannot help, whatever else moves.

SOLVE OBJECTIVE, reported second and read with suspicion. At a matched iteration count it
is confounded; it is here because a large move in it is still informative, not because a
small one means anything.

Every configuration is also run against a degree- and distance-matched SURROGATE
connectome. Without that, "structural connectivity helps" cannot be told apart from "any
long-range redistribution helps", and the coverage result already said this medium is
short of transport - so that alternative is live rather than pedantic.

  python coupling_sweep.py --workers 8
  python coupling_sweep.py --lams 0,0.003,0.01 --lags 0,32 --workers 8
"""
import argparse, time
import numpy as np

from mesh_cache import load_cortex
import fc_score, xspec, bo_step, subparcels, connectome
from best_fit import BEST_X


def span_ceiling(H, w, Tgt, dims):
    """|| P Ct P || / ||Ct|| over the top m directions of the reachable span, plus the
    span's participation rank. No solve involved, so no stopping rule to confound it."""
    A = np.concatenate([np.sqrt(w[f]) * H[f] for f in range(len(w))], axis=1)
    U, sv, _ = np.linalg.svd(np.concatenate([A.real, A.imag], axis=1),
                             full_matrices=False)
    e = np.asarray(sv, float) ** 2
    prank = float(e.sum() ** 2 / max((e ** 2).sum(), 1e-300))
    nrm = np.linalg.norm(Tgt)
    out = []
    for m in dims:
        m = min(m, U.shape[1])
        Pm = U[:, :m]
        out.append(float(np.linalg.norm(Pm @ (Pm.T @ Tgt @ Pm) @ Pm.T) / nrm))
    return out, prank


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lams", default="0,0.001,0.003,0.01,0.03,0.1")
    ap.add_argument("--lags", default="0,8,32,128")
    ap.add_argument("--surrogate", type=int, default=0, metavar="SEED",
                    help="also run a matched surrogate at each lam (-1 = skip)")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--damp", type=float, default=2e-4)
    ap.add_argument("--window", type=int, default=560)
    ap.add_argument("--whiten", type=float, default=1e-3)
    ap.add_argument("--split", type=int, default=50)
    ap.add_argument("--nvert", type=int, default=1000)
    ap.add_argument("--nfreq", type=int, default=192)
    ap.add_argument("--pad", type=int, default=1120)
    ap.add_argument("--keep", type=float, default=0.15)
    ap.add_argument("--min-mm", type=float, default=60.0, dest="min_mm")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    lams = [float(v) for v in a.lams.split(",") if v.strip()]
    lags = [int(v) for v in a.lags.split(",") if v.strip()]
    DIMS = (100, 200, 400)

    c = load_cortex("fsaverage5", verbose=False)
    t = fc_score.default_target(c, verbose=False)
    labels, tags = subparcels.split_parcels(c, subparcels.SENSORY, a.split, verbose=False)
    P = subparcels.taper_profiles(c, labels, len(tags))
    x = BEST_X.copy(); x[0] = np.log10(a.damp)
    p, save, _ = bo_step.unpack(x, c)
    sub = xspec.medoid_subset(t, a.nvert)
    n = len(sub); iu = np.triu_indices(n, 1)
    raw = np.asarray(t.target_fc()[np.ix_(sub, sub)], np.float64)
    raw = raw - raw.mean(0, keepdims=True) - raw.mean(1, keepdims=True) + raw.mean()
    Tgt = xspec.normal_scores(raw, iu)
    Tgt[np.eye(n, dtype=bool)] = 0.0

    D180 = connectome.parcel_distances(c, verbose=False)
    W0 = connectome.load_enigma(c, verbose=False)
    Wr = connectome.residual_W(W0, D180, a.keep, a.min_mm, verbose=True)
    Wsur = (None if a.surrogate < 0 else
            connectome.surrogate_W(Wr, D180, seed=a.surrogate, verbose=True))

    decay = 1.0 / (10 ** x[0] * save)
    print(f"\n  medium: damping {a.damp:g} per step (decay {decay:.0f} frames), "
          f"window {a.window} frames, {len(P)} pieces, whiten {a.whiten:g}")
    print(f"  screening {len(lams)} lam x {len(lags)} lag"
          + ("" if Wsur is None else " x {real, surrogate}"))

    rows = []
    combos = [(lam, lag, which)
              for lam in lams for lag in lags
              for which in (("real",) if (lam == 0 or Wsur is None) else
                            ("real", "surrogate"))]
    # lam=0 is the same system whatever the lag, so run it once
    combos = [cb for cb in combos if not (cb[0] == 0 and cb[1] != lags[0])]
    print(f"  {len(combos)} configurations\n")
    print(f"  {'lam':>7s} {'lag':>5s} {'W':>10s} {'span100':>8s} {'span200':>8s} "
          f"{'span400':>8s} {'spanrank':>9s} {'solve':>8s}   time")
    for lam, lag, which in combos:
        t0 = time.time()
        cpl = None
        if lam > 0:
            need = lag / save + 3.0 * decay
            if a.window <= need:
                print(f"  {lam:>7g} {lag:>5d} {which:>10s}   SKIPPED: window {a.window} "
                      f"< {need:.0f} frames needed for this lag and decay")
                continue
            cpl = connectome.CouplingOperator(
                c, Wr if which == "real" else Wsur, lam, lag)
        resp = xspec.impulse_responses(c, list(range(len(P))), p, a.window * save, save,
                                       profiles=P, verbose=False, coupling=cpl,
                                       workers=a.workers)
        R = np.pad(resp, ((0, 0), (0, max(0, a.pad - resp.shape[1])), (0, 0)))
        H, w, idx = xspec.transfer(R, t.cols[sub], a.nfreq)
        del resp, R
        sp, prank = span_ceiling(H, w, Tgt, DIMS)
        Hs = H
        if a.whiten > 0:
            Hs, _ = xspec.whiten(H, a.whiten)
        S, C = xspec.solve(Hs, w, Tgt, iters=a.iters, verbose=False)
        Cm = C.copy(); Cm[np.eye(n, dtype=bool)] = 0.0
        Tn = Tgt / np.linalg.norm(Tgt)
        obj = float((Cm * Tn).sum() / np.linalg.norm(Cm))
        rows.append((lam, lag, which, sp, prank, obj))
        print(f"  {lam:>7g} {lag:>5d} {which:>10s} {sp[0]:>8.4f} {sp[1]:>8.4f} "
              f"{sp[2]:>8.4f} {prank:>9.1f} {obj:>8.4f}   {time.time()-t0:.0f}s",
              flush=True)
        del H, Hs, S, C

    base = [r for r in rows if r[0] == 0]
    if base and rows:
        b = base[0]
        print(f"\n  --- against the lam=0 baseline (span200 {b[3][1]:.4f}, "
              f"solve {b[5]:.4f}) ---")
        real = [r for r in rows if r[2] == "real" and r[0] > 0]
        sur = [r for r in rows if r[2] == "surrogate"]
        if real:
            best = max(real, key=lambda r: r[3][1])
            print(f"  best real span200: lam {best[0]:g} lag {best[1]} -> "
                  f"{best[3][1]:+.4f} ({best[3][1]-b[3][1]:+.4f} vs baseline)")
        if sur:
            bs = max(sur, key=lambda r: r[3][1])
            print(f"  best surrogate:    lam {bs[0]:g} lag {bs[1]} -> "
                  f"{bs[3][1]:+.4f} ({bs[3][1]-b[3][1]:+.4f} vs baseline)")
            print(f"  real must beat the SURROGATE, not the baseline - and the surrogate "
                  f"shares ~46% of its edges with the real connectome, so this comparison "
                  f"is conservative in the direction of finding no difference.")
        print(f"\n  Read span200 first: it bounds what any input could do and no stopping "
              f"rule enters it. The solve column is at a matched {a.iters} iterations and "
              f"is confounded by convergence; only a large move in it means anything.")


if __name__ == "__main__":
    main()
