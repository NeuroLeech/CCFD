"""Realise and score the zero-lag and lagged solutions, on the simulated field.

The solve-stage table says the lagged objective flips the 5.2 s lead-lag correlation from
-0.107 to +0.177 for -0.051 of zero-lag fit. Those are predicted covariances. This
simulates both, and measures on the field that comes out:

  FC score (Pearson and Spearman), against everything prior;
  the REALISED lagged structure, i.e. whether the corrected propagation survives
  realisation rather than existing only in C(S).

  python lagfit.py --taus 3,8 --wa 3.0
"""
import os, argparse, time
import numpy as np

from mesh_cache import load_cortex
from paths import RESULTS
import fc_score, xspec, bo_step, subparcels, timescale, units, lagged

TR = 0.645
LAG_TR = [0, 3, 5, 8, 12]


def realised_lag(Z, taus_frames):
    """Antisymmetric part of the simulated field's lagged covariance, centred."""
    T = Z.shape[1]
    out = []
    for L in taus_frames:
        A = Z[:, :T - L] if L else Z
        B = Z[:, L:] if L else Z
        P = lagged.double_centre_ns((A @ B.T) / B.shape[1])
        out.append(0.5 * (P - P.T))
    return np.stack(out)


def cor(a, b):
    a = np.asarray(a).ravel().copy(); b = np.asarray(b).ravel().copy()
    a -= a.mean(); b -= b.mean()
    return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--taus", default="3,8", help="lags in TRs")
    ap.add_argument("--wa", default="3.0",
                    help="comma-separated weights on the antisymmetric block; each is "
                         "solved and realised separately")
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--draws", type=int, default=2)
    ap.add_argument("--spread-mm-s", type=float, default=3.0, dest="spread")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--save-frames", default="", dest="save_frames",
                    help="tag prefix; writes results/frames_<tag>_{zerolag,lagged}.npy "
                         "and the matching drive, for render_frames.py")
    a = ap.parse_args()
    taus = [int(v) for v in a.taus.split(",")]
    was = [float(v) for v in str(a.wa).split(",") if v.strip()]

    c = load_cortex("fsaverage5", verbose=False)
    tP = fc_score.default_target(c, metric="pearson", verbose=False)
    tS = fc_score.default_target(c, metric="spearman", verbose=False)
    lab, tg = subparcels.split_parcels(c, subparcels.SENSORY, 50, verbose=False)
    P = subparcels.taper_profiles(c, lab, len(tg))
    cl = timescale.plan(4, decay_s=timescale.BOLD_TAU_S, spread_mm_s=a.spread,
                        verbose=False)
    x = np.array(__import__("best_fit").BEST_X, copy=True)
    x[3] = np.log10(cl["save"]); x[0] = np.log10(cl["damp"])
    p, save, _ = bo_step.unpack(x, c)
    sub = xspec.medoid_subset(tP, 1000); n = len(sub)
    ov = cl["oversample"]
    frames = timescale.frames_for(577.0, cl["frame_s"])
    print(f"  clock TR/{ov}, save {save}, damp {cl['damp']:.4g}, "
          f"{frames} frames ({577:.0f}s); lags {taus} TR = "
          f"{[t*ov for t in taus]} model frames")

    resp = xspec.impulse_responses(c, list(range(len(P))), p, 224 * save, save,
                                   profiles=P, verbose=False, workers=a.workers)
    R = np.pad(resp, ((0, 0), (0, max(0, 4096 - resp.shape[1])), (0, 0))); del resp
    kern = units.smoothing_kernel(
        timescale.bold_fwhm_frames(cl["frame_s"], verbose=False), verbose=False)
    H, w, idx = xspec.transfer(R, tP.cols[sub], 192, kernel=kern)
    ref = R.shape[1]; del R

    Pemp = np.load("data/cache/lagged_0-3-5-8-12_1000_gsr_all.npy")
    T0 = Pemp[0]
    A_t = np.stack([0.5 * (Pemp[LAG_TR.index(t)] - Pemp[LAG_TR.index(t)].T)
                    for t in taus])
    ph0 = lagged.phases(idx, ref, [0] + [t * ov for t in taus])

    print("\n  solving...")
    t0 = time.time()
    S_zl, _ = xspec.solve(H, w, T0, iters=a.iters, verbose=False)
    print(f"    zero-lag only  [{time.time()-t0:.0f}s]", flush=True)
    sols = [("zero-lag", S_zl, "zerolag")]
    for wv in was:
        t0 = time.time()
        Sw, _ = xspec.solve_lagged(H, w, T0, A_t, ph0, iters=a.iters, verbose=False,
                                   wa=wv)
        sols.append((f"lagged wa={wv:g}", Sw, f"lag{wv:g}".replace(".", "p")))
        print(f"    lagged wa={wv:g} [{time.time()-t0:.0f}s]", flush=True)

    res = {}
    print(f"\n  {'solution':<14s} {'FC pearson':>13s} {'FC spearman':>13s} "
          + " ".join(f"{'lag '+str(t)+'TR':>13s}" for t in taus) + f" {'rank':>6s}")
    for nm, S, slug in sols:
        pear, spear, lags, rks = [], [], [], []
        for d in range(a.draws):
            A = xspec.realise(S, idx, frames, ref_frames=ref, seed=5000 + d)
            r = xspec.score_realisation(c, tP, p, A, save=save, profiles=P, kernel=kern)
            F = r["frames"]
            Zp, _ = tP.model_z(F)
            pear.append(float(tP._prep(tP.model_edges(Z=Zp)[0]) @ tP.y))
            Zs, _ = tS.model_z(F)
            spear.append(float(tS._prep(tS.model_edges(Z=Zs)[0]) @ tS.y))
            rks.append(r["rank"])
            if d == 0 and a.save_frames:
                np.save(os.path.join(RESULTS,
                                     f"frames_{a.save_frames}_{slug}.npy"), F)
                np.save(os.path.join(RESULTS,
                                     f"drive_{a.save_frames}_{slug}.npy"),
                        r["drive"].Aser)
                print(f"    wrote frames_{a.save_frames}_{slug}.npy", flush=True)
            Zb = Zp[sub].astype(np.float64)
            Zb -= Zb.mean(1, keepdims=True)
            Zb /= np.maximum(Zb.std(1, keepdims=True), 1e-12)
            Am = realised_lag(Zb, [t * ov for t in taus])
            lags.append([cor(Am[k], A_t[k]) for k in range(len(taus))])
        lags = np.array(lags)
        res[nm] = dict(pear=np.array(pear), spear=np.array(spear), lags=lags,
                       rank=np.array(rks))
        print(f"  {nm:<14s} {np.mean(pear):>+7.4f}+-{np.std(pear):<5.4f} "
              f"{np.mean(spear):>+7.4f}+-{np.std(spear):<5.4f} "
              + " ".join(f"{lags[:,k].mean():>+7.4f}+-{lags[:,k].std():<5.4f}"
                         for k in range(lags.shape[1]))
              + f" {np.mean(rks):>6.1f}", flush=True)
    print(f"\n  lag columns are the REALISED antisymmetric covariance against the "
          f"empirical one")
    if len(res) > 1:
        base = list(res)[0]
        r1 = res[base]; nd = len(r1["pear"])
        print(f"\n  difference from '{base}', {nd} draws each; 'ns' = inside 2 sd")
        print(f"  {'member':<14s} {'FC pearson':>18s} {'FC spearman':>18s} "
              + " ".join(f"{'lag '+str(t)+'TR':>18s}" for t in taus))
        for nm2 in list(res)[1:]:
            r2 = res[nm2]
            cells = []
            for k in ("pear", "spear"):
                d = r2[k].mean() - r1[k].mean()
                sd = (np.sqrt(r1[k].var(ddof=1)/nd + r2[k].var(ddof=1)/nd)
                      if nd > 1 else np.nan)
                cells.append(f"{d:+.4f}+-{sd:.4f}" + ("ns" if nd > 1 and abs(d) < 2*sd
                                                      else "  "))
            for j in range(len(taus)):
                d = r2["lags"][:, j].mean() - r1["lags"][:, j].mean()
                sd = (np.sqrt(r1["lags"][:, j].var(ddof=1)/nd
                              + r2["lags"][:, j].var(ddof=1)/nd) if nd > 1 else np.nan)
                cells.append(f"{d:+.4f}+-{sd:.4f}" + ("ns" if nd > 1 and abs(d) < 2*sd
                                                      else "  "))
            print(f"  {nm2:<14s} " + " ".join(f"{c:>18s}" for c in cells))


if __name__ == "__main__":
    main()
