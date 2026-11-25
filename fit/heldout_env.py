"""Envelope solutions scored in CLOSED FORM on vertices none of them was solved on.

A solve can fit its own vertices and still fall behind on the sheet for two separate reasons:
it has learned those vertices rather than the cortex (transfer), or its expected FC is right
but a finite simulated draw does not deliver it (realisation). The closed form - Wick's
E = 2 sum_tau rho(tau) C(tau)^2, exactly what corr_solve optimises - on a fresh vertex set
tests the first with no simulation and no sampling noise. Falling behind here means transfer;
holding up here but not in simulation means realisation.

Each solution is scored on the same held-out vertices, unsmoothed and smoothed at each
requested FWHM (truncated kernels as corr_solve uses them, so smoothing needs the covariance
at every vertex the kernels touch; H is built once at that union).

FIRST RUN, 2026-10-09, 300 held-out vertices (seed 123), spearman vs the empirical FC, and the
sheet score from simulation (577 s, 4 draws, 6 mm) beside it:

                                         closed form, held out           simulated, sheet
                                        none     3 mm     6 mm  spread    6 mm
    env400_lam0_it400 (cold, unsmoothed) +0.552  +0.589  +0.628  0.68x   +0.453
    env_sm3_n300_cold_it50 (3 mm, 50 it) +0.494  +0.521  +0.558  0.58x   +0.346
    env_sm6_n100_a (6 mm, warm, 20 it)   +0.555  +0.591  +0.626  0.53x   +0.347

The 6 mm warm solution transfers as well as the original in expectation and loses 0.11 only in
simulation; the cold 3 mm one is 0.07 behind in expectation and 0.11 in simulation. The
smoothed solutions carry less dynamic range (spread), which a finite draw's noise attenuates
more.

  python fit/heldout_env.py corrfit_env400_lam0_it400 corrfit_env_sm3_n300_cold_it50
"""
import _path  # noqa: F401
import os, time, argparse
import numpy as np

from paths import RESULTS
import envelope as env
import lagged as _lg
import corrfit as cf
import surface_smooth as ss
from interp_gap import build_H, medium_context
from corr_solve import empirical_correlation, old_objective


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("solutions", nargs="+", help="corrfit_<...> npz names, same medium")
    ap.add_argument("--src", default="env_grclip400", help="the envfit they share")
    ap.add_argument("--nheld", type=int, default=300)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--fwhm", default="3,6", help="smoothing widths to score at, mm")
    ap.add_argument("--mass", type=float, default=0.8)
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    z = np.load(os.path.join(RESULTS, f"envfit_{a.src}.npz"), allow_pickle=True)
    _, c, t, _ = medium_context(z)
    cols = np.asarray(t.cols)
    sols = {n: np.load(os.path.join(RESULTS, n + ".npz"), allow_pickle=True) for n in a.solutions}
    used = set(np.asarray(z["sub"]).tolist())
    for n, q in sols.items():
        if not np.array_equal(q["idx"], z["idx"]):
            raise SystemExit(f"  {n} was solved on different bins")
        if "smooth_fwhm" in q.files and float(q["smooth_fwhm"]) > 0:
            used |= set(np.sort(np.random.default_rng(0).choice(
                len(cols), len(q["sub"]), replace=False)).tolist())
        used |= set(np.asarray(q["sub"]).tolist())
    pool = np.setdiff1d(np.arange(len(cols)), np.fromiter(used, int))
    held = np.sort(np.random.default_rng(a.seed).choice(pool, a.nheld, replace=False))
    print(f"  {a.nheld} held-out vertices, none in any solution's solve set "
          f"({len(used)} excluded)", flush=True)

    widths = [float(v) for v in a.fwhm.split(",") if v.strip()]
    Kr = {}
    for f in widths:
        Kt, nom = ss.truncated_matrix(c, f, a.mass)
        Kr[f] = Kt[cols[held]]
        print(f"  {f:g} mm effective (gaussian {nom:.2f}, top {a.mass:.0%}): "
              f"{np.diff(Kr[f].indptr).mean():.1f} vertices per row", flush=True)
    big = np.unique(np.concatenate([cols[held]] + [k.indices for k in Kr.values()]))
    pos = {v: i for i, v in enumerate(big)}
    hp = np.array([pos[v] for v in cols[held]])
    Ks = {f: k[:, big].toarray() for f, k in Kr.items()}

    Hall, respf, kern, c, t, g = build_H(a.src, z, True, workers=a.workers, keep=big)
    idx = g["idx"]
    H = np.ascontiguousarray(Hall[idx]); del Hall
    lags, rho = env.filter_lags(respf, g["pad"], tol=1e-3)
    ph = _lg.phases(idx, g["pad"], lags)
    w = np.asarray(z["w"], float)
    T = empirical_correlation(t, held)
    iu = np.triu_indices(a.nheld, 1)

    def metrics(E):
        R = cf.dcentre(cf.correlation_form(E)[0])
        pear = float(np.corrcoef(R[iu], T[iu])[0, 1])
        _, spear = old_objective(E, t, held, False)
        return pear, spear, R[iu].std() / T[iu].std()

    print(f"\n  {'solution':34s} {'smoothing':>10s}  {'pearson(R,T)':>12s}  {'spearman':>9s}  "
          f"{'spread':>7s}")
    for n, q in sols.items():
        t0 = time.time()
        E = env.env_cov(H, w, q["S"], ph, rho, a.chunk)
        rows = [("none", E[np.ix_(hp, hp)])] + [(f"{f:g} mm", Ks[f] @ E @ Ks[f].T)
                                                 for f in widths]
        for lab, Em in rows:
            p, s, sp = metrics(Em)
            print(f"  {n:34s} {lab:>10s}  {p:+12.4f}  {s:+9.4f}  {sp:6.2f}x", flush=True)
        print(f"  {'':34s} [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
