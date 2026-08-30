"""Do differently-graded media produce DIFFERENT FCs, and do their errors complement?

A switching medium gives C = sum_r occ_r C_r with occ >= 0 summing to one, so a
non-negative combination of per-regime FCs is exactly the model. That means the payoff can
be measured before any realisation is run: solve each regime separately, then ask how much
a non-negative mixture of their FCs beats the best single one.

The interesting quantity is not whether the regimes differ - they will - but whether they
differ WHERE IT MATTERS. Two media whose FCs are 0.99 correlated with each other cannot
help each other however good each is. What is needed is decorrelated RESIDUALS: regime A
wrong about edges that regime B gets right.

Regimes differ only in how speed and damping are graded across the sheet, by the myelin,
thickness and sulcal maps. Both are renormalised so the OVERALL medium is unchanged: peak
speed matched (which also pins dt and every per-step quantity), and area-weighted mean
damping matched. So a regime is a redistribution of the same medium, not more or less of
it.

  python regime_fcs.py --precompute 3      # cache one regime's impulse responses
  python regime_fcs.py                     # solve them all and compare
"""
import argparse
import numpy as np

from mesh_cache import load_cortex
import fc_score, xspec, bo_step, subparcels, fluid as fl
from best_fit import BEST_X

# (name, speed-coefficient offset, damping-coefficient offset) over (myelin, thick, sulc)
REGIMES = [
    ("base",        [0.0, 0.0, 0.0],   [0.0, 0.0, 0.0]),
    ("speed+myel",  [0.45, 0.0, 0.0],  [0.0, 0.0, 0.0]),
    ("speed-myel",  [-0.15, 0.0, 0.0], [0.0, 0.0, 0.0]),
    ("speed+sulc",  [0.0, 0.0, 0.44],  [0.0, 0.0, 0.0]),
    ("speed+thick", [0.0, 0.45, 0.0],  [0.0, 0.0, 0.0]),
    ("damp+myel",   [0.0, 0.0, 0.0],   [0.45, 0.0, 0.0]),
    ("damp-myel",   [0.0, 0.0, 0.0],   [-0.42, 0.0, 0.0]),
]


def medium(cortex, da, db, base_p, verbose=False):
    """Base medium with the map grading changed, overall speed and damping preserved."""
    p = dict(base_p)
    p["a"] = np.asarray(base_p["a"], float) + np.asarray(da, float)
    p["b"] = np.asarray(base_p["b"], float) + np.asarray(db, float)
    area = np.asarray(cortex.A, float)
    c0, s0 = fl.fields(cortex, base_p)
    c1, s1 = fl.fields(cortex, p)
    p["c0"] *= float(c0.max()) / float(c1.max())            # peak speed -> dt unchanged
    p["sig0"] *= float(area @ s0) / float(area @ s1)        # mean damping unchanged
    if verbose:
        c2, s2 = fl.fields(cortex, p)
        print(f"    speed {c2.min():.3f}-{c2.max():.3f} (base {c0.min():.3f}-{c0.max():.3f}), "
              f"mean damping {float(area @ s2)/area.sum():.3e} "
              f"(base {float(area @ s0)/area.sum():.3e})")
    return p


def setup():
    c = load_cortex("fsaverage5", verbose=False)
    t = fc_score.default_target(c, verbose=False)
    labels, tg = subparcels.split_parcels(c, subparcels.SENSORY, 50, verbose=False)
    P = subparcels.taper_profiles(c, labels, len(tg))
    base_p, save, _ = bo_step.unpack(BEST_X, c)
    return c, t, P, base_p, save


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--precompute", type=int, default=None)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--window", type=int, default=280)
    ap.add_argument("--nvert", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=0)
    a = ap.parse_args()

    c, t, P, base_p, save = setup()
    if a.precompute is not None:
        nm, da, db = REGIMES[a.precompute]
        p = medium(c, da, db, base_p, verbose=True)
        print(f"  {nm}: computing impulse responses")
        xspec.impulse_responses(c, list(range(len(P))), p, a.window * save, save,
                                profiles=P, verbose=True, workers=a.workers)
        print(f"  {nm}: cached")
        return

    sub = xspec.medoid_subset(t, a.nvert)
    iu = np.triu_indices(len(sub), 1)
    raw = np.asarray(t.target_fc()[np.ix_(sub, sub)], np.float64)
    raw = raw - raw.mean(0, keepdims=True) - raw.mean(1, keepdims=True) + raw.mean()
    Tgt = xspec.normal_scores(raw)
    y = Tgt[iu]; y = (y - y.mean()) / y.std()

    Cs, names = [], []
    for nm, da, db in REGIMES:
        p = medium(c, da, db, base_p)
        resp = xspec.impulse_responses(c, list(range(len(P))), p, a.window * save, save,
                                       profiles=P, verbose=False)
        R = np.pad(resp, ((0, 0), (0, max(0, 1120 - resp.shape[1])), (0, 0)))
        H, w, idx = xspec.transfer(R, t.cols[sub], 192)
        S, C = xspec.solve(H, w, Tgt, iters=a.iters, verbose=False)
        v = C[iu]; v = (v - v.mean()) / v.std()
        Cs.append(v); names.append(nm)
        print(f"  {nm:12s} corr with target {float(v @ y)/len(y):+.4f}")
    X = np.stack(Cs)
    r_t = X @ y / len(y)

    print(f"\n  correlation BETWEEN regime FCs (upper triangle, {len(iu[0])} edges)")
    print(f"  {'':12s}" + "".join(f"{n[:9]:>10s}" for n in names))
    G = X @ X.T / X.shape[1]
    for i, n in enumerate(names):
        print(f"  {n:12s}" + "".join(f"{G[i, j]:10.3f}" for j in range(len(names))))

    Rres = np.stack([y - r_t[i] * X[i] for i in range(len(names))])
    Rres /= Rres.std(1, keepdims=True)
    Gr = Rres @ Rres.T / Rres.shape[1]
    print(f"\n  correlation between RESIDUALS (target minus each regime's best fit)")
    print(f"  {'':12s}" + "".join(f"{n[:9]:>10s}" for n in names))
    for i, n in enumerate(names):
        print(f"  {n:12s}" + "".join(f"{Gr[i, j]:10.3f}" for j in range(len(names))))

    from scipy.optimize import nnls
    coef, _ = nnls(X.T, y)
    fit = X.T @ coef
    r_mix = float(np.corrcoef(fit, y)[0, 1])
    best = float(r_t.max())
    print(f"\n  best single regime      {best:+.4f}  ({names[int(np.argmax(r_t))]})")
    print(f"  non-negative mixture    {r_mix:+.4f}   (+{r_mix-best:.4f})")
    print(f"  weights: " + ", ".join(f"{n}={w_:.3f}" for n, w_ in zip(names, coef)
                                     if w_ > 1e-6))
    ur, _, _, _ = np.linalg.lstsq(X.T, y, rcond=None)
    print(f"  unconstrained (not physical, an upper bound) "
          f"{float(np.corrcoef(X.T @ ur, y)[0,1]):+.4f}")


if __name__ == "__main__":
    main()
