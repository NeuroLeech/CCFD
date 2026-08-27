"""Bayesian optimisation of the medium, with the cross-spectrum solve as the inner loop.

The input is no longer searched - for any medium, xspec solves for the best possible
input cross-spectrum convexly. That leaves only the medium's own parameters, a handful of
continuous numbers, which is what Bayesian optimisation is for: tens of evaluations
rather than the thousands a GA needed to do worse.

Each evaluation is honest end to end: impulse responses for this medium, convex solve for
the input, realisation of a drive with that cross-spectrum, simulation, and the same
Spearman edge score every earlier result used. Two draws are averaged, so the objective
carries about +-0.006 of noise, which the GP is told about.

  python bo_medium.py --calls 30
"""
import os, time, pickle, argparse
import multiprocessing as mp
import numpy as np

from paths import RESULTS, CACHE
import xspec

OUT = os.path.join(RESULTS, "bo_medium")
SPACE = [("log10_c0", -0.5, 0.5), ("log10_Ld", 1.0, 2.5), ("log10_sig0", -3.0, -1.0)]


def space(c0_max=0.5):
    """Bounds in log10. c0's ceiling is an argument because the first run pinned it."""
    return [("log10_c0", -0.5, float(c0_max)), ("log10_Ld", 1.0, 2.5),
            ("log10_sig0", -3.0, -1.0)]
_W = {}


def _init(nfreq, nvert, iters, ndraw, nframes, regions=None):
    from mesh_cache import load_cortex
    from fc_score import FCTarget
    if regions:
        xspec.REGIONS = list(regions)
    c = load_cortex("fsaverage5", verbose=False)
    _W.update(cortex=c, target=FCTarget(c, verbose=False), nfreq=nfreq, nvert=nvert,
              iters=iters, ndraw=ndraw, nframes=nframes)


def _impulse(args):
    """One region's impulse response, so the 19 can be spread over the pool."""
    region, p, nsteps, save = args
    from input2 import parcel_tapers
    import fluid as fl
    c = _W["cortex"]
    T, ids = parcel_tapers(c, verbose=False)
    pos = {int(q): i for i, q in enumerate(ids)}
    s, dt, g, H = fl.build(c, p, sponge=True)
    h = T[pos[int(region)]].astype(np.float32).copy()
    ue = np.zeros(s.nE, np.float32)
    fr = [h.copy()]
    for n in range(1, nsteps):
        ue, h = s.step(ue, h, np.float32(dt), g, H)
        if n % save == 0:
            fr.append(h.copy())
    return np.asarray(fr)


def evaluate(pool, x, verbose=True):
    """-> (mean end-to-end similarity, details). x is the medium, in log10 units."""
    p = xspec.medium(10.0 ** x[2], 10.0 ** x[0], 10.0 ** x[1])
    c, target = _W["cortex"], _W["target"]
    t0 = time.time()
    resp = np.asarray(pool.map(_impulse, [(k, p, xspec.NSTEPS, xspec.SAVE)
                                          for k in xspec.REGIONS]))
    rng = np.random.default_rng(0)
    sub = np.sort(rng.choice(target.nV, _W["nvert"], replace=False))
    H, w, idx = xspec.transfer(resp, target.cols[sub], _W["nfreq"])
    Ct = np.asarray(target.target_fc()[np.ix_(sub, sub)], np.float64)
    Ct = Ct - Ct.mean(0, keepdims=True) - Ct.mean(1, keepdims=True) + Ct.mean()
    S, C = xspec.solve(H, w, Ct, iters=_W["iters"], verbose=False)

    sims, gaps, ranks = [], [], []
    for d in range(_W["ndraw"]):
        A = xspec.realise(S, idx, _W["nframes"], seed=d)
        r = xspec.score_realisation(c, target, p, A)
        sims.append(r["sim"]); gaps.append(r["gap"]); ranks.append(r["rank"])
    off = ~np.eye(len(sub), dtype=bool)
    det = dict(solve_corr=float(np.corrcoef(C[off], Ct[off])[0, 1]),
               sim=float(np.mean(sims)), sim_sd=float(np.std(sims)),
               gap=float(np.mean(gaps)), rank=float(np.mean(ranks)),
               secs=time.time() - t0, S=S, idx=idx, x=np.asarray(x))
    if verbose:
        print(f"    c0 {10**x[0]:5.2f}  Ld {10**x[1]:6.1f}  sig0 {10**x[2]:.1e}   "
              f"solve {det['solve_corr']:+.3f}  sim {det['sim']:+.4f}  "
              f"gap {det['gap']:.3f}  rank {det['rank']:5.1f}  [{det['secs']:.0f}s]",
              flush=True)
    return det["sim"], det


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--calls", type=int, default=30)
    ap.add_argument("--initial", type=int, default=10)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--nfreq", type=int, default=32)
    ap.add_argument("--nvert", type=int, default=1000)
    ap.add_argument("--iters", type=int, default=150)
    ap.add_argument("--draws", type=int, default=2)
    ap.add_argument("--frames", type=int, default=280)
    ap.add_argument("--regions", nargs="*", type=int, default=None)
    ap.add_argument("--c0-max", type=float, default=0.5, dest="c0_max",
                    help="log10 upper bound on wave speed")
    ap.add_argument("--tag", default="bo_medium")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.regions:
        xspec.REGIONS = list(a.regions)

    from skopt import Optimizer
    from skopt.space import Real
    SP = space(a.c0_max)
    space_ = [Real(lo, hi, name=nm) for nm, lo, hi in SP]
    opt = Optimizer(space_, base_estimator="GP", acq_func="EI",
                    n_initial_points=a.initial, random_state=0)

    print(f"BO over the medium: {[nm for nm, _, _ in SP]}, c0 up to "
          f"{10**a.c0_max:.1f}, {len(xspec.REGIONS)} regions {xspec.REGIONS}, "
          f"{a.calls} evaluations, input solved convexly at each one")
    ctx = mp.get_context("spawn")
    trace = []
    with ctx.Pool(a.workers, initializer=_init,
                  initargs=(a.nfreq, a.nvert, a.iters, a.draws, a.frames,
                            xspec.REGIONS)) as pool:
        _init(a.nfreq, a.nvert, a.iters, a.draws, a.frames)      # parent needs it too
        for i in range(a.calls):
            x = opt.ask()
            y, det = evaluate(pool, x)
            opt.tell(x, -y)
            trace.append(det)
            best = max(trace, key=lambda d: d["sim"])
            if (i + 1) % 5 == 0:
                print(f"  after {i+1:3d}: best sim {best['sim']:+.4f} at "
                      f"c0 {10**best['x'][0]:.2f}, Ld {10**best['x'][1]:.1f}, "
                      f"sig0 {10**best['x'][2]:.1e}", flush=True)

    best = max(trace, key=lambda d: d["sim"])
    print(f"\n  best: sim {best['sim']:+.4f} +- {best['sim_sd']:.4f}  "
          f"gap {best['gap']:.3f}  rank {best['rank']:.1f}  "
          f"(convex solve gave {best['solve_corr']:+.3f})")
    print(f"  medium: c0 {10**best['x'][0]:.3f}, Ld {10**best['x'][1]:.2f}, "
          f"sig0 {10**best['x'][2]:.3e}")
    path = os.path.join(OUT, f"{a.tag}.pkl")
    with open(path, "wb") as fh:
        pickle.dump(dict(trace=[{k: v for k, v in d.items()} for d in trace],
                         best=best, space=SP, regions=list(xspec.REGIONS)), fh)
    print(f"  wrote {path}")


if __name__ == "__main__":
    main()
