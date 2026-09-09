"""Search the medium's speed/damping grading over a map basis, with a reduced solve.

The maps are worth +0.030 of sim (a flat medium scores +0.6606 against the incumbent's
+0.6901) and their overall STRENGTH is already at an interior optimum - `--maps-scale` 0 /
1 / 1.28 gives +0.6606 / +0.6901 / +0.6665. What has never been fitted under the current
clock, target, passband or metric is their DIRECTION: the six coefficients are the bo_step
winner and predate all four.

The basis here nests the incumbent. myelin/thickness/sulc are the maps it already uses, so
the search starts exactly at the current best medium and the eigenmodes lb1..lb8 are extra
freedom on top. A null result is then interpretable - it says the incumbent direction is
locally optimal - which is not true of an eigenmode-only basis, since 32 modes hold only
18.5% of sulc and the incumbent puts a 0.35 coefficient on it.

The objective is the REDUCED solve, validated to rank the three known media correctly
(1 > 1.28 > 0) at nvert 400 / 50 iterations under both the raw-FC and the K=90 affinity
target. It preserves rank and not spacing, so it ranks media and cannot be read for effect
sizes; the winner has to be re-scored through the full pipeline.

  python fit/map_search.py --evals 400 --tag anat_lb8
"""
import _path  # noqa: F401
import os, time, argparse
import numpy as np

from mesh_cache import load_cortex
from paths import RESULTS
import fc_score, xspec, bo_step, subparcels, units, timescale, bandpass, gradients
import fluid as fl
from best_fit import BEST_X

DEFAULT_MAPS = "myelin,thickness,sulc," + ",".join(f"lb{k}" for k in range(1, 9))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--maps", default=DEFAULT_MAPS)
    ap.add_argument("--nvert", type=int, default=400)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--target-affinity", type=int, default=90, dest="target_affinity")
    ap.add_argument("--evals", type=int, default=400)
    ap.add_argument("--popsize", type=int, default=0)
    ap.add_argument("--sigma0", type=float, default=0.15)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reach-cap", type=float, default=0.0, dest="reach_cap",
                    help="cap the realised spread of c/sigma (the local reach) at this "
                         "factor. The incumbent runs at 18.97x; the LB search reached "
                         "230x for +0.0094 (0.96 sigma), so the cap is what stops the "
                         "search buying score with heterogeneity nobody asked for. "
                         "Enforced by SCALING (a-b) rather than rejecting, so no "
                         "evaluation is wasted and `b` is left untouched")
    ap.add_argument("--damping-cap", type=float, default=0.0, dest="damping_cap",
                    help="cap the realised spread of sigma at this factor. The incumbent "
                         "runs at 18.06x. Reach alone does not bound this: a medium can "
                         "hold c/sigma fixed while scaling both, giving a constant SPATIAL "
                         "scale and a wildly varying TEMPORAL one")
    ap.add_argument("--stage", default="both", choices=("both", "damping", "speed"),
                    help="which block to search. 'damping' holds a at the seed and moves "
                         "b; 'speed' holds b and moves a. Two 3-parameter stages instead "
                         "of one 6-parameter search, so the two cannot be pushed together")
    ap.add_argument("--init", default="",
                    help="mapsearch npz to start from, for the second stage")
    ap.add_argument("--check-every", type=int, default=5, dest="check_every",
                    help="every N generations, re-evaluate the current best at a STRICTER "
                         "budget (nvert 1000, 400 iterations) and log both. The 22-map run "
                         "gained +0.069 on the surrogate and +0.0094 (0.96 sigma) in the "
                         "full pipeline, so the surrogate gap has to be watched while the "
                         "search runs rather than discovered after it")
    ap.add_argument("--tag", default="mapsearch")
    a = ap.parse_args()

    maps = tuple(m.strip() for m in a.maps.split(","))
    m = len(maps)
    c = load_cortex("fsaverage5", verbose=False)
    t = fc_score.default_target(c, verbose=False)
    parcels, split = subparcels.region_set(c, "subcortical", 40, 1.0)
    labels, tags = subparcels.split_parcels(c, parcels, split, verbose=False)
    P = subparcels.shell_profiles(c, labels, len(tags), 1)
    clock = timescale.plan(4, decay_s=25.0, spread_mm_s=1.47)
    kern = units.smoothing_kernel(timescale.bold_fwhm_frames(clock["frame_s"]))
    sub = xspec.medoid_subset(t, a.nvert)
    cols = t.cols[sub]
    iu = np.triu_indices(len(sub), 1)

    raw = np.asarray(t.target_fc()[np.ix_(sub, sub)], np.float64)
    raw = raw - raw.mean(0, keepdims=True) - raw.mean(1, keepdims=True) + raw.mean()
    if a.target_affinity:
        Tg = gradients.affinity_lowrank(t.target_fc(), a.target_affinity, sub=sub)
        Tg = Tg - Tg.mean(0, keepdims=True) - Tg.mean(1, keepdims=True) + Tg.mean()
    else:
        Tg = raw
    Tgt = xspec.normal_scores(Tg, iu)
    Ctn = Tgt.copy(); np.fill_diagonal(Ctn, 0.0); Ctn /= np.linalg.norm(Ctn)

    import cortical_maps as _cm
    Mst = np.stack([_cm.load_maps(c, maps, verbose=False)[k] for k in maps])
    free = {"both": slice(0, 2 * m), "damping": slice(m, 2 * m),
            "speed": slice(0, m)}[a.stage]

    def spread(v):
        u = np.asarray(v, float) @ Mst
        return float(u.max() - u.min())

    def project(theta, blk=None):
        """Scale a candidate into the spread caps, touching only the block being searched.

        Scaling `(a-b)` to cap reach rewrites `a`, so in the damping stage it dragged speed
        along and the staging did nothing. Instead the FREE block is scaled by the largest
        t in [0, 1] that satisfies both caps, found by bisection on a linear form - cheap,
        and t = 0 is always feasible (zero damping gives 1x spread; zero speed leaves reach
        at whatever the fixed b already satisfies).

        Not a rejection: no evaluation is wasted, and `best_x` is stored already projected,
        so the medium on record is the feasible one."""
        th = np.clip(np.asarray(theta, float), -fl.COEF_LIM, fl.COEF_LIM)
        blk = free if blk is None else blk
        if not (a.reach_cap or a.damping_cap):
            return th
        base = th.copy(); base[blk] = 0.0

        def ok(t):
            v = base.copy()
            v[blk] = th[blk] * t
            if a.damping_cap and spread(v[m:]) > np.log(a.damping_cap) + 1e-12:
                return False
            if a.reach_cap and spread(v[:m] - v[m:]) > np.log(a.reach_cap) + 1e-12:
                return False
            return True

        if ok(1.0):
            return th
        lo, hi = 0.0, 1.0
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if ok(mid) else (lo, mid)
        out = base.copy(); out[blk] = th[blk] * lo
        return out

    x0 = BEST_X.copy()
    x0[3] = np.log10(clock["save"]); x0[0] = np.log10(clock["damp"])
    base_p, save, _ = bo_step.unpack(x0, c)
    nsteps = 1088 * save
    br = [None]

    def rho(theta, nvert=None, iters=None):
        """theta = [a (m), b (m)] -> the reduced solve's objective for that medium."""
        theta = project(theta)
        nvert = a.nvert if nvert is None else nvert
        iters = a.iters if iters is None else iters
        p = dict(base_p)
        p["maps"] = maps
        p["a"] = np.asarray(theta[:m], float)
        p["b"] = np.asarray(theta[m:], float)
        # cache=False: the key carries a and b, so every candidate is a miss and one
        # 47-piece set is 1.79 GB.
        resp = xspec.impulse_responses(c, list(range(len(P))), p, nsteps, save,
                                       profiles=P, verbose=False, workers=a.workers,
                                       cache=False)
        # Subset the vertices BEFORE padding: padding the full response to 4096 frames is
        # a 7.2 GB array per candidate, and transfer only ever looks at `cols`.
        sb = sub if nvert == a.nvert else xspec.medoid_subset(t, nvert)
        cl = t.cols[sb] if nvert != a.nvert else cols
        G = np.ascontiguousarray(resp[:, :, cl])
        del resp
        G = np.pad(G, ((0, 0), (0, 4096 - G.shape[1]), (0, 0)))
        H, w, idx = xspec.transfer(G, np.arange(len(cl)), 192, kernel=kern)
        if br[0] is None:
            br[0] = bandpass.transfer_response(idx, G.shape[1], clock["frame_s"], 0.01, 0.08)
        H = H * br[0][:, None, None]
        if nvert == a.nvert:
            Tg_, Ctn_ = Tgt, Ctn
        else:
            iu_ = np.triu_indices(len(sb), 1)
            Aq = gradients.affinity_lowrank(t.target_fc(), a.target_affinity, sub=sb)
            Aq = Aq - Aq.mean(0, keepdims=True) - Aq.mean(1, keepdims=True) + Aq.mean()
            Tg_ = xspec.normal_scores(Aq, iu_)
            Ctn_ = Tg_.copy(); np.fill_diagonal(Ctn_, 0.0); Ctn_ /= np.linalg.norm(Ctn_)
        _, C = xspec.solve(H, w, Tg_, iters=iters, verbose=False)
        C = C - C.mean(0, keepdims=True) - C.mean(1, keepdims=True) + C.mean()
        np.fill_diagonal(C, 0.0)
        n = np.linalg.norm(C)
        return float((C * Ctn_).sum() / n) if n > 0 else -1.0

    # the incumbent, embedded in the larger basis: its three anatomical coefficients, and
    # zero on every eigenmode. So generation 0 contains the current best medium exactly.
    seed = np.zeros(2 * m)
    seed[:3] = BEST_X[4:7]
    seed[m:m + 3] = BEST_X[7:10]
    if a.init:
        zi = np.load(a.init, allow_pickle=True)
        seed = np.asarray(zi["best_x"], float).copy()
        print(f"  seeded from {os.path.basename(a.init)} "
              f"(its surrogate best {float(zi['best']):+.6f})")
    seed = project(seed)
    print(f"  stage '{a.stage}': moving {len(seed[free])} of {2*m} coefficients")
    if a.reach_cap or a.damping_cap:
        print(f"  caps: reach {a.reach_cap or float('inf'):.4g}x, "
              f"damping {a.damping_cap or float('inf'):.4g}x; seed sits at "
              f"reach {np.exp(spread(seed[:m]-seed[m:])):.3f}x, "
              f"damping {np.exp(spread(seed[m:])):.3f}x")

    def expand(zfree):
        th = seed.copy(); th[free] = np.asarray(zfree, float); return th
    print(f"  {m} maps: {', '.join(maps)}")
    print(f"  {2*m} coefficients, box +-{fl.COEF_LIM}, seeded at the incumbent "
          f"(a={np.round(seed[:m],3)}, b={np.round(seed[m:],3)})")
    t0 = time.time()
    r0 = rho(seed)
    dt = time.time() - t0
    print(f"  incumbent rho = {r0:+.6f}   [{dt:.1f}s per candidate]")
    print(f"  budget {a.evals} candidates ~ {a.evals*dt/3600:.1f} h", flush=True)

    import cma
    opts = dict(bounds=[-fl.COEF_LIM, fl.COEF_LIM], seed=a.seed + 1,
                maxfevals=a.evals, verbose=-9)
    if a.popsize:
        opts["popsize"] = a.popsize
    es = cma.CMAEvolutionStrategy(seed[free], a.sigma0, opts)
    best, best_x, hist, n, checks = r0, seed.copy(), [], 0, []
    strict_seed = [None]
    out = os.path.join(RESULTS, f"mapsearch_{a.tag}.npz")
    while not es.stop():
        X = es.ask()
        Y = []
        for x in X:
            th = expand(x)
            v = rho(th); Y.append(-v); n += 1
            if v > best:
                best, best_x = v, project(th)
        es.tell(X, Y)
        hist.append(best)
        if a.check_every and len(hist) % a.check_every == 0:
            if strict_seed[0] is None:          # the seed's strict value never changes
                strict_seed[0] = rho(seed, nvert=1000, iters=400)
            vb, vs = rho(best_x, nvert=1000, iters=400), strict_seed[0]
            checks.append((len(hist), best, vb, vs))
            print(f"    strict check: best {vb:+.6f} vs incumbent {vs:+.6f} "
                  f"({vb-vs:+.6f})   surrogate said {best-r0:+.6f}", flush=True)
        np.savez(out, best_x=best_x, best=best, seed=seed, r0=r0, maps=np.array(maps, object),
                 hist=np.array(hist), n=n, checks=np.array(checks, float), nvert=a.nvert, iters=a.iters,
                 target_affinity=a.target_affinity)
        print(f"  gen {len(hist):3d}  evals {n:4d}  best {best:+.6f} "
              f"(incumbent {r0:+.6f}, {best-r0:+.6f})  [{(time.time()-t0)/60:.0f} min]",
              flush=True)
    print(f"\n  best a = {np.round(best_x[:m], 4)}")
    print(f"  best b = {np.round(best_x[m:], 4)}")
    print(f"  rho {best:+.6f} against incumbent {r0:+.6f} ({best-r0:+.6f})")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
