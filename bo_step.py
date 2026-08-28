"""BO over the medium in per-step units, with observation cadence and contrast maps.

Three changes from bo_medium:

1. Per-step parameterisation. c0 is not a wave speed here: dt = CFL*d_min/c ties the
   timestep to it, so distance travelled per step is CFL*d_min whatever c is, and c only
   rescales dt. Three rescalings of one medium (c0 from 10 to 0.1, everything per-step
   matched) gave solve +0.530 and sim +0.341 in all three. What the dynamics actually
   depend on are damping per step, rotation per step, and boundary absorption per step,
   so those are searched directly with c0 pinned at 1.

2. Observation cadence. The wave covers SAVE*CFL*d_min between saved frames, and SAVE was
   fixed at 25 in every run so far - the one genuine speed knob, never searched. nsteps
   follows SAVE so the number of frames stays fixed and the FC is always estimated from
   the same sample size.

3. Contrast maps. Speed and damping vary log-linearly with myelin, thickness and sulcal
   depth across the whole sheet; six coefficients, zero meaning uniform.

The solve uses medoid vertices rather than a random subset (held-out correlation +0.581
vs +0.571, realised score +0.482 vs +0.438, and a third of the time).

  python bo_step.py --calls 40
"""
import os, time, pickle, argparse
import multiprocessing as mp
import numpy as np

from paths import RESULTS
import xspec, fluid as fl
from genome import SPONGE_STRENGTH_FIXED

OUT = os.path.join(RESULTS, "bo_step")
NFRAMES = 280
COEF_LIM = 0.45                            # +-ln units per map sd


def make_space(coef=COEF_LIM, save_lo=0.7, save_hi=1.8):
    """Search bounds. The coefficient limit is an argument because the first run put
    b_myelin and b_sulc on their bounds."""
    return [("log10_dt_sig", -4.0, -1.0),   # interior damping per step
            ("log10_dt_f", -5.0, -1.5),     # rotation per step
            ("log10_dt_spg", -3.0, 0.0),    # boundary absorption per step
            ("log10_save", save_lo, save_hi),   # steps per saved frame
            ("a_myelin", -coef, coef), ("a_thick", -coef, coef),
            ("a_sulc", -coef, coef),
            ("b_myelin", -coef, coef), ("b_thick", -coef, coef),
            ("b_sulc", -coef, coef)]


SPACE = make_space()
_W = {}


def unpack(x, cortex):
    """x -> (medium dict, save, nsteps). Per-step targets are set at the fastest point."""
    save = int(round(10.0 ** x[3]))
    a, b = np.asarray(x[4:7], float), np.asarray(x[7:10], float)
    p = xspec.medium(1.0, 1.0, 52.4, a=a, b=b)            # placeholder scales
    cfield, _ = fl.fields(cortex, p)
    dt = fl.CFL * cortex.d.min() / float(cfield.max())
    sig0 = (10.0 ** x[0]) / dt
    f = (10.0 ** x[1]) / dt
    spg = (10.0 ** x[2]) / (dt * SPONGE_STRENGTH_FIXED)
    return xspec.medium(sig0, 1.0, 1.0 / f, a=a, b=b, sponge_scale=spg), save, save * NFRAMES


def _init(nfreq, iters, ndraw, regions, split=0, pad=0, realise=0, target="normal",
          region_set="sensory"):
    from mesh_cache import load_cortex
    import fc_score
    if regions:
        xspec.REGIONS = list(regions)
    c = load_cortex("fsaverage5", verbose=False)
    t = fc_score.default_target(c, verbose=False)
    _W.update(cortex=c, target=t, nfreq=nfreq, iters=iters, ndraw=ndraw,
              sub=xspec.medoid_subset(t, 1000), pad=pad, realise=realise,
              target_mode=target)
    if split:
        import subparcels
        parcels, total = subparcels.region_set(c, region_set, split)
        labels, tags = subparcels.split_parcels(c, parcels, total, verbose=False)
        _W.update(profiles=subparcels.taper_profiles(c, labels, len(tags)),
                  labels=labels, tags=tags)
    else:
        _W["profiles"] = None


def _impulse(args):
    region, p, nsteps, save = args
    from input2 import parcel_tapers
    c = _W["cortex"]
    s, dt, g, H = fl.build(c, p, sponge=True)
    if _W.get("profiles") is not None:
        h = _W["profiles"][int(region)].astype(np.float32).copy()
    else:
        T, ids = parcel_tapers(c, verbose=False)
        pos = {int(q): i for i, q in enumerate(ids)}
        h = T[pos[int(region)]].astype(np.float32).copy()
    ue = np.zeros(s.nE, np.float32)
    fr = [h.copy()]
    for n in range(1, nsteps):
        ue, h = s.step(ue, h, np.float32(dt), g, H)
        if n % save == 0:
            fr.append(h.copy())
    return np.asarray(fr)


def evaluate(pool, x, verbose=True):
    c, t, sub = _W["cortex"], _W["target"], _W["sub"]
    P = _W.get("profiles")
    p, save, nsteps = unpack(x, c)
    keys = list(range(len(P))) if P is not None else list(xspec.REGIONS)
    t0 = time.time()
    resp = np.asarray(pool.map(_impulse, [(k, p, nsteps, save) for k in keys]))
    # zero-pad before the transform: the response has long decayed, and the padding buys
    # frequency resolution, which the sweeps showed matters more than window length
    pad = max(int(_W.get("pad", 0)), resp.shape[1])
    if pad > resp.shape[1]:
        resp = np.pad(resp, ((0, 0), (0, pad - resp.shape[1]), (0, 0)))
    H, w, idx = xspec.transfer(resp, t.cols[sub], _W["nfreq"])
    Ct = np.asarray(t.target_fc()[np.ix_(sub, sub)], np.float64)
    Ct = Ct - Ct.mean(0, keepdims=True) - Ct.mean(1, keepdims=True) + Ct.mean()
    # the score is Spearman, so solve against the normal-scored target: Pearson against
    # that is a far closer surrogate for Spearman against the raw FC. Ct itself stays raw,
    # because that is what solve_corr has always been reported against.
    Tgt = xspec.normal_scores(Ct) if _W.get("target_mode", "raw") == "normal" else Ct
    S, C = xspec.solve(H, w, Tgt, iters=_W["iters"], verbose=False)

    sims, gaps, ranks = [], [], []
    rl = int(_W.get("realise") or resp.shape[1])
    for dd in range(_W["ndraw"]):
        A = xspec.realise(S, idx, rl, ref_frames=resp.shape[1], seed=dd)
        r = xspec.score_realisation(c, t, p, A, save=save, profiles=P)
        sims.append(r["sim"]); gaps.append(r["gap"]); ranks.append(r["rank"])
    off = ~np.eye(len(sub), dtype=bool)
    from scipy.stats import spearmanr
    det = dict(solve_corr=float(np.corrcoef(C[off], Ct[off])[0, 1]),
               solve_rho=float(spearmanr(C[off], Ct[off]).statistic),
               sim=float(np.mean(sims)), sim_sd=float(np.std(sims)),
               gap=float(np.mean(gaps)), rank=float(np.mean(ranks)),
               save=save, nsteps=nsteps, secs=time.time() - t0, S=S, idx=idx,
               x=np.asarray(x))
    if verbose:
        print(f"    dtsig {10**x[0]:.1e} dtf {10**x[1]:.1e} dtspg {10**x[2]:.1e} "
              f"save {save:3d} | a {np.round(x[4:7],2)} b {np.round(x[7:10],2)} | "
              f"solve {det['solve_corr']:+.3f}/{det['solve_rho']:+.3f} "
              f"sim {det['sim']:+.4f} "
              f"gap {det['gap']:.3f} rank {det['rank']:5.1f} [{det['secs']:.0f}s]",
              flush=True)
    return det["sim"], det


def _save(a, trace, best, space):
    """Write the run so far. Called after every call, so a kill costs one evaluation."""
    with open(os.path.join(OUT, f"{a.tag}.pkl"), "wb") as fh:
        pickle.dump(dict(trace=trace, best=best, space=space, split=a.split, pad=a.pad,
                         realise=a.realise, nfreq=a.nfreq, target=a.target,
                         region_set=a.region_set, calls_done=len(trace),
                         labels=_W.get("labels"), tags=_W.get("tags"),
                         regions=list(xspec.REGIONS)), fh)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--calls", type=int, default=40)
    ap.add_argument("--initial", type=int, default=14)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--nfreq", type=int, default=32)
    ap.add_argument("--iters", type=int, default=150)
    ap.add_argument("--draws", type=int, default=2)
    ap.add_argument("--regions", nargs="*", type=int, default=None)
    ap.add_argument("--tag", default="bo_step")
    ap.add_argument("--coef-lim", type=float, default=COEF_LIM, dest="coef_lim",
                    help="bound on the speed and damping map coefficients")
    ap.add_argument("--split", type=int, default=0,
                    help="drive N equal-area sensory sub-parcels instead of whole parcels")
    ap.add_argument("--pad", type=int, default=0,
                    help="zero-pad the impulse responses to this many frames")
    ap.add_argument("--realise", type=int, default=0,
                    help="frames per realisation (default: the padded window)")
    ap.add_argument("--resume", action="store_true",
                    help="continue from this tag's checkpoint instead of starting over")
    ap.add_argument("--region-set", default="sensory", dest="region_set",
                    choices=("sensory", "dmn", "sensory+dmn", "spread"),
                    help="which parcels --split divides (see subparcels.region_set)")
    ap.add_argument("--target", default="normal", choices=("normal", "raw"),
                    help="what the convex solve matches: the normal-scored target "
                         "(matches the Spearman score) or the raw FC")
    a = ap.parse_args()
    SP = make_space(a.coef_lim)
    os.makedirs(OUT, exist_ok=True)
    if a.regions:
        xspec.REGIONS = list(a.regions)

    from skopt import Optimizer
    from skopt.space import Real
    opt = Optimizer([Real(lo, hi, name=nm) for nm, lo, hi in SP],
                    base_estimator="GP", acq_func="EI",
                    n_initial_points=a.initial, random_state=0)
    done = []
    if a.resume and os.path.exists(os.path.join(OUT, f"{a.tag}.pkl")):
        # a run this long will be interrupted at least once; the checkpoint carries every
        # (x, sim) pair, and telling them back is a faithful restart of the search
        with open(os.path.join(OUT, f"{a.tag}.pkl"), "rb") as fh:
            prev = pickle.load(fh)
        done = prev["trace"]
        opt.tell([list(map(float, d["x"])) for d in done], [-d["sim"] for d in done])
        print(f"  resuming from {len(done)} recorded evaluations, best so far "
              f"{max(d['sim'] for d in done):+.4f}")
    print(f"BO in per-step units: {[nm for nm, _, _ in SP]}, map coefficients "
          f"+-{a.coef_lim} (x{np.exp(a.coef_lim):.1f} per map sd)")
    ctx = mp.get_context("spawn")
    trace = []
    with ctx.Pool(a.workers, initializer=_init,
                  initargs=(a.nfreq, a.iters, a.draws, xspec.REGIONS, a.split, a.pad,
                            a.realise, a.target, a.region_set)) as pool:
        _init(a.nfreq, a.iters, a.draws, xspec.REGIONS, a.split, a.pad, a.realise,
              a.target, a.region_set)
        # after _init, so the channel count is the one actually being driven
        nk = len(_W["profiles"]) if _W.get("profiles") is not None else len(xspec.REGIONS)
        print(f"  {nk} drive channels from '{a.region_set}', {a.calls} evaluations, "
              f"{NFRAMES} impulse frames"
              + (f" padded to {a.pad}" if a.pad else "")
              + f", realising {a.realise or a.pad or NFRAMES}, solving against the "
              f"{a.target} target", flush=True)
        trace.extend(done)
        best = max(done, key=lambda d: d["sim"]) if done else None
        for i in range(len(done), a.calls):
            x = opt.ask()
            y, det = evaluate(pool, x)
            opt.tell(x, -y)
            if best is None or det["sim"] > best["sim"]:
                best = det
            # the solved cross-spectrum is ~2 MB a call; only the winner's is worth
            # keeping, and dropping the rest is what makes a per-call checkpoint cheap
            trace.append({k: v for k, v in det.items() if k not in ("S", "idx")})
            _save(a, trace, best, SP)                  # checkpoint every call: a run this
            if (i + 1) % 5 == 0:                       # long should not be all-or-nothing
                print(f"  after {i+1:3d}: best sim {best['sim']:+.4f} "
                      f"(save {best['save']}, dtsig {10**best['x'][0]:.1e}, "
                      f"dtf {10**best['x'][1]:.1e}, dtspg {10**best['x'][2]:.1e})",
                      flush=True)

    b = best
    print(f"\n  best: sim {b['sim']:+.4f} +- {b['sim_sd']:.4f}  gap {b['gap']:.3f}  "
          f"rank {b['rank']:.1f}  (solve pearson {b['solve_corr']:+.3f}, "
          f"spearman {b['solve_rho']:+.3f})")
    print(f"  per step: damping {10**b['x'][0]:.2e}, rotation {10**b['x'][1]:.2e}, "
          f"sponge {10**b['x'][2]:.2e}, save {b['save']} steps/frame")
    print(f"  speed maps  a = {np.round(b['x'][4:7], 3)}")
    print(f"  damping maps b = {np.round(b['x'][7:10], 3)}")
    print(f"  wrote {os.path.join(OUT, a.tag)}.pkl")


if __name__ == "__main__":
    main()
