"""Concatenated segments, each with its own medium and its own input.

The proposal: run several stationary segments back to back and let their covariances add.
With zero-mean, equal-mean segments the sample covariance of the concatenation is
Sigma = sum_i w_i Sigma_i with w_i the length fractions, so a later segment can be built to
supply what the earlier ones missed.

For a FIXED medium this buys nothing - the achievable set {sum_f H S H^H : S >= 0} is a
convex cone, so a non-negative mixture of reachable covariances is itself reachable by one
segment. It buys something only when the segments differ in H.

When they do, the joint problem collapses to something already implemented. Put
T_i = w_i S_i; since w_i >= 0 and S_i >= 0, T_i >= 0, and

    Sigma = sum_i sum_f H_i(f) T_i(f) H_i(f)^H

so the lengths are absorbed into the variables and choosing lengths AND inputs together is
one convex solve over the stacked transfer function [H_1 ... H_R] with a block-diagonal PSD
constraint - which is xspec.solve(nblock=R). The block traces then say how long each
segment should be, instead of an occupancy schedule being fixed in advance.

This differs from regimes.py in the one way that matters. That switches media WITHIN a run
and carries the field state across each boundary, which made the run "a sequence of kicks,
not the mixture of stationary regimes the solve assumes". Independent segments concatenated
afterwards have no boundary transient at all.

It also differs from regime_fcs.py in what is varied. That mixed seven media which were
redistributions of ONE medium - map grading only, with peak speed and mean damping
renormalised to be unchanged - and found the mixture worth +0.015. Global speed, global
damping, the driven set and the coupling were all held fixed, and those are the axes here.

  python segments.py --library speed --oversample 4
  python segments.py --library input --oversample 4
"""
import os, argparse, time
import numpy as np

from mesh_cache import load_cortex
from paths import RESULTS
import fc_score, xspec, bo_step, subparcels, timescale, units
from best_fit import BEST_X


def library(name, base_clock):
    """-> list of (label, overrides) describing each segment's configuration.

    `overrides` may carry: spread_mm_s, decay_s (medium, via the clock), regions/split
    (which pieces are driven), lam (long-range coupling)."""
    if name == "single":
        return [("base", {})]
    if name == "speed":
        return [(f"spread{v}", dict(spread_mm_s=v)) for v in (1.5, 3.0, 6.0, 12.0)]
    if name == "decay":
        return [(f"decay{v}s", dict(decay_s=v)) for v in (4.5, 9.0, 18.0, 36.0)]
    if name == "input":
        return [("sensory", dict(regions="sensory")),
                ("sens+dmn", dict(regions="sensory+dmn")),
                ("spread", dict(regions="spread")),
                ("dmn", dict(regions="dmn"))]
    if name == "coupling":
        return [(f"lam{v}", dict(lam=v)) for v in (0.0, 0.3, 1.0, 3.0)]
    if name == "mixed":
        return [("slow", dict(spread_mm_s=1.5, decay_s=18.0)),
                ("base", {}),
                ("fast", dict(spread_mm_s=12.0, decay_s=4.5)),
                ("spread-in", dict(regions="spread")),
                ("coupled", dict(lam=1.0))]
    raise ValueError(name)


def build_segment(c, t, sub, over, base, a, cache):
    """-> (H, w, idx, ref, profiles, medium, save, coupling) for one segment."""
    clock = timescale.plan(a.oversample,
                           decay_s=over.get("decay_s", a.decay_s),
                           spread_mm_s=over.get("spread_mm_s", a.spread_mm_s),
                           verbose=False)
    x = BEST_X.copy()
    x[3] = np.log10(clock["save"]); x[0] = np.log10(clock["damp"])
    p, save, _ = bo_step.unpack(x, c)
    regions = over.get("regions", "sensory")
    key = (regions, a.split)
    if key not in cache:
        parcels, split = subparcels.region_set(c, regions, a.split, 1.0)
        lab, tg = subparcels.split_parcels(c, parcels, split, verbose=False)
        cache[key] = subparcels.taper_profiles(c, lab, len(tg))
    P = cache[key]
    cpl = None
    if over.get("lam", 0.0) > 0:
        import connectome
        D = connectome.parcel_distances(c, verbose=False)
        W = connectome.residual_W(connectome.load_enigma(c, verbose=False), D,
                                  0.15, 60.0, verbose=False)
        cpl = connectome.CouplingOperator(c, W, over["lam"], 0)
    decay_fr = 1.0 / (clock["damp"] * clock["save"])
    win = int(np.ceil(4 * decay_fr / 32.0) * 32)
    resp = xspec.impulse_responses(c, list(range(len(P))), p, win * save, save,
                                   profiles=P, verbose=False, coupling=cpl,
                                   workers=a.workers)
    R = np.pad(resp, ((0, 0), (0, max(0, a.pad - resp.shape[1])), (0, 0)))
    kern = units.smoothing_kernel(
        timescale.bold_fwhm_frames(clock["frame_s"], verbose=False), verbose=False)
    H, w, idx = xspec.transfer(R, t.cols[sub], a.nfreq, kernel=kern)
    del resp, R
    return dict(H=H, w=w, idx=idx, ref=a.pad, P=P, p=p, save=save, cpl=cpl,
                clock=clock, kern=kern)


def _nnls_path(c, t, sub, segs, Tgt, raw, iu, w, a):
    """Solve each segment on its own, then find the best NON-NEGATIVE mixture.

    The concatenation weights must be non-negative and sum to one, so the reachable set of
    concatenated covariances is the convex hull of the per-segment ones. NNLS over the
    edge vectors finds the best point in it. This is weaker than the stacked solve - each
    segment's input is fixed at its own optimum rather than chosen jointly - so it is a
    LOWER bound on what concatenation can do, and it is the only route when the segments
    have different channel counts."""
    from scipy.optimize import nnls
    n = len(sub)
    Tn = Tgt / np.linalg.norm(Tgt)
    y = Tgt[iu].astype(float)
    cols, singles = [], []
    for s in segs:
        S, C = xspec.solve(s["H"], s["w"], Tgt, iters=a.iters, verbose=False)
        Cm = C.copy(); Cm[np.eye(n, dtype=bool)] = 0.0
        obj = float((Cm * Tn).sum() / np.linalg.norm(Cm))
        s["S"] = S
        cols.append(C[iu].astype(float))
        singles.append(obj)
        print(f"    {s['label']:<10s} solved alone: objective {obj:.4f}", flush=True)
    X = np.stack(cols, axis=1)
    wts, _ = nnls(X, y)
    if wts.sum() <= 0:
        print("  NNLS returned all-zero weights")
        return
    wn = wts / wts.sum()
    fit = X @ wts
    ry = (y - y.mean()) / y.std(); rf = (fit - fit.mean()) / fit.std()
    print(f"\n  best single segment      {max(singles):.4f}  "
          f"({segs[int(np.argmax(singles))]['label']})")
    print(f"  non-negative mixture     {float(rf @ ry / len(ry)):.4f}")
    print(f"\n  {'segment':<10s} {'weight':>9s} {'seconds':>9s}")
    for s, v in zip(segs, wn):
        print(f"  {s['label']:<10s} {v:>9.4f} {v*a.seconds:>9.1f}")

    sims = []
    for d in range(a.draws):
        chunks = []
        for r, s in enumerate(segs):
            nfr = int(round(wn[r] * a.seconds / s["clock"]["frame_s"]))
            if nfr < 32:
                continue
            A = xspec.realise(s["S"], s["idx"], nfr, ref_frames=s["ref"],
                              seed=3000 + 17 * d + r)
            rr = xspec.score_realisation(c, t, s["p"], A, save=s["save"],
                                         profiles=s["P"], kernel=s["kern"])
            chunks.append(rr["frames"])
        F = np.concatenate(chunks, axis=0)
        sim = float(t._prep(t.model_edges(frames=F)[0]) @ t.y)
        sims.append(sim)
        print(f"  draw {d}: {F.shape[0]} frames from {len(chunks)} segments "
              f"-> sim {sim:+.4f}", flush=True)
    print(f"\n  concatenated over {a.draws} draws: {np.mean(sims):+.4f} "
          f"+- {np.std(sims):.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--library", default="mixed",
                    choices=("single", "speed", "decay", "input", "coupling", "mixed"))
    ap.add_argument("--oversample", type=int, default=4)
    ap.add_argument("--decay-s", type=float, default=timescale.BOLD_TAU_S, dest="decay_s")
    ap.add_argument("--spread-mm-s", type=float, default=3.0, dest="spread_mm_s")
    ap.add_argument("--seconds", type=float, default=577.0)
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--draws", type=int, default=2)
    ap.add_argument("--split", type=int, default=50)
    ap.add_argument("--nvert", type=int, default=1000)
    ap.add_argument("--nfreq", type=int, default=192)
    ap.add_argument("--pad", type=int, default=4096)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--nnls", action="store_true",
                    help="solve segments separately and mix by NNLS instead of stacking")
    ap.add_argument("--tag", default="seg")
    a = ap.parse_args()

    c = load_cortex("fsaverage5", verbose=False)
    t = fc_score.default_target(c, verbose=False)
    sub = xspec.medoid_subset(t, a.nvert); n = len(sub)
    iu = np.triu_indices(n, 1)
    raw = np.asarray(t.target_fc()[np.ix_(sub, sub)], np.float64)
    raw = raw - raw.mean(0, keepdims=True) - raw.mean(1, keepdims=True) + raw.mean()
    Tgt = xspec.normal_scores(raw, iu); Tgt[np.eye(n, dtype=bool)] = 0.0

    lib = library(a.library, None)
    print(f"  library '{a.library}': {len(lib)} segments")
    segs, cache = [], {}
    for lb, over in lib:
        t0 = time.time()
        s = build_segment(c, t, sub, over, None, a, cache)
        s["label"] = lb
        segs.append(s)
        print(f"    {lb:<10s} K={s['H'].shape[2]:<3d} save={s['save']:<3d} "
              f"spread={s['clock']['spread_mm_s']:.2f} mm/s "
              f"decay={s['clock']['decay_s']:.1f}s"
              + (f" lam={over['lam']}" if over.get("lam") else "")
              + f"  [{time.time()-t0:.0f}s]", flush=True)

    # every segment must be sampled on ONE frequency grid for the stack to be meaningful
    idx0 = segs[0]["idx"]
    for s in segs[1:]:
        if not np.array_equal(s["idx"], idx0):
            raise SystemExit("  segments landed on different frequency grids; give them "
                             "a common --pad so the stacked solve is well posed")
    Ks = [s["H"].shape[2] for s in segs]
    if len(set(Ks)) != 1 or a.nnls:
        if len(set(Ks)) != 1:
            print(f"  channel counts differ {Ks}: _project's block projection assumes "
                  f"equal blocks, so this library is solved segment-wise and mixed by "
                  f"NNLS instead of stacked.")
        return _nnls_path(c, t, sub, segs, Tgt, raw, iu, w, a)
    Hs = np.concatenate([s["H"] for s in segs], axis=2)
    w = segs[0]["w"]
    print(f"\n  stacked transfer: {Hs.shape[0]} freq x {Hs.shape[1]} vertices x "
          f"{Hs.shape[2]} channels ({len(segs)} blocks of {Ks[0]})")

    S, C = xspec.solve(Hs, w, Tgt, iters=a.iters, verbose=False, nblock=len(segs))
    Tn = Tgt / np.linalg.norm(Tgt)
    Cm = C.copy(); Cm[np.eye(n, dtype=bool)] = 0.0
    print(f"  stacked solve objective {float((Cm*Tn).sum()/np.linalg.norm(Cm)):.4f}")

    K = Ks[0]
    occ = np.array([sum(w[f] * np.trace(S[f, r*K:(r+1)*K, r*K:(r+1)*K]).real
                        for f in range(len(w))) for r in range(len(segs))])
    occ = np.clip(occ, 0, None); occ = occ / max(occ.sum(), 1e-30)
    print(f"\n  {'segment':<10s} {'power share':>12s} {'seconds':>9s} {'frames':>8s}")
    for s, o in zip(segs, occ):
        secs = o * a.seconds
        print(f"  {s['label']:<10s} {o:>12.4f} {secs:>9.1f} "
              f"{int(round(secs / s['clock']['frame_s'])):>8d}")

    # realise each segment for its share and concatenate, then score the whole thing
    sims = []
    for d in range(a.draws):
        chunks = []
        for r, s in enumerate(segs):
            nfr = int(round(occ[r] * a.seconds / s["clock"]["frame_s"]))
            if nfr < 32:
                continue
            Sr = np.ascontiguousarray(S[:, r*K:(r+1)*K, r*K:(r+1)*K])
            tr = sum(w[f] * np.trace(Sr[f]).real for f in range(len(w)))
            if tr <= 0:
                continue
            A = xspec.realise(Sr, s["idx"], nfr, ref_frames=s["ref"], seed=2000 + 17*d + r)
            rr = xspec.score_realisation(c, t, s["p"], A, save=s["save"],
                                         profiles=s["P"], kernel=s["kern"],
                                         run_fn=(None if s["cpl"] is None else
                                                 (lambda dr, ns, sv, _s=s:
                                                  __import__("fluid").run(
                                                      c, dr, _s["p"], ns, sv,
                                                      coupling=_s["cpl"]))))
            chunks.append(rr["frames"])
        F = np.concatenate(chunks, axis=0)
        sim = float(t._prep(t.model_edges(frames=F)[0]) @ t.y)
        sims.append(sim)
        print(f"  draw {d}: concatenated {F.shape[0]} frames from {len(chunks)} "
              f"segments -> sim {sim:+.4f}", flush=True)
        if d == 0:
            np.save(os.path.join(RESULTS, f"frames_{a.tag}.npy"), F)
    print(f"\n  concatenated over {a.draws} draws: {np.mean(sims):+.4f} "
          f"+- {np.std(sims):.4f}")


if __name__ == "__main__":
    main()
