"""Fit the 100-region model against an ENVELOPE observable, on the real target.

best_fit's observable is linear in the field, so a fast drive is invisible to it: an LTI
medium never moves power between frequencies, and the passband discards whatever sits
outside it. Squaring before filtering does move power - C_ij(tau)^2 in time is the
cross-spectrum convolved with itself in frequency - so components at f1 and f2 contribute
at |f1 - f2|. See fit/envelope.py for the derivation and fit/toy_recover.py for it
measured against simulation on a system whose answer is known.

THREE DIFFERENCES FROM best_fit, all forced by where the square goes.

H is RAW. xspec.transfer folds the BOLD kernel into H and best_fit multiplies the passband
on top, which is right when the observable is linear. Here the square acts on the field,
so every filter belongs in rho(tau) and H carries none of them.

The solve is on E, not C. E is quadratic in S where C is linear, so convexity is not given
by construction. On the toy it reached the same optimum from every random start to 1e-4
relative; --starts repeats that test here.

The realisation squares too. score_realisation applies kernel then passband to the FIELD;
envelope.observable squares first. Scoring the fit through the linear path would report a
number for a system nobody simulated.

COST. The lag count follows the passband: 35 lags for the BOLD kernel alone, 1,579 with
0.01-0.08 Hz at a 1e-3 tolerance, because a 0.01 Hz filter rings for ~100 s. The forward
is then ~nlag x nfreq x nvert^2, which is why --nvert defaults to 400 rather than the
1,000 best_fit solves on. At 400 the envelope forward measured 1.7x the linear one.

  python fit/envelope_fit.py --oversample 4 --decay-s 25 --spread-mm-s 1.5 \\
    --hybrid-file results/xspec_asc_first_area_100.npz --map-clip iqr --tag env100
"""
import _path  # noqa: F401  - puts the sibling code folders on sys.path
import os, time, argparse
import numpy as np

from mesh_cache import load_cortex
from paths import RESULTS
import fc_score, xspec, bo_step, subparcels, timescale, units, bandpass, provenance
import lagged as _lg
import envelope as env
from best_fit import BEST_X, normal_scores


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--hybrid-file", required=True, dest="hybrid_file",
                    help="npz with `labels`, `tags` and optionally `profiles` - an "
                         "xspec_<tag>.npz carries all three")
    ap.add_argument("--oversample", type=int, default=4)
    ap.add_argument("--spread-mm-s", type=float, default=1.5, dest="spread")
    ap.add_argument("--decay-s", type=float, default=25.0, dest="decay_s")
    ap.add_argument("--impulse-decays", type=float, default=7.0, dest="idecays")
    ap.add_argument("--pad", type=int, default=4096)
    ap.add_argument("--nfreq", type=int, default=192)
    ap.add_argument("--band", default="0.01,0.08")
    ap.add_argument("--maps-scale", type=float, default=1.0, dest="maps_scale")
    ap.add_argument("--map-clip", default="none", dest="map_clip",
                    choices=("none", "iqr", "tukey", "sd2"))
    ap.add_argument("--nvert", type=int, default=400,
                    help="solve vertices. The envelope forward is ~nlag x nfreq x nvert^2 "
                         "and nlag is ~1,579 with the passband, so this is the knob that "
                         "decides whether a solve takes minutes or a day")
    ap.add_argument("--lag-tol", type=float, default=1e-3, dest="lag_tol",
                    help="drop lags where the filter autocorrelation is below this "
                         "fraction of its peak. 1e-3 keeps 1,579 of 4,096; 1e-6 keeps "
                         "4,006 and costs 2.5x for the tail of a filter")
    ap.add_argument("--chunk", type=int, default=128, help="lags per block")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--starts", type=int, default=1,
                    help=">1 re-solves from random PSD starts. E is quadratic in S, so "
                         "whether they agree is the convexity test")
    ap.add_argument("--seconds", type=float, default=577.0)
    ap.add_argument("--draws", type=int, default=2)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--decimate", default="snapshot", choices=("snapshot", "mean"),
                    help="how steps become frames, impulses and realisation alike; 'mean' "
                         "for any medium faster than 1x (see fluid.run)")
    ap.add_argument("--tag", default="env100")
    a = ap.parse_args()
    lo, hi = (float(v) for v in a.band.split(","))

    t0 = time.time()
    c = load_cortex("fsaverage5", verbose=False)
    t = fc_score.default_target(c, verbose=True)
    cl = timescale.plan(a.oversample, decay_s=a.decay_s, spread_mm_s=a.spread,
                        verbose=False)
    x = np.array(BEST_X, copy=True)
    x[3] = np.log10(cl["save"]); x[0] = np.log10(cl["damp"])
    if a.maps_scale != 1.0:
        x[4:10] = x[4:10] * a.maps_scale
    p, save, _ = bo_step.unpack(x, c)
    p["map_clip"] = a.map_clip
    # the envelope reads EVERY frequency of the field, so a fast medium's snapshot frames,
    # which fold content above the frame Nyquist into lower bins, corrupt it even more than
    # they corrupt the linear solve; "mean" averages each frame's steps (see fluid.run)
    p["decimate"] = a.decimate

    z = np.load(a.hybrid_file, allow_pickle=True)
    labels = np.asarray(z["labels"], np.int64)
    tags = [str(v) for v in np.asarray(z["tags"])]
    P = (np.asarray(z["profiles"], np.float32) if "profiles" in z.files
         else subparcels.taper_profiles(c, labels, len(tags)))
    K = len(tags)

    decay_fr = 1.0 / (cl["damp"] * save)
    imp = int(np.ceil(max(a.idecays * decay_fr, 1.0 / (lo * cl["frame_s"])) / 64) * 64)
    pad = max(a.pad, int(2 ** np.ceil(np.log2(imp))))
    sub = xspec.medoid_subset(t, a.nvert)
    cf, sg = __import__("fluid").fields(c, p)
    print(f"  {K} channels, {a.nvert} solve vertices, medium {cl['spread_mm_s']:.2f} mm/s "
          f"decay {cl['decay_s']:.1f}s reach {cl['reach_mm']:.0f} mm, map_clip {a.map_clip}")
    print(f"  speed {cf.min():.3f}-{cf.max():.3f}, impulse window {imp} frames "
          f"({imp*cl['frame_s']:.0f}s), pad {pad} ({pad*cl['frame_s']:.0f}s)", flush=True)

    resp_t = xspec.impulse_responses(c, list(range(K)), p, imp * save, save, profiles=P,
                                     verbose=False, workers=a.workers,
                                     keep=np.asarray(t.cols)[sub])
    R = np.pad(resp_t, ((0, 0), (0, max(0, pad - resp_t.shape[1])), (0, 0)))
    del resp_t
    # RAW: no kernel here, the filtering is all in rho
    H, w, idx = xspec.transfer(R, np.arange(a.nvert), a.nfreq, kernel=None)
    del R

    nb = pad // 2 + 1
    kern = units.smoothing_kernel(timescale.bold_fwhm_frames(cl["frame_s"], verbose=False),
                                  verbose=False)
    respf = (units.kernel_response(kern, nb, pad)
             * bandpass.response(np.arange(nb) / (pad * cl["frame_s"]),
                                 cl["frame_s"], lo, hi))
    lags, rho = env.filter_lags(respf, pad, tol=a.lag_tol)
    ph = _lg.phases(idx, pad, lags)
    print(f"  {len(idx)} solved frequencies, {len(lags)} lags of {pad} carry the filter "
          f"autocorrelation (tol {a.lag_tol:g})", flush=True)

    raw = np.asarray(t.target_fc()[np.ix_(sub, sub)], np.float64)
    raw = raw - raw.mean(0, keepdims=True) - raw.mean(1, keepdims=True) + raw.mean()
    iu = np.triu_indices(a.nvert, 1)
    Tgt = normal_scores(raw, iu)
    print(f"  setup {time.time()-t0:.0f}s; solving {a.iters} iterations"
          + (f" x {a.starts} starts" if a.starts > 1 else ""), flush=True)

    best, vals = None, []
    for st in range(a.starts):
        S0 = None
        if st:
            r = np.random.default_rng(1000 + st)
            A = r.normal(size=(len(idx), K, K)) + 1j * r.normal(size=(len(idx), K, K))
            S0 = np.einsum("fab,fcb->fac", A, A.conj())
            S0 = S0 / sum(np.trace(S0[f]).real for f in range(len(idx)))
        t1 = time.time()
        S, v = env.solve(H, w, ph, rho, Tgt, iters=a.iters, S0=S0, chunk=a.chunk,
                         verbose=(a.starts == 1))
        vals.append(v)
        print(f"  start {st}: envelope objective {v:+.6f}  [{time.time()-t1:.0f}s]",
              flush=True)
        if best is None or v > best[1]:
            best = (S, v)
    if a.starts > 1:
        rel = np.std(vals) / max(abs(np.mean(vals)), 1e-30)
        print(f"  {a.starts} starts: spread {np.std(vals):.2e} ({rel:.1e} relative) "
              + ("- one optimum" if rel < 1e-3 else "- STARTS DISAGREE"))
    S = best[0]

    # ---- realise and score THROUGH THE SQUARED OBSERVABLE, not the linear one
    import fluid as fl
    from xspec import ProfileDrive
    nframes = timescale.frames_for(a.seconds, cl["frame_s"])
    sims = []
    for d in range(a.draws):
        Af = xspec.realise(S, idx, nframes, ref_frames=pad, seed=1000 + d)
        Aser = np.repeat(Af, save, axis=0)[:nframes * save] / save
        fr, _ = fl.run(c, ProfileDrive(c, P, Aser, 2e-4), p, nframes * save, save)
        U = env.observable(fr[:, np.asarray(t.cols)], kern, cl["frame_s"], (lo, hi),
                           burn=t.burn)
        Z = U - U.mean(0, keepdims=True)
        Z = (Z / np.maximum(Z.std(0, keepdims=True), 1e-300)).T
        sims.append(float(t._prep(t.model_edges(Z=Z)[0]) @ t.y))
        print(f"    draw {d}: envelope-observable sim {sims[-1]:+.4f}", flush=True)
    print(f"\n  realised over {nframes} frames ({a.seconds:.0f}s), {a.draws} draws: "
          f"sim {np.mean(sims):+.4f} +- {np.std(sims):.4f}")

    np.savez(os.path.join(RESULTS, f"envfit_{a.tag}.npz"), S=S, idx=idx, x=x, w=w,
             lags=lags, rho=rho, pad=pad, save=save, frame_s=cl["frame_s"],
             band=np.array([lo, hi]), sub=sub, profiles=P, map_clip=a.map_clip,
             decimate=a.decimate,
             objective=best[1], sim=float(np.mean(sims)), sim_sd=float(np.std(sims)),
             realise_seconds=a.seconds, draws=a.draws, **provenance.stamp(a))
    print(f"  wrote results/envfit_{a.tag}.npz")


if __name__ == "__main__":
    main()
