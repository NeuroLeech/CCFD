"""Pool D realisations' FC and score the POOLED FC, D = 1, 2, 4, 8, 16.

The target is a group average over ~100 NKI subjects, so the thing it should be compared with
is a model FC averaged over draws, not one draw (see the target-is-a-100-subject-average
note). This builds that curve for a recorded solve: each draw is realised, run, smoothed and
bandpassed exactly as xspec.score_realisation does, its 2M-edge FC vector is kept, and the
running mean is scored after 1, 2, 4, ... draws - pooling the FC, as corr_score.py does, not
averaging the scores.

Draws are seeds 1000.. so draw 0 is the one best_fit and rescore.py score. Each worker
rebuilds the medium and target once, then returns edge vectors (8 MB each) - a 134 MB frame
array never crosses the pool.

FIRST CURVES, 2026-10-04, mirror-padded bandpass, 577 s, 16 draws, 400 solve vertices:

                   D=1     D=2     D=4     D=8     D=16    gain    fit (400 v)
    maxfun 15    +0.5854 +0.5946 +0.5931 +0.5977 +0.6021  +0.017    +0.6094
    maxfun 60    +0.6643 +0.6723 +0.6749 +0.6816 +0.6846  +0.020    +0.7156
    maxfun 400   +0.6325 +0.6452 +0.6692 +0.6774 +0.6862  +0.054    +0.7834
    converged    +0.5367 +0.5778 +0.6010 +0.6197 +0.6244  +0.088    +0.8165

Pooling helps most where the solve is most optimised - by 16 draws maxfun 400 has caught
maxfun 60 - and every curve levels off below its fit. Pooling removes only estimator noise,
so its limit is the EXPECTED FC scored on all 9,310 vertices; the shortfall left at D=16
(~0.03 at maxfun 60, ~0.19 converged) is the solve not transferring off its 400 vertices.

  python fit/pool_curve.py fixgrad_mf60 flat100_fac20_fix --draws 16 --workers 4
"""
import _path  # noqa: F401
import os, time, argparse
import numpy as np

from paths import RESULTS

_W = {}


def _init(tag, seconds):
    import timescale, units
    from interp_gap import build_H
    z = np.load(os.path.join(RESULTS, f'xspec_{tag}.npz'), allow_pickle=True)
    Hall, _, kern, c, t, g = build_H(tag, z, False, workers=1)
    del Hall
    _W.update(z=z, kern=kern, c=c, t=t, g=g,
              nframes=timescale.frames_for(seconds, g['frame_s']))


def _one(seed):
    import xspec, units, bandpass
    import fluid as fl
    from fc_score import _rank_z
    z, g, c, t = _W['z'], _W['g'], _W['c'], _W['t']
    nf = _W['nframes']
    A = xspec.realise(z['S'], z['idx'], nf, ref_frames=int(z['ref_frames']), seed=seed)
    Aser = np.repeat(A, g['save'], axis=0)[:nf * g['save']] / g['save']
    fr, _ = fl.run(c, xspec.ProfileDrive(c, g['P'], Aser, 2e-4), g['p'], nf * g['save'],
                   g['save'])
    X = bandpass.apply(units.smooth_frames(fr, _W['kern']), g['frame_s'], g['lo'], g['hi'])
    del fr
    Z, flat = _rank_z(np.ascontiguousarray(X[t.burn:, t.cols].T), rank=(t.metric == 'spearman'))
    e = t.model_edges(Z=Z, flat=flat)[0]
    return np.asarray(e, np.float32), float(t._prep(e) @ t.y)



def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('tags', nargs='+')
    ap.add_argument('--draws', type=int, default=16)
    ap.add_argument('--seconds', type=float, default=577.0)
    ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args()

    import multiprocessing as mp
    from mesh_cache import load_cortex
    import fc_score
    t = fc_score.default_target(load_cortex('fsaverage5', verbose=False), verbose=False)
    marks = [d for d in (1, 2, 4, 8, 16, 32, 64) if d <= a.draws]
    for tag in a.tags:
        t0 = time.time()
        ctx = mp.get_context('spawn')
        with ctx.Pool(a.workers, initializer=_init, initargs=(tag, a.seconds)) as pool:
            esum, singles, out = None, [], []
            for d, (e, s) in enumerate(pool.imap(_one, range(1000, 1000 + a.draws))):
                esum = e.astype(np.float64) if esum is None else esum + e
                singles.append(s)
                if d + 1 in marks:
                    out.append((d + 1, float(t._prep(esum / (d + 1)) @ t.y)))
        print(f"\n  {tag}  [{time.time()-t0:.0f}s]   single draws {np.mean(singles):+.4f} "
              f"+- {np.std(singles):.4f} over {len(singles)}", flush=True)
        print("    pooled over D draws:  " +
              "   ".join(f"D={d} {v:+.4f}" for d, v in out), flush=True)


if __name__ == '__main__':
    main()
