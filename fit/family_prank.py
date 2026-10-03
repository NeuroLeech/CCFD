"""Hold the fit and spread the input over modes: prank_reg through family_member.

The maxfun ladder showed that optimising concentrates the drive - input participation ratio
89 -> 8.5 of 100 channels - and that a concentrated drive is what makes the expected FC
unrealisable: at maxfun 1500 the expectation-to-one-draw step costs 0.665 where the
400-to-9,310 extrapolation costs 0.047, and the variance floor rises from 139 s to 21,909 s.
So --maxfun has been standing in for a realisability term the objective does not have.

xspec.prank_reg IS that term - at fixed trace, minimising ||S||_F^2 maximises the
participation ratio - and xspec.family_member moves along it while holding the fit within eps
of the argmax. That constrains the cause rather than tuning a stopping point, and unlike a
penalty with a weight it answers a question with units: how much participation ratio is
available at THIS fit.

Takes a recorded solve rather than building its own, so the medium, grid and vertices are the
ones the starting point was solved on. fit/family.py does the same walk but constructs its own
solve from old defaults - pad 1120, and --whiten, which does not work here - so it would be
reporting on a different model.

IT DOES NOT WORK FROM A CONVERGED ARGMAX. Walking from the maxfun-1500 solution (fit +0.8164,
participation 8.5 of 100):

    eps     fit after    participation after    realised (argmax +0.0918)
    0.01     +0.8064          8.5                     -
    0.05     +0.7664          8.7                 +0.0969 +- 0.0122
    0.10     +0.7164          8.8                 +0.1028 +- 0.0109
    0.20     +0.6164          9.2                 +0.1021 +- 0.0106

The fit constraint binds every time and 0.20 of fit buys 0.7 of participation ratio. Against that,
the maxfun ladder reaches participation 54.2 AT fit +0.7162 - better on both axes than anything this
walk finds at any eps. So the early-stopped solution is not a member of the level set near the
argmax; it is in a different region, and family_member is a local walk that cannot cross to it.
EARLY STOPPING IS NOT EQUIVALENT TO REGULARISING THE ARGMAX, which was the assumption that made this
worth trying.

The realised figures above came from a local z-score-and-correlate that read this argmax at
+0.0918 where best_fit recorded +0.1661 for the same S, seconds and seeds. The whole of that
difference was the METRIC: the target's metric is spearman, so fc_score.model_z RANK transforms
each vertex timeseries before the FC, and an inline z-score computes Pearson FC instead. Scoring
through xspec.score_realisation reproduces +0.1661 +- 0.0465 exactly (--no-walk). The eps table has
not been re-run through the fixed path; the argmax row there should be read as +0.1661, and the
comparison it supports - that the walk buys nothing - rests on the participation ratio and the fit,
which were never affected.

  python fit/family_prank.py flat100_fac20 --eps 0.01
"""
import _path  # noqa: F401
import os, time, argparse
import numpy as np

from paths import RESULTS
import xspec, timescale, units, fc_score
import bandpass as bp
from interp_gap import build_H


def participation(S, w, mask=None):
    A = np.zeros((S.shape[1],) * 2, complex)
    for f in (range(len(w)) if mask is None else np.flatnonzero(mask)):
        A += w[f] * S[f]
    ev = np.clip(np.linalg.eigvalsh(0.5 * (A + A.conj().T)).real, 0, None)
    return float(ev.sum() ** 2 / max((ev ** 2).sum(), 1e-300))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('tag')
    ap.add_argument('--eps', type=float, default=0.01,
                    help='how far the fit may fall from the argmax while walking')
    ap.add_argument('--family-iters', type=int, default=200, dest='fiters')
    ap.add_argument('--sign', type=float, default=1.0,
                    help='+1 spreads power over modes, -1 concentrates it')
    ap.add_argument('--seconds', type=float, default=577.0)
    ap.add_argument('--draws', type=int, default=2)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--no-walk', action='store_true', dest='nowalk',
                    help='score the argmax only, which is how this driver is checked '
                         'against the figure best_fit recorded for the same solve')
    a = ap.parse_args()

    z = np.load(os.path.join(RESULTS, f'xspec_{a.tag}.npz'), allow_pickle=True)
    S0 = z['S']
    w = np.asarray(z['H_w'], float)
    ref_fr, seg = int(z['ref_frames']), int(z['segment'])
    Hall, respf, kern, c, t, g = build_H(a.tag, z, False, workers=a.workers)
    idx, sub = g['idx'], g['sub']
    H = np.ascontiguousarray(Hall[idx]); del Hall
    nV = len(sub)
    iu = np.triu_indices(nV, 1)
    raw = np.asarray(t.target_fc()[np.ix_(sub, sub)], np.float64)
    raw = raw - raw.mean(0, keepdims=True) - raw.mean(1, keepdims=True) + raw.mean()
    Tgt = xspec.normal_scores(raw, iu)
    f_hz = idx / (g['pad'] * g['frame_s'])
    inb = (f_hz >= g['lo']) & (f_hz <= g['hi'])

    print(f"  start: participation {participation(S0, w):.1f} of {S0.shape[1]}, "
          f"in band {participation(S0, w, inb):.1f}", flush=True)
    t0 = time.time()
    S = None if a.nowalk else xspec.family_member(H, w, Tgt, S0, xspec.prank_reg(), eps=a.eps,
                            iters=a.fiters, sign=a.sign, verbose=True)
    if S is not None:
        S = S[0] if isinstance(S, tuple) else S
        print(f"  after: participation {participation(S, w):.1f}, "
              f"in band {participation(S, w, inb):.1f}  [{time.time()-t0:.0f}s]", flush=True)

    from scipy.stats import spearmanr
    nframes = timescale.frames_for(a.seconds, g['frame_s'])
    todo = [('argmax', S0)] + ([] if S is None else [('family member', S)])
    for nm, Su in todo:
        C = np.einsum('f,fij->ij', 2.0 * w,
                      np.real((H @ Su) @ np.conj(np.transpose(H, (0, 2, 1)))), optimize=True)
        C = C - C.mean(0, keepdims=True) - C.mean(1, keepdims=True) + C.mean()
        # score_realisation, not a local z-score-and-correlate: the target's metric is
        # Spearman, so fc_score.model_z RANK transforms each vertex before the FC. An
        # inline Pearson version of the same pipeline read this argmax at +0.0918 where
        # best_fit recorded +0.1661, which is the whole of that discrepancy.
        rest = [xspec.realise(Su, idx, nframes, ref_frames=ref_fr, seed=1000 + d)
                for d in range(1, a.draws)]
        pool = res = None
        if a.workers > 1 and rest:
            pool, res = xspec.parallel_scores(c, t, g['p'], rest, g['save'], g['P'], kern,
                                              a.workers, band=(g['lo'], g['hi']),
                                              frame_s=g['frame_s'], segment=seg)
        A0 = xspec.realise(Su, idx, nframes, ref_frames=ref_fr, seed=1000)
        sims = [xspec.score_realisation(c, t, g['p'], A0, save=g['save'], profiles=g['P'],
                                        kernel=kern, band=(g['lo'], g['hi']),
                                        frame_s=g['frame_s'], segment=seg,
                                        diagnostics=False)['sim']]
        if pool is not None:
            sims += [r[0] for r in res.get()]
            pool.close(); pool.join()
        print(f"  {nm:<14s} closed-form spearman vs raw "
              f"{spearmanr(C[iu], raw[iu]).statistic:+.4f}   realised {np.mean(sims):+.4f} "
              f"+- {np.std(sims):.4f}", flush=True)
    if S is None:
        return
    np.savez(os.path.join(RESULTS, f'family_{a.tag}.npz'), S=S, idx=idx, w=w, sub=sub,
             eps=a.eps, sign=a.sign, src=a.tag)


if __name__ == '__main__':
    main()
