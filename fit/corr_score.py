"""Realise a corr_solve result and score it the way every recorded run is scored.

The point of the correlation-form objective is a more empirically realistic model, and
"more realistic" has to survive the actual scoring path: simulate the drive, build the model
FC the way fc_score does - z-scored, rank-transformed timecourses - and take the Spearman
against the empirical FC over all 9,310 vertices. J and the solve-vertex spread are what the
objective optimised; this is the number that is comparable with RUNS.md.

Reports the SOURCE solve's realised score from the same seeds, so the pair differs only in
which objective produced S.

  python fit/corr_score.py corrfit_lin400_lam10 --seconds 2308 --draws 2
"""
import _path  # noqa: F401
import os, time, argparse
import numpy as np
from scipy.stats import spearmanr

from paths import RESULTS
import xspec, timescale, units, bandpass as bp
import envelope as env
import corrfit as cf
from interp_gap import build_H


def realise_score(c, t, g, S, idx, kern, envelope, seconds, draws, label, save_tag=''):
    """-> list of Spearman scores on all target vertices, one per draw."""
    import fluid as fl
    from xspec import ProfileDrive
    save, P, p = g['save'], g['P'], g['p']
    nframes = timescale.frames_for(seconds, g['frame_s'])
    cols = np.asarray(t.cols)
    out, esum = [], None
    for d in range(draws):
        t0 = time.time()
        Af = xspec.realise(S, idx, nframes, ref_frames=g['pad'], seed=1000 + d)
        Aser = np.repeat(Af, save, axis=0)[:nframes * save] / save
        fr, _ = fl.run(c, ProfileDrive(c, P, Aser, 2e-4), p, nframes * save, save)
        X = np.asarray(fr[:, cols], np.float64)
        if save_tag and d == 0:
            # the WHOLE sheet for the video, not just the target's columns, and the bare
            # field with no burn trimmed so the two arrays align frame for frame. The
            # observable is saved alongside it so render_bandpass's reconstruction check
            # has something to compare against.
            full = np.asarray(fr, np.float32)
            obs = (env.observable(full, kern, g['frame_s'], (g['lo'], g['hi']), burn=0)
                   if envelope else
                   bp.apply(units.smooth_frames(np.asarray(full, np.float64), kern),
                            g['frame_s'], g['lo'], g['hi']))
            np.save(os.path.join(RESULTS, f'frames_{save_tag}_raw.npy'), full)
            np.save(os.path.join(RESULTS, f'frames_{save_tag}.npy'),
                    np.asarray(obs, np.float32))
            np.save(os.path.join(RESULTS, f'drive_{save_tag}.npy'), Aser)
            print(f'    wrote frames_{save_tag}_raw.npy, frames_{save_tag}.npy and '
                  f'drive_{save_tag}.npy ({full.shape[0]} frames x {full.shape[1]} '
                  f'vertices, {"envelope" if envelope else "linear"} observable)',
                  flush=True)
            del full, obs
        del fr
        if envelope:
            U = env.observable(X, kern, g['frame_s'], (g['lo'], g['hi']), burn=t.burn)
        else:
            U = bp.apply(units.smooth_frames(X, kern), g['frame_s'],
                         g['lo'], g['hi'])[t.burn:]
        del X
        Z = U - U.mean(0, keepdims=True)
        Z = (Z / np.maximum(Z.std(0, keepdims=True), 1e-300)).T
        ev = t.model_edges(Z=Z)[0]
        out.append(float(t._prep(ev) @ t.y))
        # POOL THE EDGE VECTOR, not the scores. Each draw's model FC estimates the same
        # population FC, so averaging the 2M-edge vectors is the all-vertex equivalent of
        # pooling covariances - 16 MB a draw. Averaging the SCORES instead leaves every
        # estimate at the single-draw attenuation and only shrinks its error bar, which is
        # what made the envelope's +0.3416 look like a converged number when it was not.
        esum = np.asarray(ev, np.float64) if esum is None else esum + ev
        pooled = float(t._prep(esum / (d + 1)) @ t.y)
        print(f'    {label} draw {d}: sim {out[-1]:+.4f}   pooled over {d+1} '
              f'{pooled:+.4f}   [{time.time()-t0:.0f}s]', flush=True)
    return out, pooled


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('tag', help='a corrfit_<...>.npz written by corr_solve')
    ap.add_argument('--seconds', type=float, default=2308.0)
    ap.add_argument('--draws', type=int, default=2)
    ap.add_argument('--skip-source', action='store_true',
                    help='do not re-score the source solve (its number is already recorded)')
    ap.add_argument('--save-frames', default='', dest='save_frames',
                    help='write frames_<tag>_raw.npy, frames_<tag>.npy and drive_<tag>.npy '
                         'from draw 0, which is what viz/render_bandpass.py reads')
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()

    q = np.load(os.path.join(RESULTS, a.tag + '.npz'), allow_pickle=True)
    src, isenv = str(q['src']), bool(q['envelope'])
    z = np.load(os.path.join(RESULTS, ('envfit_' if isenv else 'xspec_') + src + '.npz'),
                allow_pickle=True)
    Hall, respf, kern, c, t, g = build_H(src, z, isenv, workers=a.workers)
    del Hall
    idx = g['idx']
    print(f'  {a.tag}: from {src}, lam {float(q["lam"]):g}, '
          f'spread {float(q["spread"])/float(q["target_spread"]):.2f}x the target, '
          f'J {float(q["J"]):.4e}', flush=True)

    new, newp = realise_score(c, t, g, q['S'], idx, kern, isenv, a.seconds, a.draws, 'NEW',
                              save_tag=a.save_frames)
    old, oldp = (([], None) if a.skip_source else
                 realise_score(c, t, g, xspec.load_S(z), idx, kern, isenv, a.seconds, a.draws, 'OLD'))
    print(f'\n  over {a.seconds:.0f}s, {a.draws} draws, all {t.nV} vertices:')
    print(f'    correlation-form objective  per draw {np.mean(new):+.4f} +- {np.std(new):.4f}'
          f'   POOLED {newp:+.4f}')
    if old:
        print(f'    the recorded solve          per draw {np.mean(old):+.4f} +- {np.std(old):.4f}'
              f'   POOLED {oldp:+.4f}')
    np.savez(os.path.join(RESULTS, f'corrscore_{a.tag}.npz'), new=new, old=old,
             new_pooled=newp, old_pooled=(oldp if oldp is not None else np.nan),
             seconds=a.seconds, draws=a.draws, tag=a.tag, src=src)


if __name__ == '__main__':
    main()
