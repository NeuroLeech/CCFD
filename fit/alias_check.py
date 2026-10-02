"""Does decimating steps into frames fold a faster medium's content back below Nyquist?

fluid.run saves a SNAPSHOT every `save` steps - `if n % save_every == 0: frames.append(h)` -
with no anti-aliasing. `save` is set by the clock so that a frame is always TR/4 = 0.16125 s,
which means a medium that travels further per step needs more steps per frame: 4 at 1x, 8 at
2x, 12 at 3x. The frame Nyquist stays at 3.1 Hz while the field's content moves up in
frequency, so some of it can fold.

Both sides of the fit see the same decimation - the impulse responses H is built from and the
realisation that gets scored are saved the same way - so the model stays internally
consistent. What is at stake is the SWEEP: if content folds, comparing speeds is comparing
"faster medium" and "more aliasing" together, and part of the measured drop is the sampling.

WHAT THIS CAN AND CANNOT SEE. The cached responses are already decimated, so nothing here can
observe what folded. What it can observe is whether power is still RISING into the Nyquist: a
spectrum that has rolled off to nothing well before 3.1 Hz had nothing above it to fold, and
one still substantial at 3.1 Hz did. That is indicative, not a measurement of the folded
energy; the decisive test re-integrates one piece at save=1 and decimates it by hand.

WHAT IT FOUND, on the three sweep caches (all share Ld31313.8 and c0=1, so the per-step
physics is identical and only `save` and the per-step damping differ):

    medium            save  Nyq Hz  centroid    >1 Hz    >2 Hz  at Nyq / peak
    1x  1.47 mm/s        4   3.101    0.3143    2.00%    0.14%      4.581e-04
    2x  2.95 mm/s        8   3.101    1.1530   47.34%   17.14%      2.202e-01
    3x  4.42 mm/s       12   3.101    1.5468   69.83%   31.62%      5.762e-01

At 1x the response has rolled off before the frame Nyquist and there is nothing to fold. At 2x
it is still at 22% of its peak there, at 3x 58%, so both carry content above 3.1 Hz that the
snapshot decimation folded back.

The jump coincides with the speed sweep's drop. Between 1x and 2x the Nyquist content goes
0.05% -> 22% while solve-space Spearman goes +0.6530 -> +0.4805, and both then change more
slowly out to 3x. So that sweep varies speed AND folded content together from 2x onward, and
attributing its drop to the medium alone does not follow. The frequency-grid control is
unaffected - that was measured at 1x, where there is no folding.

THE FOLDING IS REAL AND IT COSTS ALMOST NOTHING. Re-run at the 3x medium with save=1, which
needs --oversample 48 rather than 40: save must be an integer, so 40 rounds it to 1 and drops
the achieved speed to 3.68 mm/s, while 48 keeps dt at 0.013437 s and the medium bit-identical
to the sweep run at 4.42 mm/s and reach 110.5 mm.

    3x, save 12, 37 passband bins    solve spearman +0.3874
    3x, save  1, 27 passband bins    solve spearman +0.3973

+0.010, and the save=1 run had the COARSER grid of the two, so at matched grid it is perhaps
+0.02. Against the 0.27 the sweep lost from 1x to 3x, removing the decimation entirely recovers
essentially none of it. The responses did carry content above the frame Nyquist and it did fold;
that folded content was not what the solve needed. So 9728014 stands as written: the sweep's
drop is the medium.

That run was killed during its realisation - 42,940 frames through a 329-tap kernel and the
bandpass makes several float64 copies of a 9,374 x 42,940 array - so it wrote no npz and there is
no realised figure or RUNS.md row for it. The solve-space numbers above come from its log.

  python fit/alias_check.py
"""
import _path  # noqa: F401
import os, argparse, glob
import numpy as np

from paths import CACHE, RESULTS
import timescale

SWEEP = [('1x  1.47 mm/s', 'flat100_nolag'), ('2x  2.95 mm/s', 'flat100_s3'),
         ('3x  4.42 mm/s', 'flat100_s4.5')]


def cache_for(tag):
    """The impulse cache the run used, found by its nsteps and save rather than rebuilt."""
    z = np.load(os.path.join(RESULTS, f'xspec_{tag}.npz'), allow_pickle=True)
    save, imp = int(z['save']), int(z['impulse_frames'])
    pat = os.path.join(CACHE, f'impulse_*_100_{imp*save}_{save}_*keep1000*.npy')
    hits = sorted(glob.glob(pat))
    if not hits:
        raise FileNotFoundError(pat)
    return hits[-1], save, imp, float(z['frame_s'])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--pieces', type=int, default=20)
    ap.add_argument('--vertices', type=int, default=200)
    a = ap.parse_args()
    rng = np.random.default_rng(0)

    print(f'\n  mean response power spectrum, {a.pieces} pieces x {a.vertices} vertices')
    print(f'  {"medium":<16s} {"save":>5s} {"Nyq Hz":>7s} {"centroid":>9s} '
          f'{">1 Hz":>8s} {">2 Hz":>8s} {"at Nyq / peak":>14s}')
    for label, tag in SWEEP:
        path, save, imp, frame_s = cache_for(tag)
        R = np.load(path, mmap_mode='r')
        pi = rng.choice(R.shape[0], min(a.pieces, R.shape[0]), replace=False)
        vi = rng.choice(R.shape[2], min(a.vertices, R.shape[2]), replace=False)
        X = np.asarray(R[np.ix_(pi, np.arange(R.shape[1]), vi)], np.float64)
        X = X - X.mean(1, keepdims=True)
        P = (np.abs(np.fft.rfft(X, axis=1)) ** 2).mean(axis=(0, 2))
        f = np.fft.rfftfreq(X.shape[1], frame_s)
        tot = P[1:].sum()
        cen = float((f[1:] * P[1:]).sum() / tot)
        print(f'  {label:<16s} {save:>5d} {f[-1]:>7.3f} {cen:>9.4f} '
              f'{100*P[1:][f[1:] > 1.0].sum()/tot:>7.2f}% '
              f'{100*P[1:][f[1:] > 2.0].sum()/tot:>7.2f}% '
              f'{P[-1]/P[1:].max():>14.3e}')
        del X, R
    print(f'\n  a spectrum still substantial at its Nyquist had content above it to fold;'
          f'\n  one that has rolled off by then did not. This cannot measure what folded.')


if __name__ == '__main__':
    main()
