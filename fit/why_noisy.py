"""Why is one draw of the envelope so much worse than one draw of the linear observable?

Both are bandpassed to 0.01-0.08 Hz, so the obvious expectation is that both have the same
number of independent samples in a given run and should be estimated equally well. This
tests that expectation instead of assuming it, by computing BOTH observables from ONE
simulation of ONE drive - same field, same realisation, the only difference being the
square - and measuring, per observable:

  the in-band SPECTRAL SHAPE        where inside 0.01-0.08 Hz the power actually sits
  the ESTIMATOR NOISE              block-split sd of the covariance estimate, per edge
  the PATTERN CONTRAST             sd across edges of the correlation-form off-diagonals
  the KURTOSIS                     how far from Gaussian each observable is

The ratio (estimator noise / pattern contrast) is what sets the variance floor T0, so
splitting it into those two factors says which one the envelope pays, and the spectral
shape says whether the two have the same N to begin with.

  python fit/why_noisy.py env_grclip400 --seconds 2308
"""
import _path  # noqa: F401
import os, time, argparse
import numpy as np
from scipy.stats import kurtosis

from paths import RESULTS
import xspec, timescale, units, bandpass as bp
from interp_gap import build_H


def efficiency(X, lo, hi, frame_s, name, nblock=16):
    """-> the numbers above for one (T, V) observable."""
    T, V = X.shape
    X = np.asarray(X, np.float64)
    X = X - X.mean(0, keepdims=True)
    f = np.fft.rfftfreq(T, frame_s)
    P = (np.abs(np.fft.rfft(X, axis=0)) ** 2).mean(1)
    inb = (f >= lo) & (f <= hi)
    cen = float((f[inb] * P[inb]).sum() / max(P[inb].sum(), 1e-300))
    lowhalf = float(P[inb & (f < 0.5 * (lo + hi))].sum() / max(P[inb].sum(), 1e-300))

    # the pattern: correlation-form off-diagonals of the FULL-record covariance
    C = np.cov(X, rowvar=False)
    d = np.sqrt(np.clip(np.diag(C), 1e-300, None))
    R = C / np.outer(d, d)
    iu = np.triu_indices(V, 1)
    contrast = float(R[iu].std())

    # the noise: the SAME quantity estimated on each block, so its scatter across blocks is
    # the estimator's own sd at T/nblock, scaled to the full record by sqrt(nblock)
    nb = T // nblock
    Rb = np.empty((nblock, len(iu[0])))
    for k in range(nblock):
        Y = X[k * nb:(k + 1) * nb]
        Ck = np.cov(Y, rowvar=False)
        dk = np.sqrt(np.clip(np.diag(Ck), 1e-300, None))
        Rb[k] = (Ck / np.outer(dk, dk))[iu]
    noise = float(Rb.std(0, ddof=1).mean() / np.sqrt(nblock))
    kur = float(np.median(kurtosis(X, axis=0, fisher=True)))
    print(f'\n  {name}')
    print(f'    in-band power centroid {cen:.4f} Hz, {100*lowhalf:.1f}% of it below '
          f'{0.5*(lo+hi):.3f} Hz')
    print(f'    pattern contrast  sd(offdiag) {contrast:.4f}')
    print(f'    estimator noise   sd per edge {noise:.4f}  (16 blocks, scaled to the '
          f'full {T} frames)')
    print(f'    noise / contrast  {noise/max(contrast,1e-300):.4f}')
    print(f'    excess kurtosis   {kur:+.3f} (median over vertices)')
    return dict(cen=cen, contrast=contrast, noise=noise, kur=kur,
                ratio=noise / max(contrast, 1e-300))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('tag')
    ap.add_argument('--seconds', type=float, default=2308.0)
    ap.add_argument('--seed', type=int, default=1000)
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()

    z = np.load(os.path.join(RESULTS, f'envfit_{a.tag}.npz'), allow_pickle=True)
    S = xspec.load_S(z)
    H, respf, kern, c, t, g = build_H(a.tag, z, True, workers=a.workers)
    del H                                    # only the medium and the grid facts are wanted
    lo, hi, frame_s, save = g['lo'], g['hi'], g['frame_s'], g['save']

    import fluid as fl
    from xspec import ProfileDrive
    nframes = timescale.frames_for(a.seconds, frame_s)
    print(f'  simulating {nframes} frames ({a.seconds:.0f}s) of the solved drive',
          flush=True)
    t0 = time.time()
    Af = xspec.realise(S, g['idx'], nframes, ref_frames=g['pad'], seed=a.seed)
    Aser = np.repeat(Af, save, axis=0)[:nframes * save] / save
    fr, _ = fl.run(c, ProfileDrive(c, g['P'], Aser, 2e-4), g['p'], nframes * save, save)
    Fr = np.asarray(fr[:, np.asarray(t.cols)[g['sub']]], np.float64)
    del fr
    print(f'    [{time.time()-t0:.0f}s]  {Fr.shape[0]} frames x {Fr.shape[1]} vertices',
          flush=True)

    sm = units.smooth_frames(Fr, kern)
    lin = bp.apply(sm, frame_s, lo, hi)[t.burn:]
    envl = bp.apply(units.smooth_frames(Fr ** 2, kern), frame_s, lo, hi)[t.burn:]
    del Fr, sm
    L = efficiency(lin, lo, hi, frame_s, 'LINEAR    bandpass(smooth(h))')
    E = efficiency(envl, lo, hi, frame_s, 'ENVELOPE  bandpass(smooth(h^2))')
    print(f'\n  noise/contrast ratio: envelope is {E["ratio"]/L["ratio"]:.2f}x the linear')
    print(f'    of which contrast  {L["contrast"]/E["contrast"]:.2f}x'
          f'   and noise  {E["noise"]/L["noise"]:.2f}x')
    print(f'  T0 scales as that ratio squared -> {(E["ratio"]/L["ratio"])**2:.0f}x,'
          f' against the measured 12713/145 = {12713/145:.0f}x')


if __name__ == '__main__':
    main()
