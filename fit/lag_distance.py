"""How far does a response have to travel before the OBSERVABLE can see it arrive late?

The passband's upper edge is 0.08 Hz, so its shortest resolvable period is 12.5 s. A wave
covers speed x 12.5 s in that time - 18 mm at 1.47 mm/s, 55 mm at 4.42 mm/s - and below that
distance the travel time is shorter than the observable can resolve, so pairs of vertices look
simultaneous however the drive is arranged. Above it, lag becomes visible. If that is what
costs the faster media their fit, the distance where the curves recover should track
speed / 0.08 Hz.

WHAT IS MEASURED, and what does not work. An energy centroid of the filtered response is
useless here: a bandpassed impulse rings for ~100 s across a 660 s padded window, so the
centroid sits near the window's middle and DECREASES with distance, which is backwards from
propagation - it reports the filter, not the arrival. What survives the filter is the lag of
the CROSS-CORRELATION BETWEEN PAIRS of vertices, since both carry the same filter response and
it largely cancels. So the lag reported here is, for each driven piece and each pair of
vertices, the lag at which their two responses correlate best, binned by the pair's own
geodesic separation.

The filter is applied to the ZERO-PADDED response, by multiplying the rfft by the same
kernel x passband response that xspec.transfer folds into H - a 175 s record cannot carry the
100 s period of a 0.01 Hz filter, and filtering it unpadded would put the filter's own
transient inside the measurement. bandpass.response returns |h|^2, real and non-negative, so
it adds no phase and no delay of its own.

WHAT IT FOUND. Every lag the model produces is below the 12.5 s the passband can resolve, at
every distance and every speed:

                       0-10   10-20   20-30   30-40   40-60   60-80  80-120  120-250
    1x  raw             0.6     1.3     2.1     3.0     4.3     5.9     8.3    12.0
        observable      0.4     0.8     1.5     2.5     4.2     6.0     8.3    11.6
    2x  observable      0.1     0.2     0.5     0.8     1.7     2.9     4.7     6.9
    3x  observable      0.0     0.1     0.2     0.3     0.7     1.4     2.6     4.3

So the premise this file was written to test - that there is a distance d_sync = speed/f_hi below
which pairs look simultaneous and above which lag becomes visible - is wrong as a threshold. The
whole model is inside that regime: the largest lag anywhere is 11.6 s. The passband never sees the
model timing, so every bit of FC structure it carries comes from amplitude and coherence.

The lags are also far below ballistic propagation. At 1x, 40-60 mm would take ~34 s at 1.47 mm/s
and the measured lag is 4.3 s, so the field is not arriving as a travelling front - its covariance
is dominated by a near-instantaneous shared component, consistent with the raw field decorrelating
in 3.8 mm (analysis/spatial_scale.py). In the observable this medium behaves more like locally
filtered noise than like a wave medium.

Beyond the reach - 37 mm at 1x - the response is negligible, so the 80-120 and 120-250 columns are
lags measured on near-zero signal and should not be leaned on.

  python fit/lag_distance.py
  python fit/lag_distance.py --nvert 160 --pieces 12
"""
import _path  # noqa: F401
import os, argparse
import numpy as np

from mesh_cache import load_cortex
from paths import CACHE, RESULTS
import xspec, bo_step, units, timescale, bandpass
import fc_score

SWEEP = [('1x  1.47 mm/s', 'flat100_nolag', 1.47),
         ('2x  2.95 mm/s', 'flat100_s3', 2.95),
         ('3x  4.42 mm/s', 'flat100_s4.5', 4.42)]
EDGES = np.array([0, 10, 20, 30, 40, 60, 80, 120, 250])


def cache_path(z, c, cols):
    """The exact key xspec.impulse_responses would have written, not a glob - nsteps and
    save are not unique across media."""
    save, imp = int(z['save']), int(z['impulse_frames'])
    p, _, _ = bo_step.unpack(np.asarray(z['x'], float), c)
    mc = str(z['map_clip']) if 'map_clip' in z.files else 'none'
    mtag = '' if mc in ('', 'none', 'None') else f'_mc{mc}'
    P = np.asarray(z['profiles'], np.float32)
    key = (f"{c.mesh}_sig{p['sig0']:.6g}_c{p['c0']:.6g}_Ld{p['Ld']:.6g}"
           f"_spg{p.get('sponge_scale', 1.0):.4g}_a{np.round(p.get('a', 0), 3)}"
           f"_b{np.round(p.get('b', 0), 3)}{mtag}_{len(P)}_{imp*save}_{save}"
           f"_{xspec.profile_tag(P)}{xspec.keep_tag(cols)}")
    return os.path.join(CACHE, 'impulse_' + key.replace(' ', '') + '.npy')


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--nvert', type=int, default=400)
    ap.add_argument('--pieces', type=int, default=40)
    ap.add_argument('--band', default='0.01,0.08')
    a = ap.parse_args()
    lo, hi = (float(v) for v in a.band.split(','))
    rng = np.random.default_rng(0)
    c = load_cortex('fsaverage5', verbose=False)
    t = fc_score.default_target(c, verbose=False)

    print("\n  |lag| of best pair cross-correlation, by the pair geodesic separation")
    print(f'  the passband resolves {1/hi:.1f} s at best; '
          f'{"":<4s}' + ''.join(f'{f"{lo_}-{hi_}":>9s}'
                                for lo_, hi_ in zip(EDGES[:-1], EDGES[1:])))
    for label, tag, mms in SWEEP:
        z = np.load(os.path.join(RESULTS, f'xspec_{tag}.npz'), allow_pickle=True)
        sub = np.asarray(z['sub'], np.int64)
        cols = np.asarray(t.cols)[sub]
        path = cache_path(z, c, cols)
        if not os.path.exists(path):
            print(f'  {label}: no cache'); continue
        R = np.load(path, mmap_mode='r')
        K, T, nV = R.shape
        pad, fs = int(z['pad']), float(z['frame_s'])
        P = np.asarray(z['profiles'], np.float32)
        pk = rng.choice(K, min(a.pieces, K), replace=False)
        vi = np.sort(rng.choice(nV, min(a.nvert, nV), replace=False))
        vcort = cols[vi]
        # the observable's response on the padded rfft grid, exactly as transfer folds it in
        nb = pad // 2 + 1
        kern = units.smoothing_kernel(timescale.bold_fwhm_frames(fs, verbose=False),
                                      verbose=False)
        respf = (units.kernel_response(kern, nb, pad)
                 * bandpass.response(np.arange(nb) / (pad * fs), fs, lo, hi))
        tt = np.arange(pad) * fs
        src = np.asarray([int(np.argmax(P[k])) for k in pk])
        # vertex_geodesic returns (len(query), nV_FULL), so the columns index by cortex
        # vertex id, not by position in the sample
        Dsv = units.vertex_geodesic(c, src)[:, vcort]         # (pieces, vertices) mm
        # pairwise cross-correlation lag. C(tau) = sum_t g_v(t) g_w(t+tau) for a band of
        # lags, then argmax over tau per pair. Lags are in FRAMES and every sweep run shares
        # frame_s = 0.16125 s, so they compare directly across speeds.
        LAGS = np.arange(-372, 373, 2)                      # +/- 60 s
        Dvv = units.vertex_geodesic(c, vcort)[:, vcort]
        iu = np.triu_indices(len(vi), 1)
        dpair = Dvv[iu]
        rows = {}
        for nm, filt in (('raw', False), ('observable', True)):
            best = np.full((len(pk), len(iu[0])), np.nan)
            for j, k in enumerate(pk):
                g = np.zeros((pad, len(vi)))
                g[:T] = np.asarray(R[k][:, vi], np.float64)
                if filt:
                    g = np.fft.irfft(np.fft.rfft(g, axis=0) * respf[:, None], n=pad, axis=0)
                g = g - g.mean(0, keepdims=True)
                sd = np.maximum(g.std(0), 1e-300)
                g = g / sd
                peak = np.full(len(iu[0]), -np.inf)
                arg = np.zeros(len(iu[0]))
                for L in LAGS:
                    if L >= 0:
                        A, B = g[:pad - L], g[L:]
                    else:
                        A, B = g[-L:], g[:pad + L]
                    Cm = (A.T @ B) / len(A)
                    val = Cm[iu]
                    up = val > peak
                    peak[up], arg[up] = val[up], L
                best[j] = np.abs(arg) * fs                   # seconds, unsigned
            rows[nm] = best
        print(f'  {label}')
        for nm in ('raw', 'observable'):
            cells = []
            for lo_, hi_ in zip(EDGES[:-1], EDGES[1:]):
                m = (dpair >= lo_) & (dpair < hi_)
                v = rows[nm][:, m]
                cells.append(f'{np.nanmean(v):>9.1f}' if m.sum() > 20 else f'{"-":>9s}')
            print(f'    {nm:<12s}' + ''.join(cells))
        del R
    print(f'\n  latency in seconds. The observable cannot distinguish arrivals closer '
          f'together than {1/hi:.1f} s.')


if __name__ == '__main__':
    main()
