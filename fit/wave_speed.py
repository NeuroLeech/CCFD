"""The medium's actual wavefront speed, from impulses integrated at every step.

spread_mm_s is a label: it sets the clock through MM_PER_STEP, which an earlier measurement put
~6x off the arrival speed the medium really has. Anything set from it - "reach = speed x decay"
in particular - inherits that error. This measures the speed directly, the way
units.model_speed does (time-to-peak of |h| at each vertex against white-surface geodesic
distance from the source, least squares over an annulus that excludes the source and the
noise floor), but at EVERY integration step rather than every frame: at 10x a wave can cross
the annulus in about a second, which is only six TR/4 frames.

Time-to-peak fails in a fast medium: the far field's largest response is the sheet-wide floor
that builds after the front has passed, so --front times the first reach of a fraction of each
vertex's own maximum. MEASURED 2026-10-06, 12 pieces, white-surface geodesic, --front 0.2:

    medium              label (spread_mm_s)   front speed     r(distance, arrival)
    1x                        1.47             8.0 mm/s            0.94
    10x, decay 25 s          15.1             80.5 mm/s            0.93
    10x, decay 1.5 s         15.1             82.5 mm/s            0.94

The media differ by exactly 10x and both run ~5.4x faster than labelled (time-to-peak gives
6.5 mm/s at 1x and a meaningless 8.6 at 10x, r 0.47). So true reach is ~5.4x the label's:
1x's 25 s decay carries a front ~200 mm, not 37.

  python fit/wave_speed.py f1m_nv2000_mf400 f10m_nv2000_mf150 --seconds 40 6 --front 0.2
"""
import _path  # noqa: F401
import os, argparse
import numpy as np

from paths import RESULTS


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('tags', nargs='+')
    ap.add_argument('--pieces', type=int, default=12)
    ap.add_argument('--seconds', type=float, nargs='+', default=None,
                    help='integration window per tag (s); default 40 s at 1x-like save, 6 s else')
    ap.add_argument('--front', type=float, default=None,
                    help='time the front: first reach of this fraction of each vertex\'s max')
    ap.add_argument('--min-mm', type=float, default=15.0)
    ap.add_argument('--max-mm', type=float, default=90.0)
    a = ap.parse_args()
    import fluid as fl, units
    from interp_gap import medium_context
    for ti, tag in enumerate(a.tags):
        z = np.load(os.path.join(RESULTS, f'xspec_{tag}.npz'), allow_pickle=True)
        kern, c, t, g = medium_context(z)
        s, dt, gg, Hf = fl.build(c, g['p'], sponge=True)
        secs = (a.seconds[ti] if a.seconds else (40.0 if g['save'] <= 8 else 6.0))
        sps = g['frame_s'] / g['save']          # real seconds per step; fl.build's dt is model time
        nst = int(secs / sps)
        P = g['P']
        rng = np.random.default_rng(0)
        ks = rng.choice(P.shape[0], a.pieces, replace=False)
        src = np.argmax(P[ks], axis=1)
        D = units.vertex_geodesic(c, src)                        # (pieces, nV), white surface
        sp, frac = [], []
        for i, k in enumerate(ks):
            h = P[k].astype(np.float32).copy(); ue = np.zeros(s.nE, np.float32)
            src_mask = P[k] > 1e-6
            hist = np.empty((nst, c.nV), np.float32); hist[0] = np.abs(h)
            for n in range(1, nst):
                ue, h = s.step(ue, h, np.float32(dt), gg, Hf)
                hist[n] = np.abs(h)
            best = hist.max(0)
            if a.front is None:
                tpk = hist.argmax(0) * sps
            else:
                # the FRONT: first step at which |h| reaches `front` of the vertex's own maximum.
                # Time-to-peak times whatever is largest, and in a fast medium the far field's
                # largest response is the sheet-wide floor that builds after the front has passed
                hit = hist >= a.front * best[None, :]
                tpk = np.where(hit.any(0), hit.argmax(0), 0) * sps
            del hist
            d = D[i]
            m = (d > a.min_mm) & (d < a.max_mm) & (tpk > 0) & (tpk < (nst - 2) * sps)
            frac.append(m.mean() / max(((d > a.min_mm) & (d < a.max_mm)).mean(), 1e-9))
            if m.sum() < 50:
                continue
            A = np.c_[np.ones(m.sum()), tpk[m]]
            beta, *_ = np.linalg.lstsq(A, d[m], rcond=None)
            r = np.corrcoef(d[m], tpk[m])[0, 1]
            sp.append((beta[1], r))
        sp = np.array(sp).reshape(-1, 2)
        if not len(sp):
            print(f"  {tag}: no usable pieces"); continue
        lab = float(g['p'].get('c0', np.nan))
        print(f"  {tag:<22s} {sps*1e3:.2f} ms/step, {nst} steps ({secs:.0f} s), save {g['save']}: "
              f"arrival speed {np.median(sp[:,0]):.1f} mm/s (IQR {np.percentile(sp[:,0],25):.1f}-"
              f"{np.percentile(sp[:,0],75):.1f}; r(distance, arrival) median {np.median(sp[:,1]):.2f}; "
              f"{len(sp)} of {a.pieces} pieces)", flush=True)


if __name__ == '__main__':
    main()
