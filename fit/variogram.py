"""Variogram effective range: the target FC's, and the model's slow field's, measured alike.

The spatial scale to give a medium is the one the data shows - the effective range of a
whole-brain variogram (~70 mm). This fits the same model to both sides so the comparison
does not depend on a speed label (spread_mm_s is a frozen label, measured ~6x off the actual
arrival speed, so 'reach = speed x decay' is not a reliable length):

  semivariance   gamma(d) = 1 - mean correlation between vertices d apart (geodesic, 4 mm bins)
  model          gamma = nugget + sill * (1 - exp(-3 d / R)), fitted by least squares;
                 R is the EFFECTIVE range, where gamma reaches 95% of its sill
  target         the empirical FC on the solve vertices (raw, not double-centred)
  model          the white-input slow field: sum_f w_f H_f H_f^H in 0.01-0.08 Hz, BOLD kernel and
                 passband applied - the medium's own spatial scale, independent of any solve
  realised       optionally the FC of a saved realisation (frames_<tag>.npy), on the same vertices

  python fit/variogram.py f1m_nv2000_mf400 f10m_nv2000_mf150 --realised
"""
import _path  # noqa: F401
import os, argparse
import numpy as np

from paths import RESULTS
import xspec, bandpass, units

EDGES = np.arange(0, 200, 4.0)


def profile(R, D):
    iu = np.triu_indices(len(R), 1)
    d, r = D[iu], R[iu]
    mid, gam, n = [], [], []
    for a in EDGES:
        m = (d >= a) & (d < a + 4)
        if m.sum() >= 30:
            mid.append(a + 2); gam.append(1 - r[m].mean()); n.append(m.sum())
    return np.array(mid), np.array(gam), np.array(n)


def fit_range(d, g, n):
    from scipy.optimize import curve_fit
    f = lambda x, nug, sill, R: nug + sill * (1 - np.exp(-3 * x / R))
    p0 = (max(g[0], 0), max(g.max() - g[0], 1e-3), 60.0)
    try:
        (nug, sill, R), _ = curve_fit(f, d, g, p0=p0, sigma=1 / np.sqrt(n),
                                      bounds=([0, 0, 1], [2, 4, 2000]), maxfev=20000)
    except Exception:
        return np.nan, np.nan, np.nan
    return R, nug, sill


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('tags', nargs='+')
    ap.add_argument('--realised', action='store_true')
    a = ap.parse_args()
    from interp_gap import medium_context
    from alias_check import responses_for
    shown = False
    for tag in a.tags:
        z = np.load(os.path.join(RESULTS, f'xspec_{tag}.npz'), allow_pickle=True)
        kern, c, t, g = medium_context(z)
        vsub = np.asarray(t.cols)[g['sub']]
        D = units.vertex_geodesic(c, vsub)[:, vsub]
        if not shown:
            T = np.asarray(t.target_fc()[np.ix_(g['sub'], g['sub'])], np.float64)
            R_, nug, sill = fit_range(*profile(T, D))
            print(f"  target FC (empirical, {len(vsub)} vertices): effective range {R_:6.1f} mm"
                  f"   (nugget {nug:.2f}, sill {sill:.2f})")
            shown = True
        Rr, save, fs = responses_for(tag)
        keep, inv = np.unique(vsub, return_inverse=True)
        idx = np.asarray(z['idx']); w = np.asarray(z['H_w'], float)
        f_hz = idx / (int(z['pad']) * fs)
        band = np.flatnonzero((f_hz >= g['lo']) & (f_hz <= g['hi']))
        H, _, _ = xspec.transfer(Rr, inv, len(idx), kernel=kern, idx=idx[band],
                                 n=max(int(z['pad']), Rr.shape[1]))
        del Rr
        H = H * bandpass.transfer_response(idx[band], int(z['ref_frames']), fs, g['lo'],
                                           g['hi'])[:, None, None]
        Cw = sum(2 * w[band][f] * np.real(H[f] @ H[f].conj().T) for f in range(len(band)))
        sd = np.sqrt(np.diag(Cw)); Rw = Cw / np.outer(sd, sd)
        R_, nug, sill = fit_range(*profile(Rw, D))
        dec = str(z['decimate']) if 'decimate' in z.files else 'snapshot'
        line = (f"  {tag:<22s} ({dec}) white-input slow field: effective range {R_:6.1f} mm"
                f"   (nugget {nug:.2f}, sill {sill:.2f})")
        if a.realised and os.path.exists(os.path.join(RESULTS, f'frames_{tag}.npy')):
            F = np.asarray(np.load(os.path.join(RESULTS, f'frames_{tag}.npy'), mmap_mode='r')
                           [t.burn:, vsub], np.float64)
            Rf = np.corrcoef(F.T)
            Rr_, _, _ = fit_range(*profile(Rf, D))
            line += f";   realised FC: {Rr_:6.1f} mm"
        print(line, flush=True)


if __name__ == '__main__':
    main()
