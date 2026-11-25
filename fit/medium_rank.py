"""The medium's own ceiling on the effective rank of the in-band field, before any solve.

The model is linear, so whatever the drive - steady, bursting, switching on and off, cancelling
locally - its effect on FC goes through its cross-spectrum S(f), and the solve searches all of
those. What bounds the structure the FIELD can carry is therefore the transfer H: if every input
reaches the solve vertices through nearly the same few patterns, no combination of inputs gets
out of them. Two numbers, from the run's own impulse responses, in the 0.01-0.08 Hz band, with
the BOLD kernel and the passband applied as in the solve:

  white-input field rank   participation ratio of sum_f w_f H_f H_f^H - the field's effective
                           rank when every input channel is driven by independent equal noise
  best possible rank       the largest participation ratio any input could give: the field's
                           in-band covariance has rank at most min(K, nV), and its eigenvalues
                           are those of H S H^H, so spreading S evenly over H's right singular
                           vectors weighted 1/sigma^2 flattens them - reported from the
                           singular values of the stacked, weighted H

  python fit/medium_rank.py nv2000_mf400 f10_nv2000_mf60
"""
import _path  # noqa: F401
import os, argparse
import numpy as np

from paths import RESULTS
import xspec, bandpass


def pr(ev):
    ev = np.clip(np.asarray(ev, float), 0, None)
    return float(ev.sum() ** 2 / max((ev ** 2).sum(), 1e-300))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('tags', nargs='+')
    a = ap.parse_args()
    from interp_gap import medium_context
    from alias_check import responses_for
    for tag in a.tags:
        z = np.load(os.path.join(RESULTS, f'xspec_{tag}.npz'), allow_pickle=True)
        kern, c, t, g = medium_context(z)
        R, save, fs = responses_for(tag)
        keep, inv = np.unique(np.asarray(t.cols)[g['sub']], return_inverse=True)
        idx = np.asarray(z['idx']); w = np.asarray(z['H_w'], float)
        f_hz = idx / (int(z['pad']) * fs)
        band = np.flatnonzero((f_hz >= g['lo']) & (f_hz <= g['hi']))
        H, _, _ = xspec.transfer(R, inv[:len(g['sub'])], len(idx), kernel=kern, idx=idx[band],
                                 n=max(int(z['pad']), R.shape[1]))
        H = H * bandpass.transfer_response(idx[band], int(z['ref_frames']), fs, g['lo'],
                                           g['hi'])[:, None, None]
        wb = w[band]
        Cw = np.zeros((H.shape[1], H.shape[1]))
        for f in range(len(band)):
            Cw += 2 * wb[f] * np.real(H[f] @ H[f].conj().T)
        ev = np.linalg.eigvalsh(Cw)
        Hs = np.concatenate([np.sqrt(wb[f]) * H[f] for f in range(len(band))], axis=0)
        sv = np.linalg.svd(Hs, compute_uv=False)
        cum = np.cumsum(ev[::-1]) / ev.sum()
        # spatial scale of the slow field: white-input correlation against geodesic distance
        import units
        vsub = np.asarray(t.cols)[g['sub']]
        Dg = units.vertex_geodesic(c, vsub)[:, vsub]
        sd = np.sqrt(np.clip(np.diag(Cw), 1e-300, None)); Rw = Cw / np.outer(sd, sd)
        iu = np.triu_indices(len(vsub), 1); dd, rr = Dg[iu], Rw[iu]
        edges = np.arange(0, 160, 4.0)
        prof = np.array([rr[(dd >= a_) & (dd < a_ + 4)].mean() if ((dd >= a_) & (dd < a_ + 4)).any()
                         else np.nan for a_ in edges])
        below = np.flatnonzero(prof < np.exp(-1))
        ell = edges[below[0]] + 2 if len(below) else np.inf
        print(f"  {tag:<20s} save {save:>3d}: white-input field rank {pr(ev):6.1f}   "
              f"modes for 90%/99% of field variance {np.searchsorted(cum, .9)+1}/"
              f"{np.searchsorted(cum, .99)+1}   input directions visible (PR of sigma^2) "
              f"{pr(sv**2):5.1f} of {H.shape[2]}   slow-field correlation falls to 1/e at "
              f"{ell:.0f} mm  (r at 10/30/60 mm: {prof[2]:+.2f}/{prof[7]:+.2f}/{prof[15]:+.2f})",
              flush=True)
        # slow-band POINT SPREAD: one region driven slowly, its in-band amplitude against
        # distance from its peak vertex. In a fast medium this was a sharp ~8 mm peak on a flat
        # sheet-wide floor, where 1x falls off gradually over 10-100 mm
        amp = np.sqrt(np.einsum('f,fvk->vk', wb, np.abs(H) ** 2))
        src = np.argmax(g['P'], axis=1)
        Dk = units.vertex_geodesic(c, src)[:, vsub]
        hw, pf = [], []
        for k in range(amp.shape[1]):
            aa = amp[:, k] / amp[:, k].max(); dk = Dk[k]
            bb = np.arange(0, 200, 5)
            mm = np.array([aa[(dk >= b) & (dk < b + 5)].mean() if ((dk >= b) & (dk < b + 5)).any()
                           else np.nan for b in bb])
            lo_ = np.flatnonzero(mm < 0.5); hw.append(bb[lo_[0]] + 2.5 if len(lo_) else np.inf)
            pf.append([np.nanmean(mm[(bb >= x - 5) & (bb <= x)]) for x in (10, 30, 60, 100)])
        pf = np.nanmedian(np.array(pf), 0)
        print(f"  {'':20s} slow-band point spread: half-width {np.median(hw):.0f} mm, "
              f"amplitude at 10/30/60/100 mm " + "/".join(f"{v:.2f}" for v in pf), flush=True)


if __name__ == '__main__':
    main()
