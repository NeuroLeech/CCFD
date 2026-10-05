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
        print(f"  {tag:<20s} save {save:>3d}: white-input field rank {pr(ev):6.1f}   "
              f"modes for 90%/99% of field variance {np.searchsorted(cum, .9)+1}/"
              f"{np.searchsorted(cum, .99)+1}   input directions visible (PR of sigma^2) "
              f"{pr(sv**2):5.1f} of {H.shape[2]}", flush=True)


if __name__ == '__main__':
    main()
