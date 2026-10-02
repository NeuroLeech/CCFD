"""With one channel per driven vertex, the solved input cross-spectrum is a MAP.

Every earlier solve carried S over 100 tiles, so its spatial structure was a 100x100 matrix
of pieces and could only be read through the tiling. With K = 1880 one-hot channels, S_f is
the cross-spectrum between driven VERTICES, so the band-integrated input covariance
sum_f w_f S_f is an object on the cortical surface and its autocorrelation against geodesic
distance is directly comparable to the field's (3.8 mm, analysis/spatial_scale.py) and to the
target FC's own structure on the same vertices.

Reported in correlation form, since the diagonal of S carries each vertex's input power and
the off-diagonal is what says how the drive is organised in space.

WHAT IT FOUND on flat1880_r20 - 1,880 one-hot channels, rank 20 permitted, 700 vertices sampled:

                              effective rank (of 700)   correlation length
      in band 0.01-0.08 Hz              4.1                   5.7 mm
      above the band                    5.8                   7.0 mm
      all bins                          7.1                   6.8 mm

Given a complete basis on the driven territory the solve did NOT build detailed spatial
structure. It chose a smooth input - 5.7 to 7.0 mm, smoother than the field's own 3.8 mm from
analysis/spatial_scale.py - carrying about four effective modes in the band that matters, with
twenty available. The expressive power went into tuning a few smooth modes precisely to the 400
solve vertices rather than into spatial detail, which is the same story the fit tells: solve-space
spearman +0.8468 against the 100-tile baseline's +0.6530, realised +0.3047 against its +0.6104.

The target comparison in this script is not apples to apples and should not be read as one: the
empirical FC arrives double-centred, so putting it in correlation form after setting the diagonal
to 1 removes much of its local structure and its numbers come out far lower than the input's for
reasons that are about the centring rather than about the drive.

  python fit/input_space.py flat1880_r20
"""
import _path  # noqa: F401
import os, argparse
import numpy as np

from mesh_cache import load_cortex
from paths import RESULTS
import units, fc_score

EDGES = np.array([0, 2, 4, 6, 8, 10, 15, 20, 25, 30, 40, 60, 90, 140, 250])


def corr_by_distance(M, D, iu, label, edges=EDGES):
    """Off-diagonal of M in correlation form, binned by geodesic distance."""
    d = np.sqrt(np.clip(np.diag(M).real, 1e-300, None))
    R = np.real(M) / np.outer(d, d)
    r, dd = R[iu], D[iu]
    half, prev = None, None
    print(f'    {label}')
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (dd >= lo) & (dd < hi)
        if m.sum() < 50:
            continue
        v = float(r[m].mean())
        print(f'      {lo:5.0f} - {hi:<4.0f} {int(m.sum()):>8d} {v:>+9.3f}')
        if half is None and prev is not None and prev[1] >= 0.5 > v:
            x0, y0 = prev
            x1 = 0.5 * (lo + hi)
            half = x0 + (y0 - 0.5) * (x1 - x0) / (y0 - v)
        prev = (0.5 * (lo + hi), v)
    if half:
        print(f'      falls to r = 0.5 at {half:.1f} mm')
    return half


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('tag')
    ap.add_argument('--nvert', type=int, default=900)
    a = ap.parse_args()

    z = np.load(os.path.join(RESULTS, f'xspec_{a.tag}.npz'), allow_pickle=True)
    S, idx, w = z['S'], np.asarray(z['idx'], np.int64), np.asarray(z['H_w'], float)
    pad, fs = int(z['pad']), float(z['frame_s'])
    lo_hz, hi_hz = (float(v) for v in z['band'])
    lab = np.asarray(z['labels'], np.int64)
    dv = np.flatnonzero(lab >= 0)
    order = np.argsort(lab[dv])                    # channel k is vertex dv[order][k]
    verts = dv[order]
    K = S.shape[1]
    if len(verts) != K:
        raise SystemExit(f'  {K} channels but {len(verts)} driven vertices - not one-hot')
    f_hz = idx / (pad * fs)
    inb = (f_hz >= lo_hz) & (f_hz <= hi_hz)
    print(f'\n  {a.tag}: {K} one-hot channels, {int(inb.sum())} of {len(idx)} bins in band')

    rng = np.random.default_rng(0)
    sel = np.sort(rng.choice(K, min(a.nvert, K), replace=False))
    c = load_cortex('fsaverage5', verbose=False)
    D = units.vertex_geodesic(c, verts[sel])[:, verts[sel]]
    iu = np.triu_indices(len(sel), 1)

    def band_sum(mask):
        A = np.zeros((len(sel), len(sel)), complex)
        for f in np.flatnonzero(mask):
            A += w[f] * S[f][np.ix_(sel, sel)]
        return A

    print(f'\n  INPUT covariance, correlation form, by geodesic distance between driven '
          f'vertices')
    for nm, mask in (('in band 0.01-0.08 Hz', inb), ('above the band', ~inb),
                     ('all bins', np.ones(len(idx), bool))):
        M = band_sum(mask)
        ev = np.clip(np.linalg.eigvalsh(0.5 * (M + M.conj().T)).real, 0, None)
        pr = (ev.sum() ** 2) / max((ev ** 2).sum(), 1e-300)
        corr_by_distance(M, D, iu, f'{nm}  (effective rank {pr:.1f} of {len(sel)})')

    t = fc_score.default_target(c, verbose=False)
    pos = {int(v): i for i, v in enumerate(t.vertices)}
    keep = np.array([i for i, v in enumerate(verts[sel]) if int(v) in pos])
    if len(keep) > 100:
        ti = np.array([pos[int(v)] for v in verts[sel][keep]])
        T = np.asarray(t.target_fc()[np.ix_(ti, ti)], np.float64)
        np.fill_diagonal(T, 1.0)
        iu2 = np.triu_indices(len(keep), 1)
        print(f'\n  for comparison, the TARGET FC on the same {len(keep)} driven vertices')
        corr_by_distance(T.astype(complex), D[np.ix_(keep, keep)], iu2, 'empirical FC')


if __name__ == '__main__':
    main()
