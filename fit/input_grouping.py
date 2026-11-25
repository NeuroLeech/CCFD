"""How a one-hot solve groups its 1,880 inputs, read without imposing the tiling.

With one channel per driven vertex the solve is free to drive vertices together within the
original regions, across them, or not at all. Which it does is set by the medium and the
bandpassed FC objective, not by the basis - so the grouping is a result about the model.

Two things hide it if the solved input is read directly, and both are handled:

  invisible input   Directions of the input the medium does not pass to the solve vertices are
                    left at the solver's random start (fit/within_region.py). So the in-band
                    input covariance is projected onto the VISIBLE subspace - the leading
                    eigenvectors of the visibility Gram V = sum_f w_f H_f^H H_f over the in-band
                    bins - and only that part is read.
  the medium's own  The visible subspace is smooth, so ANY input projected onto it looks grouped
  grouping          by proximity. The null is the solver's own random start projected the same
                    way: the grouping the medium imposes with no target information. Solved
                    against null is what the bandpass objective adds.

Then, with the tiles and areas used only afterwards to describe what came out:
  - input correlation against geodesic distance, same tile / other tile same area / other area
  - vertices clustered on input correlation, compared with the tiles and areas (ARI); cluster
    sizes, spatial extent, and how many areas each spans
  - the area-by-area mean input correlation: which territories are driven together, which in
    opposition, beyond what the null gives

WHAT IT FOUND, 2026-10-05, oh1880_nv2000_mf150 and _mf400 (the two one-hot solves with a native
factor). The visibility Gram has participation ratio 4.6 of 1,880; 8 directions hold 90% of it,
224 hold 99%, 379 hold 99.9%. Read in that subspace, against the random start as the null, the
same picture at both solves and both cut-offs:

  < 10 mm    the medium's grouping: +0.7 at 0-3 mm, +0.5 at 6-10, solved = null.
  10-25 mm   the solve keeps same-tile coherence and decorrelates across areas - at 15-25 mm,
             same tile +0.20 vs +0.20 null, other tile same area +0.08 vs +0.13, other area
             +0.00 vs +0.12. Strongest in S1 and V1: 3a/3b/2 subfields and V1 gain internal
             coherence, adjacent S1 fields are driven in opposition (3b vs 2, 3a/3b face vs OP1).
  > 25 mm    cross-area groups the null does not have: together A1-area 32 (+0.33..+0.36),
             anterior agranular insula-area 32, presubiculum-V1; opposed A1-V1 (-0.20..-0.41),
             A1-VIP, anterior insula-OFC. Clustering is dominated by one group spanning ~30 of
             38 areas, so the solved clusters match the areas LESS than the null's do (ARI vs
             areas 0.07-0.15 against 0.22-0.33).

Specific long-range pairs shift between the two solves; those listed appear in both. One seed.

  python fit/input_grouping.py oh1880_nv2000_mf400 --keep 0.999
"""
import _path  # noqa: F401
import os, re, argparse
import numpy as np

from paths import RESULTS
import xspec, units, bandpass
from within_region import ari


def visible_basis(z, w, bands):
    from interp_gap import medium_context
    kern, c, t, g = medium_context(z)
    keep, inv = np.unique(np.asarray(t.cols)[g['sub']], return_inverse=True)
    K = xspec.solution_K(z)
    resp = xspec.impulse_responses(c, list(range(K)), g['p'], int(z['impulse_frames']) * g['save'],
                                   g['save'], profiles=g['P'], verbose=False, workers=8, keep=keep)
    idx = np.asarray(z['idx'])[bands]
    H, _, _ = xspec.transfer(resp, inv[:len(g['sub'])], len(idx), kernel=kern, idx=idx,
                             n=max(int(z['pad']), resp.shape[1]))
    del resp
    H = H * bandpass.transfer_response(idx, int(z['ref_frames']), g['frame_s'], g['lo'],
                                       g['hi'])[:, None, None]
    V = np.zeros((K, K), complex)
    for f in range(len(idx)):
        V += w[bands][f] * (H[f].conj().T @ H[f])
    ev, U = np.linalg.eigh(0.5 * (V + V.conj().T))
    return ev[::-1].real, U[:, ::-1], c


def corr(A):
    sd = np.sqrt(np.clip(np.diag(A), 1e-300, None))
    return A / np.outer(sd, sd)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('tag')
    ap.add_argument('--keep', type=float, nargs='+', default=[0.99, 0.999],
                    help='visible subspace = leading eigenvectors holding this share of V')
    a = ap.parse_args()

    z = np.load(os.path.join(RESULTS, f'xspec_{a.tag}.npz'), allow_pickle=True)
    if 'S_L' not in z.files or 'S_compact_note' in z.files:
        raise SystemExit("needs a factor saved by the solve itself, to rebuild its random start")
    L, tr = z['S_L'], float(z['S_tr'])
    nf, K, r = L.shape
    w = np.asarray(z['H_w'], float)
    f_hz = np.asarray(z['idx']) / (int(z['pad']) * float(z['frame_s']))
    lo, hi = (float(v) for v in z['band'])
    bands = np.flatnonzero((f_hz >= lo) & (f_hz <= hi))
    half = nf * K * r
    L0 = np.random.default_rng(0).standard_normal(2 * half) * (1.0 / np.sqrt(K * r))
    L0 = L0[:half].reshape(nf, K, r) + 1j * L0[half:].reshape(nf, K, r)

    ev, U, c = visible_basis(z, w, bands)
    cum = np.cumsum(ev) / ev.sum()
    print(f"  {a.tag}: {len(bands)} in-band bins; visibility Gram participation ratio "
          f"{ev.sum()**2/(ev**2).sum():.1f} of {K}; eigenvectors for "
          + ", ".join(f"{q:.1%}: {int(np.searchsorted(cum, q))+1}" for q in (0.9, 0.99, 0.999)))

    oh = np.load(os.path.join(RESULTS, 'asc_first_onehot_1880.npz'), allow_pickle=True)
    vtx = np.argmax(oh['profiles'], axis=1)
    tl = np.load(os.path.join(RESULTS, 'xspec_asc_first_area_100.npz'), allow_pickle=True)
    tile = np.argmax(tl['profiles'][:, vtx], axis=0)
    names = np.array([re.sub(r'_\d+$', '', str(s)) for s in tl['tags']])[tile]
    anames, area = np.unique(names, return_inverse=True)
    D = units.vertex_geodesic(c, vtx)[:, vtx]

    def cov(LL):
        M = np.concatenate([np.sqrt(w[f]) * LL[f] for f in bands], axis=1)
        return M @ M.conj().T                              # complex, sum_f w_f S_f (in band)

    A_sol, A_ini = cov(L / np.sqrt(tr)), cov(L0)
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    iu = np.triu_indices(K, 1)
    same_t = (tile[:, None] == tile[None, :])[iu]; same_a = (area[:, None] == area[None, :])[iu]
    d = D[iu]
    for q in a.keep:
        k = int(np.searchsorted(cum, q)) + 1
        Q = U[:, :k] @ U[:, :k].conj().T
        print(f"\n  ===== visible subspace: {k} directions ({q:.1%} of the visibility) =====")
        Rs = {}
        for nm, A in (('null (random start)', A_ini), ('solved', A_sol)):
            R = corr(2.0 * np.real(Q @ A @ Q))
            Rs[nm] = R
            print(f"\n    {nm}: input correlation by distance")
            print(f"      {'mm':>7s} {'same tile':>10s} {'other tile, same area':>22s} {'other area':>11s}")
            for b0, b1 in ((0, 3), (3, 6), (6, 10), (10, 15), (15, 25), (25, 50), (50, 300)):
                m = (d >= b0) & (d < b1)
                v = lambda mm: f"{r_[mm].mean():+.3f}" if mm.any() else "-"
                r_ = R[iu]
                print(f"      {b0:>3d}-{b1:<3d} {v(m & same_t):>10s} {v(m & ~same_t & same_a):>22s} "
                      f"{v(m & ~same_a):>11s}")
            Dc = np.clip(1.0 - R, 0, 2); np.fill_diagonal(Dc, 0)
            Z = linkage(squareform(Dc, checks=False), 'average')
            for kk in (38, 100):
                lab = fcluster(Z, kk, 'maxclust')
                sizes = np.bincount(lab)[1:]
                span = [len(np.unique(area[lab == j])) for j in np.unique(lab)]
                ext = [D[np.ix_(lab == j, lab == j)].max() for j in np.unique(lab)]
                print(f"      {kk} clusters: ARI vs tiles {ari(lab, tile):.2f}, vs areas "
                      f"{ari(lab, area):.2f}; size median {int(np.median(sizes))} (max {sizes.max()}); "
                      f"areas spanned median {int(np.median(span))} (max {max(span)}); "
                      f"extent median {np.median(ext):.0f} mm")
        # area x area mean correlation, solved minus null
        na = len(anames)
        def amat(R):
            M = np.zeros((na, na))
            for i in range(na):
                for j in range(na):
                    blk = R[np.ix_(area == i, area == j)]
                    M[i, j] = (blk.sum() - (np.trace(blk) if i == j else 0)) / \
                              max(blk.size - (blk.shape[0] if i == j else 0), 1)
            return M
        Ms, Mn = amat(Rs['solved']), amat(Rs['null (random start)'])
        dmin = np.array([[D[np.ix_(area == i, area == j)].min() for j in range(na)] for i in range(na)])
        print(f"\n    areas driven together or in opposition, solved minus null "
              f"(non-adjacent pairs, closest vertices > 10 mm apart):")
        ii, jj = np.triu_indices(na, 1)
        far = dmin[ii, jj] > 10
        diff = (Ms - Mn)[ii, jj]
        order = np.argsort(diff[far])
        fi, fj, fd = ii[far], jj[far], diff[far]
        for lab_, sel in (('together', order[::-1][:8]), ('in opposition', order[:8])):
            print(f"      {lab_}:")
            for s_ in sel:
                print(f"        {anames[fi[s_]]:<10s} {anames[fj[s_]]:<10s} {fd[s_]:+.3f}  "
                      f"(solved {Ms[fi[s_], fj[s_]]:+.3f}, {dmin[fi[s_], fj[s_]]:.0f} mm apart)")
        dg = np.diag(Ms - Mn)
        print(f"      within-area coherence, solved minus null: median {np.median(dg):+.3f}; "
              f"highest " + ", ".join(f"{anames[i]} {dg[i]:+.3f}" for i in np.argsort(dg)[::-1][:5]))


if __name__ == '__main__':
    main()
