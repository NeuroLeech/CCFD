"""Inside the original regions: does a one-hot solve drive each region as one zone?

With one channel per driven vertex (1,880 one-hot), nothing in the basis says which vertices
belong together; any grouping in the solved input is the fit's own. So the 100 tapered tiles
(and the 38 source areas they were cut from) can be laid over the solution and asked about:

  coherence   input correlation between vertices, within the same tile against between tiles,
              at MATCHED geodesic distance - a tile acting as a zone shows as within > between
              at the same separation; a solve that ignores the tiling shows them equal
  modes       per tile, the effective number of independent input patterns inside it
              (participation ratio of its block of the input covariance), the share of its
              variance in the leading pattern, and whether that pattern is single-signed
              (one zone) or changes sign (a gradient or a split)
  taper       per tile, whether input power rises toward the centre (taper-like) or not -
              rank correlation of per-vertex power with distance to the tile's edge
  clusters    vertices clustered on their input correlation at 100 and 38 groups, compared
              with the tiles and the areas by adjusted Rand index

The input statistics are exact: a drawn drive's cross-spectrum IS S, so this reads S rather
than simulating. In band (0.01-0.08 Hz, the bins FC can see) by default.

THE BASELINE THAT MATTERS is the solver's random starting point. Input structure the medium
smooths away is invisible to FC, so the solve has no reason to move it from where it started:
whatever is reported for the solution has to be read against the same numbers at the start.
solve_factor's own initial factor is rebuilt exactly (seed 0) and reported as 'init'.

WHAT IT FOUND, 2026-10-05, on the 1,880 one-hot solves at 2,000 solve vertices, in band:

At the vertex level the solve does NOT drive regions as zones. At maxfun 400 an 18-vertex tile
still carries ~10 independent input patterns (init: 17.7), the leading one holds 23% of the
tile's variance, neighbours within 3 mm correlate +0.10, there is no taper (power vs distance from
the edge rho -0.11), and clustering on input correlation does not recover the tiles (ARI 0.04).
There is a faint boundary: at matched separation, same-tile pairs correlate more than
other-tile pairs (0-3 mm: 0.104 vs 0.078 same area vs 0.067 other area), found without the tiling.

But through the medium (--visibility) a tile has only 1.18 [1.12-1.28] distinguishable input
patterns at the solve vertices, and the most visible is the tile MEAN (overlap 1.00). The random
start already sends 0.89 of a tile's visible output power through its mean; the solve takes that
to 0.92-0.93. So the fit sees every tile as one zone because the medium blurs it to one, not
because the solve chose to, and the vertex-level detail above is mostly the random start left
where it was. Above the passband the input is exactly the random start: at maxfun 400 bins above
0.15 Hz moved 0-1% from it, carrying 87% of the drive's power. The solve shaped the band (59%
change) and the filter's edge just above it (0.08-0.15 Hz, 172%), 13% of the power between them.

This is why 1,880 one-hot channels realise no better than 100 tiles: the medium resolves about
one pattern per tile. Both figures are at this medium's speed and at 2,000 solve vertices (~3
inside a tile); more solve vertices or another medium could resolve more.

CAUTION comparing a stored factor with the random start: only a factor saved by the solve itself
(S_L from best_fit) is in the optimiser's coordinates. A file converted by compact_solve.py holds
an eigendecomposition factor instead, and differs from the start everywhere for that reason alone.

  python fit/within_region.py oh1880_nv2000_mf60 oh1880_nv2000_mf150 oh1880_nv2000_mf400
"""
import _path  # noqa: F401
import os, argparse, re
import numpy as np

from paths import RESULTS
import xspec, units

BINS = (0, 3, 6, 10, 15, 25)


def input_cov(L, w, mask):
    """2 Re sum_f w_f L_f L_f^H over the masked bins, without forming S."""
    M = np.concatenate([np.sqrt(w[f]) * L[f] for f in np.flatnonzero(mask)], axis=1)
    return 2.0 * np.real(M @ M.conj().T)


def ari(a, b):
    from scipy.special import comb
    a = np.unique(a, return_inverse=True)[1]; b = np.unique(b, return_inverse=True)[1]
    ct = np.zeros((a.max() + 1, b.max() + 1)); np.add.at(ct, (a, b), 1)
    s = comb(ct, 2).sum(); sa = comb(ct.sum(1), 2).sum(); sb = comb(ct.sum(0), 2).sum()
    e = sa * sb / comb(len(a), 2)
    return float((s - e) / max(0.5 * (sa + sb) - e, 1e-300))


def describe(A, tile, area, D, edge):
    sd = np.sqrt(np.clip(np.diag(A), 1e-300, None))
    R = A / np.outer(sd, sd)
    iu = np.triu_indices(len(R), 1)
    r, d = R[iu], D[iu]
    same_t = (tile[:, None] == tile[None, :])[iu]
    same_a = (area[:, None] == area[None, :])[iu]
    out = {'bins': []}
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        m = (d >= lo) & (d < hi)
        wt = r[m & same_t]; bt = r[m & ~same_t & same_a]; bo = r[m & ~same_a]
        out['bins'].append((lo, hi, wt.mean() if wt.size else np.nan, wt.size,
                            bt.mean() if bt.size else np.nan, bt.size,
                            bo.mean() if bo.size else np.nan, bo.size))
    prs, lead, single, taper, n = [], [], [], [], []
    from scipy.stats import spearmanr
    for t in np.unique(tile):
        ix = np.flatnonzero(tile == t)
        if len(ix) < 4:
            continue
        B = A[np.ix_(ix, ix)]
        ev, U = np.linalg.eigh(B); ev = np.clip(ev, 0, None)
        prs.append(ev.sum() ** 2 / max((ev ** 2).sum(), 1e-300))
        lead.append(ev[-1] / max(ev.sum(), 1e-300))
        u = U[:, -1]; maj = np.sign(u.sum()) or 1.0
        single.append(np.abs(u[np.sign(u) == maj]).sum() / max(np.abs(u).sum(), 1e-300))
        taper.append(spearmanr(np.diag(B), edge[ix]).statistic)
        n.append(len(ix))
    out.update(pr=np.array(prs), lead=np.array(lead), single=np.array(single),
               taper=np.array(taper), n=np.array(n))
    p = np.diag(A)
    out['power_pr'] = float(p.sum() ** 2 / (p ** 2).sum())
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    Dc = np.clip(1.0 - R, 0, 2); np.fill_diagonal(Dc, 0)
    Z = linkage(squareform(Dc, checks=False), 'average')
    out['ari'] = {k: (ari(fcluster(Z, k, 'maxclust'), tile), ari(fcluster(Z, k, 'maxclust'), area))
                  for k in (100, 38)}
    ev = np.clip(np.linalg.eigvalsh(A), 0, None)
    out['global_pr'] = float(ev.sum() ** 2 / (ev ** 2).sum())
    return out


def report(name, o):
    print(f"\n  {name}")
    print(f"    effective input patterns over all 1,880 vertices: {o['global_pr']:.1f}   "
          f"power spread (PR of per-vertex power): {o['power_pr']:.0f} of 1880")
    print(f"    input correlation by geodesic distance:")
    print(f"      {'mm':>7s} {'same tile':>16s} {'other tile, same area':>24s} {'other area':>16s}")
    for lo, hi, wt, nw, bt, nb, bo, no in o['bins']:
        f = lambda v, k: f"{v:+.3f} ({k:>6d})" if k else f"{'-':>15s}"
        print(f"      {lo:>2d}-{hi:<4d} {f(wt, nw):>16s} {f(bt, nb):>24s} {f(bo, no):>16s}")
    q = lambda a: f"{np.median(a):.2f} [{np.percentile(a, 25):.2f}-{np.percentile(a, 75):.2f}]"
    print(f"    per tile (median [IQR] over {len(o['n'])} tiles, {int(np.median(o['n']))} vertices each):")
    print(f"      independent patterns inside the tile   {q(o['pr'])}")
    print(f"      variance in the leading pattern        {q(o['lead'])}")
    print(f"      leading pattern single-signed (1=yes)  {q(o['single'])}")
    print(f"      power vs distance from edge (rho)      {q(o['taper'])}")
    a = o['ari']
    print(f"    clustering vertices by input correlation, agreement with (ARI):  "
          f"100 groups vs tiles {a[100][0]:.2f}, vs areas {a[100][1]:.2f};  "
          f"38 groups vs tiles {a[38][0]:.2f}, vs areas {a[38][1]:.2f}")


def visibility(z, tile, L, tr, w, mask, label):
    """How much within-tile structure survives the medium, and how the input uses it.

    The fit only sees the input through H. Per tile, the visibility Gram
    V = sum_f w_f H_f[:, tile]^H H_f[:, tile] (in band) says which within-tile input patterns
    reach the solve vertices and how strongly: its participation ratio is the number of
    patterns the fit can tell apart inside the tile, and its leading eigenvector is the most
    visible one. Then, for the solved input and for the random start alike: the share of the
    tile's OUTPUT power that comes through the tile-mean (uniform) pattern alone. A solve that
    drives each tile as a single zone puts that share near 1."""
    from interp_gap import medium_context
    import bandpass
    kern, c, t, g = medium_context(z)
    cols_need = np.asarray(t.cols)[g['sub']]
    keep, inv = np.unique(cols_need, return_inverse=True)
    K = L.shape[1]
    resp = xspec.impulse_responses(c, list(range(K)), g['p'], int(z['impulse_frames']) * g['save'],
                                   g['save'], profiles=g['P'], verbose=False, workers=8,
                                   keep=keep)
    idx = np.asarray(z['idx'])
    bands = np.flatnonzero(mask)
    H, _, _ = xspec.transfer(resp, inv[:len(g['sub'])], len(idx), kernel=kern, idx=idx[bands],
                             n=max(int(z['pad']), resp.shape[1]))
    del resp
    H = H * bandpass.transfer_response(idx[bands], int(z['ref_frames']), g['frame_s'],
                                       g['lo'], g['hi'])[:, None, None]
    wb = w[bands]
    nf, K, r = L.shape
    half = nf * K * r
    L0 = np.random.default_rng(0).standard_normal(2 * half) * (1.0 / np.sqrt(K * r))
    L0 = (L0[:half].reshape(nf, K, r) + 1j * L0[half:].reshape(nf, K, r))[bands]
    Ls = L[bands] / np.sqrt(tr)
    vis_pr, lead_flat, share_sol, share_ini = [], [], [], []
    for tt in np.unique(tile):
        cols = np.flatnonzero(tile == tt); n = len(cols)
        if n < 4:
            continue
        Hc = H[:, :, cols]                                   # (nb, nV, n)
        V = np.einsum('f,fvi,fvj->ij', wb, Hc.conj(), Hc, optimize=True)
        ev, U = np.linalg.eigh(0.5 * (V + V.conj().T)); ev = np.clip(ev.real, 0, None)
        vis_pr.append(ev.sum() ** 2 / max((ev ** 2).sum(), 1e-300))
        u = np.ones(n) / np.sqrt(n)
        lead_flat.append(abs(np.vdot(U[:, -1], u)))
        P_u = np.outer(u, u)
        for LL, out in ((Ls, share_sol), (L0, share_ini)):
            X = Hc @ LL[:, cols, :]                           # output of the tile's own input
            Xu = Hc @ (P_u @ LL[:, cols, :])                  # ...of its uniform part alone
            tot = float(np.einsum('f,fvr->', wb, np.abs(X) ** 2))
            uni = float(np.einsum('f,fvr->', wb, np.abs(Xu) ** 2))
            out.append(uni / max(tot, 1e-300))
    q = lambda a: f"{np.median(a):.2f} [{np.percentile(a, 25):.2f}-{np.percentile(a, 75):.2f}]"
    print(f"\n  {label}: what the fit can see inside a tile (median [IQR] over {len(vis_pr)} tiles)")
    print(f"    within-tile input patterns distinguishable through the medium   {q(vis_pr)}")
    print(f"    most visible pattern's overlap with the tile mean (1 = uniform) {q(lead_flat)}")
    print(f"    share of the tile's output power carried by its uniform part:")
    print(f"      random start {q(share_ini)}      solved {q(share_sol)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('tags', nargs='+')
    ap.add_argument('--all-bins', action='store_true', help='not just the passband')
    ap.add_argument('--visibility', action='store_true',
                    help='also measure what the fit can see inside each tile (needs H)')
    a = ap.parse_args()

    from mesh_cache import load_cortex
    c = load_cortex('fsaverage5', verbose=False)
    oh = np.load(os.path.join(RESULTS, 'asc_first_onehot_1880.npz'), allow_pickle=True)
    vtx = np.argmax(oh['profiles'], axis=1)
    tl = np.load(os.path.join(RESULTS, 'xspec_asc_first_area_100.npz'), allow_pickle=True)
    P = tl['profiles']
    tile = np.argmax(P[:, vtx], axis=0)
    names = [re.sub(r'_\d+$', '', str(s)) for s in tl['tags']]
    area = np.unique(np.array(names)[tile], return_inverse=True)[1]
    Dfull = units.vertex_geodesic(c, vtx)                     # (1880, nV)
    D = Dfull[:, vtx]
    in_tile = P[tile] > 1e-6                                  # (1880, nV): own tile's support
    edge = np.where(in_tile, np.inf, Dfull).min(1)            # distance to nearest non-tile vertex
    print(f"  1,880 driven vertices in {len(np.unique(tile))} tiles / {area.max()+1} areas; "
          f"distance to tile edge median {np.median(edge):.1f} mm")

    shown_init = False
    for tag in a.tags:
        z = np.load(os.path.join(RESULTS, f'xspec_{tag}.npz'), allow_pickle=True)
        L, tr = z['S_L'], float(z['S_tr'])
        w = np.asarray(z['H_w'], float)
        f_hz = np.asarray(z['idx']) / (int(z['pad']) * float(z['frame_s']))
        lo_, hi_ = (float(v) for v in z['band'])
        mask = np.ones(len(w), bool) if a.all_bins else (f_hz >= lo_) & (f_hz <= hi_)
        if not shown_init:
            nf, K, r = L.shape
            half = nf * K * r
            L0 = np.random.default_rng(0).standard_normal(2 * half) * (1.0 / np.sqrt(K * r))
            L0 = L0[:half].reshape(nf, K, r) + 1j * L0[half:].reshape(nf, K, r)
            report(f"init (solve_factor's random start, rank {r})",
                   describe(input_cov(L0, w, mask), tile, area, D, edge))
            shown_init = True
        report(f"{tag}  ({'all bins' if a.all_bins else f'{mask.sum()} in-band bins'})",
               describe(input_cov(L / np.sqrt(tr), w, mask), tile, area, D, edge))
        if a.visibility:
            visibility(z, tile, L, tr, w, mask, tag)


if __name__ == '__main__':
    main()
