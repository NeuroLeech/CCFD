"""Split parcels into contiguous sub-regions of roughly equal surface area.

A Glasser parcel is whatever size anatomy made it, so driving V1 and a small auditory
area as single units gives them wildly different spatial extents and wildly different
numbers of degrees of freedom. Splitting to a common area puts them on equal terms: big
parcels become many pieces, small ones stay whole.

Pieces are found by recursive spectral bisection. For a set of vertices, the Fiedler
vector of its own graph Laplacian orders them along the direction of loosest connection;
cutting at the area-weighted median gives two halves of equal area that each stay in one
piece. Repeatedly splitting whichever piece is currently largest lands on any k, not just
powers of two. k-means on coordinates would do neither - it does not respect the surface
graph, and it does not balance area.

  python subparcels.py --total 50
"""
import os, argparse
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from paths import CACHE

SOM = [8, 53, 9, 51]
VIS = [1, 4, 5]
AUD = [24, 173, 174, 124, 104]
SENSORY = SOM + VIS + AUD
DMN = [150, 151, 132, 30, 33, 65, 88]          # the association parcels of the old 19-set

# Parcels grouped by the thalamic nucleus that drives them, from
# AlternateListOfInputs.json. This is an ALTERNATIVE HYPOTHESIS about where patterned
# input enters, not a relaxation of one: `spread` picks parcels by farthest-point
# geometry and is the input prior switched off, whereas this names a mechanism.
#
# LGN -> V1; MGN -> A1; VPL/VPM -> 3b, 1; VA/VL -> 4, 6a, 6mp; pulvinar -> IP1, 7AL, PIT;
# mediodorsal -> 9-46d, 46, d32; limbic/amygdala -> 25, OFC, EC; insular -> AAIC.
#
# Two entries were not Glasser labels and were resolved by hand: "SMA" -> 6mp (medial
# area 6 posterior) and "AI" -> AAIC (anterior agranular insula complex).
#
# It overlaps SENSORY only at V1, 4, 3b, A1, 1. The ten it adds are where the sensory
# model measurably fails - against a whole-cortex mean accuracy of +0.629: PIT +0.350,
# 9-46d +0.394, EC +0.401, OFC +0.408, 25 +0.430, 46 +0.529, d32 +0.546, IP1 +0.619,
# 6a +0.676, 7AL +0.828.
SUBCORTICAL = [1, 24, 9, 51, 8, 96, 55, 145, 42, 22, 86, 84, 62, 164, 93, 118, 112]


def spread_sample(cortex, budget, exclude=(), verbose=True):
    """Parcels spread evenly over the whole cortex, totalling about `budget` mm2.

    Farthest-point sampling on parcel centroids: start from the parcel nearest the
    cortical centroid, then repeatedly take the parcel furthest from everything chosen so
    far, until the area budget is met. The point is a control for SENSORY that holds
    driven area (and so, at a fixed piece size, piece count) fixed while moving the drive
    off sensory cortex."""
    area = np.asarray(cortex.A, float)
    ids = np.array(sorted(set(int(x) for x in np.unique(cortex.lab)) - {0} - set(exclude)))
    cen = np.stack([cortex.V[cortex.lab == p].mean(0) for p in ids])
    a = np.array([area[cortex.lab == p].sum() for p in ids])
    d = np.linalg.norm(cen - cen.mean(0), axis=1)
    chosen = [int(np.argmin(d))]
    dist = np.linalg.norm(cen - cen[chosen[0]], axis=1)
    while a[chosen].sum() < budget and len(chosen) < len(ids):
        i = int(np.argmax(dist))
        chosen.append(i)
        dist = np.minimum(dist, np.linalg.norm(cen - cen[i], axis=1))
    out = sorted(int(ids[i]) for i in chosen)
    if verbose:
        print(f"  spread sample: {len(out)} parcels, {a[chosen].sum():.0f} mm2 "
              f"(budget {budget:.0f})")
    return out


def _subgraph(cortex, verts):
    """Adjacency of `verts` in the cortical mesh, indexed within the set."""
    pos = -np.ones(cortex.nV, np.int64)
    pos[verts] = np.arange(len(verts))
    E = cortex.edges
    keep = (pos[E[:, 0]] >= 0) & (pos[E[:, 1]] >= 0)
    a, b = pos[E[keep, 0]], pos[E[keep, 1]]
    n = len(verts)
    A = sp.coo_matrix((np.ones(len(a)), (a, b)), shape=(n, n))
    return (A + A.T).tocsr()


def _bisect(cortex, verts, area):
    """Split one vertex set in two along its Fiedler vector, balanced by area."""
    A = _subgraph(cortex, verts)
    ncomp, lab = connected_components(A, directed=False)
    if ncomp > 1:                       # already in pieces: peel off the largest
        sizes = np.array([area[verts[lab == i]].sum() for i in range(ncomp)])
        big = np.argmax(sizes)
        return verts[lab == big], verts[lab != big]
    from scipy.sparse.linalg import eigsh
    deg = np.asarray(A.sum(1)).ravel()
    L = sp.diags(deg) - A
    try:
        _, V = eigsh(L.astype(float), k=2, sigma=-1e-6, which="LM")
        f = V[:, 1]
    except Exception:
        f = np.asarray(A.sum(1)).ravel().astype(float)      # fallback: degree order
    o = np.argsort(f)
    w = area[verts[o]]
    cut = np.searchsorted(np.cumsum(w), 0.5 * w.sum())
    cut = int(np.clip(cut, 1, len(verts) - 1))
    return verts[o[:cut]], verts[o[cut:]]


def _absorb_slivers(cortex, pieces, area, floor):
    """Merge pieces below `floor` mm2 into the adjacent piece they touch most."""
    E = cortex.edges
    while len(pieces) > 1:
        sizes = np.array([area[q].sum() for q in pieces])
        i = int(np.argmin(sizes))
        if sizes[i] >= floor:
            break
        owner = -np.ones(cortex.nV, np.int64)
        for j, q in enumerate(pieces):
            owner[q] = j
        touch = np.zeros(len(pieces))
        for a, b in ((E[:, 0], E[:, 1]), (E[:, 1], E[:, 0])):
            m = (owner[a] == i) & (owner[b] >= 0) & (owner[b] != i)
            np.add.at(touch, owner[b][m], 1.0)
        j = int(np.argmax(touch)) if touch.max() > 0 else int(np.argmax(sizes))
        pieces[j] = np.concatenate([pieces[j], pieces[i]])
        pieces.pop(i)
    return pieces


def split_parcels(cortex, parcels=SENSORY, total=50, verbose=True):
    """-> (labels over cortex vertices, list of (parcel, piece) tags).

    Piece count per parcel is its area divided by the common target, so the split is
    driven by size rather than by a fixed number per parcel."""
    key = "-".join(map(str, parcels))
    if len(key) > 120:                  # 70+ parcel ids overflow the 255-byte filename
        import hashlib
        key = f"{len(parcels)}p-{hashlib.sha1(key.encode()).hexdigest()[:12]}"
    cache = os.path.join(CACHE, f"subparcels_{cortex.mesh}_{total}_{key}.npz")
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        return z["labels"], list(z["tags"])

    area = np.asarray(cortex.A, float)
    parcel_area = {p: area[cortex.lab == p].sum() for p in parcels}
    target = sum(parcel_area.values()) / total
    labels = -np.ones(cortex.nV, np.int64)
    tags, nxt = [], 0
    if verbose:
        print(f"  total sensory area {sum(parcel_area.values()):.0f} mm2, "
              f"target {target:.0f} mm2 per piece")
    for p in parcels:
        verts = np.flatnonzero(cortex.lab == p)
        k = max(1, int(round(parcel_area[p] / target)))
        pieces = [verts]
        while len(pieces) < k:
            i = int(np.argmax([area[q].sum() for q in pieces]))
            a, b = _bisect(cortex, pieces[i], area)
            if len(a) == 0 or len(b) == 0:
                break
            pieces[i:i + 1] = [a, b]
        # a bisection can shear off a sliver when the parcel is not simply connected;
        # fold anything well under target into the neighbour it shares most border with
        pieces = _absorb_slivers(cortex, pieces, area, 0.45 * target)
        for j, q in enumerate(pieces):
            labels[q] = nxt
            tags.append(f"{p}_{j}")
            nxt += 1
        if verbose:
            sizes = [area[q].sum() for q in pieces]
            print(f"  parcel {p:3d}: {parcel_area[p]:6.0f} mm2 -> {len(pieces):2d} pieces "
                  f"({min(sizes):.0f}-{max(sizes):.0f} mm2)")
    np.savez(cache, labels=labels, tags=np.array(tags, dtype=object))
    return labels, tags


def region_set(cortex, name, split, scale=1.0):
    """-> (parcel ids, `total` to pass to split_parcels) for one region set.

    `split` is honoured for SENSORY; every other set is given the `total` that reproduces
    the SAME piece area, so the sets differ in where and how much is driven, not in how
    finely it is cut. `scale` widens the 'spread' sample's area budget, which is the one
    way to drive more of the cortex without also changing the piece size."""
    area = np.asarray(cortex.A, float)
    def A(ps):
        return float(sum(area[cortex.lab == p].sum() for p in ps))
    piece = A(SENSORY) / split
    if name == "sensory":
        return SENSORY, split
    if name == "dmn":
        return DMN, int(round(A(DMN) / piece))
    if name == "sensory+dmn":
        ps = SENSORY + DMN
        return ps, int(round(A(ps) / piece))
    if name == "subcortical":
        return SUBCORTICAL, int(round(A(SUBCORTICAL) / piece))
    if name == "subcortical+sensory":
        ps = sorted(set(SUBCORTICAL) | set(SENSORY))
        return ps, int(round(A(ps) / piece))
    ps = spread_sample(cortex, A(SENSORY) * scale, exclude=SENSORY + DMN)
    return ps, int(round(A(ps) / piece))


def gauss_profiles(cortex, labels, n_pieces, fwhm=10.0, cut=1e-3, mask=None,
                   verbose=True):
    """Fixed-width Gaussian profiles centred on each piece's core. -> (n_pieces, nV).

    `taper_profiles` normalises each piece's erosion depth by that piece's OWN maximum, so
    the input's spatial smoothness is a side effect of how finely the parcels were cut: a
    165 mm2 piece falls to half amplitude 6.2 mm from its peak, a 78 mm2 piece at 2.3 mm.
    Changing the piece count therefore changes two things at once, which makes a piece-count
    sweep uninterpretable. It also leaves the profiles DISJOINT and tapering to zero at
    every border, so the total drive is a honeycomb - 24.6% of the driven area gets under
    half the peak, and a spatially uniform input over a driven region cannot be expressed
    at all.

    Here the width is a parameter in millimetres and the same for every piece, so piece
    count controls only how many channels there are. Profiles overlap, which removes the
    seams, and they are free to extend past the parcel boundary - a thalamic projection
    does not stop at an atlas edge. `mask` restricts them if that is not wanted.

    Distance is white-surface geodesic, so `fwhm` is real millimetres."""
    import units
    sigma = float(fwhm) / 2.3548
    cen = []
    for i in range(n_pieces):
        v = np.flatnonzero(labels == i)
        if not len(v):
            cen.append(0)
            continue
        # the piece's core: the vertex furthest from its own border, by erosion depth
        from input2 import _erode_once
        nb = np.zeros(cortex.nV)
        np.add.at(nb, cortex.edges[:, 0], 1)
        np.add.at(nb, cortex.edges[:, 1], 1)
        depth = np.zeros(cortex.nV)
        cur, k = (labels == i), 0
        while cur.any():
            k += 1
            depth[cur] = k
            cur = _erode_once(cur, cortex.edges, nb, cortex.nV)
        cen.append(int(v[np.argmax(depth[v])]))
    D = units.vertex_geodesic(cortex, np.asarray(cen, int))      # (n_pieces, nV), mm
    P = np.exp(-0.5 * (D / max(sigma, 1e-9)) ** 2).astype(np.float32)
    P[P < cut] = 0.0
    if mask is not None:
        P = P * np.asarray(mask, np.float32)[None, :]
    if verbose:
        tot = P.sum(0)
        on = tot > cut
        area = np.asarray(cortex.A, float)
        print(f"  gaussian profiles: FWHM {fwhm:g} mm, {n_pieces} pieces, "
              f"{int(on.sum())} vertices touched ({area[on].sum():.0f} mm2)")
        print(f"    total drive where touched: min {tot[on].min():.3f}, "
              f"mean {tot[on].mean():.3f}, max {tot[on].max():.3f}")
    return P


def taper_profiles(cortex, labels, n_pieces):
    """Smooth profile per piece: 1 at its core, 0 at its border, same recipe as
    input2.parcel_tapers but for arbitrary vertex sets."""
    from input2 import _erode_once
    nb = np.zeros(cortex.nV)
    np.add.at(nb, cortex.edges[:, 0], 1)
    np.add.at(nb, cortex.edges[:, 1], 1)
    P = np.zeros((n_pieces, cortex.nV), np.float32)
    for i in range(n_pieces):
        mask = labels == i
        depth = np.zeros(cortex.nV)
        cur, k = mask.copy(), 0
        while cur.any():
            k += 1
            depth[cur] = k
            cur = _erode_once(cur, cortex.edges, nb, cortex.nV)
        if k:
            u = depth / depth.max()
            P[i] = (u * u * (3.0 - 2.0 * u)).astype(np.float32)
        else:
            P[i] = mask.astype(np.float32)
    return P


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--total", type=int, default=50)
    ap.add_argument("--figure", action="store_true", default=True)
    a = ap.parse_args()

    from mesh_cache import load_cortex
    c = load_cortex("fsaverage5", verbose=False)
    labels, tags = split_parcels(c, SENSORY, a.total)
    n = len(tags)
    area = np.asarray(c.A, float)
    sizes = np.array([area[labels == i].sum() for i in range(n)])
    print(f"\n  {n} pieces, area {sizes.min():.0f}-{sizes.max():.0f} mm2 "
          f"(mean {sizes.mean():.0f}, sd {sizes.std():.0f}), "
          f"{int((labels >= 0).sum())} vertices covered")

    if a.figure:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.tri import Triangulation
        from render_regimes import _proj
        from paths import RESULTS
        proj = _proj(c.V, c.F)
        rng = np.random.default_rng(0)
        shuffle = rng.permutation(n)
        v = np.where(labels >= 0, shuffle[np.clip(labels, 0, None)], np.nan)
        fig = plt.figure(figsize=(3.6 * len(proj), 2.6))
        gs = fig.add_gridspec(1, len(proj), wspace=0.02)
        for k, (xy, vis, nm) in enumerate(proj):
            ax = fig.add_subplot(gs[0, k])
            keep = vis[c.F].all(1)
            ax.tripcolor(Triangulation(xy[:, 0], xy[:, 1], c.F[keep]),
                         np.nan_to_num(v, nan=-1), shading="gouraud", cmap="tab20",
                         vmin=-1, vmax=n, rasterized=True)
            ax.set_xlim(xy[:, 0].min(), xy[:, 0].max())
            ax.set_ylim(xy[:, 1].min(), xy[:, 1].max())
            ax.set_aspect("equal"); ax.axis("off"); ax.set_title(nm, fontsize=10)
        path = os.path.join(RESULTS, f"subparcels_{a.total}.png")
        fig.savefig(path, dpi=130, bbox_inches="tight")
        print(f"  wrote {path}")
