"""Input bases built from the ASCENDING PATHWAYS into cortex, four ways.

Region selection is the same throughout: whole Glasser parcels, grouped by the pathway
that drives them, motor excluded (VA/VL is the cerebellar and pallidal relay, not an
ascending sensory route). FIRST is the first-order set - nuclei whose driving afferents
come from a subcortical source - plus the non-thalamic ascending routes, olfactory and
amygdalocortical. HIGHER adds the higher-order relays, whose driving input is cortical
layer 5.

What differs between the four is how that territory becomes CHANNELS:

  tiles      Glasser picks the territory; inside it the Sereno CsurfMaps1 areas take their
             natural extent, so an area straddling two parcels stays ONE piece. Territory
             with no accepted CsurfMaps1 area falls back to its Glasser parcel (tagged
             G_*); leftovers under MIN_SUB are flood-filled into the nearest piece. The
             amygdalocortical group is never subdivided.
  subdivide  the tiles, cut further to a target COUNT. Which piece to cut next is chosen
             by area (what split_parcels does) or by DISPERSION - the piece's diameter,
             the largest Euclidean distance between any two of its vertices. Area treats a
             compact blob and a long ribbon of the same size as equally urgent; dispersion
             cuts the ribbon first.
  modes      each pathway is ONE region and its channels are the k smallest Laplacian
             eigenmodes on it: smooth, signed, overlapping, no seams. `neumann` (D_sub -
             A_sub) leaves the modes at ~0.43 of peak on the region boundary and makes
             mode 0 a flat indicator; `dirichlet` (D_full - A_sub) charges for every edge
             leaving the region, so the modes vanish into the surround (~0.19) and mode 0
             becomes a smooth interior bump - a tapered whole-region drive that falls out
             of the operator rather than an erosion rule.
  corebelt   each channel is the region's core plus ONE radial sector of its belt, so every
             channel launches from the same centre with a directional lobe. Linearity means
             h_j = h_core + h_belt_j, so handing best_fit the summed profiles is exactly
             equivalent to running core and sector together. NOTE span{core + belt_j} has
             dimension n, one LESS than span{core, belt_1..belt_n}; --core-channel adds the
             core on its own to restore it.

Disconnected pathways are handled per component, and components under MIN_COMP are
DROPPED, not merged: concatenating a stray leaves the subgraph disconnected and the
Laplacian then spends one zero-eigenvalue mode per component telling them apart.

Every mode writes an npz that `best_fit.py --regions hybrid --hybrid-file` reads - either
`labels`+`tags` (profiles made by best_fit, honouring --profile) or `profiles`+`tags`
directly, which signed eigenmodes need because no labelling can express them.

  python fit/ascending_basis.py tiles --out first.npz
  python fit/ascending_basis.py subdivide --in first.npz --count 100 --by dispersion
  python fit/ascending_basis.py modes --modes 10 --bc dirichlet
  python fit/ascending_basis.py corebelt --sectors 8 --core-channel
"""
import _path  # noqa: F401  - puts the sibling code folders on sys.path
import os, argparse
import numpy as np
import scipy.sparse as sps
from scipy.sparse.linalg import eigsh
from scipy.sparse.csgraph import connected_components

from paths import DATA
import subparcels as SP

CSURF = os.path.join(DATA, "atlases", "csurfmaps1", "lh-CsurfMaps1.annot")

FIRST_GROUPS = {
    "LGN_V1":        ["V1"],
    "MGv_A1":        ["A1"],
    "VPL_S1":        ["3a", "3b", "1", "2"],
    "VPI_S2":        ["OP1", "OP2-3", "OP4", "Ig", "PoI1", "PoI2", "PI"],
    "VPMpc_taste":   ["AAIC", "AVI"],
    "olfactory":     ["Pir"],
    "AVAM_retrospl": ["RSC", "ProS", "PreS"],
    "amygdala":      ["OFC", "pOFC", "25", "s32", "EC", "PeEc", "TGv", "TGd"],
}
HIGHER_GROUPS = {
    "pulvinar_extrastriate": ["V2", "V3", "V4", "V3A", "V3B", "LO1", "LO2", "LO3", "V4t",
                              "MT", "MST", "FST", "PIT"],
    "pulvinar_parietal":     ["LIPv", "LIPd", "VIP", "7AL", "IP0", "IP1"],
    "MGdm_belt":             ["MBelt", "LBelt", "PBelt", "A4", "A5", "RI", "52"],
    "MD_lateralPFC":         ["46", "9-46d", "p9-46v", "a9-46v"],
    "MD_FEF":                ["FEF", "8Ad", "8Av"],
    "MD_orbital":            ["47m", "47l", "11l", "13l"],
    "MD_ACC":                ["d32", "p32", "a24"],
    "LD_PCC":                ["31pd", "31pv", "v23ab", "d23ab", "7m", "POS1"],
}
NEVER_SPLIT = {"amygdala"}


def _cortex():
    from mesh_cache import load_cortex
    return load_cortex("fsaverage5", verbose=False)


def _names(c):
    return [str(s).removeprefix("L_").removesuffix("_ROI") for s in c.names]


def _groups(which):
    return dict(FIRST_GROUPS) if which == "first" else {**FIRST_GROUPS, **HIGHER_GROUPS}


def _csurf(c):
    """CsurfMaps1 on the cortex submesh. fsaverage5 is the first 10242 fsaverage vertices."""
    import nibabel as nib
    lab, _, names = nib.freesurfer.read_annot(CSURF)
    names = [n.decode() if isinstance(n, bytes) else n for n in names]
    return lab[:10242][np.asarray(c.old)], names


def _components(c, verts, min_comp):
    """-> (kept components, dropped vertex count)."""
    n, cc = connected_components(SP._subgraph(c, verts), directed=False)
    comps = [verts[cc == i] for i in range(n)]
    ok = [i for i, q in enumerate(comps) if len(q) >= min_comp]
    if not ok:
        ok = [int(np.argmax([len(q) for q in comps]))]
    return [comps[i] for i in ok], sum(len(q) for i, q in enumerate(comps) if i not in ok)


def _diameter(V, q):
    if len(q) < 2:
        return 0.0
    P = V[q]
    return float(np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2).max())


# ---- tiles ---------------------------------------------------------------------------

def tiles(c, which="first", min_sub=20, verbose=True):
    gl, area = np.asarray(c.lab), np.asarray(c.A, float)
    nm = _names(c); ID = {n: i for i, n in enumerate(nm)}
    ser, snames = _csurf(c)
    groups = _groups(which)
    T = np.zeros(c.nV, bool); nosplit = np.zeros(c.nV, bool)
    for g, ps in groups.items():
        for p in ps:
            if p not in ID:
                continue
            m = gl == ID[p]; T |= m
            if g in NEVER_SPLIT:
                nosplit |= m
    piece = np.full(c.nV, -1, np.int64); names = []
    for g, ps in groups.items():                       # amygdalocortical, whole
        if g not in NEVER_SPLIT:
            continue
        for p in ps:
            if p in ID and (gl == ID[p]).any():
                piece[gl == ID[p]] = len(names); names.append(f"G_{p}")
    splittable = T & ~nosplit
    for s in np.unique(ser[splittable]):               # CsurfMaps1 at natural extent
        if s <= 0:
            continue
        m = splittable & (ser == s) & (piece < 0)
        if m.sum() >= min_sub:
            piece[m] = len(names); names.append(snames[s])
    for g, ps in groups.items():                       # Glasser fallback
        if g in NEVER_SPLIT:
            continue
        for p in ps:
            if p not in ID:
                continue
            m = (gl == ID[p]) & (piece < 0)
            if m.sum() >= min_sub:
                piece[m] = len(names); names.append(f"G_{p}")
    E = np.asarray(c.edges)                            # flood-fill the remainder
    left = T & (piece < 0); n0 = int(left.sum())
    while left.any():
        moved = False
        for u, v in ((E[:, 0], E[:, 1]), (E[:, 1], E[:, 0])):
            take = left[u] & (piece[v] >= 0)
            if take.any():
                piece[u[take]] = piece[v[take]]; left = T & (piece < 0); moved = True
        if not moved:
            break
    if verbose:
        print(f"  tiles/{which}: {len(names)} channels, {int(T.sum())} driven vertices, "
              f"{area[T].sum():.0f} mm2 "
              f"({100 * area[T].sum() / area[gl > 0].sum():.1f}% of cortex); "
              f"{n0} flood-filled, {int((T & (piece < 0)).sum())} unassigned")
    return piece, names


# ---- subdivide -----------------------------------------------------------------------

def subdivide(c, labels, tags, count, by="area", min_v=12, min_comp=12, verbose=True):
    area, V, E = np.asarray(c.A, float), np.asarray(c.V, float), np.asarray(c.edges)
    lab0 = np.asarray(labels, np.int64).copy()
    total_moved = 0                                    # strays first - see module docstring
    for _ in range(4):
        moved = 0
        for i in range(len(tags)):
            q = np.flatnonzero(lab0 == i)
            if len(q) < 2:
                continue
            n, comp = connected_components(SP._subgraph(c, q), directed=False)
            if n < 2:
                continue
            for ci in range(n):
                frag = q[comp == ci]
                if len(frag) >= min_comp:
                    continue
                fs = set(frag.tolist()); nb = {}
                for u, v in ((E[:, 0], E[:, 1]), (E[:, 1], E[:, 0])):
                    for a_, b_ in zip(u, v):
                        if a_ in fs and lab0[b_] >= 0 and lab0[b_] != i:
                            nb[lab0[b_]] = nb.get(lab0[b_], 0) + 1
                if nb:
                    lab0[frag] = max(nb, key=nb.get)
                    moved += len(frag); total_moved += len(frag)
        if not moved:
            break
    pieces, origin = [], []
    for i, t in enumerate(tags):
        q = np.flatnonzero(lab0 == i)
        if len(q):
            pieces.append(q); origin.append(t)
    metric = (lambda q: area[q].sum()) if by == "area" else (lambda q: _diameter(V, q))
    while len(pieces) < count:
        order = np.argsort([-metric(q) for q in pieces])
        done = False
        for j in order:
            A, B = SP._bisect(c, pieces[j], area)
            if len(A) >= min_v and len(B) >= min_v:
                t = origin[j]
                pieces[j:j + 1] = [A, B]; origin[j:j + 1] = [t, t]; done = True
                break
        if not done:
            if verbose:
                print(f"  stopped at {len(pieces)}: nothing splits above {min_v} vertices")
            break
    out = -np.ones(c.nV, np.int64); names = list(origin)
    for j, q in enumerate(pieces):
        out[q] = j
    cnt = {}
    for t in names:
        cnt[t] = cnt.get(t, 0) + 1
    seen = {}
    for j, t in enumerate(names):
        if cnt[t] > 1:
            seen[t] = seen.get(t, 0); names[j] = f"{t}_{seen[t]}"; seen[t] += 1
    if verbose:
        dm = [_diameter(V, np.flatnonzero(out == i)) for i in range(len(names))]
        ar = [area[out == i].sum() for i in range(len(names))]
        print(f"  subdivide/{by}: {len(names)} channels, {total_moved} stray vertices "
              f"reassigned; area med {np.median(ar):.0f} max {max(ar):.0f} mm2; "
              f"diameter med {np.median(dm):.1f} max {max(dm):.1f} mm")
    return out, names


# ---- modes ---------------------------------------------------------------------------

def _modes(c, verts, k, bc, deg_full):
    E = np.asarray(c.edges)
    pos = -np.ones(c.nV, np.int64); pos[verts] = np.arange(len(verts))
    keep = (pos[E[:, 0]] >= 0) & (pos[E[:, 1]] >= 0)
    a, b = pos[E[keep, 0]], pos[E[keep, 1]]
    A = sps.coo_matrix((np.ones(keep.sum()), (a, b)), shape=(len(verts),) * 2)
    A = A + A.T
    d = deg_full[verts] if bc == "dirichlet" else np.asarray(A.sum(1)).ravel()
    L = sps.diags(d) - A
    _, V = eigsh(L.astype(float), k=min(k, len(verts) - 1), sigma=-1e-6, which="LM")
    return V * np.sign(V.sum(0))[None, :]


def modes(c, which="first", k=5, bc="neumann", min_comp=12, verbose=True):
    gl, area, E = np.asarray(c.lab), np.asarray(c.A, float), np.asarray(c.edges)
    nm = _names(c); ID = {n: i for i, n in enumerate(nm)}
    deg = np.zeros(c.nV); np.add.at(deg, E[:, 0], 1); np.add.at(deg, E[:, 1], 1)
    rows, tags, driven, dropped = [], [], np.zeros(c.nV, bool), 0
    for g, ps in _groups(which).items():
        v = np.flatnonzero(np.isin(gl, [ID[p] for p in ps if p in ID]))
        if len(v) == 0:
            continue
        keepc, drop = _components(c, v, min_comp); dropped += drop
        for ci, q in enumerate(keepc):
            M = _modes(c, q, k, bc, deg)
            sfx = "" if len(keepc) == 1 else f"c{ci}"
            for j in range(M.shape[1]):
                row = np.zeros(c.nV, np.float32); w = M[:, j]
                row[q] = w / max(abs(w).max(), 1e-30)
                rows.append(row); tags.append(f"{g}{sfx}_m{j}")
            driven[q] = True
    P = np.stack(rows)
    labels = -np.ones(c.nV, np.int64); labels[driven] = 0
    if verbose:
        print(f"  modes/{bc} k={k}: {P.shape[0]} channels, {int(driven.sum())} driven "
              f"vertices, {area[driven].sum():.0f} mm2 "
              f"({100 * area[driven].sum() / area[gl > 0].sum():.1f}%), "
              f"{dropped} stray vertices dropped")
    return P, tags, labels


# ---- core + belt ---------------------------------------------------------------------

def corebelt(c, which="first", sectors=8, core_frac=0.5, min_sect=6, min_comp=12,
             core_channel=False, verbose=True):
    from input2 import _erode_once
    gl, area, V, E = (np.asarray(c.lab), np.asarray(c.A, float), np.asarray(c.V, float),
                      np.asarray(c.edges))
    nb = np.zeros(c.nV); np.add.at(nb, E[:, 0], 1); np.add.at(nb, E[:, 1], 1)
    nm = _names(c); ID = {n: i for i, n in enumerate(nm)}
    rows, tags, driven = [], [], np.zeros(c.nV, bool)
    for g, ps in _groups(which).items():
        v = np.flatnonzero(np.isin(gl, [ID[p] for p in ps if p in ID]))
        if len(v) == 0:
            continue
        keepc, _ = _components(c, v, min_comp)
        for ci, q in enumerate(keepc):
            d = np.zeros(c.nV); m = np.zeros(c.nV, bool); m[q] = True
            cur, kmax = m.copy(), 0
            while cur.any():
                kmax += 1; d[cur] = kmax; cur = _erode_once(cur, E, nb, c.nV)
            u = d[q] / max(kmax, 1); t = u * u * (3.0 - 2.0 * u)
            core, belt = q[u >= core_frac], q[u < core_frac]
            if len(core) == 0:
                core = q[np.argsort(-u)[:max(1, len(q) // 4)]]
                belt = np.setdiff1d(q, core)
            X = V[q] - V[q].mean(0)
            _, _, W = np.linalg.svd(X, full_matrices=False)
            P2 = (V[belt] - V[core].mean(0)) @ W[:2].T
            ang = np.arctan2(P2[:, 1], P2[:, 0])
            ns = int(np.clip(len(belt) // min_sect, 1, sectors))
            sect = np.floor((ang + np.pi) / (2 * np.pi) * ns).astype(int) % ns
            tmap = np.zeros(c.nV); tmap[q] = t
            sfx = "" if len(keepc) == 1 else f"c{ci}"
            if core_channel and ns > 1:      # ns==1 means no belt: that channel IS the core
                row = np.zeros(c.nV, np.float32); row[core] = tmap[core]
                rows.append(row / max(row.max(), 1e-30)); tags.append(f"{g}{sfx}_core")
            for j in range(ns):
                sel = np.concatenate([core, belt[sect == j]])
                row = np.zeros(c.nV, np.float32); row[sel] = tmap[sel]
                rows.append(row / max(row.max(), 1e-30)); tags.append(f"{g}{sfx}_s{j}")
            driven[q] = True
    P = np.stack(rows)
    labels = -np.ones(c.nV, np.int64); labels[driven] = 0
    if verbose:
        ov = (P > 0).sum(0)
        print(f"  corebelt n={sectors}{' +core' if core_channel else ''}: {P.shape[0]} "
              f"channels, {int(driven.sum())} driven vertices, {area[driven].sum():.0f} mm2 "
              f"({100 * area[driven].sum() / area[gl > 0].sum():.1f}%); "
              f"{100 * (ov[driven] > 1).mean():.0f}% of driven vertices in >1 channel")
    return P, tags, labels


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=("tiles", "subdivide", "modes", "corebelt"))
    ap.add_argument("--set", default="first", choices=("first", "all"), dest="which")
    ap.add_argument("--out", default="")
    ap.add_argument("--in", default="", dest="src", help="subdivide: the tiles npz to cut")
    ap.add_argument("--min-sub", type=int, default=20, dest="min_sub")
    ap.add_argument("--count", type=int, default=75)
    ap.add_argument("--by", default="area", choices=("area", "dispersion"))
    ap.add_argument("--modes", type=int, default=5, dest="nmodes")
    ap.add_argument("--bc", default="neumann", choices=("neumann", "dirichlet"))
    ap.add_argument("--sectors", type=int, default=8)
    ap.add_argument("--core-channel", action="store_true", dest="core_channel")
    a = ap.parse_args()
    c = _cortex()
    if a.mode == "tiles":
        labels, tags = tiles(c, a.which, a.min_sub)
        z = dict(labels=labels, tags=np.array(tags, dtype=object))
        out = a.out or f"tiles_{a.which}.npz"
    elif a.mode == "subdivide":
        if not a.src:
            raise SystemExit("  subdivide needs --in (a tiles npz)")
        s = np.load(a.src, allow_pickle=True)
        labels, tags = subdivide(c, s["labels"], [str(v) for v in s["tags"]],
                                 a.count, a.by)
        z = dict(labels=labels, tags=np.array(tags, dtype=object))
        out = a.out or f"sub_{a.which}_{a.by}_{len(tags)}.npz"
    elif a.mode == "modes":
        P, tags, labels = modes(c, a.which, a.nmodes, a.bc)
        z = dict(profiles=P, tags=np.array(tags, dtype=object), labels=labels)
        out = a.out or f"modes_{a.which}_{a.bc}_k{a.nmodes}.npz"
    else:
        P, tags, labels = corebelt(c, a.which, a.sectors, core_channel=a.core_channel)
        z = dict(profiles=P, tags=np.array(tags, dtype=object), labels=labels)
        out = a.out or f"cb_{a.which}_n{a.sectors}{'_core' if a.core_channel else ''}.npz"
    np.savez(out, **z)
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
