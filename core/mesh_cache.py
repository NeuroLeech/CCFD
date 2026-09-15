"""Build and cache a cortical fluid mesh: surface + Glasser atlas + repaired metric.

The intrinsic-Delaunay repair is what makes the C-grid shallow-water scheme stable;
without it ~6% of dual edge lengths are negative and the scheme blows up in ~200
steps. The repair changes only connectivity, not vertex positions.

That repaired connectivity is the mesh the SOLVER runs on, and it is not `m`. `m` is the
extrinsic triangulation the surface arrived with; the solve uses `edges`, `d`, `l` and `A`,
all of which come from the flipped triangulation F2 with its intrinsic lengths. Anything
needing FACES of the integrated mesh - circulation round a triangle, so relative vorticity,
so momentum advection - has to use `F2`, and building it from `m.F` silently uses a
different triangulation. So F2 is carried on the Cortex and cached alongside the rest.

F2 is re-oriented before it is stored. `intrinsic_delaunay` writes F[t1] = [k, l, j] and
F[t2] = [l, k, i], which reverses the winding of both triangles in every flip: as returned
it has 1,989 directed edges traversed twice the same way and 13.2% of faces wound against
the surface normal. Winding reaches nothing else - cotan_from_lengths keys every length
through `key()`, which sorts, and the boundary count uses min/max - so re-orienting changes
no solver value, and it is checked rather than assumed (see `_orient`).
"""
import _path  # noqa: F401  - puts the sibling code folders on sys.path
import os, time
import numpy as np
from collections import defaultdict
from surf_ops import SurfaceMesh, load_fsaverage, load_glasser, submesh
from intrinsic_delaunay import intrinsic_delaunay, cotan_from_lengths

from paths import ANNOT, CACHE
HERE = os.path.dirname(os.path.abspath(__file__))


class Cortex:
    """Everything the solver and the input model need, for one surface resolution."""
    __slots__ = ("mesh", "m", "V", "F", "lab", "names", "old",
                 "edges", "d", "l", "A", "bnd", "nV", "parcels", "F2", "tri_area")

    def __repr__(self):
        return (f"<Cortex {self.mesh}: {self.nV} fluid vertices, {len(self.edges)} edges, "
                f"{len(self.parcels)} parcels>")


def _orient(F2, Vc, vertex_normal):
    """Wind every face of F2 the same way round, positively about the surface normal.

    Verified combinatorially, not trusted: in a consistently oriented triangulation no
    directed edge is traversed twice in the same direction, and that count has to be 0
    afterwards. On fsaverage5 this rule and an independent BFS propagation of orientation
    across the dual graph agree on 100% of faces, and the smallest face |cross| is 1.6, so
    no face is near-degenerate and the normal test is safe."""
    P = Vc[F2]
    nrm = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
    neg = (nrm * vertex_normal[F2].mean(1)).sum(1) < 0
    out = F2.copy()
    out[neg] = out[neg][:, [0, 2, 1]]
    seen = defaultdict(int)
    for t in out:
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            seen[(int(a), int(b))] += 1
    bad = sum(1 for v in seen.values() if v > 1)
    if bad:
        raise RuntimeError(f"F2 is not consistently oriented: {bad} directed edges "
                           f"traversed twice the same way")
    return out


def _heron(F2, edges, d):
    """Intrinsic area of each face of F2, from the repaired edge lengths.

    Heron on Lg, not 0.5|cross| on the vertex positions: a flipped edge joins vertices
    that are not adjacent on the surface, so its triangle is not the flat one spanned by
    those three points, and the intrinsic length is the one the solver's metric uses."""
    eidx = {(int(a), int(b)): n for n, (a, b) in enumerate(edges)}
    kk = lambda i, j: (i, j) if i < j else (j, i)
    L = np.empty((len(F2), 3))
    for c_, (p, q) in enumerate(((1, 2), (2, 0), (0, 1))):
        L[:, c_] = [d[eidx[kk(int(t[p]), int(t[q]))]] for t in F2]
    a, b, c = L[:, 0], L[:, 1], L[:, 2]
    sN = 0.5 * (a + b + c)
    return np.sqrt(np.maximum(sN * (sN - a) * (sN - b) * (sN - c), 0.0))


def load_cortex(mesh="fsaverage5", annot=ANNOT, verbose=True):
    cache = os.path.join(CACHE, f"cortex_{mesh}.npz")
    V, F = load_fsaverage(mesh, "infl_left")
    lab, names = load_glasser(annot, len(V))
    active = lab > 0                       # label 0 is '???' = medial wall
    Vc, Fc, old = submesh(V, F, active)
    m = SurfaceMesh(Vc, Fc)

    def build():
        L0 = {(int(a), int(b)): float(np.linalg.norm(Vc[a] - Vc[b])) for a, b in m.E}
        F2, Lg, nflip = intrinsic_delaunay(Fc.tolist(), L0)
        edges, d, w, areav = cotan_from_lengths(F2, Lg)
        l = w * d
        A = np.zeros(m.nV)
        for k, v in areav.items():
            A[k] = v
        cnt = defaultdict(int)
        for t in F2:
            for a, b in ((t[1], t[2]), (t[2], t[0]), (t[0], t[1])):
                cnt[(min(a, b), max(a, b))] += 1
        eidx = {tuple(e): n for n, e in enumerate(edges)}
        bnd = np.zeros(len(edges), bool)
        for e, c in cnt.items():
            if c == 1:
                bnd[eidx[e]] = True
        return F2, edges, d, l, A, bnd, nflip

    if os.path.exists(cache):
        z = np.load(cache)
        edges, d, l, A, bnd = z["edges"], z["d"], z["l"], z["A"], z["bnd"]
        if "F2" in z.files:
            F2, tri_area = z["F2"], z["tri_area"]
            if verbose:
                print(f"[{mesh}] cache: {m.nV} vertices, negative dual edges "
                      f"{100*(l < 0).mean():.3f}%")
        else:
            # An older cache predates F2. Rebuilding is 0.3 s, but the rebuilt metric must
            # be the one already on disk or every result measured against this cache
            # silently refers to a different medium - so it is compared, not assumed, and
            # only the faces are added.
            t0 = time.time()
            F2, e2, d2, l2, A2, b2, _ = build()
            for nm, was, now in (("edges", edges, e2), ("d", d, d2), ("l", l, l2),
                                 ("A", A, A2), ("bnd", bnd, b2)):
                if not np.array_equal(was, now):
                    raise RuntimeError(
                        f"[{mesh}] rebuilding to add F2 changed `{nm}`. The cached metric "
                        f"is not reproducible from the current code, so adding faces to it "
                        f"would describe a different mesh from the one every earlier run "
                        f"used. Delete {cache} deliberately if that is intended.")
            F2 = _orient(F2, Vc, m.vertex_normal)
            tri_area = _heron(F2, edges, d)
            np.savez(cache, edges=edges, d=d, l=l, A=A, bnd=bnd, F2=F2, tri_area=tri_area)
            if verbose:
                print(f"[{mesh}] cache: {m.nV} vertices, negative dual edges "
                      f"{100*(l < 0).mean():.3f}%; added the intrinsic faces in "
                      f"{time.time()-t0:.1f}s, metric unchanged")
    else:
        t0 = time.time()
        F2, edges, d, l, A, bnd, nflip = build()
        F2 = _orient(F2, Vc, m.vertex_normal)
        tri_area = _heron(F2, edges, d)
        np.savez(cache, edges=edges, d=d, l=l, A=A, bnd=bnd, F2=F2, tri_area=tri_area)
        if verbose:
            print(f"[{mesh}] built in {time.time()-t0:.1f}s: {m.nV} vertices, "
                  f"{nflip} Delaunay flips, negative dual edges "
                  f"{100*(m.l < 0).mean():.2f}% -> {100*(l < 0).mean():.3f}%")

    c = Cortex()
    c.mesh, c.m, c.V, c.F = mesh, m, Vc, Fc
    c.lab, c.names, c.old = lab[old], names, old
    c.edges, c.d, c.l, c.A, c.bnd = edges, d, l, A, bnd
    c.F2, c.tri_area = F2, tri_area
    c.nV = m.nV
    c.parcels = np.array(sorted(set(int(x) for x in c.lab if x > 0)))
    return c


def boundary_loops(c):
    """Number of separate boundary loops - should be 1 (the medial wall)."""
    adj = defaultdict(list)
    for a, b in c.edges[c.bnd]:
        adj[int(a)].append(int(b)); adj[int(b)].append(int(a))
    seen, loops = set(), 0
    for s in adj:
        if s in seen:
            continue
        loops += 1
        stack = [s]
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x); stack.extend(adj[x])
    return loops


if __name__ == "__main__":
    for mesh in ("fsaverage5", "fsaverage6"):
        c = load_cortex(mesh)
        print(f"  {c}")
        print(f"  boundary loops: {boundary_loops(c)}   "
              f"min dual edge length {c.l.min():+.4f}   "
              f"parcels present: {len(c.parcels)}")
