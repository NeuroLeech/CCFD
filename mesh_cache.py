"""Build and cache a cortical fluid mesh: surface + Glasser atlas + repaired metric.

The intrinsic-Delaunay repair is what makes the C-grid shallow-water scheme stable;
without it ~6% of dual edge lengths are negative and the scheme blows up in ~200
steps. The repair changes only connectivity, not vertex positions.
"""
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
                 "edges", "d", "l", "A", "bnd", "nV", "parcels")

    def __repr__(self):
        return (f"<Cortex {self.mesh}: {self.nV} fluid vertices, {len(self.edges)} edges, "
                f"{len(self.parcels)} parcels>")


def load_cortex(mesh="fsaverage5", annot=ANNOT, verbose=True):
    cache = os.path.join(CACHE, f"cortex_{mesh}.npz")
    V, F = load_fsaverage(mesh, "infl_left")
    lab, names = load_glasser(annot, len(V))
    active = lab > 0                       # label 0 is '???' = medial wall
    Vc, Fc, old = submesh(V, F, active)
    m = SurfaceMesh(Vc, Fc)

    if os.path.exists(cache):
        z = np.load(cache)
        edges, d, l, A, bnd = z["edges"], z["d"], z["l"], z["A"], z["bnd"]
        if verbose:
            print(f"[{mesh}] cache: {m.nV} vertices, negative dual edges "
                  f"{100*(l < 0).mean():.3f}%")
    else:
        t0 = time.time()
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
        np.savez(cache, edges=edges, d=d, l=l, A=A, bnd=bnd)
        if verbose:
            print(f"[{mesh}] built in {time.time()-t0:.1f}s: {m.nV} vertices, "
                  f"{nflip} Delaunay flips, negative dual edges "
                  f"{100*(m.l < 0).mean():.2f}% -> {100*(l < 0).mean():.3f}%")

    c = Cortex()
    c.mesh, c.m, c.V, c.F = mesh, m, Vc, Fc
    c.lab, c.names, c.old = lab[old], names, old
    c.edges, c.d, c.l, c.A, c.bnd = edges, d, l, A, bnd
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
