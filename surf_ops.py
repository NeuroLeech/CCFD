"""Surface loading and primal/dual (DEC) operators on a triangulated cortical surface."""
import os
import numpy as np, nibabel as nib
from scipy.sparse import csr_matrix, diags
from paths import SURF


def load_fsaverage(mesh='fsaverage6', surf='infl_left'):
    """Read the bundled surface. Previously fetched through nilearn, which put
    the files in a download cache outside this folder."""
    g = nib.load(os.path.join(SURF, mesh, surf + ".gii.gz"))
    return g.darrays[0].data.astype(np.float64), g.darrays[1].data.astype(np.int64)


def load_glasser(annot_path, n_vertices):
    """HCP-MMP1 annot is on fsaverage (163842). FreeSurfer's icosahedra are nested,
    so fsaverage5/6 are exactly the first 10242 / 40962 vertices - verified to
    1.5e-6 degrees. Resampling is therefore exact truncation, no interpolation."""
    lab, ctab, names = nib.freesurfer.read_annot(annot_path)
    names = [n.decode() if isinstance(n, bytes) else n for n in names]
    assert len(lab) >= n_vertices
    return lab[:n_vertices], names


def submesh(V, F, keep):
    fm = keep[F].all(1)
    F2 = F[fm]
    old = np.unique(F2)
    remap = -np.ones(len(V), np.int64); remap[old] = np.arange(len(old))
    return V[old], remap[F2], old


class SurfaceMesh:
    """Primal triangulation + circumcentric (Voronoi) dual.
       d_e = |x_i - x_j|;  l_e = w_e d_e with w_e = (cot a + cot b)/2 the SIGNED
       dual edge length;  l_e < 0 marks a locally non-Delaunay edge."""

    def __init__(self, V, F):
        self.V, self.F = V, F
        nV, nF = len(V), len(F)
        self.nV, self.nF = nV, nF
        E = np.vstack([F[:, [1, 2]], F[:, [2, 0]], F[:, [0, 1]]])
        opp = np.concatenate([F[:, 0], F[:, 1], F[:, 2]])
        uniq, inv = np.unique(np.sort(E, axis=1), axis=0, return_inverse=True)
        self.E, self.nE, self.edge_of_corner = uniq, len(uniq), inv

        p = V[opp]; a = V[E[:, 0]]; b = V[E[:, 1]]
        u, v = a - p, b - p
        cot = (u * v).sum(1) / np.linalg.norm(np.cross(u, v), axis=1)
        w = np.zeros(self.nE); np.add.at(w, inv, 0.5 * cot)
        self.w = w
        self.d = np.linalg.norm(V[uniq[:, 0]] - V[uniq[:, 1]], axis=1)
        self.l = w * self.d

        p0, p1, p2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
        n = np.cross(p1 - p0, p2 - p0)
        self.tri_area = 0.5 * np.linalg.norm(n, axis=1)
        self.tri_normal = n / np.linalg.norm(n, axis=1)[:, None]

        A = np.zeros(nV)
        contrib = 0.25 * w * self.d ** 2
        np.add.at(A, uniq[:, 0], 0.5 * contrib); np.add.at(A, uniq[:, 1], 0.5 * contrib)
        Ab = np.zeros(nV); np.add.at(Ab, F.ravel(), np.repeat(self.tri_area / 3.0, 3))
        self.A_voronoi, self.A_bary = A, Ab
        self.A = np.where(A > 0.2 * Ab, A, Ab)

        vn = np.zeros((nV, 3))
        for k in range(3): np.add.at(vn, F[:, k], n)
        self.vertex_normal = vn / np.linalg.norm(vn, axis=1)[:, None]

        rows = np.repeat(np.arange(self.nE), 2)
        self.d0 = csr_matrix((np.tile([-1.0, 1.0], self.nE), (rows, uniq.ravel())),
                             shape=(self.nE, nV))
        self.L = (self.d0.T @ diags(w) @ self.d0).tocsr()

        cnt = np.bincount(inv, minlength=self.nE)
        self.boundary_edge = cnt == 1
        bv = np.zeros(nV, bool); bv[uniq[self.boundary_edge].ravel()] = True
        self.boundary_vertex = bv

    def tri_grad(self, phi):
        V, F, n = self.V, self.F, self.tri_normal
        p0, p1, p2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
        g = (np.cross(n, p2 - p1) * phi[F[:, 0]][:, None]
             + np.cross(n, p0 - p2) * phi[F[:, 1]][:, None]
             + np.cross(n, p1 - p0) * phi[F[:, 2]][:, None])
        return g / (2 * self.tri_area)[:, None]

    def report(self, tag=""):
        print(f"  [{tag}] V={self.nV} E={self.nE} F={self.nF} boundary={self.boundary_vertex.sum()}")
        print(f"  [{tag}] negative dual edge length: {(self.l<0).sum()}/{self.nE} "
              f"= {100*(self.l<0).mean():.2f}%   min l_e={self.l.min():.3f}")
