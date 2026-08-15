"""Rotating shallow water on a triangulated surface, energy-neutral Coriolis.

Coriolis is built as scatter = transpose(gather):
    gather   U_i = (1/A_i) sum_e s_ie l_e (x_e - x_i) u_e
    rotate   V_i = f_i (n_i x U_i)
    scatter  (Cu)_e = (1/A_e) sum_i A_i s_ie (l_e/A_i)(x_e - x_i).V_i
so sum_e A_e u_e (Cu)_e = sum_i A_i V_i.U_i = 0 exactly (V is U rotated 90 deg).
Time stepping uses Crank-Nicolson on the Coriolis term: forward Euler on a
rotation amplifies every step no matter how good the operator is.
"""
import numpy as np
from scipy.sparse import csr_matrix, vstack


class RotSWE:
    def set_sponge(self, sigma_v):
        """Absorbing edge. sigma_v is a per-vertex damping rate (0 in the interior,
        ramping up near the boundary). Applied implicitly, so any strength is stable.
        sigma_v = None (or all zeros) leaves the boundary perfectly reflecting."""
        if sigma_v is None:
            self.sig_v = self.sig_e = None
            return
        self.sig_v = np.asarray(sigma_v, float)
        self.sig_e = 0.5 * (self.sig_v[self.Ei] + self.sig_v[self.Ej])

    def __init__(self, m, f_field, l=None, d=None, A=None, E=None, bnd_edge=None):
        self.sig_v = self.sig_e = None
        V = m.V
        self.V = V
        self.E = m.E if E is None else E
        self.l = m.l if l is None else l
        self.d = m.d if d is None else d
        self.A = m.A if A is None else A
        self.bnd_edge = m.boundary_edge if bnd_edge is None else bnd_edge
        self.nV, self.nE = len(V), len(self.E)
        Ei, Ej = self.E[:, 0], self.E[:, 1]
        self.Ei, self.Ej = Ei, Ej
        self.xe = 0.5 * (V[Ei] + V[Ej])
        self.t = (V[Ej] - V[Ei]) / self.d[:, None]
        self.n_i = m.vertex_normal
        self.f = np.asarray(f_field, float) if np.ndim(f_field) else np.full(self.nV, float(f_field))
        self.Ae = 0.5 * self.l * self.d
        self.g_i = self.l[:, None] * (self.xe - V[Ei])
        self.g_j = -self.l[:, None] * (self.xe - V[Ej])
        self._build_operators()

    def astype(self, dtype):
        """Recast operators and geometry to float32 for speed. Sparse matvec is
        memory-bound, so halving the width helps; accuracy loss is ~1e-7 relative
        per step, which is irrelevant for a parameter search."""
        self.G = self.G.astype(dtype); self.Gt = self.Gt.astype(dtype)
        self.D = self.D.astype(dtype)
        for a in ('_invA','_invAe','n_i','f','d','l','A','Ae'):
            setattr(self, a, getattr(self, a).astype(dtype))
        if self.sig_v is not None:
            self.sig_v = self.sig_v.astype(dtype); self.sig_e = self.sig_e.astype(dtype)
        self.dtype = dtype
        return self

    def _build_operators(self):
        """Precompute the gather as a sparse matrix. It is a fixed linear operator,
        so np.add.at (which is slow) can be replaced by one matvec; the scatter is
        then literally its transpose, which keeps the Coriolis term exactly
        energy-neutral by construction rather than by arithmetic that happens to
        match."""
        nV, nE = self.nV, self.nE
        rows = np.concatenate([self.Ei, self.Ej])
        cols = np.concatenate([np.arange(nE), np.arange(nE)])
        blocks = []
        for k in range(3):
            vals = np.concatenate([self.g_i[:, k], self.g_j[:, k]])
            blocks.append(csr_matrix((vals, (rows, cols)), shape=(nV, nE)))
        self.G = vstack(blocks).tocsr()            # (3nV, nE)
        self.Gt = self.G.T.tocsr()
        dv = np.concatenate([self.l, -self.l])
        self.D = csr_matrix((dv, (rows, cols)), shape=(nV, nE))
        self._invA = 1.0 / self.A
        self._invAe = 1.0 / self.Ae

    def gather(self, ue):
        U = (self.G @ ue).reshape(3, self.nV).T * self._invA[:, None]
        return U - (U * self.n_i).sum(1)[:, None] * self.n_i

    def coriolis(self, ue):
        U = self.gather(ue)
        Vr = self.f[:, None] * np.cross(self.n_i, U)
        return (self.Gt @ Vr.T.ravel()) * self._invAe

    def divergence(self, ue):
        return (self.D @ ue) * self._invA

    # --- reference implementations, kept for verification ---
    def gather_ref(self, ue):
        U = np.zeros((self.nV, 3))
        np.add.at(U, self.Ei, self.g_i * ue[:, None])
        np.add.at(U, self.Ej, self.g_j * ue[:, None])
        U /= self.A[:, None]
        return U - (U * self.n_i).sum(1)[:, None] * self.n_i

    def coriolis_ref(self, ue):
        U = self.gather_ref(ue)
        Vr = self.f[:, None] * np.cross(self.n_i, U)
        return ((self.g_i * Vr[self.Ei]).sum(1) + (self.g_j * Vr[self.Ej]).sum(1)) / self.Ae

    def divergence_ref(self, ue):
        acc = np.zeros(self.nV)
        np.add.at(acc, self.Ei, self.l * ue); np.add.at(acc, self.Ej, -self.l * ue)
        return acc / self.A

    def grad(self, h):
        return (h[self.Ej] - h[self.Ei]) / self.d

    # ---------------- batched: evaluate a whole population at once ----------------
    def gather_b(self, ue):
        """ue: (nE, P) -> U: (nV, 3, P). One sparse x dense product instead of P
        matvecs, which reuses the matrix across the population."""
        U = (self.G @ ue).reshape(3, self.nV, -1).transpose(1, 0, 2)
        U = U * self._invA[:, None, None]
        return U - (U * self.n_i[:, :, None]).sum(1)[:, None, :] * self.n_i[:, :, None]

    def coriolis_b(self, ue):
        U = self.gather_b(ue)
        n = self.n_i
        Vr = np.empty_like(U)
        Vr[:, 0, :] = n[:, 1, None]*U[:, 2, :] - n[:, 2, None]*U[:, 1, :]
        Vr[:, 1, :] = n[:, 2, None]*U[:, 0, :] - n[:, 0, None]*U[:, 2, :]
        Vr[:, 2, :] = n[:, 0, None]*U[:, 1, :] - n[:, 1, None]*U[:, 0, :]
        Vr *= self.f[:, None, None]
        flat = Vr.transpose(1, 0, 2).reshape(3*self.nV, -1)
        return (self.Gt @ flat) * self._invAe[:, None]

    def grad_b(self, h):
        return (h[self.Ej] - h[self.Ei]) * (1.0/self.d)[:, None]

    def divergence_b(self, ue):
        return (self.D @ ue) * self._invA[:, None]

    def step_b(self, ue, h, dt, g, H, niter=2):
        rhs = ue + dt * (-g * self.grad_b(h))
        Cold = self.coriolis_b(ue)
        un = rhs + dt * Cold
        for _ in range(niter - 1):
            un = rhs + 0.5 * dt * (Cold + self.coriolis_b(un))
        un[self.bnd_edge, :] = 0.0
        hn = h - dt * H * self.divergence_b(un)
        if self.sig_v is not None:
            un = un / (1.0 + dt * self.sig_e[:, None])
            hn = hn / (1.0 + dt * self.sig_v[:, None])
        return un, hn

    def step(self, ue, h, dt, g, H, niter=2):
        rhs = ue + dt * (-g * self.grad(h))
        Cold = self.coriolis(ue)
        # The first fixed-point pass starts from un = ue, so coriolis(un) == Cold
        # and the pass reduces to rhs + dt*Cold. Doing it in closed form skips one
        # of the three Coriolis evaluations per step - identical arithmetic.
        un = rhs + dt * Cold
        for _ in range(niter - 1):
            un = rhs + 0.5 * dt * (Cold + self.coriolis(un))
        un[self.bnd_edge] = 0.0
        hn = h - dt * H * self.divergence(un)
        if self.sig_v is not None:
            un = un / (1.0 + dt * self.sig_e)
            hn = hn / (1.0 + dt * self.sig_v)
        return un, hn


def sponge_profile(Vc, edges, bnd_edge, width, strength):
    """Damping rate per vertex: 0 beyond `width` of the boundary, rising smoothly
    to `strength` at the rim. Distance is measured along the surface."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra
    n = len(Vc)
    w = np.linalg.norm(Vc[edges[:, 0]] - Vc[edges[:, 1]], axis=1)
    G = csr_matrix((np.r_[w, w], (np.r_[edges[:, 0], edges[:, 1]],
                                  np.r_[edges[:, 1], edges[:, 0]])), shape=(n, n))
    seeds = np.unique(edges[bnd_edge].ravel())
    dist = dijkstra(G, indices=seeds, directed=False).min(axis=0)
    u = np.clip(1.0 - dist / max(width, 1e-9), 0.0, 1.0)
    return strength * u * u * (3.0 - 2.0 * u)          # smoothstep ramp
