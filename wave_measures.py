"""Stage-1 measures: is the field wave-like, and is it moving?

Everything comes from one object, the structure tensor of the field gradient in each
vertex's tangent plane:

    J = <grad h  grad h^T>          2x2, symmetric, area-weighted over the 1-ring

    anisotropy  A = (l1 - l2)/(l1 + l2)     1 = locally striped/wave-like, 0 = isotropic
    orientation th = principal eigenvector   (mod pi, so average with doubled angles)
    flow        v = -J^-1 <grad h  dh/dt>    the local transport velocity

A plane wave has grad h everywhere parallel to travel, so l2 = 0 and A = 1. An
isotropic field gives A -> 0. Standing and travelling patterns both score high on A,
so the third measure - how fast the orientation field turns over - is what separates
them.

No vorticity, no turbulence statistics, no phase/Hilbert machinery.
"""
import numpy as np
from scipy.sparse.csgraph import dijkstra
from scipy.sparse import csr_matrix


class WaveMeasures:
    """Precompute the geometry once; then each frame is a few bincounts."""

    def __init__(self, cortex, exclude_mm=20.0):
        c = self.c = cortex
        m = c.m
        nV, F = m.nV, m.F

        # per-vertex orthonormal tangent basis
        n = m.vertex_normal
        tmp = np.tile(np.array([1.0, 0.0, 0.0]), (nV, 1))
        bad = np.abs(n[:, 0]) > 0.9
        tmp[bad] = np.array([0.0, 1.0, 0.0])
        e1 = np.cross(n, tmp); e1 /= np.linalg.norm(e1, axis=1)[:, None]
        e2 = np.cross(n, e1)
        self.e1, self.e2 = e1, e2

        # (triangle, corner) -> vertex, flattened, for fast bincount accumulation
        self.vid = F.T.ravel().astype(np.int64)             # (3nF,)
        self.tid = np.tile(np.arange(len(F)), 3)
        self.w = m.tri_area[self.tid]
        self.e1v, self.e2v = e1[self.vid], e2[self.vid]
        self.nV = nV
        self.wsum = np.bincount(self.vid, weights=self.w, minlength=nV)

        # exclude a band at the medial-wall rim: reflections there manufacture
        # anisotropy, and a search would otherwise learn to exploit the wall
        w_e = np.linalg.norm(c.V[c.edges[:, 0]] - c.V[c.edges[:, 1]], axis=1)
        G = csr_matrix((np.r_[w_e, w_e],
                        (np.r_[c.edges[:, 0], c.edges[:, 1]],
                         np.r_[c.edges[:, 1], c.edges[:, 0]])), shape=(nV, nV))
        seeds = np.unique(c.edges[c.bnd].ravel())
        self.dist_to_edge = dijkstra(G, indices=seeds, directed=False).min(axis=0)
        self.keep = self.dist_to_edge > exclude_mm
        self.exclude_mm = exclude_mm

        # neighbour-averaging operator over kept-kept edges only, so the exclusion
        # band does not drag estimates toward zero at its inner edge
        ek = c.edges[self.keep[c.edges[:, 0]] & self.keep[c.edges[:, 1]]]
        one = np.ones(len(ek))
        Adj = csr_matrix((np.r_[one, one],
                          (np.r_[ek[:, 0], ek[:, 1]], np.r_[ek[:, 1], ek[:, 0]])),
                         shape=(nV, nV))
        self.Adj = Adj
        self.deg = np.maximum(np.asarray(Adj.sum(1)).ravel(), 1.0)
        self._baseline = None

    # ------------------------------------------------------------------ per frame
    def tensor(self, h):
        """Area-weighted structure tensor per vertex -> (Jxx, Jyy, Jxy)."""
        g = self.c.m.tri_grad(np.asarray(h, np.float64))[self.tid]
        gx = (g*self.e1v).sum(1)
        gy = (g*self.e2v).sum(1)
        nV, w, vid = self.nV, self.w, self.vid
        Jxx = np.bincount(vid, weights=w*gx*gx, minlength=nV)
        Jyy = np.bincount(vid, weights=w*gy*gy, minlength=nV)
        Jxy = np.bincount(vid, weights=w*gx*gy, minlength=nV)
        return Jxx, Jyy, Jxy

    @staticmethod
    def anisotropy(Jxx, Jyy, Jxy):
        tr = Jxx + Jyy
        disc = np.sqrt((Jxx - Jyy)**2 + 4.0*Jxy*Jxy)
        A = np.where(tr > 1e-30, disc/np.maximum(tr, 1e-30), 0.0)
        theta2 = np.arctan2(2.0*Jxy, Jxx - Jyy)       # doubled angle: orientation is mod pi
        return A, theta2, tr

    def flow(self, h, dhdt, reg=1e-3):
        """v = -J^-1 <grad h dh/dt>, Tikhonov-regularised. Also returns the
        normal-flow speed |dh/dt|/|grad h|, which is well defined even where the
        aperture problem makes the full vector unrecoverable."""
        g = self.c.m.tri_grad(np.asarray(h, np.float64))[self.tid]
        ht = np.asarray(dhdt, np.float64)[self.c.m.F].mean(1)[self.tid]
        gx = (g*self.e1v).sum(1); gy = (g*self.e2v).sum(1)
        nV, w, vid = self.nV, self.w, self.vid
        Jxx = np.bincount(vid, weights=w*gx*gx, minlength=nV)
        Jyy = np.bincount(vid, weights=w*gy*gy, minlength=nV)
        Jxy = np.bincount(vid, weights=w*gx*gy, minlength=nV)
        bx = np.bincount(vid, weights=w*gx*ht, minlength=nV)
        by = np.bincount(vid, weights=w*gy*ht, minlength=nV)
        lam = reg*np.maximum(Jxx + Jyy, 1e-30)
        a, d, b = Jxx + lam, Jyy + lam, Jxy
        det = np.maximum(a*d - b*b, 1e-30)
        vx = -( d*bx - b*by)/det
        vy = -(-b*bx + a*by)/det
        gm = np.sqrt(np.maximum(Jxx + Jyy, 0.0)/np.maximum(self.wsum, 1e-30))
        htv = np.abs(np.asarray(dhdt, np.float64))
        nspeed = np.where(gm > 1e-12, htv/np.maximum(gm, 1e-30), np.nan)
        return vx, vy, nspeed, Jxx + Jyy

    # ------------------------------------------------------------------ mesh floor
    def baseline(self, n=48, seed=0):
        """The anisotropy an unstructured field already shows on THIS mesh.

        The discrete gradient on an irregular triangulation has preferred
        directions, so white noise scores far above zero (~0.6 at fsaverage5) and
        its orientation is nearly frame-invariant - it is a property of the mesh,
        not of the field. Measured once per vertex and divided out, so a reported
        anisotropy of 0 means 'no more oriented than noise on this mesh'."""
        if self._baseline is None:
            rng = np.random.default_rng(seed)
            acc = np.zeros(self.nV)
            for _ in range(n):
                Jxx, Jyy, Jxy = self.tensor(rng.standard_normal(self.nV))
                A, _, _ = self.anisotropy(Jxx, Jyy, Jxy)
                acc += A
            self._baseline = acc/n
        return self._baseline

    def corrected(self, A):
        A0 = self.baseline()
        return (A - A0)/np.maximum(1.0 - A0, 1e-9)

    # ------------------------------------------------------------------ summary
    def summarise(self, Hs, dt_frames, weight_by_power=True):
        """Hs: (nframes, nV) field snapshots, evenly spaced by dt_frames.

        Returns scalars describing how wave-like the run is. Vertices near the
        medial wall are excluded throughout.
        """
        Hs = np.asarray(Hs, np.float64)
        nF = len(Hs)
        keep = self.keep
        Aw, th2, pw = [], [], []
        for k in range(nF):
            Jxx, Jyy, Jxy = self.tensor(Hs[k])
            A, t2, tr = self.anisotropy(Jxx, Jyy, Jxy)
            Aw.append(A[keep]); th2.append(t2[keep]); pw.append(tr[keep])
        A = np.array(Aw); th2 = np.array(th2); pw = np.array(pw)
        A_raw = A.copy()
        A0k = self.baseline()[keep]
        A = (A_raw - A0k)/np.maximum(1.0 - A0k, 1e-9)   # not clipped at 0: clipping a
        # zero-mean fluctuation biases an unstructured field upward

        # power-weighted anisotropy: quiet vertices carry no orientation information
        wts = pw/np.maximum(pw.sum(axis=1, keepdims=True), 1e-30) if weight_by_power \
            else np.full_like(pw, 1.0/pw.shape[1])
        A_mean = float((A*wts).sum(axis=1).mean())
        A_raw_mean = float((A_raw*wts).sum(axis=1).mean())

        # spatial coherence of orientation, over each vertex's neighbours
        z = np.exp(1j*th2)*pw
        zc = self._neighbour_mean(z)
        denom = self._neighbour_mean(pw)
        coh = np.abs(zc)/np.maximum(denom, 1e-30)
        coh_mean = float(np.nanmean(coh))

        pattern_tau = self._pattern_decorr(Hs[:, keep], dt_frames)

        # propagation speed and direction coherence, from consecutive frames
        speeds, dcoh = [], []
        for k in range(nF - 1):
            dh = (Hs[k+1] - Hs[k])/dt_frames
            vx, vy, ns, tr = self.flow(0.5*(Hs[k] + Hs[k+1]), dh)
            v = ns[keep]
            v = v[np.isfinite(v)]
            if len(v):
                speeds.append(np.median(v))
            dcoh.append(self._dir_coherence(vx, vy, tr))
        speed = float(np.median(speeds)) if speeds else float("nan")
        dcoh = np.array(dcoh, float)
        dir_coh = float(np.nanmean(dcoh)) if len(dcoh) else float("nan")

        return dict(
            anisotropy=A_mean,                 # mesh floor divided out
            anisotropy_raw=A_raw_mean,         # before correction, for reference
            orient_coherence=coh_mean,
            dir_coherence=dir_coh,
            pattern_tau=pattern_tau,
            speed=speed,
            anisotropy_sd_time=float((A*wts).sum(axis=1).std()),
            n_excluded=int((~keep).sum()),
        )

    def _dir_coherence(self, vx, vy, tr):
        """Do neighbouring vertices agree on which WAY the field is moving?

        The same construction as orient_coherence, but on the flow angle mod 2pi
        instead of the gradient orientation mod pi. That distinction is the point:
        a standing pattern has a coherent orientation while its flow direction
        flips sign every half cycle, so orientation cannot tell a travelling
        structure from a standing one and this can.

        Weighted by local gradient power, because where the field is flat the
        direction is numerically arbitrary and would otherwise dominate the mean.
        """
        keep = self.keep
        ang = np.arctan2(vy[keep], vx[keep])
        pw = tr[keep]
        good = np.isfinite(ang) & np.isfinite(pw)
        pw = np.where(good, np.maximum(pw, 0.0), 0.0)
        z = np.where(good, np.exp(1j*np.nan_to_num(ang))*pw, 0.0)
        zc = self._neighbour_mean(z)
        denom = self._neighbour_mean(pw)
        coh = np.abs(zc)/np.maximum(denom, 1e-30)
        return float(np.nanmean(coh))

    # ------------------------------------------------------------------ helpers
    def _neighbour_mean(self, x):
        """Neighbour average of x, given over kept vertices, shape (..., nKept).
        Only kept-kept edges contribute."""
        idx = np.flatnonzero(self.keep)
        x = np.atleast_2d(x)
        full = np.zeros(x.shape[:-1] + (self.nV,), dtype=x.dtype)
        full[..., idx] = x
        if np.iscomplexobj(full):
            out = (full.real @ self.Adj.T) + 1j*(full.imag @ self.Adj.T)
        else:
            out = full @ self.Adj.T
        return (out/self.deg)[..., idx]

    @staticmethod
    def _pattern_decorr(X, dt_frames):
        """1/e decorrelation time of the spatial pattern: how long before the field
        stops resembling itself. A frozen field never decorrelates; white noise
        decorrelates in one frame; a travelling wave sits between."""
        nF = len(X)
        maxlag = min(nF - 1, 80)
        if maxlag < 2:
            return float("nan")
        Z = X - X.mean(axis=1, keepdims=True)
        nrm = np.linalg.norm(Z, axis=1)
        good = nrm > 1e-30
        if good.sum() < 3:
            return float("nan")
        Z = Z[good]/nrm[good][:, None]
        nF = len(Z)
        vals = [float((Z[:nF-lag]*Z[lag:]).sum(1).mean()) for lag in range(min(maxlag, nF-1) + 1)]
        vals = np.array(vals)
        below = np.flatnonzero(vals < np.exp(-1.0))
        return float(below[0]*dt_frames) if len(below) else float(len(vals)*dt_frames)
