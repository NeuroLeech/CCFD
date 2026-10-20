"""Which way the field travels, and which inputs' timing sends it there.

The medium is linear, so at one frequency the field is exactly a weighted sum of the
pieces' transfer functions:

    Phi(v) = sum_k H_k(v, w) z_k

with z_k the complex amplitude of piece k at that frequency. |z_k| is how loud the piece
is and arg z_k is WHEN it peaks within the cycle, so the temporal interplay between the
inputs is the set of relative phases arg z_k - arg z_l, and nothing else. The field's own
phase arg Phi(v) then fixes the geometry of the wave: crests sit where w t + arg Phi is
constant, so they travel down the phase gradient, along -grad(arg Phi), at speed
w / |grad(arg Phi)|. Everything here is read off that one complex pattern.

Two consequences worth knowing before reading any number:

  - The BOLD kernel and the passband are filters in TIME, the same at every vertex. At
    one frequency they multiply Phi by a single complex gain, which changes no phase
    gradient. Per frequency, the fluid and the observable carry the SAME waves; the
    observable only reweights which frequencies dominate. So nothing here is asked
    "through the instrument" - the band is the only place the instrument enters.
  - A pattern's geometry is invariant to multiplying z by any complex scalar, so global
    phase and overall loudness are irrelevant. What is left - relative phase and relative
    amplitude between pieces - is exactly what the attribution below splits.

Wave descriptors of one complex pattern, all amplitude-weighted (a phase where the field
is ~0 means nothing):

  travel     Feeny's travelling index, 0 for a standing wave (Re and Im of Phi spatially
             collinear, the field just breathes) to 1 for a pure travelling one
  speed      geometric-mean phase speed, mm/s on the inflated surface
  align_*    mean cosine between the propagation direction and a named tangent axis:
             `anterior`, `dorsal`, `outward` (away from the driven territory), or a pair
             axis pointing from one pathway's territory to another's
  planar     global order parameter: length of the amplitude-weighted mean propagation
             direction. 1 when the whole field moves one way; a source, a sink or a
             rotation cancels itself and reads low
  coherence  the same over a 10 mm disc, averaged: near 1 wherever nearby crests move
             together, so it separates ORGANISED waves - planar, radial or rotating
             alike - from disordered ones
  div, curl  of the unit propagation field, per cm: div > 0 is net outflow (sources),
             div < 0 net inflow (sinks), |curl| rotation
  n_sing     phase singularities - triangles around which the phase winds by 2 pi -
             the centres of rotating waves, counted where the field is not negligible

Three ways in, from cheapest and most exact to the one that also covers the nonlinear
medium:

  pairs        drive two pathways at a controlled phase lag, sweep the lag, and read how
               the direction flips. No S needed - this is what the MEDIUM does with a lag.
  draws        the solved S(f) is a distribution over z; draw from it at the strongest
               solved bins, read every pattern, and ask how much of each descriptor the
               inputs' relative TIMING explains against their relative LOUDNESS
               (cross-validated), which pathway pairs carry it, and per piece, which way
               advancing that piece's phase steers the waves. Plus the coalitions of S,
               each read as a pattern, against the same coalition with its phases zeroed.
  realisation  the same split on a simulated run, through a narrowband analytic signal of
               the field and of the drive. Works for a torch fit, where H does not exist.

  python analysis/waves.py pairs --tag grclip100_nolag
  python analysis/waves.py draws --tag grclip100_nolag --bands 6
  python analysis/waves.py realisation --tag grclip100_nolag_577s \\
      --frames results/frames_grclip100_nolag_577s_raw.npy --hz 0.03
  python analysis/waves.py pairs --standin 25,1.5 --basis results/sub_first_area_100.npz

`--standin decay_s,spread_mm_s` replaces the solved medium with a flat one on the same
clock - no map grading, no xspec npz - for checking the machinery where the cortical map
stack is not available. It is not the fitted medium and its numbers are not the model's.
"""
import _path  # noqa: F401  - puts the sibling code folders on sys.path
import os, argparse
import numpy as np
import scipy.sparse as sps
from scipy.sparse.csgraph import dijkstra

from paths import RESULTS, CACHE


# ============================================================ phase on the surface

class Surface:
    """Per-face gradients and per-vertex divergence on the cortex submesh.

    The geometry is the extrinsic triangulation `F` on the inflated vertex positions - the
    one the pictures are drawn on - not the solver's flipped F2: a phase gradient needs
    flat triangles that actually sit on the surface, and a flipped edge joins vertices
    that are not adjacent on it. Distances are therefore inflated-surface millimetres."""

    def __init__(self, c, radius=10.0):
        V = np.asarray(c.V, float)
        F = np.asarray(c.F, np.int64)
        nrm = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
        flip = (nrm * np.asarray(c.m.vertex_normal)[F].mean(1)).sum(1) < 0
        F = F.copy()
        F[flip] = F[flip][:, [0, 2, 1]]
        nrm[flip] *= -1
        self.c, self.V, self.F = c, V, F
        self.nV, self.nF = len(V), len(F)
        two_a = np.linalg.norm(nrm, axis=1)
        self.area = 0.5 * two_a
        self.n = nrm / two_a[:, None]
        # grad u = B @ [u1 - u0, u2 - u0]: the least-squares inverse of the two edge
        # vectors, which is exact for a linear function on a flat triangle and needs no
        # convention about which way round the edges are taken
        Ed = np.stack([V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]], -1)
        self.B = Ed @ np.linalg.inv(np.einsum("fai,faj->fij", Ed, Ed))       # (nF,3,2)
        gl = np.empty((self.nF, 3, 3))                   # grad of each barycentric
        gl[:, 1], gl[:, 2] = self.B[..., 0], self.B[..., 1]
        gl[:, 0] = -(gl[:, 1] + gl[:, 2])
        self.Av = np.bincount(F.ravel(), np.repeat(self.area / 3.0, 3), self.nV)
        # weak divergence: div(X)_v = -(1/A_v) sum_f A_f grad(lambda_v) . X_f, the adjoint
        # of the gradient, so div(grad u) is the cotangent Laplacian. Stored as one sparse
        # (nV, 3 nF) matrix acting on X flattened face-major.
        rows = np.repeat(F, 3, axis=1).ravel()
        cols = (3 * np.arange(self.nF)[:, None, None] + np.arange(3)[None, None, :])
        cols = np.broadcast_to(cols, (self.nF, 3, 3)).ravel()
        vals = (-self.area[:, None, None] * gl).ravel() / self.Av[rows]
        self.Dv = sps.csr_matrix((vals, (rows, cols)), shape=(self.nV, 3 * self.nF))
        # face -> vertex averaging, area weighted
        self.FtoV = sps.csr_matrix(
            (np.repeat(self.area / 3.0, 3) / self.Av[F.ravel()],
             (F.ravel(), np.repeat(np.arange(self.nF), 3))), shape=(self.nV, self.nF))
        self.radius = float(radius)
        self.K = self._disc_kernel(radius)

    # ---- distances
    def graph(self):
        E = np.asarray(self.c.edges)
        d = np.linalg.norm(self.V[E[:, 0]] - self.V[E[:, 1]], axis=1)
        G = sps.coo_matrix((d, (E[:, 0], E[:, 1])), shape=(self.nV, self.nV))
        return (G + G.T).tocsr()

    def distance_from(self, verts):
        """Graph-geodesic distance from a vertex set, mm."""
        return dijkstra(self.graph(), indices=np.asarray(verts), min_only=True)

    def _disc_kernel(self, r):
        """Row-normalised, area-weighted Gaussian of scale r, for local averages.

        Euclidean on the inflated surface rather than geodesic: within 3r = 30 mm the
        inflated sheet is close to flat, and a geodesic kernel would need the full
        all-pairs distance matrix, 9,374^2, to build."""
        from scipy.spatial import cKDTree
        t = cKDTree(self.V)
        W = t.sparse_distance_matrix(t, 3.0 * r, output_type="coo_matrix").tocsr()
        W.data = np.exp(-0.5 * (W.data / r) ** 2)
        W = W + sps.eye(self.nV)                 # zero distance is not stored
        W = W.multiply(self.Av[None, :]).tocsr()
        return (sps.diags(1.0 / np.asarray(W.sum(1)).ravel()) @ W).tocsr()

    # ---- fields
    def scalar_grad(self, u):
        """(nV,) real -> (nF, 3) tangent gradient."""
        u = np.asarray(u, float)
        d = np.stack([u[self.F[:, 1]] - u[self.F[:, 0]],
                      u[self.F[:, 2]] - u[self.F[:, 0]]], -1)
        return np.einsum("fij,fj->fi", self.B, d)

    def axis(self, u):
        """Unit tangent field pointing up the gradient of a scalar."""
        g = self.scalar_grad(u)
        return g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-12)

    def _dphase(self, Phi):
        """Wrapped phase differences along two edges of every face. (..., nF, 2).

        Wrapped edge by edge rather than unwrapped over the mesh, so a singularity costs
        its own face and nothing else."""
        Phi = np.asarray(Phi)
        a, b, c = (Phi[..., self.F[:, k]] for k in range(3))
        return np.stack([np.angle(b * a.conj()), np.angle(c * a.conj())], -1), (a, b, c)

    def phase_grad(self, Phi):
        """(..., nV) complex -> (..., nF, 3) gradient of arg Phi, rad/mm."""
        d, _ = self._dphase(Phi)
        return np.einsum("fij,...fj->...fi", self.B, d)

    def charge(self, Phi):
        """Winding number of arg Phi round each face: +1, -1, or 0. (..., nF)"""
        _, (a, b, c) = self._dphase(Phi)
        w = (np.angle(b * a.conj()) + np.angle(c * b.conj()) + np.angle(a * c.conj()))
        return np.rint(w / (2 * np.pi)).astype(np.int8)

    def div(self, X):
        """(..., nF, 3) -> (..., nV), per mm."""
        X = np.asarray(X)
        flat = X.reshape(-1, 3 * self.nF).T
        return np.asarray(self.Dv @ flat).T.reshape(X.shape[:-2] + (self.nV,))

    def curl(self, X):
        """Normal component of curl, per mm; positive for counter-clockwise circulation
        seen from outside the surface. div(n x X) = -curl(X) for a tangent field."""
        return -self.div(np.cross(self.n, X))


def travel_index(Phi, w):
    """Feeny's travelling index of complex patterns (..., nV) under vertex weights w.

    Re and Im of Phi are two spatial patterns the field alternates between each cycle.
    Collinear, the field only breathes: a standing wave, 0. Orthogonal and equal in size,
    it travels: 1. The index is the ratio of the two singular values of [Re Phi, Im Phi]."""
    s2 = (w * Phi * Phi).sum(-1)                         # sum w Phi^2
    m2 = (w * np.abs(Phi) ** 2).sum(-1)
    rho = np.abs(s2) / np.maximum(m2, 1e-30)
    return np.sqrt(np.clip((1 - rho) / np.maximum(1 + rho, 1e-30), 0, 1))


def describe(S, Phi, hz, axes, chunk=128, sing_floor=0.1, vweight=None):
    """Wave descriptors of a stack of complex patterns. -> dict name -> (N,) array.

    `axes` is name -> (nF, 3) unit tangent field; each gives an `align_<name>`.
    `sing_floor`: a singularity is counted when the larger of its face's vertex
    amplitudes reaches this fraction of the pattern's RMS. Every singularity sits at a
    zero of the field, so the test is on its surround, not on the face itself - it keeps
    the rotations that organise visible field and drops the ones in the noise floor far
    from any input.

    `vweight` (nV,) multiplies the amplitude weighting, to read the descriptors over one
    zone - where two pathways overlap, say - rather than wherever the field is loudest."""
    Phi = np.atleast_2d(np.asarray(Phi, np.complex64))
    N = len(Phi)
    om = 2 * np.pi * float(hz)
    keys = (["travel", "speed", "planar", "coherence", "div", "curl", "n_sing", "net_charge"]
            + [f"align_{k}" for k in axes])
    out = {k: np.empty(N) for k in keys}
    Ax = {k: np.asarray(v, np.float32) for k, v in axes.items()}
    for s in range(0, N, chunk):
        P = Phi[s:s + chunk]
        amp2 = np.abs(P) ** 2
        rms = np.sqrt((amp2 * S.Av).sum(1) / S.Av.sum())
        out["travel"][s:s + chunk] = travel_index(P, S.Av)
        wf = S.area * amp2[:, S.F].mean(-1)                         # (n, nF)
        if vweight is not None:
            wf = wf * np.asarray(vweight)[S.F].mean(-1)
        wf = wf / np.maximum(wf.sum(1, keepdims=True), 1e-30)
        g = S.phase_grad(P).astype(np.float32)                       # (n, nF, 3)
        gn = np.linalg.norm(g, axis=-1)
        k = -g / np.maximum(gn, 1e-12)[..., None]                    # propagation dir
        out["speed"][s:s + chunk] = np.exp((wf * np.log(om / np.maximum(gn, 1e-9))).sum(1))
        for nm, a in Ax.items():
            out[f"align_{nm}"][s:s + chunk] = (wf * (k * a).sum(-1)).sum(1)
        # vertex weights from the face weights, so div/curl/planar are read where the
        # field is, not where it is a phase of nothing
        wv = np.asarray(S.FtoV @ wf.T).T
        wv = wv / np.maximum(wv.sum(1, keepdims=True), 1e-30)
        out["div"][s:s + chunk] = 10.0 * (wv * S.div(k)).sum(1)
        out["curl"][s:s + chunk] = 10.0 * (wv * S.curl(k)).sum(1)
        # coherence: |local mean of amplitude-weighted direction| / local mean weight
        kv = np.stack([np.asarray(S.FtoV @ (wf * k[..., j]).T).T for j in range(3)], -1)
        num = np.stack([np.asarray(S.K @ kv[..., j].T).T for j in range(3)], -1)
        den = np.asarray(S.K @ np.asarray(S.FtoV @ wf.T)).T
        loc = np.linalg.norm(num, axis=-1) / np.maximum(den, 1e-30)
        out["coherence"][s:s + chunk] = (wv * np.clip(loc, 0, 1)).sum(1)
        # the mean direction, less its component along the mean normal: on a curved
        # patch, outward tangents all tilt away from the normal and would otherwise add
        # up to a "direction" that is not along the sheet at all
        m = (wf[..., None] * k).sum(1)
        nb = wf @ S.n
        nb = nb / np.maximum(np.linalg.norm(nb, axis=-1, keepdims=True), 1e-12)
        m = m - (m * nb).sum(-1, keepdims=True) * nb
        out["planar"][s:s + chunk] = np.linalg.norm(m, axis=-1)
        q = S.charge(P)
        big = np.sqrt(amp2[:, S.F].max(-1)) >= sing_floor * rms[:, None]
        q = q * big
        out["n_sing"][s:s + chunk] = (q != 0).sum(1)
        out["net_charge"][s:s + chunk] = q.sum(1)
    return out


def wave_type(d):
    """A coarse label per pattern, from the descriptors. Provisional: the thresholds are
    round numbers, not fitted, and the continuous descriptors are what to analyse.

      standing   travel < 0.3                     the field breathes in place
      rotating   >= 1 singularity and |curl| > |div|
      source     div > 0 dominates, not planar     crests leave a centre
      sink       div < 0 dominates, not planar     crests converge
      planar     planar >= 0.5                     the whole field moves one way
      mixed      none of these, or coherence < 0.4: no organised direction"""
    lab = np.full(len(d["travel"]), "mixed", dtype=object)
    rad = np.abs(d["div"]) > np.abs(d["curl"])
    flat = d["planar"] >= 0.5
    lab[flat] = "planar"
    lab[~flat & rad & (d["div"] > 0)] = "source"
    lab[~flat & rad & (d["div"] < 0)] = "sink"
    lab[(d["n_sing"] >= 1) & ~rad] = "rotating"
    lab[(d["coherence"] < 0.4) & (lab != "rotating")] = "mixed"
    lab[d["travel"] < 0.3] = "standing"
    return lab


# ============================================================ the medium and its H

def pathway_groups(c, labels, n):
    """Piece -> ascending pathway, by the majority Glasser parcel of its vertices."""
    from ascending_basis import FIRST_GROUPS, HIGHER_GROUPS, _names
    names = _names(c)
    owner = {p: g for g, ps in {**FIRST_GROUPS, **HIGHER_GROUPS}.items() for p in ps}
    lab = np.asarray(c.lab)
    out = []
    for k in range(n):
        vs = lab[np.asarray(labels) == k]
        if not len(vs):
            out.append("?"); continue
        out.append(owner.get(names[np.bincount(vs).argmax()], "?"))
    return out


def standin_medium(c, decay_s, spread_mm_s, oversample=4):
    """A flat medium on the fit's clock: same save, per-step damping, rotation and sponge
    as best_fit would set from BEST_X, but uniform speed and damping - no maps.

    Mirrors bo_step.unpack for a medium whose speed is 1 everywhere. Not the fitted
    medium; it exists so the geometry can be checked where the map stack is absent."""
    import timescale, fluid as fl
    from best_fit import BEST_X
    from genome import SPONGE_STRENGTH_FIXED
    clk = timescale.plan(oversample, decay_s=decay_s, spread_mm_s=spread_mm_s,
                         verbose=False)
    dt = fl.CFL * c.d.min() / 1.0
    p = dict(mode="group", c0=1.0, c_group=np.ones(4), sig_group=np.ones(4),
             sig0=clk["damp"] / dt, Ld=dt / 10.0 ** BEST_X[1],
             sponge_scale=10.0 ** BEST_X[2] / (dt * SPONGE_STRENGTH_FIXED))
    return p, clk["save"], clk["frame_s"]


def load_run(c, tag=None, standin=None, basis=None):
    """-> dict(p, save, frame_s, profiles, labels, tags, nimp, S, idx, ref_frames)."""
    import subparcels
    if standin:
        dec, spr = (float(v) for v in standin.split(","))
        p, save, frame_s = standin_medium(c, dec, spr)
        z = np.load(basis, allow_pickle=True)
        labels, tags = z["labels"], [str(t) for t in z["tags"]]
        P = (np.asarray(z["profiles"], np.float32) if "profiles" in z
             else subparcels.taper_profiles(c, labels, len(tags)))
        decay_fr = dec / frame_s
        nimp = int(np.ceil(max(7 * decay_fr, 1.0 / (0.01 * frame_s)) / 64.0) * 64)
        return dict(p=p, save=save, frame_s=frame_s, profiles=P, labels=labels,
                    tags=tags, nimp=nimp, S=None, idx=None, ref_frames=None,
                    name=f"standin_d{dec:g}_s{spr:g}")
    import bo_step, interference
    z = np.load(os.path.join(RESULTS, f"xspec_{tag}.npz"), allow_pickle=True)
    x, save = z["x"], int(z["save"])
    # exactly best_fit's construction: unpack, then the clip on top
    p, _, _ = bo_step.unpack(x, c)
    if "map_clip" in z.files:
        p["map_clip"] = str(z["map_clip"])
    if "maps" in z.files and len(z["maps"]):
        p["maps"] = tuple(str(m) for m in z["maps"])
        p["a"], p["b"] = np.asarray(z["map_a"], float), np.asarray(z["map_b"], float)
    labels, tags = z["labels"], [str(t) for t in z["tags"]]
    P = (np.asarray(z["profiles"], np.float32) if "profiles" in z.files
         else subparcels.taper_profiles(c, labels, len(tags)))
    frame_s = float(z["frame_s"])
    nimp = int(z["impulse_frames"]) if "impulse_frames" in z.files else 0
    nimp = max(nimp, interference._impulse_frames(x, save, frame_s))
    return dict(p=p, save=save, frame_s=frame_s, profiles=P, labels=labels, tags=tags,
                nimp=nimp, S=z["S"], idx=z["idx"], name=tag,
                ref_frames=int(z["ref_frames"]) if "ref_frames" in z.files else None)


def transfer_fields(c, run, hz, chunk=32, verbose=True):
    """Each piece's transfer function at every vertex: (len(hz), nV, K) complex64.

    The same impulse experiment xspec.impulse_responses runs - the profile as initial
    depth, sampled every `save` steps over `nimp` frames - but all K pieces integrated at
    once through RotSWE.step_b, and Fourier-summed as it goes at only the frequencies
    asked for. Holding the responses instead would be K x nimp x nV, ~4 GB at full width
    for the 100-piece set; this is K x nV per frequency."""
    import fluid as fl
    from xspec import profile_tag
    hz = np.atleast_1d(np.asarray(hz, float))
    p, save, nimp, P = run["p"], run["save"], run["nimp"], run["profiles"]
    key = (f"{c.mesh}_sig{p['sig0']:.6g}_c{p['c0']:.6g}_Ld{p['Ld']:.6g}"
           f"_spg{p.get('sponge_scale', 1.0):.4g}_a{np.round(p.get('a', 0), 3)}"
           f"_b{np.round(p.get('b', 0), 3)}_mc{p.get('map_clip')}_{p.get('mode')}"
           f"_{nimp}_{save}_{profile_tag(P)}_{run['frame_s']:.6g}"
           f"_hz{abs(hash(tuple(np.round(hz, 7)))) % 10**8}")
    f = os.path.join(CACHE, "waveH_" + key.replace(" ", "") + ".npz")
    if os.path.exists(f):
        z = np.load(f)
        if np.allclose(z["hz"], hz):
            return z["H"]
    s, dt, g, Hd = fl.build(c, p, sponge=True)
    K = len(P)
    w = 2 * np.pi * hz * run["frame_s"]                              # rad / frame
    h = np.ascontiguousarray(P.T, np.float32)                        # (nV, K)
    ue = np.zeros((s.nE, K), np.float32)
    Hd = Hd[:, None]
    out = np.zeros((len(hz), c.nV * K), np.complex64)
    buf, n0 = [h.copy()], 0
    dt = np.float32(dt)

    def flush(buf, n0):
        ph = np.exp(-1j * np.outer(w, np.arange(n0, n0 + len(buf)))).astype(np.complex64)
        out[:] += ph @ np.stack(buf).reshape(len(buf), -1)

    for n in range(1, nimp * save):
        ue, h = s.step_b(ue, h, dt, g, Hd)
        if n % save == 0:
            buf.append(h.copy())
            if len(buf) == chunk:
                flush(buf, n0); n0 += len(buf); buf = []
        if verbose and n % (50 * save * 20) == 0:
            print(f"    transfer: frame {n // save}/{nimp}", flush=True)
    if buf:
        flush(buf, n0)
    H = out.reshape(len(hz), c.nV, K)
    np.savez(f, H=H, hz=hz)
    return H


def axes_for(S, c, run):
    """The named tangent axes every pattern is projected on."""
    driven = np.flatnonzero(np.asarray(run["labels"]) >= 0)
    return dict(anterior=S.axis(S.V[:, 1]), dorsal=S.axis(S.V[:, 2]),
                outward=S.axis(S.distance_from(driven)))


# ============================================================ attribution

def group_amplitudes(z, groups, w):
    """Net complex injection per pathway: sum over its pieces of area-integral x z."""
    names = sorted(set(groups) - {"?"})
    G = np.stack([np.asarray([g == nm for g in groups], float) * w for nm in names])
    return z @ G.T, names


def timing_features(Zg, names):
    """Relative phase and relative loudness between pathways, kept apart.

    timing: cos and sin of every pairwise phase difference - invariant to global phase
            and to every amplitude
    amp:    log |Z_g| minus its mean over pathways - invariant to global scale and to
            every phase"""
    ph = np.angle(Zg)
    T, tl = [], []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            d = ph[:, i] - ph[:, j]
            T += [np.cos(d), np.sin(d)]
            tl.append((names[i], names[j]))
    la = np.log(np.maximum(np.abs(Zg), 1e-30))
    return np.stack(T, 1), tl, la - la.mean(1, keepdims=True)


def explained(X, y, folds=5, seed=0, blocked=False):
    """Cross-validated R^2 of a ridge fit; negative clipped to 0.

    `blocked` keeps folds contiguous. Samples of a narrowband signal taken closer than its
    envelope time are near-copies of each other, and shuffled folds would then score
    interpolation between neighbours rather than prediction."""
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import cross_val_predict, KFold
    if np.std(y) < 1e-12:
        return 0.0
    cv = KFold(folds) if blocked else KFold(folds, shuffle=True, random_state=seed)
    pred = cross_val_predict(RidgeCV(alphas=np.logspace(-3, 3, 13)), X, y, cv=cv)
    return max(0.0, 1.0 - np.mean((y - pred) ** 2) / np.var(y))


def attribute(D, Zg, names, keys, top=4, blocked=False, verbose=True):
    """How much of each descriptor relative timing explains, against relative loudness.

    unique timing = R2(timing + amp) - R2(amp): what is left once loudness has had every
    chance. Then where it comes from, two ways, because the 2 x n(n-1)/2 pair columns
    carry only n-1 independent phases and removing any one pair costs nothing:

      pair alone   R2 of that one phase difference (cos, sin) by itself
      pathway      the drop in full R2 when every pair involving that pathway goes, i.e.
                   that pathway's timing removed outright

    This explains VARIATION across draws or frames. A lag that is fixed - two pathways
    driven by one source at a constant offset - shapes the mean wave and explains none of
    its variance; the pairs sweep and the in-phase control are what see that."""
    T, tl, Am = timing_features(Zg, names)
    full = np.c_[T, Am]
    ev = lambda X, y: explained(X, y, blocked=blocked)
    res = {}
    for k in keys:
        y = np.asarray(D[k], float)
        r = dict(full=ev(full, y), timing=ev(T, y), amp=ev(Am, y))
        r["unique_timing"] = r["full"] - r["amp"]
        r["pairs"] = sorted(((ev(T[:, [2 * i, 2 * i + 1]], y), pr)
                             for i, pr in enumerate(tl)), key=lambda t: -t[0])
        grp = []
        for g in names:
            keep = np.ones(full.shape[1], bool)
            for i, pr in enumerate(tl):
                if g in pr:
                    keep[[2 * i, 2 * i + 1]] = False
            grp.append((r["full"] - ev(full[:, keep], y), g))
        r["pathways"] = sorted(grp, key=lambda t: -t[0])
        res[k] = r
        if verbose:
            pa = ", ".join(f"{a}-{b} {d:.2f}" for d, (a, b) in r["pairs"][:top])
            gr = ", ".join(f"{g} {d:+.2f}" for d, g in r["pathways"][:top])
            print(f"    {k:16s} R2 full {r['full']:.3f}  timing {r['timing']:.3f}  "
                  f"amp {r['amp']:.3f}  unique timing {r['unique_timing']:+.3f}"
                  f"\n{'':22s}pair alone: {pa}\n{'':22s}pathway: {gr}")
    return res


def steering(S, Hf, z, hz, axes, keys, delta=0.2):
    """Per piece: d(descriptor)/d(phase advance of that piece), mean and mean-|.| over z.

    Advancing piece k by delta adds H_k z_k (e^{i delta} - 1) to the pattern, so the
    derivative costs one extra pattern per piece per draw and no simulation. The signed
    mean says which way bringing that input earlier pushes the waves on average; the
    mean magnitude says how strongly its timing steers them at all."""
    N, K = z.shape
    base = Hf @ z.T                                                  # (nV, N)
    sgn, mag = ({k: np.zeros(K) for k in keys} for _ in range(2))
    e = np.exp(1j * delta) - 1
    for k in range(K):
        Phi = (base + np.outer(Hf[:, k], z[:, k] * e)).T
        Dp = describe(S, Phi, hz, axes)
        Phi = (base - np.outer(Hf[:, k], z[:, k] * e)).T
        Dm = describe(S, Phi, hz, axes)
        for nm in keys:
            d = (Dp[nm] - Dm[nm]) / (2 * delta)
            sgn[nm][k], mag[nm][k] = d.mean(), np.abs(d).mean()
    return sgn, mag


# ============================================================ the three commands

KEYS = ("travel", "planar", "coherence", "div", "curl", "n_sing", "align_anterior", "align_dorsal",
        "align_outward")


def _summ(D, keys=KEYS):
    return "  ".join(f"{k.replace('align_', '>')} {np.mean(D[k]):+.3f}" for k in keys)


def cmd_pairs(a, c, S, run):
    """Two pathways, one lag: what the medium does with the timing alone."""
    groups = pathway_groups(c, run["labels"], len(run["tags"]))
    w = (run["profiles"].astype(float) * np.asarray(c.A)[None, :]).sum(1)
    names = sorted(set(groups) - {"?"})
    pairs = ([tuple(p.split(":")) for p in a.pairs.split(",")] if a.pairs
             else [(names[i], names[j]) for i in range(len(names))
                   for j in range(i + 1, len(names))])
    hz = np.asarray([float(v) for v in a.hz.split(",")])
    H = transfer_fields(c, run, hz)
    lags = np.linspace(-np.pi, np.pi, a.nlag, endpoint=False)
    terr = {g: np.flatnonzero(np.isin(np.asarray(run["labels"]),
                                      [k for k, gg in enumerate(groups) if gg == g]))
            for g in names}
    dist = {g: S.distance_from(v) for g, v in terr.items()}
    base_axes = axes_for(S, c, run)
    out = {}
    print(f"\n  {len(pairs)} pathway pairs x {len(hz)} frequencies x {a.nlag} lags; "
          f"+lag means the SECOND pathway lags, so a wave from first to second is "
          f"align > 0. Lags are phase, not seconds: at this medium's 1.5 mm/s a "
          f"conduction delay between territories is often over half a period, and wraps")
    for gi, gj in pairs:
        ax = dict(base_axes, pair=S.axis(dist[gi] - dist[gj]))
        rec = {}
        for fi, f in enumerate(hz):
            zi = np.asarray([g == gi for g in groups], float) * w
            zj = np.asarray([g == gj for g in groups], float) * w
            zi, zj = zi / zi.sum(), zj / zj.sum()                    # equal net injection
            Z = zi[None, :] + zj[None, :] * np.exp(-1j * lags)[:, None]
            D = describe(S, (H[fi] @ Z.T).T, f, ax)
            # where the two fields overlap: the only place their timing can interfere,
            # away from each territory's own outward emission
            ov = np.abs(H[fi] @ zi) * np.abs(H[fi] @ zj)
            Do = describe(S, (H[fi] @ Z.T).T, f, ax, vweight=ov / ov.max())
            D.update({f"overlap_{k}": v for k, v in Do.items()})
            rec[f] = D
            al, ao = D["align_pair"], D["overlap_align_pair"]
            print(f"    {gi:>13s} -> {gj:<13s} {f:.3f} Hz: overlap align "
                  f"{ao.min():+.2f}..{ao.max():+.2f} peak at phase lag "
                  f"{np.degrees(lags[ao.argmax()]):+4.0f} deg; whole cortex "
                  f"{al.min():+.2f}..{al.max():+.2f}; travel "
                  f"{D['travel'].min():.2f}..{D['travel'].max():.2f}")
        out[(gi, gj)] = rec
    path = os.path.join(RESULTS, f"waves_pairs_{run['name']}.npz")
    np.savez(path, hz=hz, lags=lags, pairs=np.array(pairs, dtype=object),
             rec=np.array([out], dtype=object))
    print(f"\n  wrote {path}")
    _plot_pairs(path.replace(".npz", ".png"), out, hz, lags)


def cmd_draws(a, c, S, run):
    """Draw z from the solved S at its strongest bins and attribute the waves."""
    groups = pathway_groups(c, run["labels"], len(run["tags"]))
    w = (run["profiles"].astype(float) * np.asarray(c.A)[None, :]).sum(1)
    Sx, idx, ref = run["S"], run["idx"], run["ref_frames"]
    if Sx is None:
        if not a.synthetic:
            raise SystemExit("  draws needs a solved S: --tag, or --synthetic with "
                             "--standin to exercise the pipeline")
        Sx, idx, ref = synthetic_S(groups, w, run, a.synthetic)
    hz_bins = np.asarray(idx, float) / ref / run["frame_s"]
    lo, hi = (float(v) for v in a.band.split(","))
    inb = np.flatnonzero((hz_bins >= lo) & (hz_bins <= hi))
    power = np.array([np.trace(Sx[b]).real for b in inb])
    pick = inb[np.argsort(power)[::-1][:a.bands]]
    pick = pick[np.argsort(hz_bins[pick])]
    print(f"\n  {len(pick)} strongest solved bins in {lo}-{hi} Hz: "
          + ", ".join(f"{hz_bins[b]:.4f}" for b in pick))
    H = transfer_fields(c, run, hz_bins[pick])
    axes = axes_for(S, c, run)
    rng = np.random.default_rng(a.seed)
    res = []
    for n, b in enumerate(pick):
        f = hz_bins[b]
        Sb = 0.5 * (Sx[b] + Sx[b].conj().T)
        ev, U = np.linalg.eigh(Sb)
        o = np.argsort(ev)[::-1]
        ev, U = np.clip(ev[o], 0, None), U[:, o]
        L = U * np.sqrt(ev)
        K = len(ev)
        eta = (rng.standard_normal((a.draws, K)) + 1j * rng.standard_normal((a.draws, K)))
        z = (eta / np.sqrt(2)) @ L.T                                  # (N, K) ~ CN(0, S)
        Hf = H[n]
        D = describe(S, (Hf @ z.T).T, f, axes)
        # the same draws with every piece put in phase: identical loudness, no timing
        D0 = describe(S, (Hf @ np.abs(z).T).T, f, axes)
        lab = wave_type(D)
        types, cnt = np.unique(lab, return_counts=True)
        print(f"\n  {f:.4f} Hz (period {1/f:.0f} s), {a.draws} draws from S, "
              f"{100*ev[0]/max(ev.sum(),1e-30):.0f}% in the first coalition")
        print(f"    as drawn   : {_summ(D)}")
        print(f"    in phase   : {_summ(D0)}")
        print("    types      : " + ", ".join(f"{t} {100*m/len(lab):.0f}%"
                                              for t, m in zip(types, cnt)))
        # each coalition read as a pattern, with and without its phases
        for i in range(min(a.modes, K)):
            Di = describe(S, np.stack([Hf @ U[:, i], Hf @ np.abs(U[:, i])]), f, axes)
            print(f"    coalition {i} ({100*ev[i]/max(ev.sum(),1e-30):4.1f}%): "
                  + "  ".join(f"{k.replace('align_', '>')} {Di[k][0]:+.2f}"
                              f"({Di[k][1]:+.2f})" for k in KEYS))
        Zg, names = group_amplitudes(z, groups, w)
        print("    what explains each descriptor across draws (cross-validated R2):")
        att = attribute(D, Zg, names, KEYS)
        sgn = mag = None
        if a.steer:
            ns = min(a.steer, a.draws)
            sgn, mag = steering(S, Hf, z[:ns], f, axes, KEYS)
            for k in ("align_anterior", "align_outward", "div"):
                o = np.argsort(mag[k])[::-1][:6]
                print(f"    timing steers {k} most through: "
                      + ", ".join(f"{run['tags'][j]} ({groups[j]}) {sgn[k][j]:+.3f}"
                                  for j in o))
        res.append(dict(hz=f, D=D, D0=D0, types=lab, att=att, sgn=sgn, mag=mag,
                        ev=ev, names=names))
    path = os.path.join(RESULTS, f"waves_draws_{run['name']}.npz")
    np.savez(path, res=np.array(res, dtype=object), tags=np.array(run["tags"], object),
             groups=np.array(groups, object))
    print(f"\n  wrote {path}")
    _plot_draws(path.replace(".npz", ".png"), res, c, S, run)


def synthetic_S(groups, w, run, spec):
    """A known S for exercising `draws` on the stand-in: two latent sources, each driving
    a set of pathways with fixed lags, plus independent noise per piece.

    spec "A:B:lag_s,C:D:lag_s" - pathway B follows A by lag_s seconds in source 1, and so
    on. The attribution should then find the A-B and C-D phase differences and little
    else."""
    rng = np.random.default_rng(0)
    K = len(groups)
    ref = 1024
    hz = np.geomspace(0.008, 0.1, 24)
    idx = np.unique(np.round(hz * ref * run["frame_s"]).astype(int))
    S = np.zeros((len(idx), K, K), complex)
    for n, b in enumerate(idx):
        f = b / ref / run["frame_s"]
        for src in spec.split(","):
            ga, gb, lag = src.split(":")
            u = np.zeros(K, complex)
            u[[g == ga for g in groups]] = 1.0
            u[[g == gb for g in groups]] = np.exp(-2j * np.pi * f * float(lag))
            u *= np.exp(0.3j * rng.standard_normal(K)) * (u != 0)    # within-pathway jitter
            S[n] += np.outer(u, u.conj())
        S[n] += 0.2 * np.eye(K)
    return S, idx, ref


def cmd_realisation(a, c, S, run):
    """The same attribution on a simulated run, narrowband, in time."""
    from scipy.signal import hilbert
    import bandpass
    groups = pathway_groups(c, run["labels"], len(run["tags"]))
    w = (run["profiles"].astype(float) * np.asarray(c.A)[None, :]).sum(1)
    fr = np.asarray(np.load(a.frames, mmap_mode="r"), np.float32)
    A = np.asarray(np.load(a.drive), np.float64)
    save, fs = run["save"], run["frame_s"]
    nfr = min(len(fr), len(A) // save)
    a_fr = A[:nfr * save].reshape(nfr, save, -1).sum(1)
    f0 = float(a.hz)
    lo, hi = f0 / (1 + a.rel_bw), f0 * (1 + a.rel_bw)
    print(f"\n  {nfr} frames, narrowband {lo:.4f}-{hi:.4f} Hz around {f0} Hz")
    Phi = hilbert(bandpass.apply(fr[:nfr], fs, lo, hi), axis=0)
    Za = hilbert(bandpass.apply(a_fr, fs, lo, hi), axis=0)
    e = int(a.edge_s / fs)
    keep = np.arange(e, nfr - e, max(1, int(a.every_s / fs)))
    axes = axes_for(S, c, run)
    D = describe(S, Phi[keep], f0, axes)
    lab = wave_type(D)
    types, cnt = np.unique(lab, return_counts=True)
    print(f"    {len(keep)} frames read: {_summ(D)}")
    print("    types: " + ", ".join(f"{t} {100*m/len(lab):.0f}%" for t, m in zip(types, cnt)))
    Zg, names = group_amplitudes(Za[keep], groups, w)
    # A narrowband analytic signal decorrelates over ~1/bandwidth, so a run holds about
    # duration x bandwidth independent states however finely it is sampled. The full
    # model has 2 x n(n-1)/2 + n columns; below that many states it cannot generalise,
    # and the per-pair R2 (two columns each) is the only part worth reading.
    n_eff = (len(keep) * a.every_s) * (hi - lo)
    ncol = len(names) * (len(names) - 1) + len(names)
    print(f"    ~{n_eff:.0f} independent narrowband states against {ncol} columns"
          + ("  - TOO FEW for the full model; run longer, or widen --rel-bw"
             if n_eff < 2 * ncol else ""))
    print("    what explains each descriptor across frames (cross-validated R2):")
    att = attribute(D, Zg, names, KEYS, blocked=True)
    path = os.path.join(RESULTS, f"waves_real_{os.path.basename(a.frames)[:-4]}_{f0:g}.npz")
    np.savez(path, D=np.array([D], object), att=np.array([att], object), keep=keep,
             types=lab, hz=f0)
    print(f"\n  wrote {path}")


# ============================================================ pictures

def _plot_pairs(path, out, hz, lags):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    prs = list(out)
    nc = min(len(prs), 7)
    nr = int(np.ceil(len(prs) / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(2.3 * nc, 2.0 * nr), squeeze=False,
                            sharex=True, sharey=True)
    for ax, pr in zip(axs.ravel(), prs):
        M = np.stack([out[pr][f]["overlap_align_pair"] for f in hz])
        ax.imshow(M, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-1, vmax=1,
                  extent=(-180, 180, -0.5, len(hz) - 0.5))
        ax.set_title(f"{pr[0]}\n-> {pr[1]}", fontsize=7)
        ax.set_yticks(range(len(hz)))
        ax.set_yticklabels([f"{f:.3f}" for f in hz], fontsize=6)
        ax.tick_params(labelsize=6)
    for ax in axs.ravel()[len(prs):]:
        ax.axis("off")
    fig.supxlabel("phase lag of the second pathway (deg)", fontsize=8)
    fig.supylabel("Hz", fontsize=8)
    fig.suptitle("propagation along the first -> second axis where their fields overlap "
                 "(red: first to second)",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor="white")
    plt.close(fig)
    print(f"  wrote {path}")


def _plot_draws(path, res, c, S, run):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = [k for k in KEYS if k != "n_sing"]
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.2))
    x = np.arange(len(keys))
    for r in res:
        axs[0].plot(x, [r["att"][k]["unique_timing"] for k in keys], "o-",
                    label=f"{r['hz']:.3f} Hz", ms=3)
        axs[1].plot(x, [r["att"][k]["amp"] for k in keys], "o-", ms=3)
    for ax, t in zip(axs, ("unique to relative timing", "relative loudness alone")):
        ax.set_xticks(x)
        ax.set_xticklabels([k.replace("align_", "->") for k in keys], rotation=30,
                           fontsize=7)
        ax.set_title(f"cross-validated R2: {t}", fontsize=9)
        ax.set_ylim(0, 1)
    axs[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor="white")
    plt.close(fig)
    print(f"  wrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=("pairs", "draws", "realisation"))
    ap.add_argument("--tag", default="grclip100_nolag",
                    help="a best_fit solve: results/xspec_<tag>.npz")
    ap.add_argument("--standin", default="",
                    help="decay_s,spread_mm_s: a flat medium on that clock instead of "
                         "the solved one, with --basis for the pieces")
    ap.add_argument("--basis", default=os.path.join(RESULTS, "sub_first_area_100.npz"))
    ap.add_argument("--radius", type=float, default=10.0,
                    help="mm, the disc planarity is measured over")
    ap.add_argument("--hz", default="0.02,0.04,0.06")
    ap.add_argument("--pairs", default="",
                    help="pairs: comma-separated A:B pathway pairs; default all")
    ap.add_argument("--nlag", type=int, default=24)
    ap.add_argument("--band", default="0.01,0.08")
    ap.add_argument("--bands", type=int, default=6)
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--modes", type=int, default=3)
    ap.add_argument("--steer", type=int, default=100,
                    help="draws used for the per-piece steering derivative; 0 skips it")
    ap.add_argument("--synthetic", default="",
                    help="draws with --standin: a known S, e.g. LGN_V1:VPL_S1:4")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frames", default="", help="realisation: raw field (T, nV) npy")
    ap.add_argument("--drive", default="",
                    help="realisation: per-step drive npy; default drive_<tag>.npy")
    ap.add_argument("--rel-bw", type=float, default=0.25, dest="rel_bw")
    ap.add_argument("--edge-s", type=float, default=60.0, dest="edge_s")
    ap.add_argument("--every-s", type=float, default=1.0, dest="every_s")
    a = ap.parse_args()

    from mesh_cache import load_cortex
    c = load_cortex("fsaverage5", verbose=False)
    run = load_run(c, None if a.standin else a.tag, a.standin or None, a.basis)
    S = Surface(c, a.radius)
    print(f"  {run['name']}: {len(run['tags'])} pieces, frame {run['frame_s']:.4f} s, "
          f"save {run['save']}, impulse window {run['nimp']} frames")
    if a.cmd == "pairs":
        cmd_pairs(a, c, S, run)
    elif a.cmd == "draws":
        cmd_draws(a, c, S, run)
    else:
        if not a.frames:
            raise SystemExit("  realisation needs --frames")
        a.drive = a.drive or os.path.join(RESULTS, f"drive_{a.tag}.npy")
        cmd_realisation(a, c, S, run)


if __name__ == "__main__":
    main()
