"""Long-range structural connections, as a second transport term on depth.

The fluid moves signal by propagating across the sheet, and the coverage experiment
suggested it does not move it far enough: at matched driven area, spreading the input over
the cortex beat concentrating it in sensory areas, which is what you would expect if
distant FC structure has to be supplied by putting a source near it rather than by
transporting anything there. White matter is the transport the sheet is missing.

The term is linear and instantaneous:

    dh/dt += lam * Rt ( W (R h) - deg * (R h) )

with R the area-weighted parcel average (180 x nV) and W the connectome restricted to
connections a distance rule does not already predict. Written in graph-Laplacian form, so
a uniform field is a fixed point and the term redistributes rather than creates.

Two properties matter more than the details. It is LINEAR, so the system stays LTI and the
convex solve needs no change at all - only H changes. And it is LOW RANK, applied through
180 parcel means rather than 9,374 vertices, so it costs nothing against the step it sits
in.

  python connectome.py --synthetic --check     # stability and the lam=0 identity
"""
import os, hashlib, argparse
import numpy as np

from paths import CACHE

N_PARCELS = 180                     # left hemisphere, Glasser; labels 1..180


def load_matrix(path, cortex, verbose=True):
    """Load a structural connectome and put it in cortex parcel order.

    Accepts .npy/.npz/.csv/.txt. The ordering is CHECKED rather than assumed: a matrix
    saved in a different parcel order is not detectable from its values, and would give a
    plausible-looking but meaningless coupling. If the file carries parcel names they are
    matched against cortex.names; otherwise the caller must confirm the convention."""
    if path.endswith(".npz"):
        z = np.load(path, allow_pickle=True)
        W = np.asarray(z["W"] if "W" in z else z[z.files[0]], float)
        names = [str(x) for x in z["names"]] if "names" in z else None
    elif path.endswith(".npy"):
        W, names = np.asarray(np.load(path), float), None
    else:
        W, names = np.loadtxt(path, delimiter=","), None

    if W.shape[0] == 2 * N_PARCELS:                 # bilateral: take the left block
        W = W[:N_PARCELS, :N_PARCELS]
        if verbose:
            print(f"  bilateral {2*N_PARCELS} matrix -> left-hemisphere block")
    if W.shape != (N_PARCELS, N_PARCELS):
        raise ValueError(f"expected {N_PARCELS}x{N_PARCELS}, got {W.shape}")

    if names is not None:
        want = [cortex.names[i] for i in range(1, N_PARCELS + 1)]
        order = _match_names(names, want)
        W = W[np.ix_(order, order)]
        if verbose:
            print(f"  reordered to cortex.names by parcel name")
    elif verbose:
        print("  no parcel names in the file: assuming rows are Glasser 1..180 in "
              "cortex.names order - verify this against the source before trusting it")
    W = 0.5 * (W + W.T)
    np.fill_diagonal(W, 0.0)
    return W


def _match_names(have, want):
    """Index array putting `have` into `want` order, matching on parcel name.

    Two traps, both silent if you get them wrong:

    The hemisphere prefix is KEPT. Stripping it makes L_V1 and R_V1 the same key, one
    overwrites the other in the lookup, and every left parcel resolves to its right-
    hemisphere twin - a full 180-parcel matrix of the wrong hemisphere, which looks
    entirely normal.

    The _ROI suffix is STRIPPED by slicing, not by replace. A bare replace of "L_" also
    eats the one inside 7AL_ROI, PSL_ROI, SFL_ROI, 5L_ROI and 7PL_ROI - every area whose
    name ends in L."""
    def key(s):
        s = s.strip()
        for suf in ("_ROI", "-ROI"):
            if s.endswith(suf):
                s = s[:-len(suf)]
                break
        s = s.replace("lh.", "L_").replace("rh.", "R_")
        return s.lower()
    pos = {}
    for i, s in enumerate(have):
        pos.setdefault(key(s), i)                  # first wins; duplicates are a warning
    if len(pos) != len(have):
        raise ValueError(f"duplicate parcel names: {len(have)} entries, {len(pos)} unique")
    missing = [w for w in want if key(w) not in pos]
    if missing:
        raise ValueError(f"{len(missing)} parcels absent from the file, e.g. {missing[:5]}")
    return np.array([pos[key(w)] for w in want])


def load_enigma(cortex, normalise=True, verbose=True):
    """HCP-derived group-normative structural connectivity, Glasser 360, from the ENIGMA
    Toolbox (MRtrix3 anatomically-constrained tractography over unrelated HCP adults).

    It ships parcel labels, so the ordering is checked rather than assumed - and the check
    earns its keep: the first five ENIGMA labels coincide with cortex.names, which makes
    the ordering look right when it is not.

    `normalise` divides by the largest eigenvalue of the parcel Laplacian, so that lam is
    a dimensionless coupling strength rather than a number whose safe range depends on
    whatever units the tractography counted in."""
    from enigmatoolbox.datasets import load_sc
    W, names, _, _ = load_sc(parcellation="glasser_360")
    names = [str(x) for x in names]
    want = [cortex.names[i] for i in range(1, N_PARCELS + 1)]
    order = _match_names(names, want)
    W = np.asarray(W, float)[np.ix_(order, order)]
    W = 0.5 * (W + W.T)
    np.fill_diagonal(W, 0.0)
    if normalise:
        W = W / _lap_norm(W)
    if verbose:
        lo, hi = order.min(), order.max()
        print(f"  ENIGMA glasser_360: left block is rows {lo}-{hi} of 360, "
              f"{'contiguous' if hi - lo == N_PARCELS - 1 else 'scattered'}; "
              f"density {(W > 0).mean():.3f}"
              + (", Laplacian normalised" if normalise else ""))
    return W


def _lap_norm(W):
    """Largest eigenvalue of the parcel graph Laplacian of W."""
    L = np.diag(W.sum(1)) - W
    return max(float(np.abs(np.linalg.eigvalsh(L)).max()), 1e-30)


def synthetic_matrix(cortex, D, n_edges=400, seed=0, verbose=True):
    """A stand-in connectome: long-range pairs drawn with probability rising with
    distance. Not a model of anything - it exists so the coupling term, its stability
    bound and the cache keys can be exercised before the real matrix arrives."""
    rng = np.random.default_rng(seed)
    W = np.zeros((N_PARCELS, N_PARCELS))
    iu = np.triu_indices(N_PARCELS, 1)
    p = (D[iu] / D.max()) ** 3
    pick = rng.choice(len(p), size=min(n_edges, len(p)), replace=False, p=p / p.sum())
    W[iu[0][pick], iu[1][pick]] = rng.random(len(pick))
    W = W + W.T
    if verbose:
        print(f"  synthetic connectome: {len(pick)} pairs, median distance "
              f"{np.median(D[iu][pick]):.0f} mm")
    return W


def residual_W(W, D, keep=0.15, min_mm=60.0, verbose=True):
    """Keep the long connections that a distance rule does not already predict.

    Two filters, because either alone is insufficient. The fluid already transports over
    short range, so a short connection would largely duplicate it and any gain would be
    ambiguous - hence `min_mm`. And within the long connections, log(W) is regressed on
    distance and only the strongest positive residuals survive, which leaves the ones that
    are stronger than their length explains rather than simply the biggest.

    Filtering on residual alone does not give long connections: on the ENIGMA matrix the
    top residuals have almost exactly the median distance of all connected pairs."""
    iu = np.triu_indices(W.shape[0], 1)
    w, d = W[iu], D[iu]
    m = (w > 0) & (d >= min_mm)
    if m.sum() < 10:
        raise ValueError(f"only {int(m.sum())} edges beyond {min_mm} mm")
    lw = np.log(w[m] + 1e-12)
    A = np.c_[np.ones(m.sum()), d[m]]
    beta, *_ = np.linalg.lstsq(A, lw, rcond=None)
    res = np.full(len(w), -np.inf)
    res[m] = lw - A @ beta
    thr = np.quantile(res[m], 1.0 - keep)
    out = np.zeros_like(W)
    sel = res > thr
    out[iu[0][sel], iu[1][sel]] = w[sel]
    out = out + out.T
    if verbose:
        print(f"  beyond {min_mm:.0f} mm: {int(m.sum())} of {int((w>0).sum())} edges; "
              f"log w = {beta[0]:+.2f} {beta[1]:+.4f} * mm")
        print(f"  kept top {keep:.0%} by residual: {int(sel.sum())} edges, median "
              f"distance {np.median(d[sel]):.0f} mm (all connected {np.median(d[w>0]):.0f} mm)")
    return out


class CouplingOperator:
    """h -> lam * Rt( W (R h) - deg * (R h) ), the long-range term of the depth update.

    R is the area-weighted parcel mean, so `R h` is the parcel-average depth and Rt puts a
    parcel's increment back uniformly over its vertices. The Laplacian form means a
    spatially uniform field produces exactly zero, so the term cannot pump energy in on
    its own; what it does is move depth between parcels that white matter connects."""

    def __init__(self, cortex, W, lam, dtype=np.float32):
        lab = np.asarray(cortex.lab)
        area = np.asarray(cortex.A, float)
        self.idx = [np.flatnonzero(lab == p + 1) for p in range(N_PARCELS)]
        self.wts = []
        for q in self.idx:
            a = area[q]
            self.wts.append((a / a.sum()).astype(dtype) if len(q) else a.astype(dtype))
        self.W = np.asarray(W, float).astype(dtype)
        self.deg = self.W.sum(1).astype(dtype)
        self.lam = dtype(lam)
        self.nV = cortex.nV
        self.dtype = dtype

    def parcel_mean(self, h):
        return np.array([float(self.wts[p] @ h[q]) if len(q) else 0.0
                         for p, q in enumerate(self.idx)], self.dtype)

    def __call__(self, h):
        m = self.parcel_mean(h)
        inc = self.lam * (self.W @ m - self.deg * m)
        out = np.zeros(self.nV, self.dtype)
        for p, q in enumerate(self.idx):
            if len(q):
                out[q] = inc[p]
        return out

    def spectral_bound(self):
        """Largest |eigenvalue| of the parcel-level Laplacian times lam.

        The term is explicit, so dt * bound has to stay well under 1 or the step is
        unstable - the same kind of condition the CFL bound imposes on the wave part."""
        L = np.diag(self.deg.astype(float)) - self.W.astype(float)
        return float(self.lam) * float(np.abs(np.linalg.eigvalsh(L)).max())

    def key(self):
        """Short hash for the impulse cache. Responses computed with one coupling must
        never be reused for another, and W is far too big to put in a filename."""
        hh = hashlib.sha1(np.ascontiguousarray(self.W, np.float64).tobytes()).hexdigest()
        return f"cpl{float(self.lam):.6g}_{hh[:10]}"


def parcel_distances(cortex, verbose=True):
    """Geodesic distance between all 180 parcel centroids, cached."""
    cache = os.path.join(CACHE, f"parcel_D_{cortex.mesh}_{N_PARCELS}.npy")
    if os.path.exists(cache):
        return np.load(cache)
    import ladder
    D, _ = ladder.parcel_geodesic(cortex, list(range(1, N_PARCELS + 1)), verbose=verbose)
    np.save(cache, D)
    return D


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--enigma", action="store_true",
                    help="the HCP-derived ENIGMA Toolbox glasser_360 connectome")
    ap.add_argument("--path", default=None, help="connectome file")
    ap.add_argument("--lam", type=float, default=0.05)
    ap.add_argument("--keep", type=float, default=0.15)
    ap.add_argument("--min-mm", type=float, default=60.0, dest="min_mm",
                    help="shortest connection counted as long-range")
    ap.add_argument("--check", action="store_true", help="lam=0 identity and stability")
    a = ap.parse_args()

    from mesh_cache import load_cortex
    import fluid as fl, bo_step
    from best_fit import BEST_X
    c = load_cortex("fsaverage5", verbose=False)
    D = parcel_distances(c)
    if a.enigma:
        W = load_enigma(c)
    elif a.path:
        W = load_matrix(a.path, c)
    else:
        W = synthetic_matrix(c, D)
    Wr = residual_W(W, D, a.keep, a.min_mm)

    p, save, _ = bo_step.unpack(BEST_X, c)
    s, dt, g, Hf = fl.build(c, p)
    print(f"  dt {dt:.4g}")
    for lam in (0.0, a.lam, 10 * a.lam, 100 * a.lam):
        op = CouplingOperator(c, Wr, lam)
        b = op.spectral_bound()
        print(f"  lam {lam:<8.4g} spectral bound {b:9.4g}   dt*bound {dt*b:9.4g}"
              + ("   (explicit term needs << 1)" if lam == 0.0 else ""))

    if a.check:
        rng = np.random.default_rng(0)
        h = rng.standard_normal(c.nV).astype(np.float32)
        ue = np.zeros(s.nE, np.float32)
        s.coupling = None
        u0, h0 = s.step(ue.copy(), h.copy(), np.float32(dt), g, Hf)
        s.coupling = CouplingOperator(c, Wr, 0.0)
        u1, h1 = s.step(ue.copy(), h.copy(), np.float32(dt), g, Hf)
        print(f"\n  lam=0 identical to no coupling: "
              f"{np.abs(h1 - h0).max() == 0.0 and np.abs(u1 - u0).max() == 0.0}")
        s.coupling = CouplingOperator(c, Wr, a.lam)
        _, h2 = s.step(ue.copy(), h.copy(), np.float32(dt), g, Hf)
        print(f"  lam={a.lam} changes the field: {np.abs(h2 - h0).max():.3g}")
        flat = np.ones(c.nV, np.float32)
        print(f"  uniform field is a fixed point of the term: "
              f"{np.abs(s.coupling(flat)).max():.3g}")


if __name__ == "__main__":
    main()
