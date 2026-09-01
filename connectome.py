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


def surrogate_W(W, D, nbins=10, nswap=200, seed=0, match_scale=True, verbose=True):
    """A connectome matched to W in everything except which pairs are connected.

    The control the coupling experiment needs. Without it, "structural connectivity
    improves the fit" cannot be told apart from "any long-range redistribution improves
    the fit" - and the coverage result already said this medium is short of transport, so
    that alternative is the live one, not a pedantic one.

    Degree-preserving double-edge swaps, (a,b),(c,d) -> (a,d),(c,b), restricted so that
    BOTH new edges fall in the same geodesic-distance bin as the pair that produced them.
    Degree sequence is then exact (every swap leaves all four degrees unchanged), the
    per-bin edge count is exact (by the acceptance rule), and the weight multiset is exact
    (weights travel with their edges). What is destroyed is topology: which specific
    parcels are wired to which.

    `synthetic_matrix` is NOT this - it draws pairs with a distance-biased probability and
    preserves neither degree nor weights, so it can only exercise the machinery.

    `match_scale` rescales the result so its Laplacian norm equals W's, which is what
    makes a single `lam` mean the same thing for both. _lap_norm is homogeneous of degree
    one in W, so this is an exact match, not an approximation."""
    rng = np.random.default_rng(seed)
    n = W.shape[0]
    iu = np.triu_indices(n, 1)
    sel = W[iu] > 0
    ea, eb, ew = iu[0][sel].copy(), iu[1][sel].copy(), W[iu][sel].copy()
    ed = D[ea, eb]
    nE = len(ea)
    if nE < 4:
        raise ValueError(f"only {nE} edges to rewire")
    # Equal-count bins, but the bin count has to fall with the edge count or the
    # constraint becomes unsatisfiable: requiring BOTH swapped edges to stay inside a
    # narrow bin is nearly impossible when a bin holds only a handful of edges, and the
    # result is a "surrogate" that is mostly the original graph. ~20 edges per bin is
    # what makes the acceptance rate usable on the 82-edge filtered connectome.
    nbins = int(max(2, min(nbins, nE // 20)))
    qs = np.quantile(ed, np.linspace(0, 1, nbins + 1))
    qs[0], qs[-1] = -np.inf, np.inf
    binof = np.clip(np.searchsorted(qs, ed, side="right") - 1, 0, nbins - 1)

    present = set()
    for a, b in zip(ea, eb):
        present.add((int(a), int(b)))

    done = 0
    for b in range(nbins):
        idx = np.flatnonzero(binof == b)
        if len(idx) < 2:
            continue
        lo, hi = qs[b], qs[b + 1]
        for _ in range(nswap * len(idx)):
            p1, p2 = rng.choice(idx, 2, replace=False)
            a, bb = ea[p1], eb[p1]
            cc, d = ea[p2], eb[p2]
            if rng.random() < 0.5:                       # both pairings, not just one
                cc, d = d, cc
            if len({int(a), int(bb), int(cc), int(d)}) < 4:
                continue                                 # would make a self-loop
            n1 = (min(a, d), max(a, d))
            n2 = (min(cc, bb), max(cc, bb))
            if (int(n1[0]), int(n1[1])) in present or (int(n2[0]), int(n2[1])) in present:
                continue                                 # would double an existing edge
            if not (lo <= D[n1] < hi and lo <= D[n2] < hi):
                continue                                 # would leave the distance bin
            present.discard((int(min(a, bb)), int(max(a, bb))))
            present.discard((int(min(cc, d)), int(max(cc, d))))
            present.add((int(n1[0]), int(n1[1])))
            present.add((int(n2[0]), int(n2[1])))
            ea[p1], eb[p1] = n1
            ea[p2], eb[p2] = n2
            done += 1

    out = np.zeros_like(W)
    out[ea, eb] = ew
    out = out + out.T
    if match_scale:
        out *= _lap_norm(W) / _lap_norm(out)
    keep_frac = float(((out > 0) & (W > 0)).sum()) / max(float((W > 0).sum()), 1.0)
    if keep_frac > 0.5:
        print(f"  WARNING: surrogate shares {keep_frac:.0%} of its edges with the real "
              f"connectome, so it is a weak control. Raise nswap or lower nbins.")
    if verbose:
        deg_ok = np.allclose(np.sort((out > 0).sum(1)), np.sort((W > 0).sum(1)))
        print(f"  surrogate: {done} accepted swaps over {nE} edges ({done/nE:.1f} per "
              f"edge), degree sequence preserved: {deg_ok}")
        print(f"    {nbins} distance bins; median distance {np.median(D[out > 0]):.0f} mm "
              f"(real {np.median(D[W > 0]):.0f} mm), "
              f"{int((out > 0).sum())//2} edges (real {nE}), "
              f"topology overlap {keep_frac:.1%}")
    return out


class CouplingOperator:
    """h -> lam * Rt( W (R h)(t-lag) - deg * (R h)(t-lag) ), the long-range depth term.

    R is the area-weighted parcel mean, so `R h` is the parcel-average depth and Rt puts a
    parcel's increment back uniformly over its vertices. The Laplacian form means a
    spatially uniform field produces exactly zero, so the term cannot pump energy in on
    its own; what it does is move depth between parcels that white matter connects.

    `lag` delays that transport by a whole number of STEPS. A single uniform delay, not a
    distance rule: the question it answers is whether any lag helps before whether the
    right one does. Delaying a linear term leaves the system linear AND time invariant,
    so the convex solve is untouched and only H changes - which is the only reason a lag
    is affordable here at all.

    What is buffered is the 180 PARCEL MEANS, not the 9,374-vertex field, so a lag of a
    thousand steps costs a few hundred kilobytes. The operator is therefore STATEFUL, and
    that is a trap: every integration loop must call reset() before it starts, or one
    impulse response leaks into the next. See reset()."""

    def __init__(self, cortex, W, lam, lag=0, dtype=np.float32):
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
        self.lag = int(lag)
        if self.lag < 0:
            raise ValueError(f"lag must be >= 0, got {lag}")
        self.buf = np.zeros((self.lag + 1, N_PARCELS), dtype)
        self.ptr = 0

    def reset(self):
        """Clear the delay buffer. MUST be called before each independent trajectory.

        The serial impulse loop in xspec.impulse_responses builds the solver once and
        reuses it across all K pieces, and the parallel path reuses one operator per
        worker across its share of them - so without this, piece k+1 starts inside the
        tail of piece k. That failure is silent: the responses stay finite and plausible,
        H simply stops describing the system, and it shows up only as a worse score."""
        self.buf[:] = 0.0
        self.ptr = 0

    def parcel_mean(self, h):
        return np.array([float(self.wts[p] @ h[q]) if len(q) else 0.0
                         for p, q in enumerate(self.idx)], self.dtype)

    def __call__(self, h):
        m = self.parcel_mean(h)
        # write the current mean, advance, then read: after the advance the write head
        # sits on the OLDEST entry, which is the one written `lag` steps ago. At lag=0
        # the ring is one row and this is the identity, bit for bit.
        self.buf[self.ptr] = m
        self.ptr = (self.ptr + 1) % (self.lag + 1)
        md = self.buf[self.ptr]
        inc = self.lam * (self.W @ md - self.deg * md)
        out = np.zeros(self.nV, self.dtype)
        for p, q in enumerate(self.idx):
            if len(q):
                out[q] = inc[p]
        return out

    def spectral_bound(self):
        """Largest |eigenvalue| of the parcel-level Laplacian times lam.

        The term is explicit, so dt * bound has to stay well under 1 or the step is
        unstable - the same kind of condition the CFL bound imposes on the wave part.

        This bound is NECESSARY BUT NOT SUFFICIENT once lag > 0. It is derived from the
        Laplacian being negative semidefinite, which makes the instantaneous term purely
        dissipative; a delayed feedback carries no such guarantee and can pump energy at
        frequencies where the delay turns dissipation into gain. So a lagged medium needs
        the empirical energy check in `main`, not just this number."""
        L = np.diag(self.deg.astype(float)) - self.W.astype(float)
        return float(self.lam) * float(np.abs(np.linalg.eigvalsh(L)).max())

    def key(self):
        """Short hash for the impulse cache. Responses computed with one coupling must
        never be reused for another, and W is far too big to put in a filename."""
        hh = hashlib.sha1(np.ascontiguousarray(self.W, np.float64).tobytes()).hexdigest()
        # lag appears only when set, so responses cached before lags existed stay valid -
        # at lag=0 the operator is bit-identical to the version that wrote them
        tag = f"cpl{float(self.lam):.6g}_"
        if self.lag:
            tag += f"L{self.lag}_"
        return tag + hh[:10]


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
    ap.add_argument("--lag", type=int, default=0,
                    help="uniform delay on the long-range term, in STEPS")
    ap.add_argument("--surrogate", type=int, default=None, metavar="SEED",
                    help="use a degree- and distance-matched rewiring instead of the "
                         "real topology; the control, see surrogate_W")
    ap.add_argument("--energy-steps", type=int, default=4000, dest="energy_steps",
                    help="length of the empirical stability check")
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
    if a.surrogate is not None:
        Wr = surrogate_W(Wr, D, seed=a.surrogate)

    p, save, _ = bo_step.unpack(BEST_X, c)
    s, dt, g, Hf = fl.build(c, p)
    print(f"  dt {dt:.4g}")
    for lam in (0.0, a.lam, 10 * a.lam, 100 * a.lam):
        op = CouplingOperator(c, Wr, lam, a.lag)
        b = op.spectral_bound()
        print(f"  lam {lam:<8.4g} spectral bound {b:9.4g}   dt*bound {dt*b:9.4g}"
              + ("   (explicit term needs << 1)" if lam == 0.0 else ""))

    if a.check:
        rng = np.random.default_rng(0)
        h = rng.standard_normal(c.nV).astype(np.float32)
        ue = np.zeros(s.nE, np.float32)
        dtD = np.float32(dt)
        s.coupling = None
        u0, h0 = s.step(ue.copy(), h.copy(), dtD, g, Hf)
        s.coupling = CouplingOperator(c, Wr, 0.0, a.lag)
        u1, h1 = s.step(ue.copy(), h.copy(), dtD, g, Hf)
        print(f"\n  lam=0 identical to no coupling: "
              f"{np.abs(h1 - h0).max() == 0.0 and np.abs(u1 - u0).max() == 0.0}")
        # has to be run for lag+1 steps, not 1: at lag L the buffer is still zeros for
        # the first L steps, so a one-step check reports "no effect" for a term that has
        # simply not arrived yet
        op1 = CouplingOperator(c, Wr, a.lam, a.lag); op1.reset()
        s.coupling = op1
        uu, h2 = ue.copy(), h.copy()
        for _ in range(a.lag + 1):
            uu, h2 = s.step(uu, h2, dtD, g, Hf)
        s.coupling = None
        uu, hb = ue.copy(), h.copy()
        for _ in range(a.lag + 1):
            uu, hb = s.step(uu, hb, dtD, g, Hf)
        print(f"  lam={a.lam} changes the field after lag+1={a.lag+1} steps: "
              f"{np.abs(h2 - hb).max():.3g}")
        flat = np.ones(c.nV, np.float32)
        op_flat = CouplingOperator(c, Wr, a.lam, 0)      # lag 0: the term itself, undelayed
        print(f"  uniform field is a fixed point of the term: "
              f"{np.abs(op_flat(flat)).max():.3g}")

        # ---- lag=0 must reproduce the unlagged operator exactly, step for step
        if a.lag:
            def trajectory(op, nsteps, seed=1):
                r = np.random.default_rng(seed)
                hh = r.standard_normal(c.nV).astype(np.float32)
                uu = np.zeros(s.nE, np.float32)
                s.coupling = op
                if op is not None:
                    op.reset()
                for _ in range(nsteps):
                    uu, hh = s.step(uu, hh, dtD, g, Hf)
                return hh
            d0 = trajectory(CouplingOperator(c, Wr, a.lam, 0), 40)
            dL = trajectory(CouplingOperator(c, Wr, a.lam, a.lag), 40)
            print(f"  lag={a.lag} differs from lag=0 over 40 steps: "
                  f"{np.abs(dL - d0).max():.3g}  (must be > 0, or the buffer is inert)")

        # ---- the reset test: a reused operator must match a fresh one
        r = np.random.default_rng(2)
        prof = [r.standard_normal(c.nV).astype(np.float32) for _ in range(2)]

        def run_from(op, h0v, nsteps):
            uu = np.zeros(s.nE, np.float32)
            hh = h0v.copy()
            s.coupling = op
            for _ in range(nsteps):
                uu, hh = s.step(uu, hh, dtD, g, Hf)
            return hh

        op = CouplingOperator(c, Wr, a.lam, a.lag)
        reused = []
        for k in range(2):
            op.reset()                                  # what the loops now do
            reused.append(run_from(op, prof[k], 30))
        fresh = [run_from(CouplingOperator(c, Wr, a.lam, a.lag), prof[k], 30)
                 for k in range(2)]
        err = max(float(np.abs(reused[k] - fresh[k]).max()) for k in range(2))
        print(f"  reused operator + reset == fresh operator: {err == 0.0} "
              f"(max |diff| {err:.3g})")

        op2 = CouplingOperator(c, Wr, a.lam, a.lag)
        noreset = [run_from(op2, prof[k], 30) for k in range(2)]   # deliberately no reset
        leak = float(np.abs(noreset[1] - fresh[1]).max())
        print(f"  without reset, piece 2 is contaminated by piece 1 by {leak:.3g}"
              + ("   <- so the reset is load-bearing" if leak > 0 else
                 "   (zero at lag=0, as expected: the operator is stateless there)"))

        # ---- empirical stability: a delayed feedback is not covered by spectral_bound
        op3 = CouplingOperator(c, Wr, a.lam, a.lag)
        op3.reset()
        s.coupling = op3
        rr = np.random.default_rng(3)
        hh = rr.standard_normal(c.nV).astype(np.float32)
        uu = np.zeros(s.nE, np.float32)
        area = np.asarray(c.A, float)
        e0 = None
        for n in range(a.energy_steps):
            uu, hh = s.step(uu, hh, dtD, g, Hf)
            if n % (a.energy_steps // 4) == 0 or n == a.energy_steps - 1:
                e = float(area @ (hh.astype(float) ** 2))
                if e0 is None:
                    e0 = max(e, 1e-300)
                print(f"    step {n:6d}  depth energy {e:.4e}  ({e/e0:8.3e} x start)"
                      + ("   NOT FINITE" if not np.isfinite(e) else ""))
        print(f"  energy bounded over {a.energy_steps} steps at lam={a.lam}, "
              f"lag={a.lag}: {np.isfinite(e) and e <= e0}")


if __name__ == "__main__":
    main()
