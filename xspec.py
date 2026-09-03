"""Best input cross-spectrum for a target FC, solved rather than searched.

The medium is linear, so the field covariance depends on the input only through its
cross-spectral density: C = sum_f H(f) S(f) H(f)^H, with H(f) the field response to each
driven region at frequency f. Matching FC is therefore a convex problem over S - a
Hermitian PSD matrix per frequency - and it has an optimum instead of a plateau.

S is built from coalitions: an event with signed amplitudes a and per-region time offsets
d contributes |g(f)|^2 * u u^H at each frequency, with u_k = a_k exp(-2i pi f d_k). Any
PSD S decomposes into such terms, so a library of damped coalitions is a complete
generating set for second-order structure, and the eigenvectors of the solved S read back
as the coalitions that realise it.

What is solved here is covariance matching, not the Spearman edge score the search uses,
so the number this produces is not the score - it is a candidate, to be simulated and
scored honestly afterwards.

  python xspec.py --sig0 0.01 --nfreq 32 --iters 150
"""
import os, argparse
import numpy as np

from mesh_cache import load_cortex
from input2 import parcel_tapers
from fc_score import FCTarget
from paths import CACHE
import fluid as fl

REGIONS = [8, 9, 51, 53, 1, 4, 5, 24, 173, 174, 124, 104, 150, 151, 132, 30, 33, 65, 88]
NSTEPS, SAVE = 7000, 25


def medium(sig0, c0=1.0, Ld=52.4, a=None, b=None, sponge_scale=1.0):
    return dict(mode="maps", maps=fl.MAPS_DEFAULT, c0=c0, Ld=Ld, sig0=sig0,
                a=np.zeros(3) if a is None else np.asarray(a, float),
                b=np.zeros(3) if b is None else np.asarray(b, float),
                sponge_scale=sponge_scale)


def normal_scores(C, iu=None):
    """Rank the edges of a centred FC matrix and map them onto gaussian quantiles.

    The score is Spearman, so what the solve should match is the target's edge ORDER, not
    its values. Normal-scoring fixes the target side of that: after it, Pearson against
    the transformed target is a much closer surrogate for Spearman against the raw one."""
    from scipy.stats import rankdata, norm
    if iu is None:
        iu = np.triu_indices(C.shape[0], 1)
    out = np.zeros_like(C)
    out[iu] = norm.ppf(rankdata(C[iu]) / (len(iu[0]) + 1.0))
    out = out + out.T
    return out - out.mean(0, keepdims=True) - out.mean(1, keepdims=True) + out.mean()


def medoid_subset(target, n=1000, sketch=400, seed=0):
    """Vertices chosen to represent the FC structure, not sampled at random.

    k-means on each vertex's FC profile (sketched against a few hundred random columns),
    then the medoid of each cluster. Fewer near-duplicate rows than a random draw, which
    both speeds the solve and generalises better to held-out vertices."""
    cache = os.path.join(CACHE, f"medoids_{n}_{sketch}_{seed}_{target.nV}.npy")
    if os.path.exists(cache):
        return np.load(cache)
    from sklearn.cluster import MiniBatchKMeans
    rng = np.random.default_rng(seed)
    FC = np.asarray(target.target_fc(), np.float32)
    sk = FC[:, rng.choice(target.nV, sketch, replace=False)]
    km = MiniBatchKMeans(n_clusters=n, random_state=seed, n_init=3, batch_size=2048).fit(sk)
    out = []
    for k in range(n):
        idx = np.flatnonzero(km.labels_ == k)
        if len(idx):
            out.append(idx[np.argmin(((sk[idx] - km.cluster_centers_[k]) ** 2).sum(1))])
    out = np.sort(np.unique(out))
    np.save(cache, out)
    return out


def validation_subset(target, sub, n=1000, seed=1):
    """Vertices the solve never sees, for early stopping.

    The solve fits 1,000 medoid vertices and leaves 8,217 unused, so a generalisation
    signal is already paid for. Drawn at random from the complement rather than by
    medoid, because these are meant to be a fair sample of what the solve is NOT fitting,
    not a second summary of the same structure."""
    rng = np.random.default_rng(seed)
    rest = np.setdiff1d(np.arange(target.nV), np.asarray(sub))
    return np.sort(rng.choice(rest, min(n, len(rest)), replace=False))


def impulse_responses(cortex, regions, p, nsteps=NSTEPS, save=SAVE, profiles=None,
                      verbose=True, dt=None, coupling=None, workers=0):
    """(K, nframes, nV) response of the field to one impulse in each region.

    `profiles` overrides the parcel tapers with an explicit (K, nV) profile matrix, which
    is how sub-parcel pieces are driven. `dt` overrides the CFL timestep, which switching
    runs need so that every regime is integrated on one clock.

    The cache key has to carry dt. Two regimes can share every entry of `p` that appears
    in the key and still be different systems if they are stepped at different rates, and
    a silently reused response would make H describe a system nobody simulated."""
    ptag = "" if profiles is None else f"_prof{profiles.shape[0]}x{float(profiles.sum()):.3f}"
    # a regional patch changes the medium without touching any of the scalars below, so
    # without this a patched run silently loads unpatched responses
    for nm in ("damp_patch", "speed_patch"):
        if p.get(nm):
            pk, fac = p[nm]
            ptag += f"_{nm[:4]}{len(pk)}x{float(fac):.4g}h{abs(hash(tuple(sorted(map(int, pk)))))%100000}"
    dtag = "" if dt is None else f"_dt{float(dt):.8g}"
    ctag = "" if coupling is None else "_" + coupling.key()
    key = (f"{cortex.mesh}_sig{p['sig0']:.6g}_c{p['c0']:.6g}_Ld{p['Ld']:.6g}"
           f"_spg{p.get('sponge_scale', 1.0):.4g}_a{np.round(p.get('a', 0), 3)}"
           f"_b{np.round(p.get('b', 0), 3)}_{len(regions)}_{nsteps}_{save}{ptag}{dtag}{ctag}")
    cache = os.path.join(CACHE, "impulse_" + key.replace(" ", "") + ".npy")
    if os.path.exists(cache):
        return np.load(cache)
    if profiles is None:
        T, ids = parcel_tapers(cortex, verbose=False)
        pos = {int(q): i for i, q in enumerate(ids)}
        profiles = np.stack([T[pos[int(k)]] for k in regions])
    if workers and workers > 1:
        out = _parallel_impulses(cortex, p, profiles, nsteps, save, dt, coupling,
                                 workers, verbose)
    else:
        s, dt, g, H = fl.build(cortex, p, sponge=True, dt=dt, coupling=coupling)
        out = []
        for k in range(len(profiles)):
            # the solver is built ONCE and reused across pieces, so a stateful coupling
            # term would carry piece k's tail into piece k+1
            if s.coupling is not None:
                s.coupling.reset()
            h = profiles[k].astype(np.float32).copy()
            ue = np.zeros(s.nE, np.float32)
            fr = [h.copy()]
            for n in range(1, nsteps):
                ue, h = s.step(ue, h, np.float32(dt), g, H)
                if n % save == 0:
                    fr.append(h.copy())
            out.append(np.asarray(fr))
            if verbose:
                print(f"  impulse {k:3d}: peak {np.abs(fr[0]).max():.3f} -> "
                      f"end {np.abs(fr[-1]).max():.2e}", flush=True)
        out = np.asarray(out)
    np.save(cache, out)
    return out


_W_IMP = {}


def _imp_init(cortex, p, profiles, dt, coupling):
    _W_IMP.update(cortex=cortex, p=p, profiles=profiles, dt=dt, coupling=coupling)


def _imp_one(args):
    k, nsteps, save = args
    s, dt, g, H = fl.build(_W_IMP["cortex"], _W_IMP["p"], sponge=True,
                           dt=_W_IMP["dt"], coupling=_W_IMP["coupling"])
    # each worker handles several pieces and reuses the one operator it was handed, so
    # this is the same hazard as the serial loop, not merely defensive
    if s.coupling is not None:
        s.coupling.reset()
    h = _W_IMP["profiles"][k].astype(np.float32).copy()
    ue = np.zeros(s.nE, np.float32)
    fr = [h.copy()]
    for n in range(1, nsteps):
        ue, h = s.step(ue, h, np.float32(dt), g, H)
        if n % save == 0:
            fr.append(h.copy())
    return np.asarray(fr)


def _parallel_impulses(cortex, p, profiles, nsteps, save, dt, coupling, workers, verbose):
    """One impulse per piece, across processes.

    The pieces are completely independent - each is a separate initial condition evolving
    on its own - so this is the one genuinely embarrassing parallelism in the pipeline,
    and it was being run serially everywhere except inside bo_step."""
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    K = len(profiles)
    if verbose:
        print(f"  {K} impulses over {workers} workers", flush=True)
    with ctx.Pool(workers, initializer=_imp_init,
                  initargs=(cortex, p, profiles, dt, coupling)) as pool:
        out = pool.map(_imp_one, [(k, nsteps, save) for k in range(K)])
    return np.asarray(out)


_W_SCORE = {}


def _score_init(cortex, target, p, profiles, save, kernel, band, frame_s):
    _W_SCORE.update(cortex=cortex, target=target, p=p, profiles=profiles, save=save,
                    kernel=kernel, band=band, frame_s=frame_s)


def _score_one(A):
    """One drawn realisation, scored. Returns the numbers, NOT the frames: a realisation
    is 134 MB and only one draw's frames are ever kept, so shipping them back through the
    pool would cost more than the run they came from."""
    r = score_realisation(_W_SCORE["cortex"], _W_SCORE["target"], _W_SCORE["p"], A,
                          save=_W_SCORE["save"], profiles=_W_SCORE["profiles"],
                          kernel=_W_SCORE["kernel"], band=_W_SCORE["band"],
                          frame_s=_W_SCORE["frame_s"])
    return r["sim"], r["gap"], r["rank"]


def parallel_scores(cortex, target, p, draws_A, save, profiles, kernel, workers,
                    band=None, frame_s=None):
    """Score several drawn realisations at once. -> [(sim, gap, rank), ...]

    Draws are independent runs of the same medium under different samples of the same
    S(f), so this is the same embarrassing parallelism as the impulses, and the same spawn
    pool. A `run_fn` case - a switching medium, or a coupling operator - is NOT handled
    here: the callback is a closure and does not pickle, so the caller stays serial.

    Returns (pool, async result) rather than the values, so the caller can run the draw
    whose frames it keeps in its own process while these are still going."""
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    pool = ctx.Pool(min(workers, len(draws_A)), initializer=_score_init,
                    initargs=(cortex, target, p, profiles, save, kernel, band, frame_s))
    return pool, pool.map_async(_score_one, list(draws_A))


def transfer(resp, cols, nfreq, kernel=None, idx=None):
    """FFT the impulse responses -> H (nfreq, nVsub, K) and the bin weights.

    `kernel` is a temporal filter standing between the field and the observable - the
    stage the model does not otherwise have, and which the units calibration says is
    missing by about two orders of magnitude. Being linear, it simply multiplies the
    transfer function, so the solve is unchanged and costs nothing extra. It must be the
    same kernel that score_realisation applies to the realised frames; anything else and
    H describes a system that is never scored.

    `idx` overrides the default geometric grid with an explicit set of bins. The default
    spreads samples evenly in LOG frequency, which is right when every frequency is
    equally in play and wrong once the input is held to the BOLD spectrum: that puts 63%
    of the power in 0.01-0.03 Hz, which the geometric grid covers with three samples,
    while spending a hundred on the decade above 0.1 Hz that holds 0.2%. See
    timescale.band_grid."""
    G = resp[:, :, cols]                                  # (K, nframes, nVsub)
    F = np.fft.rfft(G, axis=1)                            # (K, nbins, nVsub)
    nb = F.shape[1]
    if kernel is not None:
        import units
        F = F * units.kernel_response(kernel, nb, G.shape[1])[None, :, None]
    if idx is None:
        idx = np.unique(np.round(np.geomspace(1, nb - 1, nfreq)).astype(int))
    else:
        idx = np.unique(np.asarray(idx, int))
        idx = idx[(idx >= 1) & (idx <= nb - 1)]
    w = np.gradient(idx).astype(float)                    # each sample stands for a band
    return np.ascontiguousarray(F[:, idx].transpose(1, 2, 0)), w, idx


# ------------------------------------------------------- the admissible family
# FC is a second moment, so the medium being LTI makes C depend on the input only through
# S(f). The solve returns ONE argmax of a scale-invariant ratio over a convex cone, but
# the set {S : rho(S) >= rho* - eps} has real dimension: many different inputs reproduce
# the same FC to within noise. These regularisers pick out named members of that family,
# so the question "which dynamics are consistent with this FC" can be asked instead of
# assumed away.
#
# Each returns (value, gradient) with the gradient in the same per-frequency Hermitian
# layout as S. They are used by `family_member`, which MAXIMISES them (sign=+1) or
# minimises them (sign=-1) subject to the fit staying within eps of the argmax - not as
# penalties with a weight. A single weight trading fit against a regulariser was tried and
# does not work here: the objective is a scale-invariant ratio on a cone under trace
# normalisation, and each of these interacts with that normalisation differently.


def logdet_reg(delta=0.05):
    """Entropy of the input. sign=+1 spreads power over modes, sign=-1 concentrates it.

    This is the regulariser the project's own history asks for. Field rank tracks the
    realised score across the whole iteration sweep (48, 54, 55, 52, 46, 37, 31), and
    whitening's gain came with rank rising 64.6 -> 107.9 - converging concentrates the
    input into too few modes and the low-rank solution generalises badly off the solve
    vertices. Early stopping has been standing in for an explicit entropy term; this is
    the explicit term.

    log det is the log volume of the Gaussian with covariance S, so maximising it at
    fixed fit is the maximum-entropy member: the least committed input consistent with
    the target. sign=-1 gives the minimum-rank member, the fewest coalitions. The contrast
    between the two AT EQUAL FIT is the identifiability result.

    delta is relative to each frequency's mean eigenvalue, and is what keeps log det
    finite on the boundary of the cone where the PSD projection puts it. It has to be
    generous: log det is a barrier, so a small delta gives a gradient of order 1/delta
    that swamps everything else. prank_reg is the better-behaved choice for the same
    purpose."""
    def R(S):
        nf, K = S.shape[0], S.shape[1]
        val, G = 0.0, np.empty_like(S)
        for f in range(nf):
            A = 0.5 * (S[f] + S[f].conj().T)
            d = delta * max(np.trace(A).real / K, 1e-30)
            A = A + d * np.eye(K)
            val += float(np.linalg.slogdet(A)[1])
            Ai = np.linalg.inv(A)
            # delta is set from the trace, so it MOVES with A and the gradient is not
            # simply A^-1: d/dA log det(A + delta*tr(A)/K * I) carries the extra
            # (delta/K) tr(A^-1) I term. Dropping it leaves a ~13% gradient error that a
            # finite-difference check catches and nothing else does.
            G[f] = Ai + (delta / K) * float(np.trace(Ai).real) * np.eye(K)
        z = float(nf * K)
        return val / z, G / z
    return R


def prank_reg():
    """Participation-ratio rank of the input. sign=+1 spreads power over modes.

    Preferred over logdet_reg in practice. At fixed trace, minimising ||S||_F^2 maximises
    the participation ratio sum(e)^2/sum(e^2) - which is EXACTLY the effective-rank
    statistic fc_score.effective_rank already reports and the one that tracks the realised
    score across the iteration sweep. So this regulariser moves the diagnostic the project
    has been watching all along, rather than a proxy for it.

    It is also numerically tame where log det is not: the gradient -2S is bounded, whereas
    log det is a BARRIER whose gradient diverges as the PSD projection pushes eigenvalues
    to zero, which makes it dominate the fit almost regardless of how it is weighted.

    One trap: the gradient -2S is ENTIRELY radial, and the iterate is renormalised to unit
    trace every step, so without the tangent-space projection in family_member this
    regulariser does nothing at all and silently returns the argmax unchanged."""
    def R(S):
        nf = S.shape[0]
        z = float(nf * S.shape[1] ** 2)
        val = -sum(float((np.abs(S[f]) ** 2).sum()) for f in range(nf))
        return val / z, (-2.0 * S) / z
    return R


def offdiag_reg(eps=1e-8):
    """Coordination between driven pieces. sign=-1 drives them independently.

    The diagonal of S(f) is how much power each piece carries; the off-diagonal is how
    coordinated they are. Penalising the off-diagonal asks whether the target needs
    coalitions at all, or only a spatial profile of independent power. Smoothed |x| so the
    gradient exists on the axis."""
    def R(S):
        nf, K = S.shape[0], S.shape[1]
        off = ~np.eye(K, dtype=bool)
        val, G = 0.0, np.zeros_like(S)
        for f in range(nf):
            A = S[f]
            m = np.sqrt(np.abs(A) ** 2 + eps)
            val += float(m[off].sum())
            G[f][off] = (A / m)[off]
        z = float(nf * K * K)
        return val / z, G / z
    return R


def distance_reg(Dpiece, scale):
    """Coordination that decays with distance between pieces.

    Quadratic weight rising as (d/scale)^2, so sign=-1 buys input coalitions that are
    LOCAL by construction. This is the falsifiable version of the coalition claim
    that did not survive the finer split: instead of reading offsets back out of a free
    solve and asking whether they happen to respect geometry, geometry is imposed and the
    cost in fit is measured."""
    Wd = (np.asarray(Dpiece, float) / float(scale)) ** 2
    np.fill_diagonal(Wd, 0.0)

    def R(S):
        nf = S.shape[0]
        val, G = 0.0, np.empty_like(S)
        for f in range(nf):
            val += float((Wd * np.abs(S[f]) ** 2).sum())
            G[f] = 2.0 * Wd * S[f]
        z = float(nf * Wd.size)
        return val / z, G / z
    return R


def _stack_conj(H):
    """H (nf, nV, K) -> (Ph, Qh), contiguous and real, with H^H = Ph + i Qh.

    Built once per solve. H never changes inside the solve, so every per-frequency
    conjugate transpose the objective would otherwise take is hoisted out of the loop."""
    return (np.ascontiguousarray(H.real.transpose(0, 2, 1)),
            np.ascontiguousarray(-H.imag.transpose(0, 2, 1)))


def _stack_cols(H):
    """H (nf, nV, K) -> (Hr, Hi), each (nV, nf*K), contiguous and real."""
    nV = H.shape[1]
    return (np.ascontiguousarray(H.real.transpose(1, 0, 2).reshape(nV, -1)),
            np.ascontiguousarray(H.imag.transpose(1, 0, 2).reshape(nV, -1)))


def _cov_from_factor(Ph, Qh, sw, L):
    """sum_f 2 w_f Re(H_f S_f H_f^H) with S_f = L_f L_f^H, as two REAL gemms.

    S is PSD on the feasible set, so it factors, and then

        2 w_f H_f S_f H_f^H = B_f B_f^H,     B_f = sqrt(2 w_f) H_f L_f

    Stacking B^H over frequencies into Z (nf*K, nV) - contiguous, because Ph and Qh are
    already stored transposed - and splitting real from imaginary,

        sum_f Re(B_f B_f^H) = Zr^T Zr + Zi^T Zi

    which is two real gemms in place of nf complex triple products: a quarter of the real
    flops, and one large call rather than nf with an inner dimension of K."""
    Lh = L.conj().transpose(0, 2, 1) * sw
    Lr, Li = np.ascontiguousarray(Lh.real), np.ascontiguousarray(Lh.imag)
    nV = Ph.shape[2]
    Zr = (Lr @ Ph - Li @ Qh).reshape(-1, nV)
    Zi = (Lr @ Qh + Li @ Ph).reshape(-1, nV)
    return Zr.T @ Zr + Zi.T @ Zi


def _project(T, nblock=1, share=False, freq_keep=None, psd=True, want_factor=False):
    """Project each frequency onto the feasible set, in place.

    `freq_keep` is a boolean over frequencies; excluded ones are set to zero, which is the
    projection onto "the input carries no power in this band". That is the temporal
    constraint - what input timescales can produce this FC - and it is convex, so the
    solve is unchanged.

    `psd=False` drops the PSD clip and keeps only Hermitian symmetry. That is NOT a model
    - a Hermitian S with negative eigenvalues is not a cross-spectrum of anything - it is
    a diagnostic. The reachable-span bound assumes some PSD S realises the target's
    projection into the span, and nothing guarantees that. Running the same solve without
    the cone says how much of the gap to that bound is the CONSTRAINT and how much is the
    solver failing to reach a point it is allowed to reach.

    With one block that is the PSD cone. With R blocks - which is how a switching medium
    enters, see regimes.py - the feasible set is BLOCK-DIAGONAL PSD: regime r drives its
    own input, and there is no cross-regime input covariance to estimate because the
    regimes never run at the same time. Zeroing the off-diagonal blocks is the Frobenius
    projection onto block-diagonal matrices, and each diagonal block then projects onto
    the PSD cone independently.

    `share` additionally constrains every block to be equal - the tighter claim that one
    input is played through a medium that changes, rather than input and medium switching
    together. Averaging Hermitian blocks and then clipping keeps the result feasible.

    `want_factor` returns L alongside, with L L^H = T to rounding. The eigendecomposition
    the PSD clip already performs is what a factor costs, so it is free here and saves the
    objective a Cholesky per frequency. T itself is bit-identical either way; only the
    extra return value is new. With `psd=False` there is no factor and L is None."""
    nf = T.shape[0]
    K = T.shape[1] // nblock
    L = np.zeros(T.shape, T.dtype) if (want_factor and psd) else None
    for f in range(nf):
        if freq_keep is not None and not freq_keep[f]:
            T[f] = 0.0
            continue
        if nblock > 1:
            B = np.stack([T[f][r * K:(r + 1) * K, r * K:(r + 1) * K]
                          for r in range(nblock)])
            if share:
                B[:] = B.mean(0)
            T[f] = 0.0
            for r in range(nblock):
                M = 0.5 * (B[r] + B[r].conj().T)
                ev, U = np.linalg.eigh(M)
                sl = slice(r * K, (r + 1) * K)
                T[f][sl, sl] = (U * np.clip(ev, 0, None)) @ U.conj().T
                if L is not None:
                    L[f][sl, sl] = U * np.sqrt(np.clip(ev, 0, None))
        else:
            M = 0.5 * (T[f] + T[f].conj().T)
            if not psd:
                T[f] = M
                continue
            ev, U = np.linalg.eigh(M)
            T[f] = (U * np.clip(ev, 0, None)) @ U.conj().T
            if L is not None:
                L[f] = U * np.sqrt(np.clip(ev, 0, None))
    return (T, L) if want_factor else T


def whiten(H, eps=1e-3):
    """Change of variables making the transfer well conditioned. -> (H_tilde, L).

    The stacked H is violently anisotropic - 25 directions carry 22% of the energy - so
    plain projected gradient crawls in the weak ones and the objective is still climbing
    after 4,000 steps. Preconditioning the GRADIENT does not fix it: the PSD projection
    that follows each step undoes the rescaling, and it stalls within ten iterations.

    Reparameterising does. With M = H^H H and L its Cholesky factor, put
    Q = L^H S L. Then H S H^H = (H L^-H) Q (H L^-H)^H, so solving with H_tilde = H L^-H
    is the same problem in a basis where H_tilde^H H_tilde = I, and Q is PSD exactly when
    S is. The solver needs no change at all; only the answer has to be mapped back with
    `unwhiten`. eps is relative to the mean eigenvalue and keeps near-null directions from
    being inflated without limit."""
    nf, nV, K = H.shape
    Ht = np.empty_like(H)
    L = np.empty((nf, K, K), complex)
    for f in range(nf):
        M = H[f].conj().T @ H[f]
        M = 0.5 * (M + M.conj().T)
        M += eps * (np.trace(M).real / K) * np.eye(K)
        L[f] = np.linalg.cholesky(M)
        Ht[f] = np.linalg.solve(L[f], H[f].conj().T).conj().T      # H L^-H
    return Ht, L


def unwhiten(Q, L):
    """Q in the whitened basis -> S in the original one: S = L^-H Q L^-1.

    The order matters and is easy to get backwards: L^-1 Q L^-H is a different matrix and
    gives a C that does not match the one the solve reported."""
    S = np.empty_like(Q)
    for f in range(len(Q)):
        Lh = L[f].conj().T
        Z = np.linalg.solve(Lh, Q[f])                            # L^-H Q
        S[f] = np.linalg.solve(Lh, Z.conj().T).conj().T          # (L^-H Z^H)^H = Z L^-1
    return S


def solve(H, w, Ct, iters=300, verbose=True, nblock=1, share=False, trace=None,
          val_H=None, val_Ct=None, val_every=5, S0=None, freq_keep=None, psd=True,
          spec=None, ref=False):
    """Maximise corr(C(S), Ct) over S(f) >= 0, by projected gradient on the ratio.

    Correlation rather than squared error: with a free scale, least squares is minimised
    by shrinking the model to nothing, which is what an earlier version of this did. The
    ratio <C,Ct>/||C|| is scale invariant, so the solution is a shape rather than a size,
    and the PSD projection is the only constraint that has to be enforced.

    `nblock` > 1 expects H to be R regimes stacked along the region axis, each already
    scaled by sqrt(occupancy); the model expression is then unchanged and only the
    projection differs. See _project.

    `val_H` and `val_Ct` turn on early stopping against held-out vertices, which this
    solve needs rather than merely benefits from. The objective never converges - it is
    still climbing after 4,000 steps and never stalls - while the realised score peaks
    around 25 steps and falls away after, because converging concentrates the input into
    fewer modes and the low-rank solution does not generalise off the solve vertices. A
    fixed iteration count is therefore a hidden regularisation parameter whose right value
    differs per configuration, which makes configurations incomparable. Scoring Spearman
    on vertices the solve never sees, and keeping the best S, makes that choice explicit
    and per-configuration.

    See `whiten` for the conditioning problem and the fix, which is a change of variables
    applied OUTSIDE this function rather than anything in here.

    `freq_keep` restricts the input to a frequency band - a convex constraint, so nothing
    about the method changes. `spec` is the stronger version of the same idea: an array of
    target OUTPUT power per solved frequency, which the model is held to.

    The distinction matters. S(f) is K x K over the DRIVEN PIECES, not over vertices - its
    diagonal is each piece's input power at f, its off-diagonal their cross-spectrum. What
    can be measured is the output, the BOLD power spectrum, and the model's output power at
    frequency f is

        p_f = sum_v [H_f S_f H_f^H]_vv = Re tr(S_f H_f^H H_f)

    which is LINEAR in S_f. So "the model must have the spectrum fMRI actually has" is a
    convex constraint, and a far tighter statement than fencing off a band: the empirical
    spectrum falls as f^-2.6 with 84% of its variance in 0.01-0.1 Hz, and a hard band
    treats every frequency inside it as equally available.

    It is enforced by rescaling each S_f onto its target power after the PSD projection.
    That is a retraction onto the constraint set along the scaling ray rather than the
    Euclidean projection onto it - scaling by a positive constant preserves the cone, so
    the iterate stays feasible, which is what the method needs. Members of the admissible family at MATCHED fit are found by
    `family_member` instead of by a penalty here: a single mu trading fit against a
    penalty is not controllable on this problem, because the objective is a scale-
    invariant ratio on a cone under trace normalisation and each regulariser interacts
    with that normalisation differently. Constraining the fit and moving along the
    penalty is both better behaved and the question actually being asked.

    Pass a list as `trace` to get the objective per accepted step back. Whether the solve
    ran out of iterations or stalled is not cosmetic: a block-diagonal problem with R
    times the channels is a harder problem at the same iteration count, and comparing its
    result against the single-block one then measures the optimiser rather than the model.
    R=3 strictly contains R=1 whenever one regime is the base medium, so a lower number
    there is a convergence failure by construction.

    `ref=True` runs the objective as per-frequency loops over complex H. That is what the
    default path is checked against - it is the same arithmetic in a different order, so
    the two agree to rounding rather than bit for bit, and a long solve can take a
    different backtracking path once the difference reaches the accept test."""
    nf, nV, K = H.shape
    # white input to start, unless a warm start is supplied. The starting point matters:
    # this is a ratio objective on a cone, and projected gradient on it has no convergence
    # rate - it is still climbing after 4,000 steps and never stalls.
    S = (np.stack([np.eye(K, dtype=complex) for _ in range(nf)]) if S0 is None
         else np.array(S0, dtype=complex, copy=True))
    if freq_keep is not None:
        freq_keep = np.asarray(freq_keep, bool)
        S[~freq_keep] = 0.0            # the start must satisfy the constraint too
    Mf = None
    if spec is not None:
        spec = np.asarray(spec, float)
        if len(spec) != nf:
            raise ValueError(f"spec has {len(spec)} entries, need {nf}")
        spec = np.clip(spec, 0.0, None)
        spec = spec / max(spec.sum(), 1e-300)
        Mf = np.stack([H[f].conj().T @ H[f] for f in range(nf)])

    def _spec_fix(T, L=None):
        """Rescale each frequency onto its target output power. Feasible, not Euclidean.

        A per-frequency POSITIVE scaling of S scales its factor by the square root, so L
        is carried through in step rather than refactorised."""
        for f in range(nf):
            if spec[f] <= 0:
                T[f] = 0.0
                if L is not None:
                    L[f] = 0.0
                continue
            pw = float(np.real(np.trace(T[f] @ Mf[f]))) * w[f]
            if pw > 1e-300:
                T[f] *= spec[f] / pw
                if L is not None:
                    L[f] *= np.sqrt(spec[f] / pw)
        return T
    Ctn = Ct.copy()
    np.fill_diagonal(Ctn, 0.0)         # the diagonal is not part of the objective
    Ctn /= np.linalg.norm(Ctn)

    # H is fixed for the whole solve, so its transposes are built once here rather than
    # taken nf times per objective evaluation.
    Ph, Qh = _stack_conj(H)
    Hr, Hi = _stack_cols(H)
    sw = np.sqrt(2.0 * np.asarray(w, float))[:, None, None]
    w2 = (2.0 * np.asarray(w, float))[:, None, None]

    def model(S, L=None):
        if L is None or ref:
            C = np.zeros((nV, nV))
            for f in range(nf):
                C += w[f] * 2.0 * np.real((H[f] @ S[f]) @ H[f].conj().T)
        else:
            C = _cov_from_factor(Ph, Qh, sw, L)
        C = C - C.mean(0, keepdims=True) - C.mean(1, keepdims=True) + C.mean()
        np.fill_diagonal(C, 0.0)           # the diagonal is not part of the objective
        return C

    def adjoint(M):
        """d<C(S), M>/dS, per frequency; M must already be masked and centred."""
        Mc = M - M.mean(0, keepdims=True) - M.mean(1, keepdims=True) + M.mean()
        if ref:
            return np.stack([w[f] * 2.0 * (H[f].conj().T @ Mc @ H[f])
                             for f in range(nf)])
        # Mc is REAL, so promoting it to complex costs four real gemms where two will do.
        # One pass against the stacked H per part, then a small per-frequency contraction.
        Yr = (Mc @ Hr).reshape(nV, nf, K).transpose(1, 0, 2)
        Yi = (Mc @ Hi).reshape(nV, nf, K).transpose(1, 0, 2)
        return ((Ph @ Yr - Qh @ Yi) + 1j * (Ph @ Yi + Qh @ Yr)) * w2

    def obj(S, L=None):
        C = model(S, L)
        n = np.linalg.norm(C)
        return (float((C * Ctn).sum() / n) if n > 0 else -1.0), C, n

    SL = None
    if spec is not None:
        S, SL = _project(S, nblock, share, freq_keep, psd, want_factor=True)
        _spec_fix(S, SL)
    val, C, n = obj(S, SL)
    A0 = adjoint(Ctn)                  # Ctn never changes, so this is loop-invariant
    step = 1.0 / max(np.linalg.norm(A0), 1e-30)


    watching = val_H is not None and val_Ct is not None
    if watching:
        from scipy.stats import rankdata
        iu_v = np.triu_indices(val_H.shape[1], 1)
        yv = rankdata(np.asarray(val_Ct)[iu_v])
        yv = (yv - yv.mean()) / max(yv.std(), 1e-30)

        vPh, vQh = _stack_conj(val_H)

        def val_score(S, L=None):
            if L is None or ref:
                Cv = np.zeros((val_H.shape[1],) * 2)
                for f in range(nf):
                    Cv += w[f] * 2.0 * np.real((val_H[f] @ S[f]) @ val_H[f].conj().T)
            else:
                Cv = _cov_from_factor(vPh, vQh, sw, L)
            r = rankdata(Cv[iu_v])
            r = (r - r.mean()) / max(r.std(), 1e-30)
            return float(r @ yv / len(yv))

        best_v, best_S, best_at = val_score(S, SL), S.copy(), 0
        best_L = None if SL is None else SL.copy()
    if trace is not None:
        trace.append(float(val))
    if verbose:
        print(f"    start (white input)  corr = {val:+.4f}")
    stalled_at = None
    for it in range(iters):
        G = A0 / n - (val / n) * adjoint(C / n)
        for _ in range(12):                        # backtracking on the step size
            T, TL = _project(S + step * G, nblock, share, freq_keep, psd,
                             want_factor=True)
            if spec is not None:
                T = _spec_fix(T, TL)      # already normalised: spec sums to one
            tr = sum(np.trace(T[f]).real for f in range(nf))
            if spec is not None:
                pass
            elif psd and tr > 0:
                T = T / tr
                if TL is not None:
                    TL = TL / np.sqrt(tr)
            elif not psd:
                nn = float(np.linalg.norm(T))    # trace can vanish without the cone
                if nn > 0:
                    T = T / nn
            v2, C2, n2 = obj(T, TL)
            if v2 > val:
                S, SL, val, C, n = T, TL, v2, C2, n2
                step *= 1.6
                if trace is not None:
                    trace.append(float(val))
                break
            step *= 0.4
        else:
            stalled_at = it                         # no uphill step remains
            break
        if watching and (it % val_every == 0 or it == iters - 1):
            v = val_score(S, SL)
            if v > best_v:
                best_v, best_S, best_at = v, S.copy(), it
                best_L = None if SL is None else SL.copy()
        if verbose and (it % 25 == 0 or it == iters - 1):
            print(f"    iter {it:4d}  corr = {val:+.4f}"
                  + (f"   held-out {val_score(S, SL):+.4f}" if watching else ""),
                  flush=True)
    if watching:
        S, SL = best_S, best_L
    if trace is not None:
        rep = dict(final=float(val), steps=len(trace) - 1,
                   stalled_at=stalled_at, iters=iters)
        if watching:
            rep.update(held_out=float(best_v), stopped_at=int(best_at))
        trace.append(rep)
    return S, model(S, SL)


def in_original_basis(R, L):
    """Wrap a regulariser so it constrains S, while the solve runs on Q = L^H S L.

    Load-bearing. `whiten` reparameterises the problem for conditioning, and a regulariser
    applied to the whitened variable constrains the WRONG object: the rank of Q is not the
    rank of S, so a "maximum-entropy input" found in the whitened basis is
    maximum-entropy in a coordinate system with no physical meaning.

    S = L^-H Q L^-1 is linear, so with the real inner product <A,B> = Re tr(A^H B),

        dR = <G_S, L^-H dQ L^-1> = Re tr(L^-1 G_S^H L^-H dQ)  =>  G_Q = L^-1 G_S L^-H

    Note the order. The VARIABLE maps as L^-H Q L^-1 (that is `unwhiten`); the GRADIENT
    maps as L^-1 G L^-H, the other way round. Reusing unwhiten for the gradient looks
    right, type-checks, and is wrong - it was wrong here first, and a finite-difference
    check is the only thing that catches it, since the search still runs and still
    produces a plausible answer."""
    def RW(Q):
        v, g = R(unwhiten(Q, L))
        out = np.empty_like(g)
        for f in range(len(g)):
            Z = np.linalg.solve(L[f], g[f])                   # L^-1 G
            out[f] = np.linalg.solve(L[f], Z.conj().T).conj().T   # (L^-1 Z^H)^H = Z L^-H
        return v, out
    return RW


def family_member(H, w, Ct, S_star, R, eps=0.01, iters=200, sign=1.0, nblock=1,
                  share=False, freq_keep=None, verbose=True):
    """Move away from the argmax along R, without letting the fit fall by more than eps.

    FC is a second moment, so the medium being LTI makes the whole of the input's effect
    pass through S(f). The solve returns one point; what the data actually determine is
    the LEVEL SET {S : rho(S) >= rho* - eps}, which has real dimension. Two inputs in it
    are equally consistent with the target, so anything that differs between them is
    something FC cannot decide.

    This is a feasible-direction method: propose a step along R's gradient, project back
    onto the PSD cone and the trace sphere, and accept only if R improves AND the fit is
    still within eps of rho*. It answers "how different can the input be at the same fit",
    which a penalty parameter cannot - a single mu trades fit against R at an exchange
    rate nobody chose, and on this problem the trade is not even monotone.

    `sign` +1 maximises R, -1 minimises it, so one regulariser gives both ends of an axis:
    prank_reg at +1 is the maximum-entropy member and at -1 the minimum-rank one.

    -> (S, report). The report carries the fit actually achieved, so a member that could
    not be moved is visible as such rather than being quietly reported as a discovery."""
    nf, nV, K = H.shape
    off = ~np.eye(nV, dtype=bool)
    Ctn = Ct.copy(); Ctn[~off] = 0.0
    Ctn /= np.linalg.norm(Ctn)

    def fit(S):
        C = np.zeros((nV, nV))
        for f in range(nf):
            C += w[f] * 2.0 * np.real((H[f] @ S[f]) @ H[f].conj().T)
        C = C - C.mean(0, keepdims=True) - C.mean(1, keepdims=True) + C.mean()
        C[~off] = 0.0
        n = np.linalg.norm(C)
        return (float((C * Ctn).sum() / n) if n > 0 else -1.0), C

    S = np.array(S_star, dtype=complex, copy=True)
    rho0, _ = fit(S)
    floor = rho0 - eps
    r0 = sign * R(S)[0]
    best_r, cur_r = r0, r0
    step = None
    accepted, blocked = 0, 0
    for it in range(iters):
        g = sign * R(S)[1]
        # Project onto the trace-preserving tangent space. The iterate is renormalised to
        # unit total trace every step, so any radial component of the gradient is undone
        # by that renormalisation and is simply wasted. For the entropy regulariser it is
        # not merely wasted but fatal: the gradient of -||S||^2 is -2S, which is ENTIRELY
        # radial, so without this projection the maximum-entropy member cannot be reached
        # at all and the search silently reports the argmax back unchanged.
        tr_g = sum(np.trace(g[f]).real for f in range(nf))
        g = g - (tr_g / (nf * K)) * np.eye(K)[None, :, :]
        gn = float(np.linalg.norm(g))
        if gn < 1e-14:
            break
        if step is None:                      # first step sized to move S by ~10%
            step = 0.1 * float(np.linalg.norm(S)) / gn
        moved = False
        for _ in range(14):
            T = _project(S + step * g, nblock, share, freq_keep)
            tr = sum(np.trace(T[f]).real for f in range(nf))
            if tr > 0:
                T = T / tr
            r2 = sign * R(T)[0]
            if r2 > cur_r:
                f2, _ = fit(T)
                if f2 >= floor:
                    S, cur_r = T, r2
                    step *= 1.6
                    accepted += 1
                    moved = True
                    break
                blocked += 1                  # R would improve, the fit constraint bites
            step *= 0.4
        if not moved:
            break
        if verbose and (it % 25 == 0):
            print(f"    family iter {it:4d}  fit {fit(S)[0]:+.4f} (floor {floor:+.4f})  "
                  f"R {sign*cur_r:+.5g}", flush=True)
    rho, _ = fit(S)
    rep = dict(rho_star=float(rho0), rho=float(rho), eps=float(eps),
               R_start=float(sign * r0), R_end=float(sign * cur_r),
               accepted=accepted, blocked_by_fit=blocked,
               binding=bool(rho <= floor + 1e-9))
    if verbose:
        print(f"    -> fit {rho0:+.4f} -> {rho:+.4f} (floor {floor:+.4f}), R "
              f"{sign*r0:+.5g} -> {sign*cur_r:+.5g}, {accepted} steps, "
              f"{'fit constraint binding' if rep['binding'] else 'stopped on R'}")
    return S, rep


def solve_lagged(H, w, T0, A_tgt, ph0, iters=300, verbose=True, S0=None,
                 freq_keep=None, wa=1.0, trace=None):
    """Match the zero-lag covariance AND the antisymmetric part of lagged covariances.

    Writing M_f = H_f S_f H_f^H, which is Hermitian, the lagged covariance splits as

        Phi(tau) = sum_f w_f 2 [ cos(th) Re(M_f)  -  sin(th) Im(M_f) ]
                                 symmetric            antisymmetric

    with th = 2 pi f tau. At tau = 0 the sine vanishes, so the zero-lag objective sees only
    Re(M_f) and is BLIND to Im(M_f) - the phase structure of the model's cross-spectrum is
    unconstrained by it. The antisymmetric parts are orthogonal to the symmetric part by
    construction, so adding them double-counts nothing.

    `ph0` must be the phases for [0] + taus, i.e. the zero-lag row first. The model map and
    its adjoint are `lagged.model_lagged` / `lagged.adjoint_lagged`, which are
    finite-difference checked as a matched pair; an earlier version of this function
    inlined its own copy of the adjoint with a sign error on the i, which no test covered
    and which showed up only as a solve that stalled after two accepted steps of 400.

    `wa` weights the antisymmetric block against the zero-lag block."""
    import lagged as _lg
    nf, nV, K = H.shape
    nt = ph0.shape[0] - 1

    T0n = _lg.double_centre_ns(np.asarray(T0, float).copy())
    An = np.stack([0.5 * (_lg.double_centre_ns(np.asarray(A_tgt[k], float).copy())
                          - _lg.double_centre_ns(np.asarray(A_tgt[k], float).copy()).T)
                   for k in range(nt)])
    nrm = np.sqrt(np.linalg.norm(T0n) ** 2 + (wa ** 2) * np.linalg.norm(An) ** 2)
    T0n, An = T0n / nrm, An * (wa / nrm)

    S = (np.stack([np.eye(K, dtype=complex) for _ in range(nf)]) if S0 is None
         else np.array(S0, dtype=complex, copy=True))
    if freq_keep is not None:
        freq_keep = np.asarray(freq_keep, bool)
        S[~freq_keep] = 0.0

    def blocks(S):
        P = _lg.model_lagged(H, w, S, ph0)
        Cs = P[0]
        Ca = np.stack([0.5 * (P[k + 1] - P[k + 1].T) for k in range(nt)])
        return Cs, Ca

    def adj(Ms, Ma):
        """Gradient of <Cs,Ms> + sum_k <Ca_k, Ma_k>. Ma must be antisymmetric, so
        <antisym(Phi_k), Ma_k> = <Phi_k, Ma_k> and this is exactly adjoint_lagged."""
        return _lg.adjoint_lagged(H, w, np.concatenate([Ms[None], Ma], axis=0), ph0)

    def obj(S):
        Cs, Ca = blocks(S)
        n = np.sqrt(np.linalg.norm(Cs) ** 2 + np.linalg.norm(Ca) ** 2)
        if n <= 0:
            return -1.0, Cs, Ca, 0.0
        return float((Cs * T0n).sum() + (Ca * An).sum()) / n, Cs, Ca, n

    val, Cs, Ca, n = obj(S)
    step = 1.0 / max(np.linalg.norm(adj(T0n, An)), 1e-30)
    if trace is not None:
        trace.append(float(val))
    if verbose:
        print(f"    start  corr = {val:+.4f}  ({nt} lags, wa={wa})")
    fails = 0
    for it in range(iters):
        G = adj(T0n, An) / n - (val / n) * adj(Cs / n, Ca / n)
        moved = False
        for _ in range(30):
            T = _project(S + step * G, 1, False, freq_keep)
            tr = sum(np.trace(T[f]).real for f in range(nf))
            if tr > 0:
                T = T / tr
            v2, Cs2, Ca2, n2 = obj(T)
            if v2 > val:
                S, val, Cs, Ca, n = T, v2, Cs2, Ca2, n2
                step *= 1.6
                fails = 0
                moved = True
                if trace is not None:
                    trace.append(float(val))
                break
            step *= 0.4
        if not moved:
            fails += 1
            step *= 1e-3
            if fails >= 3:
                break
        if verbose and (it % 25 == 0 or it == iters - 1):
            print(f"    iter {it:4d}  corr = {val:+.4f}", flush=True)
    return S, blocks(S)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sig0", type=float, default=0.01)
    ap.add_argument("--c0", type=float, default=1.0)
    ap.add_argument("--Ld", type=float, default=52.4)
    ap.add_argument("--nfreq", type=int, default=32)
    ap.add_argument("--nvert", type=int, default=1200)
    ap.add_argument("--iters", type=int, default=150)
    a = ap.parse_args()

    cortex = load_cortex("fsaverage5", verbose=False)
    target = FCTarget(cortex, verbose=False)
    p = medium(a.sig0, a.c0, a.Ld)
    print(f"medium: c0 {a.c0}, Ld {a.Ld}, sig0 {a.sig0};  {len(REGIONS)} regions")

    resp = impulse_responses(cortex, REGIONS, p)
    rng = np.random.default_rng(0)
    sub = np.sort(rng.choice(target.nV, a.nvert, replace=False))
    H, w, idx = transfer(resp, target.cols[sub], a.nfreq)
    print(f"  H: {H.shape[0]} frequencies x {H.shape[1]} vertices x {H.shape[2]} regions")

    Ct = np.asarray(target.target_fc()[np.ix_(sub, sub)], np.float64)
    Ct = Ct - Ct.mean(0, keepdims=True) - Ct.mean(1, keepdims=True) + Ct.mean()

    print("  solving for the best input cross-spectrum:")
    S, C = solve(H, w, Ct, iters=a.iters)

    off = ~np.eye(len(sub), dtype=bool)
    from scipy.stats import spearmanr
    print(f"\n  best covariance match: pearson {np.corrcoef(C[off], Ct[off])[0,1]:+.4f}, "
          f"spearman {spearmanr(C[off], Ct[off]).statistic:+.4f}")
    power = np.array([np.trace(S[f]).real * w[f] for f in range(len(S))])
    ranks = [float((lambda e: e.sum() ** 2 / max((e ** 2).sum(), 1e-30))
                   (np.clip(np.linalg.eigvalsh(S[f]), 0, None))) for f in range(len(S))]
    print(f"  input power spectrum: peak at frequency bin {idx[np.argmax(power)]} of "
          f"{resp.shape[1]//2}, spread over {float(power.sum()**2/(power**2).sum()):.1f} "
          f"of {len(power)} bands")
    print(f"  cross-spectrum rank per frequency: min {min(ranks):.1f}, "
          f"median {np.median(ranks):.1f}, max {max(ranks):.1f} (of {H.shape[2]})")
    np.savez(os.path.join(CACHE, "xspec_solution.npz"), S=S, idx=idx, w=w,
             regions=REGIONS, sig0=a.sig0, c0=a.c0, Ld=a.Ld, sub=sub)
    print(f"  wrote {os.path.join(CACHE, 'xspec_solution.npz')}")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------- realisation
def draw_eta(rng, K, law="gaussian"):
    """A K-vector innovation with E[eta eta^H] = I, from one of several laws.

    This is the third axis of the admissible family, and the sharpest one. The realisation
    draws z_f = L_f eta with S = L L^H, so ANY eta with identity covariance reproduces
    S(f) exactly - and therefore reproduces the static FC exactly. What it does not
    reproduce is anything higher than second order: burstiness, tail weight, how the
    energy is clumped in time. Static FC is blind to all of it by construction, which is
    the whole point; FCD and dwell time are not.

      gaussian  complex normal - what the pipeline has always drawn
      phase     unit modulus, uniform random phase: no amplitude randomness at all
      heavy     unit-modulus phase times a heavy-tailed radius with E[r^2] = 1, so the
                same second moment carried by rarer, larger excursions

    Each has E[eta eta^H] = I exactly, so the three are second-order indistinguishable and
    the invariance is a check to run, not an assumption to make."""
    if law == "gaussian":
        return (rng.standard_normal(K) + 1j * rng.standard_normal(K)) / np.sqrt(2)
    u = np.exp(2j * np.pi * rng.random(K))
    if law == "phase":
        return u
    if law == "heavy":
        # lognormal radius, normalised so E[r^2] = 1: sigma sets the tail weight
        sig = 1.0
        r = np.exp(sig * rng.standard_normal(K) - sig * sig)
        return u * r
    raise ValueError(f"unknown law {law!r}")


def realise(S, idx, nframes, ref_frames=None, seed=0, law="gaussian"):
    """Draw a drive whose cross-spectrum is S: at each bin, z = L eta with S = L L^H.

    S is solved on a coarse grid, so it is interpolated first - in FREQUENCY, not in bin
    index. Bin b of an N-sample series is the frequency b/N, so interpolating by index
    would move the whole spectrum whenever the run length changed; `ref_frames` is the
    length S was solved at. Linear interpolation of PSD matrices stays PSD, so every bin
    remains a valid covariance, and power is not extrapolated beyond the solved band."""
    K = S.shape[1]
    nb = nframes // 2 + 1
    ref = ref_frames or 280
    f_src = np.asarray(idx, float) / ref                  # cycles per frame
    f_dst = np.arange(nb) / float(nframes)
    Sf = np.empty((nb, K, K), complex)
    for a in range(K):
        for b in range(K):
            Sf[:, a, b] = np.interp(f_dst, f_src, S[:, a, b].real, left=0.0, right=0.0) \
                + 1j * np.interp(f_dst, f_src, S[:, a, b].imag, left=0.0, right=0.0)
    rng = np.random.default_rng(seed)
    Z = np.zeros((nb, K), complex)
    for f in range(1, nb):
        ev, U = np.linalg.eigh(0.5 * (Sf[f] + Sf[f].conj().T))
        L = U * np.sqrt(np.clip(ev, 0, None))
        Z[f] = L @ draw_eta(rng, K, law)
    A = np.fft.irfft(Z, n=nframes, axis=0)
    return A / max(A.std(), 1e-30)


class ProfileDrive:
    """Minimal drive object: explicit spatial profiles and per-step amplitudes."""

    def __init__(self, cortex, profiles, Aser, amp):
        self.P = np.ascontiguousarray(profiles, np.float32)
        self.w = (self.P.astype(np.float64) * np.asarray(cortex.A)[None, :]).sum(1)
        rms = float(np.sqrt((Aser ** 2).mean()))
        self.Aser = (Aser * (amp / rms) if rms > 1e-12 else Aser).astype(np.float32)
        self.dead = rms <= 1e-12


def score_realisation(cortex, target, p, A_frames, save=SAVE, amp=2e-4, balance=False,
                      seed=0, profiles=None, run_fn=None, kernel=None, band=None,
                      frame_s=None):
    """Hold each drawn sample over its block of steps, run, and score for real.

    `run_fn(drive, nsteps, save) -> (frames, dt)` replaces the plain integration, which is
    how a switching medium is scored without this module having to know about regimes.py
    (which imports this one). p is then unused.

    `band` is the passband the TARGET was filtered to, as (lo_hz, hi_hz). The data XCP-D
    produced is bandpassed, so the observable has to be too or the model is scored on
    power the target cannot contain. It must be the same band whose response multiplied H
    in transfer - the identity the whole convex solve rests on is that the scored system
    is the solved one."""
    from input2 import RegionDrive
    from fc_moran import MoranMatch
    nsteps = len(A_frames) * save
    A = np.repeat(A_frames, save, axis=0)[:nsteps] / save
    if profiles is not None:
        d = ProfileDrive(cortex, profiles, A, amp)
    else:
        d = RegionDrive(cortex, REGIONS, A, amp=amp, nsteps=nsteps,
                        tapers=parcel_tapers(cortex, verbose=False))
    if balance:
        ww = float(d.w @ d.w)
        d.Aser = d.Aser - np.outer(d.Aser @ d.w / ww, d.w)
    d.Aser = (d.Aser * (amp / np.sqrt((d.Aser ** 2).mean()))).astype(np.float32)
    frames, _ = (run_fn(d, nsteps, save) if run_fn is not None
                 else fl.run(cortex, d, p, nsteps, save))
    if kernel is not None:                  # the observable, not the field; see transfer
        import units
        frames = units.smooth_frames(frames, kernel)
    if band is not None:
        import bandpass
        if frame_s is None:
            raise ValueError("band needs frame_s; the filter is defined in Hz")
        frames = bandpass.apply(frames, frame_s, band[0], band[1])
    Z, _ = target.model_z(frames)
    sim = float(target._prep(target.model_edges(Z=Z)[0]) @ target.y)
    mm = MoranMatch(cortex, target)
    return dict(sim=sim, gap=mm.gap(Z), rank=target.effective_rank(frames[target.burn:]),
                frames=frames, drive=d)
