"""Stage 2: do the model's spatial patterns look like the brain's?

Both sides are reduced to 180 Glasser parcel means per frame. That is what makes
the comparison possible at all: MSC is fs_LR 32k and the model runs on fsaverage5,
but the parcel ids are identical in both (verified: 1=V1, 4=V2, 9=3b, 24=A1), so no
surface resampling is needed anywhere.

Each frame is then demeaned across parcels. The model's field is exactly zero-mean
by construction (mass balance), BOLD is not - without matching that, the dominant
clusters in the fMRI would be global-signal-high vs global-signal-low, patterns the
model cannot produce even in principle.

Canonical patterns are found by k-medoids under correlation distance. Overlap is
measured by pooling both sets, clustering them together, and asking how mixed each
cluster is: a cluster of only fMRI frames is a pattern the model never produces,
one of only model frames is a pattern the model invents.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import hcp_utils as hcp
from scipy.optimize import linear_sum_assignment

from get_msc import load_all
from mesh_cache import load_cortex

N_PARC = 180


# ------------------------------------------------------------------ parcellation
def fmri_parcels(censor=True, verbose=True):
    """MSC left cortex -> (180, nframes), each frame demeaned and unit norm."""
    X, verts, nsurf, tr = load_all(censor=censor)          # (29696, T)
    lab = np.asarray(hcp.mmp["map_all"])[:X.shape[0]]
    P = np.zeros((N_PARC, X.shape[1]), np.float32)
    for p in range(1, N_PARC + 1):
        m = lab == p
        if m.any():
            P[p-1] = X[m].mean(0)
    if verbose:
        print(f"  fMRI: {X.shape[0]} vertices x {X.shape[1]} frames "
              f"-> {P.shape[0]} parcels, TR {tr:.2f}s")
    return _normalise(P), tr


def model_parcels(H, cortex, verbose=True):
    """H: (nframes, nV) model field -> (180, nframes) parcel means, normalised."""
    lab = cortex.lab
    A = cortex.A
    P = np.zeros((N_PARC, len(H)), np.float32)
    for p in range(1, N_PARC + 1):
        m = lab == p
        if m.any():
            w = A[m]
            P[p-1] = (H[:, m] * w).sum(1) / w.sum()        # area-weighted mean
    if verbose:
        print(f"  model: {H.shape[1]} vertices x {len(H)} frames -> {P.shape[0]} parcels")
    return _normalise(P)


def _normalise(P):
    """Demean each frame across parcels, then unit-norm, so only pattern shape counts."""
    P = np.asarray(P, np.float64)
    P = P - P.mean(axis=0, keepdims=True)
    n = np.linalg.norm(P, axis=0)
    keep = n > 1e-12
    return (P[:, keep] / n[keep]).astype(np.float32)


def smooth_decimate(P, factor=10):
    """Temporal low-pass and downsample in one step.

    With no stimulus events there is nothing to align a haemodynamic lag to, so the
    HRF reduces to its low-pass effect and this single operation stands in for both
    the HRF and the downsampling to a TR."""
    sig = factor/2.0
    k = np.arange(-int(3*sig), int(3*sig)+1)
    w = np.exp(-0.5*(k/sig)**2); w /= w.sum()
    S = np.apply_along_axis(lambda x: np.convolve(x, w, mode="same"), 1, P)
    return _normalise(S[:, ::factor])


# ------------------------------------------------------------------ clustering
def _corr_dist(X):
    """1 - Pearson r between columns (frames). X columns are already unit norm."""
    G = X.T @ X
    np.clip(G, -1.0, 1.0, out=G)
    return 1.0 - G


def kmedoids(D, k, rng, n_init=8, iters=60):
    """Plain alternating k-medoids on a precomputed distance matrix."""
    n = len(D)
    best = None
    for _ in range(n_init):
        med = rng.choice(n, k, replace=False)
        for _ in range(iters):
            lab = np.argmin(D[:, med], axis=1)
            new = med.copy()
            for j in range(k):
                idx = np.flatnonzero(lab == j)
                if len(idx) == 0:
                    new[j] = rng.integers(n); continue
                sub = D[np.ix_(idx, idx)].sum(0)
                new[j] = idx[int(np.argmin(sub))]
            if np.array_equal(np.sort(new), np.sort(med)):
                med = new; break
            med = new
        lab = np.argmin(D[:, med], axis=1)
        cost = float(D[np.arange(n), med[lab]].sum())
        if best is None or cost < best[0]:
            best = (cost, med.copy(), lab.copy())
    return best[1], best[2], best[0]


def compare(Pf, Pm, k=6, seed=0, verbose=True):
    """Pool equal numbers of fMRI and model frames, cluster together, report mixing."""
    rng = np.random.default_rng(seed)
    n = min(Pf.shape[1], Pm.shape[1])
    fi = rng.choice(Pf.shape[1], n, replace=False)
    mi = rng.choice(Pm.shape[1], n, replace=False)
    X = np.hstack([Pf[:, fi], Pm[:, mi]])
    src = np.r_[np.zeros(n, int), np.ones(n, int)]
    D = _corr_dist(X)
    med, lab, cost = kmedoids(D, k, rng)
    rows = []
    for j in range(k):
        idx = np.flatnonzero(lab == j)
        if len(idx) == 0:
            continue
        frac = float((src[idx] == 1).mean())      # fraction from the model
        rows.append((j, len(idx), frac))
    mixing = float(np.mean([min(f, 1-f)/0.5 for _, _, f in rows]))

    # how well does each fMRI medoid correlate with its best model medoid?
    Df = _corr_dist(Pf); Dm = _corr_dist(Pm)
    mf, _, _ = kmedoids(Df, k, rng)
    mm, _, _ = kmedoids(Dm, k, rng)
    Cx = Pf[:, mf].T @ Pm[:, mm]
    best_match = np.abs(Cx).max(axis=1)

    if verbose:
        print(f"\n  pooled k-medoids, k={k}, {n} frames from each source")
        print(f"  {'cluster':>8} {'n':>6} {'% model':>9}")
        for j, cnt, frac in rows:
            print(f"  {j:8d} {cnt:6d} {100*frac:8.0f}%")
        print(f"  mixing index {mixing:.3f}   (1 = every cluster balanced, "
              f"0 = fully segregated)")
        print(f"  best |r| of each fMRI medoid to any model medoid: "
              f"{np.array2string(best_match, precision=2)}")
        print(f"  mean {best_match.mean():.3f}")
    return dict(mixing=mixing, rows=rows, best_match=best_match,
                medoids_f=mf, medoids_m=mm, Pf=Pf, Pm=Pm)


from paths import CACHE as HERE


def fmri_reference(k=6, n_sub=2500, seed=0):
    """The k canonical fMRI patterns, computed once and cached.

    Held fixed across the whole search so every genome is scored against the same
    reference; recomputing them per generation would move the target."""
    path = os.path.join(HERE, f"fmri_ref_k{k}.npz")
    if os.path.exists(path):
        return np.load(path)["ref"]
    Pf, tr = fmri_parcels(verbose=False)
    rng = np.random.default_rng(seed)
    idx = rng.choice(Pf.shape[1], min(n_sub, Pf.shape[1]), replace=False)
    X = Pf[:, idx]
    med, lab, cost = kmedoids(_corr_dist(X), k, rng)
    ref = np.ascontiguousarray(X[:, med])
    sizes = np.bincount(lab, minlength=k)
    np.savez(path, ref=ref, sizes=sizes)
    print(f"  fMRI reference: {k} medoids from {X.shape[1]} frames, "
          f"cluster sizes {sizes.tolist()}")
    return ref


def match_score(Pm, ref, seed=0):
    """Mean |r| between the fixed fMRI reference patterns and the model's own
    medoids, under the best one-to-one assignment.

    NOT USABLE AS A FITNESS TERM. Measured: clustering one half of the fMRI and
    matching it to the other half scores 0.138-0.156; doing the same to white
    noise scores 0.092-0.158. The ceiling lies inside the null, because BOLD
    frames have no k=6 cluster structure to find (k-medoids cost on real frames
    is 0.96x the cost on noise). Kept for reference only."""
    k = ref.shape[1]
    if Pm.shape[1] < k:
        return 0.0
    rng = np.random.default_rng(seed)
    med, _, _ = kmedoids(_corr_dist(Pm), k, rng, n_init=4, iters=30)
    C = np.abs(ref.T @ Pm[:, med])
    r_, c_ = linear_sum_assignment(-C)
    return float(C[r_, c_].mean())


# --------------------------------------------------- the two fitness measures
# Measured, not assumed. CEIL is held-out fMRI under the identical procedure in
# both cases. The FLOORs are NOT the same kind of thing and must not be read as
# if they were: white noise is the floor of similarity, but on richness noise
# scores ~0.94, i.e. ABOVE the fMRI, so the floor there is the worst model run
# instead. The two terms are summed, so these are what makes them commensurable.
SIM_FLOOR, SIM_CEIL = 0.270, 0.370     # white noise / held-out fMRI
RICH_FLOOR, RICH_CEIL = 0.230, 0.922   # worst model run / fMRI  (noise ~0.94)


def frame_bank(n_bank=3000, seed=0):
    """(bank, held) - disjoint pools of real fMRI frames, 180 parcels each.

    Built from the raw MSC files on first call, then cached. The split is fixed
    by `seed`, so the held-out pool is never one the bank has already seen."""
    path = os.path.join(HERE, "fmri_bank.npz")
    if not os.path.exists(path):
        Pf, tr = fmri_parcels(verbose=True)
        perm = np.random.default_rng(seed).permutation(Pf.shape[1])
        np.savez(path,
                 bank=np.ascontiguousarray(Pf[:, perm[:n_bank]]),
                 held=np.ascontiguousarray(Pf[:, perm[n_bank:]]))
        print(f"  frame bank: {n_bank} kept, {Pf.shape[1]-n_bank} held out")
    z = np.load(path)
    return z["bank"], z["held"]


N_PC = 20


def fmri_pcs(k=N_PC):
    """Top-k principal components of the fMRI parcel timeseries. Computed once.

    These are not claimed to be brain states, modes, or anything else. They are a
    fixed low-dimensional summary of where the fMRI puts its variance, held
    constant so every genome is scored against the same thing."""
    path = os.path.join(HERE, f"fmri_pcs_k{k}.npz")
    if os.path.exists(path):
        return np.load(path)["pcs"]
    bank, held = frame_bank()
    U, S, _ = np.linalg.svd(np.hstack([bank, held]), full_matrices=False)
    pcs = np.ascontiguousarray(U[:, :k])
    np.savez(path, pcs=pcs, sv=S[:k])
    return pcs


def pcs_of(P, k=N_PC):
    """Top-k principal components of a parcel timeseries."""
    U, _, _ = np.linalg.svd(P, full_matrices=False)
    return U[:, :min(k, U.shape[1])]


def similarity(Pm, ref_pcs, k=N_PC):
    """median over fMRI PCs of the best |r| to any of the model's own PCs.

    A relative measure for driving a search, not an absolute claim about how
    brain-like a run is. Median rather than mean so a couple of very well matched
    components cannot carry the score."""
    M = pcs_of(Pm, k)
    return float(np.median(np.abs(ref_pcs.T @ M).max(1)))


def richness(Pm):
    """1 - mean |r| over all pairs of the model's own frames.

    How many distinct configurations the flow actually visits. White noise scores
    ~0.94 and the fMRI 0.922, so this term cannot be won by incoherence alone
    beyond what the brain already does; it can only be lost by repeating one
    pattern, which is what every run so far does (0.23-0.60)."""
    G = np.abs(Pm.T @ Pm)
    n = G.shape[0]
    if n < 2:
        return 0.0
    return float(1.0 - (G.sum() - np.trace(G)) / (n*(n-1)))


def scaled(sim, rich):
    """Both terms on a common scale: 0 = white noise, 1 = the brain."""
    s = (sim - SIM_FLOOR) / (SIM_CEIL - SIM_FLOOR)
    r = (rich - RICH_FLOOR) / (RICH_CEIL - RICH_FLOOR)
    return float(s), float(r)
