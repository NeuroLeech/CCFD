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


def impulse_responses(cortex, regions, p, nsteps=NSTEPS, save=SAVE, profiles=None,
                      verbose=True):
    """(K, nframes, nV) response of the field to one impulse in each region.

    `profiles` overrides the parcel tapers with an explicit (K, nV) profile matrix, which
    is how sub-parcel pieces are driven."""
    ptag = "" if profiles is None else f"_prof{profiles.shape[0]}x{float(profiles.sum()):.3f}"
    key = (f"{cortex.mesh}_sig{p['sig0']:.6g}_c{p['c0']:.6g}_Ld{p['Ld']:.6g}"
           f"_spg{p.get('sponge_scale', 1.0):.4g}_a{np.round(p.get('a', 0), 3)}"
           f"_b{np.round(p.get('b', 0), 3)}_{len(regions)}_{nsteps}_{save}{ptag}")
    cache = os.path.join(CACHE, "impulse_" + key.replace(" ", "") + ".npy")
    if os.path.exists(cache):
        return np.load(cache)
    if profiles is None:
        T, ids = parcel_tapers(cortex, verbose=False)
        pos = {int(q): i for i, q in enumerate(ids)}
        profiles = np.stack([T[pos[int(k)]] for k in regions])
    s, dt, g, H = fl.build(cortex, p, sponge=True)
    out = []
    for k in range(len(profiles)):
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


def transfer(resp, cols, nfreq):
    """FFT the impulse responses -> H (nfreq, nVsub, K) and the bin weights."""
    G = resp[:, :, cols]                                  # (K, nframes, nVsub)
    F = np.fft.rfft(G, axis=1)                            # (K, nbins, nVsub)
    nb = F.shape[1]
    idx = np.unique(np.round(np.geomspace(1, nb - 1, nfreq)).astype(int))
    w = np.gradient(idx).astype(float)                    # each sample stands for a band
    return np.ascontiguousarray(F[:, idx].transpose(1, 2, 0)), w, idx


def solve(H, w, Ct, iters=300, verbose=True):
    """Maximise corr(C(S), Ct) over S(f) >= 0, by projected gradient on the ratio.

    Correlation rather than squared error: with a free scale, least squares is minimised
    by shrinking the model to nothing, which is what an earlier version of this did. The
    ratio <C,Ct>/||C|| is scale invariant, so the solution is a shape rather than a size,
    and the PSD projection is the only constraint that has to be enforced."""
    nf, nV, K = H.shape
    S = np.stack([np.eye(K, dtype=complex) for _ in range(nf)])     # white input to start
    off = ~np.eye(nV, dtype=bool)
    Ctn = Ct.copy(); Ctn[~off] = 0.0
    Ctn /= np.linalg.norm(Ctn)

    def model(S):
        C = np.zeros((nV, nV))
        for f in range(nf):
            C += w[f] * 2.0 * np.real((H[f] @ S[f]) @ H[f].conj().T)
        C = C - C.mean(0, keepdims=True) - C.mean(1, keepdims=True) + C.mean()
        C[~off] = 0.0
        return C

    def adjoint(M):
        """d<C(S), M>/dS, per frequency; M must already be masked and centred."""
        Mc = M - M.mean(0, keepdims=True) - M.mean(1, keepdims=True) + M.mean()
        return np.stack([w[f] * 2.0 * (H[f].conj().T @ Mc @ H[f]) for f in range(nf)])

    def obj(S):
        C = model(S)
        n = np.linalg.norm(C)
        return (float((C * Ctn).sum() / n) if n > 0 else -1.0), C, n

    val, C, n = obj(S)
    step = 1.0 / max(np.linalg.norm(adjoint(Ctn)), 1e-30)
    if verbose:
        print(f"    start (white input)  corr = {val:+.4f}")
    for it in range(iters):
        G = adjoint(Ctn) / n - (val / n) * adjoint(C / n)
        for _ in range(12):                        # backtracking on the step size
            T = S + step * G
            for f in range(nf):                    # project each onto PSD
                T[f] = 0.5 * (T[f] + T[f].conj().T)
                ev, U = np.linalg.eigh(T[f])
                T[f] = (U * np.clip(ev, 0, None)) @ U.conj().T
            tr = sum(np.trace(T[f]).real for f in range(nf))
            if tr > 0:
                T = T / tr
            v2, C2, n2 = obj(T)
            if v2 > val:
                S, val, C, n = T, v2, C2, n2
                step *= 1.6
                break
            step *= 0.4
        else:
            break                                   # no uphill step remains
        if verbose and (it % 25 == 0 or it == iters - 1):
            print(f"    iter {it:4d}  corr = {val:+.4f}", flush=True)
    return S, model(S)


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
def realise(S, idx, nframes, ref_frames=None, seed=0):
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
        eta = (rng.standard_normal(K) + 1j * rng.standard_normal(K)) / np.sqrt(2)
        Z[f] = L @ eta
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
                      seed=0, profiles=None):
    """Hold each drawn sample over its block of steps, run, and score for real."""
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
    frames, _ = fl.run(cortex, d, p, nsteps, save)
    Z, _ = target.model_z(frames)
    sim = float(target._prep(target.model_edges(Z=Z)[0]) @ target.y)
    mm = MoranMatch(cortex, target)
    return dict(sim=sim, gap=mm.gap(Z), rank=target.effective_rank(frames[target.burn:]),
                frames=frames, drive=d)
