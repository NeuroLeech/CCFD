"""Does the group data carry consistent lead-lag structure, or only zero-lag coherence?

Moving the target from zero-lag FC to a cross-spectral density is only worth it if the
off-diagonal PHASE survives averaging over subjects. Resting state has no common timing,
so a lead-lag relation that differs between subjects averages away and the group CSD
collapses to a frequency-resolved real matrix - coherence magnitude and nothing more. In
that case the extra machinery buys a better-conditioned objective but no new constraint on
the dynamics, and lagged covariance at a few tau is the cheaper target.

Two measurements, either of which settles it.

PHASE CONSISTENCY. Per vertex pair and frequency, |sum_s C_s| / sum_s |C_s| over subjects:
1 means every subject agrees on the phase, 0 means it is random. Compared against the
value expected from averaging that many random phases, sqrt(pi)/2/sqrt(n), so "high" is
measured rather than asserted.

LAG ASYMMETRY. The group lagged covariance Phi(tau) = <x(t) x(t+tau)^T>. Its SYMMETRIC
part is what zero-lag FC already sees smeared in time; its ANTISYMMETRIC part is pure
lead-lag and is exactly what a second moment at tau=0 cannot represent. The ratio
||Phi - Phi^T|| / ||Phi + Phi^T|| is the quantity that decides whether Stage 2 has
anything to fit.

  python csd_probe.py --nvert 400
"""
import argparse
import numpy as np

from mesh_cache import load_cortex
import fc_score, xspec, fc_group_nki as nki

TR = 0.645


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--nvert", type=int, default=400)
    ap.add_argument("--nsub", type=int, default=40)
    ap.add_argument("--window", type=int, default=256, help="Welch window, frames")
    ap.add_argument("--gsr", action="store_true",
                    help="regress out the global mean timecourse first. The FC target is "
                         "double-centred, so the global component is already removed "
                         "there; consistent lead-lag in fMRI is classically contaminated "
                         "by vascular transit of that component")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    c = load_cortex("fsaverage5", verbose=False)
    t = fc_score.default_target(c, verbose=False)
    sub = xspec.medoid_subset(t, 1000)
    rng = np.random.default_rng(a.seed)
    pick = np.sort(rng.choice(sub, min(a.nvert, len(sub)), replace=False))
    verts = t.vertices[pick]
    files = nki.subject_files("left")[:a.nsub]
    n = len(pick)
    print(f"  {len(files)} subjects, {n} vertices, TR {TR}s, Welch window {a.window} "
          f"frames ({a.window*TR:.0f} s, {1/(a.window*TR):.4f} Hz bins)")

    W = a.window
    nb = W // 2 + 1
    fhz = np.fft.rfftfreq(W, d=TR)
    win = np.hanning(W)
    acc_c = np.zeros((nb, n, n), complex)      # sum of complex CSD over segments
    acc_m = np.zeros((nb, n, n))               # sum of |CSD|
    nseg = 0
    LAGS = [0, 1, 2, 3, 5, 8, 12, 20]
    acc_lag = {L: np.zeros((n, n)) for L in LAGS}
    nsub_ok = 0
    for si, p in enumerate(files):
        X = nki.load_subject(p)[verts].astype(np.float64)
        X -= X.mean(1, keepdims=True)
        sd = X.std(1, keepdims=True); sd[sd == 0] = 1.0
        X /= sd
        if a.gsr:
            g = X.mean(0)                       # global mean timecourse
            g = g - g.mean()
            X = X - np.outer((X @ g) / max(float(g @ g), 1e-30), g)
        T = X.shape[1]
        for st in range(0, T - W + 1, W // 2):
            seg = X[:, st:st + W] * win
            F = np.fft.rfft(seg, axis=1)
            for b in range(nb):
                v = F[:, b]
                Cb = np.outer(v, v.conj())
                acc_c[b] += Cb
                acc_m[b] += np.abs(Cb)
            nseg += 1
        for L in LAGS:
            Y = X[:, L:] if L else X
            Z = X[:, :T - L] if L else X
            acc_lag[L] += (Z @ Y.T) / Y.shape[1]
        nsub_ok += 1
        if (si + 1) % 10 == 0:
            print(f"    {si+1}/{len(files)} subjects", flush=True)

    off = ~np.eye(n, dtype=bool)
    cons = np.abs(acc_c) / np.maximum(acc_m, 1e-300)
    chance = np.sqrt(np.pi) / 2.0 / np.sqrt(nseg)
    print(f"\n  --- phase consistency across {nseg} segments "
          f"(chance level {chance:.4f}) ---")
    print(f"  {'band (Hz)':>14s} {'consistency':>12s} {'x chance':>9s} "
          f"{'|Im|/|Re| of group CSD':>23s}")
    for lo, hi in ((0.005, 0.01), (0.01, 0.03), (0.03, 0.06), (0.06, 0.1), (0.1, 0.3)):
        m = (fhz > lo) & (fhz <= hi)
        if not m.any():
            continue
        cv = np.array([cons[b][off].mean() for b in np.flatnonzero(m)]).mean()
        G = acc_c[m].sum(0)
        ir = np.linalg.norm(G[off].imag) / max(np.linalg.norm(G[off].real), 1e-300)
        print(f"  {lo:6.3f}-{hi:<7.3f} {cv:>12.4f} {cv/chance:>9.1f} {ir:>23.4f}")

    print(f"\n  --- lagged covariance, group mean over {nsub_ok} subjects ---")
    print(f"  {'lag':>5s} {'seconds':>8s} {'||sym||':>10s} {'||antisym||':>12s} "
          f"{'antisym/sym':>12s}")
    for L in LAGS:
        P = acc_lag[L] / nsub_ok
        S_ = 0.5 * (P + P.T); A_ = 0.5 * (P - P.T)
        ns, na = np.linalg.norm(S_[off]), np.linalg.norm(A_[off])
        print(f"  {L:>5d} {L*TR:>8.2f} {ns:>10.4f} {na:>12.4f} "
              f"{na/max(ns,1e-300):>12.4f}")
    print(f"\n  the antisymmetric part is what a zero-lag second moment cannot carry;")
    print(f"  the |Im|/|Re| column is the same question in the frequency domain")


if __name__ == "__main__":
    main()
