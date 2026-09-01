"""Empirical lagged covariance from NKI, and the model's prediction of it.

Zero-lag FC is a second moment, so it cannot separate processes that share it - which is
why the admissible family spans 22x in input concentration at matched fit. The lagged
covariance Phi(tau) = <x(t) x(t+tau)^T> can: its ANTISYMMETRIC part is pure lead-lag and
is exactly what tau=0 cannot represent. Measured on this data it is worth 19% of the
symmetric part at 5.2 s and 36% at 7.7 s, and it survives removal of the global signal.

For an LTI medium driven by a stationary input with cross-spectrum S(f), the output lagged
covariance is the inverse transform of the output cross-spectrum:

    Phi(tau) = sum_f w_f * 2 * Re( exp(2i pi f tau) H_f S_f H_f^H )

which is STILL LINEAR in S, so the convex solve carries over unchanged in structure - only
the model map and its adjoint acquire a phase. At tau = 0 it reduces to the existing
expression exactly.

Two things are deliberate. Each Phi(tau) is double-centred exactly as the zero-lag target
is, which removes the global component - the classical carrier of vascular rather than
neural lag. And the model side must use UNRANKED timeseries: the rank transform is
nonlinear, so with it in place Phi(tau) is not the inverse transform of anything.

  python lagged.py --taus 3,5,8,12
"""
import os, argparse
import numpy as np

from mesh_cache import load_cortex
from paths import CACHE
import fc_score, xspec, fc_group_nki as nki

TR = 0.645


def double_centre_ns(A):
    """Double-centre a possibly NON-symmetric matrix, then mask the diagonal.

    fc_score.double_centre assumes symmetry (it reads one set of row means); a lagged
    covariance is not symmetric and that is the whole point of using it.

    The means here INCLUDE the diagonal, unlike fc_score's. That is deliberate and it is
    what makes the operation J A J with J = I - 11^T/n, symmetric and idempotent, hence
    self-adjoint. Excluding the diagonal from the means - the natural-looking choice, and
    what this function did first - breaks self-adjointness, so the gradient of any
    objective built on it silently loses a term. It cost a factor of 2.6 in a
    finite-difference check and nothing else would have caught it. See `centre_adjoint`."""
    B = A - A.mean(1, keepdims=True) - A.mean(0, keepdims=True) + A.mean()
    np.fill_diagonal(B, 0.0)
    return B


def centre_adjoint(M):
    """Adjoint of double_centre_ns: mask FIRST, then centre.

    double_centre_ns is Z(J A J) with Z the diagonal mask. Both Z and J.J are self-adjoint,
    so the adjoint of their composition is the composition in the other order."""
    N = np.array(M, float, copy=True)
    np.fill_diagonal(N, 0.0)
    return N - N.mean(1, keepdims=True) - N.mean(0, keepdims=True) + N.mean()


def empirical(taus, sub_vertices, nsub=None, gsr=True, verbose=True):
    """Group-mean lagged covariance on the given vertices. -> (n_tau, n, n)."""
    key = (f"lagged_{'-'.join(map(str, taus))}_{len(sub_vertices)}_"
           f"{'gsr' if gsr else 'raw'}_{nsub or 'all'}.npy")
    path = os.path.join(CACHE, key)
    if os.path.exists(path):
        if verbose:
            print(f"  loaded {path}")
        return np.load(path)
    files = nki.subject_files("left")[:nsub]
    n = len(sub_vertices)
    acc = np.zeros((len(taus), n, n))
    for si, p in enumerate(files, 1):
        X = nki.load_subject(p)[sub_vertices].astype(np.float64)
        X -= X.mean(1, keepdims=True)
        sd = X.std(1, keepdims=True); sd[sd == 0] = 1.0
        X /= sd
        if gsr:
            g = X.mean(0); g -= g.mean()
            X = X - np.outer((X @ g) / max(float(g @ g), 1e-30), g)
        T = X.shape[1]
        for k, L in enumerate(taus):
            A = X[:, :T - L] if L else X
            B = X[:, L:] if L else X
            acc[k] += (A @ B.T) / B.shape[1]
        if verbose and si % 20 == 0:
            print(f"    {si}/{len(files)} subjects", flush=True)
    acc /= len(files)
    out = np.stack([double_centre_ns(acc[k]) for k in range(len(taus))])
    np.save(path, out)
    if verbose:
        print(f"  wrote {path}")
    return out


def phases(idx, ref_frames, taus):
    """exp(-2i pi f tau) per (tau, frequency), f = idx/ref_frames cycles per frame.

    The sign is load-bearing and was wrong first. With x(t) = int X(f) e^{2i pi f t} df,

        Phi(tau) = E[x(t) x(t+tau)^H] = int P(f) e^{-2i pi f tau} df

    and since Re(e^{+i th} M) = cos Re(M) - sin Im(M) while Re(e^{-i th} M) = cos Re(M) +
    sin Im(M), the wrong sign flips the ANTISYMMETRIC part exactly and leaves the
    symmetric part untouched. Nothing in a gradient check catches it - the adjoint was
    consistent with the model, both simply described the transpose of the intended
    quantity. What caught it was comparing the PREDICTED lagged covariance against the one
    the simulation actually produced: the wrong sign scores -0.78, the right one +0.78."""
    f = np.asarray(idx, float) / float(ref_frames)
    return np.exp(-2j * np.pi * np.outer(np.asarray(taus, float), f))


def model_lagged(H, w, S, ph):
    """-> (n_tau, nV, nV) predicted lagged covariances, double-centred.

    ph is from `phases`. At tau = 0 the phase is 1 and this is the existing model map."""
    nf, nV, _ = H.shape
    out = np.zeros((ph.shape[0], nV, nV))
    for f in range(nf):
        M = (H[f] @ S[f]) @ H[f].conj().T
        for k in range(ph.shape[0]):
            out[k] += w[f] * 2.0 * np.real(ph[k, f] * M)
    return np.stack([double_centre_ns(out[k]) for k in range(out.shape[0])])


def adjoint_lagged(H, w, Ms, ph):
    """d<Phi, M>/dS per frequency, summed over lags. Ms already centred and masked.

    Derivation: <Phi(tau), M> = sum_f w_f 2 Re( exp(i th) tr(S H^H M^T H) ), so with the
    real inner product <A,B> = Re tr(A^H B) the gradient is w_f 2 exp(-i th) H^H M H,
    Hermitianised. At tau = 0 this is the existing adjoint."""
    nf, nV, K = H.shape
    Ms = np.stack([centre_adjoint(Ms[k]) for k in range(ph.shape[0])])
    G = np.zeros((nf, K, K), complex)
    for f in range(nf):
        acc = np.zeros((K, K), complex)
        for k in range(ph.shape[0]):
            acc += np.conj(ph[k, f]) * (H[f].conj().T @ Ms[k] @ H[f])
        acc = w[f] * 2.0 * acc
        G[f] = 0.5 * (acc + acc.conj().T)
    return G


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--taus", default="0,3,5,8,12")
    ap.add_argument("--nvert", type=int, default=1000)
    ap.add_argument("--nsub", type=int, default=None)
    ap.add_argument("--raw", action="store_true", help="do NOT remove the global signal")
    a = ap.parse_args()
    taus = [int(v) for v in a.taus.split(",")]

    c = load_cortex("fsaverage5", verbose=False)
    t = fc_score.default_target(c, verbose=False)
    sub = xspec.medoid_subset(t, a.nvert)
    print(f"  {len(sub)} solve vertices, lags {taus} "
          f"({', '.join(f'{L*TR:.2f}s' for L in taus)})")
    P = empirical(taus, t.vertices[sub], a.nsub, gsr=not a.raw)
    n = P.shape[1]
    off = ~np.eye(n, dtype=bool)
    print(f"\n  {'lag':>5s} {'seconds':>8s} {'||sym||':>10s} {'||antisym||':>12s} "
          f"{'antisym/sym':>12s} {'corr with tau=0':>16s}")
    z = P[0][off]; z = (z - z.mean()) / z.std()
    for k, L in enumerate(taus):
        S_ = 0.5 * (P[k] + P[k].T); A_ = 0.5 * (P[k] - P[k].T)
        ns, na = np.linalg.norm(S_[off]), np.linalg.norm(A_[off])
        v = P[k][off]; v = (v - v.mean()) / max(v.std(), 1e-30)
        print(f"  {L:>5d} {L*TR:>8.2f} {ns:>10.4f} {na:>12.4f} "
              f"{na/max(ns,1e-30):>12.4f} {float(v@z/len(v)):>16.4f}")
    print(f"\n  the last column is how much each lag ADDS: a value near 1 means that lag "
          f"is already implied by the zero-lag matrix")


if __name__ == "__main__":
    main()
