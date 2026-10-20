"""The envelope observable: square the field FIRST, then filter.

`best_fit`'s observable is linear in the field - BOLD kernel, then passband - so every
filter can be folded into the transfer function and the covariance stays linear in the
input cross-spectrum. That is what makes the solve convex, and it is also what makes a
fast drive invisible: an LTI system never moves power between frequencies, so input above
the passband produces output above the passband and the filter discards it.

Squaring does move power. The drive is Gaussian and the medium linear, so the field is
Gaussian and Wick is exact:

    cov(h_i(t)^2, h_j(s)^2) = 2 C_ij(t-s)^2

and filtering afterwards weights each lag by the filter's autocorrelation:

    cov(u_i, u_j) = 2 sum_tau rho(tau) C_ij(tau)^2       u = filter(h^2)

C_ij(tau)^2 in time is the cross-spectrum convolved with itself in frequency, so a pair of
components at f1 and f2 contributes at |f1 - f2|. Fast input, slow observable.

TWO THINGS THE CALLER MUST GET RIGHT.

`H` has to be the RAW transfer function - `xspec.transfer(..., kernel=None)`, and not
multiplied by the passband. The square acts on the field, so the filtering belongs in
`rho` and nowhere else. Passing best_fit's `H` squares an already-smoothed field, which is
a different quantity: measured on the toy, that error alone moved the closed form's
agreement with simulation from +0.99 to +0.80.

The lag count is set by the filter, not by taste. The BOLD kernel alone spans 35 lags; with
the 0.01-0.08 Hz passband it is 1,579 at a 1e-3 tolerance, because a 0.01 Hz filter rings
for ~100 s. That is 11.8 GiB of C(tau) at 1,000 vertices, so the forward and the adjoint
both chunk over lags and recompute C per block rather than holding it.
"""
import _path  # noqa: F401  - puts the sibling code folders on sys.path
import numpy as np

import xspec


def filter_lags(resp, ref_frames, tol=1e-3):
    """-> (lags, rho). The filter's autocorrelation, truncated where it is negligible.

    `resp` is the observable's amplitude response on the rfft grid - the BOLD kernel times
    the passband. The envelope is squared THEN filtered, so what weights each lag is the
    filter applied twice, which is its autocorrelation."""
    rho = np.fft.irfft(np.abs(np.asarray(resp)) ** 2, n=ref_frames)
    keep = np.abs(rho) > tol * np.abs(rho).max()
    return np.arange(ref_frames)[keep], rho[keep]


def _M(H, S):
    """H_f S_f H_f^H per frequency, as batched matmuls.

    NOT np.einsum("fva,fab,fwb->fvw", ...) without optimize=True: that plans no pairwise
    contraction and materialises an (f,v,a,b,w) intermediate - 1e13 elements at 1,000
    vertices and 100 channels, which kills the process before it prints anything."""
    return (H @ S) @ np.conj(np.transpose(H, (0, 2, 1)))


def env_cov(H, w, S, ph, rho, chunk=128, M=None):
    """-> E, the covariance of filter(h^2). Chunked over lags."""
    nf, nV, _ = H.shape
    M = _M(H, S) if M is None else M
    Mf = M.reshape(nf, -1)
    E = np.zeros((nV, nV))
    for a in range(0, len(rho), chunk):
        b = min(a + chunk, len(rho))
        C = 2.0 * np.real((ph[a:b] * w[None, :]) @ Mf).reshape(b - a, nV, nV)
        E += 2.0 * np.einsum("t,tij->ij", rho[a:b], C ** 2, optimize=True)
    return E


def env_grad(H, w, G, S, ph, rho, chunk=128, M=None):
    """dJ/dS given dJ/dE = G. Chunked over lags, recomputing C per block.

    dE/dC_k is 4 rho_k C_k elementwise; the rest is the adjoint of the linear map
    S -> Phi(tau), which is w_f 2 exp(-i th) H^H M H Hermitianised - the same expression
    lagged.adjoint_lagged derives, without its centring step, because the square has to
    precede any linear operation on the observable."""
    nf, nV, K = H.shape
    M = _M(H, S) if M is None else M
    Mf = M.reshape(nf, -1)
    Y = np.zeros((nf, nV, nV), complex)
    for a in range(0, len(rho), chunk):
        b = min(a + chunk, len(rho))
        C = 2.0 * np.real((ph[a:b] * w[None, :]) @ Mf).reshape(b - a, nV, nV)
        Ms = (4.0 * rho[a:b, None, None]) * C * G[None]
        Y += (np.conj(ph[a:b]).T @ Ms.reshape(b - a, -1)).reshape(nf, nV, nV)
    out = np.empty((nf, K, K), complex)
    for f in range(nf):
        acc = (w[f] * 2.0) * (H[f].conj().T @ Y[f] @ H[f])
        out[f] = 0.5 * (acc + acc.conj().T)
    return out


def solve(H, w, ph, rho, T, iters=200, S0=None, chunk=128, verbose=True, trace=None):
    """max corr(E(S), T) over S >= 0, by projected gradient on the PSD cone.

    The same scheme xspec.solve uses, with the one difference that is the open question:
    E is QUADRATIC in S where the FC objective is linear in it, so convexity is not given
    by construction. On the toy it reached the same optimum from every random start to
    within 1e-4 relative; that is a measurement on one problem, not a proof."""
    nf, nV, K = H.shape
    S = (np.stack([np.eye(K, dtype=complex) for _ in range(nf)]) if S0 is None
         else np.array(S0, complex, copy=True))
    iu = np.triu_indices(nV, 1)
    t = np.asarray(T[iu], float)
    t = t - t.mean()
    t /= max(np.linalg.norm(t), 1e-300)

    def obj(S):
        E = env_cov(H, w, S, ph, rho, chunk)
        e = E[iu] - E[iu].mean()
        n = max(np.linalg.norm(e), 1e-300)
        return float(e @ t / n), E, n

    val, E, n = obj(S)
    if trace is not None:
        trace.append(val)
    step, fails = 1.0, 0
    for it in range(iters):
        # HALF on each of (i,j) and (j,i): the objective reads the upper triangle while
        # env_grad sums over every entry, so mirroring unhalved double-counts each
        # off-diagonal. Finite differences caught exactly that factor of two once.
        g = (t - val * (E[iu] - E[iu].mean()) / n) / n
        G = np.zeros((nV, nV))
        G[iu] = g
        G = 0.5 * (G + G.T)
        grad = env_grad(H, w, G, S, ph, rho, chunk)
        moved = False
        for _ in range(40):
            Tn = xspec._project(S + step * grad, 1, False, None)
            tr = sum(np.trace(Tn[f]).real for f in range(nf))
            if tr > 0:
                Tn = Tn / tr
            v2, E2, n2 = obj(Tn)
            if v2 > val:
                S, val, E, n = Tn, v2, E2, n2
                step *= 1.6
                moved = True
                if trace is not None:
                    trace.append(val)
                break
            step *= 0.4
        if not moved:
            fails += 1
            step *= 1e-3
            if fails >= 3:
                break
        if verbose and (it % 25 == 0 or it == iters - 1):
            print(f"    iter {it:4d}  corr {val:+.6f}", flush=True)
    return S, val


def observable(frames, kernel, frame_s, band, burn=0):
    """The envelope observable from simulated frames: square, smooth, bandpass.

    In that order. Filtering before squaring removes the carriers before they can beat,
    which is the whole mechanism - and it is what the linear path does."""
    import units, bandpass as bp
    X = np.asarray(frames, np.float64) ** 2
    X = units.smooth_frames(X, kernel) if kernel is not None else X
    if band is not None:
        X = bp.apply(X, frame_s, band[0], band[1])
    return X[burn:]
