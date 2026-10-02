"""Fit the MAGNITUDE of FC as well as its order: squared error on the correlation form.

Every objective in this project so far maximises corr(model_edges, target_edges), a
scale-invariant ratio. That was forced: the model covariance C has a free scale, and least
squares against a free scale is minimised by shrinking the model to nothing. The cost is
that the fit cannot see dynamic range at all, and it shows - measured on 400 vertices, the
linear solve's off-diagonal spread is 1.56x the empirical FC's and the envelope's is 0.49x,
while both score about +0.72 Spearman. A rank metric cannot distinguish them. See
fit/dyn_range.py.

NORMALISING REMOVES THE REASON FOR THE RATIO. With d = diag(M),

    R = M / sqrt(d_i d_j)

is EXACTLY invariant to S -> alpha S, because M -> alpha M carries d -> alpha d. So squared
error on R has no shrink-to-zero solution, the trace normalisation the solvers already apply
becomes harmless gauge fixing rather than something the objective fights, and magnitude
becomes a legitimate target. R is also the form the realised score actually reads: it
z-scores every vertex timecourse, which is why scoring an unnormalised covariance reported
+0.7691 for the envelope where the matching figure is +0.7472.

    J(S) = sum_{i<j} ( dcentre(R)_ij - T_ij )^2

with T the empirical FC in the same correlation form and double-centred the same way, and
NOT normal-scored: normal_scores replaces the target's values with gaussian quantiles, which
is exactly the information this objective exists to use.

THE GRADIENT HAS A TERM THE OLD ONE DID NOT. R depends on M through the numerator and
through the diagonal, so dJ/dM carries a diagonal correction:

    Y_ij  = Ghat_ij / sqrt(d_i d_j)                  Ghat = dcentre(dJ/dR)
    Y_kk += -(1/d_k) sum_j Ghat_kj R_kj

Nothing in xspec or envelope expects that - xspec.solve's adjoint double-centres its
argument internally, which would centre the diagonal term as well - so both adjoints are
written out here and checked against finite differences. A constant factor in a gradient
survives a backtracking line search unnoticed, which is how env_grad once ran for a whole
session with a factor of two in it.

MEASURED ON THE LINEAR FIT, 400 vertices, 200 iterations, same medium and grid as
grclip400_nolag, which the old objective solved to spread 1.58x the target:

    lam     spread  x target   pearson(R,T)   old objective   spearman vs empirical
    -        0.2028    1.58x        -            +0.7226            +0.7159
    0        0.1025    0.80x      +0.7302        +0.6171            +0.6549
    1        0.1134    0.89x      +0.7401        +0.6237            +0.6617
    10       0.1248    0.98x      +0.7337        +0.6260            +0.6576
    100      0.1280    1.00x      +0.6564        +0.5681            +0.5864

lam=10 lands the spread on the target while holding pearson at +0.734, which is where the
unconstrained squared-error optimum already was - so matching the magnitude costs almost
nothing against least squares. What it costs is against the OLD objective: spearman +0.658
rather than +0.716. Realised over 2,308 s on all 9,310 vertices that trade is cheaper still:
+0.6356 +- 0.0033 against the recorded solve's +0.6537 +- 0.0017, a cost of 0.018 for an
exact dynamic-range match. lam=0 realises +0.6277 and lam=100 collapses to +0.5557.

THE SAME SWEEP ON THE ENVELOPE, and the two models are not symmetric:

    lam     spread  x target   pearson(R,T)   spearman vs empirical
    -        0.0626    0.49x      +0.7322           +0.7798
    0        0.1068    0.83x      +0.9061           +0.7313
    10       0.1249    0.98x      +0.7446           +0.5757
    100      0.1273    0.99x      +0.4675           +0.3072

The envelope starts BELOW the target and has to be pushed up, where the linear starts above
and comes down. Coming down is free; going up is not. At lam=0, with no constraint at all,
asking the envelope to fit magnitudes took its spread from 0.49x to 0.83x and its pearson
from +0.7322 to +0.9061 - so its flatness was never a ceiling, and the fit it reaches on the
correlation form is well above anything the linear reaches. Forcing the last 17% costs
pearson 0.16 at lam=10 and 0.44 at lam=100. Both envelope runs at 150 iterations were still
falling fast (lam=10: J 718 -> 660 over the last 24), so those are lower bounds.

AND IT IS THE ENVELOPE THAT THIS OBJECTIVE IS FOR. Realised over 2,308 s on all 9,310
vertices, same seeds for both:

    ENVELOPE  correlation-form objective (lam 0)   sim +0.5019 +- 0.0216
              the recorded solve                   sim +0.3416 +- 0.0183
    LINEAR    correlation-form objective (lam 10)  sim +0.6356 +- 0.0033
              the recorded solve                   sim +0.6537 +- 0.0017

So it is worth +0.160 of Spearman to the envelope and costs the linear 0.018. The envelope's
gain has a mechanism beyond the better fit: its realised score was attenuated by estimator
noise as r = contrast / sqrt(contrast^2 + noise^2), and raising the contrast from 0.49x to
0.83x of the target raises r directly. Fitting the magnitude and fixing the sampling are the
same intervention, because low contrast was what made the fourth moment so hard to estimate
- see fit/why_noisy.py, where the contrast factor was the dominant one.

AT 400 ITERATIONS, where 150 was not close to converged (envelope lam=10 fell J 660 -> 414):

                            spread  x target   pearson(R,T)   spearman vs empirical
    envelope  old            0.0626    0.49x      +0.7322           +0.7798
    envelope  lam 0          0.1102    0.86x      +0.9278           +0.7559
    envelope  lam 10         0.1253    0.98x      +0.8408           +0.6721
    linear    old            0.2028    1.58x      +0.7351           +0.7159
    linear    lam 10         0.1248    0.98x      +0.7564           +0.6721

At MATCHED dynamic range and matched iterations the two models tie exactly on Spearman and
the envelope fits the values far better, +0.8408 against +0.7564. Under the OLD objective
their pearson was +0.7322 against +0.7351 - indistinguishable - so the rank objective was
hiding a real difference between the models rather than finding none.

The best single model is the envelope at lam 0: 0.86x the empirical spread with pearson
+0.9278, which is above every other row here, and a Spearman only 0.024 below what the old
objective reached while carrying 1.8x its dynamic range.

REALISED, 2,308 s, 6 draws pooled as edge vectors, all 9,310 vertices, matched seeds:

    envelope  lam 0, 400 it    per draw +0.4998 +- 0.0237    POOLED +0.6237
    envelope  old objective    per draw +0.3569 +- 0.0325    POOLED +0.5225
    linear    lam 10           per draw +0.6356 +- 0.0033    (r 0.97, already converged)
    linear    old objective    per draw +0.6537 +- 0.0017    (r 0.98, already converged)

+0.101 pooled for the envelope, and the per-draw figures understate it by 0.12 because each
draw is attenuated - which is exactly why pooling the edge vector rather than the scores
matters. Neither envelope row has converged at 6 draws; the new one is nearer its asymptote
because its higher contrast attenuates less. The linear needs no pooling at all.

ITERATION COUNT IS A REGULARISATION PARAMETER HERE TOO, exactly as xspec.solve's docstring
says it is for the old objective. Continuing the envelope lam=0 solve from 400 to 1,000
iterations improves EVERY solve-side number - J 188 -> 139, spread 0.86x -> 0.89x, pearson
+0.9278 -> +0.9470, solve-vertex spearman +0.7559 -> +0.7724, the last now within 0.007 of
what the old objective reached while carrying 1.8x its dynamic range - and makes the REALISED
score worse:

    pooled draws              1       2       3       4       5       6      12
    400 iterations        +0.4929 +0.5512 +0.5869 +0.5955 +0.6165 +0.6237    -
    1,000 iterations      +0.4652 +0.4990 +0.5369 +0.5529 +0.5731 +0.5858 +0.6139

0.038 worse at matched draws. So J on 400 solve vertices and the score on 9,310 part company
past a few hundred iterations, and --iters has to be chosen on held-out vertices rather than
run to convergence.

The 400-iteration solution pooled over 12 draws reaches +0.6507 and has flattened there
(+0.6475, +0.6501, +0.6507 over the last three).

POOLED THE SAME WAY, THE LINEAR REACHES +0.6807. Its 2-draw per-draw mean was +0.6356, so
pooling lifted it by 0.045, not the 0.01-0.015 an r of 0.978 suggested. Matched at 12 pooled
draws on all 9,310 vertices:

    linear    lam 10, 400 it     POOLED +0.6807   (plateaued by draw 6)
    envelope  lam 0,  400 it     POOLED +0.6507   (plateaued by draw 10)

So the envelope sits 0.030 below the linear on the realised metric, and the earlier reading
that it had drawn level came from comparing its pooled figure against the linear's unpooled
one. The envelope's advantage is confined to the solve side: at matched dynamic range its
pearson on the correlation form is +0.8408 against +0.7564, and at lam 0 it is +0.9470 while
carrying 0.89x the empirical spread.

The linear under the OLD objective was not pooled - the run was stopped after one of twelve
draws, at +0.6520 - so there is no 12-draw figure for it.

The attenuation argument is approximate, not exact: at lam=0 it predicts spread/sd(T) =
pearson = 0.730 and the measured ratio is 0.801, because the prediction holds the model's
SHAPE fixed and only optimises amplitude, while here the shape co-adapts.

  python fit/corrfit.py --selfcheck
"""
import _path  # noqa: F401
import numpy as np

import xspec
import envelope as env


def dcentre(X):
    """Double-centre. Self-adjoint and idempotent, so it is its own adjoint in the chain."""
    X = np.asarray(X, float)
    return X - X.mean(0, keepdims=True) - X.mean(1, keepdims=True) + X.mean()


def correlation_form(M):
    """-> (R, d). The UNcentred diagonal sets the scale, which is what z-scoring the vertex
    timecourses does; centring is applied to the edges afterwards, not before."""
    d = np.clip(np.diag(np.asarray(M, float)), 1e-300, None)
    return np.asarray(M, float) / np.sqrt(np.outer(d, d)), d


def loss(M, T, iu, lam=0.0, s_T=None):
    """-> (J, Y) with Y = dJ/dM, including the diagonal term from the normalisation.

    `lam` adds N*lam*(sd(R) - sd(T))^2, which is needed rather than optional if the aim is
    matched magnitude. Least squares alone does NOT give it: for a model correlating rho
    with the target, the squared-error optimum sits at amplitude rho*sd(T), because
    shrinking an imperfect prediction toward the mean lowers squared error. Measured on the
    linear fit, lam=0 took the spread from 1.58x the target straight past 1.0 to 0.80x. So
    magnitude has to be CONSTRAINED, not merely included."""
    R, d = correlation_form(M)
    Rc = dcentre(R)
    r = Rc[iu] - np.asarray(T)[iu]
    J = float(r @ r)
    # G symmetric with G[iu] = G[il] = r and a zero diagonal, so the FULL-matrix inner
    # product <G, dRc> equals dJ - the factor of two from mirroring is the factor of two
    # in d(r^2)/dr, not an extra one
    g = r.copy()
    if lam > 0.0:
        n = len(r)
        sT = float(np.asarray(T)[iu].std()) if s_T is None else float(s_T)
        rc = Rc[iu] - Rc[iu].mean()
        sR = float(np.sqrt((rc @ rc) / n))
        J += lam * n * (sR - sT) ** 2
        # d/d rc of lam*n*(sR - sT)^2, with sR = ||rc||/sqrt(n)
        g = g + lam * (sR - sT) * rc / max(sR, 1e-300)
    G = np.zeros_like(Rc)
    G[iu] = g
    G = G + G.T
    Gh = dcentre(G)
    Y = Gh / np.sqrt(np.outer(d, d))
    Y[np.diag_indices_from(Y)] -= (Gh * R).sum(1) / d
    return J, Y


# ------------------------------------------------------------------ the two forwards
def linear_pair(H, w):
    """-> (forward, adjoint) for C = sum_f w_f 2Re(H_f S_f H_f^H)."""
    wf = np.asarray(w, float)

    def forward(S):
        return np.einsum('f,fij->ij', 2.0 * wf,
                         np.real((H @ S) @ np.conj(np.transpose(H, (0, 2, 1)))),
                         optimize=True)

    def adjoint(Y, S):
        Yc = np.asarray(Y, float)
        out = np.empty((H.shape[0], H.shape[2], H.shape[2]), complex)
        for f in range(H.shape[0]):
            a = (2.0 * wf[f]) * (H[f].conj().T @ Yc @ H[f])
            out[f] = 0.5 * (a + a.conj().T)
        return out

    return forward, adjoint


def envelope_pair(H, w, ph, rho, chunk=128):
    """-> (forward, adjoint) for E = 2 sum_tau rho(tau) C(tau)^2."""
    def forward(S):
        return env.env_cov(H, w, S, ph, rho, chunk)

    def adjoint(Y, S):
        return env.env_grad(H, w, np.asarray(Y, float), S, ph, rho, chunk)

    return forward, adjoint


def solve(forward, adjoint, T, nf, K, iters=200, S0=None, verbose=True, trace=None,
          lam=0.0):
    """min J over S >= 0, projected gradient with a backtracking line search.

    The trace normalisation is kept, but it means something different here: J is exactly
    invariant to S -> alpha S, so fixing the trace removes a gauge freedom rather than
    constraining the model. Any S and its multiples are the same point of the objective."""
    S = (np.stack([np.eye(K, dtype=complex) for _ in range(nf)]) if S0 is None
         else np.array(S0, complex, copy=True))
    nV = T.shape[0]
    iu = np.triu_indices(nV, 1)

    def norm_trace(X):
        tr = sum(np.trace(X[f]).real for f in range(nf))
        return X / tr if tr > 0 else X

    S = norm_trace(xspec._project(S, 1, False, None))
    J, Y = loss(forward(S), T, iu, lam)
    if trace is not None:
        trace.append(J)
    step, fails = 1.0, 0
    for it in range(iters):
        g = adjoint(Y, S)
        moved = False
        for _ in range(40):
            Sn = norm_trace(xspec._project(S - step * g, 1, False, None))
            J2, Y2 = loss(forward(Sn), T, iu, lam)
            if J2 < J:
                S, J, Y = Sn, J2, Y2
                step *= 1.6
                moved = True
                if trace is not None:
                    trace.append(J)
                break
            step *= 0.4
        if not moved:
            fails += 1
            step *= 1e-3
            if fails >= 3:
                break
        if verbose and (it % 25 == 0 or it == iters - 1):
            R, _ = correlation_form(forward(S))
            print(f'    iter {it:4d}  J {J:.6e}  spread {dcentre(R)[iu].std():.4f}',
                  flush=True)
    return S, J


# ------------------------------------------------------------------------ self-check
def _selfcheck(nf=4, nV=9, K=3, seed=0, eps=1e-6):
    """Finite-difference both adjoints. A constant factor here is invisible downstream."""
    rng = np.random.default_rng(seed)
    H = (rng.normal(size=(nf, nV, K)) + 1j * rng.normal(size=(nf, nV, K))) / np.sqrt(K)
    w = np.abs(rng.normal(size=nf)) + 0.5
    A = rng.normal(size=(nf, K, K)) + 1j * rng.normal(size=(nf, K, K))
    S = np.einsum('fab,fcb->fac', A, A.conj())
    T = dcentre(np.tanh(rng.normal(size=(nV, nV))))
    T = 0.5 * (T + T.T)
    iu = np.triu_indices(nV, 1)

    # real filter lags, not random: rho must be symmetric in the way the lag pairing
    # assumes or E is not symmetric and the Hermitianised adjoint is not its adjoint
    ref = 64
    resp = np.zeros(ref // 2 + 1)
    resp[2:7] = 1.0
    lags, rho = env.filter_lags(resp, ref, tol=1e-12)
    ph = __import__('lagged').phases(np.arange(1, nf + 1), ref, lags)

    ok = True
    for lam in (0.0, 3.0):
      print(f'  lam = {lam:g}')
      for nm, (fwd, adj) in (('linear', linear_pair(H, w)),
                             ('envelope', envelope_pair(H, w, ph, rho, chunk=32))):
        J0, Y = loss(fwd(S), T, iu, lam)
        g = adj(Y, S)
        # a Hermitian perturbation, since S is constrained Hermitian
        B = rng.normal(size=(nf, K, K)) + 1j * rng.normal(size=(nf, K, K))
        D = 0.5 * (B + np.conj(np.transpose(B, (0, 2, 1))))
        ana = float(sum(np.real(np.vdot(g[f], D[f])) for f in range(nf)))
        Jp = loss(fwd(S + eps * D), T, iu, lam)[0]
        Jm = loss(fwd(S - eps * D), T, iu, lam)[0]
        num = (Jp - Jm) / (2 * eps)
        rel = abs(ana - num) / max(abs(num), 1e-30)
        ok &= rel < 1e-5
        print(f'    {nm:<9s} analytic {ana:+.8f}  numeric {num:+.8f}  '
              f'relative {rel:.2e}  {"ok" if rel < 1e-5 else "DISAGREE"}')
    print(f'  {"gradients agree" if ok else "GRADIENT IS WRONG"}')
    return ok


if __name__ == '__main__':
    import sys
    if '--selfcheck' in sys.argv:
        raise SystemExit(0 if _selfcheck() else 1)
    print(__doc__)
