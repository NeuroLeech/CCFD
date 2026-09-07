"""Solve for the DIFFERENCE between two conditions, instead of solving each and subtracting.

Fitting ON and OFF separately gave inputs that agree to 0.994 and fields that agree to
0.990, on targets that are not identical. That is what an objective dominated by what the
two conditions SHARE will do: the solve reproduces its target at about 0.75, so a quarter
of the structure is unexplained already, and if the between-condition difference is a few
percent of the variance it sits entirely inside that residual and pulls on nothing.

The fix follows from the model being linear. C(S) = sum_f w_f 2 Re(H S H^H) is linear in
S and H is shared by both arms, so

    C(S_on) - C(S_off) = C(S_on - S_off)

and the difference between the two predicted FCs is itself a model output. So the
difference can be made the objective directly:

    maximise  corr( C(S) - C(S_off),  FC_on - FC_off )   over S >= 0

which is the same projected-gradient problem `xspec.solve` already solves, with the target
replaced by the target DIFFERENCE and a fixed matrix subtracted from the model. Two things
change and both matter:

  NO TRACE NORMALISATION. The usual objective is scale-invariant in S because C(S) is
  homogeneous; subtracting the fixed C(S_off) destroys that, so the SIZE of S is now part
  of the answer and cannot be normalised away.

  THE STARTING POINT IS THE ANSWER SO FAR. Starting at S_on evaluates exactly what the two
  independent solves already achieve on the difference, so the first number this prints is
  the baseline the search has to beat.

Then the second question, which is not the same one: can the model express the difference
AT ALL without giving up the fit it already has? That is `--eps`, which holds
corr(C(S), FC_off) within eps of what S_off achieves while maximising the difference
objective - the constrained version of the same ascent. If the difference objective can be
driven up freely but collapses under that constraint, the model can represent the
difference only by abandoning the condition it was fitted to.

  python onoff_solve.py --a ck_check_on_out --b ck_check_off_out
"""
import os, argparse
import numpy as np

from paths import RESULTS
import xspec
from onoff_fields import load, rebuild_H, tri, _pearson


def centre_mask(C):
    """Double-centre with means over all entries, then drop the diagonal.

    Means over ALL entries, not excluding the diagonal: a centring that excludes it is not
    self-adjoint, and every gradient built on it silently loses a term - PLAN section 6
    records the factor of 2.6 that cost. This is the same operation `xspec.solve.model`
    applies, in the same order."""
    C = C - C.mean(0, keepdims=True) - C.mean(1, keepdims=True) + C.mean()
    np.fill_diagonal(C, 0.0)
    return C


def make_ops(H, w):
    """-> (model, adjoint) with exactly xspec.solve's conventions."""
    nf, nV, K = H.shape
    Ph, Qh = xspec._stack_conj(H)
    Hr, Hi = xspec._stack_cols(H)
    w2 = (2.0 * np.asarray(w, float))[:, None, None]

    def model(S):
        C = np.zeros((nV, nV))
        for f in range(nf):
            C += w[f] * 2.0 * np.real((H[f] @ S[f]) @ H[f].conj().T)
        return centre_mask(C)

    def adjoint(M):
        Mc = M - M.mean(0, keepdims=True) - M.mean(1, keepdims=True) + M.mean()
        Yr = (Mc @ Hr).reshape(nV, nf, K).transpose(1, 0, 2)
        Yi = (Mc @ Hi).reshape(nV, nf, K).transpose(1, 0, 2)
        return ((Ph @ Yr - Qh @ Yi) + 1j * (Ph @ Yi + Qh @ Yr)) * w2

    return model, adjoint


def ascend(model, adjoint, K, dTn, S0, iters=200, fit=None, floor=None, verbose=True):
    """Projected gradient on corr(C(S) - K, dTn), S >= 0.

    `fit`/`floor` add the constraint that a second objective stay above a floor - the
    difference maximised at MATCHED fit, which is a different question from the difference
    maximised outright. A step is accepted only if the difference objective improves AND
    the constraint still holds, the same accept rule `family_member` uses."""
    S = np.array(S0, dtype=complex, copy=True)

    def rho(S):
        D = model(S) - K
        n = np.linalg.norm(D)
        # a zero difference is not a bad one, it is an undefined one: report it as 0 so
        # the very first accepted step is an improvement rather than a comparison
        # against a sentinel
        return (float((D * dTn).sum() / n) if n > 1e-30 else 0.0), D, n

    val, D, n = rho(S)
    A0 = adjoint(dTn)
    step = 0.1 * float(np.linalg.norm(S)) / max(np.linalg.norm(A0), 1e-30)
    # Starting AT S_off makes D identically zero, and the gradient of <D,T>/||D|| is
    # undefined there - which is exactly where part (d) has to start, since the fit is
    # being held to that arm. At D = 0 the ratio's denominator contributes nothing and
    # the ascent direction is the linear part alone.
    hist = [val]
    blocked = 0
    for it in range(iters):
        G = A0 if n < 1e-30 else A0 / n - (val / n) * adjoint(D / n)
        moved = False
        for _ in range(14):
            T = xspec._project(S + step * G)
            v2, D2, n2 = rho(T)
            if v2 > val + (1e-12 if n < 1e-30 else 0.0):
                if fit is None or fit(T) >= floor:
                    S, val, D, n = T, v2, D2, n2
                    step *= 1.6
                    moved = True
                    break
                blocked += 1
            step *= 0.4
        if not moved:
            break
        hist.append(val)
        if verbose and it % 20 == 0:
            print(f"    iter {it:4d}  difference corr {val:+.4f}"
                  + ("" if fit is None else f"   fit {fit(S):+.4f} (floor {floor:+.4f})"),
                  flush=True)
    return S, val, dict(iters=len(hist) - 1, blocked=blocked, hist=hist)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--a", default="ck_check_on_out", help="the ON arm")
    ap.add_argument("--b", default="ck_check_off_out", help="the OFF arm")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--eps", default="0.002,0.01,0.05",
                    help="fit budgets for the matched-fit run (part d)")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    from mesh_cache import load_cortex
    import fc_score
    c = load_cortex("fsaverage5", verbose=False)
    zA, zB = load(a.a), load(a.b)
    sub = np.asarray(zA["sub"])
    if not np.array_equal(sub, np.asarray(zB["sub"])):
        raise SystemExit("  the two arms were solved on different vertices")

    tA = fc_score.FCTarget(c, fc_path=str(zA["fc_path"]), centre="double", verbose=False)
    tB = fc_score.FCTarget(c, fc_path=str(zB["fc_path"]), centre="double", verbose=False)
    rawA = centre_mask(np.asarray(tA.target_fc()[np.ix_(sub, sub)], np.float64))
    rawB = centre_mask(np.asarray(tB.target_fc()[np.ix_(sub, sub)], np.float64))

    print(f"  A = {a.a}\n  B = {a.b}\n  {len(sub)} solve vertices, "
          f"{len(tri(rawA))} edges")
    print(f"\n  HOW DIFFERENT ARE THE TARGETS")
    print(f"    corr(FC_A, FC_B) over the solve edges: {_pearson(tri(rawA), tri(rawB)):.6f}")
    dT = centre_mask(rawA - rawB)
    print(f"    ||FC_A - FC_B|| / ||FC_A||: "
          f"{np.linalg.norm(tri(dT))/np.linalg.norm(tri(rawA)):.4f}")

    H, w, idx = rebuild_H(c, zA, tA.cols[sub], workers=a.workers)
    model, adjoint = make_ops(H, w)
    SA, SB = np.asarray(zA["S"]), np.asarray(zB["S"])
    CA, CB = model(SA), model(SB)
    print(f"    corr(C_A, C_B) for the two solved models: "
          f"{_pearson(tri(CA), tri(CB)):.6f}")

    dTn = dT / np.linalg.norm(dT)
    D0 = CA - CB
    rho0 = float((D0 * dTn).sum() / max(np.linalg.norm(D0), 1e-30))
    print(f"\n  BASELINE: the two independent solves' own difference against the "
          f"target difference")
    print(f"    corr(C_A - C_B, FC_A - FC_B) = {rho0:+.6f}")
    for nm, C, raw in (("A", CA, rawA), ("B", CB, rawB)):
        print(f"    (for reference, corr(C_{nm}, FC_{nm}) = "
              f"{_pearson(tri(C), tri(raw)):+.4f})")

    # ---- (a) the difference as the objective, unconstrained --------------------------
    print(f"\n  (a) MAXIMISE THE DIFFERENCE OBJECTIVE, S >= 0, no fit constraint")
    best = (None, rho0, "the existing S_A")
    for nm, S0 in (("from S_A", SA), ("from S_B", SB),
                   ("from white", np.stack([np.eye(SA.shape[1], dtype=complex)
                                            for _ in range(SA.shape[0])])
                    * float(np.abs(SA).max()))):
        S, v, rep = ascend(model, adjoint, CB, dTn, S0, iters=a.iters, verbose=False)
        print(f"    {nm:<12s} -> {v:+.6f}  after {rep['iters']} accepted steps")
        if v > best[1]:
            best = (S, v, nm)
    print(f"    best {best[1]:+.6f} ({best[2]}), against the baseline {rho0:+.6f}")

    # ---- (d) the same at matched fit to the OFF target -------------------------------
    TgtB = xspec.normal_scores(rawB)
    TgtBn = TgtB.copy(); np.fill_diagonal(TgtBn, 0.0)
    TgtBn /= np.linalg.norm(TgtBn)

    def fitB(S):
        C = model(S)
        n = np.linalg.norm(C)
        return float((C * TgtBn).sum() / n) if n > 0 else -1.0

    f0 = fitB(SB)
    print(f"\n  (d) THE SAME, HOLDING THE FIT TO {a.b} WITHIN eps OF {f0:+.4f}")
    print(f"    {'eps':>8s} {'difference corr':>16s} {'fit kept':>10s} "
          f"{'steps':>6s} {'blocked':>8s}")
    out = {}
    for eps in [float(v) for v in a.eps.split(",")]:
        S, v, rep = ascend(model, adjoint, CB, dTn, SB, iters=a.iters,
                           fit=fitB, floor=f0 - eps, verbose=False)
        print(f"    {eps:>8g} {v:>16.6f} {fitB(S):>10.4f} {rep['iters']:>6d} "
              f"{rep['blocked']:>8d}")
        out[f"S_eps{eps:g}"] = S

    p = os.path.join(RESULTS, f"onoff_solve_{a.a}_{a.b}.npz")
    np.savez(p, rho0=rho0, best=best[1], S_free=best[0] if best[0] is not None else SA,
             dT=dT, sub=sub, **out)
    print(f"\n  wrote {p}")


if __name__ == "__main__":
    main()
