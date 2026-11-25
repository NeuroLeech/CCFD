"""Does either model reproduce the empirical FC's DYNAMIC RANGE, or only its ordering?

Every score in this project is a Spearman, which is rank based and therefore scale free: a
nearly flat model FC whose small deviations happen to be ordered like the empirical FC
scores as well as one that reproduces its spread. So "the envelope fits better" and "the
envelope looks nothing like the data" can both be true, and the metric cannot tell them
apart. This compares the off-diagonal spread of each model's CLOSED FORM - the prediction
itself, no simulation and no estimator noise - against the empirical FC's own spread, on
the same solve vertices.

WHAT IT FOUND, on 400 vertices with the clipped graded medium:

                               mean       sd            p5       p95   spearman vs emp
    EMPIRICAL FC            +0.0007   0.1289       -0.1613   +0.2575          -
    linear                  -0.0011   0.2008 1.56x -0.2781   +0.3506      +0.7161
    envelope                +0.0030   0.0628 0.49x -0.0676   +0.1102      +0.7472

Neither matches, and they miss in opposite directions: the envelope predicts HALF the
spread of the real thing and the linear 1.56x too much. The envelope scores higher while
being the one further from the data in absolute terms, which is exactly what a rank metric
cannot see.

IS THE FLATNESS STRUCTURAL? NO, AND THE OBVIOUS ARGUMENT THAT IT IS, IS WRONG.

E_ij = 2 sum_tau rho(tau) C_ij(tau)^2 involves the Hadamard square of C, and squaring a
correlation maps [-1,1] -> [0,1], losing the sign and compressing the range - squaring the
empirical FC compresses its spread 3.53x, from 0.1280 to 0.0362. That looks like a ceiling
the envelope cannot clear, and it is not one: the solve produced 0.0628, which is 1.7x past
it, and that alone disproves it.

The argument needs rho >= 0 and rho is 60.7% negative over the 1,579 kept lags, with
positive mass +0.9581 against negative -0.9626 - cancelling to 0.5%. That cancellation is
exact in principle, because sum_tau rho(tau) is the filter's squared response at DC and the
passband excludes DC. So E is a DIFFERENCE of Hadamard squares, with no positivity and no
compression bound, and what it measures is how squared correlation VARIES WITH LAG rather
than its level - which is also why its edges can be negative.

What the reachable spread actually is remains unmeasured. xspec.family_member moves along a
regulariser while holding the fit within eps of the argmax, so correlation-form spread as
that regulariser would answer it as a number and separate "the solve picked a flat member of
the admissible family" from "the family is flat".

CENTRE AND NORMALISE THE SAME WAY ON BOTH SIDES, or the comparison measures the convention.
The target is double-centred, so the model has to be. And Spearman is NOT invariant to
dividing each edge by sqrt(d_i d_j): the realised scoring path z-scores every vertex
timecourse, so the model FC it builds is in correlation form, and scoring an unnormalised
covariance instead reports +0.7691 for the envelope where the matching number is +0.7472.

  python fit/dyn_range.py
"""
import _path  # noqa: F401
import os, argparse
import numpy as np
from scipy.stats import spearmanr

from paths import RESULTS
import xspec
from interp_gap import build_H, spectra, cov_linear, cov_envelope
import envelope as env
import lagged as _lg


def off_diagonals(tag, isenv, workers=8):
    """-> (uncentred, double-centred) correlation-form off-diagonals of the closed form."""
    f = os.path.join(RESULTS, ('envfit_' if isenv else 'xspec_') + tag + '.npz')
    z = np.load(f, allow_pickle=True)
    H, respf, kern, c, t, g = build_H(tag, z, isenv, workers=workers)
    Si, _ = spectra(xspec.load_S(z), g['idx'], g['nb'])
    allb = np.arange(1, g['nb'])
    if isenv:
        lags, rho = env.filter_lags(respf, g['pad'], tol=1e-3)
        M = cov_envelope(H, Si, allb, _lg.phases(allb, g['pad'], lags), rho)
    else:
        M = cov_linear(H, Si, allb)
    del H
    iu = np.triu_indices(len(M), 1)
    # the UNcentred diagonal sets the scale in both, because that is what z-scoring the
    # vertex timecourses does - centring is applied to the edges afterwards, not before
    d = np.sqrt(np.clip(np.diag(M), 1e-300, None))
    Mc = M - M.mean(0, keepdims=True) - M.mean(1, keepdims=True) + M.mean()
    return (M / np.outer(d, d))[iu], (Mc / np.outer(d, d))[iu], g['sub'], t


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--linear', default='grclip400_nolag')
    ap.add_argument('--envelope', default='env_grclip400')
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()

    rows, sub, t = {}, None, None
    for nm, tag, isenv in (('linear', a.linear, False), ('envelope', a.envelope, True)):
        u, dc, sub, t = off_diagonals(tag, isenv, a.workers)
        rows[nm] = (u, dc)
    emp = np.asarray(t.target_fc()[np.ix_(sub, sub)], np.float64)
    e = emp[np.triu_indices(len(sub), 1)]

    def line(nm, v, ref=None):
        print(f'  {nm:<26s} {v.mean():>+8.4f} {v.std():>8.4f}'
              f'{"" if ref is None else f" {v.std()/ref:>5.2f}x"}'
              f' {np.percentile(v,5):>+9.4f} {np.percentile(v,95):>+8.4f} '
              f'{"-" if ref is None else f"{spearmanr(v, e).statistic:+.4f}":>16s}')

    print(f'\n  off-diagonal spread on {len(sub)} vertices, {len(e)} edges')
    print(f'  {"":<26s} {"mean":>8s} {"sd":>8s} {"":>6s} {"p5":>9s} {"p95":>8s} '
          f'{"spearman vs emp":>16s}')
    line('EMPIRICAL FC', e)
    for nm in ('linear', 'envelope'):
        line(nm + ', uncentred', rows[nm][0], e.std())
        line(nm + ', double-centred', rows[nm][1], e.std())


if __name__ == '__main__':
    main()
