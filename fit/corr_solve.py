"""Re-solve an existing fit against the correlation-form SQUARED ERROR objective.

Takes a recorded solve's npz and reuses its medium, frequency grid, solve vertices and
transfer function unchanged, so the only thing that differs from the run it is named after
is what is being minimised. That makes the pair directly comparable, which re-deriving the
setup from command-line flags would not.

The old objective, corr(model_edges, normal_scores(target)), cannot see dynamic range: it is
a scale-invariant ratio against a target whose values have been replaced by gaussian
quantiles. This one is

    J = sum_{i<j} ( dcentre(R)_ij - T_ij )^2,      R = M / sqrt(d_i d_j)

against the empirical FC in the same correlation form, values intact. See fit/corrfit.py for
why normalising is what makes squared error well posed, and for the gradient's extra
diagonal term.

BOTH objectives are reported every time. They are different quantities and the new one is
not comparable with anything in RUNS.md, so a run that only printed J would be unreadable
against the record.

  python fit/corr_solve.py grclip400_nolag --iters 200
  python fit/corr_solve.py env_grclip400 --envelope --iters 100 --init warm
"""
import _path  # noqa: F401
import os, time, argparse
import numpy as np
from scipy.stats import spearmanr

from paths import RESULTS
import xspec, provenance
import envelope as env
import lagged as _lg
import corrfit as cf
from interp_gap import build_H, double_centre
import fc_score


def empirical_correlation(t, sub):
    """The empirical FC as the CORRELATION matrix it is, on `sub`, double-centred.

    t.target_fc() returns the already-double-centred matrix, which cannot be put back into
    correlation form, so the file is re-read and its diagonal set to 1 before centring."""
    fc = np.asarray(np.load(t.fc_path, mmap_mode='r'))
    vts = np.load(fc_score.vertices_path(t.fc_path))
    idx = np.searchsorted(vts, t.vertices)
    M = fc[np.ix_(idx[sub], idx[sub])].astype(np.float64)
    M = 0.5 * (M + M.T)
    np.fill_diagonal(M, 1.0)
    return cf.dcentre(M)


def old_objective(M, t, sub, centre):
    """corr against the normal-scored target - the quantity every recorded run reports."""
    raw = np.asarray(t.target_fc()[np.ix_(sub, sub)], np.float64)
    raw = cf.dcentre(raw)
    iu = np.triu_indices(len(sub), 1)
    Tg = xspec.normal_scores(raw, iu)
    A = double_centre(M) if centre else np.asarray(M, float)
    a = A[iu] - A[iu].mean()
    b = Tg[iu] - Tg[iu].mean()
    return (float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))),
            float(spearmanr(A[iu], raw[iu]).statistic))


def _start(a, S_old):
    """The initial S: white, the recorded solve, or another corrfit result to continue."""
    if a.init_from:
        q = np.load(os.path.join(RESULTS, a.init_from + '.npz'), allow_pickle=True)
        print(f'  continuing from {a.init_from}: J {float(q["J"]):.4e}, spread '
              f'{float(q["spread"])/float(q["target_spread"]):.2f}x the target', flush=True)
        return q['S']
    return S_old if a.init == 'warm' else None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('tag')
    ap.add_argument('--envelope', action='store_true')
    ap.add_argument('--iters', type=int, default=200)
    ap.add_argument('--init', default='white', choices=('white', 'warm'),
                    help="'warm' starts from the recorded S, so J can only improve on what "
                         "the old objective reached; 'white' is the independent run")
    ap.add_argument('--lag-tol', type=float, default=1e-3, dest='lag_tol')
    ap.add_argument('--init-from', default='', dest='init_from',
                    help="continue from another corrfit npz by tag. The solve keeps no "
                         "state beyond S and the step size, so a resumed run is not "
                         "identical to one long run - the step restarts at 1.0 and the "
                         "line search has to find its scale again - but it beats "
                         "discarding the iterations already paid for")
    ap.add_argument('--lam', type=float, default=0.0,
                    help='weight on (sd(R) - sd(T))^2. 0 is pure squared error, which '
                         'SHRINKS the spread to rho*sd(T); >0 pushes it back')
    ap.add_argument('--chunk', type=int, default=128)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--out', default='')
    a = ap.parse_args()

    f = os.path.join(RESULTS, ('envfit_' if a.envelope else 'xspec_') + a.tag + '.npz')
    z = np.load(f, allow_pickle=True)
    S_old = xspec.load_S(z)
    w = np.asarray(z['w' if a.envelope else 'H_w'], float)
    Hall, respf, kern, c, t, g = build_H(a.tag, z, a.envelope, workers=a.workers)
    idx, sub, nV, K = g['idx'], g['sub'], g['nV'], g['K']
    # build_H returns every rfft bin because interp_gap compares quadratures; the SOLVE
    # lives on the recorded grid, so take those bins and let the rest go
    H = np.ascontiguousarray(Hall[idx]); del Hall
    iu = np.triu_indices(nV, 1)

    if a.envelope:
        lags, rho = env.filter_lags(respf, g['pad'], tol=a.lag_tol)
        ph = _lg.phases(idx, g['pad'], lags)
        fwd, adj = cf.envelope_pair(H, w, ph, rho, chunk=a.chunk)
        print(f'  {len(lags)} lags carry the filter autocorrelation', flush=True)
    else:
        fwd, adj = cf.linear_pair(H, w)

    T = empirical_correlation(t, sub)
    print(f'  target: empirical FC in correlation form, spread {T[iu].std():.4f}',
          flush=True)
    R0, _ = cf.correlation_form(fwd(S_old))
    o0, s0 = old_objective(fwd(S_old), t, sub, not a.envelope)
    p0 = float(np.corrcoef(cf.dcentre(R0)[iu], T[iu])[0, 1])
    print(f'  the recorded solve: J {float(((cf.dcentre(R0)[iu]-T[iu])**2).sum()):.6e}  '
          f'spread {cf.dcentre(R0)[iu].std():.4f} ({cf.dcentre(R0)[iu].std()/T[iu].std():.2f}x '
          f'the target)  pearson(R,T) {p0:+.4f}  old objective {o0:+.4f}  '
          f'spearman {s0:+.4f}', flush=True)

    t0, tr = time.time(), []
    S, J = cf.solve(fwd, adj, T, len(idx), K, iters=a.iters, lam=a.lam,
                    S0=_start(a, S_old), trace=tr)
    M = fwd(S)
    Rc = cf.dcentre(cf.correlation_form(M)[0])
    o1, s1 = old_objective(M, t, sub, not a.envelope)
    pear = float(np.corrcoef(Rc[iu], T[iu])[0, 1])
    print(f'\n  solved [{time.time()-t0:.0f}s]: J {J:.6e} (from {tr[0]:.6e})')
    print(f'    spread {Rc[iu].std():.4f} ({Rc[iu].std()/T[iu].std():.2f}x the target, '
          f'was {cf.dcentre(R0)[iu].std()/T[iu].std():.2f}x)')
    print(f'    pearson(R, T) {pear:+.4f}  - least squares predicts spread/sd(T) = pearson,'
          f' measured {Rc[iu].std()/T[iu].std():.4f}')
    print(f'    old objective {o1:+.4f} (was {o0:+.4f})   '
          f'spearman vs empirical {s1:+.4f} (was {s0:+.4f})')
    out = a.out or f'corrfit_{a.tag}' + ('' if a.init == 'white' else '_warm')
    np.savez(os.path.join(RESULTS, out + '.npz'), S=S, idx=idx, w=w, sub=sub, J=J,
             J0=tr[0], spread=Rc[iu].std(), target_spread=T[iu].std(),
             old_objective=o1, spearman=s1, old_objective_before=o0, spearman_before=s0,
             pearson=pear, pearson_before=p0, lam=a.lam, envelope=a.envelope,
             src=a.tag,
             **provenance.stamp(a))
    print(f'  wrote results/{out}.npz')


if __name__ == '__main__':
    main()
