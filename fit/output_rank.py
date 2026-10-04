"""Does the EXPECTED output covariance's rank predict the realised score, where the input's does not?

Two measurements on 2026-10-03 removed the input's participation ratio as the thing that controls
realisability. The adjoint fix in xspec.solve_factor (574fd43) moved the realised score from +0.1661
to +0.2621 while leaving the input participation ratio at 8.5 -> 8.3, and --prank-mu drove that ratio
to its ceiling of 100 of 100 and made the realised score FALL. What did move with the realised score
both times was on the output side: the field's effective rank, fc_score.effective_rank of the
realised frames.

That is measured FROM a realisation, so it cannot be optimised against. This asks whether its
expectation can be: the participation ratio of the eigenvalues of

    C_raw = sum_f 2 w_f Re(H_f S_f H_f^H)

which is PSD, is a function of the solve alone, costs one einsum and no simulation, and is the same
functional form effective_rank applies to the realised field. If it tracks the realised score across
the maxfun ladder where the input's ratio does not, it is a candidate penalty on the side the
measurements point at.

NOTE WHICH C. The objective double-centres C and zeroes its diagonal, which makes it indefinite, so
the participation ratio is taken on the UNCENTRED covariance - the one the field actually has. Both
are reported, because the centred one is what the fit sees.

IT DOES TRACK, over 10 runs spanning two different mechanisms - the maxfun ladder and the
--prank-mu sweep, which moves the input's ratio in the opposite direction:

    tag                      fit  input PR  output PR  |C_c| PR  realised   field rank
    fixgrad_mf15         +0.6094      89.1       12.4      12.0   +0.5569      13.7
    fixgrad_mf30         +0.6655      81.3       13.1      12.9   +0.6064      14.6
    fixgrad_mf45         +0.6972      71.0       13.6      14.0   +0.6161      13.8
    fixgrad_mf60         +0.7156      58.3       13.7      14.5   +0.6231      13.7
    fixgrad_mf150        +0.7536      23.2       13.4      17.1   +0.5624       8.6
    fixgrad_mf400        +0.7834      11.4        9.9      22.7   +0.4042       5.8
    flat100_fac20_fix    +0.8165       8.3       11.9      43.4   +0.2621       2.9
    flat100_mu0.1        +0.8184      99.8        9.6      52.6   +0.1989       3.4
    flat100_mu0.5        +0.8184     100.0        9.4      53.4   +0.2015       3.4
    flat100_mu2.0        +0.8184     100.0        9.2      53.5   +0.1981       3.3

  spearman of the realised score against, over those 10 runs:
    input PR         -0.4788        output PR        +0.9636
    |C_centred| PR   -0.8182        field rank       +0.8545        fit  -0.8182

The input's ratio has the WRONG SIGN. The output's beats even field rank, which can only be
measured after realising. Read it as an ordering and not as a scale relation: output PR spans
9.2-13.7 where the realised score spans 0.20-0.62, and mf15 at 12.4 against the converged run at
11.9 is nearly a tie on the predictor with 0.29 between the outcomes.

The |C_centred| column is the other half of the story and runs the other way, 12.0 -> 53.5. The fit
is taken on the double-centred, zero-diagonal C, so the solve can keep adding structure to the
off-diagonal part it is scored on - that column rising - while the covariance the FIELD actually has
stays dominated by a dozen modes or fewer. The realised FC is estimated from the field, so its
effective sample size is set by the UNCENTRED spectrum, and the two diverge by a factor of four as
the solve converges.

  python fit/output_rank.py fixgrad_mf15 fixgrad_mf30 ... flat100_fac20_fix
"""
import _path  # noqa: F401
import os, argparse
import numpy as np

from paths import RESULTS
import xspec
from interp_gap import build_H


def pr(ev):
    ev = np.clip(np.asarray(ev, float), 0.0, None)
    return float(ev.sum() ** 2 / max((ev ** 2).sum(), 1e-300))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('tags', nargs='+')
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()

    from scipy.stats import spearmanr
    rows = []
    for tag in a.tags:
        f = os.path.join(RESULTS, f'xspec_{tag}.npz')
        if not os.path.exists(f):
            print(f'  {tag}: missing'); continue
        z = np.load(f, allow_pickle=True)
        S, w = xspec.load_S(z), np.asarray(z['H_w'], float)
        Hall, _, _, c, t, g = build_H(tag, z, False, workers=a.workers)
        H = np.ascontiguousarray(Hall[g['idx']]); del Hall
        sub = g['sub']
        C = np.einsum('f,fij->ij', 2.0 * w,
                      np.real((H @ S) @ np.conj(np.transpose(H, (0, 2, 1)))), optimize=True)
        C = 0.5 * (C + C.T)
        ev_raw = np.linalg.eigvalsh(C)
        Cc = C - C.mean(0, keepdims=True) - C.mean(1, keepdims=True) + C.mean()
        raw = np.asarray(t.target_fc()[np.ix_(sub, sub)], np.float64)
        raw = raw - raw.mean(0, keepdims=True) - raw.mean(1, keepdims=True) + raw.mean()
        iu = np.triu_indices(len(sub), 1)
        A = np.einsum('f,fij->ij', w, S, optimize=True); A = 0.5 * (A + A.conj().T)
        rows.append(dict(
            tag=tag,
            fit=float(spearmanr(Cc[iu], raw[iu]).statistic),
            in_pr=pr(np.linalg.eigvalsh(A).real),
            out_pr=pr(ev_raw),
            out_pr_c=pr(np.abs(np.linalg.eigvalsh(Cc))),
            sim=float(z['sim']), sd=float(z['sim_sd']),
            field=float(z['field_rank'])))
        del H, C

    print(f"\n  {'tag':<20s} {'fit':>7s} {'input PR':>9s} {'output PR':>10s} "
          f"{'|C_c| PR':>9s} {'realised':>9s} {'+-':>7s} {'field rank':>11s}")
    for r in rows:
        print(f"  {r['tag']:<20s} {r['fit']:+7.4f} {r['in_pr']:9.1f} {r['out_pr']:10.1f} "
              f"{r['out_pr_c']:9.1f} {r['sim']:+9.4f} {r['sd']:7.4f} {r['field']:11.1f}")
    if len(rows) > 2:
        sim = np.array([r['sim'] for r in rows])
        print(f"\n  spearman of realised score against, over {len(rows)} runs:")
        for k, nm in (('in_pr', 'input PR'), ('out_pr', 'output PR'),
                      ('out_pr_c', '|C_centred| PR'), ('field', 'field rank'),
                      ('fit', 'fit')):
            v = np.array([r[k] for r in rows])
            print(f"    {nm:<16s} {spearmanr(v, sim).statistic:+.4f}")


if __name__ == '__main__':
    main()
