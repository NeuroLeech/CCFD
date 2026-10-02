"""With one channel per driven vertex, the solved input cross-spectrum is a MAP.

Every earlier solve carried S over 100 tiles, so its spatial structure was a 100x100 matrix
of pieces and could only be read through the tiling. With K = 1880 one-hot channels, S_f is
the cross-spectrum between driven VERTICES, so the band-integrated input covariance
sum_f w_f S_f is an object on the cortical surface and its autocorrelation against geodesic
distance is directly comparable to the field's (3.8 mm, analysis/spatial_scale.py) and to the
target FC's own structure on the same vertices.

Reported in correlation form, since the diagonal of S carries each vertex's input power and
the off-diagonal is what says how the drive is organised in space.

WHAT IT FOUND on flat1880_r20 - 1,880 one-hot channels, rank 20 permitted, 700 vertices sampled:

                              effective rank (of 700)   correlation length
      in band 0.01-0.08 Hz              4.1                   5.7 mm
      above the band                    5.8                   7.0 mm
      all bins                          7.1                   6.8 mm

Given a complete basis on the driven territory the solve did NOT build detailed spatial
structure. It chose a smooth input - 5.7 to 7.0 mm, smoother than the field's own 3.8 mm from
analysis/spatial_scale.py - carrying about four effective modes in the band that matters, with
twenty available. The expressive power went into tuning a few smooth modes precisely to the 400
solve vertices rather than into spatial detail, which is the same story the fit tells: solve-space
spearman +0.8468 against the 100-tile baseline's +0.6530, realised +0.3047 against its +0.6104.

RANK IS NOT THE BINDING FREEDOM. The same basis at three ranks:

    rank 3     solve +0.8494    realised +0.2804 +- 0.0430
    rank 8     solve +0.8475    realised +0.2919 +- 0.1005
    rank 20    solve +0.8468    realised +0.3047 +- 0.0234

Flat in both columns, so three modes overfit exactly as well as twenty. With one-hot channels each
mode can be an ARBITRARY function on 1,880 vertices - 5,640 spatial degrees of freedom at rank 3 -
where the 100-tile basis gives each mode 100, every tile smooth over ~19 vertices. The tiling was
acting as a spatial regulariser, WHICH IS WRONG - see below.

THE OVERFITTING IS THE SOLVER, not the basis. The same 100 tapered tiles, only the solver differing:

    100 tiles, factor rank 20, maxfun 1500    solve +0.8108    realised +0.1661
    100 tiles, gradient, 400 iterations       solve +0.6530    realised +0.6104

which is what xspec.solve's docstring already says - the realised score peaks early and falls away
as the objective converges, so --iters is a hidden regularisation parameter - and L-BFGS converging
properly removes it. Every comparison between a factored large-K run and the gradient-solver
baseline was confounded by that. At matched solver the channel count does not hurt:

    basis                channels   solve-space    realised
    100 tapered tiles       100       +0.8108      +0.1661 +- 0.0465
    90 smooth modes          90       +0.7906      +0.3108 +- 0.0634
    263 smooth modes        263       +0.8261      +0.2552 +- 0.0290
    618 smooth modes        618       +0.8352      +0.3098 +- 0.0044
    1,880 one-hot          1880       +0.8468      +0.3047 +- 0.0234

Solve-space rises monotonically with channels while realised stays flat at 0.25-0.31, the disjoint
tiles the lone outlier below. All sit far under +0.6104, so regularisation dominates the basis and
the expressive-power question cannot be settled with a solver that converges.

MAXFUN IS THE KNOB, AND AT THE RIGHT BUDGET THE CHANNEL COUNT DOES NOT MATTER. The factor solver's
evaluation budget is its analogue of --iters, and 60 evaluations recovers everything convergence
destroyed:

    basis            maxfun   solve-space    realised            gap    rank
    100 tiles            60     +0.7162    +0.6196 +- 0.0262    0.061   13.7
    100 tiles          1500     +0.8108    +0.1661 +- 0.0465    0.282    4.6
    1,880 one-hot        60     +0.7327    +0.6074 +- 0.0127    0.053   15.8
    1,880 one-hot      1500     +0.8468    +0.3047 +- 0.0234    0.221    3.8

THE FULL LADDER. Realised score is an inverted U in optimisation effort, for both bases:

    maxfun    100 tiles  solve / realised     1,880 one-hot  solve / realised
        15       +0.6094 / +0.5549 +- 0.0166     +0.5954 / +0.5333 +- 0.0031
        30       +0.6653 / +0.6049 +- 0.0177     +0.6701 / +0.5888 +- 0.0100
        45       +0.6976 / +0.6126 +- 0.0176          -   / +0.6053 +- 0.0068
        60       +0.7162 / +0.6196 +- 0.0262     +0.7327 / +0.6074 +- 0.0127
       150       +0.7532 / +0.5718 +- 0.0501     +0.7866 / +0.5934 +- 0.0020
       400       +0.7830 / +0.3924 +- 0.0104     +0.8289 / +0.4875 +- 0.0213
      1500       +0.8108 / +0.1661 +- 0.0465     +0.8468 / +0.3047 +- 0.0234

      incumbent gradient solver, --iters 400:   +0.6530 / +0.6104

Three readings. The peak is near maxfun 45-60 and it is sharp - solve-space keeps climbing the whole
way while realised turns over - so the stopping point is the dominant regularisation and the
incumbent --iters 400 was close to but not at the optimum. More channels do NOT help at the peak:
+0.6074 against +0.6196, inside both error bars, from a basis complete on the driven territory. And
past the peak more channels degrade more SLOWLY, +0.4875 against +0.3924 at 400 and +0.3047 against
+0.1661 at 1500, so the channel count buys robustness to over-optimisation rather than a better
optimum.

The best realised figure anywhere in this sweep is +0.6196 against the incumbent's +0.6104, a gain of
0.009 inside a +-0.026 error bar.

WHY THE REALISATION IS LIMITED, which is the part that explains all of the above. Decomposing the gap
with fit/interp_gap.py at both ends of the ladder:

                        closed form   one draw,      one draw,     corr(draw,     T0
                        400 verts     400 verts      9310 verts    closed form)
    maxfun   60          +0.7302       +0.6506        +0.6229        +0.8978      139 s
    maxfun 1500          +0.8150       +0.1497        +0.1024        +0.1602   21,909 s

It is NOT spatial overfitting. At maxfun 1500 extrapolation from 400 vertices to 9,310 costs 0.047
while the expectation-to-one-draw step costs 0.665, and the draw's empirical FC correlates with its own
closed-form prediction at r = 0.16. The solve finds an S whose EXPECTED FC matches the target at
+0.8150 and which cannot be realised in a scan: the variance floor goes from 139 s to 21,909 s, a
factor of 157, so that expectation needs ~22,000 s of data to become visible.

The mechanism is concentration. Input participation ratio against optimisation effort, out of 100
channels:

    maxfun      15     30     45     60    150    400   1500
    input PR  89.1   80.8   68.9   54.2   20.4   10.0    8.5
    in-band   20.3   15.0   10.3    8.7    7.2    6.5    7.9
    realised +.5549 +.6049 +.6126 +.6196 +.5718 +.3924 +.1661

Optimising concentrates the drive into fewer modes, a low-rank drive has a high variance floor, and the
realised score peaks where the input still spreads over ~54 of 100 modes. That is why more channels
cannot help: the failure mode is concentration and extra channels are extra freedom to concentrate, so
1,880 one-hot channels reach ~8 effective modes sooner rather than buying anything.

So maxfun was standing in for a realisability term the objective does not have. xspec.prank_reg is
exactly that quantity - "Participation-ratio rank of the input. sign=+1 spreads power over modes" - and
family_member already moves along a regulariser while holding the fit within eps of the argmax, which
would constrain the cause rather than tuning a magic number.

The target comparison in this script is not apples to apples and should not be read as one: the
empirical FC arrives double-centred, so putting it in correlation form after setting the diagonal
to 1 removes much of its local structure and its numbers come out far lower than the input's for
reasons that are about the centring rather than about the drive.

  python fit/input_space.py flat1880_r20
"""
import _path  # noqa: F401
import os, argparse
import numpy as np

from mesh_cache import load_cortex
from paths import RESULTS
import units, fc_score

EDGES = np.array([0, 2, 4, 6, 8, 10, 15, 20, 25, 30, 40, 60, 90, 140, 250])


def corr_by_distance(M, D, iu, label, edges=EDGES):
    """Off-diagonal of M in correlation form, binned by geodesic distance."""
    d = np.sqrt(np.clip(np.diag(M).real, 1e-300, None))
    R = np.real(M) / np.outer(d, d)
    r, dd = R[iu], D[iu]
    half, prev = None, None
    print(f'    {label}')
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (dd >= lo) & (dd < hi)
        if m.sum() < 50:
            continue
        v = float(r[m].mean())
        print(f'      {lo:5.0f} - {hi:<4.0f} {int(m.sum()):>8d} {v:>+9.3f}')
        if half is None and prev is not None and prev[1] >= 0.5 > v:
            x0, y0 = prev
            x1 = 0.5 * (lo + hi)
            half = x0 + (y0 - 0.5) * (x1 - x0) / (y0 - v)
        prev = (0.5 * (lo + hi), v)
    if half:
        print(f'      falls to r = 0.5 at {half:.1f} mm')
    return half


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('tag')
    ap.add_argument('--nvert', type=int, default=900)
    a = ap.parse_args()

    z = np.load(os.path.join(RESULTS, f'xspec_{a.tag}.npz'), allow_pickle=True)
    S, idx, w = z['S'], np.asarray(z['idx'], np.int64), np.asarray(z['H_w'], float)
    pad, fs = int(z['pad']), float(z['frame_s'])
    lo_hz, hi_hz = (float(v) for v in z['band'])
    lab = np.asarray(z['labels'], np.int64)
    dv = np.flatnonzero(lab >= 0)
    order = np.argsort(lab[dv])                    # channel k is vertex dv[order][k]
    verts = dv[order]
    K = S.shape[1]
    if len(verts) != K:
        raise SystemExit(f'  {K} channels but {len(verts)} driven vertices - not one-hot')
    f_hz = idx / (pad * fs)
    inb = (f_hz >= lo_hz) & (f_hz <= hi_hz)
    print(f'\n  {a.tag}: {K} one-hot channels, {int(inb.sum())} of {len(idx)} bins in band')

    rng = np.random.default_rng(0)
    sel = np.sort(rng.choice(K, min(a.nvert, K), replace=False))
    c = load_cortex('fsaverage5', verbose=False)
    D = units.vertex_geodesic(c, verts[sel])[:, verts[sel]]
    iu = np.triu_indices(len(sel), 1)

    def band_sum(mask):
        A = np.zeros((len(sel), len(sel)), complex)
        for f in np.flatnonzero(mask):
            A += w[f] * S[f][np.ix_(sel, sel)]
        return A

    print(f'\n  INPUT covariance, correlation form, by geodesic distance between driven '
          f'vertices')
    for nm, mask in (('in band 0.01-0.08 Hz', inb), ('above the band', ~inb),
                     ('all bins', np.ones(len(idx), bool))):
        M = band_sum(mask)
        ev = np.clip(np.linalg.eigvalsh(0.5 * (M + M.conj().T)).real, 0, None)
        pr = (ev.sum() ** 2) / max((ev ** 2).sum(), 1e-300)
        corr_by_distance(M, D, iu, f'{nm}  (effective rank {pr:.1f} of {len(sel)})')

    t = fc_score.default_target(c, verbose=False)
    pos = {int(v): i for i, v in enumerate(t.vertices)}
    keep = np.array([i for i, v in enumerate(verts[sel]) if int(v) in pos])
    if len(keep) > 100:
        ti = np.array([pos[int(v)] for v in verts[sel][keep]])
        T = np.asarray(t.target_fc()[np.ix_(ti, ti)], np.float64)
        np.fill_diagonal(T, 1.0)
        iu2 = np.triu_indices(len(keep), 1)
        print(f'\n  for comparison, the TARGET FC on the same {len(keep)} driven vertices')
        corr_by_distance(T.astype(complex), D[np.ix_(keep, keep)], iu2, 'empirical FC')


if __name__ == '__main__':
    main()
