"""How much does the solve's FREQUENCY GRID cost, measured rather than assumed?

The solve carries S on a coarse grid - 135 bins of 2,049 - and integrates

    C = sum_{f in idx} w_f 2Re(H_f S_f H_f^H),    w_f = gradient(idx)

so each sample stands for the block around it. `xspec.realise` does something else: it
LINEARLY INTERPOLATES S across every bin of the rfft grid before drawing. The drive that
is actually simulated therefore has a different spectrum from the one the solve scored,
and the difference is a candidate explanation for the solve-to-realised gap.

This measures it. The same solved S is pushed through three quadratures on the SAME H,
evaluated at every bin:

  C_quad    sum over idx with the weights w       - what the objective maximised
  C_interp  sum over all bins, tent interpolant   - the infinite-data limit of what is drawn
  C_block   sum over all bins, piecewise constant - what the weights w already assume

C_interp is the expectation of the realised covariance, so score(C_quad) - score(C_interp)
is the part of the gap that is NOT sampling noise and NOT the solve-to-score vertex
extrapolation. If the two agree, the grid is exonerated and the gap is elsewhere.

Each objective is scored the way ITS OWN solver scores it, which is not optional: for the
linear fit `xspec.solve`'s `model` double-centres C and zeros the diagonal before
correlating, and a reconstruction that skips that does not reproduce the number the run
reported. The envelope's `envelope.solve` instead takes a plain Pearson on the upper
triangle.

WHAT IT MEASURED. The grid is not the gap. For the linear fit the tent interpolant and
the quadrature agree at r = +1.0000 and the objective moves +0.7186 -> +0.7180, so the
coarse grid costs 0.0006. For the envelope it is real but small: +0.7690 -> +0.7470, a
cost of 0.022 at r = +0.9851, where the realised score was +0.198.

What IS the gap is estimator variance in the realised fourth moment, and `--realise`
measures it rather than asserting it. Scoring the closed form, the simulation on the solve
vertices, and the simulation on all 9,310 separately shows the envelope does BETTER off
the solve set than on it, so extrapolation is not the loss; and the split half shows
r(h1,h2) matching r(h1,ref)*r(h2,ref) to 0.01 in both paths, so the shortfall is additive
independent noise and not a pipeline that computes the wrong quantity.

  corr(empirical, closed form) against realisation length, one draw each:
                288s     577s    1154s    2308s      variance floor T0
    linear        -      0.912   0.961    0.978       ~106 s
    envelope    0.128    0.171   0.288    0.392       ~12,700 s

Fitting r^2 = T/(T+T0): the envelope needs about 120x the data the linear observable needs
for the same relative accuracy, because it is a fourth moment of a signal whose passband
leaves ~40 independent samples in 577 s. The realised score then tracks r almost exactly:
r x (closed form score) predicts +0.128 against +0.123 measured at 577 s, +0.293 against
+0.286 at 2,308 s, and for the linear +0.702 against +0.7057. So +0.198 at 577 s is a
statement about the estimator, not about the model, and a realised envelope score near
+0.70 needs r ~ 0.94, which is ~96,000 s of simulated time - about 42 draws at 2,308 s.

  python fit/interp_gap.py grclip100_nolag          # linear, results/xspec_<tag>.npz
  python fit/interp_gap.py env_grclip400 --envelope # envelope, results/envfit_<tag>.npz
  python fit/interp_gap.py env_grclip400 --envelope --realise 2308 --draws 1
"""
import _path  # noqa: F401
import os, sys, time, argparse
import numpy as np
from scipy.stats import spearmanr

from mesh_cache import load_cortex
from paths import RESULTS
import fc_score, xspec, bo_step, timescale, units, bandpass
import lagged as _lg
import envelope as env


def double_centre(C, zero_diag=True):
    """What xspec.solve's `model` scores. The diagonal is not part of the objective."""
    C = np.asarray(C, float).copy()
    C = C - C.mean(0, keepdims=True) - C.mean(1, keepdims=True) + C.mean()
    if zero_diag:
        np.fill_diagonal(C, 0.0)
    return C


def spectra(S, idx, nb):
    """-> (S_interp, S_block) on every bin of the rfft grid.

    S_interp is `xspec.realise`'s tent interpolant, evaluated at the pad grid's own
    frequencies. S_block assigns each sample to the bins between the midpoints of its
    neighbours, which is the partition whose widths ARE gradient(idx)."""
    K = S.shape[1]
    f_src, f_dst = np.asarray(idx, float) / nb, np.arange(nb) / float(nb)
    Si = np.empty((nb, K, K), np.complex64)
    for a in range(K):
        for b in range(K):
            Si[:, a, b] = (np.interp(f_dst, f_src, S[:, a, b].real, left=0., right=0.)
                           + 1j * np.interp(f_dst, f_src, S[:, a, b].imag,
                                            left=0., right=0.))
    edge = np.concatenate([[idx[0]],
                           np.round(0.5 * (idx[1:] + idx[:-1])).astype(int), [nb]])
    own = np.full(nb, -1)
    for f in range(len(idx)):
        own[edge[f]:edge[f + 1]] = f
    Sb = np.where(own[:, None, None] >= 0,
                  S[np.clip(own, 0, None)], 0).astype(np.complex64)
    return Si, Sb


def build_H(tag, z, envelope, workers=8):
    """-> (H at every rfft bin, respf, cortex, target, grid facts).

    `envelope` decides whether the observable's filters are folded into H (linear) or left
    out of it entirely (envelope, where they belong in rho and nowhere else)."""
    idx, x, sub = z['idx'], z['x'], z['sub']
    pad, save, frame_s = int(z['pad']), int(z['save']), float(z['frame_s'])
    lo, hi = (float(v) for v in z['band'])
    P, mc = np.asarray(z['profiles'], np.float32), str(z['map_clip'])
    K, nV = z['S'].shape[1], len(sub)

    c = load_cortex('fsaverage5', verbose=False)
    t = fc_score.default_target(c, verbose=False)
    p, _, _ = bo_step.unpack(x, c)
    p['map_clip'] = mc
    if 'impulse_frames' in z.files:
        imp = int(z['impulse_frames'])
    else:                       # envelope_fit does not store it; rebuild its rule
        cl = timescale.plan(4, decay_s=25.0, spread_mm_s=1.5, verbose=False)
        decay_fr = 1.0 / (cl['damp'] * save)
        imp = int(np.ceil(max(7.0 * decay_fr, 1.0 / (lo * frame_s)) / 64) * 64)
    print(f'  {tag}: {K} channels, {nV} vertices, {len(idx)} solved bins, '
          f'pad {pad} ({pad*frame_s:.0f}s), impulse {imp} frames', flush=True)

    t0 = time.time()
    resp = xspec.impulse_responses(c, list(range(K)), p, imp * save, save, profiles=P,
                                  verbose=False, workers=workers,
                                  keep=np.asarray(t.cols)[sub])
    R = np.pad(resp, ((0, 0), (0, max(0, pad - resp.shape[1])), (0, 0))); del resp
    F = np.fft.rfft(R, axis=1).astype(np.complex64); del R
    nb = F.shape[1]
    kern = units.smoothing_kernel(timescale.bold_fwhm_frames(frame_s, verbose=False),
                                  verbose=False)
    respf = (units.kernel_response(kern, nb, pad)
             * bandpass.response(np.arange(nb) / (pad * frame_s), frame_s, lo, hi))
    if not envelope:
        F *= respf[None, :, None].astype(np.complex64)
    H = np.ascontiguousarray(F.transpose(1, 2, 0)); del F
    print(f'  H at all {nb} bins{"" if envelope else " (kernel x passband folded in)"}, '
          f'{H.nbytes/2**30:.2f} GiB  [{time.time()-t0:.0f}s]', flush=True)
    return H, respf, kern, c, t, dict(idx=idx, sub=sub, pad=pad, nb=nb, frame_s=frame_s,
                                      lo=lo, hi=hi, nV=nV, K=K, p=p, P=P, save=save)


def cov_linear(H, S, bins, weight=None, block=48):
    """sum_j weight_j 2Re(H_{bins[j]} S_j H^H). S is indexed by POSITION in `bins`, which
    is what lets the coarse quadrature and the all-bins sums share one code path."""
    nV = H.shape[1]
    bins = np.asarray(bins)
    S = S if len(S) == len(bins) else S[bins]
    C = np.zeros((nV, nV))
    for a in range(0, len(bins), block):
        k, sl = bins[a:a + block], slice(a, a + block)
        Hk = H[k].astype(np.complex128)
        M = (Hk @ S[sl].astype(np.complex128)) @ np.conj(np.transpose(Hk, (0, 2, 1)))
        ww = np.ones(len(k)) if weight is None else np.asarray(weight, float)[sl]
        C += 2.0 * np.einsum('t,tij->ij', ww, M.real, optimize=True)
    return C


def cov_envelope(H, S, bins, ph, rho, weight=None, lchunk=64, fblock=256):
    """E = 2 sum_tau rho(tau) C(tau)^2, with C(tau) built from `bins` only.

    envelope.env_cov holds M for every solved frequency at once, which is 5.2 GiB at the
    full 2,049 bins. This chunks over frequency as well as lag and keeps M in single
    precision: the quantity wanted is a correlation to four figures, not E itself."""
    nV = H.shape[1]
    bins = np.asarray(bins)
    S = S if len(S) == len(bins) else S[bins]
    w = np.ones(len(bins)) if weight is None else np.asarray(weight, float)
    Mf = np.empty((len(bins), nV * nV), np.complex64)
    for a in range(0, len(bins), fblock):
        k, sl = bins[a:a + fblock], slice(a, a + fblock)
        Hk = H[k].astype(np.complex64)
        Mf[sl] = (((Hk @ S[sl].astype(np.complex64))
                   @ np.conj(np.transpose(Hk, (0, 2, 1)))).reshape(len(k), -1))
    phw = (np.asarray(ph, np.complex64) * w[None, :].astype(np.float32))
    E = np.zeros((nV, nV))
    for a in range(0, len(rho), lchunk):
        b = min(a + lchunk, len(rho))
        C = 2.0 * np.real(phw[a:b] @ Mf).reshape(b - a, nV, nV).astype(np.float64)
        E += 2.0 * np.einsum('t,tij->ij', rho[a:b], C ** 2, optimize=True)
    return E


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('tag')
    ap.add_argument('--envelope', action='store_true')
    ap.add_argument('--lag-tol', type=float, default=1e-3, dest='lag_tol')
    ap.add_argument('--realise', type=float, default=0.0,
                    help='also SIMULATE the solved drive for this many seconds and report '
                         'the empirical covariance against the closed form. This is what '
                         'separates the three remaining candidates: a pipeline that does '
                         'not match the closed form, the finite realisation, and the '
                         'extrapolation from the solve vertices to all 9,310')
    ap.add_argument('--draws', type=int, default=1)
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()

    f = os.path.join(RESULTS, ('envfit_' if a.envelope else 'xspec_') + a.tag + '.npz')
    z = np.load(f, allow_pickle=True)
    S = z['S']
    w = np.asarray(z['w' if a.envelope else 'H_w'], float)
    H, respf, kern, c, t, g = build_H(a.tag, z, a.envelope, workers=a.workers)
    idx, nb, nV, sub = g['idx'], g['nb'], g['nV'], g['sub']
    Si, Sb = spectra(S, idx, nb)
    allb = np.arange(1, nb)

    t1 = time.time()
    if a.envelope:
        lags, rho = env.filter_lags(respf, g['pad'], tol=a.lag_tol)
        print(f'  {len(lags)} lags carry the filter autocorrelation (tol {a.lag_tol:g})',
              flush=True)
        Cq = cov_envelope(H, S, idx, _lg.phases(idx, g['pad'], lags), rho, weight=w)
        print(f'    quadrature done [{time.time()-t1:.0f}s]', flush=True)
        ph_all = _lg.phases(allb, g['pad'], lags)
        Ci = cov_envelope(H, Si, allb, ph_all, rho)
        Cb = cov_envelope(H, Sb, allb, ph_all, rho)
        centre, label = False, 'E'
    else:
        Cq = cov_linear(H, S, idx, weight=w)
        Ci = cov_linear(H, Si, allb)
        Cb = cov_linear(H, Sb, allb)
        centre, label = True, 'C'
    print(f'  three covariances  [{time.time()-t1:.0f}s]', flush=True)

    raw = np.asarray(t.target_fc()[np.ix_(sub, sub)], np.float64)
    raw = raw - raw.mean(0, keepdims=True) - raw.mean(1, keepdims=True) + raw.mean()
    iu = np.triu_indices(nV, 1)
    Tgt = xspec.normal_scores(raw, iu)
    if centre:
        Cq, Ci, Cb = (double_centre(M) for M in (Cq, Ci, Cb))

    def obj(M):
        """The solver's own objective: Pearson against the normal-scored target."""
        m = M[iu] - M[iu].mean()
        return float(m @ (Tgt[iu] - Tgt[iu].mean())
                     / (np.linalg.norm(m) * np.linalg.norm(Tgt[iu] - Tgt[iu].mean())))

    print(f'\n  {"quadrature":<34s} {"objective":>10s} {"vs raw (r)":>11s} '
          f'{"vs raw (rho)":>13s}')
    rows = ((f'{label}_quad    what we solve', Cq),
            (f'{label}_interp  what we draw', Ci),
            (f'{label}_block   block spectrum', Cb))
    for nm, M in rows:
        print(f'  {nm:<34s} {obj(M):>+10.4f} {np.corrcoef(M[iu], raw[iu])[0,1]:>+11.4f} '
              f'{spearmanr(M[iu], raw[iu]).statistic:>+13.4f}')
    print(f'\n  corr({label}_quad, {label}_interp) = '
          f'{np.corrcoef(Cq[iu], Ci[iu])[0,1]:+.4f}   '
          f'spearman {spearmanr(Cq[iu], Ci[iu]).statistic:+.4f}')
    print(f'  corr({label}_quad, {label}_block)  = '
          f'{np.corrcoef(Cq[iu], Cb[iu])[0,1]:+.4f}   '
          f'spearman {spearmanr(Cq[iu], Cb[iu]).statistic:+.4f}')

    f_hz = np.arange(nb) / (g['pad'] * g['frame_s'])
    inb = (f_hz >= g['lo']) & (f_hz <= g['hi'])
    inq = inb[idx]
    trq, tri, trb = (np.einsum('fkk->f', M).real for M in (S, Si, Sb))
    print(f'\n  drive power in {g["lo"]}-{g["hi"]} Hz: quadrature '
          f'{np.sum((w*trq)[inq])/np.sum(w*trq):.4f}, interp '
          f'{tri[inb].sum()/tri.sum():.4f}, block {trb[inb].sum()/trb.sum():.4f}')
    if a.realise > 0:
        realise(a, c, t, g, S, kern, Ci, Tgt, iu, raw, centre, label)


def realise(a, c, t, g, S, kern, Cref, Tgt, iu, raw, centre, label):
    """Simulate the solved drive and compare the EMPIRICAL covariance three ways.

    `Cref` is the closed form over every bin - the infinite-data limit of this very
    simulation - so corr(empirical, Cref) asks whether the pipeline computes the quantity
    the solve modelled, with no reference to the target at all. Scoring the same empirical
    matrix on the SOLVE vertices and then on all 9,310 separates the finite realisation
    from the extrapolation off the solve set, which no closed form can reach at the
    envelope's cost."""
    import fluid as fl
    from xspec import ProfileDrive
    save, P, p = g['save'], g['P'], g['p']
    nframes = timescale.frames_for(a.realise, g['frame_s'])
    cols = np.asarray(t.cols)
    print(f'\n  simulating {a.draws} draw(s) of {nframes} frames ({a.realise:.0f}s)',
          flush=True)
    for d in range(a.draws):
        t2 = time.time()
        Af = xspec.realise(S, g['idx'], nframes, ref_frames=g['pad'], seed=1000 + d)
        Aser = np.repeat(Af, save, axis=0)[:nframes * save] / save
        fr, _ = fl.run(c, ProfileDrive(c, P, Aser, 2e-4), p, nframes * save, save)
        if a.envelope:
            U = env.observable(fr[:, cols], kern, g['frame_s'], (g['lo'], g['hi']),
                               burn=t.burn)
        else:
            import bandpass as bp
            U = bp.apply(units.smooth_frames(np.asarray(fr[:, cols], np.float64), kern),
                         g['frame_s'], g['lo'], g['hi'])[t.burn:]
        del fr
        Z = U - U.mean(0, keepdims=True)
        Z = (Z / np.maximum(Z.std(0, keepdims=True), 1e-300)).T
        all9 = float(t._prep(t.model_edges(Z=Z)[0]) @ t.y)
        Us = np.asarray(U[:, g['sub']], np.float64)

        def emp(X):
            E = np.cov(X, rowvar=False)
            return double_centre(E) if centre else E

        Ee = emp(Us)
        e = Ee[iu] - Ee[iu].mean()
        tg = Tgt[iu] - Tgt[iu].mean()
        print(f'    draw {d}: corr(empirical, {label}_interp) '
              f'{np.corrcoef(Ee[iu], Cref[iu])[0,1]:+.4f}  |  objective on the '
              f'{len(g["sub"])} solve vertices {float(e @ tg / (np.linalg.norm(e) * np.linalg.norm(tg))):+.4f}'
              f', spearman vs raw {spearmanr(Ee[iu], raw[iu]).statistic:+.4f}'
              f'  |  sim on all {Z.shape[0]} {all9:+.4f}   [{time.time()-t2:.0f}s]',
              flush=True)
        # SPLIT HALF, free: the two halves share one truth, so if the shortfall against
        # the closed form is sampling then r(h1,h2) ~ r(h1,ref) * r(h2,ref). A r(h1,h2)
        # well above that product is structure the halves share and the closed form does
        # not have, which is a mismatch rather than noise.
        h = len(Us) // 2
        E1, E2 = emp(Us[:h]), emp(Us[h:2 * h])
        r1 = np.corrcoef(E1[iu], Cref[iu])[0, 1]
        r2 = np.corrcoef(E2[iu], Cref[iu])[0, 1]
        r12 = np.corrcoef(E1[iu], E2[iu])[0, 1]
        print(f'      split half ({h} frames each): r(h1,ref) {r1:+.4f}  r(h2,ref) '
              f'{r2:+.4f}  r(h1,h2) {r12:+.4f}  vs r1*r2 {r1*r2:+.4f}'
              f'  -> {"sampling" if abs(r12 - r1*r2) < 0.1 else "SHARED STRUCTURE the closed form lacks"}',
              flush=True)


if __name__ == '__main__':
    main()
