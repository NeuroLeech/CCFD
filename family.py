"""Which dynamics are consistent with this FC, rather than which one is optimal?

FC is a second moment. The medium is LTI, so the field covariance depends on the input
only through its cross-spectral density S(f) - and the solve returns ONE argmax of a
scale-invariant ratio over a convex cone. What the data actually determine is the level
set {S : rho(S) >= rho* - eps}, which has real dimension. Everything that differs between
two members of it is something functional connectivity cannot decide.

Three axes, each answering a different version of "what else would have worked".

INPUT STRUCTURE, at matched fit. xspec.family_member walks away from the argmax along a
named direction while holding the fit within eps: maximum entropy against minimum rank,
locally-coordinated coalitions against free ones, independent channels against coupled
ones. Reported as realised scores, so a member that turns out to fit BETTER once
simulated is visible - which is a live possibility, since field rank has tracked the
realised score everywhere it has been measured.

INPUT TIMESCALE. --bands restricts S(f) to a frequency band and re-solves. This is convex,
so nothing about the method changes, and it is the direct handle on what input timescales
can produce this FC - the question the units calibration leaves open, the model frame
being about 1/100 of a TR.

REALISATION LAW. Same S(f), different draw: gaussian, unit-modulus phase, heavy-tailed.
All three have E[eta eta^H] = I, so all three reproduce the static FC exactly and differ
only above second order. The invariance is CHECKED here rather than assumed, because the
score is Spearman on rank-transformed timecourses and a non-Gaussian margin moves ranks -
the size of that effect is how much "FC is only a second moment" is not quite true for
this pipeline.

  python family.py --members --frames 4480
  python family.py --laws --frames 4480
  python family.py --bands 4
"""
import os, time, argparse
import numpy as np

from mesh_cache import load_cortex
from paths import RESULTS
import fc_score, xspec, bo_step, subparcels, best_fit


def setup(a):
    c = load_cortex("fsaverage5", verbose=False)
    t = fc_score.default_target(c, verbose=False)
    labels, tags = subparcels.split_parcels(c, subparcels.SENSORY, a.split, verbose=False)
    P = subparcels.taper_profiles(c, labels, len(tags))
    x = best_fit.BEST_X.copy()
    x[0] = np.log10(a.damp)
    p, save, _ = bo_step.unpack(x, c)
    sub = xspec.medoid_subset(t, a.nvert)
    val = xspec.validation_subset(t, sub, a.nvert)
    n = len(sub)
    iu = np.triu_indices(n, 1)
    raw = np.asarray(t.target_fc()[np.ix_(sub, sub)], np.float64)
    raw = raw - raw.mean(0, keepdims=True) - raw.mean(1, keepdims=True) + raw.mean()
    Tgt = xspec.normal_scores(raw, iu)
    Tgt[np.eye(n, dtype=bool)] = 0.0

    resp = xspec.impulse_responses(c, list(range(len(P))), p, a.window * save, save,
                                   profiles=P, verbose=False, workers=a.workers)
    R = np.pad(resp, ((0, 0), (0, max(0, a.pad - resp.shape[1])), (0, 0)))
    ref = R.shape[1]
    H, w, idx = xspec.transfer(R, t.cols[sub], a.nfreq)
    del resp, R
    Lw = None
    if a.whiten > 0:
        H, Lw = xspec.whiten(H, a.whiten)
    print(f"  {len(P)} pieces, {H.shape[0]} frequencies, {n} solve vertices, "
          f"damping {a.damp:g}, whiten {a.whiten:g}")
    return dict(c=c, t=t, P=P, p=p, save=save, sub=sub, val=val, Tgt=Tgt, H=H, w=w,
                idx=idx, ref=ref, Lw=Lw, labels=labels, tags=tags)


def summarise(S, w, idx, ref, P, cortex, labels):
    """Where the input's power sits, spatially and spectrally, and how spread it is."""
    nf, K = S.shape[0], S.shape[1]
    pw = np.array([np.trace(S[f]).real * w[f] for f in range(nf)])
    pw = np.clip(pw, 0, None)
    ranks = [float((lambda e: e.sum() ** 2 / max((e ** 2).sum(), 1e-30))
                   (np.clip(np.linalg.eigvalsh(0.5 * (S[f] + S[f].conj().T)), 0, None)))
             for f in range(nf)]
    per_piece = np.array([sum(w[f] * S[f, k, k].real for f in range(nf))
                          for k in range(K)])
    per_piece = np.clip(per_piece, 0, None)
    per_piece = per_piece / max(per_piece.sum(), 1e-30)
    return dict(
        band_spread=float(pw.sum() ** 2 / max((pw ** 2).sum(), 1e-30)),
        peak_bin=int(idx[int(np.argmax(pw))]),
        mean_rank=float(np.mean(ranks)),
        piece_spread=float(1.0 / max((per_piece ** 2).sum(), 1e-30)),
        piece_power=per_piece)


def score(k, S, a, seed=1000, law="gaussian", draws=None):
    """Realise, simulate, score honestly. -> (mean, sd, gap, rank)."""
    draws = draws or a.draws
    sims, gaps, rks = [], [], []
    for d in range(draws):
        A = xspec.realise(S, k["idx"], a.frames, ref_frames=k["ref"], seed=seed + d,
                          law=law)
        r = xspec.score_realisation(k["c"], k["t"], k["p"], A, save=k["save"],
                                    profiles=k["P"])
        sims.append(r["sim"]); gaps.append(r["gap"]); rks.append(r["rank"])
    return float(np.mean(sims)), float(np.std(sims)), float(np.mean(gaps)), \
        float(np.mean(rks))


AXES = [
    ("argmax",       None,                   0.0),
    ("max-entropy",  "prank",                +1.0),
    ("min-rank",     "prank",                -1.0),
    ("local-coal",   "distance",             -1.0),
    ("independent",  "offdiag",              -1.0),
    ("coordinated",  "offdiag",              +1.0),
]


def make_reg(name, k):
    if name == "prank":
        return xspec.prank_reg()
    if name == "offdiag":
        return xspec.offdiag_reg()
    if name == "distance":
        lab = np.asarray(k["labels"])
        cen = np.stack([k["c"].V[lab == i].mean(0) for i in range(len(k["P"]))])
        D = np.linalg.norm(cen[:, None, :] - cen[None, :, :], axis=2)
        return xspec.distance_reg(D, float(np.median(D[D > 0])))
    raise ValueError(name)


def run_members(k, a):
    print(f"\n  solving the argmax ({a.iters} iterations)")
    S0, _ = xspec.solve(k["H"], k["w"], k["Tgt"], iters=a.iters, verbose=False)
    rows = []
    for nm, reg, sg in AXES:
        t0 = time.time()
        if reg is None:
            S, rep = S0, dict(rho=None, rho_star=None, binding=False)
        else:
            R = make_reg(reg, k)
            if k["Lw"] is not None:
                # the solve runs on Q = L^H S L, but "maximum-entropy input" has to mean
                # maximum-entropy S, not maximum-entropy Q
                R = xspec.in_original_basis(R, k["Lw"])
            S, rep = xspec.family_member(k["H"], k["w"], k["Tgt"], S0, R,
                                         eps=a.eps, iters=a.family_iters, sign=sg,
                                         verbose=False)
        Su = xspec.unwhiten(S, k["Lw"]) if k["Lw"] is not None else S
        st = summarise(Su, k["w"], k["idx"], k["ref"], k["P"], k["c"], k["labels"])
        m, sd, gap, rk = score(k, Su, a)
        rows.append((nm, rep, st, m, sd, gap, rk))
        fit_s = "  (argmax)" if rep["rho"] is None else f"{rep['rho']:+.4f}"
        print(f"    {nm:<13s} solve {fit_s:>9s}  realised {m:+.4f} ± {sd:.4f}  "
              f"field rank {rk:6.1f}  gap {gap:.3f}  S rank {st['mean_rank']:5.2f}  "
              f"pieces {st['piece_spread']:5.1f}  [{time.time()-t0:.0f}s]", flush=True)
    _report(rows, a)
    return rows


def _report(rows, a):
    base = [r for r in rows if r[0] == "argmax"][0]
    print(f"\n  --- the admissible family at eps = {a.eps} ---")
    print(f"  {'member':<13s} {'realised':>10s} {'vs argmax':>10s} {'field rank':>11s} "
          f"{'S(f) rank':>10s} {'pieces used':>12s}")
    for nm, rep, st, m, sd, gap, rk in rows:
        print(f"  {nm:<13s} {m:>+10.4f} {m-base[3]:>+10.4f} {rk:>11.1f} "
              f"{st['mean_rank']:>10.2f} {st['piece_spread']:>12.1f}")
    ms = [r[3] for r in rows]
    rks = [r[6] for r in rows]
    pcs = [r[2]["piece_spread"] for r in rows]
    print(f"  input spread over pieces: {min(pcs):.1f} to {max(pcs):.1f} of "
          f"{len(rows and [1]) and rows[0][2]['piece_power'].size} "
          f"({max(pcs)/max(min(pcs),1e-9):.0f}x)")
    print(f"\n  realised score spans {min(ms):+.4f} to {max(ms):+.4f} "
          f"({max(ms)-min(ms):.4f}) across members held within {a.eps} of the same solve "
          f"objective")
    print(f"  field rank spans {min(rks):.1f} to {max(rks):.1f} "
          f"({max(rks)/max(min(rks),1e-9):.1f}x)")
    print(f"  -> whatever differs across these is something FC does not determine")


def run_laws(k, a):
    print(f"\n  solving the argmax ({a.iters} iterations)")
    S, _ = xspec.solve(k["H"], k["w"], k["Tgt"], iters=a.iters, verbose=False)
    if k["Lw"] is not None:
        S = xspec.unwhiten(S, k["Lw"])
    print(f"\n  --- same S(f), different realisation law ---")
    print(f"  {'law':<10s} {'realised':>10s} {'sd':>8s} {'field rank':>11s} "
          f"{'gap':>7s} {'kurtosis':>9s}")
    out = []
    for law in ("gaussian", "phase", "heavy"):
        A = xspec.realise(S, k["idx"], a.frames, ref_frames=k["ref"], seed=1000, law=law)
        kurt = float((((A - A.mean()) / A.std()) ** 4).mean())
        m, sd, gap, rk = score(k, S, a, law=law)
        out.append((law, m, sd, gap, rk, kurt))
        print(f"  {law:<10s} {m:>+10.4f} {sd:>8.4f} {rk:>11.1f} {gap:>7.3f} "
              f"{kurt:>9.2f}", flush=True)
    g = out[0]
    print(f"\n  static FC is second-order blind, so all three SHOULD score alike.")
    for law, m, sd, gap, rk, kurt in out[1:]:
        print(f"    {law:<9s} differs from gaussian by {m-g[1]:+.4f} "
              f"(gaussian draw scatter {g[2]:.4f}) - "
              f"{'within noise, as predicted' if abs(m-g[1]) < 2.5*max(g[2],1e-6) else 'OUTSIDE draw scatter: the rank transform sees the margin'}")
    print(f"  drive kurtosis {g[5]:.2f} -> " +
          ", ".join(f"{l} {kk:.2f}" for l, _, _, _, _, kk in out[1:]) +
          "; identical FC, different temporal signature")
    return out


def run_bands(k, a):
    nf = k["H"].shape[0]
    edges = np.linspace(0, nf, a.bands + 1).astype(int)
    print(f"\n  --- input restricted to one band at a time ({a.bands} bands of "
          f"{nf} frequencies) ---")
    print(f"  {'band':<12s} {'bins':>10s} {'solve':>9s} {'realised':>10s} "
          f"{'field rank':>11s}")
    S_all, _ = xspec.solve(k["H"], k["w"], k["Tgt"], iters=a.iters, verbose=False)
    Su = xspec.unwhiten(S_all, k["Lw"]) if k["Lw"] is not None else S_all
    m, sd, gap, rk = score(k, Su, a, draws=max(1, a.draws // 2))
    print(f"  {'all':<12s} {nf:>10d} {'':>9s} {m:>+10.4f} {rk:>11.1f}")
    for b in range(a.bands):
        keep = np.zeros(nf, bool)
        keep[edges[b]:edges[b + 1]] = True
        S, _ = xspec.solve(k["H"], k["w"], k["Tgt"], iters=a.iters, verbose=False,
                           freq_keep=keep)
        Su = xspec.unwhiten(S, k["Lw"]) if k["Lw"] is not None else S
        m, sd, gap, rk = score(k, Su, a, draws=max(1, a.draws // 2))
        lo, hi = k["idx"][edges[b]], k["idx"][edges[b + 1] - 1]
        print(f"  {f'bins {lo}-{hi}':<12s} {int(keep.sum()):>10d} {'':>9s} "
              f"{m:>+10.4f} {rk:>11.1f}", flush=True)


def check():
    """Verify the family machinery on a small synthetic problem.

    Four things, all of which have already been wrong once. The regulariser gradients are
    finite-differenced in BOTH bases, because the whitened wrapper's operand order is a
    trap that leaves the search running and producing plausible-looking answers. mu=0 no
    longer exists as a path, but the unregularised solve must still be untouched by the
    freq_keep argument. And the realisation laws must have identity innovation covariance,
    since that is the only reason they preserve S(f) at all."""
    rng = np.random.default_rng(1)
    nf, nV, K = 6, 40, 6
    H = (rng.standard_normal((nf, nV, K)) + 1j * rng.standard_normal((nf, nV, K)))
    Ht, L = xspec.whiten(H, 1e-3)
    A = rng.standard_normal((nf, K, K)) + 1j * rng.standard_normal((nf, K, K))
    S = np.stack([a @ a.conj().T for a in A])
    Q = np.stack([L[f].conj().T @ S[f] @ L[f] for f in range(nf)])
    D0 = np.abs(np.arange(K)[:, None] - np.arange(K)[None, :]).astype(float)

    print("  regulariser gradients vs finite differences:")
    worst = 0.0
    for nm, R in (("prank", xspec.prank_reg()), ("offdiag", xspec.offdiag_reg()),
                  ("logdet", xspec.logdet_reg()),
                  ("distance", xspec.distance_reg(D0, 2.0))):
        for basis, RR, X in (("original", R, S),
                             ("whitened", xspec.in_original_basis(R, L), Q)):
            _, g = RR(X)
            eps = 1e-6
            D = rng.standard_normal(X.shape) + 1j * rng.standard_normal(X.shape)
            D = 0.5 * (D + np.conj(np.transpose(D, (0, 2, 1))))
            num = (RR(X + eps * D)[0] - RR(X - eps * D)[0]) / (2 * eps)
            ana = float(np.real(np.sum(np.conj(g) * D)))
            err = abs(num - ana) / max(abs(num), 1e-12)
            worst = max(worst, err)
            print(f"    {nm:<9s} {basis:<9s} rel err {err:.2e}"
                  + ("   FAIL" if err > 1e-4 else ""))
    print(f"  worst gradient error {worst:.2e} ({'OK' if worst < 1e-4 else 'FAIL'})")

    X = rng.standard_normal((nV, 20)); Ct = X @ X.T / 20
    Ct = Ct - Ct.mean(0, keepdims=True) - Ct.mean(1, keepdims=True) + Ct.mean()
    np.fill_diagonal(Ct, 0.0)
    w = np.ones(nf)
    S0, _ = xspec.solve(H, w, Ct, iters=60, verbose=False)
    S1, _ = xspec.solve(H, w, Ct, iters=60, verbose=False, freq_keep=np.ones(nf, bool))
    print(f"  all-frequencies freq_keep is a no-op: {np.array_equal(S0, S1)}")
    keep = np.zeros(nf, bool); keep[1:3] = True
    S2, _ = xspec.solve(H, w, Ct, iters=60, verbose=False, freq_keep=keep)
    out = max(float(np.abs(S2[f]).max()) for f in range(nf) if not keep[f])
    print(f"  band-limited solve puts no power outside the band: {out == 0.0}")

    print("  innovation covariance E[eta eta^H] (must be I; MC error at n=1e5):")
    for law in ("gaussian", "phase", "heavy"):
        E = np.zeros((4, 4), complex)
        n = 100_000
        for _ in range(n):
            e = xspec.draw_eta(rng, 4, law)
            E += np.outer(e, e.conj())
        print(f"    {law:<9s} max |E - I| = {np.abs(E / n - np.eye(4)).max():.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true",
                    help="verify the machinery on a synthetic problem and exit")
    ap.add_argument("--members", action="store_true", help="the input-structure axis")
    ap.add_argument("--laws", action="store_true", help="the realisation-law axis")
    ap.add_argument("--bands", type=int, default=0, help="the input-timescale axis")
    ap.add_argument("--eps", type=float, default=0.01,
                    help="how far the fit may fall to buy a different input")
    ap.add_argument("--iters", type=int, default=50, help="argmax solve iterations")
    ap.add_argument("--family-iters", type=int, default=200, dest="family_iters")
    ap.add_argument("--frames", type=int, default=1120)
    ap.add_argument("--draws", type=int, default=2)
    ap.add_argument("--damp", type=float, default=2e-4)
    ap.add_argument("--window", type=int, default=560)
    ap.add_argument("--whiten", type=float, default=1e-3)
    ap.add_argument("--split", type=int, default=50)
    ap.add_argument("--nvert", type=int, default=1000)
    ap.add_argument("--nfreq", type=int, default=192)
    ap.add_argument("--pad", type=int, default=1120)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--tag", default="family")
    a = ap.parse_args()
    if a.check:
        check()
        return
    if not (a.members or a.laws or a.bands):
        a.members = True

    k = setup(a)
    if a.members:
        run_members(k, a)
    if a.laws:
        run_laws(k, a)
    if a.bands:
        run_bands(k, a)


if __name__ == "__main__":
    main()
