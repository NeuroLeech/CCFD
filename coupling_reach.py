"""Does fast long-range wiring let the field reach patterns it could not before, and
does that improve the FC match?

Two questions, kept separate because they can have different answers: the wiring could
enlarge the reachable set without any of the new directions being useful, or it could
improve the fit merely by re-weighting directions it already had.

REACH. C = sum_f w_f 2 Re(H S H^H), so what the field can express lives in the real span
of H's columns. With coupling that span is U1, without it U0. Principal angles between
them say how much of U1 is genuinely NEW - directions no input to the uncoupled medium
could produce. Then, of that new part, how much does the uncoupled fit's RESIDUAL need?
New directions the residual does not load on are irrelevant however many there are.

MATCH. Solve, realise, simulate and score, end to end, at each coupling strength. This is
the only thing that answers "can this improve the fit", and no span statistic substitutes
for it.

The strength range is chosen by MEASUREMENT, not precedent: the relative change in the
impulse response tells you where the term stops being inert. On this medium that is
lam ~ 0.3 upward; lam <= 0.03 moves the field by under 8% and is not a test of anything.

  python coupling_reach.py --lams 0,0.3,1,3 --workers 8
"""
import argparse, time
import numpy as np

from mesh_cache import load_cortex
import fc_score, xspec, bo_step, subparcels, connectome
from best_fit import BEST_X


def field_modes(frames, t, k=6):
    """The spatial modes the field ACTUALLY expresses: leading eigenvectors of the
    realised field's spatial covariance, on the scored vertices.

    The span says what the medium COULD reach; this says what it did. They are different
    questions and the coupling result turns on the difference - span400 falls with
    coupling while the realised score rises."""
    Z, _ = t.model_z(frames)
    Z = Z - Z.mean(1, keepdims=True)
    G = (Z.T @ Z)                                  # (T, T), T << nV
    ev, W = np.linalg.eigh(G)
    o = np.argsort(ev)[::-1][:k]
    M = Z @ W[:, o]
    M /= np.maximum(np.linalg.norm(M, axis=0, keepdims=True), 1e-30)
    return M, np.clip(ev[o], 0, None)


def mode_overlap(M0, M1):
    """Principal angles between two sets of field modes; 1 = identical subspace."""
    return np.clip(np.linalg.svd(M0.T @ M1, compute_uv=False), 0.0, 1.0)


def real_span(H, w):
    """Orthonormal basis for the real span of H's columns, and its singular values."""
    A = np.concatenate([np.sqrt(w[f]) * H[f] for f in range(len(w))], axis=1)
    U, sv, _ = np.linalg.svd(np.concatenate([A.real, A.imag], axis=1),
                             full_matrices=False)
    return U, sv


def novelty(U0, U1, m=400):
    """How much of the coupled span's top-m directions is orthogonal to the uncoupled one.

    Principal angles via the singular values of U0^T U1: a value near 1 means that
    direction of U1 already lay in U0, near 0 means it is new. Reported as the number of
    effectively new directions and the energy fraction they carry."""
    A, B = U0[:, :m], U1[:, :m]
    c = np.linalg.svd(A.T @ B, compute_uv=False)
    c = np.clip(c, 0.0, 1.0)
    return c


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lams", default="0,0.3,1,3")
    ap.add_argument("--lags", default="0")
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--frames", type=int, default=4480)
    ap.add_argument("--draws", type=int, default=2)
    ap.add_argument("--damp", type=float, default=2e-4)
    ap.add_argument("--window", type=int, default=560)
    ap.add_argument("--whiten", type=float, default=1e-3)
    ap.add_argument("--split", type=int, default=50)
    ap.add_argument("--nvert", type=int, default=1000)
    ap.add_argument("--nfreq", type=int, default=192)
    ap.add_argument("--pad", type=int, default=1120)
    ap.add_argument("--keep", type=float, default=0.15)
    ap.add_argument("--min-mm", type=float, default=60.0, dest="min_mm")
    ap.add_argument("--span-dims", type=int, default=400, dest="span_dims")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-realise", action="store_true", dest="no_realise")
    ap.add_argument("--oversample", type=int, default=0, metavar="K",
                    help="run on the fMRI clock at frames of TR/K (see timescale.py)")
    ap.add_argument("--decay-s", type=float, default=None, dest="decay_s")
    ap.add_argument("--spread-mm-s", type=float, default=None, dest="spread_mm_s")
    ap.add_argument("--seconds", type=float, default=577.0)
    ap.add_argument("--nmode", type=int, default=6,
                    help="field modes to compare against the uncoupled ones")
    a = ap.parse_args()

    lams = [float(v) for v in a.lams.split(",") if v.strip()]
    lags = [int(v) for v in a.lags.split(",") if v.strip()]

    c = load_cortex("fsaverage5", verbose=False)
    t = fc_score.default_target(c, verbose=False)
    labels, tags = subparcels.split_parcels(c, subparcels.SENSORY, a.split, verbose=False)
    P = subparcels.taper_profiles(c, labels, len(tags))
    x = BEST_X.copy(); x[0] = np.log10(a.damp)
    kern = None
    if a.oversample:
        import timescale, units
        clock = timescale.plan(a.oversample,
                               decay_s=(a.decay_s or timescale.BOLD_TAU_S),
                               spread_mm_s=a.spread_mm_s)
        x[3] = np.log10(clock["save"]); x[0] = np.log10(clock["damp"])
        decay_fr = 1.0 / (clock["damp"] * clock["save"])
        a.window = max(a.window, int(np.ceil(4 * decay_fr / 32.0) * 32))
        a.pad = max(a.pad, 4096)
        a.frames = timescale.frames_for(a.seconds, clock["frame_s"])
        kern = units.smoothing_kernel(
            timescale.bold_fwhm_frames(clock["frame_s"], verbose=False), verbose=False)
        print(f"  fMRI clock: frame {clock['frame_s']:.4f}s, save {clock['save']}, "
              f"window {a.window} frames, pad {a.pad}, realise {a.frames} frames")
    p, save, _ = bo_step.unpack(x, c)
    sub = xspec.medoid_subset(t, a.nvert)
    n = len(sub); iu = np.triu_indices(n, 1)
    raw = np.asarray(t.target_fc()[np.ix_(sub, sub)], np.float64)
    raw = raw - raw.mean(0, keepdims=True) - raw.mean(1, keepdims=True) + raw.mean()
    Tgt = xspec.normal_scores(raw, iu)
    Tgt[np.eye(n, dtype=bool)] = 0.0
    Tn = Tgt / np.linalg.norm(Tgt)

    D180 = connectome.parcel_distances(c, verbose=False)
    W = connectome.residual_W(connectome.load_enigma(c, verbose=False), D180,
                              a.keep, a.min_mm, verbose=True)
    print(f"\n  damping {a.damp:g}/step, window {a.window} frames, {len(P)} pieces, "
          f"solve {a.iters} iters, realise {a.frames} frames x {a.draws} draws")

    U0 = Res0 = None
    modes = []
    print(f"\n  {'lam':>6s} {'lag':>4s} {'dt*bnd':>8s} {'span400':>8s} {'new dirs':>9s} "
          f"{'resid in new':>13s} {'solve':>8s} {'realised':>18s} {'rank':>7s}")
    for lam in lams:
        for lag in (lags if lam > 0 else [0]):
            t0 = time.time()
            cpl = None
            bnd = 0.0
            if lam > 0:
                cpl = connectome.CouplingOperator(c, W, lam, lag)
                bnd = float(np.float64(cpl.spectral_bound()))
            resp = xspec.impulse_responses(c, list(range(len(P))), p, a.window * save,
                                           save, profiles=P, verbose=False,
                                           coupling=cpl, workers=a.workers)
            R = np.pad(resp, ((0, 0), (0, max(0, a.pad - resp.shape[1])), (0, 0)))
            H, w, idx = xspec.transfer(R, t.cols[sub], a.nfreq, kernel=kern)
            ref = R.shape[1]
            del resp, R
            U, sv = real_span(H, w)
            m = min(a.span_dims, U.shape[1])
            Pm = U[:, :m]
            span = float(np.linalg.norm(Pm @ (Pm.T @ Tgt @ Pm) @ Pm.T)
                         / np.linalg.norm(Tgt))

            newdirs, resid_new = "-", "-"
            if U0 is None:
                U0 = U.copy()
            else:
                cs = novelty(U0, U, m)
                nnew = int((cs < 0.9).sum())          # angle > ~26 deg from the old span
                newdirs = f"{nnew}"
                if Res0 is not None and nnew > 0:
                    # of the uncoupled fit's residual, how much lives in directions the
                    # coupled span has and the uncoupled one does not?
                    Q = U[:, :m] - U0[:, :m] @ (U0[:, :m].T @ U[:, :m])
                    q, _ = np.linalg.qr(Q)
                    keep = np.linalg.norm(Q, axis=0) > 1e-8
                    q = q[:, keep]
                    pr = q @ (q.T @ Res0 @ q) @ q.T
                    resid_new = f"{np.linalg.norm(pr)/np.linalg.norm(Res0):.3f}"

            Hs = H
            Lw = None
            if a.whiten > 0:
                Hs, Lw = xspec.whiten(H, a.whiten)
            S, C = xspec.solve(Hs, w, Tgt, iters=a.iters, verbose=False)
            if Lw is not None:
                S = xspec.unwhiten(S, Lw)
            C = np.zeros((n, n))
            for f in range(len(w)):
                C += w[f] * 2.0 * np.real((H[f] @ S[f]) @ H[f].conj().T)
            C = C - C.mean(0, keepdims=True) - C.mean(1, keepdims=True) + C.mean()
            C[np.eye(n, dtype=bool)] = 0.0
            obj = float((C * Tn).sum() / np.linalg.norm(C))

            if lam == 0:
                tv = Tgt[iu]; cv = C[iu]
                tv = (tv - tv.mean()) / tv.std(); cv = (cv - cv.mean()) / cv.std()
                beta = float(tv @ cv) / len(tv)
                Res0 = np.zeros((n, n)); Res0[iu] = tv - beta * cv
                Res0 = Res0 + Res0.T

            sim_s, rk_s = "(skipped)", "-"
            if not a.no_realise:
                run_fn = None
                if cpl is not None:
                    import fluid as fl
                    run_fn = (lambda dr, ns, sv_, _p=p, _c=cpl:
                              fl.run(c, dr, _p, ns, sv_, coupling=_c))
                sims, rks = [], []
                for d in range(a.draws):
                    A_ = xspec.realise(S, idx, a.frames, ref_frames=ref, seed=1000 + d)
                    r = xspec.score_realisation(c, t, p, A_, save=save, profiles=P,
                                                run_fn=run_fn, kernel=kern)
                    sims.append(r["sim"]); rks.append(r["rank"])
                    if d == 0:
                        modes.append((lam, lag, field_modes(r["frames"], t, a.nmode)))
                sim_s = f"{np.mean(sims):+.4f} +- {np.std(sims):.4f}"
                rk_s = f"{np.mean(rks):.1f}"
            print(f"  {lam:>6g} {lag:>4d} {bnd*0.174:>8.3g} {span:>8.4f} {newdirs:>9s} "
                  f"{resid_new:>13s} {obj:>8.4f} {sim_s:>18s} {rk_s:>7s}"
                  f"   [{time.time()-t0:.0f}s]", flush=True)
            del H, Hs, S, C

    if len(modes) > 1:
        (l0, g0, (M0, e0)) = modes[0]
        print(f"\n  --- do the FIELD MODES change? (leading {M0.shape[1]}, "
              f"principal angles against lam=0) ---")
        print(f"  {'lam':>6s} {'lag':>4s}  " +
              "  ".join(f"m{i+1}" for i in range(M0.shape[1])) + "   mean")
        for lam, lag, (M, e) in modes[1:]:
            cs = mode_overlap(M0, M)
            print(f"  {lam:>6g} {lag:>4d}  " +
                  "  ".join(f"{v:.2f}" for v in cs) + f"   {cs.mean():.3f}")
        print(f"  1.00 = the same subspace; lower means the coupling is expressing "
              f"spatial patterns the uncoupled medium did not")


if __name__ == "__main__":
    main()
