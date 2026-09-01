"""The gap between the best fit and 1.0, decomposed into named, measured losses.

+0.6433 against a target whose own reliability caps any score at +0.9679. That leaves
0.325 of model failure, and "where does it go" has been asked repeatedly and answered
qualitatively. This answers it as a ladder, where every rung is a measured number and
every step between two rungs is one identifiable mechanism:

  reliability ceiling      sqrt(split-half reliability) - no model can beat this
  span ceiling             the best ANY input to this medium could do, || P Ct P || / ||Ct||
  solve objective          what the projected gradient actually reached
  ... surrogate loss       the solve matches Pearson-against-normal-scores; the score is
                           Spearman-against-raw, and the two are not the same objective
  ... fidelity loss        C(S) is a prediction; the simulation is the thing scored
  ... generalisation loss  the solve saw 1,000 medoid vertices, the score sees 9,217
  reported score           the number quoted

Read the steps, not the rungs: a large drop names where the effort should go.

  python ceiling.py --tag whiten
"""
import os, argparse
import numpy as np
from scipy.stats import spearmanr, rankdata

from mesh_cache import load_cortex
from paths import RESULTS
import fc_score, xspec, bo_step, subparcels


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", default="whiten", help="a saved best_fit run")
    ap.add_argument("--ceiling", type=float, default=0.9679,
                    help="from reliability.py --splits 3")
    ap.add_argument("--nvert", type=int, default=1000)
    ap.add_argument("--nfreq", type=int, default=192)
    ap.add_argument("--pad", type=int, default=1120)
    ap.add_argument("--window", type=int, default=560)
    ap.add_argument("--split", type=int, default=50)
    ap.add_argument("--span-dims", type=int, default=400, dest="span_dims")
    ap.add_argument("--unconstrained", type=int, default=0, metavar="ITERS",
                    help="also solve WITHOUT the PSD cone, to split the conditioning "
                         "step into 'the constraint binds' and 'the solver does not "
                         "reach a point it is allowed to reach'. The span ceiling assumes "
                         "some PSD S realises the target's projection into the span, and "
                         "nothing guarantees that")
    ap.add_argument("--refit", type=int, default=0, metavar="ITERS",
                    help="re-solve from scratch at this many iterations instead of using "
                         "the saved S, for a like-for-like comparison")
    a = ap.parse_args()

    c = load_cortex("fsaverage5", verbose=False)
    t = fc_score.default_target(c, verbose=False)
    z = np.load(os.path.join(RESULTS, f"xspec_{a.tag}.npz"), allow_pickle=True)
    S, idx, x, save = z["S"], z["idx"], z["x"], int(z["save"])
    p, _, _ = bo_step.unpack(x, c)
    labels, tags = subparcels.split_parcels(c, subparcels.SENSORY, a.split, verbose=False)
    P = subparcels.taper_profiles(c, labels, len(tags))
    print(f"  {a.tag}: {S.shape[1]} pieces, {S.shape[0]} frequencies, "
          f"per-step damping {10**x[0]:.2e}, save {save}")

    sub = xspec.medoid_subset(t, a.nvert)
    val = xspec.validation_subset(t, sub, a.nvert)
    n = len(sub)
    iu = np.triu_indices(n, 1)
    raw = np.asarray(t.target_fc()[np.ix_(sub, sub)], np.float64)
    raw = raw - raw.mean(0, keepdims=True) - raw.mean(1, keepdims=True) + raw.mean()
    Tgt = xspec.normal_scores(raw, iu)
    Tgt[np.eye(n, dtype=bool)] = 0.0

    resp = xspec.impulse_responses(c, list(range(len(P))), p, a.window * save, save,
                                   profiles=P, verbose=False)
    R = np.pad(resp, ((0, 0), (0, max(0, a.pad - resp.shape[1])), (0, 0)))
    H, w, _ = xspec.transfer(R, t.cols[sub], a.nfreq)
    del resp, R

    # ---- rung: span ceiling. C = sum_f w_f 2 Re(H S H^H) forces range(C) into the real
    # span of H's columns, so no input whatsoever can reach outside it.
    A = np.concatenate([np.sqrt(w[f]) * H[f] for f in range(len(w))], axis=1)
    U, sv, _ = np.linalg.svd(np.concatenate([A.real, A.imag], axis=1),
                             full_matrices=False)
    m = min(a.span_dims, U.shape[1])
    Pm = U[:, :m]
    span = float(np.linalg.norm(Pm @ (Pm.T @ Tgt @ Pm) @ Pm.T) / np.linalg.norm(Tgt))
    del A

    # ---- rung: what the solve reached, in its own currency and in the score's
    C = np.zeros((n, n))
    for f in range(len(w)):
        C += w[f] * 2.0 * np.real((H[f] @ S[f]) @ H[f].conj().T)
    C = C - C.mean(0, keepdims=True) - C.mean(1, keepdims=True) + C.mean()
    Tn = Tgt / np.linalg.norm(Tgt)
    Cm = C.copy(); Cm[np.eye(n, dtype=bool)] = 0.0
    obj = float((Cm * Tn).sum() / np.linalg.norm(Cm))
    solve_sp = float(spearmanr(C[iu], raw[iu]).statistic)

    extra = []
    if a.unconstrained or a.refit:
        Hs, Lw = xspec.whiten(H, 1e-3)
        it = a.refit or a.unconstrained
        Sc, Cc = xspec.solve(Hs, w, Tgt, iters=it, verbose=False)
        Sc = xspec.unwhiten(Sc, Lw)
        Cc = np.zeros((n, n))
        for f in range(len(w)):
            Cc += w[f] * 2.0 * np.real((H[f] @ Sc[f]) @ H[f].conj().T)
        Cc = Cc - Cc.mean(0, keepdims=True) - Cc.mean(1, keepdims=True) + Cc.mean()
        Cc[np.eye(n, dtype=bool)] = 0.0
        o_psd = float((Cc * Tn).sum() / np.linalg.norm(Cc))
        extra.append((f"re-solved, PSD, {it} iters", o_psd))
        if a.unconstrained:
            Su, _ = xspec.solve(Hs, w, Tgt, iters=a.unconstrained, verbose=False,
                                psd=False)
            Su = xspec.unwhiten(Su, Lw)
            Cu = np.zeros((n, n))
            for f in range(len(w)):
                Cu += w[f] * 2.0 * np.real((H[f] @ Su[f]) @ H[f].conj().T)
            Cu = Cu - Cu.mean(0, keepdims=True) - Cu.mean(1, keepdims=True) + Cu.mean()
            Cu[np.eye(n, dtype=bool)] = 0.0
            o_un = float((Cu * Tn).sum() / np.linalg.norm(Cu))
            mineig = min(float(np.linalg.eigvalsh(0.5 * (Su[f] + Su[f].conj().T)).min())
                         for f in range(len(Su)))
            extra.append((f"re-solved, NO PSD cone, {a.unconstrained} iters "
                          f"(min eig {mineig:+.2g})", o_un))

    # ---- rung: the simulation, on the same vertices, then on held-out ones
    frames = np.asarray(np.load(os.path.join(RESULTS, f"frames_{a.tag}.npy"),
                                mmap_mode="r"))
    Z, _ = t.model_z(frames)
    del frames

    def sim_block(vs):
        T_, V = Z.shape[1], Z.shape[0]
        Zv = Z[vs]
        F = (Zv @ Zv.T) / T_
        if t.centre == "double":
            ssum = Z.sum(0)
            dg = (Z * Z).sum(1) / T_
            mm = ((Z @ ssum) / T_ - dg) / (V - 1)
            grand = (float(ssum @ ssum) / T_ - float(dg.sum())) / (V * (V - 1))
            F = F - mm[vs][:, None] - mm[vs][None, :] + grand
        return F

    Fs = sim_block(sub)
    sim_solve = float(spearmanr(Fs[iu], raw[iu]).statistic)
    rawv = np.asarray(t.target_fc()[np.ix_(val, val)], np.float64)
    rawv = rawv - rawv.mean(0, keepdims=True) - rawv.mean(1, keepdims=True) + rawv.mean()
    Fv = sim_block(val)
    iuv = np.triu_indices(len(val), 1)
    sim_val = float(spearmanr(Fv[iuv], rawv[iuv]).statistic)
    reported = float(t._prep(t.model_edges(Z=Z)[0]) @ t.y)

    rungs = [
        ("perfect fit",                       1.0,        None),
        ("target reliability ceiling",        a.ceiling,  "target noise: no model can pass this"),
        (f"span ceiling ({m} directions)",    span,       "outside the medium's reach from these pieces"),
        ("solve objective reached",           obj,        "CONDITIONING: the solve does not reach its own span"),
        ("same solution, Spearman vs raw",    solve_sp,   "SURROGATE: solving Pearson-on-normal-scores is not the score"),
        ("simulated, on the solve vertices",  sim_solve,  "FIDELITY: C(S) is a prediction, the simulation is what is scored"),
        ("simulated, on held-out vertices",   sim_val,    "GENERALISATION: 1,000 medoids do not stand for 9,217"),
        ("reported score (2M edges)",         reported,   "sampling and the full vertex set"),
    ]
    print(f"\n  {'rung':<38s} {'value':>8s} {'step':>8s}   what the step is")
    prev = None
    for name, v, why in rungs:
        step = "" if prev is None else f"{v - prev:+8.4f}"
        print(f"  {name:<38s} {v:>+8.4f} {step:>8s}   {why or ''}")
        prev = v
    print(f"\n  total {rungs[0][1] - reported:.4f} from perfect; of that, "
          f"{1.0 - a.ceiling:.4f} is target noise and {a.ceiling - reported:.4f} is ours.")
    biggest = max(range(1, len(rungs)), key=lambda i: rungs[i - 1][1] - rungs[i][1])
    print(f"  largest single step: {rungs[biggest][2]} "
          f"({rungs[biggest-1][1] - rungs[biggest][1]:.4f})")

    if extra:
        print(f"\n  --- is the conditioning step the CONE or the SOLVER? ---")
        for name, v in extra:
            print(f"  {name:<52s} {v:>+8.4f}")
        if len(extra) == 2:
            psd_v, un_v = extra[0][1], extra[1][1]
            print(f"\n  span ceiling {span:+.4f}; PSD solve {psd_v:+.4f}; "
                  f"same solver with the cone removed {un_v:+.4f}.")
            frac = (un_v - psd_v) / max(span - psd_v, 1e-12)
            print(f"  Dropping the cone recovers {un_v - psd_v:+.4f} of the "
                  f"{span - psd_v:.4f} gap ({frac:.0%}), so at LEAST that much of it is "
                  f"the PSD constraint: the target's projection into the span is not the "
                  f"covariance of any input, and the span bound is loose by at least that "
                  f"much.")
            print(f"  The remaining {span - un_v:.4f} is the solver failing to reach "
                  f"points it is already allowed to reach. Both shares are LOWER bounds "
                  f"on the cone's part and UPPER bounds on the solver's: the "
                  f"unconstrained solve is run by the same projected gradient and is "
                  f"itself solver-limited, so a better optimiser would raise it and move "
                  f"the split further towards the cone.")


if __name__ == "__main__":
    main()
