"""Does restricting the SOLVE to reliable vertices help, or only restricting the score?

Two different things have been conflated. Scoring the existing model on reliable vertices
raises it from +0.6293 to +0.6615 - but that model was fitted to all 9,217, so the gain is
just "we stopped grading it on noisy vertices". Restricting the FIT is the untested one:
the solve picks 1,000 medoid vertices from the whole sheet, so it currently spends capacity
matching targets that are partly noise, and cannot trade that capacity for anything.

Both arms are scored on the SAME reliable vertex set, so the only difference is which
vertices the solve was allowed to see.

  python reliable_fit.py --thr 0.92
"""
import argparse, os, time
import numpy as np

from mesh_cache import load_cortex
from paths import CACHE, RESULTS
import fc_score, xspec, bo_step, subparcels, timescale, units


def medoids_from(target, pool, n, sketch=400, seed=0):
    """k-means medoids restricted to a candidate pool of vertices."""
    from sklearn.cluster import MiniBatchKMeans
    rng = np.random.default_rng(seed)
    FC = np.asarray(target.target_fc(), np.float32)
    sk = FC[np.ix_(pool, rng.choice(target.nV, sketch, replace=False))]
    km = MiniBatchKMeans(n_clusters=n, random_state=seed, n_init=3, batch_size=2048).fit(sk)
    out = []
    for k in range(n):
        idx = np.flatnonzero(km.labels_ == k)
        if len(idx):
            out.append(pool[idx[np.argmin(((sk[idx] - km.cluster_centers_[k]) ** 2).sum(1))]])
    return np.sort(np.unique(out))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--thr", type=float, default=0.92, help="reliability threshold")
    ap.add_argument("--nvert", type=int, default=1000, help="solve vertices")
    ap.add_argument("--nscore", type=int, default=2000, help="vertices used for scoring")
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--draws", type=int, default=3)
    ap.add_argument("--spread-mm-s", type=float, default=3.0, dest="spread")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    c = load_cortex("fsaverage5", verbose=False)
    t = fc_score.default_target(c, metric="pearson", verbose=False)
    rel = np.load(os.path.join(CACHE, "vertex_quality_1500_0_9217.npz"))["rel"]
    pool = np.flatnonzero(rel > a.thr)
    print(f"  reliability > {a.thr}: {len(pool)} of {t.nV} vertices "
          f"(mean rel {rel[pool].mean():.4f} vs {rel.mean():.4f} overall)")

    lab, tg = subparcels.split_parcels(c, subparcels.SENSORY, 50, verbose=False)
    P = subparcels.taper_profiles(c, lab, len(tg))
    cl = timescale.plan(4, decay_s=timescale.BOLD_TAU_S, spread_mm_s=a.spread,
                        verbose=False)
    x = np.array(__import__("best_fit").BEST_X, copy=True)
    x[3] = np.log10(cl["save"]); x[0] = np.log10(cl["damp"])
    p, save, _ = bo_step.unpack(x, c)
    frames = timescale.frames_for(577.0, cl["frame_s"])

    # the scoring set: reliable vertices the SOLVE never sees, in either arm
    rng = np.random.default_rng(7)
    all_solve = xspec.medoid_subset(t, a.nvert)
    rel_solve = medoids_from(t, pool, a.nvert)
    seen = np.union1d(all_solve, rel_solve)
    cand = np.setdiff1d(pool, seen)
    score_v = np.sort(rng.choice(cand, min(a.nscore, len(cand)), replace=False))
    print(f"  solve sets: {len(all_solve)} from all vertices, {len(rel_solve)} from the "
          f"reliable pool ({len(np.intersect1d(all_solve, rel_solve))} shared)")
    print(f"  scoring on {len(score_v)} reliable vertices neither solve saw")

    resp = xspec.impulse_responses(c, list(range(len(P))), p, 224 * save, save,
                                   profiles=P, verbose=False, workers=a.workers)
    R = np.pad(resp, ((0, 0), (0, max(0, 4096 - resp.shape[1])), (0, 0))); del resp
    kern = units.smoothing_kernel(
        timescale.bold_fwhm_frames(cl["frame_s"], verbose=False), verbose=False)
    ref = R.shape[1]

    Gs = np.asarray(t.target_fc()[np.ix_(score_v, score_v)], np.float64)
    Gs = Gs - Gs.mean(0, keepdims=True) - Gs.mean(1, keepdims=True) + Gs.mean()
    ius = np.triu_indices(len(score_v), 1)
    ys = Gs[ius]; ys = (ys - ys.mean()) / ys.std()

    print(f"\n  {'solve vertices':<22s} {'score (held-out reliable)':>26s} "
          f"{'all-vertex score':>18s} {'rank':>7s}")
    for nm, sv in (("all 9,217", all_solve), (f"reliable only", rel_solve)):
        H, w, idx = xspec.transfer(R, t.cols[sv], 192, kernel=kern)
        raw = np.asarray(t.target_fc()[np.ix_(sv, sv)], np.float64)
        raw = raw - raw.mean(0, keepdims=True) - raw.mean(1, keepdims=True) + raw.mean()
        iu = np.triu_indices(len(sv), 1)
        Tgt = xspec.normal_scores(raw, iu); Tgt[np.eye(len(sv), dtype=bool)] = 0.0
        t0 = time.time()
        S, _ = xspec.solve(H, w, Tgt, iters=a.iters, verbose=False)
        sc, full, rks = [], [], []
        for d in range(a.draws):
            A = xspec.realise(S, idx, frames, ref_frames=ref, seed=9000 + d)
            r = xspec.score_realisation(c, t, p, A, save=save, profiles=P, kernel=kern)
            Z, _ = t.model_z(r["frames"])
            full.append(float(t._prep(t.model_edges(Z=Z)[0]) @ t.y))
            rks.append(r["rank"])
            Zv = Z[score_v].astype(np.float64)
            Zv -= Zv.mean(1, keepdims=True)
            Zv /= np.maximum(Zv.std(1, keepdims=True), 1e-12)
            F = (Zv @ Zv.T) / Zv.shape[1]
            F = F - F.mean(0, keepdims=True) - F.mean(1, keepdims=True) + F.mean()
            v = F[ius]; v = (v - v.mean()) / max(v.std(), 1e-30)
            sc.append(float(v @ ys / len(v)))
        print(f"  {nm:<22s} {np.mean(sc):>+15.4f}+-{np.std(sc):<9.4f} "
              f"{np.mean(full):>+10.4f}+-{np.std(full):<6.4f} {np.mean(rks):>7.1f} "
              f"  [{time.time()-t0:.0f}s]", flush=True)
        del H, S
    print(f"\n  first column is the like-for-like comparison: both arms scored on the "
          f"same\n  reliable vertices that neither solve was fitted to.")


if __name__ == "__main__":
    main()
