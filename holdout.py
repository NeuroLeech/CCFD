"""Is the solved cross-spectrum fitting the group FC, or fitting its noise?

The solve has a lot of freedom: S is one Hermitian PSD matrix per frequency, so K driven
pieces at F frequencies carry F*K*(K+1)/2 real parameters - 596k of them at K=100, more
than the 500k edges of the 1000-vertex matrix it is fitted to. A better score under more
freedom is not by itself evidence of a better model.

The control splits the subjects rather than the vertices. Solve against a group FC built
from one half of the subjects, then score the realisation against the OTHER half. Nothing
about the target's noise is shared between the two, so anything the solve gained by
fitting sampling noise is lost here, while anything it gained by fitting real structure
survives. The in-sample half is scored too, and the difference is the overfit.

  python holdout.py --regions spread --spread-scale 2 --frames 4480
"""
import os, argparse, time
import numpy as np

from mesh_cache import load_cortex
import fc_score
from paths import RESULTS, CACHE
import xspec, bo_step, subparcels, fc_group_nki as nki
from best_fit import BEST_X


def half_targets(target, seed=0, verbose=True):
    """Group FC from each half of the subjects, built exactly as the target was.

    Exactly matters. The matrix the model is scored against is the DOUBLE-CENTRED group
    FC (fc_centre.py), and double centring removes the global component - which is both
    the largest and the most reliable part of an FC matrix. Half-targets built without it
    would answer a question about a different matrix, and would flatter the reliability.

    So the full 9217 x 9217 matrix is formed per half, double-centred whole, and only then
    read out on the sampled edges and the solve vertices. -> (edges, solve blocks, sub)."""
    from scipy.stats import rankdata
    from fc_score import double_centre
    cache = os.path.join(CACHE, f"halves_dc_{seed}_{target.nV}_{len(target.i)}.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        return [z["eA"], z["eB"]], [z["mA"], z["mB"]], z["sub"]
    files = nki.subject_files("left")
    n = len(files)
    sub = xspec.medoid_subset(target, 1000)
    rng = np.random.default_rng(seed)
    which = rng.permutation(n) % 2                      # balanced halves
    acc = [np.zeros((target.nV, target.nV), np.float32) for _ in range(2)]
    cnt = [0, 0]
    t0 = time.time()
    for s, path in enumerate(files):
        h = int(which[s])
        X = nki.load_subject(path)[target.vertices]
        Z = rankdata(X, axis=1).astype(np.float32)
        Z -= Z.mean(1, keepdims=True)
        Z /= np.maximum(Z.std(1, keepdims=True), 1e-12)
        S = (Z @ Z.T) / Z.shape[1]
        np.clip(S, -0.9999, 0.9999, out=S)
        acc[h] += np.arctanh(S)
        cnt[h] += 1
        if verbose and ((s + 1) % 20 == 0 or s == n - 1):
            print(f"  {s+1:3d}/{n} subjects  [{time.time()-t0:.0f}s]", flush=True)
        del X, Z, S
    e, m = [], []
    for h in (0, 1):
        FC = np.tanh(acc[h] / cnt[h])
        np.fill_diagonal(FC, 1.0)
        FC = double_centre(FC, inplace=True)
        e.append(np.asarray(FC[target.i, target.j], np.float32))
        m.append(np.asarray(FC[np.ix_(sub, sub)], np.float64))
        acc[h] = None
        del FC
    if verbose:
        print(f"  halves of {cnt[0]} and {cnt[1]} subjects, double-centred like the target")
    np.savez(cache, eA=e[0], eB=e[1], mA=m[0], mB=m[1], sub=sub)
    return e, m, sub


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--regions", default="spread",
                    choices=("sensory", "dmn", "sensory+dmn", "spread"))
    ap.add_argument("--spread-scale", type=float, default=2.0, dest="spread_scale")
    ap.add_argument("--split", type=int, default=50)
    ap.add_argument("--frames", type=int, default=4480)
    ap.add_argument("--draws", type=int, default=2)
    ap.add_argument("--iters", type=int, default=150)
    ap.add_argument("--nfreq", type=int, default=192)
    ap.add_argument("--pad", type=int, default=1120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="holdout")
    a = ap.parse_args()

    c = load_cortex("fsaverage5", verbose=False)
    t = fc_score.default_target(c, verbose=True)
    e, m, sub = half_targets(t, a.seed)
    yA, yB = t._prep(e[0]), t._prep(e[1])
    print(f"  the two halves agree with each other at spearman {float(yA @ yB):+.4f}, "
          f"and with the 99-subject target at {float(yA @ t.y):+.4f} / "
          f"{float(yB @ t.y):+.4f}")

    parcels, split = subparcels.region_set(c, a.regions, a.split, a.spread_scale)
    labels, tags = subparcels.split_parcels(c, parcels, split, verbose=False)
    P = subparcels.taper_profiles(c, labels, len(tags))
    p, save, _ = bo_step.unpack(BEST_X, c)
    K, F = len(P), None
    resp = xspec.impulse_responses(c, list(range(K)), p, 280 * save, save,
                                   profiles=P, verbose=False)
    R = np.pad(resp, ((0, 0), (0, max(0, a.pad - resp.shape[1])), (0, 0)))
    H, w, idx = xspec.transfer(R, t.cols[sub], a.nfreq)
    F = H.shape[0]
    print(f"  {K} pieces, {F} frequencies: the solve carries "
          f"{F*K*(K+1)//2:,} real parameters against "
          f"{len(sub)*(len(sub)-1)//2:,} solve edges")

    # solve against half A only
    A_fc = m[0] - m[0].mean(0, keepdims=True) - m[0].mean(1, keepdims=True) + m[0].mean()
    S, C = xspec.solve(H, w, xspec.normal_scores(A_fc), iters=a.iters, verbose=False)
    from scipy.stats import spearmanr
    iu = np.triu_indices(len(sub), 1)
    B_fc = m[1] - m[1].mean(0, keepdims=True) - m[1].mean(1, keepdims=True) + m[1].mean()
    print(f"  solve on half A: spearman vs A {spearmanr(C[iu], A_fc[iu]).statistic:+.4f} "
          f"(in sample), vs B {spearmanr(C[iu], B_fc[iu]).statistic:+.4f} (held out)")

    rows = []
    for d in range(a.draws):
        Adr = xspec.realise(S, idx, a.frames, ref_frames=R.shape[1], seed=1000 + d)
        r = xspec.score_realisation(c, t, p, Adr, save=save, profiles=P)
        Z, _ = t.model_z(r["frames"])
        v = t._prep(t.model_edges(Z=Z)[0])
        rows.append((float(v @ yA), float(v @ yB), float(v @ t.y)))
    rows = np.array(rows)
    mu, sd = rows.mean(0), rows.std(0)
    print(f"\n  realised over {a.frames} frames, {a.draws} draws:")
    print(f"    vs half A (solved on)  {mu[0]:+.4f} +- {sd[0]:.4f}")
    print(f"    vs half B (held out)   {mu[1]:+.4f} +- {sd[1]:.4f}")
    print(f"    vs all 99 subjects     {mu[2]:+.4f} +- {sd[2]:.4f}")
    print(f"  overfit to the solved half: {mu[0]-mu[1]:+.4f}")
    np.savez(os.path.join(RESULTS, f"xspec_{a.tag}.npz"), S=S, idx=idx, x=BEST_X,
             save=save, labels=labels, tags=np.array(tags, dtype=object), scores=rows)


if __name__ == "__main__":
    main()
