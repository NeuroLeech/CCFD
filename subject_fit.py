"""Is the GROUP FC something a single run could produce, and what is one subject's ceiling?

The model simulates one run. Its field has effective rank ~15; a single fMRI run measures
10.9-13.4. The GROUP FC has effective rank 93.3, because it averages 99 heterogeneous
subjects - an average of rank-13 matrices is not itself rank 13. So the target may be an
object no single process can be, and the residual would then contain between-subject
variability that no amount of model improvement can reach.

Three measurements, none of which need a model:

  RANK. Effective rank of one subject's FC against the group's, on the same vertices.
  CEILING. Within-subject split-half: FC from the first half of a run against the second.
    That is the ceiling on fitting THAT subject, and with one scan each it is the only
    within-subject reliability available. It also gives a per-vertex version, which is what
    "restrict the objective to reliable vertices" needs - the group-half reliability
    measured earlier is a BETWEEN-subject quantity and answers a different question.
  REACHABILITY. How well one subject's FC is approximated by a low-rank matrix, against
    the group's - the same curve computed for the group reached 0.973 at rank 12.

  python subject_fit.py --nsub 8
"""
import argparse
import numpy as np
from scipy.stats import rankdata

from mesh_cache import load_cortex
import fc_score, xspec, fc_group_nki as nki


def eff_rank(ev):
    e = np.clip(np.asarray(ev, float), 0, None)
    return float(e.sum() ** 2 / max((e ** 2).sum(), 1e-300))


def fc_of(X, centre=True):
    Z = X - X.mean(1, keepdims=True)
    sd = Z.std(1, keepdims=True); sd[sd == 0] = 1.0
    Z = Z / sd
    C = (Z @ Z.T) / Z.shape[1]
    if centre:
        C = C - C.mean(0, keepdims=True) - C.mean(1, keepdims=True) + C.mean()
        np.fill_diagonal(C, 1.0)
    return C


def cor_off(A, B, iu):
    a, b = A[iu], B[iu]
    a = a - a.mean(); b = b - b.mean()
    return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--nsub", type=int, default=8)
    ap.add_argument("--nvert", type=int, default=1000)
    a = ap.parse_args()

    c = load_cortex("fsaverage5", verbose=False)
    t = fc_score.default_target(c, metric="pearson", verbose=False)
    sub = xspec.medoid_subset(t, a.nvert)
    n = len(sub); iu = np.triu_indices(n, 1)
    verts = t.vertices[sub]
    G = np.asarray(t.target_fc()[np.ix_(sub, sub)], np.float64)
    G = G - G.mean(0, keepdims=True) - G.mean(1, keepdims=True) + G.mean()
    print(f"  {n} medoid vertices; group FC effective rank "
          f"{eff_rank(np.abs(np.linalg.eigvalsh(G))):.1f}")

    files = nki.subject_files("left")[:a.nsub]
    print(f"\n  {'subject':<12s} {'FC eff rank':>12s} {'vs group':>9s} "
          f"{'split-half':>11s} {'ceiling':>8s}")
    rows, halves = [], []
    for p in files:
        X = nki.load_subject(p)[verts].astype(np.float64)
        C = fc_of(X)
        T = X.shape[1]
        C1, C2 = fc_of(X[:, :T // 2]), fc_of(X[:, T // 2:])
        sh = cor_off(C1, C2, iu)
        # Spearman-Brown: reliability of the full run from its halves
        rel = 2 * sh / (1 + sh) if sh > -1 else np.nan
        rows.append((p.split("/")[-2], eff_rank(np.abs(np.linalg.eigvalsh(C))),
                     cor_off(C, G, iu), sh, np.sqrt(max(rel, 0))))
        halves.append((C1, C2))
        print(f"  {rows[-1][0]:<12s} {rows[-1][1]:>12.1f} {rows[-1][2]:>9.4f} "
              f"{sh:>11.4f} {rows[-1][4]:>8.4f}", flush=True)
    R = np.array([[r[1], r[2], r[3], r[4]] for r in rows])
    print(f"  {'mean':<12s} {R[:,0].mean():>12.1f} {R[:,1].mean():>9.4f} "
          f"{R[:,2].mean():>11.4f} {R[:,3].mean():>8.4f}")

    print(f"\n  a single subject's FC correlates {R[:,1].mean():.4f} with the group;")
    print(f"  its own split-half reliability implies a ceiling of {R[:,3].mean():.4f}")
    print(f"  so fitting ONE subject is bounded near {R[:,3].mean():.3f}, and the group "
          f"target\n  sits {R[:,1].mean():.3f} away from any individual by construction")

    print(f"\n  best rank-r approximation, Pearson on off-diagonal edges:")
    print(f"  {'rank':>6s} {'one subject':>12s} {'group':>9s}")
    ev_s, V_s = np.linalg.eigh(fc_of(nki.load_subject(files[0])[verts].astype(np.float64)))
    o = np.argsort(np.abs(ev_s))[::-1]; ev_s, V_s = ev_s[o], V_s[:, o]
    Cs = fc_of(nki.load_subject(files[0])[verts].astype(np.float64))
    ev_g, V_g = np.linalg.eigh(G)
    o = np.argsort(np.abs(ev_g))[::-1]; ev_g, V_g = ev_g[o], V_g[:, o]
    for r in (5, 12, 20, 47, 93):
        As = (V_s[:, :r] * ev_s[:r]) @ V_s[:, :r].T
        Ag = (V_g[:, :r] * ev_g[:r]) @ V_g[:, :r].T
        print(f"  {r:>6d} {cor_off(As, Cs, iu):>12.4f} {cor_off(Ag, G, iu):>9.4f}")

    # per-vertex within-subject reliability, averaged over the subjects loaded
    print(f"\n  per-vertex within-subject reliability (split-half, mean over "
          f"{len(halves)} subjects):")
    accs = np.zeros(n)
    for C1, C2 in halves:
        m = ~np.eye(n, dtype=bool)
        for i in range(n):
            x, y = C1[i][m[i]], C2[i][m[i]]
            x = x - x.mean(); y = y - y.mean()
            accs[i] += float(x @ y / max(np.linalg.norm(x) * np.linalg.norm(y), 1e-30))
    accs /= len(halves)
    print(f"    mean {accs.mean():+.4f}, median {np.median(accs):+.4f}, "
          f"range {accs.min():+.3f} to {accs.max():+.3f}")
    for thr in (0.0, 0.2, 0.3, 0.4, 0.5):
        k = accs > thr
        print(f"    reliability > {thr:.1f}: {int(k.sum()):>4d} of {n} vertices "
              f"({k.mean():.0%})")
    np.savez("data/cache/subject_reliability.npz", rel=accs, sub=sub)
    print(f"  wrote data/cache/subject_reliability.npz")


if __name__ == "__main__":
    main()
