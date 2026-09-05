"""Per-vertex noise ceiling: how much of each vertex's FC is reproducible at all?

The failure map is spatially structured and the worst parcels - pOFC, EC, PeEc, TPOJ,
ventral temporal - are where fMRI has least signal. Some of the residual is therefore not
the model's to explain. This measures how much.

tSNR is NOT available: the NKI preprocessed surfaces are demeaned, so the per-vertex mean
is ~1e-8 and mean/std is identically zero. Two things that ARE available and bear more
directly on the question:

TEMPORAL SD. Signal amplitude per vertex in the units the data ships in. A dropout region
has little BOLD variance, so this is a proxy for acquisition quality, though it is not
normalised by anything.

PER-VERTEX RELIABILITY, which is the quantity that actually bounds the model. Split the 99
subjects into two groups, build a group FC from each, and correlate every vertex's FC
PROFILE between the two. No subject is shared, so this is the ceiling on how well any
model can predict that vertex's connectivity - the per-vertex analogue of what
reliability.py computes for the matrix as a whole.

Both are then set against the model's per-vertex accuracy from diag_maps, and the score is
recomputed on vertices above a reliability threshold, which is the "predict only the good
vertices" question.

  python vertex_quality.py --tag whiten
"""
import os, argparse
import numpy as np
from scipy.stats import rankdata

from mesh_cache import load_cortex
from paths import RESULTS, CACHE
import fc_score, fc_group_nki as nki


def _load(src, idx):
    """One subject's (nV, T) timeseries on the target's vertices, from either source.

    `src` is a nilearn path or an rbc.Run. The two releases are different preprocessing of
    overlapping people, so which one is used has to follow the TARGET - a reliability built
    from nilearn subjects and cached under the RBC target's vertex count would be the same
    silent mismatch the hardcoded 9217 filename in band_fail was."""
    if isinstance(src, str):
        return nki.load_subject(src)[idx]
    import rbc
    return rbc.load(src, verbose=False)[0][idx]


def group_profiles(files, idx, part, metric="spearman"):
    """Group-average FC profile for every vertex against a common partner set.

    -> (nV, npart). Accumulated in Fisher z exactly as fc_group_nki does, so this is the
    same estimator as the target, restricted to a column subset."""
    acc = np.zeros((len(idx), len(part)), np.float64)
    ppos = np.searchsorted(idx, part)
    for k, p in enumerate(files, 1):
        X = _load(p, idx)
        Z = rankdata(X, axis=1).astype(np.float32) if metric == "spearman" else X.copy()
        Z -= Z.mean(1, keepdims=True)
        sd = Z.std(1, keepdims=True); sd[sd == 0] = 1.0
        Z /= sd
        C = (Z @ Z[ppos].T) / Z.shape[1]
        np.clip(C, -0.9999, 0.9999, out=C)
        acc += np.arctanh(C)
        if k % 20 == 0:
            print(f"    {k}/{len(files)} subjects", flush=True)
    return np.tanh(acc / len(files))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", default="whiten")
    ap.add_argument("--npart", type=int, default=1500,
                    help="common partner set each vertex's profile is taken against")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--source", default="nilearn", choices=("nilearn", "rbc"),
                    help="which release the split-half subjects come from. This must "
                         "match whatever the TARGET was built from: the reliability is a "
                         "ceiling on that target, and the cache is keyed only by vertex "
                         "count, which both releases share")
    ap.add_argument("--cohort", type=int, default=100)
    a = ap.parse_args()

    c = load_cortex("fsaverage5", verbose=False)
    t = fc_score.default_target(c, verbose=True)
    if a.source == "rbc":
        import json, rbc
        subs = json.load(open(os.path.join(
            CACHE, f"rbc_cohort_{a.cohort}_seed0.json")))["subjects"]
        files = rbc.cohort_runs(subs, specs=(("rest", "645"),))[("rest", "645")]
    else:
        files = nki.subject_files("left")
    print(f"  {len(files)} subjects from {a.source}")

    # the source belongs in the key: both releases can produce the same vertex count
    sfx = "" if a.source == "nilearn" else f"_{a.source}"
    cache = os.path.join(CACHE, f"vertex_quality_{a.npart}_{a.seed}_{t.nV}{sfx}.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        rel, sd_map, part = z["rel"], z["sd"], z["part"]
        print(f"  loaded {cache}")
    else:
        idx = t.vertices                       # full-mesh ids the FC covers, sorted
        rng = np.random.default_rng(a.seed)
        part = np.sort(rng.choice(idx, a.npart, replace=False))
        which = rng.permutation(len(files)) % 2
        halves = [[f for f, wv in zip(files, which) if wv == h] for h in (0, 1)]
        print(f"  splitting {len(halves[0])} / {len(halves[1])} subjects, "
              f"{a.npart} common partners")
        Ps = []
        for h in (0, 1):
            print(f"  half {h+1}:")
            Ps.append(group_profiles(halves[h], idx, part))
        A, B = Ps
        # a vertex must not enter its own profile
        ppos = np.searchsorted(idx, part)
        keep = np.ones((len(idx), len(part)), bool)
        for j, v in enumerate(ppos):
            keep[v, j] = False
        rel = np.empty(len(idx))
        for i in range(len(idx)):
            k = keep[i]
            x, y = A[i][k], B[i][k]
            x = x - x.mean(); y = y - y.mean()
            d = np.linalg.norm(x) * np.linalg.norm(y)
            rel[i] = float(x @ y / d) if d > 0 else 0.0
        print("  temporal SD map:")
        sd_map = np.zeros(len(idx))
        for k, p in enumerate(files, 1):
            sd_map += _load(p, idx).std(1)
            if k % 40 == 0:
                print(f"    {k}/{len(files)}", flush=True)
        sd_map /= len(files)
        np.savez(cache, rel=rel, sd=sd_map, part=part)
        print(f"  wrote {cache}")

    print(f"\n  per-vertex reliability (between disjoint halves of the subjects):")
    print(f"    mean {rel.mean():+.4f}, median {np.median(rel):+.4f}, "
          f"range {rel.min():+.3f} to {rel.max():+.3f}")
    print(f"  temporal SD: median {np.median(sd_map):.4g}, "
          f"range {sd_map.min():.3g} to {sd_map.max():.3g}")
    print(f"  reliability vs temporal SD: r = {np.corrcoef(rel, sd_map)[0,1]:+.3f}")

    mp = os.path.join(RESULTS, f"diag_maps_{a.tag}.npz")
    if not os.path.exists(mp):
        print(f"\n  {mp} not found - run diag_maps.py --tag {a.tag} for the comparison")
        return
    acc = np.load(mp)["acc"]
    print(f"\n  model accuracy vs per-vertex reliability: "
          f"r = {np.corrcoef(acc, rel)[0,1]:+.3f}")
    print(f"  model accuracy vs temporal SD:            "
          f"r = {np.corrcoef(acc, sd_map)[0,1]:+.3f}")

    print(f"\n  {'reliability >':>14s} {'vertices':>9s} {'mean ceiling':>13s} "
          f"{'model accuracy':>15s} {'accuracy/ceiling':>17s}")
    for thr in (0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9):
        m = rel > thr
        if m.sum() < 50:
            continue
        print(f"  {thr:>14.2f} {int(m.sum()):>9d} {rel[m].mean():>13.4f} "
              f"{acc[m].mean():>15.4f} {acc[m].mean()/rel[m].mean():>17.3f}")

    lab = np.asarray(c.lab)[t.cols]
    rows = []
    for p in np.unique(lab):
        if p == 0:
            continue
        m = lab == p
        if m.sum() < 5:
            continue
        nm = c.names[p].replace("_ROI", "") if p < len(c.names) else str(p)
        rows.append((float(rel[m].mean()), float(acc[m].mean()), int(m.sum()), nm))
    rows.sort()
    print(f"\n  {'parcel':<12s} {'reliability':>12s} {'accuracy':>10s}   lowest 12 by "
          f"reliability")
    for r_, a_, n_, nm in rows[:12]:
        print(f"  {nm:<12s} {r_:>12.4f} {a_:>10.4f}")
    np.savez(os.path.join(RESULTS, f"vertex_quality_{a.tag}.npz"),
             rel=rel, sd=sd_map, acc=acc, cols=t.cols)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from render_regimes import _proj
    from plot_fc_map import surface_row
    proj = _proj(c.V, c.F)
    rows_f = [("per-vertex\nreliability", rel, "viridis",
               (float(np.percentile(rel, 2)), float(np.percentile(rel, 98)))),
              ("temporal SD", sd_map, "magma",
               (float(np.percentile(sd_map, 2)), float(np.percentile(sd_map, 98)))),
              ("model accuracy", acc, "viridis",
               (float(np.percentile(acc, 2)), float(np.percentile(acc, 98)))),
              ("accuracy MINUS\nreliability", acc - rel, "RdBu_r",
               (-0.4, 0.4))]
    fig = plt.figure(figsize=(3.6 * len(proj), 2.5 * len(rows_f)))
    gs = fig.add_gridspec(len(rows_f), len(proj), hspace=0.04, wspace=0.02)
    for r, (lb, v, cm, lims) in enumerate(rows_f):
        surface_row(fig, gs, r, proj, v, c, t.cols, cm, lims, lb)
    fig.suptitle("what is reproducible, and what the model reaches", fontsize=10)
    pth = os.path.join(RESULTS, f"vertex_quality_{a.tag}.png")
    fig.savefig(pth, dpi=135, bbox_inches="tight")
    print(f"\n  wrote {pth}")


if __name__ == "__main__":
    main()
