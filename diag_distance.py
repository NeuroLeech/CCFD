"""Where in the cortex does the fit fail? Predictive accuracy against distance from drive.

The sensory model plateaus near r = 0.6, so roughly two thirds of the edge variance is
unaccounted for, and a single number cannot say whether that is spread evenly over the
sheet or concentrated somewhere. This asks the simplest spatial version of the question:
for each vertex, how well does the model reproduce that vertex's whole FC profile, and how
far is it from the nearest driven piece?

Per-vertex accuracy is Spearman between the model's and the target's FC profile over a
common random subset, so every vertex is scored against the same set of partners.
Distance is white-surface geodesic to the nearest driven vertex, computed in one pass with
a zero-weight super-source rather than one Dijkstra per seed.

Target and model profile strength are reported alongside, because accuracy that simply
tracks how much correlation there is to predict means something different from accuracy
that falls where the model stops reaching.

  python diag_distance.py --tag sel_sensory
"""
import os, argparse
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra
from scipy.stats import rankdata

from mesh_cache import load_cortex
import fc_score
from paths import RESULTS
import subparcels, ladder


def distance_to_drive(cortex, driven_mask):
    """White-surface geodesic (mm) from every vertex to the nearest driven vertex."""
    G, _ = ladder._white_graph(cortex)
    n = cortex.nV
    seeds = np.flatnonzero(driven_mask)
    # one extra node joined to every seed at zero cost: one Dijkstra, not one per seed
    G = G.tocoo()
    rows = np.r_[G.row, np.full(len(seeds), n), seeds]
    cols = np.r_[G.col, seeds, np.full(len(seeds), n)]
    dat = np.r_[G.data, np.zeros(len(seeds)), np.zeros(len(seeds))]
    A = sp.coo_matrix((dat, (rows, cols)), shape=(n + 1, n + 1)).tocsr()
    return dijkstra(A, indices=n)[:n]


def zscore_rows(X):
    X = X - X.mean(1, keepdims=True)
    return X / np.maximum(X.std(1, keepdims=True), 1e-12)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", default="sel_sensory")
    ap.add_argument("--nvert", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", type=int, default=50)
    a = ap.parse_args()

    c = load_cortex("fsaverage5", verbose=False)
    # default_target, NOT FCTarget(c): the bare constructor loads the
    # pre-double-centred file with centre='none', which centres the target and
    # leaves the model un-centred - the exact asymmetry f409535 fixed in the
    # scoring path. A diagnostic run that way describes a matrix nobody fits.
    t = fc_score.default_target(c, verbose=False)
    labels, tags = subparcels.split_parcels(c, subparcels.SENSORY, a.split, verbose=False)
    driven = labels >= 0
    print(f"  {int(driven.sum())} driven vertices in {len(tags)} pieces")

    d_all = distance_to_drive(c, driven)
    d = d_all[t.cols]                                    # aligned to the FC vertex order

    rng = np.random.default_rng(a.seed)
    v = np.sort(rng.choice(t.nV, a.nvert, replace=False))
    print(f"  {len(v)} random vertices, distance to drive "
          f"{d[v].min():.0f}-{d[v].max():.0f} mm")

    frames = np.load(os.path.join(RESULTS, f"frames_{a.tag}.npy"), mmap_mode="r")
    Z, _ = t.model_z(np.asarray(frames))
    Zs = zscore_rows(Z[v].astype(np.float64))
    M = (Zs @ Zs.T) / Zs.shape[1]
    T = np.asarray(t.target_fc()[np.ix_(v, v)], np.float64)

    off = ~np.eye(len(v), dtype=bool)
    acc = np.empty(len(v))
    for i in range(len(v)):
        m = off[i]
        acc[i] = np.corrcoef(rankdata(M[i][m]), rankdata(T[i][m]))[0, 1]
    tstr = np.array([np.abs(T[i][off[i]]).mean() for i in range(len(v))])
    mstr = np.array([np.abs(M[i][off[i]]).mean() for i in range(len(v))])

    dv = d[v]
    edges = np.array([0, 5, 10, 20, 30, 40, 60, 80, 120, 200])
    print(f"\n  {'distance (mm)':>16s} {'n':>5s} {'accuracy':>9s} "
          f"{'|target FC|':>12s} {'|model FC|':>11s}")
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (dv >= lo) & (dv < hi)
        if m.sum() < 5:
            continue
        print(f"  {lo:6.0f} - {hi:<7.0f} {int(m.sum()):5d} {acc[m].mean():+9.3f} "
              f"{tstr[m].mean():12.4f} {mstr[m].mean():11.4f}")
    print(f"\n  overall accuracy {acc.mean():+.3f};  "
          f"corr(accuracy, distance) = {np.corrcoef(acc, dv)[0,1]:+.3f}")
    print(f"  driven vertices (d = 0): {acc[dv == 0].mean():+.3f} "
          f"({int((dv == 0).sum())} of {len(v)})")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].scatter(dv, acc, s=6, alpha=0.35, color="#3b6ea5", edgecolors="none")
    ctr = 0.5 * (edges[:-1] + edges[1:])
    mean = [acc[(dv >= lo) & (dv < hi)].mean() if ((dv >= lo) & (dv < hi)).sum() >= 5
            else np.nan for lo, hi in zip(edges[:-1], edges[1:])]
    ax[0].plot(ctr, mean, "o-", color="#c1442e", lw=2, label="binned mean")
    ax[0].set_xlabel("geodesic distance to nearest driven vertex (mm)")
    ax[0].set_ylabel("per-vertex FC accuracy (Spearman)")
    ax[0].axhline(0, color="0.6", lw=0.8)
    ax[0].legend(frameon=False, fontsize=9)
    ax[0].set_title(f"{a.tag}: accuracy vs distance from drive", fontsize=10)
    ax[1].plot(ctr, [tstr[(dv >= lo) & (dv < hi)].mean()
                     for lo, hi in zip(edges[:-1], edges[1:])], "o-",
               color="#444444", label="|target FC|")
    ax[1].plot(ctr, [mstr[(dv >= lo) & (dv < hi)].mean()
                     for lo, hi in zip(edges[:-1], edges[1:])], "o-",
               color="#c1442e", label="|model FC|")
    ax[1].set_xlabel("geodesic distance to nearest driven vertex (mm)")
    ax[1].set_ylabel("mean |FC| of the vertex's profile")
    ax[1].legend(frameon=False, fontsize=9)
    ax[1].set_title("is there signal there to predict?", fontsize=10)
    for b in ax:
        b.spines[["top", "right"]].set_visible(False)
    path = os.path.join(RESULTS, f"diag_distance_{a.tag}.png")
    fig.tight_layout(); fig.savefig(path, dpi=140)
    print(f"  wrote {path}")


if __name__ == "__main__":
    main()
