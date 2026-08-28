"""Is the fit failing at short range, long range, or evenly?

The per-vertex version of this question said the failure is not spatial - accuracy barely
moves with distance from the drive, and is no better at the driven vertices themselves.
That leaves the possibility that it is the LENGTH of the connection rather than the
location of the vertex: a wave medium has an obvious characteristic scale, and a fit that
gets nearby pairs right and distant pairs wrong would look uniform when scored per vertex.

Edges are binned by white-surface geodesic distance between their endpoints, and scored
within each bin, so each number is a Spearman over pairs of comparable length.

  python diag_edges.py --tag sel_sensory
"""
import os, argparse
import numpy as np
from scipy.stats import spearmanr

from mesh_cache import load_cortex
from fc_score import FCTarget
from paths import RESULTS
import units


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", default="sel_sensory")
    ap.add_argument("--nvert", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    c = load_cortex("fsaverage5", verbose=False)
    t = FCTarget(c, verbose=False)
    rng = np.random.default_rng(a.seed)
    v = np.sort(rng.choice(t.nV, a.nvert, replace=False))

    D = units.vertex_geodesic(c, t.cols[v])[:, t.cols[v]]      # (n, n) mm
    frames = np.load(os.path.join(RESULTS, f"frames_{a.tag}.npy"), mmap_mode="r")
    Z, _ = t.model_z(np.asarray(frames))
    Zs = Z[v].astype(np.float64)
    Zs = (Zs - Zs.mean(1, keepdims=True)) / np.maximum(Zs.std(1, keepdims=True), 1e-12)
    M = (Zs @ Zs.T) / Zs.shape[1]
    T = np.asarray(t.target_fc()[np.ix_(v, v)], np.float64)

    iu = np.triu_indices(len(v), 1)
    d, m, tg = D[iu], M[iu], T[iu]
    print(f"  {len(d)} edges, lengths {d.min():.0f}-{d.max():.0f} mm\n")
    edges = np.array([0, 10, 20, 30, 40, 60, 80, 100, 130, 160, 250])
    print(f"  {'edge length (mm)':>18s} {'n':>7s} {'accuracy':>9s} "
          f"{'mean target':>12s} {'mean model':>11s}")
    ctr, acc, tm, mm = [], [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        k = (d >= lo) & (d < hi)
        if k.sum() < 200:
            continue
        r = spearmanr(m[k], tg[k]).statistic
        ctr.append(0.5 * (lo + hi)); acc.append(r)
        tm.append(tg[k].mean()); mm.append(m[k].mean())
        print(f"  {lo:7.0f} - {hi:<8.0f} {int(k.sum()):7d} {r:+9.3f} "
              f"{tg[k].mean():+12.4f} {m[k].mean():+11.4f}")
    print(f"\n  all edges together: {spearmanr(m, tg).statistic:+.3f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(ctr, acc, "o-", color="#3b6ea5", lw=2)
    ax[0].axhline(spearmanr(m, tg).statistic, color="0.6", ls="--", lw=1,
                  label="all edges")
    ax[0].set_xlabel("edge length (mm, white-surface geodesic)")
    ax[0].set_ylabel("Spearman(model, target) within bin")
    ax[0].set_title(f"{a.tag}: accuracy by edge length", fontsize=10)
    ax[0].legend(frameon=False, fontsize=9)
    ax[1].plot(ctr, tm, "o-", color="#444444", label="target")
    ax[1].plot(ctr, mm, "o-", color="#c1442e", label="model")
    ax[1].axhline(0, color="0.6", lw=0.8)
    ax[1].set_xlabel("edge length (mm)")
    ax[1].set_ylabel("mean FC in bin")
    ax[1].set_title("distance dependence of FC itself", fontsize=10)
    ax[1].legend(frameon=False, fontsize=9)
    for b in ax:
        b.spines[["top", "right"]].set_visible(False)
    p = os.path.join(RESULTS, f"diag_edges_{a.tag}.png")
    fig.tight_layout(); fig.savefig(p, dpi=140)
    print(f"  wrote {p}")


if __name__ == "__main__":
    main()
