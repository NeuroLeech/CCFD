"""Model FC against empirical FC, edge by edge.

The score is a single correlation over 2M sampled edges, which says nothing about the
SHAPE of the relationship - whether the model is well calibrated, compresses the range,
saturates, or misses one tail. This draws it.

Both sides are the double-centred values actually scored, on the target's own fixed edge
sample. Density is drawn as a hexbin because two million points cannot be scattered.

  python plot_edges.py --tags sc2_sen,sc2_sub47
"""
import os, argparse
import numpy as np
from scipy.stats import spearmanr

from mesh_cache import load_cortex
from paths import RESULTS
import fc_score


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tags", default="sc2_sen,sc2_sub47")
    ap.add_argument("--labels", default="sensory,subcortical")
    ap.add_argument("--nplot", type=int, default=400_000)
    ap.add_argument("--out", default="edges_scatter.png")
    a = ap.parse_args()
    tags = [s for s in a.tags.split(",") if s]
    labs = [s for s in a.labels.split(",") if s]
    if len(labs) != len(tags):
        labs = tags

    c = load_cortex("fsaverage5", verbose=False)
    t = fc_score.default_target(c, verbose=False)
    FC = t.target_fc()
    y = np.asarray(FC[t.i, t.j], np.float64)
    del FC
    print(f"  {len(y)} sampled edges; target range {y.min():+.3f} to {y.max():+.3f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(tags), figsize=(5.6 * len(tags), 5.2), squeeze=False)
    rng = np.random.default_rng(0)
    sel = np.sort(rng.choice(len(y), min(a.nplot, len(y)), replace=False))

    for k, (tag, lab) in enumerate(zip(tags, labs)):
        F = np.asarray(np.load(os.path.join(RESULTS, f"frames_{tag}.npy"), mmap_mode="r"))
        Z, _ = t.model_z(F); del F
        x = np.asarray(t.model_edges(Z=Z)[0], np.float64)
        pe = float(np.corrcoef(x, y)[0, 1])
        sp = float(spearmanr(x, y).statistic)
        b = float(np.polyfit(y, x, 1)[0])
        ax = axes[0, k]
        hb = ax.hexbin(y[sel], x[sel], gridsize=110, bins="log", cmap="magma",
                       mincnt=1, linewidths=0)
        lim = (min(y[sel].min(), x[sel].min()), max(y[sel].max(), x[sel].max()))
        ax.plot(lim, lim, color="0.75", lw=1.0, ls="--", label="y = x")
        xs = np.linspace(y.min(), y.max(), 50)
        ax.plot(xs, np.polyval(np.polyfit(y, x, 1), xs), color="#3ec1d3", lw=1.6,
                label=f"fit, slope {b:.2f}")
        # binned mean of the model within deciles of the target
        # x-coordinate must be the empirical MEAN in the bin, not the midpoint of the
        # quantile edges. The top bin spans q95 to the maximum and is strongly skewed, so
        # its midpoint sits far above its mean and the point falls below the diagonal for
        # that reason alone - which manufactures an apparent saturation that is not there.
        qs = np.quantile(y, np.linspace(0, 1, 21))
        ctr, mn = [], []
        for lo, hi in zip(qs[:-1], qs[1:]):
            m = (y >= lo) & (y < hi)
            if m.sum() > 50:
                ctr.append(y[m].mean()); mn.append(x[m].mean())
        ax.plot(ctr, mn, "o-", color="#7CFC00", ms=3.5, lw=1.4, label="binned mean")
        ax.set_xlabel("empirical FC (double-centred)")
        if k == 0:
            ax.set_ylabel("model FC (double-centred)")
        ax.set_title(f"{lab}   pearson {pe:+.4f}   spearman {sp:+.4f}", fontsize=11)
        ax.legend(frameon=False, fontsize=9, loc="upper left")
        ax.spines[["top", "right"]].set_visible(False)
        fig.colorbar(hb, ax=ax, label="edges per hexagon", shrink=0.85)
        print(f"  {lab:<14s} pearson {pe:+.4f}  spearman {sp:+.4f}  slope {b:.3f}  "
              f"model sd {x.std():.4f} vs target sd {y.std():.4f} "
              f"(ratio {x.std()/y.std():.3f})", flush=True)
        del Z, x
    fig.tight_layout()
    p = os.path.join(RESULTS, a.out)
    fig.savefig(p, dpi=140)
    print(f"  wrote {p}")


if __name__ == "__main__":
    main()
