"""Where the model's edge distribution departs from the empirical one, and why.

The scalar score hides the shape. The scatter shows the model tracking the diagonal up to
empirical FC ~0.2 and then flattening, so the strongest empirical edges are systematically
under-produced, while the overall edge spread is 1.4-1.6x too WIDE. Those two facts
together mean too much variance in the middle of the range and too little at the top.

This asks what the saturating edges are: how long, how far from the drive, and whether the
compression is a property of the model's own distribution (a ceiling it cannot exceed) or
of the mapping (it produces large values, just not for the right edges).

  python saturation.py --tags sc2_sen,sc2_sub47
"""
import os, argparse
import numpy as np

from mesh_cache import load_cortex
from paths import RESULTS
import fc_score, subparcels, units


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tags", default="sc2_sen,sc2_sub47")
    ap.add_argument("--labels", default="sensory,subcortical")
    ap.add_argument("--regions", default="sensory,subcortical",
                    help="driven set per tag, for distance-to-drive")
    ap.add_argument("--splits", default="50,40")
    ap.add_argument("--nvert", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    tags = a.tags.split(","); labs = a.labels.split(",")
    regs = a.regions.split(","); spl = [int(v) for v in a.splits.split(",")]

    c = load_cortex("fsaverage5", verbose=False)
    t = fc_score.default_target(c, verbose=False)
    rng = np.random.default_rng(a.seed)
    v = np.sort(rng.choice(t.nV, a.nvert, replace=False))
    n = len(v); iu = np.triu_indices(n, 1)
    G = np.asarray(t.target_fc()[np.ix_(v, v)], np.float64)
    G = G - G.mean(0, keepdims=True) - G.mean(1, keepdims=True) + G.mean()
    D = units.vertex_geodesic(c, t.cols[v])[:, t.cols[v]]
    y, dd = G[iu], D[iu]
    print(f"  {n} vertices, {len(y)} edges; empirical FC {y.min():+.3f} to {y.max():+.3f}")

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 3, figsize=(16.5, 9))

    qs = np.linspace(0.001, 0.999, 400)
    for k, (tag, lab, rg, sp) in enumerate(zip(tags, labs, regs, spl)):
        F = np.asarray(np.load(os.path.join(RESULTS, f"frames_{tag}.npy"), mmap_mode="r"))
        Z, _ = t.model_z(F); del F
        Zs = Z[v].astype(np.float64)
        Zs -= Zs.mean(1, keepdims=True)
        Zs /= np.maximum(Zs.std(1, keepdims=True), 1e-12)
        M = (Zs @ Zs.T) / Zs.shape[1]
        M = M - M.mean(0, keepdims=True) - M.mean(1, keepdims=True) + M.mean()
        x = M[iu]
        col = ["#c1442e", "#3b6ea5"][k]

        # distance to the nearest driven vertex, for this tag's own driven set
        parcels, total = subparcels.region_set(c, rg, sp)
        lab_, tg_ = subparcels.split_parcels(c, parcels, total, verbose=False)
        from diag_distance import distance_to_drive
        dv = distance_to_drive(c, lab_ >= 0)[t.cols[v]]
        ed = np.minimum(dv[iu[0]], dv[iu[1]])      # nearer endpoint to the drive

        # 1. marginal distributions
        ax[0,0].hist(x, bins=200, range=(-0.5, 1.05), histtype="step", density=True,
                     color=col, label=f"{lab} model")
        # 2. Q-Q
        ax[0,1].plot(np.quantile(y, qs), np.quantile(x, qs), color=col, lw=1.8, label=lab)
        # 3. conditional mean, fine bins of the empirical
        qq = np.quantile(y, np.linspace(0, 1, 41))
        ctr, mn, hi = [], [], []
        for lo, h in zip(qq[:-1], qq[1:]):
            m = (y >= lo) & (y < h)
            if m.sum() > 100:
                # empirical MEAN in the bin, not the midpoint of the quantile edges;
                # the top bin is skewed and its midpoint sits far above its mean
                ctr.append(y[m].mean()); mn.append(x[m].mean())
                hi.append(np.quantile(x[m], 0.95))
        ax[0,2].plot(ctr, mn, "o-", color=col, ms=3, lw=1.6, label=f"{lab} mean")
        ax[0,2].plot(ctr, hi, ":", color=col, lw=1.2, label=f"{lab} 95th pct")

        # 4. accuracy within edge-length bins
        eb = np.array([0,10,20,30,40,60,80,120,250])
        cc, rr = [], []
        for lo, h in zip(eb[:-1], eb[1:]):
            m = (dd >= lo) & (dd < h)
            if m.sum() > 500:
                cc.append(0.5*(lo+h))
                rr.append(float(np.corrcoef(x[m], y[m])[0,1]))
        ax[1,0].plot(cc, rr, "o-", color=col, lw=1.8, label=lab)

        # 5. accuracy vs distance from the DRIVE
        db = np.array([0,5,10,20,30,45,60,90,200])
        cc2, rr2 = [], []
        for lo, h in zip(db[:-1], db[1:]):
            m = (ed >= lo) & (ed < h)
            if m.sum() > 500:
                cc2.append(0.5*(lo+h))
                rr2.append(float(np.corrcoef(x[m], y[m])[0,1]))
        ax[1,1].plot(cc2, rr2, "o-", color=col, lw=1.8, label=lab)

        # 6. where do the STRONG empirical edges live, and what does the model give them?
        top = y >= np.quantile(y, 0.99)
        print(f"\n  {lab}: top 1% of empirical edges (n={int(top.sum())}, "
              f"empirical mean {y[top].mean():+.3f})")
        print(f"    model gives them {x[top].mean():+.3f}  "
              f"(model's own top 1% is {np.quantile(x, 0.99):+.3f})")
        print(f"    their length: median {np.median(dd[top]):.0f} mm; "
              f"distance to drive: median {np.median(ed[top]):.0f} mm")
        print(f"    model max {x.max():+.3f} vs empirical max {y.max():+.3f}; "
              f"sd ratio {x.std()/y.std():.3f}")
        ax[1,2].plot(cc2, [float(np.mean(x[(ed>=lo)&(ed<h)])) for lo,h in
                           zip(db[:-1], db[1:]) if ((ed>=lo)&(ed<h)).sum()>500],
                     "o-", color=col, lw=1.8, label=f"{lab} model")
        del Z, Zs, M

    ax[0,0].hist(y, bins=200, range=(-0.5,1.05), histtype="step", density=True,
                 color="k", lw=1.6, label="empirical")
    ax[0,0].set_xlabel("FC"); ax[0,0].set_ylabel("density")
    ax[0,0].set_title("edge distributions", fontsize=10); ax[0,0].set_yscale("log")
    lim = (-0.45, 1.05)
    ax[0,1].plot(lim, lim, "--", color="0.6", lw=1)
    ax[0,1].set_xlabel("empirical quantile"); ax[0,1].set_ylabel("model quantile")
    ax[0,1].set_title("Q-Q: is the model's RANGE the problem?", fontsize=10)
    ax[0,2].plot(ctr, ctr, "--", color="0.6", lw=1)
    ax[0,2].set_xlabel("empirical FC"); ax[0,2].set_ylabel("model FC")
    ax[0,2].set_title("conditional mean and 95th pct", fontsize=10)
    ax[1,0].set_xlabel("edge length (mm)"); ax[1,0].set_ylabel("pearson within bin")
    ax[1,0].set_title("accuracy by edge length", fontsize=10)
    ax[1,1].set_xlabel("distance of nearer endpoint to drive (mm)")
    ax[1,1].set_ylabel("pearson within bin")
    ax[1,1].set_title("accuracy by distance from the DRIVE", fontsize=10)
    ax[1,2].set_xlabel("distance to drive (mm)"); ax[1,2].set_ylabel("mean model FC")
    ax[1,2].set_title("model FC amplitude vs distance from drive", fontsize=10)
    for b in ax.ravel():
        b.legend(frameon=False, fontsize=8); b.spines[["top","right"]].set_visible(False)
    fig.tight_layout()
    p = os.path.join(RESULTS, "saturation.png")
    fig.savefig(p, dpi=135); print(f"\n  wrote {p}")


if __name__ == "__main__":
    main()
