"""Do the sensory systems anticorrelate in the model, and do they in the data?

The edge-length diagnostic shows the model's FC never goes negative: the target's mean FC
crosses zero around 80-100 mm and reaches -0.03 at long range, while the model stays at
+0.03. A linear medium driven by an additive source injects depth of one sign and spreads
it, so sustained anticorrelation has to come from the input's cross-spectrum rather than
from the dynamics - and whether the solve actually uses that freedom is checkable.

Piece timecourses are the field averaged over each piece's vertices per frame. The
modality timecourses average those within somatomotor, visual and auditory. The same
quantities are computed from the empirical target (mean target FC between the same vertex
sets) so the model has something to be wrong against.

  python diag_modality.py --tag sel_sensory
"""
import os, argparse
import numpy as np

from mesh_cache import load_cortex
from fc_score import FCTarget
from paths import RESULTS
import subparcels

GROUPS = [("SOM", subparcels.SOM, "#c1442e"),
          ("VIS", subparcels.VIS, "#3b6ea5"),
          ("AUD", subparcels.AUD, "#2e8b57")]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", default="sel_sensory")
    ap.add_argument("--split", type=int, default=50)
    a = ap.parse_args()

    c = load_cortex("fsaverage5", verbose=False)
    t = FCTarget(c, verbose=False)
    labels, tags = subparcels.split_parcels(c, subparcels.SENSORY, a.split, verbose=False)
    K = len(tags)
    parcel_of = np.array([int(s.split("_")[0]) for s in tags])
    mod = np.array([next((g for g, ps, _ in GROUPS if p in ps), "?")
                    for p in parcel_of])
    order = np.concatenate([np.flatnonzero(mod == g) for g, _, _ in GROUPS])
    print("  pieces per modality: "
          + ", ".join(f"{g} {int((mod == g).sum())}" for g, _, _ in GROUPS))

    frames = np.asarray(np.load(os.path.join(RESULTS, f"frames_{a.tag}.npy"),
                                mmap_mode="r"))[t.burn:]
    area = np.asarray(c.A, float)
    X = np.empty((K, frames.shape[0]))
    for k in range(K):
        q = np.flatnonzero(labels == k)
        w = area[q] / area[q].sum()
        X[k] = frames[:, q] @ w
    Xz = (X - X.mean(1, keepdims=True)) / np.maximum(X.std(1, keepdims=True), 1e-12)
    Cm = (Xz @ Xz.T) / Xz.shape[1]

    # the empirical analogue: mean target FC between the same vertex sets
    pos = {int(o): i for i, o in enumerate(t.vertices)}
    vidx = [np.array([pos[int(c.old[q])] for q in np.flatnonzero(labels == k)
                      if int(c.old[q]) in pos]) for k in range(K)]
    FC = t.target_fc()
    Ct = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            if len(vidx[i]) and len(vidx[j]):
                Ct[i, j] = float(np.asarray(FC[np.ix_(vidx[i], vidx[j])]).mean())

    # both sides must be the same quantity: mean piece-to-piece correlation within or
    # between groups, off-diagonal. Correlating group-MEAN timecourses instead would put
    # a trivial 1.0 on the model diagonal and compare it against a target block mean.
    def block(C, gi, gj):
        A = C[np.ix_(mod == gi, mod == gj)]
        if gi == gj:
            n = A.shape[0]
            return float(A[~np.eye(n, dtype=bool)].mean())
        return float(A.mean())

    print(f"\n  mean piece-to-piece correlation, model / target:")
    print(f"  {'':6s}" + "".join(f"{g:>19s}" for g, _, _ in GROUPS))
    for g, _, _ in GROUPS:
        row = f"  {g:6s}"
        for h, _, _ in GROUPS:
            row += f"   {block(Cm, g, h):+.3f} / {block(Ct, g, h):+.3f}"
        print(row)

    off = ~np.eye(K, dtype=bool)
    print(f"\n  piece-level FC, model: mean {Cm[off].mean():+.3f}, "
          f"{100*(Cm[off] < 0).mean():.1f}% negative, min {Cm[off].min():+.3f}")
    print(f"  piece-level FC, target: mean {Ct[off].mean():+.3f}, "
          f"{100*(Ct[off] < 0).mean():.1f}% negative, min {Ct[off].min():+.3f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 5.4))
    for k, (C, name) in enumerate(((Cm, "model"), (Ct, "target (group NKI)"))):
        A = C[np.ix_(order, order)]
        lim = float(np.abs(A[~np.eye(K, dtype=bool)]).max())
        im = ax[k].imshow(A, cmap="RdBu_r", vmin=-lim, vmax=lim)
        fig.colorbar(im, ax=ax[k], fraction=0.046)
        b = 0
        for g, _, col in GROUPS:
            n = int((mod == g).sum())
            ax[k].add_patch(plt.Rectangle((b - .5, b - .5), n, n, fill=False,
                                          ec=col, lw=2))
            ax[k].text(b + n / 2 - .5, -2.2, g, color=col, ha="center", fontsize=11)
            b += n
        ax[k].set_title(f"{name}: 47 sensory pieces", fontsize=10)
        ax[k].set_xticks([]); ax[k].set_yticks([])
    p = os.path.join(RESULTS, f"diag_modality_{a.tag}.png")
    fig.tight_layout(); fig.savefig(p, dpi=140)
    print(f"  wrote {p}")


if __name__ == "__main__":
    main()
