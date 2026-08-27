"""Render vertexwise FC maps on the cortex mesh, empirical next to model.

Both rows are drawn from the same Cortex object and the same projection the run videos
use, and both are indexed through FCTarget's alignment, so anything anatomically out of
place is a real misalignment rather than a plotting difference.

Top rows are mean FC per vertex (weighted degree: how strongly a vertex correlates with
the rest of the sheet). Below that, seed maps for a few Glasser parcels - a seed map is
the sharper check, because a correctly aligned projection puts the empirical peak on the
seed and keeps V1's map in occipital cortex.

  python plot_fc_map.py
  python plot_fc_map.py --seeds 1 24 150 --frames results/frames.npy
"""
import os, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mesh_cache import load_cortex
from render_regimes import _proj
from fc_score import FCTarget
from paths import RESULTS


def seed_vertex(cortex, parcel, cols):
    """Vertex of `parcel` closest to its centroid, restricted to vertices the FC covers.
    -> (index into the aligned FC/model matrices, index into cortex)."""
    inpar = np.flatnonzero(cortex.lab == parcel)
    inpar = np.intersect1d(inpar, cols)
    if not len(inpar):
        raise ValueError(f"parcel {parcel} has no vertices in the FC matrix")
    v = cortex.V[inpar]
    ci = inpar[np.argmin(((v - v.mean(0)) ** 2).sum(1))]
    return int(np.flatnonzero(cols == ci)[0]), int(ci)


def surface_row(fig, gs, r, proj, vals, cortex, cols, cmap, lims, label, mark=None):
    """One map, three views, drawn as the shaded surface rather than one dot per vertex.

    A dot per vertex makes any vertex-level variation read as speckle and hides the
    field; Gouraud shading over the front-facing triangles shows the map itself."""
    from matplotlib.tri import Triangulation
    full = np.zeros(cortex.nV)
    full[cols] = vals
    vmin, vmax = lims
    for k, (xy, vis, nm) in enumerate(proj):
        ax = fig.add_subplot(gs[r, k])
        keep = vis[cortex.F].all(1)                       # front-facing triangles only
        tri = Triangulation(xy[:, 0], xy[:, 1], cortex.F[keep])
        ax.tripcolor(tri, full, shading="gouraud", cmap=cmap, vmin=vmin, vmax=vmax,
                     rasterized=True)
        if mark is not None and vis[mark]:
            ax.scatter(xy[mark, 0], xy[mark, 1], s=60, facecolors="none",
                       edgecolors="k", linewidths=1.6, zorder=5)
        ax.set_xlim(xy[:, 0].min(), xy[:, 0].max())
        ax.set_ylim(xy[:, 1].min(), xy[:, 1].max())
        ax.set_aspect("equal"); ax.axis("off")
        if r == 0:
            ax.set_title(nm, fontsize=10)
        if k == 0:
            ax.text(-0.03, 0.5, f"{label}\n{vmin:+.3f} to {vmax:+.3f}", rotation=90,
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=8.5, family="monospace")


def figure(cortex, target, frames=None, seeds=(1, 24, 150), out="results/fc_maps.png",
           verbose=True):
    """Empirical and (optionally) model maps: mean FC, then one seed map per parcel."""
    proj = _proj(cortex.V, cortex.F)
    cols = target.cols
    E = target.target_fc().astype(np.float32)
    M = target.model_fc(frames) if frames is not None else None
    n = E.shape[0]

    def degree(FC):
        return (FC.sum(1) - np.diag(FC)) / (n - 1)

    # mean FC is strictly positive with a narrow spread, so it gets a percentile scale of
    # its own; seed maps are signed and take symmetric limits from the bulk, not the tail
    def deg_lims(v):
        return float(np.percentile(v, 2)), float(np.percentile(v, 98))

    def seed_lims(v):
        sc = float(np.percentile(np.abs(v), 95))
        return -sc, sc

    dtag = os.path.basename(target.fc_path).split("_")[0]      # sub-MSC01 / group-NKI100
    rows = [(f"{dtag}  mean FC", degree(E), "inferno", deg_lims, None)]
    if M is not None:
        rows.append(("model  mean FC", degree(M), "inferno", deg_lims, None))
    for p in seeds:
        si, ci = seed_vertex(cortex, p, cols)
        name = cortex.names[p].replace("_ROI", "") if p < len(cortex.names) else str(p)
        rows.append((f"{dtag}  seed {name}", E[si], "RdBu_r", seed_lims, ci))
        if M is not None:
            rows.append((f"model  seed {name}", M[si], "RdBu_r", seed_lims, ci))

    fig = plt.figure(figsize=(3.6 * len(proj), 2.5 * len(rows)))
    gs = fig.add_gridspec(len(rows), len(proj), hspace=0.04, wspace=0.02)
    for r, (label, vals, cmap, lim_fn, mark) in enumerate(rows):
        surface_row(fig, gs, r, proj, vals, cortex, cols, cmap, lim_fn(vals), label, mark)
    fig.suptitle(os.path.basename(target.fc_path), fontsize=9, family="monospace")
    fig.savefig(out, dpi=135, bbox_inches="tight")
    plt.close(fig)
    if verbose:
        print(f"  wrote {out}")
        print("  seed map peaks (inflated-surface distance from the seed):")
        for p in seeds:
            si, ci = seed_vertex(cortex, p, cols)
            for tag, FC in ((dtag, E), ("model", M)) if M is not None else ((dtag, E),):
                row = FC[si].copy(); row[si] = -np.inf
                pk = int(np.argmax(row))
                d = float(np.linalg.norm(cortex.V[cols[pk]] - cortex.V[ci]))
                lab = cortex.lab[cols[pk]]
                nm = (cortex.names[lab].replace("_ROI", "")
                      if lab < len(cortex.names) else str(lab))
                print(f"    parcel {p:3d} {tag:12s}: peak r={row[pk]:+.3f} at {d:5.1f} mm, in {nm}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--space", default="fsaverage5")
    ap.add_argument("--fc", default=None, help="FC .npy (default: newest glasser-mask)")
    ap.add_argument("--frames", default="results/frames.npy")
    ap.add_argument("--seeds", nargs="*", type=int, default=[1, 24, 150],
                    help="Glasser parcel ids to seed (1=V1, 24=A1, 150=10r)")
    ap.add_argument("--burn", type=int, default=50)
    ap.add_argument("--centre", default="none", choices=("none", "double"))
    ap.add_argument("--out", default=os.path.join(RESULTS, "fc_maps.png"))
    a = ap.parse_args()

    cortex = load_cortex(a.space, verbose=False)
    target = FCTarget(cortex, fc_path=a.fc, burn=a.burn, centre=a.centre, verbose=True)
    frames = np.load(a.frames) if os.path.exists(a.frames) else None
    figure(cortex, target, frames, a.seeds, a.out)


if __name__ == "__main__":
    main()
