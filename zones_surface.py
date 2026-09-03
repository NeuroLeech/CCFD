"""The zones of influence on the surface, drawn to be read rather than to fit in a stack.

zones.py already writes its maps, but as twelve stacked rows in one figure, at a size that
makes the boundaries guesswork. This draws the three partitions large, in four views, with
the borders explicitly stroked: a zone map is a claim about WHERE one system gives way to
another, and a flat-shaded label field hides exactly that.

Three rows, the same colours throughout:

  attributed         which nucleus supplies the vertex's variance (Shapley share).
                     Model-internal, exact, and with no empirical counterpart.
  model territory    winner-take-all on seed FC from each group's driven vertices.
  empirical          the same rule on the NKI target, using no model quantity at all.

Rows 2 and 3 are the pair that can be compared; row 1 is the thing only the model can
answer, placed above them so the difference is visible rather than described.

Contested vertices - where the top two shares are within the margin - are stippled on the
attributed row. They are 16.6% of the sheet at 10 points, and a hard-edged partition that
does not show them overstates how sharp the zones are.

  python zones_surface.py --tag pr_taper
"""
import os, argparse
import numpy as np

from paths import RESULTS
from zones import NUCLEI

# lateral / medial / dorsal, as every other figure in the project. A ventral view was
# tried and dropped: the medial wall is cut out of this mesh, so from below the rim shows
# edge-on as a thin stripe across the panel that reads as a rendering fault rather than as
# geometry, and the limbic and orbitofrontal zones it would add are already on the medial.
VIEWS = ((8, 180, "lateral"), (8, 0, "medial"), (90, 180, "dorsal"))


def borders(cortex, lab_full, valid, xy, vis):
    """Line segments along mesh edges whose endpoints carry different labels.

    Drawn per view so only front-facing geometry contributes; an edge behind the surface
    would otherwise stroke a border across the middle of a zone it does not touch."""
    F = cortex.F
    segs = []
    for i, j in ((0, 1), (1, 2), (2, 0)):
        a, b = F[:, i], F[:, j]
        m = (valid[a] & valid[b] & vis[a] & vis[b] & (lab_full[a] != lab_full[b]))
        if m.any():
            segs.append(np.stack([xy[a[m]], xy[b[m]]], axis=1))
    return np.concatenate(segs) if segs else np.zeros((0, 2, 2))


def panel(ax, cortex, xy, vis, full, cmap, lims, lab_full=None, valid=None,
          stipple=None):
    from matplotlib.tri import Triangulation
    from matplotlib.collections import LineCollection
    # faces touching a vertex the FC target does not cover are dropped rather than shaded:
    # those 157 vertices carry no label, and filling them with 0 drew them as the first
    # zone in the palette - a patch of LGN wherever the target happened not to reach
    keep = vis[cortex.F].all(1) & valid[cortex.F].all(1)
    tri = Triangulation(xy[:, 0], xy[:, 1], cortex.F[keep])
    ax.tripcolor(tri, full, shading="gouraud", cmap=cmap, vmin=lims[0], vmax=lims[1],
                 rasterized=True)
    if lab_full is not None:
        seg = borders(cortex, lab_full, valid, xy, vis)
        if len(seg):
            ax.add_collection(LineCollection(seg, colors="k", linewidths=0.7,
                                             alpha=0.75, zorder=4))
    if stipple is not None:
        m = stipple & vis
        ax.scatter(xy[m, 0], xy[m, 1], s=0.6, c="w", alpha=0.55, linewidths=0, zorder=5)
    ax.set_xlim(xy[:, 0].min(), xy[:, 0].max())
    ax.set_ylim(xy[:, 1].min(), xy[:, 1].max())
    ax.set_aspect("equal"); ax.axis("off")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", default="pr_taper")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    from mesh_cache import load_cortex
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from render_regimes import _proj

    c = load_cortex("fsaverage5", verbose=False)
    z = np.load(os.path.join(RESULTS, f"zones_{a.tag}.npz"), allow_pickle=True)
    cols_v = z["cols"]
    names = [str(x) for x in z["names"]]
    nG = len(names)
    share = z["shapley"]
    tot = share.sum()
    pct = [share[:, g].sum() / tot for g in range(nG)]

    proj = _proj(c.V, c.F, VIEWS)
    palette = plt.cm.tab10(np.linspace(0, 1, 10))[:nG]
    cmap = ListedColormap(palette)

    valid = np.zeros(c.nV, bool); valid[cols_v] = True

    def to_full(v, fill=0.0):
        f = np.full(c.nV, fill, float); f[cols_v] = v; return f

    rows = [
        ("attributed zone", "Shapley share of diag(C)", z["lab_fit"], z["contested"]),
        ("model territory", "seed FC, winner-take-all", z["lab_mod"], None),
        ("empirical territory", "same rule, NKI 99", z["lab_emp"], None),
    ]

    fig = plt.figure(figsize=(4.6 * len(proj), 3.6 * len(rows) + 1.0))
    gs = fig.add_gridspec(len(rows), len(proj), hspace=0.02, wspace=0.01,
                          top=0.93, bottom=0.10)
    for r, (title, sub, lab, cont) in enumerate(rows):
        labf = to_full(lab, -1).astype(int)
        full = to_full(lab.astype(float), 0.0)
        stip = to_full((cont if cont is not None else np.zeros(len(lab), bool)
                        ).astype(float), 0.0) > 0.5 if cont is not None else None
        for k, (xy, vis, nm) in enumerate(proj):
            ax = fig.add_subplot(gs[r, k])
            panel(ax, c, xy, vis, full, cmap, (-0.5, nG - 0.5), labf, valid, stip)
            if r == 0:
                ax.set_title(nm, fontsize=11, pad=6)
            if k == 0:
                # one rotated label outside the axes; the second line used to be drawn at
                # x=0.03, which is INSIDE the panel and printed the caption over the brain
                ax.text(-0.045, 0.5, f"{title}\n{sub}", rotation=90,
                        transform=ax.transAxes, ha="center", va="center",
                        fontsize=10.5, linespacing=1.8, color="0.15")

    handles = [plt.Line2D([], [], marker="s", ls="", markersize=9, color=palette[g],
                          label=f"{names[g]}  {pct[g]:.0%}") for g in range(nG)]
    handles.append(plt.Line2D([], [], marker="o", ls="", markersize=6, color="none",
                              markerfacecolor="w", markeredgecolor="0.45",
                              markeredgewidth=0.8,
                              label="contested (top two < 10 pts)"))
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=11,
               bbox_to_anchor=(0.5, 0.005))
    fig.suptitle(f"zones of influence by thalamic nucleus  ·  {a.tag}  ·  "
                 f"percentages are share of total field variance", fontsize=12)

    out = a.out or os.path.join(RESULTS, f"zones_surface_{a.tag}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  {nG} zones over {int(valid.sum())} vertices, {len(proj)} views")
    for g in range(nG):
        print(f"    {names[g]:<10s} {pct[g]:>6.1%}")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
