"""Where the input actually enters: the driven pieces, drawn and counted.

Every other diagnostic refers to "the drive" as if it were one thing. It is 47 pieces cut
to equal area out of 17 parcels, grouped by the thalamic nucleus that names them, and the
groups are nothing like equal: one nucleus contributes a single piece and another
fourteen. That asymmetry is anatomy passed through an equal-AREA split rather than a
choice, but it sets what any per-system comparison can mean, so it belongs on the page
next to the results rather than in a docstring.

Three maps: the pieces themselves, the same coloured by nucleus, and geodesic distance to
the nearest driven vertex - the last being the axis diag_distance.py scores against.

  python inputs_map.py --tag pr_taper
"""
import os, argparse
import numpy as np

from paths import RESULTS
from zones import NUCLEI, group_index
from diag_distance import distance_to_drive


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", default="pr_taper")
    a = ap.parse_args()

    from mesh_cache import load_cortex
    import fc_score
    c = load_cortex("fsaverage5", verbose=False)
    t = fc_score.default_target(c, verbose=False)
    z = np.load(os.path.join(RESULTS, f"xspec_{a.tag}.npz"), allow_pickle=True)
    labels, tags = z["labels"], list(z["tags"])
    names, gidx = group_index(tags)
    parcel_of_piece = np.array([int(s.split("_")[0]) for s in tags])
    area = np.asarray(c.A, float)

    def nm(i):
        s = c.names[i]
        return (s.decode() if isinstance(s, bytes) else str(s)).removeprefix("L_").removesuffix("_ROI")

    print(f"  {a.tag}: {len(tags)} pieces over {len(set(parcel_of_piece.tolist()))} parcels")
    print(f"\n  {'nucleus':<10s} {'role':<19s} {'parcels':<26s} {'pieces':>6s} "
          f"{'mm2':>7s} {'mm2/piece':>10s}")
    tot_pieces = tot_area = 0
    for g, (nucleus, ps, role) in enumerate(NUCLEI):
        mm2 = float(sum(area[c.lab == q].sum() for q in ps))
        npc = len(gidx[g])
        tot_pieces += npc; tot_area += mm2
        print(f"  {nucleus:<10s} {role:<19s} {', '.join(nm(q) for q in ps):<26s} "
              f"{npc:>6d} {mm2:>7.0f} {mm2/max(npc,1):>10.0f}")
    print(f"  {'total':<10s} {'':<19s} {'':<26s} {tot_pieces:>6d} {tot_area:>7.0f} "
          f"{tot_area/tot_pieces:>10.0f}")

    driven = labels >= 0
    d_all = distance_to_drive(c, driven)
    dv = d_all[t.cols]
    print(f"\n  driven vertices {int(driven.sum())} of {c.nV} "
          f"({driven.sum()/c.nV:.1%} of the sheet); distance to the nearest of them "
          f"0-{dv.max():.0f} mm, median {np.median(dv):.0f} mm")
    sens = [g for g, (n_, p_, r_) in enumerate(NUCLEI) if n_ in ("LGN", "MGN", "VPL/VPM")]
    ns = sum(len(gidx[g]) for g in sens)
    print(f"  of the {len(tags)} pieces, {ns} belong to a primary sensory nucleus "
          f"(LGN/MGN/VPL-VPM) and {len(tags)-ns} do not")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from render_regimes import _proj
    from plot_fc_map import surface_row

    proj = _proj(c.V, c.F)
    nucleus_of_piece = np.full(len(tags), -1)
    for g in range(len(NUCLEI)):
        nucleus_of_piece[gidx[g]] = g
    nuc_map = np.where(labels >= 0, nucleus_of_piece[np.clip(labels, 0, None)], -1)

    cols = plt.cm.tab10(np.linspace(0, 1, 10))[:len(NUCLEI)]
    piece_cmap = ListedColormap(np.vstack([[0.9, 0.9, 0.9, 1.0],
                                           plt.cm.turbo(np.linspace(0, 1, len(tags)))]))
    nuc_cmap = ListedColormap(np.vstack([[0.9, 0.9, 0.9, 1.0], cols]))

    rows = [(f"{len(tags)} driven pieces", (labels + 1)[t.cols].astype(float),
             piece_cmap, (-0.5, len(tags) + 0.5)),
            ("grouped by thalamic nucleus", (nuc_map + 1)[t.cols].astype(float),
             nuc_cmap, (-0.5, len(NUCLEI) + 0.5)),
            ("geodesic distance to drive (mm)", dv, "viridis", (0, float(dv.max())))]
    fig = plt.figure(figsize=(3.4 * len(proj), 2.4 * len(rows)))
    gs = fig.add_gridspec(len(rows), len(proj), hspace=0.06, wspace=0.02)
    for r, (lab, vals, cm, lims) in enumerate(rows):
        surface_row(fig, gs, r, proj, vals, c, t.cols, cm, lims, lab)
    handles = [plt.Line2D([], [], marker="s", ls="", color=cols[g],
                          label=f"{NUCLEI[g][0]} ({len(gidx[g])})")
               for g in range(len(NUCLEI))]
    fig.legend(handles=handles, loc="lower center", ncol=len(NUCLEI), frameon=False,
               fontsize=9)
    out = os.path.join(RESULTS, f"inputs_{a.tag}.png")
    fig.savefig(out, dpi=115, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
