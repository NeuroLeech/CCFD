"""Video of where the driven pieces reinforce and where they cancel, frame by frame.

interference.py measures the same thing averaged over the run; this is the instantaneous
version, which is what shows fronts rather than totals. Per frame:

    total        sum_k f_k        the field, as every other video shows it
    interference (sum_k f_k)^2 - sum_k f_k^2

The second is positive where the contributions arriving at that vertex reinforce and
negative where they cancel, and it is exactly the cross-term 2*sum_{k<l} f_k f_l - the
part of the field's instantaneous power that is not attributable to any single piece. It
is NOT the field squared: a vertex can be at high amplitude with no interference at all if
one piece supplies everything.

Both rows use a fixed symmetric scale across the clip, so a quiet stretch reads as quiet.
The two rows are in different units - amplitude and amplitude squared - so their scales
are set and printed separately, and no comparison between the two colour bars is meant.

  python viz/render_interference.py --tag g7_s1.5_d25 --start 200 --n 600
"""
import _path  # noqa: F401  - puts the sibling code folders on sys.path
import os, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.tri import Triangulation

from mesh_cache import load_cortex
from render_regimes import _proj
from paths import RESULTS, VIDEOS
import timescale
from interference import contributions


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", default="g7_s1.5_d25")
    ap.add_argument("--start", type=int, default=200)
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--clip", type=float, default=99.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    c = load_cortex("fsaverage5", verbose=False)
    r = contributions(c, a.tag, window=(a.start, a.start + a.n))
    w0, w1 = r["win"]
    tot = r["tot"][w0:w1]
    cross = tot ** 2 - r["win_sq"]            # 2 * sum_{k<l} f_k f_l
    frame_s = r["frame_s"]

    lim_f = float(np.percentile(np.abs(tot), a.clip))
    lim_c = float(np.percentile(np.abs(cross), a.clip))
    neg = float((cross < 0).mean())
    print(f"  window {w0}-{w1} ({(w1-w0)*frame_s:.0f} s); scales +-{lim_f:.3e} (field), "
          f"+-{lim_c:.3e} (interference)")
    print(f"  vertex-frames where the cross term is negative (cancelling): {neg:.1%}")

    proj = _proj(c.V, c.F)
    rows = [("field\nsum of 47 pieces", tot, "RdBu_r", lim_f),
            ("interference\n(sum f)^2 - sum f^2", cross, "PuOr_r", lim_c)]
    fig = plt.figure(figsize=(4.0 * len(proj), 8.4), facecolor="black")
    gs = fig.add_gridspec(3, len(proj), height_ratios=[3.1, 3.1, 1.0],
                          hspace=0.08, wspace=0.02)
    arts = []
    for ri, (lab, X, cm, lim) in enumerate(rows):
        for k, (xy, vis, nm) in enumerate(proj):
            ax = fig.add_subplot(gs[ri, k], facecolor="black")
            keep = vis[c.F].all(1)
            tri = Triangulation(xy[:, 0], xy[:, 1], c.F[keep])
            im = ax.tripcolor(tri, X[0], shading="gouraud", cmap=cm,
                              vmin=-lim, vmax=lim, rasterized=True)
            ax.set_xlim(xy[:, 0].min(), xy[:, 0].max())
            ax.set_ylim(xy[:, 1].min(), xy[:, 1].max())
            ax.set_aspect("equal"); ax.axis("off")
            if ri == 0:
                ax.set_title(nm, color="0.75", fontsize=10)
            if k == 0:
                ax.text(-0.10, 0.5, lab, rotation=90, transform=ax.transAxes,
                        ha="center", va="center", color="0.8", fontsize=9,
                        linespacing=1.6)
            arts.append((im, ri))

    # the running share of instantaneous power that is cross-term, over the window
    axd = fig.add_subplot(gs[2, :], facecolor="black")
    num = cross.sum(1)
    den = r["win_sq"].sum(1)
    frac = num / np.maximum(den, 1e-30)
    axd.plot(np.arange(len(frac)), frac, lw=0.9, color="tab:orange")
    axd.axhline(0.0, color="0.5", lw=0.8)
    cursor = axd.axvline(0, color="w", lw=1.0)
    axd.set_xlim(0, len(frac) - 1)
    axd.set_ylabel("cross / incoherent", color="0.75", fontsize=9)
    for sp in axd.spines.values():
        sp.set_color("0.3")
    axd.tick_params(colors="0.6", labelsize=8)
    axd.set_xlabel(f"saved frame  ({frame_s:.4f} s each)", color="0.75", fontsize=9)
    print(f"  cortex-summed cross/incoherent over the window: "
          f"mean {frac.mean():+.3f}, range {frac.min():+.3f} to {frac.max():+.3f}")

    def update(i):
        for im, ri in arts:
            im.set_array(rows[ri][1][i])
        cursor.set_xdata([i, i])
        return [im for im, _ in arts] + [cursor]

    out = a.out or os.path.join(VIDEOS, f"interference_{a.tag}_{w0}-{w1}.mp4")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    ani = animation.FuncAnimation(fig, update, frames=len(tot), blit=False)
    ani.save(out, writer=animation.FFMpegWriter(fps=a.fps, bitrate=4000),
             savefig_kwargs=dict(facecolor="black"))
    plt.close(fig)
    print(f"  wrote {out}  ({len(tot)/a.fps:.0f} s at {a.fps} fps)")


if __name__ == "__main__":
    main()
