"""Video of a field that best_fit.py already produced, straight from its saved frames.

render_winner.py re-integrates a searched genome; this one renders what is on disk. Every
best_fit run writes results/frames_<tag>.npy (the field, one row per saved frame) and
results/drive_<tag>.npy (the injected amplitudes, one row per STEP), so a run can be
watched afterwards without touching the solver.

The drive is plotted underneath with a time cursor, because a frame of the field is hard
to read without knowing what was going into it at that moment. Drive rows are per step and
field rows are per saved frame, so the drive is decimated by `--save` to line the two up.

  python render_frames.py --tag long4480 --n 600            # the best sensory model
  python render_frames.py --tag spread2x_4480 --start 1000 --n 400
"""
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


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", default="long4480", help="results/frames_<tag>.npy")
    ap.add_argument("--start", type=int, default=0, help="first saved frame to show")
    ap.add_argument("--n", type=int, default=600, help="frames to show")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--save", type=int, default=33, help="steps per saved frame")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--clip", type=float, default=99.0,
                    help="percentile of |field| that saturates the colour scale")
    ap.add_argument("--ndrive", type=int, default=6,
                    help="drive channels to trace (the loudest are chosen)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    c = load_cortex("fsaverage5", verbose=False)
    F = np.load(os.path.join(RESULTS, f"frames_{a.tag}.npy"), mmap_mode="r")
    sel = np.arange(a.start, min(a.start + a.n * a.stride, len(F)), a.stride)
    H = np.asarray(F[sel], np.float32)
    dpath = os.path.join(RESULTS, f"drive_{a.tag}.npy")
    D = np.load(dpath, mmap_mode="r")[::a.save] if os.path.exists(dpath) else None
    if D is not None:
        D = np.asarray(D[sel[0]:sel[-1] + 1:a.stride], np.float32)
        loud = np.argsort(D.std(0))[::-1][:a.ndrive]
    print(f"  {a.tag}: {len(F)} saved frames, showing {len(H)} from {a.start} "
          f"(stride {a.stride})" + ("" if D is None else f", {D.shape[1]} drive channels"))

    # a fixed symmetric scale across the whole clip: per-frame scaling would hide exactly
    # the amplitude structure the video is meant to show
    lim = float(np.percentile(np.abs(H), a.clip))
    proj = _proj(c.V, c.F)

    fig = plt.figure(figsize=(4.0 * len(proj), 4.4), facecolor="black")
    gs = fig.add_gridspec(2, len(proj), height_ratios=[3.1, 1.0], hspace=0.06, wspace=0.02)
    arts = []
    for k, (xy, vis, nm) in enumerate(proj):
        ax = fig.add_subplot(gs[0, k], facecolor="black")
        keep = vis[c.F].all(1)
        tri = Triangulation(xy[:, 0], xy[:, 1], c.F[keep])
        im = ax.tripcolor(tri, H[0], shading="gouraud", cmap="RdBu_r",
                          vmin=-lim, vmax=lim, rasterized=True)
        ax.set_xlim(xy[:, 0].min(), xy[:, 0].max())
        ax.set_ylim(xy[:, 1].min(), xy[:, 1].max())
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(nm, color="0.75", fontsize=10)
        arts.append(im)

    axd = fig.add_subplot(gs[1, :], facecolor="black")
    if D is not None:
        t = np.arange(len(D))
        off = 2.5 * float(np.median(D.std(0)[loud]) + 1e-12)
        for j, k in enumerate(loud):
            axd.plot(t, D[:, k] + j * off, lw=0.7, color=plt.cm.viridis(j / max(a.ndrive - 1, 1)))
        cursor = axd.axvline(0, color="w", lw=1.0)
        axd.set_xlim(0, len(D) - 1)
        axd.set_ylabel("drive", color="0.75", fontsize=9)
    else:
        cursor = None
    axd.set_yticks([])
    for sp in axd.spines.values():
        sp.set_color("0.3")
    axd.tick_params(colors="0.6", labelsize=8)
    axd.set_xlabel("saved frame", color="0.75", fontsize=9)

    def update(i):
        for im in arts:
            im.set_array(H[i])
        if cursor is not None:
            cursor.set_xdata([i, i])
        return arts + ([cursor] if cursor is not None else [])

    out = a.out or os.path.join(VIDEOS, f"field_{a.tag}_{a.start}-{a.start+len(H)*a.stride}.mp4")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    ani = animation.FuncAnimation(fig, update, frames=len(H), blit=False)
    ani.save(out, writer=animation.FFMpegWriter(fps=a.fps, bitrate=4000),
             savefig_kwargs=dict(facecolor="black"))
    plt.close(fig)
    print(f"  wrote {out}  ({len(H)/a.fps:.0f} s at {a.fps} fps)")


if __name__ == "__main__":
    main()
