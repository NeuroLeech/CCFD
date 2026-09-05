"""Video of the INPUT: what is injected into the sheet, not what the sheet does with it.

Every other video here shows the field. This shows the drive that produced it, on the same
surface and the same clock, so the two can be watched against each other: the input is 47
fixed spatial profiles whose amplitudes vary, and the field is what the medium makes of
that. The pieces never move; only their amplitudes do.

The injected map at frame n is `A[n] @ P` - the piece profiles weighted by that frame's
amplitudes. drive_<tag>.npy stores the per-STEP series that fluid.run integrated, which is
score_realisation's `np.repeat(A_frames, save) / save`, so a frame's own amplitude is the
mean over its block of `save` steps. Taking every save-th step instead would sample one
step of the block and miss nothing here, since the value is constant across it - but the
mean is what the frame actually injected and does not depend on that staying true.

Amplitude is signed and the colour scale is symmetric: `h += A @ P` adds to depth every
step, so a piece with a positive amplitude is pushing depth up under its taper and a
negative one is pulling it down. The scale is the 99th percentile of |injected|, fixed
across the clip, so the quiet stretches read as quiet rather than being renormalised.

  python render_drive.py --tag g7_s1.5_d25 --save 4 --n 600
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
import timescale


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", default="g7_s1.5_d25")
    ap.add_argument("--start", type=int, default=200)
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--save", type=int, default=4, help="steps per saved frame")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--clip", type=float, default=99.0)
    ap.add_argument("--ndrive", type=int, default=6)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    import subparcels
    frame_s = timescale.TR / 4.0
    c = load_cortex("fsaverage5", verbose=False)
    z = np.load(os.path.join(RESULTS, f"xspec_{a.tag}.npz"), allow_pickle=True)
    labels, tags = z["labels"], list(z["tags"])
    P = subparcels.taper_profiles(c, labels, len(tags))
    Aser = np.asarray(np.load(os.path.join(RESULTS, f"drive_{a.tag}.npy")), np.float32)

    nfr = len(Aser) // a.save
    A = Aser[:nfr * a.save].reshape(nfr, a.save, -1).mean(1)      # per-frame amplitudes
    sel = np.arange(a.start, min(a.start + a.n, nfr))
    inj = A[sel] @ P                                              # (n, nV) injected map
    lim = float(np.percentile(np.abs(inj), a.clip))
    print(f"  {a.tag}: {len(Aser)} steps -> {nfr} frames, {len(tags)} pieces")
    print(f"  showing {len(sel)} frames from {a.start} ({len(sel)*frame_s:.0f} s), "
          f"scale +-{lim:.3e}")
    act = np.abs(A[sel]).mean(0)
    order = np.argsort(act)[::-1]
    print(f"  loudest pieces: " + ", ".join(f"{tags[k]}" for k in order[:6]))

    proj = _proj(c.V, c.F)
    Dl = A[sel]
    loud = order[:a.ndrive]
    fig = plt.figure(figsize=(4.0 * len(proj), 4.4), facecolor="black")
    gs = fig.add_gridspec(2, len(proj), height_ratios=[3.1, 1.0], hspace=0.06, wspace=0.02)
    arts = []
    for k, (xy, vis, nm) in enumerate(proj):
        ax = fig.add_subplot(gs[0, k], facecolor="black")
        keep = vis[c.F].all(1)
        tri = Triangulation(xy[:, 0], xy[:, 1], c.F[keep])
        im = ax.tripcolor(tri, inj[0], shading="gouraud", cmap="PuOr_r",
                          vmin=-lim, vmax=lim, rasterized=True)
        ax.set_xlim(xy[:, 0].min(), xy[:, 0].max())
        ax.set_ylim(xy[:, 1].min(), xy[:, 1].max())
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(nm, color="0.75", fontsize=10)
        if k == 0:
            ax.text(-0.06, 0.5, "injected input\n47 pieces", rotation=90,
                    transform=ax.transAxes, ha="center", va="center",
                    color="0.8", fontsize=9, linespacing=1.6)
        arts.append(im)

    axd = fig.add_subplot(gs[1, :], facecolor="black")
    tt = np.arange(len(Dl))
    off = 2.5 * float(np.median(Dl.std(0)[loud]) + 1e-12)
    for j, k in enumerate(loud):
        axd.plot(tt, Dl[:, k] + j * off, lw=0.7,
                 color=plt.cm.viridis(j / max(a.ndrive - 1, 1)))
    cursor = axd.axvline(0, color="w", lw=1.0)
    axd.set_xlim(0, len(Dl) - 1)
    axd.set_ylabel("amplitude", color="0.75", fontsize=9)
    axd.set_yticks([])
    for sp in axd.spines.values():
        sp.set_color("0.3")
    axd.tick_params(colors="0.6", labelsize=8)
    axd.set_xlabel(f"saved frame  ({frame_s:.4f} s each)", color="0.75", fontsize=9)

    def update(i):
        for im in arts:
            im.set_array(inj[i])
        cursor.set_xdata([i, i])
        return arts + [cursor]

    out = a.out or os.path.join(VIDEOS, f"drive_{a.tag}_{a.start}-{sel[-1]+1}.mp4")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    ani = animation.FuncAnimation(fig, update, frames=len(sel), blit=False)
    ani.save(out, writer=animation.FFMpegWriter(fps=a.fps, bitrate=4000),
             savefig_kwargs=dict(facecolor="black"))
    plt.close(fig)
    print(f"  wrote {out}  ({len(sel)/a.fps:.0f} s at {a.fps} fps)")


if __name__ == "__main__":
    main()
