"""One realisation, shown raw and bandpassed: what the target's filter keeps and removes.

The pair of videos already in docs/ differ in medium AND in filter, because each was
solved under its own objective. This holds everything fixed except the filter: one run's
frames on the top row, the SAME frames through bandpass.apply on the bottom. Same medium,
same drive, same realisation, same window - the only difference is the 0.01-0.08 Hz
passband XCP-D applied to the data and best_fit --bandpass applies to the observable.

The filter is applied to the whole run and the window is cut afterwards. Filtering a
600-frame excerpt instead would put the transient of a filter whose slowest component has
a 100 s period inside the clip.

COLOUR SCALES ARE PER ROW, and printed. The filtered field keeps about an eighth of the
power, so roughly a third of the amplitude, and a shared scale would render the bottom row
nearly blank - the same reason render_regimes scales its two conditions separately. The
point here is which SPATIAL STRUCTURE survives, so each row is scaled to its own range and
the ratio is stated rather than shown.

  python render_bandpass.py --tag rbc_nobp --n 600
"""
import os, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

from mesh_cache import load_cortex
from render_regimes import _proj
from paths import RESULTS, VIDEOS
import bandpass, timescale


def resimulate_raw(c, tag, frame_s):
    """-> (T, nV) the run's field with the bandpass OFF, from its own saved drive.

    Nothing is re-solved and nothing is re-drawn: drive_<tag>.npy is the per-step amplitude
    series score_realisation actually integrated, so feeding it back through the same
    medium reproduces that realisation exactly. Only the filter is left off. The BOLD
    smoothing kernel stays, because it is part of the observable in both rows."""
    import bo_step, subparcels, units, fluid as fl
    from xspec import ProfileDrive
    z = np.load(os.path.join(RESULTS, f"xspec_{tag}.npz"), allow_pickle=True)
    labels, tags, x = z["labels"], list(z["tags"]), z["x"]
    p, save, _ = bo_step.unpack(x, c)
    P = subparcels.taper_profiles(c, labels, len(tags))
    Aser = np.asarray(np.load(os.path.join(RESULTS, f"drive_{tag}.npy")), np.float32)
    d = ProfileDrive(c, P, Aser, 2e-4)
    print(f"  resimulating {tag}: {len(Aser)} steps, save {save}, {len(tags)} pieces")
    frames, _ = fl.run(c, d, p, len(Aser), save)
    kern = units.smoothing_kernel(timescale.bold_fwhm_frames(frame_s, verbose=False),
                                  verbose=False)
    return units.smooth_frames(frames, kern)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", default="rbc_nobp")
    ap.add_argument("--start", type=int, default=200)
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--save", type=int, default=16, help="steps per saved frame, for the drive")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--clip", type=float, default=99.0)
    ap.add_argument("--band", default="0.01,0.08")
    ap.add_argument("--ndrive", type=int, default=6)
    ap.add_argument("--resimulate", action="store_true",
                    help="the tag was fitted WITH --bandpass, so its saved frames are "
                         "already filtered and refiltering them would filter twice. Re-run "
                         "the integration from the run's own saved drive with the filter "
                         "off to recover the raw field: same medium, same drive, same "
                         "realisation")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    lo, hi = (float(v) for v in a.band.split(","))
    frame_s = timescale.TR / 4.0
    c = load_cortex("fsaverage5", verbose=False)
    saved = np.asarray(np.load(os.path.join(RESULTS, f"frames_{a.tag}.npy")), np.float32)
    if a.resimulate:
        F = resimulate_raw(c, a.tag, frame_s)
        G = bandpass.apply(F, frame_s, lo, hi)
        # the reconstruction is only right if refiltering it returns what the run saved
        m = min(len(G), len(saved))
        r = float(np.corrcoef(G[:m].ravel(), saved[:m].ravel())[0, 1])
        print(f"  resimulated raw field; refiltering it against the run's saved frames: "
              f"r = {r:.6f}")
        if r < 0.99:
            raise SystemExit("  the reconstruction does not reproduce the saved run")
    else:
        F, G = saved, bandpass.apply(saved, frame_s, lo, hi)

    def power_in_band(X):
        Xc = X - X.mean(0, keepdims=True)
        P = (np.abs(np.fft.rfft(Xc, axis=0)) ** 2).mean(1)
        f = np.fft.rfftfreq(X.shape[0], frame_s)
        return float(P[(f >= lo) & (f <= hi)].sum() / max(P[1:].sum(), 1e-300))

    pb_raw, pb_filt = power_in_band(F), power_in_band(G)
    sel = np.arange(a.start, min(a.start + a.n, len(F)))
    Hs = [np.asarray(F[sel]), np.asarray(G[sel])]
    lims = [float(np.percentile(np.abs(h), a.clip)) for h in Hs]
    print(f"  {a.tag}: {len(F)} frames, showing {len(sel)} from {a.start}")
    print(f"  power in {lo}-{hi} Hz: raw {pb_raw:.1%}, filtered {pb_filt:.1%}")
    print(f"  amplitude scale: raw {lims[0]:.3e}, filtered {lims[1]:.3e} "
          f"({lims[1]/max(lims[0],1e-30):.2f}x)")

    dpath = os.path.join(RESULTS, f"drive_{a.tag}.npy")
    D = None
    if os.path.exists(dpath):
        D = np.asarray(np.load(dpath, mmap_mode="r")[::a.save][sel[0]:sel[-1] + 1], np.float32)
        loud = np.argsort(D.std(0))[::-1][:a.ndrive]

    proj = _proj(c.V, c.F)
    labels = [f"raw field\n{pb_raw:.0%} of power in band",
              f"bandpassed {lo}-{hi} Hz\n{pb_filt:.0%} of power in band"]
    fig = plt.figure(figsize=(4.0 * len(proj), 8.4), facecolor="black")
    gs = fig.add_gridspec(3, len(proj), height_ratios=[3.1, 3.1, 1.0],
                          hspace=0.08, wspace=0.02)
    arts = []
    for r in range(2):
        for k, (xy, vis, nm) in enumerate(proj):
            ax = fig.add_subplot(gs[r, k], facecolor="black")
            keep = vis[c.F].all(1)
            tri = Triangulation(xy[:, 0], xy[:, 1], c.F[keep])
            im = ax.tripcolor(tri, Hs[r][0], shading="gouraud", cmap="RdBu_r",
                              vmin=-lims[r], vmax=lims[r], rasterized=True)
            ax.set_xlim(xy[:, 0].min(), xy[:, 0].max())
            ax.set_ylim(xy[:, 1].min(), xy[:, 1].max())
            ax.set_aspect("equal"); ax.axis("off")
            if r == 0:
                ax.set_title(nm, color="0.75", fontsize=10)
            if k == 0:
                # well clear of the axes: at -0.03 the two-line label sat on top of the
                # lateral surface, which is the panel it is labelling
                ax.text(-0.10, 0.5, labels[r], rotation=90, transform=ax.transAxes,
                        ha="center", va="center", color="0.8", fontsize=9,
                        linespacing=1.6)
            arts.append((im, r))

    axd = fig.add_subplot(gs[2, :], facecolor="black")
    cursor = None
    if D is not None:
        tt = np.arange(len(D))
        off = 2.5 * float(np.median(D.std(0)[loud]) + 1e-12)
        for j, k in enumerate(loud):
            axd.plot(tt, D[:, k] + j * off, lw=0.7,
                     color=plt.cm.viridis(j / max(a.ndrive - 1, 1)))
        cursor = axd.axvline(0, color="w", lw=1.0)
        axd.set_xlim(0, len(D) - 1)
        axd.set_ylabel("drive", color="0.75", fontsize=9)
    axd.set_yticks([])
    for sp in axd.spines.values():
        sp.set_color("0.3")
    axd.tick_params(colors="0.6", labelsize=8)
    axd.set_xlabel(f"saved frame  ({frame_s:.4f} s each)", color="0.75", fontsize=9)

    def update(i):
        for im, r in arts:
            im.set_array(Hs[r][i])
        if cursor is not None:
            cursor.set_xdata([i, i])
        return [im for im, _ in arts] + ([cursor] if cursor is not None else [])

    out = a.out or os.path.join(VIDEOS, f"bandpass_pair_{a.tag}_{a.start}-{sel[-1]+1}.mp4")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    import matplotlib.animation as animation
    ani = animation.FuncAnimation(fig, update, frames=len(sel), blit=False)
    ani.save(out, writer=animation.FFMpegWriter(fps=a.fps, bitrate=4000),
             savefig_kwargs=dict(facecolor="black"))
    plt.close(fig)
    print(f"  wrote {out}  ({len(sel)/a.fps:.0f} s at {a.fps} fps)")


if __name__ == "__main__":
    main()
