"""The visual model as a film: the same field raw and bandpassed, side by side in time.

The model runs at TR/4 and its medium carries waves whose structure lives well above the
frequencies BOLD can report. The comparison the fit is scored on is the BANDPASSED one -
0.01-0.08 Hz, what XCP-D left in the data - so the question this answers is what the
scored observable is made of, and what it threw away to get there. Same field on both
rows: one realisation, one drive, one medium, and the only difference is the filter.

The drive is the resting input the solve found, PLUS the checkerboard as a modulation on
V1 - a periodic 20 s on / 20 s off boxcar injected through the empirical V1 response
profile, at a depth of `--alpha` times the resting drive's own RMS. So this is the model
of the scanner run rather than of V1 in isolation: the stimulus arrives on top of ongoing
activity, which is what sets how far above the background it rises.

Filtering is applied to the WHOLE run and the display window cut afterwards, because the
slowest passband component has a 100 s period and filtering a short excerpt would put its
transient inside the clip.

COLOUR SCALES ARE PER ROW and printed. The filtered field keeps a fraction of the power,
and a shared scale would render the bottom row nearly blank; the point is which spatial
structure survives, so each row is scaled to its own range and the ratio is stated.

  python viz/render_visual.py --tag g7_s1.5_d25 --alpha 0.1
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
import bandpass, timescale


def boxcar_periodic(t, on=20.0, off=20.0, start=20.0):
    """20 s on, 20 s off, repeating - the checkerboard's design, run the whole clip."""
    ph = np.mod(t - start, on + off)
    return ((t >= start) & (ph < on)).astype(np.float64)


def build_run(c, tag, nframes, alpha, seed=0, amp=2e-4, verbose=True):
    """-> (raw field, boxcar on the frame grid, frame_s).

    The resting drive is a fresh draw from the solved cross-spectrum, normalised to the
    same RMS score_realisation uses so the modulation depth means what it says. The
    checkerboard enters as one extra channel whose profile is the empirical V1 map, which
    is the best-scoring drive shape found - see checkerboard_modulation.
    """
    import bo_step, subparcels, units, xspec, fc_score
    import fluid as fl
    from xspec import ProfileDrive
    import checkerboard_modulation as cmo

    z = np.load(os.path.join(RESULTS, f"xspec_{tag}.npz"), allow_pickle=True)
    x, save = z["x"], int(z["save"])
    labels, tags = z["labels"], [str(s) for s in z["tags"]]
    p, _, _ = bo_step.unpack(x, c)
    P = subparcels.taper_profiles(c, labels, len(tags))
    frame_s = timescale.TR / 4.0
    t = fc_score.default_target(c, verbose=False)

    sel = [i for i, s in enumerate(tags) if int(s.split("_")[0]) == 1]
    emp = np.asarray(np.load(os.path.join(
        os.path.dirname(RESULTS), "data", "cache",
        f"checkerboard_group_100_{t.nV}.npz"), allow_pickle=True)["response"], float)
    g = cmo.vertex_profile(c, t, tag, sel, emp, 1)

    A = xspec.realise(z["S"], z["idx"], nframes,
                      ref_frames=int(z["ref_frames"]) if "ref_frames" in z.files else 4096,
                      seed=seed)
    tt = np.arange(nframes) * frame_s
    b = boxcar_periodic(tt)
    rms = np.sqrt((A ** 2).mean())
    A = np.c_[A, alpha * rms * b]                      # the stimulus as one more channel
    Pall = np.vstack([P, g[None, :].astype(P.dtype)])

    nsteps = nframes * save
    Aser = np.repeat(A, save, axis=0)[:nsteps] / save
    d = ProfileDrive(c, Pall, Aser, amp)
    d.Aser = (d.Aser * (amp / np.sqrt((d.Aser ** 2).mean()))).astype(np.float32)
    if verbose:
        print(f"  {nframes} frames ({nframes*frame_s:.0f} s), {len(tags)} resting pieces "
              f"+ 1 checkerboard channel at alpha {alpha:g}")
    frames, _ = fl.run(c, d, p, nsteps, save)
    return np.asarray(frames, np.float32), b, frame_s


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", default="g7_s1.5_d25")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--frames", type=int, default=1912, help="frames to simulate")
    ap.add_argument("--start", type=int, default=478)
    ap.add_argument("--n", type=int, default=956, help="frames to show")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--clip", type=float, default=99.0)
    ap.add_argument("--band", default="0.01,0.08")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    lo, hi = (float(v) for v in a.band.split(","))
    c = load_cortex("fsaverage5", verbose=False)
    F, b, frame_s = build_run(c, a.tag, a.frames, a.alpha, seed=a.seed)
    G = bandpass.apply(F, frame_s, lo, hi)

    def power_in_band(X):
        Xc = X - X.mean(0, keepdims=True)
        Pw = (np.abs(np.fft.rfft(Xc, axis=0)) ** 2).mean(1)
        f = np.fft.rfftfreq(X.shape[0], frame_s)
        return float(Pw[(f >= lo) & (f <= hi)].sum() / max(Pw[1:].sum(), 1e-300))

    pb_raw, pb_filt = power_in_band(F), power_in_band(G)
    sel = np.arange(a.start, min(a.start + a.n, len(F)), a.stride)
    Hs = [np.asarray(F[sel]), np.asarray(G[sel])]
    lims = [float(np.percentile(np.abs(h), a.clip)) for h in Hs]
    print(f"  showing {len(sel)} of {len(F)} frames from {a.start} "
          f"({len(sel)*a.stride*frame_s:.0f} s of model time)")
    print(f"  power in {lo}-{hi} Hz: raw {pb_raw:.1%}, filtered {pb_filt:.1%}")
    print(f"  amplitude scale: raw {lims[0]:.3e}, filtered {lims[1]:.3e} "
          f"({lims[1]/max(lims[0],1e-30):.2f}x)")

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
                ax.text(-0.10, 0.5, labels[r], rotation=90, transform=ax.transAxes,
                        ha="center", va="center", color="0.8", fontsize=9, linespacing=1.6)
            arts.append((im, r))

    axd = fig.add_subplot(gs[2, :], facecolor="black")
    bb = b[sel]
    axd.fill_between(np.arange(len(sel)), 0, bb, step="mid", color="0.75", lw=0)
    cursor = axd.axvline(0, color="tab:red", lw=1.4)
    axd.set_xlim(0, len(sel) - 1); axd.set_ylim(-0.1, 1.2)
    axd.set_yticks([]); axd.set_ylabel("checkerboard", color="0.75", fontsize=9)
    for sp in axd.spines.values():
        sp.set_color("0.3")
    axd.tick_params(colors="0.6", labelsize=8)
    axd.set_xlabel(f"frame  ({a.stride*frame_s:.3f} s each)", color="0.75", fontsize=9)

    def update(i):
        for im, r in arts:
            im.set_array(Hs[r][i])
        cursor.set_xdata([i, i])
        return [im for im, _ in arts] + [cursor]

    out = a.out or os.path.join(VIDEOS, f"visual_{a.tag}_a{a.alpha:g}.mp4")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    ani = animation.FuncAnimation(fig, update, frames=len(sel), blit=False)
    ani.save(out, writer=animation.FFMpegWriter(fps=a.fps, bitrate=4000),
             savefig_kwargs=dict(facecolor="black"))
    plt.close(fig)
    print(f"  wrote {out}  ({len(sel)/a.fps:.0f} s at {a.fps} fps)")


if __name__ == "__main__":
    main()
