"""The drive's dominant patterns and the cortical field, side by side and in time.

One draw (577 s, seed 1000) from a one-hot solve is run through the medium. Then:

  input gradients   the diffusion components of the 1,880 input timecourses
                    (viz/input_embedding.py's npz), and their TIMECOURSES: each vertex's drive
                    weighted by its value on the gradient map and summed - how strongly the
                    drive is doing that spatial pattern at each moment
  field gradients   the same embedding on the cortical field (the bandpassed, BOLD-smoothed
                    observable over every vertex), and the field projected onto its own maps

A static figure puts the input and field gradient maps next to each other. A movie shows the
drive on its vertices, the raw field and the observable, each z-scored per vertex over the run,
with the drive-pattern and field-pattern timecourses underneath and a cursor. The drive
patterns are the BANDPASSED drive projected on the input gradients; each trace is z-scored.

FIRST RUN, 2026-10-05, oh1880_nv2000_mf400. The field gradients are large-scale and smooth
(shares 25/22/19%): a central/sensorimotor-and-medial-parietal pattern, an occipital one, a
frontal-parietal one. Peak cross-correlation of drive pattern against field pattern within
+-30 s (positive lag: the field follows):

              field 1          field 2          field 3
    drive 1  +0.74 (2.1 s)    +0.36 (2.1 s)    -0.55 (-0.8 s)
    drive 2  -0.71 (3.7 s)    +0.49 (9.4 s)    +0.60 (0.0 s)
    drive 3  +0.76 (3.9 s)    -0.35 (-11 s)    +0.72 (7.1 s)

All three drive patterns feed the field's leading pattern, which follows by 2-4 s. Entries
near 0.35-0.5 are within what slow band-limited series and a best-of-lags search give by chance.

  python viz/drive_field_movie.py --tag oh1880_nv2000_mf400
"""
import _path  # noqa: F401
import os, re, argparse
import numpy as np

from paths import RESULTS, VIDEOS

NC = 3


def zs(X):
    X = np.asarray(X, np.float64)
    X = X - X.mean(0, keepdims=True)
    return np.asarray(X / np.maximum(X.std(0, keepdims=True), 1e-300), np.float32)


def project(X, G):
    """timecourse of each gradient pattern: X (T, V) onto centred, unit-norm maps G (V, k)"""
    Gc = G - np.median(G, axis=0, keepdims=True)
    Gc = Gc / np.linalg.norm(Gc, axis=0, keepdims=True)
    P = X @ Gc
    return (P - P.mean(0)) / P.std(0)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", default="oh1880_nv2000_mf400")
    ap.add_argument("--seconds", type=float, default=577.0)
    ap.add_argument("--step", type=int, default=2, help="model frames per video frame")
    ap.add_argument("--fps", type=int, default=25)
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.tri import Triangulation
    import matplotlib.animation as animation
    import xspec, bandpass, timescale, units
    import fluid as fl
    from interp_gap import medium_context
    from render_regimes import _proj
    from input_embedding import embed, VIEWS, _surface

    z = np.load(os.path.join(RESULTS, f"xspec_{a.tag}.npz"), allow_pickle=True)
    kern, c, t, g = medium_context(z)
    fs, lo, hi = g["frame_s"], g["lo"], g["hi"]
    nfr = timescale.frames_for(a.seconds, fs)
    A = xspec.realise(None, np.asarray(z["idx"]), nfr, ref_frames=int(z["ref_frames"]),
                      seed=1000, factors=xspec.load_factors(z))
    Aser = np.repeat(A, g["save"], axis=0)[:nfr * g["save"]] / g["save"]
    raw, _ = fl.run(c, xspec.ProfileDrive(c, g["P"], Aser, 2e-4), g["p"], nfr * g["save"], g["save"])
    obs = bandpass.apply(units.smooth_frames(raw, kern), fs, lo, hi)
    print(f"  simulated {a.seconds:.0f} s: {raw.shape}")

    oh = np.load(os.path.join(RESULTS, "asc_first_onehot_1880.npz"), allow_pickle=True)
    vtx = np.argmax(oh["profiles"], axis=1)
    emb = np.load(os.path.join(RESULTS, "figs", f"input_embedding_{a.tag}.npz"), allow_pickle=True)
    Gin = emb["G_unfiltered"][:, :NC]
    cols = np.asarray(t.cols)
    burn = t.burn
    Gf, lamf = embed(np.asarray(obs[burn:, cols], np.float64), n=6)
    Gf = Gf[:, :NC]
    print(f"  field embedding: leading eigenvalue shares "
          + ", ".join(f"{v/lamf.sum():.0%}" for v in lamf))

    drive_u = A                                           # (T, 1880), as injected
    drive_b = bandpass.apply(A, fs, lo, hi)
    Pin_u, Pin_b = project(drive_u, Gin), project(drive_b, Gin)
    Pf = project(zs(obs[:, cols]), Gf)
    np.savez(os.path.join(RESULTS, "figs", f"drive_field_{a.tag}.npz"), Gin=Gin, Gf=Gf,
             lamf=lamf, cols=cols, vtx=vtx, Pin_u=Pin_u, Pin_b=Pin_b, Pf=Pf)

    # ---- static: input gradient maps beside field gradient maps
    proj = _proj(c.V, c.F, VIEWS)
    driven = np.zeros(c.nV, bool); driven[vtx] = True
    allv = np.zeros(c.nV, bool); allv[cols] = True
    tile_full = np.full(c.nV, -1)
    fig = plt.figure(figsize=(18, 4.6 * NC))
    gs = fig.add_gridspec(NC, 4, hspace=0.12, wspace=0.02, top=0.93, bottom=0.02,
                          left=0.01, right=0.99)
    for k in range(NC):
        for side, (G, mask, idxs, name) in enumerate(((Gin, driven, vtx, "input"),
                                                     (Gf, allv, cols, "field"))):
            gk = G[:, k] - np.median(G[:, k]); lim = np.percentile(np.abs(gk), 98)
            full = np.zeros(c.nV); full[idxs] = gk
            for vi, (xy, vis, nm) in enumerate(proj):
                ax = fig.add_subplot(gs[k, side * 2 + vi])
                _surface(ax, c, xy, vis, mask, full, "RdBu_r", (-lim, lim), tile_full)
                if side == 1:      # the field covers the sheet; show all of it
                    ax.set_xlim(xy[:, 0].min(), xy[:, 0].max()); ax.set_ylim(xy[:, 1].min(), xy[:, 1].max())
                ax.set_title(f"{name} gradient {k+1}  {nm}", fontsize=11, loc="left")
    fig.suptitle(f"{a.tag}: input gradients (the 1,880 drive timecourses, unfiltered) beside "
                 f"field gradients (the bandpassed observable)", x=0.01, ha="left", fontsize=12)
    path = os.path.join(RESULTS, "figs", f"drive_field_gradients_{a.tag}.png")
    fig.savefig(path, dpi=80); plt.close(fig)
    print(f"  wrote {path}")

    # ---- movie
    sel = np.arange(0, nfr, a.step)
    Dz = np.zeros((len(sel), c.nV), np.float32); Dz[:, vtx] = zs(drive_b)[sel]
    Rz = zs(raw)[sel]; Oz = zs(obs)[sel]
    rows = [("the drive, 0.01-0.08 Hz,\non its 1,880 vertices", Dz, driven),
            ("the field, raw", Rz, None), ("the observable:\nsmoothed + 0.01-0.08 Hz", Oz, None)]
    fig = plt.figure(figsize=(9.0, 13.0), facecolor="black")
    gsm = fig.add_gridspec(5, 2, height_ratios=[2.6, 2.6, 2.6, 1.3, 1.3], hspace=0.10,
                           wspace=0.02, top=0.94, bottom=0.05, left=0.07, right=0.99)
    arts = []
    for r, (lab, H, mask) in enumerate(rows):
        for vi, (xy, vis, nm) in enumerate(proj):
            ax = fig.add_subplot(gsm[r, vi], facecolor="black")
            keep = vis[c.F].all(1) & (mask[c.F].all(1) if mask is not None else True)
            if mask is not None:
                base = vis[c.F].all(1)
                ax.tripcolor(Triangulation(xy[:, 0], xy[:, 1], c.F[base]), np.full(base.sum(), 0.75),
                             cmap="Greys", vmin=0, vmax=1, rasterized=True)
            im = ax.tripcolor(Triangulation(xy[:, 0], xy[:, 1], c.F[keep]), H[0], shading="gouraud",
                              cmap="RdBu_r", vmin=-2.5, vmax=2.5, rasterized=True)
            ax.set_xlim(xy[:, 0].min(), xy[:, 0].max()); ax.set_ylim(xy[:, 1].min(), xy[:, 1].max())
            ax.set_aspect("equal"); ax.axis("off")
            if r == 0:
                ax.set_title(nm, color="0.75", fontsize=10)
            if vi == 0:
                ax.text(-0.06, 0.5, lab, rotation=90, transform=ax.transAxes, ha="center",
                        va="center", color="0.8", fontsize=9)
            arts.append((im, H))
    tt = np.arange(nfr) * fs
    curs = []
    for r, (P, lab) in enumerate(((Pin_b, "drive patterns\n(input gradients 1-3)"),
                                  (Pf, "field patterns\n(field gradients 1-3)"))):
        ax = fig.add_subplot(gsm[3 + r, :], facecolor="black")
        for k in range(NC):
            ax.plot(tt, P[:, k] + 6 * k, lw=0.8, color=plt.cm.plasma(0.15 + 0.3 * k))
        curs.append(ax.axvline(0, color="w", lw=1.0))
        ax.set_xlim(0, tt[-1]); ax.set_yticks([6 * k for k in range(NC)])
        ax.set_yticklabels([str(k + 1) for k in range(NC)], color="0.7", fontsize=8)
        ax.set_ylabel(lab, color="0.8", fontsize=8)
        for sp in ax.spines.values():
            sp.set_color("0.3")
        ax.tick_params(colors="0.6", labelsize=8)
        if r == 1:
            ax.set_xlabel("seconds", color="0.75", fontsize=9)
    ttl = fig.text(0.5, 0.975, "", color="0.85", ha="center", fontsize=11)

    def update(i):
        for im, H in arts:
            im.set_array(H[i])
        for cu in curs:
            cu.set_xdata([tt[sel[i]]] * 2)
        ttl.set_text(f"{a.tag}   t = {tt[sel[i]]:6.1f} s   (z-scored per vertex)")
        return []

    os.makedirs(VIDEOS, exist_ok=True)
    out = os.path.join(VIDEOS, f"drive_field_{a.tag}.mp4")
    ani = animation.FuncAnimation(fig, update, frames=len(sel), blit=False)
    ani.save(out, writer=animation.FFMpegWriter(fps=a.fps, bitrate=5000),
             savefig_kwargs=dict(facecolor="black"))
    plt.close(fig)
    print(f"  wrote {out}  ({len(sel)/a.fps:.0f} s at {a.fps} fps, {os.path.getsize(out)/2**20:.0f} MB)")


if __name__ == "__main__":
    main()
