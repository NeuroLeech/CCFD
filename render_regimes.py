"""One video per parameter regime, both input conditions in the same frame.

Reads the stored vertex x time fields - nothing is re-simulated. Each video shows
the anterior pair (10r+10v) on the top row and V1 on the bottom, in three views,
with the drive trace and a moving cursor underneath so the field can be read
against what is driving it.

Colour scale is set per condition, not shared: the two conditions differ in peak
amplitude by up to 6x at the same regime, and a shared scale would render one of
them invisible. The scale is printed on each row.
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from mesh_cache import load_cortex

from paths import FIELDS, VIDEOS
OUT = os.path.join(VIDEOS, "regimes")
VIEWS = ((8, 180, "lateral"), (8, 0, "medial"), (90, 180, "dorsal"))
CONDS = [("A_10r10v", "10r + 10v"), ("B_V1", "V1")]
CLIP = 96.0
FPS = 16


def _proj(Vc, Fc):
    n3 = np.cross(Vc[Fc[:, 1]]-Vc[Fc[:, 0]], Vc[Fc[:, 2]]-Vc[Fc[:, 0]])
    vn = np.zeros_like(Vc)
    for k in range(3):
        np.add.at(vn, Fc[:, k], n3)
    vn /= np.linalg.norm(vn, axis=1)[:, None]
    out = []
    for elev, azim, nm in VIEWS:
        e, a = np.radians(elev), np.radians(azim)
        d = np.array([np.cos(e)*np.cos(a), np.cos(e)*np.sin(a), np.sin(e)])
        right = np.array([-np.sin(a), np.cos(a), 0.0])
        if abs(elev) > 80:
            right = np.array([0.0, 1.0, 0.0])
        up = np.cross(d, right)
        out.append((np.c_[Vc @ right, Vc @ up], (vn @ d) > -0.10, nm))
    return out


def render(pi, rec_by_cond, proj, path):
    data = {}
    for cond, _ in CONDS:
        z = np.load(os.path.join(FIELDS, f"p{pi:02d}_{cond}.npz"))
        data[cond] = (z["H"], z["drive"], float(z["dt_frame"]))
    nF = min(len(data[c][0]) for c, _ in CONDS)
    p = rec_by_cond["A_10r10v"]

    fig = plt.figure(figsize=(4.0*len(VIEWS), 7.4))
    gs = fig.add_gridspec(3, len(VIEWS), height_ratios=[1.0, 1.0, 0.34],
                          hspace=0.06, wspace=0.02)
    scs, vlims = {}, {}
    for r, (cond, clab) in enumerate(CONDS):
        Hs = data[cond][0]
        vl = float(np.percentile(np.abs(Hs), CLIP))
        vlims[cond] = vl
        scs[cond] = []
        for k, (xy, vis, nm) in enumerate(proj):
            ax = fig.add_subplot(gs[r, k])
            sc = ax.scatter(xy[vis, 0], xy[vis, 1], c=np.zeros(vis.sum()),
                            s=3.0, linewidths=0, cmap="RdBu_r",
                            vmin=-vl, vmax=vl)
            ax.set_xlim(xy[:, 0].min(), xy[:, 0].max())
            ax.set_ylim(xy[:, 1].min(), xy[:, 1].max())
            ax.set_aspect("equal"); ax.axis("off")
            if r == 0:
                ax.set_title(nm, fontsize=10)
            if k == 0:
                ax.text(-0.04, 0.5, f"{clab}\n±{vl:.1e}", rotation=90,
                        transform=ax.transAxes, ha="center", va="center",
                        fontsize=9, family="monospace")
            scs[cond].append(sc)

    axd = fig.add_subplot(gs[2, :])
    tvec = np.arange(nF)*data["A_10r10v"][2]
    for cond, clab in CONDS:
        d = data[cond][1][:nF]
        d = d/max(np.abs(d).max(), 1e-30)
        axd.plot(tvec, d[:, 0], lw=0.9, label=clab)
    axd.set_xlim(tvec[0], tvec[-1]); axd.set_ylim(-1.15, 1.15)
    axd.set_yticks([]); axd.set_xlabel("time units", fontsize=9)
    axd.tick_params(labelsize=8)
    axd.legend(fontsize=8, ncol=2, loc="upper right", framealpha=0.7)
    axd.set_ylabel("drive", fontsize=9)
    cursor = axd.axvline(tvec[0], color="k", lw=1.2)

    fig.suptitle(
        f"regime p{pi:02d}   Ld {p['Ld']:.0f} mm (L/Ld {228/p['Ld']:.1f})   "
        f"sponge {p['sponge_strength']:.2f} over {p['sponge_width']:.0f} mm   "
        f"drive tau {p['tau']:.0f}   silent {p['silent']:.2f}",
        fontsize=11)
    txt = fig.text(0.5, 0.955, "", ha="center", fontsize=10, family="monospace")

    def upd(i):
        for cond, _ in CONDS:
            Hs = data[cond][0]
            for sc, (_, vis, _) in zip(scs[cond], proj):
                sc.set_array(Hs[i][vis])
        cursor.set_xdata([tvec[i], tvec[i]])
        txt.set_text(f"t = {tvec[i]:7.1f}")
        return []

    ani = animation.FuncAnimation(fig, upd, frames=nF, blit=False)
    ani.save(path, writer=animation.FFMpegWriter(fps=FPS, bitrate=3200))
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    c = load_cortex("fsaverage5", verbose=False)
    proj = _proj(c.V, c.F)
    man = json.load(open(os.path.join(FIELDS, "manifest.json")))
    by_pi = {}
    for r in man:
        by_pi.setdefault(r["pi"], {})[r["cond"]] = r
    for pi in sorted(by_pi):
        t0 = time.time()
        path = os.path.join(OUT, f"regime_p{pi:02d}.mp4")
        render(pi, by_pi[pi], proj, path)
        print(f"  wrote {os.path.basename(path)}  [{time.time()-t0:.0f}s]", flush=True)
    print(f"\n  {len(by_pi)} videos in {OUT}")


if __name__ == "__main__":
    main()
