"""How input timing steers waves: one patch, its four steering patterns, cause and effect.

For a single 10 mm patch on the lateral S1 strip, the four input patterns (steering modes,
per unit of injection per mm^2) that drive the most energy flow through it

    up the strip | down the strip | out of the patch (radiate) | into it (converge)

at one frequency, each drawn twice: the INPUT - when each place is driven hardest within
the cycle - and the RESULT - when the wave peaks at each place, with the time-averaged
energy flow drawn as arrows round the patch. Timing is in seconds within the cycle,
measured from when the wave peaks at the patch itself, so "the bottom of the strip is
driven 3 s before the wave passes the patch" is read straight off the colours.

A static figure and an animation over one cycle (input left, wave right, blue below
zero and red above), so the sweep of the inputs and the travel of the wave can be watched.

  python viz/render_steering.py --tag nobp_sm6_s3_box
"""
import _path  # noqa: F401  - puts the sibling code folders on sys.path
import os, argparse
import numpy as np

from paths import RESULTS
import waves, interplay

INK, MUTED, BG = "#1a1a19", "#6b6a64", "#e4e2dc"
DIVERGING = ("#184f95", "#6da7ec", "#f0efec", "#ec8a88", "#a12a29")
ROWS = (("along", "+", "pushes waves UP the strip"),
        ("along", "-", "pushes waves DOWN the strip"),
        ("expand", "+", "makes the patch RADIATE"),
        ("expand", "-", "makes the patch CONVERGE"))


def view_basis(elev=8.0, azim=180.0):
    """The 2-D projection render_regimes._proj uses, as (right, up, eye) vectors."""
    e, a = np.radians(elev), np.radians(azim)
    d = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    right = np.array([-np.sin(a), np.cos(a), 0.0])
    return right, np.cross(d, right), d


def pick_patch(c, S, run, radius, right, up, eye):
    """The S1-strip vertex deepest inside the driven strip, near its mid-height, facing the
    lateral view; and the patch round it, its axis pointing up the strip."""
    from scipy.sparse.csgraph import dijkstra
    names = np.array([str(v).removeprefix("L_").removesuffix("_ROI") for v in c.names])
    s1 = np.isin(names[np.asarray(c.lab)], ["3a", "3b", "1", "2"])
    s1 &= np.asarray(run["labels"]) >= 0
    nrm = np.asarray(c.m.vertex_normal)
    facing = nrm @ eye > 0.6
    edge = S.distance_from(np.flatnonzero(~s1))
    z = S.V[:, 2]
    zmid = np.median(z[s1 & facing])
    score = np.where(s1 & facing & (edge >= 3.0), -np.abs(z - zmid) + 0.5 * edge, -np.inf)
    q = int(np.argmax(score))
    d = dijkstra(S.graph(), indices=[q])[0]
    pv = np.flatnonzero(d <= radius)
    n = nrm[pv].mean(0); n /= np.linalg.norm(n)
    # "up the strip": the strip's own long axis near the patch, oriented dorsally
    near = np.flatnonzero(s1 & (d < 25))
    X = S.V[near] - S.V[near].mean(0)
    ax = np.linalg.svd(X - (X @ n)[:, None] * n, full_matrices=False)[2][0]
    if ax[2] < 0:
        ax = -ax
    e1 = ax - (ax @ n) * n; e1 /= np.linalg.norm(e1)
    return dict(kind="grid", pv=pv, centre=S.V[q], vertex=q, e1=e1, e2=np.cross(n, e1),
                n=n, dist=d, euc=np.linalg.norm(S.V - S.V[q], axis=1))


def faded(colors, amp, gamma=0.7):
    """Blend toward the background grey where the amplitude is small, so strength is read
    as how much colour there is, never as how dark it is."""
    a = np.clip(amp, 0, 1)[:, None] ** gamma
    bg = np.array([int(BG[i:i + 2], 16) / 255 for i in (1, 3, 5)])
    return colors[:, :3] * a + bg * (1 - a)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", default="nobp_sm6_s3_box")
    ap.add_argument("--hz", type=float, default=0.0848)
    ap.add_argument("--radius", type=float, default=10.0)
    ap.add_argument("--frames", type=int, default=36)
    ap.add_argument("--fps", type=int, default=12)
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    from matplotlib.tri import Triangulation
    import cmocean
    from mesh_cache import load_cortex

    c = load_cortex("fsaverage5", verbose=False)
    run = waves.load_run(c, a.tag)
    S = waves.Surface(c)
    right, up, eye = view_basis()
    P = pick_patch(c, S, run, a.radius, right, up, eye)
    hzb = run["idx"] / run["ref_frames"] / run["frame_s"]
    grid = np.unique([int(np.argmin(np.abs(np.log(hzb / f))))
                      for f in np.geomspace(0.005, 0.5, 14)])
    H, U, depth = waves.transfer_fields(c, run, hzb[grid], velocity=True)
    fi = int(np.argmin(np.abs(hzb[grid] - a.hz)))
    hz = float(hzb[grid][fi])
    T, om = 1.0 / hz, 2 * np.pi * hz
    Pf = np.asarray(run["profiles"], np.float32)
    area, nrm = np.asarray(c.A), np.asarray(c.m.vertex_normal)
    M = interplay.steering_matrices(P, H[fi], U[fi], depth, area, nrm, S.V)
    W = interplay.area_metric(c, Pf)
    lab = np.asarray(run["labels"])
    names = np.array([str(v).removeprefix("L_").removesuffix("_ROI") for v in c.names])
    print(f"  patch at {names[c.lab[P['vertex']]]}, {hz:.4f} Hz (cycle {T:.1f} s)")

    xy = np.c_[S.V @ right, S.V @ up]
    vis = (nrm @ eye) > -0.1
    keep = vis[c.F].all(1)
    tri = Triangulation(xy[:, 0], xy[:, 1], c.F[keep])
    Fk = c.F[keep]
    polys = xy[Fk]
    cyc = cmocean.cm.phase
    div = LinearSegmentedColormap.from_list("d", DIVERGING)

    modes = []
    for nm, sgn, title in ROWS:
        lam, x = interplay.area_modes(M[nm], W)[sgn]
        h = H[fi] @ x
        u = np.einsum("vck,k->vc", U[fi], x)
        # one clock for the row: t = 0 when the wave peaks at the patch centre
        x = x * np.exp(-1j * np.angle(h[P["vertex"]]))
        h, u = H[fi] @ x, np.einsum("vck,k->vc", U[fi], x)
        qin = x @ Pf
        F = depth[:, None] * 0.5 * np.real(np.conj(h)[:, None] * u)
        tin = (-np.angle(qin) / om) % T
        th = (-np.angle(h) / om) % T
        flow_axis = float(np.real(x.conj() @ M[nm] @ x))
        modes.append(dict(title=title, x=x, qin=qin, h=h, F=F, tin=tin, th=th, nm=nm,
                          sgn=sgn, flow=flow_axis))
        a_in = np.abs(qin) / np.percentile(np.abs(qin)[lab >= 0], 98)
        w_in = np.abs(qin) ** 2 * area
        lag_up = (S.V @ P["e1"])
        # where the strongly driven input peaks early vs late, along the strip
        m_ = (a_in > 0.3)
        r = (np.corrcoef(lag_up[m_], ((tin[m_] - T / 2) % T)) [0, 1]
             if m_.sum() > 10 else np.nan)
        print(f"    {title:28s}: {m_.sum():4d} strongly driven vertices, "
              f"{np.sqrt((w_in * P['dist'] ** 2).sum() / w_in.sum()):.0f} mm rms from the "
              f"patch; r(height up the strip, input peak time) {r:+.2f}")

    # ------------------------------------------------------------ static figure
    def draw_surface(ax, rgb):
        fc = rgb[Fk].mean(1)
        ax.add_collection(PolyCollection(polys, facecolors=fc, edgecolors=fc,
                                         linewidths=0.2))

    def furniture(ax, arrow=True):
        ax.tricontour(tri, (lab >= 0).astype(float), levels=[0.5], colors=MUTED,
                      linewidths=0.5)
        ax.tricontour(tri, P["euc"], levels=[a.radius], colors=INK, linewidths=1.6)
        if arrow:
            p0 = P["centre"] @ np.c_[right, up]
            d2 = P["e1"] @ np.c_[right, up]
            ax.annotate("", p0 + 18 * d2, p0, arrowprops=dict(arrowstyle="-|>", lw=1.6,
                                                              color=MUTED, ls="--"))
        ax.set_xlim(xy[vis, 0].min(), xy[vis, 0].max())
        ax.set_ylim(xy[vis, 1].min(), xy[vis, 1].max())
        ax.set_aspect("equal"); ax.axis("off")

    fig = plt.figure(figsize=(13.5, 15.5), facecolor="white")
    gs = fig.add_gridspec(len(ROWS) + 1, 3, width_ratios=(1, 1, 0.32),
                          height_ratios=[1] * len(ROWS) + [0.05], hspace=0.12, wspace=0.03)
    near = np.flatnonzero((P["dist"] < 32) & vis)
    # thin the arrows to one per ~7 mm
    pick, taken = [], np.zeros(len(near), bool)
    for i_ in np.argsort(P["dist"][near]):
        if not taken[i_]:
            pick.append(near[i_])
            taken |= np.linalg.norm(S.V[near] - S.V[near[i_]], axis=1) < 7.0
    pick = np.array(pick)
    for r_, md in enumerate(modes):
        ax = fig.add_subplot(gs[r_, 0])
        a_in = np.abs(md["qin"]) / np.percentile(np.abs(md["qin"])[lab >= 0], 98)
        rgb = faded(cyc(md["tin"] / T), np.where(lab >= 0, a_in, 0.0))
        draw_surface(ax, rgb); furniture(ax)
        ax.text(-0.02, 0.5, md["title"], transform=ax.transAxes, rotation=90, ha="right",
                va="center", fontsize=11, color=INK)
        if r_ == 0:
            ax.set_title("INPUT: when each place is driven hardest\n(faded = barely driven)",
                         fontsize=10, color=INK)
        ax = fig.add_subplot(gs[r_, 1])
        a_h = np.abs(md["h"]) / np.percentile(np.abs(md["h"])[vis], 99)
        draw_surface(ax, faded(cyc(md["th"] / T), a_h))
        furniture(ax)
        Fp = md["F"][pick]
        d2 = Fp @ np.c_[right, up]
        L = np.linalg.norm(Fp, axis=1)
        sc = 9.0 / np.percentile(L, 95)
        ax.quiver(xy[pick, 0], xy[pick, 1], d2[:, 0] * sc, d2[:, 1] * sc, angles="xy",
                  scale_units="xy", scale=1, width=0.0045, headwidth=3.5, color=INK,
                  alpha=0.8)
        # the patch's own net flow, bold
        net = md["F"][P["pv"]] * area[P["pv"], None]
        nv = net.sum(0) @ np.c_[right, up]
        dirn = np.linalg.norm(net.sum(0)) / max(np.linalg.norm(net, axis=1).sum(), 1e-30)
        p0 = P["centre"] @ np.c_[right, up]
        if md["nm"] == "along" and dirn > 0.25:          # a net direction only means
                                                          # something for a push one way
            ax.annotate("", p0 + 22 * nv / np.linalg.norm(nv), p0,
                        arrowprops=dict(arrowstyle="-|>", lw=3.2, color="#e34948"))
        if r_ == 0:
            ax.set_title("RESULT: when the wave peaks; arrows = where its energy flows",
                         fontsize=10, color=INK)
    # the clock legend: a ring of peak times
    lax = fig.add_subplot(gs[0, 2], projection="polar")
    th_ = np.linspace(0, 2 * np.pi, 361)
    lax.scatter(th_, np.ones_like(th_), c=cyc(th_ / (2 * np.pi)), s=60)
    lax.set_theta_zero_location("N"); lax.set_theta_direction(-1)
    lax.set_yticks([]); lax.set_ylim(0, 1.2)
    tk = np.arange(0, T, 3.0)
    lax.set_xticks(2 * np.pi * tk / T)
    lax.set_xticklabels([f"{v:.0f} s" for v in tk], fontsize=8, color=INK)
    lax.spines["polar"].set_visible(False)
    lax.set_title(f"peak time within the\n{T:.1f} s cycle\n(0 = wave peaks at the patch)",
                  fontsize=8, color=INK, pad=12)
    nax = fig.add_subplot(gs[1:3, 2]); nax.axis("off")
    nax.text(0, 1, "black circle: the 10 mm patch\ndashed arrow: 'up the strip'\nred arrow: "
             "the patch's\nnet energy flow\ngrey outline: "
             "where input can\nbe injected\n\neach row is the input\npattern that, per "
             "unit of\ninjection, drives the most\nenergy flow through the\npatch in that "
             f"way, at\n{hz:.3f} Hz, in this medium", va="top", fontsize=8.5, color=INK)
    fig.subplots_adjust(top=0.93)
    fig.suptitle("How input timing steers waves through one patch of the S1 strip",
                 fontsize=13, color=INK, y=0.975)
    out = os.path.join(RESULTS, f"steering_{run['name']}")
    fig.savefig(out + ".png", dpi=130, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}.png")

    # ------------------------------------------------------------ animation
    from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
    fig, axs = plt.subplots(2, 4, figsize=(16, 7.6), facecolor="white")
    arts = []
    for k, md in enumerate(modes):
        ai, aw = axs[k // 2, 2 * (k % 2)], axs[k // 2, 2 * (k % 2) + 1]
        si = np.percentile(np.abs(md["qin"])[lab >= 0], 98)
        sw = np.percentile(np.abs(md["h"])[vis], 99)
        ti = ai.tripcolor(tri, np.zeros(c.nV), cmap=div, norm=TwoSlopeNorm(0, -si, si),
                          shading="gouraud", rasterized=True)
        tw = aw.tripcolor(tri, np.zeros(c.nV), cmap=div, norm=TwoSlopeNorm(0, -sw, sw),
                          shading="gouraud", rasterized=True)
        for ax in (ai, aw):
            furniture(ax)
        Fp = md["F"][pick]; d2 = Fp @ np.c_[right, up]
        sc = 7.0 / np.percentile(np.linalg.norm(Fp, axis=1), 95)
        aw.quiver(xy[pick, 0], xy[pick, 1], d2[:, 0] * sc, d2[:, 1] * sc, angles="xy",
                  scale_units="xy", scale=1, width=0.0028, color=INK, alpha=0.6)
        ai.set_title(f"{md['title']}\ninput", fontsize=9, color=INK)
        aw.set_title("wave", fontsize=9, color=INK)
        arts.append((ti, tw, md))
    stamp = fig.text(0.5, 0.01, "", ha="center", fontsize=10, color=INK)
    fig.suptitle(f"input (left of each pair) and the wave it makes (right), over one "
                 f"{T:.1f} s cycle at {hz:.3f} Hz - blue below zero, red above",
                 fontsize=11, color=INK)

    def frame(n):
        t = n * T / a.frames
        ph = np.exp(1j * om * t)
        for ti, tw, md in arts:
            ti.set_array(np.where(lab >= 0, np.real(md["qin"] * ph), 0.0))
            tw.set_array(np.real(md["h"] * ph))
        stamp.set_text(f"t = {t:4.1f} s")
        return [a_[0] for a_ in arts] + [a_[1] for a_ in arts] + [stamp]

    anim = FuncAnimation(fig, frame, frames=a.frames, blit=False)
    anim.save(out + ".gif", writer=PillowWriter(fps=a.fps), dpi=80)
    print(f"  wrote {out}.gif")
    try:
        anim.save(out + ".mp4", writer=FFMpegWriter(fps=a.fps), dpi=110)
        print(f"  wrote {out}.mp4")
    except Exception as e:                                  # no ffmpeg: the gif stands
        print(f"  no mp4 ({e})")
    plt.close(fig)


if __name__ == "__main__":
    main()
