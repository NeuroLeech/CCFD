"""How input timing steers waves through one patch: cause and effect, on the surface.

For a single 10 mm patch, at one frequency, the input patterns (steering modes, per unit of
injection per mm^2) that drive the most energy flow through it in a given way, each drawn
twice: the INPUT - when each place is driven hardest within the cycle - and the RESULT -
when the wave peaks at each place, with the time-averaged energy flow as arrows round the
patch. Times are seconds within the cycle, from when the wave peaks at the patch itself.

  --site s1       a patch mid-way along the lateral S1 strip, inside the input territory:
                  the patterns that push its waves up the strip, down it, make it radiate,
                  and make it converge
  --site frontal  a patch on lateral frontal cortex near the top of the principal gradient,
                  far from any input: the patterns that push its waves up and down the
                  gradient; then what the FITTED drive does there. The fitted drive is a sum
                  of independent timing patterns (eigenvectors of its cross-spectrum S), and
                  the patch's average flow splits over them exactly, with no cross terms; the
                  one carrying most of it is drawn, and its wave is split into the input
                  regions that produce it - each region's wave alone, and everything else -
                  with how much of the patch's flow each gives alone and how much only their
                  interference gives

A static figure and an animation over one cycle (blue below zero, red above).

  python viz/render_steering.py --site s1 --hz 0.0848
  python viz/render_steering.py --site frontal --hz 0.2513
"""
import _path  # noqa: F401  - puts the sibling code folders on sys.path
import os, argparse
import numpy as np

from paths import RESULTS, CACHE
import waves, interplay

INK, MUTED, BG = "#1a1a19", "#6b6a64", "#e4e2dc"
DIVERGING = ("#184f95", "#6da7ec", "#f0efec", "#ec8a88", "#a12a29")
REGION = ("#2a78d6", "#eb6834")            # categorical slots 1 and 2: region A, region B
NET = "#e34948"


def view_basis(elev=8.0, azim=180.0):
    """The 2-D projection render_regimes._proj uses, as (right, up, eye) vectors."""
    e, a = np.radians(elev), np.radians(azim)
    d = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    right = np.array([-np.sin(a), np.cos(a), 0.0])
    return right, np.cross(d, right), d


def _patch(S, c, q, radius, ax):
    """The patch round vertex q, its axis `ax` made tangent."""
    from scipy.sparse.csgraph import dijkstra
    d = dijkstra(S.graph(), indices=[q])[0]
    pv = np.flatnonzero(d <= radius)
    nrm = np.asarray(c.m.vertex_normal)
    n = nrm[pv].mean(0); n /= np.linalg.norm(n)
    e1 = ax - (ax @ n) * n; e1 /= np.linalg.norm(e1)
    return dict(kind="grid", pv=pv, centre=S.V[q], vertex=int(q), e1=e1,
                e2=np.cross(n, e1), n=n, dist=d, euc=np.linalg.norm(S.V - S.V[q], axis=1))


def pick_s1(c, S, run, radius, eye):
    """Mid-height on the lateral S1 strip, deep inside it; axis up the strip."""
    names = np.array([str(v).removeprefix("L_").removesuffix("_ROI") for v in c.names])
    s1 = np.isin(names[np.asarray(c.lab)], ["3a", "3b", "1", "2"])
    s1 &= np.asarray(run["labels"]) >= 0
    facing = np.asarray(c.m.vertex_normal) @ eye > 0.6
    edge = S.distance_from(np.flatnonzero(~s1))
    z = S.V[:, 2]
    zmid = np.median(z[s1 & facing])
    q = int(np.argmax(np.where(s1 & facing & (edge >= 3.0), -np.abs(z - zmid) + 0.5 * edge,
                               -np.inf)))
    near = np.flatnonzero(s1 & (S.distance_from([q]) < 25))
    X = S.V[near] - S.V[near].mean(0)
    ax = np.linalg.svd(X, full_matrices=False)[2][0]
    return _patch(S, c, q, radius, ax if ax[2] >= 0 else -ax)


def pick_frontal(c, S, run, radius, eye, g1, min_dist=30.0):
    """Lateral frontal cortex, facing the view, at least `min_dist` mm from any input, as
    high on the principal gradient as possible; axis up the gradient."""
    lab = np.asarray(run["labels"])
    dterr = S.distance_from(np.flatnonzero(lab >= 0))
    facing = np.asarray(c.m.vertex_normal) @ eye > 0.6
    front = S.V[:, 1] > np.percentile(S.V[:, 1], 62)
    # the gradient smoothed over ~10 mm, so one vertex's value does not pick the patch
    g_s = np.asarray(S.K @ g1)
    q = int(np.argmax(np.where(facing & front & (dterr >= min_dist), g_s, -np.inf)))
    P = _patch(S, c, q, radius, np.array([0.0, 1.0, 0.0]))
    inside = np.isin(S.F, P["pv"]).all(1)
    ax = S.scalar_grad(g1)[inside].mean(0)
    P2 = _patch(S, c, q, radius, ax)
    P2["dterr"] = float(dterr[q])
    return P2


def faded(colors, amp, gamma=0.7):
    """Blend toward the background grey where the amplitude is small, so strength is read
    as how much colour there is, never as how dark it is."""
    a = np.clip(amp, 0, 1)[:, None] ** gamma
    bg = np.array([int(BG[i:i + 2], 16) / 255 for i in (1, 3, 5)])
    return colors[:, :3] * a + bg * (1 - a)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", default="nobp_sm6_s3_box")
    ap.add_argument("--site", default="s1", choices=("s1", "frontal"))
    ap.add_argument("--hz", type=float, default=0.0848)
    ap.add_argument("--radius", type=float, default=10.0)
    ap.add_argument("--frames", type=int, default=36)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--no-anim", action="store_true", dest="no_anim")
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
    R2 = np.c_[right, up]
    lab = np.asarray(run["labels"])
    names = np.array([str(v).removeprefix("L_").removesuffix("_ROI") for v in c.names])
    if a.site == "s1":
        P = pick_s1(c, S, run, a.radius, eye)
        rows = (("along", "+", "pushes waves UP the strip"),
                ("along", "-", "pushes waves DOWN the strip"),
                ("expand", "+", "makes the patch RADIATE"),
                ("expand", "-", "makes the patch CONVERGE"))
        axis_label = "'up the strip'"
    else:
        g1 = np.load(os.path.join(CACHE, f"cortical_maps_{c.mesh}.npz"))["grad1"]
        P = pick_frontal(c, S, run, a.radius, eye, g1)
        rows = (("along", "+", "pushes waves UP the gradient\n(toward transmodal)"),
                ("along", "-", "pushes waves DOWN the gradient\n(toward sensorimotor)"))
        axis_label = "'up the principal gradient'"
    hzb = run["idx"] / run["ref_frames"] / run["frame_s"]
    grid = np.unique([int(np.argmin(np.abs(np.log(hzb / f))))
                      for f in np.geomspace(0.005, 0.5, 14)])
    H, U, depth = waves.transfer_fields(c, run, hzb[grid], velocity=True)
    fi = int(np.argmin(np.abs(hzb[grid] - a.hz)))
    hz = float(hzb[grid][fi])
    T, om = 1.0 / hz, 2 * np.pi * hz
    Hf, Uf = H[fi], U[fi]
    Pf = np.asarray(run["profiles"], np.float32)
    area, nrm = np.asarray(c.A), np.asarray(c.m.vertex_normal)
    M = interplay.steering_matrices(P, Hf, Uf, depth, area, nrm, S.V)
    W = interplay.area_metric(c, Pf)
    extra = f", {P['dterr']:.0f} mm from the nearest input" if "dterr" in P else ""
    print(f"  {a.site} patch at {names[c.lab[P['vertex']]]}{extra}; {hz:.4f} Hz "
          f"(cycle {T:.1f} s)")

    def pattern(z, title, ref=None):
        """A drive pattern z (K,) as surface fields, on one clock: t = 0 when the wave
        peaks at the patch centre (or at `ref`'s clock when given)."""
        h = Hf @ z
        rot = np.exp(-1j * np.angle(h[P["vertex"]])) if ref is None else ref
        z = z * rot
        h = Hf @ z
        u = np.einsum("vck,k->vc", Uf, z)
        qin = z @ Pf
        return dict(title=title, z=z, qin=qin, h=h, rot=rot,
                    F=depth[:, None] * 0.5 * np.real(np.conj(h)[:, None] * u),
                    tin=(-np.angle(qin) / om) % T, th=(-np.angle(h) / om) % T)

    panels = []                              # rows of (kind, pattern, extras)
    for nm, sgn, title in rows:
        lam, x = interplay.area_modes(M[nm], W)[sgn]
        md = pattern(x, title); md["nm"] = nm
        panels.append([("input", md, {}), ("wave", md, {"net": nm == "along"})])
        w_in = np.abs(md["qin"]) ** 2 * area
        print(f"    {title.splitlines()[0]:30s}: inputs {np.sqrt((w_in * P['dist'] ** 2).sum() / w_in.sum()):.0f} mm rms from the patch")

    story = []
    if a.site == "frontal":
        # what the fitted drive does here: its independent timing patterns, and their shares
        b = grid[fi]
        Sf = 0.5 * (run["S"][b] + run["S"][b].conj().T)
        Fa, Fc = (float(np.real(np.trace(M[k] @ Sf))) for k in ("along", "across"))
        n_hat = np.array([Fa, Fc]) / np.hypot(Fa, Fc)
        Mn = n_hat[0] * M["along"] + n_hat[1] * M["across"]
        ev, Ux = np.linalg.eigh(Sf)
        ev = np.clip(ev, 0, None)
        contrib = ev * np.real(np.einsum("ki,kl,li->i", Ux.conj(), Mn, Ux))
        total = contrib.sum()
        o = np.argsort(-contrib)
        n80 = int(np.searchsorted(np.cumsum(contrib[o]) / total, 0.8) + 1)
        i0 = o[0]
        z = np.sqrt(ev[i0]) * Ux[:, i0]
        up_ = Fa / np.hypot(Fa, Fc)
        md = pattern(z, "the fitted drive's\nmain pattern here")
        share = contrib[i0] / total
        # split its flow at the patch over the inputs, and find the regions that carry it
        Szz = np.outer(md["z"], md["z"].conj())
        ck, _, _ = interplay.source_map(M, Szz, n_hat)
        D, cen, adj = interplay.geometry(c, S, run["labels"], len(Pf))
        regs = [r for r in interplay.source_regions(ck, adj) if r[1] > 0][:2]
        flow = lambda zz: float(np.real(zz.conj() @ Mn @ zz))
        f_all = flow(md["z"])
        parts, used = [], np.zeros(len(Pf), bool)
        for (mem, _), col, nm_ in zip(regs, REGION, "AB"):
            zz = np.zeros_like(md["z"]); zz[mem] = md["z"][mem]; used[mem] = True
            pt = pattern(zz, f"region {nm_} alone", ref=1.0)
            pt.update(col=col, mem=mem, share=flow(zz) / f_all,
                      dist=float(np.mean(D[mem, P["vertex"]])))
            parts.append(pt)
        zz = np.where(used, 0, md["z"])
        rest = pattern(zz, "everything else", ref=1.0)
        rest.update(col=None, mem=None, share=flow(zz) / f_all)
        inter = 1 - sum(p_["share"] for p_ in parts) - rest["share"]
        panels.append([("input", md, {"regions": parts}), ("wave", md, {"net": True})])
        panels.append([("wave", p_, {"region": p_, "ref": md}) for p_ in parts]
                      + [("wave", rest, {"ref": md})])
        # the net flow's direction, in gradient terms and in anatomical ones
        fvec = n_hat[0] * P["e1"] + n_hat[1] * P["e2"]
        rel = ("UP the gradient" if up_ > 0.3 else "DOWN the gradient" if up_ < -0.3
               else "ACROSS the gradient")
        anat = " and ".join(nm_ for v_, nm_ in ((fvec[1], "forward"), (-fvec[1], "backward"),
                                                (fvec[2], "upward"), (-fvec[2], "downward"))
                            if v_ > 0.4)
        story = [f"the fitted drive at {hz:.3f} Hz makes this patch's waves flow",
                 f"{rel} (cos {up_:+.2f}), i.e. {anat or 'obliquely'} on the cortex;",
                 f"of its independent timing patterns, the strongest here carries "
                 f"{share:.0%} of that flow ({n80} carry 80%)",
                 "", "that pattern's wave at the patch, split by where its inputs are:"]
        for p_, nm_ in zip(parts, "AB"):
            story.append(f"  region {nm_} ({len(p_['mem'])} pieces, ~{p_['dist']:.0f} mm "
                         f"away) alone: {p_['share']:+.0%}")
        story += [f"  everything else alone: {rest['share']:+.0%}",
                  f"  interference between them: {inter:+.0%}"]
        for line in story:
            print("    " + line)

    # ------------------------------------------------------------ drawing
    xy = np.c_[S.V @ right, S.V @ up]
    vis = (nrm @ eye) > -0.1
    keep = vis[c.F].all(1)
    tri = Triangulation(xy[:, 0], xy[:, 1], c.F[keep])
    Fk = c.F[keep]
    polys = xy[Fk]
    cyc = cmocean.cm.phase
    div = LinearSegmentedColormap.from_list("d", DIVERGING)
    near = np.flatnonzero((P["dist"] < 32) & vis)
    pick, taken = [], np.zeros(len(near), bool)
    for i_ in np.argsort(P["dist"][near]):
        if not taken[i_]:
            pick.append(near[i_])
            taken |= np.linalg.norm(S.V[near] - S.V[near[i_]], axis=1) < 7.0
    pick = np.array(pick)
    local = P["dist"] < 40                    # waves are faded against their size HERE

    def draw_surface(ax, rgb):
        fc = rgb[Fk].mean(1)
        ax.add_collection(PolyCollection(polys, facecolors=fc, edgecolors=fc,
                                         linewidths=0.2))

    def furniture(ax, region=None):
        ax.tricontour(tri, (lab >= 0).astype(float), levels=[0.5], colors=MUTED,
                      linewidths=0.5)
        for rg in ([region] if isinstance(region, dict) else region or []):
            ax.tricontour(tri, np.isin(lab, rg["mem"]).astype(float), levels=[0.5],
                          colors=rg["col"], linewidths=2.0)
        ax.tricontour(tri, P["euc"], levels=[a.radius], colors=INK, linewidths=1.6)
        p0 = P["centre"] @ R2
        ax.annotate("", p0 + 18 * (P["e1"] @ R2), p0,
                    arrowprops=dict(arrowstyle="-|>", lw=1.6, color=MUTED, ls="--"))
        ax.set_xlim(xy[vis, 0].min(), xy[vis, 0].max())
        ax.set_ylim(xy[vis, 1].min(), xy[vis, 1].max())
        ax.set_aspect("equal"); ax.axis("off")

    def arrows(ax, md, net, scale_from=None):
        Fp = md["F"][pick]
        d2 = Fp @ R2
        ref = (scale_from or md)["F"][pick]
        sc = 9.0 / max(np.percentile(np.linalg.norm(ref, axis=1), 95), 1e-30)
        # on another pattern's scale an arrow can be many times longer; cap it at 2x, so a
        # part that alone pushes harder than the whole still reads as harder, not as noise
        L = np.linalg.norm(d2, axis=1) * sc
        d2 = d2 * (np.minimum(L, 18.0) / np.maximum(L, 1e-30))[:, None]
        ax.quiver(xy[pick, 0], xy[pick, 1], d2[:, 0] * sc, d2[:, 1] * sc, angles="xy",
                  scale_units="xy", scale=1, width=0.0045, headwidth=3.5, color=INK,
                  alpha=0.8)
        if net:
            nf = (md["F"][P["pv"]] * area[P["pv"], None])
            dirn = np.linalg.norm(nf.sum(0)) / max(np.linalg.norm(nf, axis=1).sum(), 1e-30)
            if dirn > 0.25:
                v = nf.sum(0) @ R2
                p0 = P["centre"] @ R2
                ax.annotate("", p0 + 22 * v / np.linalg.norm(v), p0,
                            arrowprops=dict(arrowstyle="-|>", lw=3.2, color=NET))

    def draw(ax, kind, md, ex):
        if kind == "input":
            a_in = np.abs(md["qin"]) / np.percentile(np.abs(md["qin"])[lab >= 0], 98)
            draw_surface(ax, faded(cyc(md["tin"] / T), np.where(lab >= 0, a_in, 0.0)))
            furniture(ax, ex.get("regions"))
        else:
            refmd = ex.get("ref", md)
            amp = np.abs(md["h"]) / np.percentile(np.abs(refmd["h"])[local & vis], 95)
            draw_surface(ax, faded(cyc(md["th"] / T), amp))
            furniture(ax, ex.get("region"))
            arrows(ax, md, ex.get("net", False), scale_from=ex.get("ref"))

    ncol = 3
    nrow = len(panels)
    fig = plt.figure(figsize=(6.6 * ncol, 4.1 * nrow + 0.6), facecolor="white")
    H_in = 4.1 * nrow + 0.6
    gs = fig.add_gridspec(nrow, ncol, hspace=0.16, wspace=0.03, top=1 - 1.6 / H_in)
    for r_, row in enumerate(panels):
        for k, (kind, md, ex) in enumerate(row):
            ax = fig.add_subplot(gs[r_, k])
            draw(ax, kind, md, ex)
            if k == 0 and len(row) == 2:
                ax.text(-0.02, 0.5, md["title"], transform=ax.transAxes, rotation=90,
                        ha="right", va="center", fontsize=11, color=INK)
            if len(row) == 2:
                ttl = ("INPUT: when each place is driven hardest (faded = barely driven)"
                       if kind == "input" else
                       "RESULT: when the wave peaks; arrows = where its energy flows")
                if r_ == 0 or (a.site == "frontal" and r_ == len(rows)):
                    ax.set_title(ttl, fontsize=10, color=INK)
            else:
                share = md.get("share", np.nan)
                ax.set_title(f"{md['title']}: {share:+.0%} of this pattern's flow at the patch",
                             fontsize=10, color=md.get("col") or INK)
        if len(row) == 2:
            if r_ == 0:
                lax = fig.add_subplot(gs[0, 2], projection="polar")
                th_ = np.linspace(0, 2 * np.pi, 361)
                lax.scatter(th_, np.ones_like(th_), c=cyc(th_ / (2 * np.pi)), s=60)
                lax.set_theta_zero_location("N"); lax.set_theta_direction(-1)
                lax.set_yticks([]); lax.set_ylim(0, 1.2)
                tk = np.linspace(0, T, 5)[:-1]
                lax.set_xticks(2 * np.pi * tk / T)
                lax.set_xticklabels([f"{v:.1f} s" for v in tk], fontsize=8, color=INK)
                lax.spines["polar"].set_visible(False)
                lax.set_title(f"peak time within the {T:.1f} s cycle\n"
                              f"(0 = wave peaks at the patch)", fontsize=9, color=INK, pad=14)
            elif r_ == 1:
                nax = fig.add_subplot(gs[1, 2]); nax.axis("off")
                nax.text(0.04, 0.95, "black circle: the 10 mm patch\n"
                         f"dashed arrow: {axis_label}\nred arrow: the patch's net "
                         "energy flow\ngrey outline: where input can be injected\n\n"
                         "rows above: the input pattern that,\nper unit of injection, drives "
                         "the most\nenergy flow through the patch that way",
                         va="top", fontsize=9, color=INK, transform=nax.transAxes)
            elif story and r_ == len(rows):
                nax = fig.add_subplot(gs[r_, 2]); nax.axis("off")
                nax.text(0.04, 0.95, "\n".join(story), va="top", fontsize=9, color=INK,
                         transform=nax.transAxes)
    site = ("one patch of the S1 strip" if a.site == "s1" else
            f"a frontal patch ({names[c.lab[P['vertex']]]}), {P.get('dterr', 0):.0f} mm "
            f"from any input")
    fig.suptitle(f"How input timing steers waves through {site}, at {hz:.3f} Hz "
                 f"({T:.1f} s cycle)", fontsize=14, color=INK, y=1 - 0.2 / H_in)
    out = os.path.join(RESULTS, f"steering_{a.site}_{run['name']}_{hz:.3f}Hz")
    fig.savefig(out + ".png", dpi=120, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}.png")
    if a.no_anim:
        return

    # ------------------------------------------------------------ animation
    from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
    cells = []
    for row in panels:
        for kind, md, ex in row:
            cells.append((kind, md, ex))
    ncol_a = 4
    nrow_a = int(np.ceil(len(cells) / ncol_a))
    fig, axs = plt.subplots(nrow_a, ncol_a, figsize=(4 * ncol_a, 3.6 * nrow_a + 0.4),
                            facecolor="white", squeeze=False)
    arts = []
    for ax, (kind, md, ex) in zip(axs.ravel(), cells):
        if kind == "input":
            vals, sc = md["qin"], np.percentile(np.abs(md["qin"])[lab >= 0], 98)
        else:
            vals = md["h"]
            sc = np.percentile(np.abs(ex.get("ref", md)["h"])[local & vis], 95)
        tp = ax.tripcolor(tri, np.zeros(c.nV), cmap=div, norm=TwoSlopeNorm(0, -sc, sc),
                          shading="gouraud", rasterized=True)
        furniture(ax, ex.get("regions") if kind == "input" else ex.get("region"))
        if kind == "wave":
            arrows(ax, md, ex.get("net", False), scale_from=ex.get("ref"))
        ttl = md["title"].replace("\n", " ")
        ax.set_title(f"{ttl}\n{'input' if kind == 'input' else 'wave'}", fontsize=8,
                     color=md.get("col") or INK)
        arts.append((tp, vals, kind == "input"))
    for ax in axs.ravel()[len(cells):]:
        ax.axis("off")
    stamp = fig.text(0.5, 0.01, "", ha="center", fontsize=10, color=INK)
    fig.suptitle(f"input and the wave it makes over one {T:.1f} s cycle at {hz:.3f} Hz - "
                 f"blue below zero, red above", fontsize=11, color=INK)

    def frame(n):
        t = n * T / a.frames
        ph = np.exp(1j * om * t)
        for tp, vals, is_in in arts:
            v = np.real(vals * ph)
            tp.set_array(np.where(lab >= 0, v, 0.0) if is_in else v)
        stamp.set_text(f"t = {t:4.2f} s")
        return [x[0] for x in arts] + [stamp]

    anim = FuncAnimation(fig, frame, frames=a.frames, blit=False)
    anim.save(out + ".gif", writer=PillowWriter(fps=a.fps), dpi=70)
    print(f"  wrote {out}.gif")
    try:
        anim.save(out + ".mp4", writer=FFMpegWriter(fps=a.fps), dpi=100)
        print(f"  wrote {out}.mp4")
    except Exception as e:                                  # no ffmpeg: the gif stands
        print(f"  no mp4 ({e})")
    plt.close(fig)


if __name__ == "__main__":
    main()
