"""The adaptive medium driven by a fitted drive: the raw field, its energy, the observable, and
the damping gain, as a movie.

One draw (577 s, seed 10000 - the first of adaptive_fit's authoritative draws) of a drive saved
by fit/adaptive_fit.py, run over the full sheet by torch_swe.run_adaptive with the run's own
k, tau_a and a_ref. Four rows, two views each:

  raw field h         frame-averaged surface height, z-scored per vertex over the run
  raw energy h^2      frame-averaged local energy (what BOLD is read from), z-scored per vertex
  observable          energy -> BOLD kernel -> 0.01-0.08 Hz, z-scored per vertex
  damping gain        1 + k a / a_ref, in absolute units (1 = no adaptation), shared scale

Underneath: the gain's mean and its 5-95% range across vertices over time, and the
observable's leading spatial pattern's timecourse. The burn-in that scoring discards is
shaded.

FIRST RUN, 2026-10-07, f10m_d0.87_learn_v4_ckpt (iteration 40: k 0.267, tau_a 30 s). The gain
settles within ~60 s and then hardly moves: median 1.22, 5-95% across vertices 1.07-1.69.
Of its variance after the burn-in, 98% is ACROSS vertices (the time-mean map) and 2% over
time within a vertex, 31% of that 2% in 0.01-0.08 Hz. The map is high along the central
sulcus and in medial occipital cortex. At this k and tau_a the adaptation acts as a fixed
regional damping map set by where the drive puts energy, not as a time-varying gain.

  python viz/adaptive_movie.py --tag f10m_d0.87_learn_v4_ckpt
"""
import _path  # noqa: F401
import os, argparse
import numpy as np

from paths import RESULTS, VIDEOS


def zs(X):
    X = np.asarray(X, np.float64)
    X = X - X.mean(0, keepdims=True)
    return np.asarray(X / np.maximum(X.std(0, keepdims=True), 1e-300), np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", default="f10m_d0.87_learn_v4_ckpt",
                    help="results/adaptfit_<tag>.npz (a checkpoint's best G is used)")
    ap.add_argument("--seconds", type=float, default=577.0)
    ap.add_argument("--seed", type=int, default=0, help="drive seed is 10000 + this")
    ap.add_argument("--step", type=int, default=2, help="model frames per video frame")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--bitrate", type=int, default=3000)
    ap.add_argument("--device", default="mps")
    a = ap.parse_args()

    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.tri import Triangulation
    import matplotlib.animation as animation
    import fc_score, units, bandpass, torch_swe
    from mesh_cache import load_cortex
    from adaptive_fit import load_envfit, AdaptiveFit
    from render_regimes import _proj
    from input_embedding import VIEWS, embed

    z = np.load(os.path.join(RESULTS, f"adaptfit_{a.tag}.npz"), allow_pickle=True)
    G0 = z["best_G"] if "best_G" in z.files else z["G"]
    k, tau, aref = float(z["k"]), float(z["tau"]), float(z["a_ref"])
    c = load_cortex("fsaverage5", verbose=False)
    t = fc_score.default_target(c, verbose=False)
    ref = load_envfit(str(z["ref"]))
    ev = AdaptiveFit(ref, c, t, a.seconds, k, tau, device=a.device, tau_max=tau)
    ev.a_ref = aref
    G = torch.tensor(G0, dtype=torch.complex64 if ev.dtype == torch.float32 else torch.complex128)
    gen = torch.Generator().manual_seed(10_000 + a.seed)
    with torch.no_grad():
        A = ev.drive(G, ev.eta(gen)).detach().cpu().numpy()
    o, _ = torch_swe.run_adaptive(ev.sw, A, ref["P"], ev.dt, ev.g, ev.Hf, ev.save, k=k,
                                  tau_steps=ev.tau_steps, a_ref=aref)
    fs, (lo, hi) = ref["frame_s"], ref["band"]
    obs = bandpass.apply(units.smooth_frames(o["energy"], ev.kern), fs, lo, hi)
    gain = 1.0 + k * o["a"] / aref
    nfr = obs.shape[0]
    burn_s = ev.burn * fs
    print(f"  {a.tag}: k {k:.3f}, tau_a {tau:.1f} s; simulated {a.seconds:.0f} s, {nfr} frames; "
          f"gain after burn: median {np.median(gain[ev.burn:]):.3f}, "
          f"5-95% {np.percentile(gain[ev.burn:], 5):.3f}-{np.percentile(gain[ev.burn:], 95):.3f}")

    # how the gain varies: across vertices (time-mean map) against within a vertex over time
    gb = gain[ev.burn:]
    sp = float(gb.mean(0).var()); tm = float(gb.var(0).mean())
    gbp = bandpass.apply(gb - gb.mean(0), fs, lo, hi)
    inb = float(gbp.var(0).mean())
    print(f"  gain variance after burn: across vertices (time-mean map) {sp:.3g}, "
          f"over time within vertices {tm:.3g} ({tm/(sp+tm):.0%} of the total), "
          f"of which in 0.01-0.08 Hz {inb/tm:.0%}")

    cols = np.asarray(t.cols)
    Gf, lamf = embed(np.asarray(obs[ev.burn:, cols], np.float64), n=3)
    gk = Gf[:, 0] - np.median(Gf[:, 0]); gk /= np.linalg.norm(gk)
    P1 = zs(obs[:, cols]) @ gk; P1 = (P1 - P1.mean()) / P1.std()

    proj = _proj(c.V, c.F, VIEWS)
    sel = np.arange(0, nfr, a.step)
    glo, ghi = 1.0, float(np.percentile(gain[ev.burn:], 99.5))
    rows = [("raw field h", zs(o["h"])[sel], "RdBu_r", (-2.5, 2.5)),
            ("raw energy h²", zs(o["energy"])[sel], "RdBu_r", (-2.5, 2.5)),
            ("observable: energy,\nBOLD kernel, 0.01-0.08 Hz", zs(obs)[sel], "RdBu_r", (-2.5, 2.5)),
            (f"damping gain\n1 + k a/a_ref ({glo:g}-{ghi:.2f})", gain[sel].astype(np.float32),
             "magma", (glo, ghi))]
    fig = plt.figure(figsize=(9.0, 15.0), facecolor="black")
    gsm = fig.add_gridspec(6, 2, height_ratios=[2.5, 2.5, 2.5, 2.5, 1.2, 1.2], hspace=0.10,
                           wspace=0.02, top=0.95, bottom=0.04, left=0.07, right=0.99)
    arts = []
    for r, (lab, H, cmap, (vmin, vmax)) in enumerate(rows):
        for vi, (xy, vis, nm) in enumerate(proj):
            ax = fig.add_subplot(gsm[r, vi], facecolor="black")
            keep = vis[c.F].all(1)
            im = ax.tripcolor(Triangulation(xy[:, 0], xy[:, 1], c.F[keep]), H[0],
                              shading="gouraud", cmap=cmap, vmin=vmin, vmax=vmax, rasterized=True)
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
    q5, q50, q95 = np.percentile(gain, [5, 50, 95], axis=1)
    ax = fig.add_subplot(gsm[4, :], facecolor="black")
    ax.fill_between(tt, q5, q95, color="tab:orange", alpha=0.35, lw=0)
    ax.plot(tt, q50, color="tab:orange", lw=0.9)
    ax.set_ylabel("gain\nmedian, 5-95%", color="0.8", fontsize=8)
    ax2 = fig.add_subplot(gsm[5, :], facecolor="black")
    ax2.plot(tt, P1, color="tab:cyan", lw=0.8)
    ax2.set_ylabel("observable\npattern 1", color="0.8", fontsize=8)
    ax2.set_xlabel("seconds", color="0.75", fontsize=9)
    for x in (ax, ax2):
        x.axvspan(0, burn_s, color="0.25", lw=0)
        x.set_xlim(0, tt[-1])
        for s in x.spines.values():
            s.set_color("0.3")
        x.tick_params(colors="0.6", labelsize=8)
        curs.append(x.axvline(0, color="w", lw=1.0))
    ttl = fig.text(0.5, 0.977, "", color="0.85", ha="center", fontsize=11)

    def update(i):
        for im, H in arts:
            im.set_array(H[i])
        for cu in curs:
            cu.set_xdata([tt[sel[i]]] * 2)
        ttl.set_text(f"{a.tag}  k {k:.2f}  tau_a {tau:.0f} s   t = {tt[sel[i]]:6.1f} s"
                     + ("   (burn-in, not scored)" if tt[sel[i]] < burn_s else ""))
        return []

    os.makedirs(VIDEOS, exist_ok=True)
    out = os.path.join(VIDEOS, f"adaptive_{a.tag}.mp4")
    ani = animation.FuncAnimation(fig, update, frames=len(sel), blit=False)
    ani.save(out, writer=animation.FFMpegWriter(fps=a.fps, bitrate=a.bitrate),
             savefig_kwargs=dict(facecolor="black"))
    plt.close(fig)
    print(f"  wrote {out}  ({len(sel)/a.fps:.0f} s at {a.fps} fps, {os.path.getsize(out)/2**20:.0f} MB)")


if __name__ == "__main__":
    main()
