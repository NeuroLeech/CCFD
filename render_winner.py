"""Videos of a searched genome: the field it produces, and the drive that produced it.

The drive is rendered on the same surface and the same projection as the field, so the
two videos can be watched against each other - what is injected on the left, what the
fluid does with it on the right. Both carry the latent traces and a time cursor beneath,
because a frame of either is uninterpretable without knowing where in the drive it sits.

A third output is static: each latent's spatial pattern, sum_k L[k,j] * taper_k, which is
the map that latent switches on and off. Those three maps plus the traces are the whole
input, so they are worth seeing on their own.

  python render_winner.py                                   # newest cma_fc pickle
  python render_winner.py --pkl results/cma_fc/cma_fc_pop32_gen50_seed0.pkl --fps 20
"""
import os, glob, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.tri import Triangulation
import pickle

from mesh_cache import load_cortex
from input_model import NetworkDrive, parcel_tapers
from swe_rot import RotSWE, sponge_profile
from render_regimes import _proj
from ga_fc import decode, AMP_FIXED
from genome import LD_FIXED, SPONGE_STRENGTH_FIXED, SPONGE_WIDTH_FIXED
from run_ou import CFL, C, G, H
from paths import RESULTS, VIDEOS


def run_saving_drive(cortex, drive, nsteps, save_every, sponge, fp=None):
    """Same integration as the search uses, but the injected field is kept as well.
    With `fp` (fluid.decode output) the medium carries its searched speed and damping
    fields rather than the frozen scalars."""
    if fp is not None:
        import fluid as fl
        s, dt, gD_, HD_ = fl.build(cortex, fp, sponge)
    else:
        dt = CFL * cortex.d.min() / C
        s = RotSWE(cortex.m, C / LD_FIXED, l=cortex.l, d=cortex.d, A=cortex.A,
                   E=cortex.edges, bnd_edge=cortex.bnd)
        if sponge:
            s.set_sponge(sponge_profile(cortex.V, cortex.edges, cortex.bnd,
                                        SPONGE_WIDTH_FIXED, SPONGE_STRENGTH_FIXED))
        s.astype(np.float32)
        for attr in ("sig_v", "sig_e"):
            v = getattr(s, attr, None)
            if v is not None:
                setattr(s, attr, v.astype(np.float32))
        gD_, HD_ = np.float32(G), np.float32(H)

    Aser, P = drive.Aser.astype(np.float32), drive.P.astype(np.float32)
    h = np.zeros(cortex.nV, np.float32)
    ue = np.zeros(s.nE, np.float32)
    dtD, gD, HD = np.float32(dt), gD_, HD_
    fields, drives = [], []
    for n in range(nsteps):
        src = Aser[n] @ P
        h += src
        ue, h = s.step(ue, h, dtD, gD, HD)
        if n % save_every == 0:
            fields.append(h.copy()); drives.append(src.copy())
    return np.asarray(fields), np.asarray(drives), dt


def movie(frames, cortex, proj, S, save_every, dt, path, title, cmap="RdBu_r",
          clip=98.0, fps=16, trace_label="drive (per region)", verbose=True):
    """One row of surface views over time, drive traces and a cursor beneath.

    `S` is whatever timecourse belongs under the movie - region drive amplitudes for the
    cross-spectrum runs. It was labelled "latents" when the input was a latent factor
    model; there are no latents in the cross-spectrum pipeline, so the label is now an
    argument rather than a leftover."""
    vl = float(np.percentile(np.abs(frames), clip))
    fig = plt.figure(figsize=(4.2 * len(proj), 4.4))
    gs = fig.add_gridspec(2, len(proj), height_ratios=[1.0, 0.30], hspace=0.05,
                          wspace=0.02)
    meshes = []
    for k, (xy, vis, nm) in enumerate(proj):
        ax = fig.add_subplot(gs[0, k])
        keep = vis[cortex.F].all(1)
        tri = Triangulation(xy[:, 0], xy[:, 1], cortex.F[keep])
        tm = ax.tripcolor(tri, frames[0], shading="gouraud", cmap=cmap,
                          vmin=-vl, vmax=vl, rasterized=True)
        ax.set_xlim(xy[:, 0].min(), xy[:, 0].max())
        ax.set_ylim(xy[:, 1].min(), xy[:, 1].max())
        ax.set_aspect("equal"); ax.axis("off"); ax.set_title(nm, fontsize=10)
        meshes.append(tm)

    axd = fig.add_subplot(gs[1, :])
    t_saved = np.arange(len(frames)) * save_every * dt
    t_full = np.arange(len(S)) * dt
    ncol = min(S.shape[1], 6)
    for j in range(ncol):
        axd.plot(t_full, S[:, j] / max(np.abs(S).max(), 1e-30) + 2.2 * j, lw=0.7)
    axd.set_xlim(t_full[0], t_full[-1]); axd.set_yticks([])
    axd.set_xlabel("time units", fontsize=9); axd.tick_params(labelsize=8)
    axd.set_ylabel(trace_label, fontsize=9)
    cursor = axd.axvline(0, color="k", lw=1.2)
    fig.suptitle(f"{title}    scale +-{vl:.2e}", fontsize=10, family="monospace")

    def upd(i):
        for tm in meshes:
            tm.set_array(frames[i])
        cursor.set_xdata([t_saved[i], t_saved[i]])
        return []

    os.makedirs(os.path.dirname(path), exist_ok=True)
    animation.FuncAnimation(fig, upd, frames=len(frames)).save(
        path, writer=animation.FFMpegWriter(fps=fps, bitrate=3600))
    plt.close(fig)
    if verbose:
        print(f"  wrote {path}")


def latent_maps(cortex, drive, L, proj, path):
    """Each latent's spatial pattern: the map it switches on and off."""
    pats = [L[:, j] @ drive.P for j in range(L.shape[1])]
    fig = plt.figure(figsize=(3.6 * len(proj), 2.5 * len(pats)))
    gs = fig.add_gridspec(len(pats), len(proj), hspace=0.04, wspace=0.02)
    for r, p in enumerate(pats):
        vl = float(np.percentile(np.abs(p), 99.5))
        for k, (xy, vis, nm) in enumerate(proj):
            ax = fig.add_subplot(gs[r, k])
            keep = vis[cortex.F].all(1)
            ax.tripcolor(Triangulation(xy[:, 0], xy[:, 1], cortex.F[keep]), p,
                         shading="gouraud", cmap="RdBu_r", vmin=-vl, vmax=vl,
                         rasterized=True)
            ax.set_xlim(xy[:, 0].min(), xy[:, 0].max())
            ax.set_ylim(xy[:, 1].min(), xy[:, 1].max())
            ax.set_aspect("equal"); ax.axis("off")
            if r == 0:
                ax.set_title(nm, fontsize=10)
            if k == 0:
                ax.text(-0.03, 0.5, f"latent {r}", rotation=90, transform=ax.transAxes,
                        ha="center", va="center", fontsize=9, family="monospace")
    fig.savefig(path, dpi=135, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def fluid_maps(cortex, fp, proj, path):
    """The medium itself: wave speed and damping across the sheet."""
    import fluid as fl
    from matplotlib.tri import Triangulation
    c, sig = fl.fields(cortex, fp)      # handles group and map parameterisations alike
    rows = [("wave speed", c, "viridis"), ("damping rate", sig, "magma")]
    fig = plt.figure(figsize=(3.6 * len(proj), 2.5 * len(rows)))
    gs = fig.add_gridspec(len(rows), len(proj), hspace=0.04, wspace=0.02)
    for r, (label, v, cmap) in enumerate(rows):
        lo, hi = float(np.percentile(v, 1)), float(np.percentile(v, 99))
        for k, (xy, vis, nm) in enumerate(proj):
            ax = fig.add_subplot(gs[r, k])
            keep = vis[cortex.F].all(1)
            ax.tripcolor(Triangulation(xy[:, 0], xy[:, 1], cortex.F[keep]), v,
                         shading="gouraud", cmap=cmap, vmin=lo, vmax=hi, rasterized=True)
            ax.set_xlim(xy[:, 0].min(), xy[:, 0].max())
            ax.set_ylim(xy[:, 1].min(), xy[:, 1].max())
            ax.set_aspect("equal"); ax.axis("off")
            if r == 0:
                ax.set_title(nm, fontsize=10)
            if k == 0:
                ax.text(-0.03, 0.5, f"{label}\n{lo:.3g} to {hi:.3g}", rotation=90,
                        transform=ax.transAxes, ha="center", va="center",
                        fontsize=8.5, family="monospace")
    fig.savefig(path, dpi=135, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pkl", default=None, help="search result (default: newest cma_fc)")
    ap.add_argument("--draw-seed", type=int, default=90_000, dest="draw_seed")
    ap.add_argument("--fps", type=int, default=16)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()

    pkl = a.pkl or sorted(glob.glob(os.path.join(RESULTS, "cma_fc", "*.pkl")),
                          key=os.path.getmtime)[-1]
    d = pickle.load(open(pkl, "rb"))
    K, r = d["K"], d["r"]
    tag = a.tag or os.path.basename(pkl).replace(".pkl", "")
    print(f"{os.path.basename(pkl)}: best {d['best'][0]:+.4f} at generation {d['best'][2]}")

    cortex = load_cortex("fsaverage5", verbose=False)
    dt = CFL * cortex.d.min() / C
    xb, fp = d["best"][1], None
    if d.get("fluid"):
        import fluid as fl
        fp = fl.decode(xb[-fl.N_PARAM:])
        xb = xb[:-fl.N_PARAM]
        cfield = fp["c0"] * fl.group_field(cortex, fp["c_group"])
        dt = fl.CFL * cortex.d.min() / float(cfield.max())
        print(f"  fluid: c0 {fp['c0']:.2f}, Ld {fp['Ld']:.1f}, sig0 {fp['sig0']:.1e}, "
              f"speed x{np.round(fp['c_group'],2)}, damping x{np.round(fp['sig_group'],2)}")

    if d.get("generator") == "ladder":
        import ladder
        from input2 import parcel_tapers as region_tapers
        regs = d["regions"]
        D, _ = ladder.parcel_geodesic(cortex, regs, verbose=False)
        p = ladder.decode(xb, d.get("rung", 4))
        print("  ladder: " + ", ".join(f"{k} {v:.3g}" for k, v in p.items()))
        drive = ladder.make_drive(cortex, xb, d["nsteps"], dt, amp=AMP_FIXED,
                                  rung=d.get("rung", 4), seed=a.draw_seed,
                                  tapers=region_tapers(cortex, verbose=False),
                                  regions=regs, D=D)
        traces = drive.Aser
    else:
        L, silent, tau = decode(xb, K, r, d["tau_lim"], d["silent_lim"], d.get("gain_lim"))
        print(f"  taus {np.round(tau,1)}  silent {np.round(silent,3)}  "
              f"gains {np.round(np.linalg.norm(L,axis=0),3)}")
        drive = NetworkDrive(cortex, d["regions"], L, silent, tau, AMP_FIXED, d["nsteps"],
                             dt, seed=a.draw_seed,
                             tapers=parcel_tapers(cortex, verbose=False),
                             balance="spatial")
        traces = drive.S
    fields, drives, dt = run_saving_drive(cortex, drive, d["nsteps"], 25, d["sponge"], fp)
    print(f"  {len(fields)} frames, field peak {np.abs(fields).max():.2e}, "
          f"drive peak {np.abs(drives).max():.2e}")
    np.save(os.path.join(RESULTS, f"frames_{tag}.npy"), fields)

    proj = _proj(cortex.V, cortex.F)
    if fp is not None:
        fluid_maps(cortex, fp, proj, os.path.join(RESULTS, f"fluid_maps_{tag}.png"))
    movie(fields, cortex, proj, traces, 25, dt,
          os.path.join(VIDEOS, f"{tag}_field.mp4"), f"{tag}  field h", fps=a.fps)
    movie(drives, cortex, proj, traces, 25, dt,
          os.path.join(VIDEOS, f"{tag}_drive.mp4"), f"{tag}  drive (injected per step)",
          fps=a.fps)
    if d.get("generator") != "ladder":
        latent_maps(cortex, drive, L, proj,
                    os.path.join(RESULTS, f"latent_maps_{tag}.png"))


if __name__ == "__main__":
    main()
