"""Collect one fitted model into a folder: parameters, fields, and the figures.

Everything here describes ONE solution - `results/torchfit_<tag>.npz` - in the medium that
solution was fitted in, which for a --learn-maps run is not the baseline medium and has to
be rebuilt from the saved coefficients. Surface maps of discrete things (which piece, which
pathway) are drawn with flat shading: Gouraud would interpolate between neighbouring
pieces and invent boundaries that are not in the model.

  python analysis/report.py --tag wamap2 --parts all
"""
import _path  # noqa: F401  - puts the sibling code folders on sys.path
import os, argparse, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

from mesh_cache import load_cortex
from paths import RESULTS
from render_regimes import _proj

# the thalamic nucleus that drives each parcel, from the AlternateListOfInputs comment in
# fit/subparcels.py - the grouping the `subcortical` set was built from
NUCLEUS = {
    "LGN": [1], "MGN": [24], "VPL/VPM": [9, 51], "VA/VL": [8, 96, 55],
    "pulvinar": [145, 42, 22], "mediodorsal": [86, 84, 62],
    "limbic/amygdala": [164, 93, 118], "insular": [112],
}
PARCEL_NUC = {p: n for n, ps in NUCLEUS.items() for p in ps}


def nucleus_of(tag):
    """The driving nucleus for a piece, or "?" when this basis does not carry one.

    NUCLEUS is keyed by the Glasser indices of the 47-piece `subcortical` set. The
    ascending bases name their pieces differently - 'V1_0', 'G_OFC_1', '3b_2' - so the
    leading token is usually not a number and there is nothing to look up. Returning "?"
    says the grouping is absent on this basis, which is true; it is not a claim that the
    piece has no thalamic input.
    """
    head = str(tag).split("_")[0]
    return PARCEL_NUC.get(int(head), "?") if head.isdigit() else "?"


def load_all(tag, ref_tag="torchref"):
    import torch, fc_score
    from torch_fit import Fit, load_reference
    z = np.load(os.path.join(RESULTS, f"torchfit_{tag}.npz"))
    ref = load_reference(ref_tag)
    c = load_cortex("fsaverage5", verbose=False)
    t = fc_score.default_target(c, verbose=False)
    fit = Fit(ref, c, t, float(z["seconds"]), device="mps", amp=float(z["amp"]),
              nl_adv=float(z["nl_adv"]), nl_flux=float(z["nl_flux"]), Ld=float(z["Ld"]),
              maps_scale=float(z["maps_scale"]))
    if "map_a" in z.files:
        fit.init_medium(float(z["learn_maps"]))
        with torch.no_grad():
            fit.a_raw.copy_(torch.as_tensor(z["map_a"], dtype=fit.dtype, device=fit.dev))
            fit.b_raw.copy_(torch.as_tensor(z["map_b"], dtype=fit.dtype, device=fit.dev))
    rz = np.load(os.path.join(RESULTS, f"xspec_{ref_tag}.npz"), allow_pickle=True)
    return z, ref, c, t, fit, rz


def flat_surface(fig, gs, r, proj, vals, c, cmap, lims, label, cats=None):
    """One map, three views, FLAT shaded - one colour per triangle, no interpolation.

    A face takes the value of its vertices: the majority for a categorical map (so a piece
    boundary stays where it is), the mean for a continuous one."""
    vals = np.asarray(vals)
    for k, (xy, vis, nm) in enumerate(proj):
        ax = fig.add_subplot(gs[r, k])
        keep = vis[c.F].all(1)
        F = c.F[keep]
        if cats:
            v3 = vals[F]
            fv = np.where(v3[:, 0] == v3[:, 1], v3[:, 0],
                          np.where(v3[:, 0] == v3[:, 2], v3[:, 0], v3[:, 1]))
        else:
            fv = vals[F].mean(1)
        tri = Triangulation(xy[:, 0], xy[:, 1], F)
        ax.tripcolor(tri, facecolors=fv, cmap=cmap, vmin=lims[0], vmax=lims[1],
                     shading="flat", rasterized=True, edgecolors="none")
        ax.set_xlim(xy[:, 0].min(), xy[:, 0].max())
        ax.set_ylim(xy[:, 1].min(), xy[:, 1].max())
        ax.set_aspect("equal"); ax.axis("off")
        if r == 0:
            ax.set_title(nm, fontsize=10)
        if k == 0:
            ax.text(-0.03, 0.5, label, rotation=90, transform=ax.transAxes,
                    ha="center", va="center", fontsize=8.5, family="monospace")


def impulse_window(fit, decays=7.0, cap=3520):
    """Frames an impulse needs to decay, from the medium the fit ended with.

    interference.py takes this from the run's own damping, and it has to be taken from the
    LEARNED damping here: wamap2 moved map_b from [-0.03, 0.35, 0.35] to [-0.41, -0.44,
    -0.53], so its slowest point decays in 441 frames against the incumbent's, and a window
    sized for the incumbent truncates the response. A 1088-frame window reproduced the
    saved field at r = 0.953, below the 0.99 interference.py requires of itself."""
    sig = fit.medium()[1].detach().cpu().numpy()
    dec = 1.0 / (fit.dt * sig * fit.save)
    return int(min(cap, np.ceil(decays * dec.max() / 64.0) * 64))


def impulse_and_pieces(fit, P, A, save, nfr, imp_frames=None, verbose=True):
    """Per-piece contribution to the field, in the medium the fit actually used.

    xspec.impulse_responses builds the medium from `p`, and a learned grading is not
    expressible as one - L*tanh((a.M)/L) is not a'.M - so the responses are integrated here
    with the same torch stepper the fit descended. wamap2 is linear (nl_flux and nl_adv
    both 0), so superposition still holds and interference.py's decomposition is valid;
    only the medium it is evaluated in differs.

    -> (total field, per-piece variance per vertex, per-piece response peak)"""
    import torch, torch_swe
    from scipy.signal import fftconvolve
    K, nV = P.shape
    if imp_frames is None:
        imp_frames = impulse_window(fit)
    if verbose:
        print(f"    impulse window {imp_frames} frames "
              f"({imp_frames * save} steps per piece)", flush=True)
    a = A[:nfr * save].reshape(nfr, save, -1).sum(1)          # injected per frame
    H = fit.apply_medium()
    tot = np.zeros((nfr, nV), np.float32)
    var_k = np.zeros((K, nV), np.float32)
    for k in range(K):
        with torch.no_grad():
            ue, h = fit.sw.zeros()
            h = torch.as_tensor(P[k], dtype=fit.dtype, device=fit.dev).clone()
            fr = [h.clone()]
            for n in range(1, imp_frames * save):
                ue, h = fit.sw.step(ue, h, torch.as_tensor(fit.dt, dtype=fit.dtype,
                                                           device=fit.dev),
                                    torch.as_tensor(fit.g, dtype=fit.dtype,
                                                    device=fit.dev), H)
                if n % save == 0:
                    fr.append(h.clone())
            R = torch.stack(fr).cpu().numpy()
        f = fftconvolve(a[:, k:k + 1], R, axes=0)[:nfr]
        tot += f
        var_k[k] = f.var(0)
        if verbose and (k + 1) % 10 == 0:
            print(f"    {k+1}/{K} pieces", flush=True)
        del f, R
    return tot, var_k


def energy_flux(fit, A, P, save, every=25, verbose=True):
    """Time-averaged <h u> per vertex: where the field carries energy, and which way.

    The time-mean VELOCITY of a wave field driven by a zero-mean input is ~0 - the flow
    oscillates - so it says nothing about propagation. The energy flux does: for shallow
    water it is the transport term, and its average is the net direction the field moves
    energy. -> (nV, 3) in the mesh's own coordinates."""
    import torch, torch_swe
    H = fit.apply_medium()
    dt = torch.as_tensor(fit.dt, dtype=fit.dtype, device=fit.dev)
    g = torch.as_tensor(fit.g, dtype=fit.dtype, device=fit.dev)
    At = torch.as_tensor(A, dtype=fit.dtype, device=fit.dev)
    Pt = torch.as_tensor(np.asarray(P), dtype=fit.dtype, device=fit.dev)
    acc = torch.zeros(fit.sw.nV, 3, dtype=fit.dtype, device=fit.dev)
    n_acc = 0
    with torch.no_grad():
        ue, h = fit.sw.zeros()
        for n in range(len(A)):
            h = h + At[n] @ Pt
            ue, h = fit.sw.step(ue, h, dt, g, H)
            if n % every == 0:
                acc += h[:, None] * fit.sw.gather(ue)
                n_acc += 1
            if verbose and n % 4000 == 0:
                print(f"    step {n}/{len(A)}", flush=True)
    return (acc / max(n_acc, 1)).cpu().numpy()


def observable(frames, ref, t, band=True):
    """Frames -> (vertices, time) z-scored rows, the thing FC is computed from."""
    import bandpass
    X = np.asarray(frames, np.float32)
    if band:
        X = bandpass.apply(X, float(ref["frame_s"]), *ref["band"])
    Z, _ = t.model_z(X)
    return Z


def sec_summary(out, tag, z, ref, c, t, rz, fit, log):
    lines = [f"# {tag}", "",
             "One fitted model: the input cross-spectrum G (and, here, the medium's",
             "grading) solved by gradient descent through the simulation.", "",
             "## Configuration", ""]
    amp = float(z["amp"])
    cfg = [("input basis", f"{rz['profiles'].shape[0]} pieces, "
                           f"regions={rz['regions']}, split={int(rz['split'])}"),
           ("realisation", f"{float(z['seconds']):.0f} s "
                           f"({fit.nframes} frames, {fit.nsteps} steps, save={fit.save})"),
           ("passband", f"{ref['band'][0]}-{ref['band'][1]} Hz (the target's own filter)"),
           ("solve vertices", f"{len(ref['sub'])} medoids of {t.nV}"),
           ("drive amplitude", f"{amp:.3g} rms ({amp/2e-4:.0f}x the incumbent)"),
           ("nl_flux", f"{float(z['nl_flux']):g}  (H+h in the depth update)"),
           ("nl_adv", f"{float(z['nl_adv']):g}  (vector-invariant momentum advection)"),
           ("Ld", f"{float(z['Ld']):.1f}  -> inertial period "
                  f"{2*np.pi*float(z['Ld'])/fit.dt*(float(ref['frame_s'])/fit.save):.0f} s"),
           ("iterations", f"{int(z['best_iter'])}"),
           ("best realised sim", f"{float(z['best_sim']):+.4f}")]
    if "map_a" in z.files:
        H, sig = fit.medium()
        H = H.detach().cpu().numpy(); sig = sig.detach().cpu().numpy()
        cfg += [("medium", f"FITTED, multiplier bounded to "
                           f"[1/{float(z['learn_maps']):g}, {float(z['learn_maps']):g}] on depth"),
                ("  map coefficients a (speed)", np.round(z["map_a"], 4).tolist()),
                ("  map coefficients b (damping)", np.round(z["map_b"], 4).tolist()),
                ("  H range", f"{H.min():.3f}-{H.max():.3f} ({H.max()/H.min():.1f}x); "
                              f"fixed medium is {fit.Hf.min():.3f}-{fit.Hf.max():.3f} "
                              f"({fit.Hf.max()/fit.Hf.min():.1f}x)"),
                ("  damping range", f"{sig.min():.2e}-{sig.max():.2e} "
                                    f"({sig.max()/sig.min():.1f}x)")]
    else:
        cfg += [("medium", "fixed (the incumbent BEST_X grading)")]
    lines += [f"- **{k}**: {v}" for k, v in cfg]
    lines += ["", "## Files", "",
              "| file | what |", "|---|---|",
              "| `video_raw.mp4` | the field, no passband |",
              "| `video_bandpassed.mp4` | the same run through 0.01-0.08 Hz |",
              "| `video_pair.mp4` | both rows together, one scale per row |",
              "| `scatter_edges.png` | simulated against empirical FC, edgewise |",
              "| `scatter_regions.png` | the same parcel-averaged |",
              "| `input_regions.png` | which piece drives each vertex, flat shaded |",
              "| `input_drive.png` | each piece's drive sd and mean abs |",
              "| `interference.png` | coherent/incoherent, and pieces reaching a vertex |",
              "| `pathways.png` | share of drive variance by thalamic nucleus |",
              "| `velocity.png` | time-averaged energy flux direction |",
              "| `medium.png` | the three maps, and the speed and damping they make |",
              "| `gradients.png` | first five FC gradients, empirical and simulated |",
              "| `summary.json` | the numbers behind this page |"]
    open(os.path.join(out, "summary.md"), "w").write("\n".join(lines) + "\n")
    print("  wrote summary.md")


def sec_medium(out, c, fit, z):
    """The three anatomical maps, and the speed and damping they combine into."""
    from cortical_maps import load_maps
    proj = _proj(c.V, c.F)
    names = tuple(str(v) for v in fit.p["maps"])
    mp = load_maps(c, names, verbose=False)
    H, sig = (x.detach().cpu().numpy() for x in fit.medium()) if "map_a" in z.files \
        else (np.asarray(fit.Hf), np.asarray(fit.sw.sig_v.cpu()))
    rows = [(f"{n}\n(z-scored)", mp[n], "PuOr_r",
             (-2.5, 2.5)) for n in names]
    rows += [("wave speed\nc = sqrt(H)", np.sqrt(H), "viridis",
              (float(np.sqrt(H).min()), float(np.sqrt(H).max()))),
             ("damping\n(per step)", np.log10(np.maximum(sig, 1e-12)), "magma",
              (float(np.log10(sig).min()), float(np.log10(sig).max())))]
    fig = plt.figure(figsize=(3.9 * len(proj), 2.7 * len(rows)))
    gs = fig.add_gridspec(len(rows), len(proj), hspace=0.06, wspace=0.02)
    for i, (lab, v, cm, lims) in enumerate(rows):
        tl = f"{lab}\n{lims[0]:+.3g} to {lims[1]:+.3g}"
        flat_surface(fig, gs, i, proj, v, c, cm, lims, tl)
    fig.savefig(os.path.join(out, "medium.png"), dpi=130, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("  wrote medium.png")


def sec_inputs(out, c, rz, drive):
    """Which piece drives each vertex, and how hard each piece is driven."""
    proj = _proj(c.V, c.F)
    lab = np.asarray(rz["labels"]).astype(float)       # -1 outside any piece
    tags = [str(v) for v in rz["tags"]]
    P = np.asarray(rz["profiles"], np.float64)
    sd = drive.std(0)
    ma = np.abs(drive).mean(0)
    # paint each piece's statistic onto its own territory
    def paint(vals):
        outv = np.full(c.nV, np.nan)
        for k in range(len(tags)):
            m = np.asarray(rz["labels"]) == k
            outv[m] = vals[k]
        return outv
    fig = plt.figure(figsize=(3.9 * len(proj), 2.7 * 3))
    gs = fig.add_gridspec(3, len(proj), hspace=0.06, wspace=0.02)
    L = np.where(lab < 0, np.nan, lab)
    flat_surface(fig, gs, 0, proj, np.nan_to_num(L, nan=-1), c, "tab20",
                 (-1, len(tags)), f"piece index\n0-{len(tags)-1}, flat shaded", cats=True)
    v1 = paint(sd)
    flat_surface(fig, gs, 1, proj, np.nan_to_num(v1, nan=0), c, "inferno",
                 (0, float(np.nanmax(v1))), f"drive sd\n0 to {np.nanmax(v1):.2e}")
    v2 = paint(ma)
    flat_surface(fig, gs, 2, proj, np.nan_to_num(v2, nan=0), c, "inferno",
                 (0, float(np.nanmax(v2))), f"drive mean |a|\n0 to {np.nanmax(v2):.2e}")
    fig.savefig(os.path.join(out, "input_regions.png"), dpi=130, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)

    order = np.argsort(sd)[::-1]
    fig, ax = plt.subplots(2, 1, figsize=(13, 6.4), sharex=True)
    nuc = [nucleus_of(t) for t in tags]
    uniq = sorted(set(nuc)); col = {n: plt.cm.tab10(i % 10) for i, n in enumerate(uniq)}
    for a_, v, nm in ((ax[0], sd, "drive sd"), (ax[1], ma, "drive mean |a|")):
        a_.bar(range(len(tags)), v[order], color=[col[nuc[i]] for i in order])
        a_.set_ylabel(nm); a_.grid(alpha=0.25, axis="y")
    ax[1].set_xticks(range(len(tags)))
    ax[1].set_xticklabels([tags[i] for i in order], rotation=90, fontsize=6.5)
    ax[0].legend(handles=[plt.Line2D([], [], color=col[n], lw=6, label=n) for n in uniq],
                 fontsize=7, ncol=4)
    ax[0].set_title("per-piece drive, ordered by sd; colour = driving nucleus", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "input_drive.png"), dpi=130, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("  wrote input_regions.png, input_drive.png")
    return sd, ma


def sec_scatter(out, c, t, Z):
    """Simulated against empirical FC, edgewise and parcel-averaged."""
    from scipy.stats import spearmanr
    rng = np.random.default_rng(0)
    i, j = t.i, t.j
    if i is None:
        n = t.nV; iu = np.triu_indices(n, 1)
        k = rng.choice(len(iu[0]), 200000, replace=False)
        i, j = iu[0][k], iu[1][k]
    else:
        k = rng.choice(len(i), min(200000, len(i)), replace=False)
        i, j = i[k], j[k]
    T = Z.shape[1]
    mod = np.einsum("ij,ij->i", Z[i], Z[j]) / T
    emp = np.asarray(t.target_fc()[i, j], np.float64)
    pe = float(np.corrcoef(mod, emp)[0, 1]); sp = float(spearmanr(mod, emp).statistic)

    lab = np.asarray(c.lab)[t.cols]
    ids = np.array(sorted(set(int(v) for v in np.unique(lab)) - {0}))
    Zp = np.stack([Z[lab == p].mean(0) for p in ids])
    Zp -= Zp.mean(1, keepdims=True)
    Zp /= np.maximum(Zp.std(1, keepdims=True), 1e-12)
    Mp = (Zp @ Zp.T) / T
    FC = t.target_fc()
    Ep = np.stack([np.asarray(FC[lab == p][:, lab == q]).mean()
                   for p in ids for q in ids]).reshape(len(ids), len(ids))
    iu = np.triu_indices(len(ids), 1)
    pe_r = float(np.corrcoef(Mp[iu], Ep[iu])[0, 1])
    sp_r = float(spearmanr(Mp[iu], Ep[iu]).statistic)

    for nm, x, y, a, b, ttl in (("scatter_edges", emp, mod, pe, sp,
                                 f"vertex edges (n={len(emp):,})"),
                                ("scatter_regions", Ep[iu], Mp[iu], pe_r, sp_r,
                                 f"parcel pairs (n={len(iu[0]):,}, {len(ids)} parcels)")):
        fig, ax = plt.subplots(figsize=(5.4, 5.2))
        ax.hexbin(x, y, gridsize=90, bins="log", cmap="viridis", mincnt=1)
        lo = min(np.min(x), np.min(y)); hi = max(np.max(x), np.max(y))
        ax.plot([lo, hi], [lo, hi], color="0.75", lw=1, ls="--")
        z_ = np.polyfit(x, y, 1)
        ax.plot([lo, hi], np.polyval(z_, [lo, hi]), color="#c1442e", lw=1.4)
        ax.set_xlabel("empirical FC"); ax.set_ylabel("simulated FC")
        ax.set_title(f"{ttl}\npearson {a:+.4f}   spearman {b:+.4f}", fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"{nm}.png"), dpi=140, bbox_inches="tight",
                    facecolor="white")
        plt.close(fig)
    print(f"  wrote scatter_edges.png (r={pe:+.4f}), scatter_regions.png (r={pe_r:+.4f})")
    return dict(edge_pearson=pe, edge_spearman=sp, region_pearson=pe_r,
                region_spearman=sp_r, n_parcels=int(len(ids)))


def sec_interference_pathways(out, c, t, rz, var_k, tot):
    """Coherent/incoherent, how many pieces reach a vertex, and which pathway they are."""
    proj = _proj(c.V, c.F)
    tags = [str(v) for v in rz["tags"]]
    cols = t.cols
    coh = tot.var(0); inc = var_k.sum(0)
    ratio = np.where(inc > 0, coh / np.maximum(inc, 1e-30), np.nan)
    vk = var_k[:, cols]; recv = vk.sum(0)
    live = recv > 1e-6 * np.median(recv[recv > 0])
    eff = np.full(len(cols), np.nan)
    share = vk[:, live] / recv[live]
    eff[live] = 1.0 / (share ** 2).sum(0)
    full_eff = np.zeros(c.nV); full_eff[cols] = np.nan_to_num(eff)

    fig = plt.figure(figsize=(3.9 * len(proj), 2.7 * 2))
    gs = fig.add_gridspec(2, len(proj), hspace=0.06, wspace=0.02)
    flat_surface(fig, gs, 0, proj, np.nan_to_num(np.clip(ratio, 0, 2), nan=1.0), c,
                 "RdBu_r", (0, 2), "coherent / incoherent\n0 to 2")
    lim = (float(np.nanpercentile(eff, 2)), float(np.nanpercentile(eff, 98)))
    flat_surface(fig, gs, 1, proj, full_eff, c, "viridis", lim,
                 f"pieces reaching the vertex\n{lim[0]:.1f} to {lim[1]:.1f}")
    fig.savefig(os.path.join(out, "interference.png"), dpi=130, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)

    nuc = [nucleus_of(tg) for tg in tags]
    uniq = sorted(set(nuc))
    G = np.stack([var_k[[k for k, n in enumerate(nuc) if n == u]].sum(0) for u in uniq])
    tot_v = G.sum(0)
    ok = tot_v > 1e-6 * np.median(tot_v[tot_v > 0])
    Sh = np.zeros_like(G); Sh[:, ok] = G[:, ok] / tot_v[ok]
    nrow = len(uniq) + 1
    fig = plt.figure(figsize=(3.9 * len(proj), 2.5 * nrow))
    gs = fig.add_gridspec(nrow, len(proj), hspace=0.06, wspace=0.02)
    for r, u in enumerate(uniq):
        flat_surface(fig, gs, r, proj, Sh[r], c, "inferno", (0, 1),
                     f"{u}\nshare of drive variance")
    dom = np.where(ok, np.argmax(Sh, 0), -1).astype(float)
    flat_surface(fig, gs, nrow - 1, proj, dom, c, "tab10", (-1, len(uniq)),
                 "dominant pathway\n" + ", ".join(f"{i}={u}" for i, u in enumerate(uniq)),
                 cats=True)
    fig.savefig(os.path.join(out, "pathways.png"), dpi=130, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("  wrote interference.png, pathways.png")
    return dict(ratio_mean=float(np.nanmean(ratio[cols])),
                ratio_median=float(np.nanmedian(ratio[cols])),
                frac_cancelling=float(np.nanmean(ratio[cols] < 1)),
                eff_pieces_median=float(np.nanmedian(eff)),
                pathways={u: float(np.nanmean(Sh[r][cols])) for r, u in enumerate(uniq)})


def sec_velocity(out, c, flux):
    """Arrows for the time-averaged energy flux, projected into each view.

    The projection is applied by finite difference - project V and V + eps*flux and take
    the difference - so the arrows follow whatever _proj does without needing its matrix."""
    proj = _proj(c.V, c.F)
    mag = np.linalg.norm(flux, axis=1)
    sc = np.percentile(mag[mag > 0], 99) if (mag > 0).any() else 1.0
    step = max(1, c.nV // 1200)
    fig = plt.figure(figsize=(4.4 * len(proj), 4.0))
    gs = fig.add_gridspec(1, len(proj), wspace=0.02)
    eps = 0.02 * float(np.linalg.norm(c.V.max(0) - c.V.min(0))) / max(sc, 1e-30)
    for k, (xy, vis, nm) in enumerate(proj):
        ax = fig.add_subplot(gs[0, k])
        keep = vis[c.F].all(1)
        tri = Triangulation(xy[:, 0], xy[:, 1], c.F[keep])
        ax.tripcolor(tri, facecolors=mag[c.F[keep]].mean(1), cmap="Greys",
                     vmin=0, vmax=sc, shading="flat", rasterized=True)
        xy2 = _proj(c.V + eps * flux, c.F)[k][0]
        sel = np.flatnonzero(vis)[::step]
        u = xy2[sel, 0] - xy[sel, 0]; v = xy2[sel, 1] - xy[sel, 1]
        ax.quiver(xy[sel, 0], xy[sel, 1], u, v, mag[sel], cmap="autumn",
                  scale=None, width=0.0022, headwidth=3.5, alpha=0.9)
        ax.set_xlim(xy[:, 0].min(), xy[:, 0].max())
        ax.set_ylim(xy[:, 1].min(), xy[:, 1].max())
        ax.set_aspect("equal"); ax.axis("off"); ax.set_title(nm, fontsize=10)
    fig.suptitle("time-averaged energy flux <h u>; shading = magnitude", fontsize=10)
    fig.savefig(os.path.join(out, "velocity.png"), dpi=140, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("  wrote velocity.png")


def sec_gradients(out, c, t, Z, ncomp=5):
    """The first five FC gradients, empirical and simulated, on the same vertices."""
    from gradients import gradients as grad_embed
    proj = _proj(c.V, c.F)
    T = Z.shape[1]
    M = (Z @ Z.T) / T
    E = np.asarray(t.target_fc(), np.float32)
    ge, ve = grad_embed(E, n_comp=ncomp + 1)
    gm, vm = grad_embed(np.asarray(M, np.float32), n_comp=ncomp + 1)
    used, rows, stats = set(), [], []
    for a in range(ncomp):
        cc = [abs(np.corrcoef(ge[:, a], gm[:, b])[0, 1]) if b not in used else -1
              for b in range(gm.shape[1])]
        b = int(np.argmax(cc))
        used.add(b)
        r = float(np.corrcoef(ge[:, a], gm[:, b])[0, 1])
        s = np.sign(r) if r != 0 else 1.0
        rows.append((f"G{a+1} empirical", ge[:, a]))
        rows.append((f"G{a+1} simulated\nmatched |r| {abs(r):.3f}", s * gm[:, b]))
        stats.append(dict(gradient=a + 1, matched_component=b, abs_r=abs(r)))
    fig = plt.figure(figsize=(3.9 * len(proj), 2.4 * len(rows)))
    gs = fig.add_gridspec(len(rows), len(proj), hspace=0.06, wspace=0.02)
    for i, (lab, v) in enumerate(rows):
        full = np.zeros(c.nV); full[t.cols] = v
        lim = float(np.percentile(np.abs(v), 98))
        flat_surface(fig, gs, i, proj, full, c, "Spectral_r", (-lim, lim), lab)
    fig.savefig(os.path.join(out, "gradients.png"), dpi=125, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("  wrote gradients.png  (|r| " +
          ", ".join(f"G{s['gradient']} {s['abs_r']:.3f}" for s in stats) + ")")
    return stats


def sec_videos(out, tag, vtag):
    """The pair render, plus a single-row raw and bandpassed video."""
    import shutil, subprocess
    src = os.path.join(RESULTS, "videos", f"bandpass_pair_{vtag}_200-800.mp4")
    if os.path.exists(src):
        shutil.copy(src, os.path.join(out, "video_pair.mp4"))
        print("  copied video_pair.mp4")
    for row, nm in ((0, "video_raw.mp4"), (1, "video_bandpassed.mp4")):
        dst = os.path.join(out, nm)
        # the pair is two rows of three views; crop the row rather than re-render
        r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src,
                            "-vf", f"crop=iw:ih/2:0:{'0' if row == 0 else 'ih/2'}",
                            "-c:v", "libx264", "-pix_fmt", "yuv420p", dst],
                           capture_output=True, text=True)
        print(f"  {'wrote ' + nm if r.returncode == 0 else nm + ' FAILED: ' + r.stderr[:120]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", default="wamap2")
    ap.add_argument("--ref", default="torchref")
    ap.add_argument("--vtag", default=None, help="the *v render tag; default <tag>v")
    ap.add_argument("--parts", default="all")
    a = ap.parse_args()
    vtag = a.vtag or (a.tag + "v")
    out = os.path.join(RESULTS, a.tag)
    os.makedirs(out, exist_ok=True)
    want = (lambda p: True) if a.parts == "all" else (lambda p: p in a.parts.split(","))

    z, ref, c, t, fit, rz = load_all(a.tag, a.ref)
    frames = np.load(os.path.join(RESULTS, f"frames_{vtag}.npy"), mmap_mode="r")
    drive = np.asarray(np.load(os.path.join(RESULTS, f"drive_{vtag}.npy")), np.float64)
    print(f"  {a.tag}: frames {frames.shape}, drive {drive.shape}")
    stats = {"tag": a.tag, "best_sim": float(z["best_sim"]),
             "best_iter": int(z["best_iter"])}

    if want("summary"):
        sec_summary(out, a.tag, z, ref, c, t, rz, fit, None)
    if want("medium"):
        sec_medium(out, c, fit, z)
    if want("inputs"):
        sd, ma = sec_inputs(out, c, rz, drive)
        stats["drive_sd_range"] = [float(sd.min()), float(sd.max())]
    Z = None
    if want("scatter") or want("gradients"):
        Z = observable(np.asarray(frames), ref, t)
    if want("scatter"):
        stats.update(sec_scatter(out, c, t, Z))
    if want("gradients"):
        stats["gradients"] = sec_gradients(out, c, t, Z)
    if want("interference"):
        P = np.asarray(rz["profiles"], np.float32)
        tot, var_k = impulse_and_pieces(fit, P, drive.astype(np.float32), fit.save,
                                        frames.shape[0])
        # frames_<vtag>.npy is the BOLD-kernel-smoothed field; tot is the raw one, and a
        # field against its own low-passed version correlates ~0.95 however right the
        # decomposition is. Filtering is linear and commutes with the superposition, so
        # the total is smoothed for the check while the per-piece variances stay raw -
        # interference in the FIELD is the question, not interference in the observable.
        import units as _u, timescale as _ts
        _k = _u.smoothing_kernel(_ts.bold_fwhm_frames(float(ref["frame_s"])),
                                 verbose=False)
        chk = _u.smooth_frames(tot, _k)
        r = float(np.corrcoef(np.asarray(chk[600:-600]).ravel(),
                              np.asarray(frames[600:-600]).ravel())[0, 1])
        print(f"  superposition against the saved field: r = {r:.6f}"
              + ("" if r >= 0.99 else "   BELOW 0.99 - the decomposition does not "
                                      "reproduce the run, do not read the panels"))
        stats["superposition_r"] = r
        stats.update(sec_interference_pathways(out, c, t, rz, var_k, tot))
        del tot, var_k
    if want("velocity"):
        flux = energy_flux(fit, drive.astype(np.float32), rz["profiles"], fit.save)
        np.save(os.path.join(out, "energy_flux.npy"), flux)
        sec_velocity(out, c, flux)
    if want("videos"):
        sec_videos(out, a.tag, vtag)
    # MERGE, never overwrite. `stats` only holds what THIS run computed, so a --parts
    # subset - re-rendering just the videos, say - would otherwise truncate the summary
    # to the three keys it starts with and silently destroy every number the full run
    # had already written.
    sp = os.path.join(out, "summary.json")
    merged = {}
    if os.path.exists(sp):
        try:
            merged = json.load(open(sp))
        except (ValueError, OSError):
            merged = {}
    merged.update(stats)
    json.dump(merged, open(sp, "w"), indent=2)
    print(f"  wrote {sp}"
          + ("" if len(merged) == len(stats) else f"  (merged into {len(merged)} keys)"))


if __name__ == "__main__":
    main()
