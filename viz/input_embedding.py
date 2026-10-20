"""Diffusion embedding of a one-hot solve's 1,880 input timecourses, on the surface.

A drive is drawn from the solved cross-spectrum (realise, one long draw), giving one input
timecourse per driven vertex. Their correlation matrix is the affinity; BrainSpace's standard
gradient pipeline (top 10% per row, normalised-angle kernel, diffusion map) embeds it, and the
leading components are drawn on the surface with the 100 original region borders stroked, so
how the solve apportions input within and across regions - centre against edge, near against
far - can be seen rather than summarised.

Three versions, one figure each:
  unfiltered   the drive as injected, every frequency
  bandpassed   the same drive through the 0.01-0.08 Hz filter: the part the objective sees
  null         the solver's random starting factor, drawn and bandpassed the same way - what
               the embedding shows when nothing has been fitted

FIRST LOOK, 2026-10-05, oh1880_nv2000_mf400, one 2,308 s draw. The random start embeds as
noise (six near-equal diffusion eigenvalues, speckle); the fitted drive has two dominant
components. The structure is about the TERRITORY'S OUTER EDGE, not the region borders: the
magnitude of each component falls with distance to the nearest undriven vertex (rank
correlation -0.22 to -0.39, filtered and unfiltered; null -0.02 to +0.02), while distance to a
vertex's own region border shows nothing (-0.10 to +0.05). Vertices bordering undriven cortex get
the most distinctive timecourses. The unfiltered drive embeds more cleanly than the bandpassed
one: its unfitted high-frequency part is uncorrelated across vertices and drops out of the
affinity, and it includes 0.08-0.15 Hz, the range the solve shaped most.

  python viz/input_embedding.py --tag oh1880_nv2000_mf400
"""
import _path  # noqa: F401
import os, re, argparse
import numpy as np

from paths import RESULTS

VIEWS = ((8, 180, "lateral"), (8, 0, "medial"))


def embed(X, n=6):
    from brainspace.gradient import GradientMaps
    R = np.corrcoef(X.T)
    gm = GradientMaps(n_components=n, approach="dm", kernel="normalized_angle", random_state=0)
    gm.fit(R, sparsity=0.9)
    return gm.gradients_, gm.lambdas_


def _surface(ax, c, xy, vis, driven, vals, cmap, lims, tile_full, shading="gouraud"):
    from matplotlib.tri import Triangulation
    from matplotlib.collections import LineCollection
    from zones_surface import borders
    base = vis[c.F].all(1)
    ax.tripcolor(Triangulation(xy[:, 0], xy[:, 1], c.F[base]), np.zeros(base.sum()),
                 cmap="Greys", vmin=-1, vmax=6, rasterized=True)
    on = base & driven[c.F].all(1)
    ax.tripcolor(Triangulation(xy[:, 0], xy[:, 1], c.F[on]), vals if shading == "gouraud"
                 else vals[c.F[on]].mean(1), shading=shading, cmap=cmap, vmin=lims[0],
                 vmax=lims[1], rasterized=True)
    seg = borders(c, tile_full, driven, xy, vis)
    if len(seg):
        ax.add_collection(LineCollection(seg, colors="0.15", linewidths=0.3, alpha=0.5))
    m = driven & vis
    ax.set_xlim(xy[m, 0].min() - 4, xy[m, 0].max() + 4)
    ax.set_ylim(xy[m, 1].min() - 4, xy[m, 1].max() + 4)
    ax.set_aspect("equal"); ax.axis("off")


def figure(c, proj, vtx, tile_full, area_full, area, anames, G, lam, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    nC = 4
    driven = np.zeros(c.nV, bool); driven[vtx] = True
    na = len(anames)
    pal = np.vstack([plt.cm.tab20(np.linspace(0, 1, 20)), plt.cm.tab20b(np.linspace(0, 1, 20))])[:na]
    fig = plt.figure(figsize=(17, 5.2 * (nC + 1)))
    gs = fig.add_gridspec(nC + 1, 3, width_ratios=[1, 1, 0.42], hspace=0.08, wspace=0.02,
                          top=0.95, bottom=0.01, left=0.01, right=0.93)
    afull = np.zeros(c.nV); afull[vtx] = area
    for vi, (xy, vis, nm) in enumerate(proj):
        ax = fig.add_subplot(gs[0, vi])
        _surface(ax, c, xy, vis, driven, afull, ListedColormap(pal), (-0.5, na - 0.5),
                 tile_full, shading="flat")
        ax.set_title(("reference: the 38 areas, coloured as the labels on the right   "
                      if vi == 0 else "") + nm, fontsize=11, loc="left")
    for k in range(nC):
        g = G[:, k] - np.median(G[:, k])
        lim = np.percentile(np.abs(g), 98)
        full = np.zeros(c.nV); full[vtx] = g
        for vi, (xy, vis, nm) in enumerate(proj):
            ax = fig.add_subplot(gs[k + 1, vi])
            _surface(ax, c, xy, vis, driven, full, "RdBu_r", (-lim, lim), tile_full)
            ax.set_title((f"component {k+1}  ({lam[k]/lam.sum():.0%} of the leading "
                          f"{len(lam)})   " if vi == 0 else "") + nm, fontsize=11, loc="left")
    M = np.array([[G[area == i, k].mean() for k in range(nC)] for i in range(na)])
    M = M - np.median(G[:, :nC], axis=0)
    M = M / np.abs(M).max(0, keepdims=True)
    order = np.argsort(M[:, 0])
    ax = fig.add_subplot(gs[1:, 2])
    ax.imshow(M[order], cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.yaxis.tick_right()
    ax.set_yticks(range(na))
    ax.set_yticklabels(np.asarray(anames)[order], fontsize=9, fontweight="bold")
    for tl_, i in zip(ax.get_yticklabels(), order):
        tl_.set_color(pal[i])
    ax.set_xticks(range(nC)); ax.set_xticklabels([f"{k+1}" for k in range(nC)])
    ax.set_xlabel("component"); ax.set_title("area means", fontsize=11)
    fig.suptitle(title, fontsize=13, x=0.01, ha="left")
    fig.savefig(path, dpi=80); plt.close(fig)
    print(f"  wrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", default="oh1880_nv2000_mf400")
    ap.add_argument("--seconds", type=float, default=2308.0)
    ap.add_argument("--out", default=os.path.join(RESULTS, "figs"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    import xspec, bandpass, timescale
    from mesh_cache import load_cortex
    from render_regimes import _proj
    c = load_cortex("fsaverage5", verbose=False)
    z = np.load(os.path.join(RESULTS, f"xspec_{a.tag}.npz"), allow_pickle=True)
    fs = float(z["frame_s"]); lo, hi = (float(v) for v in z["band"])
    nfr = timescale.frames_for(a.seconds, fs)
    idx, ref = np.asarray(z["idx"]), int(z["ref_frames"])

    oh = np.load(os.path.join(RESULTS, "asc_first_onehot_1880.npz"), allow_pickle=True)
    vtx = np.argmax(oh["profiles"], axis=1)
    tl = np.load(os.path.join(RESULTS, "xspec_asc_first_area_100.npz"), allow_pickle=True)
    tile = np.argmax(tl["profiles"][:, vtx], axis=0)
    tile_full = np.full(c.nV, -1); tile_full[vtx] = tile
    names = np.array([re.sub(r"_\d+$", "", str(s)) for s in tl["tags"]])[tile]
    anames, area = np.unique(names, return_inverse=True)
    area_full = np.full(c.nV, -1); area_full[vtx] = area
    proj = _proj(c.V, c.F, VIEWS)
    import units
    Dfull = units.vertex_geodesic(c, vtx)                      # (1880, nV)
    driven = np.zeros(c.nV, bool); driven[vtx] = True
    d_terr = np.where(driven[None, :], np.inf, Dfull).min(1)   # to the nearest undriven vertex
    own = tl["profiles"][tile] > 1e-6
    d_tile = np.where(own, np.inf, Dfull).min(1)              # to the nearest vertex outside its region

    A = xspec.realise(None, idx, nfr, ref_frames=ref, seed=1000, factors=xspec.load_factors(z))
    L = z["S_L"]; nf, K, r = L.shape; half = nf * K * r
    L0 = np.random.default_rng(0).standard_normal(2 * half) * (1.0 / np.sqrt(K * r))
    L0 = L0[:half].reshape(nf, K, r) + 1j * L0[half:].reshape(nf, K, r)
    A0 = xspec.realise(None, idx, nfr, ref_frames=ref, seed=1000, factors=list(L0))
    versions = (("unfiltered", A, "the drive as injected, all frequencies"),
                ("bandpassed", bandpass.apply(A, fs, lo, hi), "the drive through 0.01-0.08 Hz"),
                ("null", bandpass.apply(A0, fs, lo, hi),
                 "the solver's random start, through 0.01-0.08 Hz (nothing fitted)"))
    from scipy.stats import spearmanr
    print(f"  distance to the edge of the driven territory: median {np.median(d_terr):.1f} mm; "
          f"to the vertex's own region border: median {np.median(d_tile):.1f} mm")
    keep = {}
    for nm, X, desc in versions:
        G, lam = embed(np.asarray(X, np.float64))
        keep[nm] = G
        print(f"  {nm:<11s} rank correlation of each component with distance to the edge of the "
              f"driven territory / to the region border:")
        for k in range(4):
            print(f"    component {k+1}: {spearmanr(G[:, k], d_terr).statistic:+.2f} / "
                  f"{spearmanr(G[:, k], d_tile).statistic:+.2f}    |value| vs territory edge "
                  f"{spearmanr(np.abs(G[:, k] - np.median(G[:, k])), d_terr).statistic:+.2f}")
        figure(c, proj, vtx, tile_full, area_full, area, anames, G, lam,
               f"{a.tag}: input timecourses of the 1,880 driven vertices, {desc}  "
               f"({a.seconds:.0f} s)",
               os.path.join(a.out, f"input_embedding_{a.tag}_{nm}.png"))
    np.savez(os.path.join(a.out, f"input_embedding_{a.tag}.npz"), vtx=vtx, tile=tile, area=area,
             anames=anames, d_terr=d_terr, d_tile=d_tile, **{f"G_{k}": v for k, v in keep.items()})


if __name__ == "__main__":
    main()
