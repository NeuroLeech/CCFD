"""The edge effect in a one-hot solve's input, plotted directly.

viz/input_embedding.py found the input's leading patterns organised by distance to the OUTER
EDGE of the driven territory - the border with cortex that gets no direct input - and not by
the input regions' own borders. The reading to test: the solve grades its input from that edge
so the waves it launches are directed. That predicts two things, both read here from the solved
cross-spectrum itself (exact, no draw), against the solver's random start:

  graded     input power, and the input's similarity to the nearest edge vertex, change
             steadily with distance from the edge
  directed   the input LAGS (or leads) the nearest edge vertex by an amount that grows with
             distance - a phase gradient, which is what launches a travelling wave one way.
             Lag from the in-band cross-spectrum phase: x_i(t) = x_j(t - tau) gives
             S_ij(f) = |X|^2 exp(-2 pi i f tau), so tau = -arg S_ij / (2 pi f), averaged over the
             band weighted by |S_ij| w_f. Positive: the vertex lags the edge.

Plus the magnitude of the embedding components against the same distance, from the npz
viz/input_embedding.py writes.

FIRST RUN, 2026-10-05, oh1880_nv2000_mf400 (524 of 1,880 driven vertices touch undriven
cortex; median depth 7 mm). GRADED, yes: edge vertices get 1.7x the median in-band input power,
falling steadily to 0.84x by ~10 mm (random start flat at 1.0), and the edge band is coherent -
+0.27 correlation with the nearest edge vertex within 5 mm, zero by ~15 mm. PHASE-GRADED, no:
the lag behind the nearest edge vertex stays within ~+-0.3 s, inside the random start's spread,
which over 12-100 s periods is a negligible phase. The solve builds a strong in-phase band along
the territory edge rather than staggering the input in time.

  python viz/edge_effect.py --tag oh1880_nv2000_mf400
"""
import _path  # noqa: F401
import os, re, argparse
import numpy as np

from paths import RESULTS

BINS = (0, 2.5, 5, 7.5, 10, 13, 17, 25)


def binned(d, y):
    out = []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        m = (d >= lo) & (d < hi)
        out.append((0.5 * (lo + hi), np.median(y[m]), np.percentile(y[m], 25),
                    np.percentile(y[m], 75), m.sum()) if m.sum() >= 5 else
                   (0.5 * (lo + hi), np.nan, np.nan, np.nan, m.sum()))
    return np.array(out)


def stats(L, w, band_m, f_hz, edge_of, cols=None):
    """power in band and over all bins, zero-lag correlation and lag to the nearest edge vertex"""
    K = L.shape[1]
    Pb = np.zeros(K); Pa = np.zeros(K)
    num = np.zeros(K, complex); lagn = np.zeros(K); lagd = np.zeros(K)
    for f in range(L.shape[0]):
        Lf = L[f]
        p = np.einsum('kr,kr->k', Lf, Lf.conj()).real * w[f]
        Pa += 2 * p
        if not band_m[f]:
            continue
        Pb += 2 * p
        s = np.einsum('kr,kr->k', Lf, Lf[edge_of].conj()) * w[f]   # S_{v, e(v)}
        num += s
        tau = -np.angle(s) / (2 * np.pi * f_hz[f])
        lagn += np.abs(s) * tau; lagd += np.abs(s)
    r0 = 2 * num.real / np.sqrt(Pb * Pb[edge_of])
    return Pb, Pa, r0, lagn / np.maximum(lagd, 1e-300)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", default="oh1880_nv2000_mf400")
    ap.add_argument("--out", default=os.path.join(RESULTS, "figs"))
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mesh_cache import load_cortex
    import units
    c = load_cortex("fsaverage5", verbose=False)
    z = np.load(os.path.join(RESULTS, f"xspec_{a.tag}.npz"), allow_pickle=True)
    L, tr = z["S_L"] / np.sqrt(float(z["S_tr"])), 1.0
    nf, K, r = L.shape
    w = np.asarray(z["H_w"], float)
    f_hz = np.asarray(z["idx"]) / (int(z["pad"]) * float(z["frame_s"]))
    lo, hi = (float(v) for v in z["band"])
    band_m = (f_hz >= lo) & (f_hz <= hi)
    half = nf * K * r
    L0 = np.random.default_rng(0).standard_normal(2 * half) * (1.0 / np.sqrt(K * r))
    L0 = L0[:half].reshape(nf, K, r) + 1j * L0[half:].reshape(nf, K, r)

    oh = np.load(os.path.join(RESULTS, "asc_first_onehot_1880.npz"), allow_pickle=True)
    vtx = np.argmax(oh["profiles"], axis=1)
    driven = np.zeros(c.nV, bool); driven[vtx] = True
    E = np.asarray(c.edges)
    on_edge = np.zeros(c.nV, bool)
    m = driven[E[:, 0]] != driven[E[:, 1]]
    on_edge[E[m, 0]] = True; on_edge[E[m, 1]] = True
    is_edge = on_edge[vtx]                                     # driven vertices touching undriven
    Dfull = units.vertex_geodesic(c, vtx)
    D = Dfull[:, vtx]
    d_terr = np.where(driven[None, :], np.inf, Dfull).min(1)
    eidx = np.flatnonzero(is_edge)
    edge_of = eidx[np.argmin(D[:, eidx], axis=1)]
    edge_of[is_edge] = eidx[np.argsort(D[np.ix_(eidx, eidx)], axis=1)[:, 1]]  # an edge vertex's
    # own nearest is itself; compare it with the next edge vertex instead
    print(f"  {is_edge.sum()} of {K} driven vertices touch undriven cortex; distance to the "
          f"edge median {np.median(d_terr):.1f} mm, max {d_terr.max():.1f} mm")

    S = {nm: stats(LL, w, band_m, f_hz, edge_of) for nm, LL in (("solved", L), ("random start", L0))}
    emb = os.path.join(a.out, f"input_embedding_{a.tag}.npz")
    G = np.load(emb, allow_pickle=True) if os.path.exists(emb) else None

    fig, axs = plt.subplots(1, 4, figsize=(20, 4.6))
    col = {"solved": "C3", "random start": "0.5"}
    def draw(ax, y_by, title, ylab, norm=False):
        for nm, y in y_by.items():
            if norm:
                y = y / np.median(y)
            b = binned(d_terr, y)
            ax.plot(b[:, 0], b[:, 1], "-o", color=col.get(nm, None), label=nm, ms=4)
            ax.fill_between(b[:, 0], b[:, 2], b[:, 3], color=col.get(nm, None), alpha=0.15)
        ax.set_xlabel("distance from the edge of the driven territory (mm)")
        ax.set_ylabel(ylab); ax.set_title(title, fontsize=10); ax.legend(fontsize=8)
    draw(axs[0], {k: v[0] for k, v in S.items()}, "input power, 0.01-0.08 Hz",
         "power / median", norm=True)
    draw(axs[1], {k: v[2] for k, v in S.items()}, "correlation with the nearest edge vertex",
         "zero-lag correlation, in band")
    draw(axs[2], {k: v[3] for k, v in S.items()}, "lag behind the nearest edge vertex",
         "seconds (+ = lags the edge)")
    if G is not None:
        for k, cc in ((0, "C3"), (1, "C1")):
            g = G["G_unfiltered"][:, k]
            b = binned(d_terr, np.abs(g - np.median(g)))
            axs[3].plot(b[:, 0], b[:, 1], "-o", color=cc, ms=4, label=f"component {k+1}")
        g = G["G_null"][:, 0]
        b = binned(d_terr, np.abs(g - np.median(g)))
        axs[3].plot(b[:, 0], b[:, 1] / np.nanmax(b[:, 1]) * np.nanmax(axs[3].lines[0].get_ydata()),
                    "-o", color="0.5", ms=4, label="random start comp 1 (rescaled)")
        axs[3].set_xlabel("distance from the edge of the driven territory (mm)")
        axs[3].set_ylabel("|component value|"); axs[3].legend(fontsize=8)
        axs[3].set_title("embedding components (unfiltered drive)", fontsize=10)
    fig.suptitle(f"{a.tag}: how the solved input changes from the edge of the driven territory "
                 f"inward  (median and IQR per distance bin)", x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = os.path.join(a.out, f"edge_effect_{a.tag}.png")
    fig.savefig(path, dpi=100); plt.close(fig)
    print(f"  wrote {path}")
    for nm, (Pb, Pa, r0, lag) in S.items():
        b = binned(d_terr, Pb / np.median(Pb)); bl = binned(d_terr, lag); br = binned(d_terr, r0)
        print(f"  {nm:<13s} power (edge -> interior): " + " ".join(f"{v:.2f}" for v in b[:, 1]))
        print(f"  {'':13s} corr with edge vertex:     " + " ".join(f"{v:+.2f}" for v in br[:, 1]))
        print(f"  {'':13s} lag behind edge (s):       " + " ".join(f"{v:+.2f}" for v in bl[:, 1]))
    print("  distance bins (mm): " + " ".join(f"{lo}-{hi}" for lo, hi in zip(BINS[:-1], BINS[1:])))


if __name__ == "__main__":
    main()
