"""Which vertices fail in the 10-20 mm band, where are they, and what drives them?

The accuracy-by-edge-length curve has a minimum at 10-30 mm that survives piece diameter,
medium reach, coupling, input profile shape and driven coverage. A curve cannot say
whether the miss is spread evenly over the sheet or concentrated somewhere, so this scores
each vertex on ITS OWN short edges only.

Per-vertex band accuracy is Spearman between the model's and the target's FC profile
restricted to partners whose white-surface geodesic distance falls inside the band. Every
vertex is scored against its own annulus - the partner set differs by construction, which
is the point - and the annulus is the whole sheet rather than a subsample, so the ~120
partners per vertex are all of them and not a thinned draw.

Three controls travel with the map, because a low number has three uninteresting causes:
the vertex is bad everywhere (ALL-PARTNER accuracy), there is nothing there to predict
(band |target FC| and per-vertex reliability from vertex_quality), or the annulus is
small (partner count).

  python band_fail.py --tag pr_taper --regions subcortical
"""
import os, argparse
import numpy as np
from scipy.stats import rankdata

from mesh_cache import load_cortex
from paths import RESULTS, CACHE
import fc_score, subparcels, ladder
from diag_maps import model_fc_rect
from diag_distance import distance_to_drive


def spearman_rows(M, T, mask):
    """Per-row Spearman over a per-row partner mask."""
    n = M.shape[0]
    out = np.full(n, np.nan)
    for i in range(n):
        k = mask[i]
        if k.sum() < 20:
            continue
        a, b = rankdata(M[i][k]), rankdata(T[i][k])
        a = (a - a.mean()) / max(a.std(), 1e-12)
        b = (b - b.mean()) / max(b.std(), 1e-12)
        out[i] = float(a @ b / len(a))
    return out


def parcel_table(cortex, cols, vals, n, k=15, label="band accuracy"):
    lab = np.asarray(cortex.lab)[cols]
    rows = []
    for p in np.unique(lab):
        if p == 0:
            continue
        m = (lab == p) & np.isfinite(vals)
        if m.sum() < 8:
            continue
        nm = cortex.names[p].replace("_ROI", "") if p < len(cortex.names) else str(p)
        rows.append((float(vals[m].mean()), int(m.sum()), float(n[m].mean()), nm))
    rows.sort()
    print(f"\n  {label}: WORST {k} Glasser parcels")
    print(f"    {'parcel':<14s}{'acc':>8s}{'nvert':>7s}{'partners':>10s}")
    for v, cnt, np_, nm in rows[:k]:
        print(f"    {nm:<14s}{v:+8.3f}{cnt:7d}{np_:10.0f}")
    print(f"  {label}: BEST {k} Glasser parcels")
    for v, cnt, np_, nm in rows[-k:][::-1]:
        print(f"    {nm:<14s}{v:+8.3f}{cnt:7d}{np_:10.0f}")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", default="pr_taper")
    ap.add_argument("--regions", default="subcortical")
    ap.add_argument("--split", type=int, default=50)
    ap.add_argument("--band", default="10,20")
    ap.add_argument("--block", type=int, default=512)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    lo, hi = [float(v) for v in a.band.split(",")]
    out = a.out or f"band_fail_{a.tag}_{int(lo)}_{int(hi)}"

    c = load_cortex("fsaverage5", verbose=False)
    t = fc_score.default_target(c, verbose=False)
    parcels, total = subparcels.region_set(c, a.regions, a.split)
    labels, tags = subparcels.split_parcels(c, parcels, total, verbose=False)
    driven = labels >= 0
    print(f"  {a.tag}: {a.regions}, {len(tags)} pieces, "
          f"{int(driven.sum())} driven vertices; band {lo:.0f}-{hi:.0f} mm")

    # nearest driven vertex, and which PIECE (hence which Glasser parcel) it belongs to
    d_drive = distance_to_drive(c, driven)[t.cols]
    G, _ = ladder._white_graph(c)
    from scipy.sparse.csgraph import dijkstra
    seeds = np.flatnonzero(driven)
    Dsd = dijkstra(G, indices=seeds)[:, t.cols]           # (nseed, nV)
    near_piece = labels[seeds][np.argmin(Dsd, 0)]
    near_parcel = np.array([int(tags[p].split("_")[0]) for p in near_piece])
    del Dsd

    frames = np.asarray(np.load(os.path.join(RESULTS, f"frames_{a.tag}.npy"),
                                mmap_mode="r"))
    Z, _ = t.model_z(frames)
    del frames
    TFC = np.asarray(t.target_fc(), np.float32)
    print(f"  {Z.shape[0]} vertices x {Z.shape[1]} frames; target {TFC.shape}")

    nV = t.nV
    acc_b = np.full(nV, np.nan); acc_a = np.full(nV, np.nan)
    npart = np.zeros(nV, np.int32)
    tstr = np.full(nV, np.nan); mstr = np.full(nV, np.nan)
    for s in range(0, nV, a.block):
        blk = np.arange(s, min(s + a.block, nV))
        D = dijkstra(G, indices=t.cols[blk])[:, t.cols]        # (nb, nV) mm
        M = model_fc_rect(t, Z, blk).T.astype(np.float64)      # (nb, nV)
        T = TFC[blk].astype(np.float64)
        band = (D >= lo) & (D < hi)
        allm = np.ones_like(band)
        allm[np.arange(len(blk)), blk] = False
        acc_b[blk] = spearman_rows(M, T, band)
        acc_a[blk] = spearman_rows(M, T, allm)
        npart[blk] = band.sum(1)
        for k, i in enumerate(blk):
            if band[k].sum():
                tstr[i] = np.abs(T[k][band[k]]).mean()
                mstr[i] = np.abs(M[k][band[k]]).mean()
        print(f"    {min(s + a.block, nV)}/{nV}", end="\r", flush=True)
        del D, M, T, band, allm
    print(f"  scored {int(np.isfinite(acc_b).sum())} of {nV} vertices "
          f"(median {np.median(npart)} partners in band)          ")

    ok = np.isfinite(acc_b) & np.isfinite(acc_a)
    print(f"\n  band accuracy   mean {np.nanmean(acc_b):+.4f}  sd {np.nanstd(acc_b):.4f}  "
          f"range {np.nanmin(acc_b):+.3f} to {np.nanmax(acc_b):+.3f}")
    print(f"  all-partner     mean {np.nanmean(acc_a):+.4f}  sd {np.nanstd(acc_a):.4f}")
    print(f"  corr(band, all-partner) {np.corrcoef(acc_b[ok], acc_a[ok])[0,1]:+.3f}"
          f"   - how much of the band miss is just a globally bad vertex")

    rel = None
    p = os.path.join(CACHE, "vertex_quality_1500_0_9217.npz")
    if os.path.exists(p):
        rel = np.load(p)["rel"]
        m = ok & np.isfinite(rel)
        print(f"  corr(band accuracy, per-vertex reliability) "
              f"{np.corrcoef(acc_b[m], rel[m])[0,1]:+.3f}")
    print(f"  corr(band accuracy, |target FC| in band) "
          f"{np.corrcoef(acc_b[ok], tstr[ok])[0,1]:+.3f}")
    print(f"  corr(band accuracy, partner count) "
          f"{np.corrcoef(acc_b[ok], npart[ok].astype(float))[0,1]:+.3f}")

    print(f"\n  by distance to the nearest driven vertex")
    print(f"    {'mm':>12s}{'n':>7s}{'band':>9s}{'all':>9s}{'|target|':>10s}"
          f"{'|model|':>9s}")
    ed = np.array([0, 1, 5, 10, 20, 30, 40, 60, 80, 250])
    for x, y in zip(ed[:-1], ed[1:]):
        m = ok & (d_drive >= x) & (d_drive < y)
        if m.sum() < 20:
            continue
        print(f"    {x:4.0f} - {y:<5.0f}{int(m.sum()):7d}{acc_b[m].mean():+9.3f}"
              f"{acc_a[m].mean():+9.3f}{tstr[m].mean():10.4f}{mstr[m].mean():9.4f}")

    parcel_table(c, t.cols, acc_b, npart.astype(float),
                 label=f"{int(lo)}-{int(hi)} mm accuracy")
    print(f"\n  the same vertices' ALL-PARTNER accuracy, worst-band parcels first")
    lab = np.asarray(c.lab)[t.cols]
    rows = []
    for pp in np.unique(lab):
        if pp == 0:
            continue
        m = (lab == pp) & ok
        if m.sum() < 8:
            continue
        nm = c.names[pp].replace("_ROI", "") if pp < len(c.names) else str(pp)
        rows.append((acc_b[m].mean(), acc_a[m].mean(), tstr[m].mean(),
                     d_drive[m].mean(), int(m.sum()), nm))
    rows.sort()
    print(f"    {'parcel':<14s}{'band':>8s}{'all':>8s}{'band-all':>10s}"
          f"{'|target|':>10s}{'d(drive)':>10s}")
    for b, al, ts, dd, cnt, nm in rows[:15]:
        print(f"    {nm:<14s}{b:+8.3f}{al:+8.3f}{b-al:+10.3f}{ts:10.4f}{dd:10.1f}")

    print(f"\n  by NEAREST driven parcel (the input region the vertex sits closest to)")
    print(f"    {'input parcel':<14s}{'n':>7s}{'band':>9s}{'all':>9s}"
          f"{'d(drive)':>10s}")
    grp = []
    for pp in np.unique(near_parcel):
        m = ok & (near_parcel == pp)
        if m.sum() < 20:
            continue
        nm = c.names[pp].replace("_ROI", "") if pp < len(c.names) else str(pp)
        grp.append((acc_b[m].mean(), acc_a[m].mean(), d_drive[m].mean(),
                    int(m.sum()), nm))
    grp.sort()
    for b, al, dd, cnt, nm in grp:
        print(f"    {nm:<14s}{cnt:7d}{b:+9.3f}{al:+9.3f}{dd:10.1f}")

    np.savez(os.path.join(RESULTS, f"{out}.npz"), acc_band=acc_b, acc_all=acc_a,
             npart=npart, tstr=tstr, mstr=mstr, d_drive=d_drive,
             near_parcel=near_parcel, cols=t.cols, band=np.array([lo, hi]))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from render_regimes import _proj
    from plot_fc_map import surface_row
    proj = _proj(c.V, c.F)
    fig = plt.figure(figsize=(4.0 * len(proj), 3.3 * 4))
    gs = fig.add_gridspec(4, len(proj), hspace=0.04, wspace=0.02)
    def lim(v, q=0.02):
        v = v[np.isfinite(v)]
        return (float(np.quantile(v, q)), float(np.quantile(v, 1 - q)))
    fill = lambda v: np.where(np.isfinite(v), v, np.nanmean(v))
    surface_row(fig, gs, 0, proj, fill(acc_b), c, t.cols, "magma", lim(acc_b),
                f"{int(lo)}-{int(hi)} mm accuracy")
    surface_row(fig, gs, 1, proj, fill(acc_a), c, t.cols, "magma", lim(acc_a),
                "all-partner accuracy")
    dd = fill(acc_b) - fill(acc_a)
    v = max(abs(np.quantile(dd, 0.02)), abs(np.quantile(dd, 0.98)))
    surface_row(fig, gs, 2, proj, dd, c, t.cols, "coolwarm", (-v, v), "band minus all")
    surface_row(fig, gs, 3, proj, np.where(driven[t.cols], 1.0, 0.0), c, t.cols,
                "Greys", (0, 1.4), "driven")
    p = os.path.join(RESULTS, f"{out}.png")
    fig.savefig(p, dpi=130, bbox_inches="tight")
    print(f"\n  wrote {p}")


if __name__ == "__main__":
    main()
