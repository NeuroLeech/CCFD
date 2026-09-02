"""How much of a vertex's 10-20 mm profile is reproducible at all?

Per-vertex accuracy inside a narrow annulus is a Spearman over ~120 partners whose FC
values may be almost identical to one another. Where the target has little to separate
those partners, a low score means nothing about the model. reliability.py answers this for
the matrix as a whole and vertex_quality.py for whole profiles; neither answers it inside a
band, and the band is where the model fails.

Same estimator as the target and as holdout.half_targets: split the 99 subjects into
halves, build a group FC per half in Fisher z, double-centre the FULL matrix (the centring
is what the model is scored against, and it cannot be done from the band alone), and only
then read out each vertex's band partners. The correlation between the halves is the
ceiling on that vertex's band accuracy.

  python band_ceiling.py --band 10,20
"""
import os, argparse, time
import numpy as np
from scipy.stats import rankdata
from scipy.sparse.csgraph import dijkstra

from mesh_cache import load_cortex
from paths import RESULTS, CACHE
import fc_score, ladder, fc_group_nki as nki
from fc_score import double_centre


def band_lists(cortex, cols, lo, hi, block=1024):
    """-> list of partner index arrays, one per vertex, in FC vertex order."""
    G, _ = ladder._white_graph(cortex)
    out = []
    for s in range(0, len(cols), block):
        blk = cols[s:s + block]
        D = dijkstra(G, indices=blk)[:, cols]
        B = (D >= lo) & (D < hi)
        out.extend([np.flatnonzero(B[k]) for k in range(len(blk))])
        del D, B
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--band", default="10,20")
    ap.add_argument("--tag", default="pr_taper")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    lo, hi = [float(v) for v in a.band.split(",")]

    c = load_cortex("fsaverage5", verbose=False)
    t = fc_score.default_target(c, verbose=False)
    part = band_lists(c, t.cols, lo, hi)
    npart = np.array([len(p) for p in part])
    print(f"  band {lo:.0f}-{hi:.0f} mm: median {np.median(npart):.0f} partners per vertex")

    # spread of the target's own values inside the annulus - the quantity a Spearman
    # over that annulus is trying to rank. Little spread, nothing to rank.
    TFC = np.asarray(t.target_fc(), np.float32)
    sd_b = np.array([float(TFC[i, p].std()) if len(p) > 2 else np.nan
                     for i, p in enumerate(part)])
    print(f"  target sd within band: mean {np.nanmean(sd_b):.4f}, "
          f"vs sd over ALL partners {float(TFC[0].std()):.4f} (vertex 0)")
    del TFC

    cache = os.path.join(CACHE, f"band_ceiling_{int(lo)}_{int(hi)}_{a.seed}.npz")
    if os.path.exists(cache):
        z = np.load(cache); ceil = z["ceil"]
    else:
        files = nki.subject_files("left")
        rng = np.random.default_rng(a.seed)
        which = rng.permutation(len(files)) % 2          # same convention as holdout
        acc = [np.zeros((t.nV, t.nV), np.float32) for _ in range(2)]
        cnt = [0, 0]
        t0 = time.time()
        for s, path in enumerate(files):
            h = int(which[s])
            X = nki.load_subject(path)[t.vertices]
            Z = rankdata(X, axis=1).astype(np.float32)
            Z -= Z.mean(1, keepdims=True)
            Z /= np.maximum(Z.std(1, keepdims=True), 1e-12)
            S = (Z @ Z.T) / Z.shape[1]
            np.clip(S, -0.9999, 0.9999, out=S)
            acc[h] += np.arctanh(S)
            cnt[h] += 1
            if (s + 1) % 10 == 0 or s == len(files) - 1:
                print(f"    {s+1:3d}/{len(files)}  [{time.time()-t0:.0f}s]", flush=True)
            del X, Z, S
        H = []
        for h in (0, 1):
            FC = np.tanh(acc[h] / cnt[h])
            np.fill_diagonal(FC, 1.0)
            H.append(double_centre(FC, inplace=True))
            acc[h] = None
        print(f"  halves of {cnt[0]} and {cnt[1]} subjects, double-centred like the target")
        ceil = np.full(t.nV, np.nan)
        for i, p in enumerate(part):
            if len(p) < 20:
                continue
            u, v = rankdata(H[0][i, p]), rankdata(H[1][i, p])
            u = (u - u.mean()) / max(u.std(), 1e-12)
            v = (v - v.mean()) / max(v.std(), 1e-12)
            ceil[i] = float(u @ v / len(u))
        del H
        np.savez(cache, ceil=ceil, npart=npart, sd_band=sd_b)

    z = np.load(os.path.join(RESULTS, f"band_fail_{a.tag}_{int(lo)}_{int(hi)}.npz"))
    acc_b, acc_a, d_drive = z["acc_band"], z["acc_all"], z["d_drive"]
    ok = np.isfinite(ceil) & np.isfinite(acc_b)
    print(f"\n  BAND CEILING (half A vs half B, {int(ok.sum())} vertices)")
    print(f"    mean {np.nanmean(ceil):+.4f}  sd {np.nanstd(ceil):.4f}  "
          f"range {np.nanmin(ceil):+.3f} to {np.nanmax(ceil):+.3f}")
    print(f"    model band accuracy mean {np.nanmean(acc_b):+.4f}")
    print(f"    corr(model band accuracy, ceiling) {np.corrcoef(acc_b[ok], ceil[ok])[0,1]:+.3f}")
    m2 = ok & np.isfinite(sd_b)
    print(f"    corr(model band accuracy, target sd in band) "
          f"{np.corrcoef(acc_b[m2], sd_b[m2])[0,1]:+.3f}")
    print(f"    corr(ceiling, target sd in band) {np.corrcoef(ceil[m2], sd_b[m2])[0,1]:+.3f}")

    print(f"\n  by distance to the nearest driven vertex")
    print(f"    {'mm':>12s}{'n':>7s}{'ceiling':>9s}{'model':>9s}{'shortfall':>11s}"
          f"{'sd(band)':>10s}")
    ed = np.array([0, 1, 5, 10, 20, 30, 40, 60, 250])
    for x, y in zip(ed[:-1], ed[1:]):
        m = ok & (d_drive >= x) & (d_drive < y)
        if m.sum() < 20:
            continue
        print(f"    {x:4.0f} - {y:<5.0f}{int(m.sum()):7d}{ceil[m].mean():+9.3f}"
              f"{acc_b[m].mean():+9.3f}{(ceil[m]-acc_b[m]).mean():+11.3f}"
              f"{np.nanmean(sd_b[m]):10.4f}")

    lab = np.asarray(c.lab)[t.cols]
    rows = []
    for p in np.unique(lab):
        if p == 0:
            continue
        m = (lab == p) & ok
        if m.sum() < 8:
            continue
        nm = c.names[p].replace("_ROI", "") if p < len(c.names) else str(p)
        rows.append((acc_b[m].mean(), ceil[m].mean(), np.nanmean(sd_b[m]), int(m.sum()), nm))
    rows.sort()
    print(f"\n  Glasser parcels, worst model band accuracy first")
    print(f"    {'parcel':<14s}{'model':>9s}{'ceiling':>9s}{'shortfall':>11s}{'sd(band)':>10s}")
    for b, cl, sd, cnt, nm in rows[:18]:
        print(f"    {nm:<14s}{b:+9.3f}{cl:+9.3f}{cl-b:+11.3f}{sd:10.4f}")
    print(f"  ... best")
    for b, cl, sd, cnt, nm in rows[-10:][::-1]:
        print(f"    {nm:<14s}{b:+9.3f}{cl:+9.3f}{cl-b:+11.3f}{sd:10.4f}")

    print(f"\n  parcels ranked by SHORTFALL (ceiling minus model), worst first")
    rows.sort(key=lambda r: r[1] - r[0], reverse=True)
    for b, cl, sd, cnt, nm in rows[:18]:
        print(f"    {nm:<14s}{b:+9.3f}{cl:+9.3f}{cl-b:+11.3f}{sd:10.4f}")

    np.savez(os.path.join(RESULTS, f"band_ceiling_{int(lo)}_{int(hi)}.npz"),
             ceil=ceil, sd_band=sd_b, npart=npart)
    print(f"\n  wrote results/band_ceiling_{int(lo)}_{int(hi)}.npz")


if __name__ == "__main__":
    main()
