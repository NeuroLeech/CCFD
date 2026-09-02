"""Does the short-range miss come from the model being too SMOOTH at short range?

The 10-20 mm failure is worst on and beside the driven parcels, and worst where a primary
area abuts a belt (A1/LBelt/PBelt, V1/V8/VVC, 3b/area 1). Those are places where two
vertices 15 mm apart have genuinely different FC profiles. A wave medium driven with one
taper over a whole parcel cannot produce that: it makes near neighbours near-identical.

LOCAL HOMOGENEITY is the direct measure. For each vertex, the mean correlation between its
own FC profile and the profiles of its partners in the band, computed over a common
partner set so model and target are measured the same way. If the reading above is right
the model's homogeneity should exceed the target's, the excess should be largest exactly
where band accuracy is lowest, and it should NOT be a global offset.

  python band_homog.py --tag pr_taper --regions subcortical
"""
import os, argparse
import numpy as np
from scipy.sparse.csgraph import dijkstra

from mesh_cache import load_cortex
from paths import RESULTS
import fc_score, subparcels, ladder
from diag_maps import model_fc_rect
from diag_distance import distance_to_drive


def zrows(X):
    X = X - X.mean(1, keepdims=True)
    return X / np.maximum(X.std(1, keepdims=True), 1e-12)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", default="pr_taper")
    ap.add_argument("--regions", default="subcortical")
    ap.add_argument("--split", type=int, default=40)
    ap.add_argument("--band", default="10,20")
    ap.add_argument("--npart", type=int, default=2000)
    ap.add_argument("--block", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    lo, hi = [float(v) for v in a.band.split(",")]

    c = load_cortex("fsaverage5", verbose=False)
    t = fc_score.default_target(c, verbose=False)
    parcels, total = subparcels.region_set(c, a.regions, a.split)
    labels, tags = subparcels.split_parcels(c, parcels, total, verbose=False)
    d_drive = distance_to_drive(c, labels >= 0)[t.cols]

    frames = np.asarray(np.load(os.path.join(RESULTS, f"frames_{a.tag}.npy"),
                                mmap_mode="r"))
    Z, _ = t.model_z(frames); del frames
    rng = np.random.default_rng(a.seed)
    part = np.sort(rng.choice(t.nV, a.npart, replace=False))
    TFC = np.asarray(t.target_fc(), np.float32)
    Pt = zrows(TFC[:, part].astype(np.float64))
    Pm = zrows(model_fc_rect(t, Z, part).astype(np.float64))
    del TFC, Z
    print(f"  profiles over {len(part)} common partners; band {lo:.0f}-{hi:.0f} mm")

    G, _ = ladder._white_graph(c)
    nV, nP = t.nV, len(part)
    h_emp = np.full(nV, np.nan); h_mod = np.full(nV, np.nan); n_b = np.zeros(nV, int)
    for s in range(0, nV, a.block):
        blk = np.arange(s, min(s + a.block, nV))
        D = dijkstra(G, indices=t.cols[blk])[:, t.cols]
        B = ((D >= lo) & (D < hi)).astype(np.float64)
        cnt = B.sum(1)
        # mean over band partners of the profile correlation with this vertex
        h_emp[blk] = np.einsum("ip,ip->i", Pt[blk], (B @ Pt) / np.maximum(cnt, 1)[:, None]) / nP
        h_mod[blk] = np.einsum("ip,ip->i", Pm[blk], (B @ Pm) / np.maximum(cnt, 1)[:, None]) / nP
        n_b[blk] = cnt
        h_emp[blk[cnt < 20]] = np.nan; h_mod[blk[cnt < 20]] = np.nan
        print(f"    {blk[-1]+1}/{nV}", end="\r", flush=True)
        del D, B
    ok = np.isfinite(h_emp) & np.isfinite(h_mod)
    print(f"  {int(ok.sum())} vertices scored                    ")

    z = np.load(os.path.join(RESULTS, f"band_fail_{a.tag}_{int(lo)}_{int(hi)}.npz"))
    acc = z["acc_band"]; okk = ok & np.isfinite(acc)
    ex = h_mod - h_emp
    print(f"\n  local homogeneity over {lo:.0f}-{hi:.0f} mm partners")
    print(f"    target  mean {np.nanmean(h_emp):+.4f}  sd {np.nanstd(h_emp):.4f}")
    print(f"    model   mean {np.nanmean(h_mod):+.4f}  sd {np.nanstd(h_mod):.4f}")
    print(f"    excess  mean {np.nanmean(ex):+.4f}  sd {np.nanstd(ex):.4f}  "
          f"(model minus target)")
    print(f"\n  corr(band accuracy, excess homogeneity) "
          f"{np.corrcoef(acc[okk], ex[okk])[0,1]:+.3f}")
    print(f"  corr(band accuracy, target homogeneity) "
          f"{np.corrcoef(acc[okk], h_emp[okk])[0,1]:+.3f}")
    print(f"  corr(band accuracy, model  homogeneity) "
          f"{np.corrcoef(acc[okk], h_mod[okk])[0,1]:+.3f}")

    print(f"\n  by distance to the nearest driven vertex")
    print(f"    {'mm':>12s}{'n':>7s}{'target h':>10s}{'model h':>9s}{'excess':>9s}"
          f"{'band acc':>10s}")
    ed = np.array([0, 1, 5, 10, 20, 30, 40, 60, 250])
    for x, y in zip(ed[:-1], ed[1:]):
        m = okk & (d_drive >= x) & (d_drive < y)
        if m.sum() < 20:
            continue
        print(f"    {x:4.0f} - {y:<5.0f}{int(m.sum()):7d}{h_emp[m].mean():+10.4f}"
              f"{h_mod[m].mean():+9.4f}{ex[m].mean():+9.4f}{acc[m].mean():+10.3f}")

    lab = np.asarray(c.lab)[t.cols]
    rows = []
    for p in np.unique(lab):
        if p == 0:
            continue
        m = (lab == p) & okk
        if m.sum() < 8:
            continue
        nm = c.names[p].replace("_ROI", "") if p < len(c.names) else str(p)
        rows.append((acc[m].mean(), h_emp[m].mean(), h_mod[m].mean(), int(m.sum()), nm))
    rows.sort()
    print(f"\n  Glasser parcels, worst band accuracy first")
    print(f"    {'parcel':<14s}{'band acc':>10s}{'target h':>10s}{'model h':>9s}"
          f"{'excess':>9s}")
    for b, he, hm, cnt, nm in rows[:15]:
        print(f"    {nm:<14s}{b:+10.3f}{he:+10.4f}{hm:+9.4f}{hm-he:+9.4f}")
    print(f"  ... best band accuracy")
    for b, he, hm, cnt, nm in rows[-10:][::-1]:
        print(f"    {nm:<14s}{b:+10.3f}{he:+10.4f}{hm:+9.4f}{hm-he:+9.4f}")

    np.savez(os.path.join(RESULTS, f"band_homog_{a.tag}_{int(lo)}_{int(hi)}.npz"),
             h_emp=h_emp, h_mod=h_mod, n_band=n_b, cols=t.cols)
    print(f"\n  wrote results/band_homog_{a.tag}_{int(lo)}_{int(hi)}.npz")


if __name__ == "__main__":
    main()
