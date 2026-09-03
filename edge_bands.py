"""Accuracy by edge length, several runs side by side, on one shared vertex sample.

diag_edges does this for one tag and recomputes the geodesics each time. Comparing runs
needs the SAME vertices and the same bins, or the bin populations differ and the columns
are not comparable.

  python edge_bands.py --tags sh_base,sh_002,sh_004,sh_008,sh_long
"""
import os, argparse
import numpy as np
from scipy.stats import spearmanr

from mesh_cache import load_cortex
from paths import RESULTS
import fc_score, units


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tags", default="sh_base,sh_002,sh_004,sh_008,sh_long")
    ap.add_argument("--labels", default="")
    ap.add_argument("--nvert", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    tags = [t for t in a.tags.split(",") if t]
    labs = [l for l in a.labels.split(",") if l] or tags

    c = load_cortex("fsaverage5", verbose=False)
    t = fc_score.default_target(c, verbose=False)
    rng = np.random.default_rng(a.seed)
    v = np.sort(rng.choice(t.nV, a.nvert, replace=False))
    iu = np.triu_indices(len(v), 1)
    D = units.vertex_geodesic(c, t.cols[v])[:, t.cols[v]]
    T = np.asarray(t.target_fc()[np.ix_(v, v)], np.float64)
    T = T - T.mean(0, keepdims=True) - T.mean(1, keepdims=True) + T.mean()
    d, tg = D[iu], T[iu]

    ed = np.array([0, 10, 20, 30, 40, 60, 80, 120, 250])
    masks = [(d >= lo) & (d < hi) for lo, hi in zip(ed[:-1], ed[1:])]
    print(f"  {len(v)} vertices, {len(d)} edges; bin sizes "
          + " ".join(str(int(m.sum())) for m in masks))
    hdr = "".join(f"{lo:>4.0f}-{hi:<4.0f}" for lo, hi in zip(ed[:-1], ed[1:]))
    print(f"\n  {'config':<14s}{'all':>8s}  {hdr}")
    for tag, lab in zip(tags, labs):
        p = os.path.join(RESULTS, f"frames_{tag}.npy")
        if not os.path.exists(p):
            print(f"  {lab:<14s}  (no frames_{tag}.npy)")
            continue
        Z, _ = t.model_z(np.asarray(np.load(p, mmap_mode="r")))
        Zs = Z[v].astype(np.float64)
        Zs -= Zs.mean(1, keepdims=True)
        Zs /= np.maximum(Zs.std(1, keepdims=True), 1e-12)
        M = (Zs @ Zs.T) / Zs.shape[1]
        M = M - M.mean(0, keepdims=True) - M.mean(1, keepdims=True) + M.mean()
        m = M[iu]
        row = "".join(f"{spearmanr(m[k], tg[k]).statistic:9.3f}" for k in masks)
        print(f"  {lab:<14s}{spearmanr(m, tg).statistic:8.3f}  {row}", flush=True)
        del Z, Zs, M, m


if __name__ == "__main__":
    main()
