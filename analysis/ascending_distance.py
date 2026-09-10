"""Score binned by geodesic distance from the driven set.

For every vertex, correlate its FC row (model against target) and bin the mean by the
vertex's distance to the nearest driven vertex. This measures whether waves actually
account for FC far from the input or only near it.

  python analysis/ascending_distance.py --tag ascend_085
"""
import _path  # noqa: F401
import os, argparse
import numpy as np

from mesh_cache import load_cortex
from paths import RESULTS
import fc_score
from diag_distance import distance_to_drive


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", default="ascend_085")
    ap.add_argument("--nbins", type=int, default=6)
    a = ap.parse_args()

    c = load_cortex("fsaverage5", verbose=False)
    t = fc_score.default_target(c, verbose=False)
    z = np.load(os.path.join(RESULTS, f"xspec_{a.tag}.npz"), allow_pickle=True)
    P = np.asarray(z["profiles"], np.float32)
    driven = np.abs(P).sum(0) > 0
    print(f"  {a.tag}: {int(driven.sum())} driven vertices "
          f"({c.A[driven].sum()/c.A.sum():.2%} of area)")

    F = np.asarray(np.load(os.path.join(RESULTS, f"frames_{a.tag}.npy")), np.float32)
    Z, _ = t.model_z(F)
    T = Z.shape[1]
    M = (Z @ Z.T) / T
    ssum = Z.sum(0)
    diag = (Z * Z).sum(1) / T
    V = Z.shape[0]
    m = ((Z @ ssum) / T - diag) / (V - 1)
    grand = (float(ssum @ ssum) / T - float(diag.sum())) / (V * (V - 1))
    M = M - m[:, None] - m[None, :] + grand
    Tf = np.asarray(t.target_fc(), np.float64)
    Tf = Tf - Tf.mean(1, keepdims=True)
    M = M - M.mean(1, keepdims=True)
    denom = np.sqrt((M ** 2).sum(1) * (Tf ** 2).sum(1))
    r = np.divide((M * Tf).sum(1), denom, out=np.zeros_like(denom), where=denom > 0)

    d = distance_to_drive(c, driven)[t.cols]
    edges = np.percentile(d, np.linspace(0, 100, a.nbins + 1))
    print(f"\n  distance bins (mm): {np.round(edges, 1)}")
    out = {}
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        m = (d >= lo) & (d < hi)
        out[f"mean_r_{i}"] = float(r[m].mean())
        out[f"n_{i}"] = int(m.sum())
        print(f"    {lo:6.1f}-{hi:6.1f} mm  n={int(m.sum()):>5d}  mean row corr "
              f"{r[m].mean():+.4f}")
    np.savez(os.path.join(RESULTS, f"ascending_distance_{a.tag}.npz"),
             edges=edges, r=r, d=d, driven=driven, **out)
    print(f"  wrote results/ascending_distance_{a.tag}.npz")


if __name__ == "__main__":
    main()
