"""Is the mesh the limit, or the physics?

Refining fsaverage5 -> fsaverage6 only helps if the model is currently unable to REPRESENT
structure at 10-20 mm. That is a question about the fields, not the fit, and it is answered
without any solve: take the cached impulse responses, treat (piece, time) as samples, and
measure how fast the field decorrelates over the sheet.

If the fields decorrelate over tens of millimetres, the grid is idle - the physics is
already smoother than fsaverage5 can carry, and four times the vertices would carry the
same fields at higher cost. If they decorrelate over a few millimetres, the grid is the
binding constraint and fsaverage6 is the fix.

Mesh spacing is quoted alongside, on the same surface the distances are measured on.

  python spatial_scale.py --cache <impulse .npy>
"""
import os, argparse, glob
import numpy as np

from mesh_cache import load_cortex
from paths import CACHE
import units


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache", default=None)
    ap.add_argument("--nvert", type=int, default=1200)
    ap.add_argument("--ntime", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    path = a.cache or os.path.join(
        CACHE, "impulse_fsaverage5_sig0.00641038_c1_Ld15827.6_spg0.03832_"
        "a[-0.3-0.050.01]_b[-0.030.350.35]_47_3584_16_prof47x854.499.npy")
    R = np.load(path, mmap_mode="r")
    K, T, nV = R.shape
    print(f"  {os.path.basename(path)}\n  {K} pieces x {T} frames x {nV} vertices")

    c = load_cortex("fsaverage5", verbose=False)
    rng = np.random.default_rng(a.seed)
    v = np.sort(rng.choice(nV, a.nvert, replace=False))
    ts = np.unique(np.linspace(1, T - 1, a.ntime).astype(int))
    X = np.asarray(R[:, ts][:, :, v], np.float64).reshape(-1, len(v))   # samples x vertices
    keep = X.std(1) > 0
    X = X[keep]
    X -= X.mean(0, keepdims=True)
    X /= np.maximum(X.std(0, keepdims=True), 1e-30)
    C = (X.T @ X) / X.shape[0]
    print(f"  {X.shape[0]} (piece, frame) samples")

    D = units.vertex_geodesic(c, v)[:, v]
    iu = np.triu_indices(len(v), 1)
    d, r = D[iu], C[iu]

    E = np.unique(np.sort(np.r_[c.F[:, [0, 1]], c.F[:, [1, 2]], c.F[:, [0, 2]]], 1), axis=0)
    L = np.linalg.norm(c.V[E[:, 0]] - c.V[E[:, 1]], axis=1)
    Dn = units.vertex_geodesic(c, E[:, 0][:2000])
    nn = np.array([np.partition(Dn[i][Dn[i] > 0], 5)[:6].mean() for i in range(len(Dn))])
    print(f"  mesh: mean edge {L.mean():.2f} mm (inflated), "
          f"mean white-surface nearest-neighbour spacing {nn.mean():.2f} mm")

    print(f"\n  field spatial autocorrelation, white-surface geodesic")
    print(f"    {'mm':>12s}{'n pairs':>10s}{'corr':>9s}")
    ed = np.array([0, 2, 4, 6, 8, 10, 15, 20, 25, 30, 40, 60, 90, 140, 250])
    prev = None
    half = None
    for lo, hi in zip(ed[:-1], ed[1:]):
        m = (d >= lo) & (d < hi)
        if m.sum() < 50:
            continue
        val = float(r[m].mean())
        print(f"    {lo:5.0f} - {hi:<4.0f}{int(m.sum()):10d}{val:+9.3f}")
        if half is None and prev is not None and prev[1] >= 0.5 > val:
            x0, y0 = prev; x1, y1 = 0.5 * (lo + hi), val
            half = x0 + (y0 - 0.5) * (x1 - x0) / (y0 - y1)
        prev = (0.5 * (lo + hi), val)
    if half:
        print(f"\n  the field falls to r = 0.5 at {half:.1f} mm")
        print(f"  that is {half / nn.mean():.1f} mesh spacings on fsaverage5, "
              f"{half / (nn.mean() / 2):.1f} on fsaverage6")


if __name__ == "__main__":
    main()
