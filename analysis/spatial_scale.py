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

SPEED DOES NOT CHANGE THE FIELD'S SPATIAL SCALE. Measured 2026-10-02 on the three speed-sweep
caches, 1000 mapped columns, white-surface geodesic:

    geodesic      1x 1.47 mm/s   2x 2.95 mm/s   3x 4.42 mm/s
     2 -  4 mm       +0.586         +0.578         +0.559
     4 -  6          +0.369         +0.365         +0.341
     8 - 10          +0.158         +0.161         +0.156
    10 - 15          +0.061         +0.067         +0.070
    15 - 20          +0.018         +0.020         +0.031
    r = 0.5 at        3.8 mm         3.7 mm         3.5 mm

It could not have been otherwise. Each step advances the wave a fixed distance - c*dt = CFL*d_min -
so mm/s is a labelling of steps per second and the INSTANTANEOUS spatial pattern is set by Ld and
the mesh, which are identical across the sweep. What changes with speed is how fast that pattern
evolves in seconds, so the 0.01-0.08 Hz observable integrates over more of its evolution and more
cancels. That is consistent with the realised field's rank falling 12.5 -> 7.6 -> 2.5 while the raw
field's spatial scale holds: the filter does it, not the medium's spatial structure.

Separately, the field decorrelates at 3.5-3.8 mm - about 1.2 mesh spacings - and carries r ~ 0.06
at 10-15 mm at EVERY speed, so the 10-30 mm problem is not speed-dependent.

  python analysis/spatial_scale.py --cache <impulse .npy>
  python analysis/spatial_scale.py --from-run flat100_s4.5 --nvert 900 --ntime 60
"""
import _path  # noqa: F401  - puts the sibling code folders on sys.path
import os, argparse, glob
import numpy as np

from mesh_cache import load_cortex
from paths import CACHE
import units


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache", default=None)
    ap.add_argument("--from-run", default=None, dest="from_run",
                    help="an xspec_<tag> whose cache is vertex-RESTRICTED. A keep1000 cache's "
                         "column j is cortex vertex t.cols[sub[j]], not j, so treating columns "
                         "as cortex indices would measure geodesics between the wrong vertices. "
                         "This reads sub from the run and finds its cache by nsteps and save")
    ap.add_argument("--nvert", type=int, default=1200)
    ap.add_argument("--ntime", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    path = a.cache or os.path.join(
        CACHE, "impulse_fsaverage5_sig0.00641038_c1_Ld15827.6_spg0.03832_"
        "a[-0.3-0.050.01]_b[-0.030.350.35]_47_3584_16_prof47x854.499.npy")
    cols = None
    if a.from_run:
        from paths import RESULTS
        import fc_score
        z = np.load(os.path.join(RESULTS, f"xspec_{a.from_run}.npz"), allow_pickle=True)
        # CONSTRUCT the key, do not glob it. nsteps and save alone are not unique - 4352_4
        # matches both the flat sweep cache and the graded-clipped keep400 one, and globbing
        # silently returned the wrong medium at the wrong width with a mismatched `sub`,
        # which reads out as ~0 correlation at every distance.
        import xspec, bo_step
        save, imp = int(z["save"]), int(z["impulse_frames"])
        c0 = load_cortex("fsaverage5", verbose=False)
        t0 = fc_score.default_target(c0, verbose=False)
        sub = np.asarray(z["sub"], np.int64)
        cols = np.asarray(t0.cols)[sub]
        p0, _, _ = bo_step.unpack(np.asarray(z["x"], float), c0)
        mc = str(z["map_clip"]) if "map_clip" in z.files else "none"
        mtag = "" if mc in ("", "none", "None") else f"_mc{mc}"
        P0 = np.asarray(z["profiles"], np.float32)
        key = (f"{c0.mesh}_sig{p0['sig0']:.6g}_c{p0['c0']:.6g}_Ld{p0['Ld']:.6g}"
               f"_spg{p0.get('sponge_scale', 1.0):.4g}_a{np.round(p0.get('a', 0), 3)}"
               f"_b{np.round(p0.get('b', 0), 3)}{mtag}_{len(P0)}_{imp*save}_{save}"
               f"_{xspec.profile_tag(P0)}{xspec.keep_tag(cols)}")
        path = os.path.join(CACHE, "impulse_" + key.replace(" ", "") + ".npy")
        if not os.path.exists(path):
            raise SystemExit(f"  no cache at {os.path.basename(path)}")
        print(f"  {a.from_run}: save {save}, {len(cols)} columns mapped through t.cols[sub]")

    R = np.load(path, mmap_mode="r")
    K, T, nV = R.shape
    print(f"  {os.path.basename(path)}\n  {K} pieces x {T} frames x {nV} vertices")

    c = load_cortex("fsaverage5", verbose=False)
    rng = np.random.default_rng(a.seed)
    v = np.sort(rng.choice(nV, min(a.nvert, nV), replace=False))
    vcort = v if cols is None else np.asarray(cols)[v]
    ts = np.unique(np.linspace(1, T - 1, a.ntime).astype(int))
    X = np.asarray(R[:, ts][:, :, v], np.float64).reshape(-1, len(v))   # samples x vertices
    keep = X.std(1) > 0
    X = X[keep]
    X -= X.mean(0, keepdims=True)
    X /= np.maximum(X.std(0, keepdims=True), 1e-30)
    C = (X.T @ X) / X.shape[0]
    print(f"  {X.shape[0]} (piece, frame) samples")

    D = units.vertex_geodesic(c, vcort)[:, vcort]
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
