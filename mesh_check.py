"""Mesh convergence: is fsaverage5 resolving the fields, or smoothing them?

The 10-20 mm miss survives every input and medium parameter tried, and the fields
themselves fall to r = 0.5 over 7.7 mm - under three vertex spacings on fsaverage5. That
is close enough to the grid limit that the failure could be discretisation rather than
physics, and the way to find out is the standard convergence check: integrate the SAME
continuum medium on a finer mesh and see whether the answer moves.

The medium is carried across as the continuum dict `p`, not as per-step numbers. Every
rate in it is per unit time (bo_step.unpack divides the per-step targets by dt), so
handing the same p to a finer mesh gives the same physics with a smaller timestep - the
timestep follows the CFL bound on its own. `save` is raised to hold the frame duration
fixed, so a frame is the same 0.1613 s on both.

fsaverage5's vertices nest exactly inside fsaverage6 (white-surface coordinates agree to
1.4 microns on all 9,373 shared vertices), so the two runs are compared vertex to vertex
with no interpolation on the readout side. The drive is carried the other way, nearest
neighbour, so both meshes are pushed at the same places.

What this does NOT isolate: the two inflated surfaces are separate inflations of the same
white surface, not a subdivision of one another (median 3 mm apart), so a difference here
is refinement plus a small metric change. Agreement is therefore the strong result.

  python mesh_check.py
"""
import os, argparse, time
import numpy as np

from mesh_cache import load_cortex
from paths import RESULTS
import bo_step, xspec, subparcels, ladder, timescale, fluid as fl
from best_fit import BEST_X


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--regions", default="subcortical")
    ap.add_argument("--split", type=int, default=40)
    ap.add_argument("--oversample", type=int, default=4)
    ap.add_argument("--spread-mm-s", type=float, default=6.0, dest="spread")
    ap.add_argument("--impulse-frames", type=int, default=224, dest="nframe")
    ap.add_argument("--workers", type=int, default=0)
    a = ap.parse_args()

    c5 = load_cortex("fsaverage5", verbose=False)
    c6 = load_cortex("fsaverage6", verbose=False)
    clock = timescale.plan(a.oversample, decay_s=timescale.BOLD_TAU_S,
                           spread_mm_s=a.spread)
    x = BEST_X.copy()
    x[3] = np.log10(clock["save"]); x[0] = np.log10(clock["damp"])
    p, save5, _ = bo_step.unpack(x, c5)

    f5, _ = fl.fields(c5, p); f6, _ = fl.fields(c6, p)
    dt5 = fl.CFL * c5.d.min() / float(f5.max())
    dt6 = fl.CFL * c6.d.min() / float(f6.max())
    save6 = int(round(save5 * dt5 / dt6))
    print(f"  fs5: {c5.nV} vertices, dt {dt5:.5g}, save {save5} -> frame "
          f"{save5*dt5:.4g} model-time")
    print(f"  fs6: {c6.nV} vertices, dt {dt6:.5g}, save {save6} -> frame "
          f"{save6*dt6:.4g} model-time  ({save6*dt6/(save5*dt5)-1:+.2%})")

    parcels, total = subparcels.region_set(c5, a.regions, a.split)
    lab5, tags = subparcels.split_parcels(c5, parcels, total, verbose=False)
    P5 = subparcels.taper_profiles(c5, lab5, len(tags))
    print(f"  {len(tags)} pieces, profile sum {float(P5.sum()):.3f}")

    # carry the drive to fs6 by nearest white-surface neighbour
    _, V5 = ladder._white_graph(c5)
    _, V6 = ladder._white_graph(c6)
    from scipy.spatial import cKDTree
    nn = cKDTree(V5).query(V6)[1]
    P6 = np.ascontiguousarray(P5[:, nn])
    print(f"  drive mapped to fs6: sum {float(P6.sum()):.1f} "
          f"({float(P6.sum())/float(P5.sum()):.2f}x, the vertex-density factor)")

    t0 = time.time()
    R5 = xspec.impulse_responses(c5, list(range(len(tags))), p, a.nframe * save5, save5,
                                 profiles=P5, verbose=False)
    print(f"  fs5 responses {R5.shape}  [{time.time()-t0:.0f}s]", flush=True)
    t0 = time.time()
    R6 = xspec.impulse_responses(c6, list(range(len(tags))), p, a.nframe * save6, save6,
                                 profiles=P6, verbose=True, workers=a.workers)
    print(f"  fs6 responses {R6.shape}  [{time.time()-t0:.0f}s]", flush=True)

    o5 = np.asarray(c5.old); o6 = np.asarray(c6.old)
    common = np.intersect1d(o5, o6)
    i5 = np.searchsorted(o5, common); i6 = np.searchsorted(o6, common)
    print(f"  comparing on {len(common)} shared vertices")

    A = np.asarray(R5[:, :, i5], np.float64)
    B = np.asarray(R6[:, :, i6], np.float64)
    amp = np.sqrt((B ** 2).mean()) / np.sqrt((A ** 2).mean())
    print(f"\n  amplitude: fs6 rms / fs5 rms = {amp:.3f} "
          f"(drive density factor {float(P6.sum())/float(P5.sum()):.2f})")

    def corr(u, v):
        u = u - u.mean(); v = v - v.mean()
        n = np.linalg.norm(u) * np.linalg.norm(v)
        return float(u @ v / n) if n > 0 else np.nan

    print(f"\n  per-frame spatial agreement, mean over {len(tags)} pieces")
    print(f"    {'frame':>8s}{'seconds':>9s}{'corr':>9s}{'fs5 rms':>11s}{'fs6/fs5':>9s}")
    for fr in [1, 2, 4, 8, 16, 32, 56, 84, 112, 168, 223]:
        if fr >= A.shape[1]:
            continue
        cs = [corr(A[k, fr], B[k, fr]) for k in range(len(tags))]
        r5 = np.sqrt((A[:, fr] ** 2).mean()); r6 = np.sqrt((B[:, fr] ** 2).mean())
        print(f"    {fr:8d}{fr*clock['frame_s']:9.2f}{np.mean(cs):+9.4f}"
              f"{r5:11.3e}{r6/max(r5,1e-30):9.3f}")
    allc = [corr(A[k].ravel(), B[k].ravel()) for k in range(len(tags))]
    print(f"  whole response, per piece: mean {np.mean(allc):+.4f}, "
          f"worst {np.min(allc):+.4f}, best {np.max(allc):+.4f}")

    # the band that matters: vertices 10-20 mm from the piece, at the arrival time
    from scipy.sparse.csgraph import dijkstra
    G5, _ = ladder._white_graph(c5)
    print(f"\n  agreement restricted by distance from the driven piece")
    print(f"    {'mm':>12s}{'corr':>9s}{'fs6/fs5 rms':>13s}")
    cen = [int(np.argmax(P5[k])) for k in range(len(tags))]
    D = dijkstra(G5, indices=[c for c in cen])[:, i5]
    ed = np.array([0, 5, 10, 20, 30, 40, 60, 90, 250])
    for lo, hi in zip(ed[:-1], ed[1:]):
        cs, rr = [], []
        for k in range(len(tags)):
            m = (D[k] >= lo) & (D[k] < hi)
            if m.sum() < 30:
                continue
            cs.append(corr(A[k][:, m].ravel(), B[k][:, m].ravel()))
            r5 = np.sqrt((A[k][:, m] ** 2).mean())
            rr.append(np.sqrt((B[k][:, m] ** 2).mean()) / max(r5, 1e-30))
        if cs:
            print(f"    {lo:5.0f} - {hi:<4.0f}{np.mean(cs):+9.4f}{np.mean(rr):13.3f}")

    # does the finer grid carry finer spatial structure?
    import units
    rng = np.random.default_rng(0)
    v = np.sort(rng.choice(len(common), 1200, replace=False))
    Dg = units.vertex_geodesic(c5, i5[v])[:, i5[v]]
    iu = np.triu_indices(len(v), 1)
    ts = np.unique(np.linspace(1, A.shape[1] - 1, 120).astype(int))
    print(f"\n  field spatial autocorrelation on each mesh")
    print(f"    {'mm':>12s}{'fs5':>9s}{'fs6':>9s}")
    halves = {}
    curves = {}
    for nm, X in (("fs5", A), ("fs6", B)):
        Y = X[:, ts][:, :, v].reshape(-1, len(v))
        Y = Y[Y.std(1) > 0]
        Y = Y - Y.mean(0, keepdims=True)
        Y = Y / np.maximum(Y.std(0, keepdims=True), 1e-30)
        C = (Y.T @ Y) / Y.shape[0]
        curves[nm] = C[iu]
    d = Dg[iu]
    eg = np.array([0, 2, 4, 6, 8, 10, 15, 20, 25, 30, 40, 60, 90, 250])
    prev = {}
    for lo, hi in zip(eg[:-1], eg[1:]):
        m = (d >= lo) & (d < hi)
        if m.sum() < 50:
            continue
        vals = {nm: float(curves[nm][m].mean()) for nm in curves}
        print(f"    {lo:5.0f} - {hi:<4.0f}{vals['fs5']:+9.3f}{vals['fs6']:+9.3f}")
        for nm in curves:
            if nm not in halves and nm in prev and prev[nm][1] >= 0.5 > vals[nm]:
                x0, y0 = prev[nm]; x1, y1 = 0.5 * (lo + hi), vals[nm]
                halves[nm] = x0 + (y0 - 0.5) * (x1 - x0) / (y0 - y1)
            prev[nm] = (0.5 * (lo + hi), vals[nm])
    for nm, hv in halves.items():
        print(f"    {nm}: r = 0.5 at {hv:.2f} mm")

    np.savez(os.path.join(RESULTS, "mesh_check.npz"),
             per_piece_corr=np.array(allc), amp=amp,
             save5=save5, save6=save6, dt5=dt5, dt6=dt6)
    print(f"\n  wrote results/mesh_check.npz")


if __name__ == "__main__":
    main()
