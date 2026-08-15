"""Run a spread of dynamical regimes with the input held fixed, and keep the full
vertex x time field.

The point is NOT to score these runs. It is to have a set of spatiotemporal
patterns that are known to differ, so that candidate measures can be tried on them
and judged by whether they actually separate the runs - practically, not
theoretically. Nothing here selects, ranks, or optimises.

Two input conditions, fixed across every run:

    A   10r + 10v  (parcels 65, 88)   anterior, small, adjacent
    B   V1         (parcel 1)         posterior, primary sensory, eroded to
                                      approximately the size of the anterior pair

Only NON-input parameters vary between runs: the fluid regime (Ld, sponge) and
the drive's temporal statistics (tau, sparsity). Region identity, loadings and
amplitude are held constant, so any difference between runs is the regime.

Mass balance is TEMPORAL here, not spatial - see NetworkDrive. A single region
cannot be driven at all under the spatial constraint, and using different balance
modes for the two conditions would confound region identity with balance mode.
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
from mesh_cache import load_cortex
from input_model import parcel_tapers, NetworkDrive, _erode_once
from swe_rot import RotSWE

from paths import FIELDS as OUT
NSTEPS = 7000
SAVE_EVERY = 25            # -> 280 frames, ~13 per wave crossing
AMP = 2.0e-4
C, G, H, CFL = 1.0, 1.0, 1.0, 0.35

CONDITIONS = {"A_10r10v": [65, 88], "B_V1": [1]}


def eroded_taper(c, pid, target):
    """Taper on the deep core of a parcel, shrunk to about `target` vertices.

    V1 at fsaverage5 is 259 vertices against 28 and 44 for 10r and 10v, so
    comparing them directly would confound region identity with region size.
    Erosion peels from the boundary inward, so the core that survives is the part
    of the parcel furthest from any other area."""
    nb = np.zeros(c.nV)
    np.add.at(nb, c.edges[:, 0], 1)
    np.add.at(nb, c.edges[:, 1], 1)
    mask = (c.lab == pid)
    depth = np.zeros(c.nV)
    cur = mask.copy()
    k = 0
    while cur.any():
        k += 1
        depth[cur] = k
        cur = _erode_once(cur, c.edges, nb, c.nV)
    # deepest level whose core is still at least `target` vertices
    lvl = max([j for j in range(1, k+1) if (depth >= j).sum() >= target] or [1])
    core = depth >= lvl
    u = np.zeros(c.nV)
    u[core] = (depth[core] - lvl + 1)/max(k - lvl + 1, 1)
    return (u*u*(3.0 - 2.0*u)).astype(np.float32), int(core.sum()), lvl


def sample_params(n, seed=0):
    """Random non-input parameters. Log-uniform where the scale is multiplicative."""
    rng = np.random.default_rng(seed)
    return [dict(
        Ld=float(10**rng.uniform(np.log10(5.0), np.log10(200.0))),
        sponge_strength=float(rng.uniform(0.0, 1.0)),
        sponge_width=float(rng.uniform(5.0, 60.0)),
        tau=float(10**rng.uniform(np.log10(20.0), np.log10(400.0))),
        silent=float(rng.uniform(0.50, 0.92)),
    ) for _ in range(n)]


def main(n_param=10, seed=0):
    os.makedirs(OUT, exist_ok=True)
    c = load_cortex("fsaverage5")
    T, ids = parcel_tapers(c, verbose=False)
    pos = {int(p): i for i, p in enumerate(ids)}

    # V1 eroded to the size of the anterior pair
    n_ant = int((c.lab == 65).sum() + (c.lab == 88).sum())
    v1_taper, v1_n, v1_lvl = eroded_taper(c, 1, n_ant)
    T = T.copy()
    T[pos[1]] = v1_taper
    print(f"  10r {int((c.lab==65).sum())} + 10v {int((c.lab==88).sum())} = {n_ant} vertices")
    print(f"  V1 {int((c.lab==1).sum())} -> eroded to {v1_n} vertices (depth >= {v1_lvl})")

    # geodesic distance to the medial wall, for the sponge; computed once
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra
    we = np.linalg.norm(c.V[c.edges[:, 0]] - c.V[c.edges[:, 1]], axis=1)
    Gm = csr_matrix((np.r_[we, we], (np.r_[c.edges[:, 0], c.edges[:, 1]],
                                     np.r_[c.edges[:, 1], c.edges[:, 0]])),
                    shape=(c.nV, c.nV))
    dist = dijkstra(Gm, indices=np.unique(c.edges[c.bnd].ravel()),
                    directed=False).min(axis=0)

    s = RotSWE(c.m, C/25.0, l=c.l, d=c.d, A=c.A, E=c.edges, bnd_edge=c.bnd)
    s.astype(np.float32)
    dt = float(CFL*c.d.min()/C)
    params = sample_params(n_param, seed)
    manifest = []

    print(f"\n  dt {dt:.4f}, {NSTEPS} steps = {NSTEPS*dt:.0f} time units, "
          f"saving every {SAVE_EVERY} -> {NSTEPS//SAVE_EVERY} frames")
    print(f"  {len(params)} parameter sets x {len(CONDITIONS)} input conditions\n")

    for pi, p in enumerate(params):
        s.f[:] = np.float32(C/p["Ld"])
        u = np.clip(1.0 - dist/max(p["sponge_width"], 1e-9), 0.0, 1.0)
        s.set_sponge(p["sponge_strength"]*u*u*(3.0 - 2.0*u))
        s.sig_v = s.sig_v.astype(np.float32)
        s.sig_e = s.sig_e.astype(np.float32)

        for cond, regs in CONDITIONS.items():
            t0 = time.time()
            drive = NetworkDrive(c, regs, np.ones((len(regs), 1)), [p["silent"]],
                                 [p["tau"]], AMP, NSTEPS, dt, seed=0,
                                 tapers=(T, ids), balance="temporal")
            Aser = drive.Aser.astype(np.float32)
            Pmat = drive.P.astype(np.float32)
            h = np.zeros(c.nV, np.float32)
            ue = np.zeros(s.nE, np.float32)
            dtD, gD, HD = np.float32(dt), np.float32(G), np.float32(H)
            frames = []
            diverged = False
            for n in range(NSTEPS):
                h += Aser[n] @ Pmat
                ue, h = s.step(ue, h, dtD, gD, HD)
                if n % SAVE_EVERY == 0:
                    if not np.isfinite(h).all():
                        diverged = True
                        break
                    frames.append(h.copy())
            Hs = np.array(frames, np.float32)
            name = f"p{pi:02d}_{cond}"
            np.savez(os.path.join(OUT, name + ".npz"), H=Hs,
                     drive=drive.Aser.astype(np.float32)[::SAVE_EVERY],
                     dt_frame=dt*SAVE_EVERY, regions=np.array(regs))
            rec = dict(name=name, cond=cond, regions=regs, pi=pi,
                       diverged=bool(diverged), nframes=int(len(Hs)),
                       dt_frame=float(dt*SAVE_EVERY),
                       peak_frac=float(np.abs(Hs).max()/H) if len(Hs) else float("nan"),
                       **p)
            manifest.append(rec)
            print(f"  {name:16s} Ld {p['Ld']:6.1f} spStr {p['sponge_strength']:.2f} "
                  f"spW {p['sponge_width']:4.1f} tau {p['tau']:6.1f} "
                  f"sil {p['silent']:.2f} | {len(Hs):3d} frames "
                  f"peak {100*rec['peak_frac']:7.3f}% "
                  f"{'DIVERGED ' if diverged else ''}[{time.time()-t0:.1f}s]",
                  flush=True)

    with open(os.path.join(OUT, "manifest.json"), "w") as fjs:
        json.dump(manifest, fjs, indent=1)
    tot = sum(os.path.getsize(os.path.join(OUT, m["name"]+".npz")) for m in manifest)
    print(f"\n  wrote {len(manifest)} fields to {OUT}  ({tot/1e6:.0f} MB)")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 10)
