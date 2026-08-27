"""Solve for the drive that best makes this fluid trace out a target field sequence.

reach.py shows the target's eigenpatterns lie inside the fluid's reachable span, so any
remaining failure is temporal: one drive has to produce every frame at once, through a
propagator that keeps ringing after each injection. That is a linear least squares.

Field frames are linear in the drive, so frame t is sum_k sum_{tau<=t} a_k(tau) G_k(t-tau)
where G_k is region k's response to one block of injection. The design matrix's columns
are shifted copies of those responses, which makes its Gram block-Toeplitz: entry
((k,tau),(k',tau')) depends only on tau-tau', so the whole 5040 x 5040 system is built
from 18 x 18 cross-correlation sequences rather than from the matrix itself.

The fitted drive is then run through the real integrator and scored, which is the honest
number: the best this fluid can do on this objective from these regions, whatever input
family we might invent.

  python inverse.py --k 20
"""
import argparse
import numpy as np
from scipy.sparse.linalg import eigsh

from mesh_cache import load_cortex
from swe_rot import RotSWE, sponge_profile
from input2 import parcel_tapers, RegionDrive
from fc_score import FCTarget
from fc_moran import MoranMatch
from genome import LD_FIXED, SPONGE_STRENGTH_FIXED, SPONGE_WIDTH_FIXED
from run_ou import run, CFL, C, G, H
import ladder


def block_responses(cortex, regions, nsteps, save_every, sponge=True, fp=None,
                    verbose=True):
    """Response to one block of injection (save_every steps at unit rate) per region.

    With `fp` (a fluid.decode dict) the responses are taken at that operating point
    instead of the frozen one, so the ceiling can be recomputed wherever the search has
    moved the medium to."""
    if fp is not None:
        import fluid as fl
        s, dt, gD, HD = fl.build(cortex, fp, sponge)
    else:
        dt = CFL * cortex.d.min() / C
        s = RotSWE(cortex.m, C / LD_FIXED, l=cortex.l, d=cortex.d, A=cortex.A,
                   E=cortex.edges, bnd_edge=cortex.bnd)
        if sponge:
            s.set_sponge(sponge_profile(cortex.V, cortex.edges, cortex.bnd,
                                        SPONGE_WIDTH_FIXED, SPONGE_STRENGTH_FIXED))
        s.astype(np.float32)
        for at in ("sig_v", "sig_e"):
            v = getattr(s, at, None)
            if v is not None:
                setattr(s, at, v.astype(np.float32))
        gD, HD = np.float32(G), np.float32(H)
    T, ids = parcel_tapers(cortex, verbose=False)
    pos = {int(p): i for i, p in enumerate(ids)}
    dtD = np.float32(dt)

    out = []
    for p in regions:
        prof = (T[pos[int(p)]] / save_every).astype(np.float32)
        h = np.zeros(cortex.nV, np.float32); ue = np.zeros(s.nE, np.float32)
        frames = []
        for n in range(nsteps):
            if n < save_every:
                h += prof
            ue, h = s.step(ue, h, dtD, gD, HD)
            if n % save_every == 0:
                frames.append(h.copy())
        out.append(np.asarray(frames))
        if verbose:
            print(f"  region {p:3d} block response built", flush=True)
    return np.asarray(out)                       # (K, nF, nV)


def solve_drive(Gk, F, ridge=1e-3, verbose=True):
    """Least-squares drive a (K, nF) whose convolution with Gk best matches F (nF, nV)."""
    K, nF, nV = Gk.shape
    Gf = Gk.astype(np.float64)

    # cross-correlations C[k,k',d] = sum_t G_k(t) . G_k'(t+d), d >= 0
    Cc = np.zeros((K, K, nF))
    for d in range(nF):
        Cc[:, :, d] = Gf[:, :nF - d].reshape(K, -1) @ \
            Gf[:, d:].reshape(K, -1).T / 1.0 if d == 0 else \
            np.einsum('ktv,ltv->kl', Gf[:, :nF - d], Gf[:, d:])
    Cc[:, :, 0] = np.einsum('ktv,ltv->kl', Gf, Gf)

    M = np.zeros((K * nF, K * nF))
    for a in range(nF):
        for b in range(nF):
            d = b - a
            M[a * K:(a + 1) * K, b * K:(b + 1) * K] = (Cc[:, :, d] if d >= 0
                                                       else Cc[:, :, -d].T)
    rhs = np.zeros(K * nF)
    Ff = F.astype(np.float64)
    for a in range(nF):
        rhs[a * K:(a + 1) * K] = np.einsum('ktv,tv->k', Gf[:, :nF - a], Ff[a:])

    M[np.diag_indices_from(M)] += ridge * np.trace(M) / len(M)
    x = np.linalg.solve(M, rhs)
    a = x.reshape(nF, K).T
    fit = np.zeros_like(Ff)
    for t in range(nF):
        for tau in range(t + 1):
            fit[t] += a[:, tau] @ Gf[:, t - tau]
    r2 = float(1.0 - ((Ff - fit) ** 2).sum() / (Ff ** 2).sum())
    if verbose:
        print(f"  drive solved: fit to the target sequence R2 = {r2:.3f}")
    return a, r2


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--nsteps", type=int, default=7000)
    ap.add_argument("--save-every", type=int, default=25, dest="save_every")
    ap.add_argument("--tau", type=float, default=30.0, help="target timecourse timescale")
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--fluid-pkl", default=None, dest="fluid_pkl",
                    help="search pickle whose winning fluid parameters to use")
    a = ap.parse_args()

    cortex = load_cortex("fsaverage5", verbose=False)
    target = FCTarget(cortex, verbose=False)
    mm = MoranMatch(cortex, target)
    regions = ladder.CORE_POS + ladder.CORE_NEG + ladder.SATELLITES
    nF = a.nsteps // a.save_every

    fp = None
    if a.fluid_pkl:
        import pickle, fluid as fl
        d = pickle.load(open(a.fluid_pkl, "rb"))
        fp = fl.decode(d["best"][1][-fl.N_PARAM:])
        print(f"  fluid from {a.fluid_pkl.split('/')[-1]}: c0 {fp['c0']:.2f}, "
              f"Ld {fp['Ld']:.1f}, sig0 {fp['sig0']:.1e}, "
              f"speed x{np.round(fp['c_group'],2)}, damping x{np.round(fp['sig_group'],2)}")

    # target sequence: the FC's leading eigenpatterns with independent smooth timecourses
    FC = np.asarray(target.target_fc(), np.float64)
    ev, E = eigsh(FC, k=a.k, which="LA")
    o = np.argsort(ev)[::-1]; ev, E = ev[o], E[:, o]
    rng = np.random.default_rng(0)
    ts = rng.standard_normal((nF, a.k))
    alpha = np.exp(-1.0 / max(a.tau, 1e-9))
    for i in range(1, nF):
        ts[i] = alpha * ts[i - 1] + np.sqrt(1 - alpha ** 2) * ts[i]
    Fseq = (ts * np.sqrt(ev)) @ E.T                     # (nF, nV_aligned)
    full = np.zeros((nF, cortex.nV))
    full[:, target.cols] = Fseq
    print(f"  target sequence: {nF} frames of the top {a.k} eigenpatterns, tau {a.tau}")

    Gk = block_responses(cortex, regions, a.nsteps, a.save_every, fp=fp)
    drive_a, r2 = solve_drive(Gk, full, ridge=a.ridge)

    # run the real integrator with the fitted drive, then score it
    Aser = np.repeat(drive_a.T, a.save_every, axis=0)[:a.nsteps]
    d = RegionDrive(cortex, regions, Aser, amp=2e-4, nsteps=a.nsteps,
                    tapers=parcel_tapers(cortex, verbose=False))
    if fp is not None:
        import fluid as fl
        frames, _ = fl.run(cortex, d, fp, a.nsteps, a.save_every, sponge=True)
    else:
        frames, _ = run(cortex, d, a.nsteps, a.save_every, sponge=True, verbose=False)
    Z, _ = target.model_z(frames)
    sim = float(target._prep(target.model_edges(Z=Z)[0]) @ target.y)
    print(f"\n  best-possible drive, run through the model: sim {sim:+.4f}  "
          f"moran gap {mm.gap(Z):.4f}")
    print(f"  drive: {drive_a.shape[0]} regions x {drive_a.shape[1]} blocks, "
          f"rms {np.sqrt((drive_a**2).mean()):.3e}, "
          f"per-region rms spread {np.sqrt((drive_a**2).mean(1)).std()/np.sqrt((drive_a**2).mean()):.2f}")
    np.save("results/inverse_drive.npy", drive_a)
    np.save("results/frames_inverse.npy", frames)
    print("  wrote results/inverse_drive.npy, results/frames_inverse.npy")


if __name__ == "__main__":
    main()
