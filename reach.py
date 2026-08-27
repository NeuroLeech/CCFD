"""What spatial patterns can this fluid produce at all, from these regions?

The model is linear, so the set of fields reachable from K driven regions is the span of
their impulse responses over time - the Krylov subspace of the propagator applied to the
region profiles. Any drive whatsoever produces a field inside that span, so if the
target's leading FC eigenpatterns lie outside it, no input search can succeed and the
limit is the fluid or the region set rather than the input parameterisation.

This asks the question directly: drive each region with a single impulse, collect every
saved frame of every response as a basis, and measure how much of each target
eigenvector that basis can reproduce.

  python reach.py --k 20
"""
import argparse
import numpy as np
from scipy.sparse.linalg import eigsh

from mesh_cache import load_cortex
from swe_rot import RotSWE, sponge_profile
from input2 import parcel_tapers
from fc_score import FCTarget
from genome import LD_FIXED, SPONGE_STRENGTH_FIXED, SPONGE_WIDTH_FIXED
from run_ou import CFL, C, G, H
import ladder


def impulse_responses(cortex, regions, nsteps=7000, save_every=25, sponge=True,
                      verbose=True):
    """-> (K, nsaved, nV). One unit impulse into each region, then free evolution."""
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
    T, ids = parcel_tapers(cortex, verbose=False)
    pos = {int(p): i for i, p in enumerate(ids)}
    dtD, gD, HD = np.float32(dt), np.float32(G), np.float32(H)

    out = []
    for k, p in enumerate(regions):
        h = T[pos[int(p)]].astype(np.float32).copy()
        ue = np.zeros(s.nE, np.float32)
        frames = [h.copy()]
        for n in range(1, nsteps):
            ue, h = s.step(ue, h, dtD, gD, HD)
            if n % save_every == 0:
                frames.append(h.copy())
        out.append(np.asarray(frames))
        if verbose:
            print(f"  region {p:3d}: {len(frames)} frames, "
                  f"final/initial amplitude {np.abs(frames[-1]).max()/np.abs(frames[0]).max():.3f}",
                  flush=True)
    return np.asarray(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--k", type=int, default=20, help="target eigenvectors to test")
    ap.add_argument("--nsteps", type=int, default=7000)
    ap.add_argument("--save-every", type=int, default=25, dest="save_every")
    a = ap.parse_args()

    cortex = load_cortex("fsaverage5", verbose=False)
    target = FCTarget(cortex, verbose=True)
    regions = ladder.CORE_POS + ladder.CORE_NEG + ladder.SATELLITES
    print(f"  {len(regions)} driven regions, impulse responses over {a.nsteps} steps")

    R = impulse_responses(cortex, regions, a.nsteps, a.save_every)
    B = R.reshape(-1, R.shape[-1])[:, target.cols].astype(np.float32)   # (K*T, nV)
    B -= B.mean(1, keepdims=True)
    print(f"  reachable basis: {B.shape[0]} response frames over {B.shape[1]} vertices")

    Gm = (B @ B.T).astype(np.float64)
    ev = np.linalg.eigvalsh(Gm)
    eff = float(ev.sum() ** 2 / (ev ** 2).sum())
    tolr = ev.max() * 1e-10
    print(f"  numerical rank {int((ev > tolr).sum())}, effective dimension {eff:.1f}")

    FC = np.asarray(target.target_fc(), np.float64)
    evals, E = eigsh(FC, k=a.k, which="LA")
    order = np.argsort(evals)[::-1]
    evals, E = evals[order], E[:, order]

    Ginv = np.linalg.pinv(Gm, rcond=1e-10)
    print(f"\n  how much of each target eigenpattern the fluid can build:")
    r2s = []
    for j in range(a.k):
        v = E[:, j].astype(np.float32); v = v - v.mean()
        c = Ginv @ (B @ v)
        fit = B.T @ c.astype(np.float32)
        r2 = float(1.0 - ((v - fit) ** 2).sum() / (v ** 2).sum())
        r2s.append(r2)
        if j < 8 or j == a.k - 1:
            print(f"    eigenvector {j:2d} (eigenvalue {evals[j]:8.1f}):  R2 = {r2:.3f}")
    print(f"  mean R2 over the top {a.k}: {np.mean(r2s):.3f}   "
          f"(top 5: {np.mean(r2s[:5]):.3f})")


if __name__ == "__main__":
    main()
