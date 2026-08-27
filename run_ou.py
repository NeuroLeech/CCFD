"""Run the frozen-regime model driven by sparse latent OU input, score it, draw it.

The fluid is fixed at the regime genome.py freezes (Ld 52.4, i.e. f = c/52.4), and the
regions are fixed too. The only thing this varies is the latent structure: r OU latents,
each with its own timescale and silent fraction, and a K x r loading matrix saying how
they mix onto the driven regions. That is the object a search would move.

  python run_ou.py                       # seed 0
  python run_ou.py --seed 3 --r 4 --nsteps 12000
  python run_ou.py --seed 0 --sponge     # absorbing medial wall instead of reflecting

Scored against the double-centred NKI group FC by default, since maximising that score
is the point; --fc pins a different target.
"""
import os, time, argparse
import numpy as np

from mesh_cache import load_cortex
from input_model import NetworkDrive, parcel_tapers
from swe_rot import RotSWE, sponge_profile
from genome import (LD_FIXED, SPONGE_STRENGTH_FIXED, SPONGE_WIDTH_FIXED, AMP_FIXED,
                    SILENT_LIM, TAU_LIM, LOAD_LIM)
from fc_score import FCTarget
from paths import RESULTS

REGIONS = [9, 1, 24, 150, 30, 65, 132]      # the set run_v3.py drives
CFL, C, G, H = 0.347, 1.0, 1.0, 1.0
NSTEPS, SAVE_EVERY, BURN = 7000, 25, 50


def latent_params(rng, K, r):
    """Loadings, silent fractions and timescales, drawn in the ranges genome.py uses."""
    return (rng.uniform(-LOAD_LIM, LOAD_LIM, (K, r)),
            rng.uniform(*SILENT_LIM, r),
            rng.uniform(*TAU_LIM, r))


def run(cortex, drive, nsteps=NSTEPS, save_every=SAVE_EVERY, sponge=False, verbose=True):
    """-> (frames (nsaved, nV), dt). Diverged runs raise rather than return garbage."""
    dt = CFL * cortex.d.min() / C
    s = RotSWE(cortex.m, C / LD_FIXED, l=cortex.l, d=cortex.d, A=cortex.A,
               E=cortex.edges, bnd_edge=cortex.bnd)
    if sponge:
        s.set_sponge(sponge_profile(cortex.V, cortex.edges, cortex.bnd,
                                    SPONGE_WIDTH_FIXED, SPONGE_STRENGTH_FIXED))
    s.astype(np.float32)
    for attr in ("sig_v", "sig_e"):               # None unless a sponge was set
        v = getattr(s, attr, None)
        if v is not None:
            setattr(s, attr, v.astype(np.float32))

    Aser = drive.Aser.astype(np.float32)
    P = drive.P.astype(np.float32)
    h = np.zeros(cortex.nV, np.float32)
    ue = np.zeros(s.nE, np.float32)
    dtD, gD, HD = np.float32(dt), np.float32(G), np.float32(H)

    frames, t0 = [], time.time()
    for n in range(nsteps):
        h += Aser[n] @ P
        ue, h = s.step(ue, h, dtD, gD, HD)
        if n % save_every == 0:
            if not np.isfinite(h).all():
                raise FloatingPointError(f"diverged at step {n}")
            frames.append(h.copy())
    frames = np.asarray(frames)
    if verbose:
        print(f"  {len(frames)} frames in {time.time()-t0:.0f}s, "
              f"dt {dt:.4f}, {nsteps*dt:.0f} time units, "
              f"peak {100*np.abs(frames).max()/H:.2f}% of depth")
    return frames, dt


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--space", default="fsaverage5")
    ap.add_argument("--seed", type=int, default=0, help="draws the latent structure")
    ap.add_argument("--r", type=int, default=3, help="number of latents")
    ap.add_argument("--regions", nargs="*", type=int, default=REGIONS)
    ap.add_argument("--nsteps", type=int, default=NSTEPS)
    ap.add_argument("--save-every", type=int, default=SAVE_EVERY, dest="save_every")
    ap.add_argument("--sponge", action="store_true", help="absorbing medial wall")
    ap.add_argument("--balance", default="spatial", choices=("spatial", "temporal"))
    ap.add_argument("--fc", default=None, help="FC target (default: newest for the space)")
    ap.add_argument("--centre", default="none", choices=("none", "double"),
                    help="'double' if the target file is not already double-centred")
    ap.add_argument("--seeds-plot", nargs="*", type=int, default=[1, 24, 150],
                    dest="seeds_plot")
    ap.add_argument("--tag", default=None, help="output name stem")
    ap.add_argument("--no-plot", action="store_true")
    a = ap.parse_args()

    cortex = load_cortex(a.space, verbose=False)
    dt = CFL * cortex.d.min() / C
    rng = np.random.default_rng(a.seed)
    K = len(a.regions)
    L, silent, tau = latent_params(rng, K, a.r)

    print(f"{a.space}: {cortex.nV} vertices, Ld {LD_FIXED:.1f} (f = c/{LD_FIXED:.1f}), "
          f"sponge {'on' if a.sponge else 'off'}, dt {dt:.4f}")
    drive = NetworkDrive(cortex, a.regions, L, silent, tau, AMP_FIXED, a.nsteps, dt,
                         seed=a.seed, tapers=parcel_tapers(cortex, verbose=False),
                         balance=a.balance)
    print(f"  latent seed {a.seed}: taus " + ", ".join(f"{t:.1f}" for t in tau) +
          "  silent " + ", ".join(f"{s*100:.0f}%" for s in silent))
    drive.describe()

    frames, _ = run(cortex, drive, a.nsteps, a.save_every, a.sponge)

    tag = a.tag or f"ou_seed{a.seed}_r{a.r}"
    fpath = os.path.join(RESULTS, f"frames_{tag}.npy")
    np.save(fpath, frames)
    print(f"  wrote {fpath}  {frames.shape}")

    target = FCTarget(cortex, fc_path=a.fc, burn=BURN, centre=a.centre)
    r, info = target.score(frames, report=True)
    print(f"  FC score {r:+.4f}   ({info['frames_used']} frames used, "
          f"{info['flat_vertices']} flat vertices)")

    if not a.no_plot:
        from plot_fc_map import figure
        figure(cortex, target, frames, a.seeds_plot,
               os.path.join(RESULTS, f"fc_maps_{tag}.png"))


if __name__ == "__main__":
    main()
