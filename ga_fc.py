"""GA over the latent structure alone, maximising FC similarity to the empirical matrix.

Fixed: the fluid (Ld 52.4, sponge as given), the driven regions, the amplitude, the run
length. Searched: the K x r loading matrix and each latent's timescale and silent
fraction - nothing else. A genome is therefore just a point in [0,1]^(K*r + 2r), and the
fitness is fc_score.FCTarget.score of the run it produces.

The OU realisation is redrawn every generation (drive seed = generation), and elites are
re-scored under the new draw rather than carrying their old fitness forward. Without that
the GA converges on the latent noise it happened to be handed rather than on structure
that survives a fresh draw; the gap between an elite's old and new score is recorded as
`reliability` and is the first thing to read if the search looks like it is working.

  python ga_fc.py --pop 24 --gens 15 --workers 8
  python ga_fc.py --no-sponge --gens 30
"""
import os, sys, time, pickle, argparse
import multiprocessing as mp
import numpy as np

from genome import LOAD_LIM, SILENT_LIM, TAU_LIM, AMP_FIXED
from paths import RESULTS

REGIONS = [9, 1, 24, 150, 30, 65, 132]
R_LATENT = 3
NSTEPS, SAVE_EVERY = 7000, 25
ELITE, TOURN = 2, 2
OUT = os.path.join(RESULTS, "ga_fc")

_W = {}


# ------------------------------------------------------------------ genome
GAIN_LIM = (0.3, 1.0)      # per-latent gain, as a ratio; see decode


def n_genes(K, r, gains=False):
    return K * r + 2 * r + (r if gains else 0)


def decode(x, K, r, tau_lim=TAU_LIM, silent_lim=SILENT_LIM, gain_lim=None):
    """[0,1]^n -> loadings, silent fractions, timescales, one of each per latent.

    With `gain_lim` set, r extra genes carry a per-latent gain and each loading column
    is unit-normalised before being scaled by it. Column norm already carried relative
    latent amplitude implicitly, but unbounded (a column can collapse to zero, killing a
    latent) and degenerate with NetworkDrive's overall RMS normalisation, which rescales
    everything anyway. Separating direction from gain gives CMA a coordinate it can move,
    and gain_lim keeps the ratio between latents inside a stated range."""
    x = np.clip(np.asarray(x, float), 0.0, 1.0)
    i = 0
    L = (x[i:i + K * r].reshape(K, r) * 2.0 - 1.0) * LOAD_LIM; i += K * r
    silent = silent_lim[0] + x[i:i + r] * (silent_lim[1] - silent_lim[0]); i += r
    tau = tau_lim[0] + x[i:i + r] * (tau_lim[1] - tau_lim[0]); i += r
    if gain_lim is not None:
        g = gain_lim[0] + x[i:i + r] * (gain_lim[1] - gain_lim[0]); i += r
        nrm = np.linalg.norm(L, axis=0)
        nrm[nrm < 1e-12] = 1.0
        L = (L / nrm) * g
    assert i == len(x)
    return L, silent, tau


def mutate(x, rng, sigma=0.15):
    return np.clip(x + rng.normal(0.0, sigma, x.shape), 0.0, 1.0)


def crossover(a, b, rng):
    m = rng.random(a.shape) < 0.5
    return np.where(m, a, b)


# ------------------------------------------------------------------ evaluation
def _init(space, regions, r, nsteps, save_every, sponge, fc_path, centre,
          tau_lim=TAU_LIM, silent_lim=SILENT_LIM, gain_lim=None, moran_lambda=0.0,
          moran_tol=0.0, generator="ou", rung=4, fluid=False, fluid_mode="group",
          ladder_groups=None, rank_min=10.0, rank_lambda=0.0):
    from mesh_cache import load_cortex
    from input_model import parcel_tapers
    from fc_score import FCTarget
    c = load_cortex(space, verbose=False)
    _W.update(cortex=c, tapers=parcel_tapers(c, verbose=False), regions=regions, r=r,
              nsteps=nsteps, save_every=save_every, sponge=sponge,
              tau_lim=tau_lim, silent_lim=silent_lim, gain_lim=gain_lim,
              target=FCTarget(c, fc_path=fc_path, centre=centre, verbose=False))
    _W.update(generator=generator, rung=rung, fluid=fluid, fluid_mode=fluid_mode)
    if generator == "ladder":
        import ladder
        regs = (ladder.set_groups(*ladder_groups) if ladder_groups
                else ladder.CORE_POS + ladder.CORE_NEG + ladder.SATELLITES)
        D, _ = ladder.parcel_geodesic(c, regs, verbose=False)
        from input2 import parcel_tapers as region_tapers
        _W.update(ladder_regions=regs, ladder_D=D,
                  ladder_tapers=region_tapers(c, verbose=False))
    if moran_lambda:
        from fc_moran import MoranMatch
        _W["target"].attach_moran(MoranMatch(c, _W["target"]), moran_lambda, moran_tol)
    if rank_lambda:
        _W["target"].attach_rank(rank_min, rank_lambda)


def _evaluate(args):
    x, seed = args
    from input_model import NetworkDrive
    from run_ou import run, CFL, C
    c, t = _W["cortex"], _W["target"]
    K, r = len(_W["regions"]), _W["r"]
    dt = CFL * c.d.min() / C
    tau = silent = None
    try:
        fp = None
        if _W.get("fluid"):
            import fluid as fl
            nf = fl.n_param(_W["fluid_mode"])
            x, xf = x[:-nf], x[-nf:]
            fp = fl.decode_mode(xf, _W["fluid_mode"])
            cfield, _ = fl.fields(c, fp)
            dt = fl.CFL * c.d.min() / float(cfield.max())   # CFL at the fastest point
        if _W.get("generator") == "ladder":
            import ladder
            drive = ladder.make_drive(c, x, _W["nsteps"], dt, amp=AMP_FIXED,
                                      rung=_W["rung"], seed=seed,
                                      tapers=_W["ladder_tapers"],
                                      regions=_W["ladder_regions"], D=_W["ladder_D"])
        else:
            L, silent, tau = decode(x, K, r, _W["tau_lim"], _W["silent_lim"],
                                    _W["gain_lim"])
            drive = NetworkDrive(c, _W["regions"], L, silent, tau, AMP_FIXED,
                                 _W["nsteps"], dt, seed=seed, tapers=_W["tapers"],
                                 balance="spatial")
        if drive.dead:
            return dict(fitness=-1e9, ok=False, reason="dead drive")
        if fp is not None:
            import fluid as fl
            frames, _ = fl.run(c, drive, fp, _W["nsteps"], _W["save_every"], _W["sponge"])
        else:
            frames, _ = run(c, drive, _W["nsteps"], _W["save_every"], _W["sponge"],
                            verbose=False)
        fit, info = t.score(frames, report=True)
    except (FloatingPointError, ValueError, np.linalg.LinAlgError) as e:
        return dict(fitness=-1e9, ok=False, reason=str(e))
    return dict(fitness=float(fit), ok=True, flat=info["flat_vertices"],
                similarity=info["similarity"], moran_gap=info["moran_gap"],
                rank=info.get("rank", np.nan),
                peak=float(np.abs(frames).max()),
                taus=None if tau is None else tau.tolist(),
                silent=None if silent is None else silent.tolist())


# ------------------------------------------------------------------ the search
def run_ga(pop=24, gens=15, workers=8, space="fsaverage5", regions=REGIONS, r=R_LATENT,
           nsteps=NSTEPS, save_every=SAVE_EVERY, sponge=True, fc_path=None,
           centre="none", seed0=0, verbose=True):
    K = len(regions)
    rng = np.random.default_rng(seed0)
    X = rng.random((pop, n_genes(K, r)))
    history, best = [], None
    os.makedirs(OUT, exist_ok=True)

    ctx = mp.get_context("spawn")
    with ctx.Pool(workers, initializer=_init,
                  initargs=(space, regions, r, nsteps, save_every, sponge, fc_path,
                            centre)) as pool:
        prev_fit = None
        for gen in range(gens):
            t0 = time.time()
            res = pool.map(_evaluate, [(x, gen) for x in X])   # new OU draw each gen
            fit = np.array([q["fitness"] for q in res])
            order = np.argsort(fit)[::-1]

            # elites carried in from last generation were re-scored under this draw
            rel = np.nan
            if prev_fit is not None:
                rel = float(np.corrcoef(prev_fit[:ELITE], fit[:ELITE])[0, 1]) \
                    if ELITE > 2 else float(np.mean(fit[:ELITE] - prev_fit[:ELITE]))
            ok = np.array([q["ok"] for q in res])
            if verbose:
                print(f"  gen {gen:3d}  best {fit[order[0]]:+.4f}  "
                      f"mean {fit[ok].mean():+.4f}  ok {ok.sum():2d}/{pop}  "
                      f"elite re-score shift {rel:+.4f}  [{time.time()-t0:.0f}s]",
                      flush=True)
            history.append(dict(gen=gen, fitness=fit.copy(), X=X.copy(),
                                res=res, elite_shift=rel))
            if best is None or fit[order[0]] > best[0]:
                best = (float(fit[order[0]]), X[order[0]].copy(), gen)

            # elitism + tournament selection
            newX = [X[i].copy() for i in order[:ELITE]]
            while len(newX) < pop:
                pick = lambda: order[min(rng.integers(0, pop, TOURN))]
                child = crossover(X[pick()], X[pick()], rng)
                newX.append(mutate(child, rng))
            X = np.array(newX)
            prev_fit = fit[order][:ELITE].copy()

    path = os.path.join(OUT, f"ga_fc_pop{pop}_gen{gens}_seed{seed0}.pkl")
    with open(path, "wb") as fh:
        pickle.dump(dict(history=history, best=best, regions=regions, r=r, K=K,
                         nsteps=nsteps, sponge=sponge, fc_path=fc_path,
                         centre=centre, pop=pop, gens=gens, seed0=seed0), fh)
    if verbose:
        print(f"\n  best {best[0]:+.4f} at generation {best[2]}")
        print(f"  wrote {path}")
    return history, best, path


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pop", type=int, default=24)
    ap.add_argument("--gens", type=int, default=15)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--space", default="fsaverage5")
    ap.add_argument("--r", type=int, default=R_LATENT)
    ap.add_argument("--regions", nargs="*", type=int, default=REGIONS)
    ap.add_argument("--nsteps", type=int, default=NSTEPS)
    ap.add_argument("--no-sponge", action="store_true")
    ap.add_argument("--fc", default=None)
    ap.add_argument("--centre", default="none", choices=("none", "double"))
    ap.add_argument("--seed", type=int, default=0, dest="seed0")
    a = ap.parse_args()
    print(f"GA over the latent only: K={len(a.regions)} regions, r={a.r} latents, "
          f"{n_genes(len(a.regions), a.r)} genes, pop {a.pop} x {a.gens} gens, "
          f"sponge {'off' if a.no_sponge else 'on'}, {a.workers} workers")
    run_ga(pop=a.pop, gens=a.gens, workers=a.workers, space=a.space, regions=a.regions,
           r=a.r, nsteps=a.nsteps, sponge=not a.no_sponge, fc_path=a.fc,
           centre=a.centre, seed0=a.seed0)


if __name__ == "__main__":
    main()
