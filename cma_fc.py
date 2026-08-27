"""CMA-ES over the latent structure, maximising FC similarity. Three changes from ga_fc.

1. Wider bounds. The GA's one detectable gradient was towards longer latent timescales,
   and it was pressed against TAU_LIM's ceiling of 30 with 28% of top-decile latents
   sitting on the cap. Silence is widened too, all the way to zero: a continuous
   anticorrelated sine drive reaches ~0.3, and genome.py's floor of 0.75 put that whole
   regime out of reach.

2. Every candidate is scored on the mean of N_DRAWS OU realisations. One draw carries
   sd ~0.014 against a between-genome sd of ~0.029, which is enough noise for a
   selection-on-luck bias of ~0.03 on the best of 32 - the GA's apparent gains were
   mostly that. Three draws cuts it to ~0.008.

3. CMA-ES instead of the GA. 27 continuous dimensions with a fixed mutation sigma of
   0.15 never let the GA's population concentrate (gene sd 0.282 -> 0.277 over 30
   generations, versus 0.289 for a uniform population); CMA adapts the step size and
   the covariance instead.

Draw seeds advance every generation, so nothing is scored twice on the same noise, and
the final best is re-scored on fresh draws before being reported.

  python cma_fc.py --pop 32 --gens 15 --workers 12
"""
import os, time, pickle, argparse
import multiprocessing as mp
import numpy as np

import ga_fc
from ga_fc import (_init, _evaluate, n_genes, decode, REGIONS, R_LATENT, NSTEPS,
                   SAVE_EVERY)
from paths import RESULTS

# The 120 ceiling let the search win by slowing the drive until the field was almost
# static (10-19 independent swings per run, effectively rank 2). Back to 30, with a low
# floor so fast latents are reachable, and silence still free to go to zero.
TAU_LIM_NARROW = (2.0, 30.0)
SILENT_LIM_WIDE = (0.0, 0.95)    # was (0.75, 0.90); 0 = drive never switches off
GAIN_LIM = (0.3, 1.0)            # per-latent gain ratio; None disables the gain genes
MORAN_LAMBDA = 2.0               # fitness = similarity - lambda * max(0, gap - tol)
MORAN_TOL = 0.10                 # dead zone. This is a guardrail against fields that
                                 # are simply blurred, not an attempt to reproduce the
                                 # empirical correlogram: the fluid's own parameters
                                 # (rotation, wave speed) move that measure too, and are
                                 # not being searched here. For reference on the scale,
                                 # 4 mm of extra smoothing shifts it 0.018, 8 mm 0.048,
                                 # and the hand-made cosine sits 0.11 away.
N_DRAWS = 3
SIGMA0 = 0.25                    # initial CMA step, on the [0,1] hypercube
OUT = os.path.join(RESULTS, "cma_fc")


def evaluate_population(pool, X, seeds):
    """-> (mean fitness, per-draw fitness, ok mask, mean similarity, mean moran gap)."""
    jobs = [(x, s) for x in X for s in seeds]        # flat, so 12 workers stay busy
    res = pool.map(_evaluate, jobs)
    sh = (len(X), len(seeds))
    F = np.array([q["fitness"] for q in res]).reshape(sh)
    ok = np.array([q["ok"] for q in res]).reshape(sh).all(1)
    sim = np.array([q.get("similarity", np.nan) for q in res], float).reshape(sh)
    gap = np.array([q.get("moran_gap", np.nan) for q in res], float).reshape(sh)
    rnk = np.array([q.get("rank", np.nan) for q in res], float).reshape(sh)
    F = np.where(ok[:, None], F, -1e9)
    return (F.mean(1), F, ok, np.nanmean(sim, 1), np.nanmean(gap, 1),
            np.nanmean(rnk, 1))


def run_cma(pop=32, gens=15, workers=12, space="fsaverage5", regions=REGIONS,
            r=R_LATENT, nsteps=NSTEPS, save_every=SAVE_EVERY, sponge=True, fc_path=None,
            centre="none", seed0=0, n_draws=N_DRAWS, sigma0=SIGMA0,
            tau_lim=TAU_LIM_NARROW, silent_lim=SILENT_LIM_WIDE, gain_lim=GAIN_LIM,
            moran_lambda=MORAN_LAMBDA, moran_tol=MORAN_TOL, generator="ou", rung=4,
            x0="mid", fluid=False, fluid_mode="group", preset="default",
            rank_min=10.0, rank_lambda=0.0, verbose=True):
    import cma
    K = len(regions)
    groups = None
    if generator == "ladder":
        import ladder
        groups = ladder.PRESETS[preset]
        n = ladder.N_PARAM
        regions = groups[0] + groups[1] + groups[2]
    else:
        n = n_genes(K, r, gains=gain_lim is not None)
    if fluid:
        import fluid as fl
        n += fl.n_param(fluid_mode)
    os.makedirs(OUT, exist_ok=True)

    if x0 == "rung0" and generator == "ladder":
        import ladder as _l
        start = _l.x0_rung0()
    else:
        start = np.full(n - (fl.n_param(fluid_mode) if fluid else 0), 0.5)
    if fluid:
        # frozen-equivalent fluid: unit speed, Ld 52.4, damping at the floor
        f0 = np.full(fl.n_param(fluid_mode), 0.5)
        f0[0] = 0.5                                   # c0 = 1
        f0[1] = (np.log10(52.4) - 1.0) / 1.5          # the regime everything so far used
        f0[2 if fluid_mode == "maps" else 6] = 0.0    # damping at the floor
        start = np.concatenate([start, f0])
    es = cma.CMAEvolutionStrategy(
        start, sigma0,
        dict(popsize=pop, bounds=[0.0, 1.0], seed=seed0 + 1, verbose=-9))

    history, best = [], None
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers, initializer=_init,
                  initargs=(space, regions, r, nsteps, save_every, sponge, fc_path,
                            centre, tau_lim, silent_lim, gain_lim,
                            moran_lambda, moran_tol, generator, rung, fluid,
                            fluid_mode, groups, rank_min, rank_lambda)) as pool:
        for gen in range(gens):
            t0 = time.time()
            X = np.array(es.ask())
            seeds = [1000 + gen * n_draws + k for k in range(n_draws)]
            mean_fit, F, ok, sim, gap, rnk = evaluate_population(pool, X, seeds)
            es.tell(list(X), list(-mean_fit))         # cma minimises

            i = int(np.argmax(mean_fit))
            draw_sd = float(F[ok].std(1).mean()) if ok.any() else np.nan
            if verbose:
                print(f"  gen {gen:3d}  best {mean_fit[i]:+.4f}  "
                      f"mean {mean_fit[ok].mean():+.4f}  ok {ok.sum():2d}/{pop}  "
                      f"draw sd {draw_sd:.4f}  cma sigma {es.sigma:.3f}"
                      + (f"  [sim {sim[i]:+.3f} gap {gap[i]:.3f}" if moran_lambda else "")
                      + (f" rank {rnk[i]:.1f}]" if rank_lambda else
                         ("]" if moran_lambda else ""))
                      + f"  [{time.time()-t0:.0f}s]", flush=True)
            history.append(dict(gen=gen, X=X.copy(), mean_fit=mean_fit.copy(),
                                per_draw=F.copy(), ok=ok.copy(), sigma=float(es.sigma),
                                similarity=sim.copy(), moran_gap=gap.copy(),
                                rank=rnk.copy(),
                                seeds=seeds))
            if best is None or mean_fit[i] > best[0]:
                best = (float(mean_fit[i]), X[i].copy(), gen)

        # honest number for the winner: draws it has never been scored on
        fresh = [90_000 + k for k in range(8)]
        _, Fb, _, simb, gapb, rnkb = evaluate_population(pool, best[1][None, :], fresh)

    if generator == "ladder":
        import ladder
        L, silent, tau = None, None, None
        xb = best[1]
        if fluid:
            import fluid as fl
            nf = fl.n_param(fluid_mode)
            decoded = ladder.decode(xb[:-nf], rung)
            fpb = fl.decode_mode(xb[-nf:], fluid_mode)
            decoded.update({f"fluid_{k}": v for k, v in fpb.items()})
        else:
            decoded = ladder.decode(xb, rung)
    else:
        L, silent, tau = decode(best[1], K, r, tau_lim, silent_lim, gain_lim)
        decoded = dict(loadings=L, silent=silent, tau=tau)
    if verbose:
        print(f"\n  best during search {best[0]:+.4f} (generation {best[2]})")
        print(f"  re-scored on 8 fresh draws: {Fb.mean():+.4f} +- {Fb.std():.4f}"
              + (f"   (similarity {simb[0]:+.4f}, moran gap {gapb[0]:.4f}"
                 f"{f', rank {rnkb[0]:.1f}' if rank_lambda else ''})"
                 if moran_lambda else ""))
        if generator == "ladder":
            for k, v in decoded.items():
                if isinstance(v, str) or (isinstance(v, tuple)
                                          and all(isinstance(q, str) for q in v)):
                    txt = str(v)
                elif np.isscalar(v):
                    txt = f"{v:8.3f}"
                else:
                    txt = np.array2string(np.asarray(v), precision=2)
                print(f"    {k:14s} {txt}")
        else:
            print(f"  taus {np.round(tau, 1)}   silent {np.round(silent, 3)}"
                  + (f"   latent gains {np.round(np.linalg.norm(L, axis=0), 3)}"
                     if gain_lim else ""))

    path = os.path.join(OUT, f"cma_{generator}_pop{pop}_gen{gens}_seed{seed0}.pkl")
    with open(path, "wb") as fh:
        pickle.dump(dict(history=history, best=best, fresh=Fb, regions=regions, r=r, K=K,
                         nsteps=nsteps, sponge=sponge, fc_path=fc_path, centre=centre,
                         pop=pop, gens=gens, seed0=seed0, n_draws=n_draws,
                         tau_lim=tau_lim, silent_lim=silent_lim, gain_lim=gain_lim,
                         moran_lambda=moran_lambda, moran_tol=moran_tol,
                         generator=generator, rung=rung, x0=x0, fluid=fluid,
                         fluid_mode=fluid_mode, preset=preset, groups=groups,
                         rank_min=rank_min, rank_lambda=rank_lambda,
                         best_decoded=decoded), fh)
    if verbose:
        print(f"  wrote {path}")
    return history, best, path


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pop", type=int, default=32)
    ap.add_argument("--gens", type=int, default=15)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--space", default="fsaverage5")
    ap.add_argument("--r", type=int, default=R_LATENT)
    ap.add_argument("--regions", nargs="*", type=int, default=REGIONS)
    ap.add_argument("--nsteps", type=int, default=NSTEPS)
    ap.add_argument("--draws", type=int, default=N_DRAWS, dest="n_draws")
    ap.add_argument("--sigma0", type=float, default=SIGMA0)
    ap.add_argument("--tau-lim", nargs=2, type=float, default=list(TAU_LIM_NARROW),
                    dest="tau_lim")
    ap.add_argument("--gain-lim", nargs=2, type=float, default=list(GAIN_LIM),
                    dest="gain_lim", help="per-latent gain range; 0 0 disables")
    ap.add_argument("--silent-lim", nargs=2, type=float, default=list(SILENT_LIM_WIDE),
                    dest="silent_lim")
    ap.add_argument("--no-sponge", action="store_true")
    ap.add_argument("--fc", default=None)
    ap.add_argument("--centre", default="none", choices=("none", "double"))
    ap.add_argument("--moran-lambda", type=float, default=MORAN_LAMBDA,
                    dest="moran_lambda", help="0 disables the spatial-scale penalty")
    ap.add_argument("--moran-tol", type=float, default=MORAN_TOL, dest="moran_tol",
                    help="dead zone: gaps below this cost nothing")
    ap.add_argument("--generator", default="ou", choices=("ou", "ladder"))
    ap.add_argument("--rung", type=int, default=4, help="ladder rung, 0-4")
    ap.add_argument("--preset", default="default", choices=("default", "sensory"),
                    help="ladder region groups; 'sensory' is visual vs somatomotor "
                         "with auditory left to the distance kernel")
    ap.add_argument("--fluid-mode", default="group", choices=("group", "maps"),
                    dest="fluid_mode", help="'maps' grades the medium by myelin, "
                                            "thickness and sulcal depth")
    ap.add_argument("--rank-min", type=float, default=10.0, dest="rank_min")
    ap.add_argument("--rank-lambda", type=float, default=0.0, dest="rank_lambda",
                    help="0 disables the field-rank floor")
    ap.add_argument("--fluid", action="store_true",
                    help="also search wave speed, damping and rotation as fields")
    ap.add_argument("--x0", default="mid", choices=("mid", "rung0"),
                    help="ladder only: start from the dipole corner instead of centre")
    ap.add_argument("--seed", type=int, default=0, dest="seed0")
    a = ap.parse_args()
    gl = None if tuple(a.gain_lim) == (0.0, 0.0) else tuple(a.gain_lim)
    if a.generator == "ladder":
        import ladder
        head = (f"CMA-ES over the ladder: rung {a.rung}, {ladder.N_PARAM} parameters, "
                f"{len(ladder.CORE_POS)}+{len(ladder.CORE_NEG)}+{len(ladder.SATELLITES)}"
                f" regions")
    else:
        head = (f"CMA-ES over the latent only: K={len(a.regions)} regions, "
                f"r={a.r} latents, "
                f"{n_genes(len(a.regions), a.r, gains=gl is not None)} genes, "
                f"tau {tuple(a.tau_lim)}, silent {tuple(a.silent_lim)}, gain {gl}")
    print(f"{head}, pop {a.pop} x {a.gens} gens, "
          f"{a.n_draws} draws each ({a.pop*a.gens*a.n_draws} runs), "
          f"moran lambda {a.moran_lambda} tol {a.moran_tol}, "
          f"rank floor {a.rank_min} lambda {a.rank_lambda}, preset {a.preset}, "
          f"sponge {'off' if a.no_sponge else 'on'}, {a.workers} workers")
    run_cma(pop=a.pop, gens=a.gens, workers=a.workers, space=a.space, regions=a.regions,
            r=a.r, nsteps=a.nsteps, sponge=not a.no_sponge, fc_path=a.fc,
            centre=a.centre, seed0=a.seed0, n_draws=a.n_draws, sigma0=a.sigma0,
            tau_lim=tuple(a.tau_lim), silent_lim=tuple(a.silent_lim), gain_lim=gl,
            moran_lambda=a.moran_lambda, moran_tol=a.moran_tol,
            generator=a.generator, rung=a.rung, x0=a.x0, fluid=a.fluid,
            fluid_mode=a.fluid_mode, preset=a.preset, rank_min=a.rank_min,
            rank_lambda=a.rank_lambda)


if __name__ == "__main__":
    main()
