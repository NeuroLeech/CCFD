"""Genetic algorithm over the input AND the fluid: which regions are driven, how,
on what timescales, and in what dynamical regime.

    fitness = W_SIM*similarity + W_RICH*richness + W_SPEED*speed + W_WAVE*wave

Two of the terms face the fMRI and two face the fluid. Each was checked against a
floor and a ceiling before being used, because the term this replaced turned out
to score white noise and the brain identically:

    similarity  does each pattern the model produces resemble one the brain
                actually visits?  noise 0.270, fMRI 0.370. A single frozen
                pattern scores well here, which is why richness sits beside it.
    richness    how many distinct configurations does the flow visit?  noise
                ~0.94, fMRI 0.922, every run so far 0.23-0.60. Noise scores well
                here, which is why similarity sits beside it.
    speed       normal-flow transport speed, the stage-1 measure with the widest
                spread and the best ranking reliability (0.86 at 7000 steps).
    wave        anisotropy * (1 - var_mod): travelling structure specifically.
                noise 0.06, standing wave 0.29, travelling wave 0.91. var_mod
                alone is minimised by noise, exactly as richness alone is.

Both are scaled so 1 is the fMRI, but their zeros are NOT the same thing and it
matters. similarity's zero is white noise. richness's zero is the worst model run,
because noise scores ~0.94 on richness - ABOVE the brain's 0.922 - and so sits at
+1.03 on the scaled axis, at the ceiling rather than the floor. That is exactly
why richness cannot be selected on alone: the cheapest way to win it is to become
noise, and only similarity forbids that.

The brain sits at the corner where both are high and nothing we have run reaches it.

The previous fitness used k-medoid matching against canonical fMRI patterns. It
was replaced because it cannot distinguish the brain from white noise: split-half
fMRI scores 0.138-0.156 on it and white noise 0.092-0.158. See stage2.match_score.

Both sides live in 180 Glasser parcels, which is what lets fs_LR fMRI and
fsaverage5 model output be compared without any surface resampling.

Each generation uses one noise seed, changed between generations, so genomes are
compared on equal footing within a generation without the search locking onto a
single realisation. Elites carry over unchanged into the next generation and are
therefore re-scored under a new seed - that is what makes across-seed reliability
recoverable from the history afterwards.

tau_ratio (field timescale / drive timescale) and dir_coherence are recorded but
NOT selected on, so that any movement in them is an observation rather than
something the search was told to produce.
"""
import sys, os, time, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
import multiprocessing as mp

K, R = 8, 3
POP, NGEN = 32, 4
# sim and rich are already on a common 0=noise, 1=brain scale. speed has no brain
# reference (fMRI at TR 2.2 s has no velocity field), so it is divided by a
# nominal SPEED_REF to put it on the same order; wave is a product of two
# quantities that are each already O(1).
# The fluid is frozen (see genome.LD_FIXED). This search asks one question: does
# the ARRANGEMENT of the input - which regions, how they load onto the latents,
# and each latent's timescale and sparsity - raise the correlation with fMRI
# states. So sim carries the whole fitness and everything else is recorded only.
#
# Note sim is uncapped and a single near-frozen pattern can exceed the brain's own
# self-similarity on it (0.38 against 0.37 measured). Set --w-rich above 0 to make
# that route unattractive.
W_SIM, W_RICH, W_SPEED, W_WAVE = 1.0, 0.0, 0.0, 0.0
SPEED_REF = 0.5
NSTEPS = 7000
# weak selection pressure: this is a long exploratory run and the point is
# to cover the input space, not to converge on generation 10
ELITE, TOURN = 2, 2
from paths import RESULTS as OUT

_W = {}


def _init(nsteps, w):
    from explore import Rig
    from stage2 import fmri_pcs
    _W["rig"] = Rig("fsaverage5", fluid=dict(nsteps=nsteps))
    _W["pcs"] = fmri_pcs()
    _W["w"] = w


def _evaluate(args):
    gx, gregions, gK, gr, seed = args
    from genome import Genome
    from explore import run_genome
    g = Genome(np.asarray(gregions), np.asarray(gx), gK, gr)
    r = run_genome(g, _W["rig"], seed=seed)
    if not r["ok"]:
        return dict(fitness=-1e9, ok=False, reason=r.get("reason", "?"))
    from stage2 import _normalise, smooth_decimate, similarity, richness
    Pm = smooth_decimate(_normalise(r["parcels"]),
                         factor=_W["rig"].f["decimate"])
    # raw, unscaled: this drives a search by ranking genomes against each other,
    # so an absolute floor and ceiling would buy nothing.
    sim, rich = similarity(Pm, _W["pcs"]), richness(Pm)
    s, rr = sim, rich
    # Travelling structure, not just structure. var_mod alone is minimised by white
    # noise, exactly as rich alone is; multiplying by anisotropy is what excludes
    # that. On synthetics this reads 0.06 (noise), 0.29 (standing), 0.91 (travelling).
    wave = float(r["anisotropy"]*(1.0 - r["var_mod"]))
    spd = float(r["wave_speed"])/SPEED_REF
    w = _W["w"]
    # leaving the linear regime is penalised rather than banned: the search should
    # feel the boundary, not fall off a cliff at it
    pen = 0.0 if r["linear_ok"] else 1.0 - min(r["peak_frac"]/0.10, 3.0)/3.0
    raw = w["sim"]*s + w["rich"]*rr + w["speed"]*spd + w["wave"]*wave
    return dict(fitness=raw*(1.0 if r["linear_ok"] else 0.3*pen),
                ok=True, sim=sim, rich=rich, s=s, rr=rr,
                wave=wave, spd=spd, speed=r["wave_speed"],
                var_mod=r["var_mod"], tau_ratio=r["tau_ratio"],
                dir_coh=r["dir_coherence"],     # recorded, not selected on
                n_model_frames=int(Pm.shape[1]), anis=r["anisotropy"],
                peak=r["peak_frac"], linear_ok=r["linear_ok"],
                regions=r["regions"], silents=r["silents"], taus=r["taus"],
                Ld=r["Ld"], sponge_strength=r["sponge_strength"],
                sponge_width=r["sponge_width"])


def run_ga(pop=POP, ngen=NGEN, nsteps=NSTEPS, workers=8, seed0=0, verbose=True,
           w=None, out="ga_history.pkl"):
    w = w or dict(sim=W_SIM, rich=W_RICH, speed=W_SPEED, wave=W_WAVE)
    from genome import random_genome, mutate, crossover
    from mesh_cache import load_cortex
    parcels = load_cortex("fsaverage5", verbose=False).parcels
    rng = np.random.default_rng(seed0)
    P = [random_genome(rng, K, R, parcels) for _ in range(pop)]

    ctx = mp.get_context("spawn")
    history = []
    with ctx.Pool(workers, initializer=_init, initargs=(nsteps, w)) as pool:
        for gen in range(ngen):
            t0 = time.time()
            args = [(g.x, g.regions, g.K, g.r, gen) for g in P]
            res = pool.map(_evaluate, args)
            fit = np.array([r["fitness"] for r in res])
            order = np.argsort(-fit)
            el = time.time() - t0
            best = res[order[0]]
            if verbose:
                nb = sum(1 for r in res if not r.get("linear_ok", True))
                ok = [r for r in res if r.get("ok")]
                med = lambda k: np.median([r[k] for r in ok])
                print(f"  gen {gen:3d}: best {fit[order[0]]:6.3f} "
                      f"[sim {best['s']:+.2f} rich {best['rr']:+.2f} "
                      f"spd {best['spd']:.2f} wav {best['wave']:.2f}]  "
                      f"| med fit {np.median(fit):6.3f}  "
                      f"med sim {med('s'):+.2f} rich {med('rr'):+.2f} "
                      f"spd {med('spd'):.2f} wav {med('wave'):.2f}  "
                      f"| Ld {med('Ld'):5.1f} tauR {med('tau_ratio'):.2f}  "
                      f"{nb}/{pop} nonlin [{el:.0f}s]", flush=True)
            history.append(dict(gen=gen, fitness=fit.copy(), seed=gen,
                                order=order.copy(), res=res,
                                pop=[ (g.x.copy(), g.regions.copy()) for g in P ]))
            # written every generation, not at the end: a long run should be
            # inspectable while it is still going and survive being stopped
            with open(os.path.join(OUT, out), "wb") as f:
                pickle.dump(dict(history=history, K=K, R=R, nsteps=nsteps,
                                 w=w, pop=pop), f)
            if gen == ngen - 1:
                break
            # elitism + tournament selection
            newP = [P[i] for i in order[:ELITE]]
            while len(newP) < pop:
                pick = lambda: P[min(rng.choice(pop, TOURN), key=lambda i: -fit[i])]
                child = crossover(pick(), pick(), rng)
                newP.append(mutate(child, rng, parcels))
            P = newP
    return P, history


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", type=int, default=POP)
    ap.add_argument("--gens", type=int, default=NGEN)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--nsteps", type=int, default=NSTEPS)
    ap.add_argument("--w-sim", type=float, default=W_SIM)
    ap.add_argument("--w-rich", type=float, default=W_RICH)
    ap.add_argument("--w-speed", type=float, default=W_SPEED)
    ap.add_argument("--w-wave", type=float, default=W_WAVE)
    ap.add_argument("--out", type=str, default="ga_history.pkl")
    a = ap.parse_args()
    W = dict(sim=a.w_sim, rich=a.w_rich, speed=a.w_speed, wave=a.w_wave)
    print(f"GA: pop {a.pop}, {a.gens} generations, {a.nsteps} steps/run, "
          f"{a.workers} workers, K={K} r={R}")
    print(f"  weights {W}")
    print("  sim/rich scale: 0 = white noise, 1 = empirical fMRI")
    t0 = time.time()
    P, H = run_ga(a.pop, a.gens, a.nsteps, a.workers, w=W, out=a.out)
    print(f"total {time.time()-t0:.0f}s")
    b = H[-1]["res"][H[-1]["order"][0]]
    print(f"\nbest genome: sim {b['sim']:.3f} ({b['s']:+.2f})  "
          f"rich {b['rich']:.3f} ({b['rr']:+.2f})  wave {b['wave']:.3f}  "
          f"speed {b['speed']:.3f}")
    print(f"  anisotropy {b['anis']:.3f}  var_mod {b['var_mod']:.3f}  "
          f"tau_ratio {b['tau_ratio']:.2f}  dir_coh {b['dir_coh']:.3f}  "
          f"peak {100*b['peak']:.1f}% of depth "
          f"({b['n_model_frames']} model frames)")
    print(f"  fluid: Ld {b['Ld']:.1f} mm (L/Ld {228/b['Ld']:.1f})  "
          f"sponge {b['sponge_strength']:.2f} over {b['sponge_width']:.0f} mm")
    print(f"  regions: {b['regions']}")
    print(f"  sparsity {np.round(b['silents'],2)}  taus {np.round(b['taus'],0)}")
    print(f"  wrote {a.out}")
