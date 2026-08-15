"""Genome: what a genetic algorithm searches over.

Fixed K regions and r latents. Two parts, because they need different operators:

    regions   K distinct integers drawn from the 180 Glasser parcels  (discrete)
    x         K*r loadings + r thresholds + r timescales             (continuous)

The continuous part is stored as a unit hypercube vector in [0,1]^n, so any GA that
can mutate/crossover a bounded real vector works unchanged; `decode` maps it to
physical values.

The fluid parameters are searched alongside the input. Which three, and why not
the others:

    Ld        deformation radius. Sets the Coriolis parameter f = c/Ld, and with
              it the scale above which rotation governs the response. The surface
              is 228 mm across, so this gene spans L/Ld from about 1 to 45.
    sponge_*  the medial wall, continuously from perfectly reflecting to fully
              absorbing.

    c, g, H   NOT searched. Only the product gH sets the wave speed; g/H alone
              rescales u against h, which does nothing in a linear model. And
              dt = cfl*d_min/c, so making c a gene would change either the
              simulated duration or the step count per genome - i.e. the run
              length, which is deliberately fixed. c is in any case largely
              redundant with the tau genes, both acting on the ratio of drive
              time to crossing time.

    amp       NOT searched, held at AMP_FIXED. The drive is normalised to unit RMS
              before amp is applied, and the model is linear, so amp cannot change
              any normalised measure - it only slides the run up and down the
              linear-regime penalty. It was costing a gene for nothing.
"""
import numpy as np
from dataclasses import dataclass

# physical ranges for the continuous part
LOAD_LIM = 2.0             # loadings in [-LOAD_LIM, +LOAD_LIM]
SILENT_LIM = (0.75, 0.90)  # target fraction of the run with NO drive, PER LATENT
TAU_LIM = (15.0, 30.0)     # time units, PER LATENT

# ---- the frozen regime -------------------------------------------------------
# Regime p00 of the parameter sweep, picked by eye from the rendered videos. The
# fluid is FIXED from here on: this search varies only the arrangement of the
# input, so that any change in fMRI similarity is attributable to the input
# rather than to the regime, which otherwise dominates it.
LD_FIXED = 52.410806126395016              # L/Ld = 4.4
SPONGE_STRENGTH_FIXED = 0.2697867137638703
SPONGE_WIDTH_FIXED = 7.253543816490708
AMP_FIXED = 2.0e-4


@dataclass
class Genome:
    regions: np.ndarray       # (K,) parcel ids
    x: np.ndarray             # (K*r + 2r,) in [0,1]
    K: int
    r: int

    def copy(self):
        return Genome(self.regions.copy(), self.x.copy(), self.K, self.r)


N_FLUID = 0               # the fluid is frozen, not searched


def n_continuous(K, r):
    return K*r + 2*r + N_FLUID


def decode(g):
    """Unit hypercube -> physical parameters."""
    K, r = g.K, g.r
    x = np.clip(g.x, 0.0, 1.0)
    i = 0
    L = (x[i:i+K*r].reshape(K, r)*2.0 - 1.0)*LOAD_LIM;            i += K*r
    silent = SILENT_LIM[0] + x[i:i+r]*(SILENT_LIM[1]-SILENT_LIM[0]); i += r
    tau = TAU_LIM[0] + x[i:i+r]*(TAU_LIM[1]-TAU_LIM[0]);          i += r
    Ld, s_str, s_w = LD_FIXED, SPONGE_STRENGTH_FIXED, SPONGE_WIDTH_FIXED
    assert i == len(x), (i, len(x))
    return dict(regions=np.asarray(g.regions, int), loadings=L,
                silents=silent, taus=tau, amp=AMP_FIXED,
                Ld=float(Ld), sponge_strength=float(s_str),
                sponge_width=float(s_w))


def random_genome(rng, K, r, parcels):
    return Genome(regions=np.asarray(rng.choice(parcels, K, replace=False), int),
                  x=rng.random(n_continuous(K, r)), K=K, r=r)


def validate(g, parcels):
    """Reasons this genome cannot be evaluated. Empty list means fine."""
    bad = []
    if len(set(int(p) for p in g.regions)) != g.K:
        bad.append("repeated regions")
    if not set(int(p) for p in g.regions) <= set(int(p) for p in parcels):
        bad.append("region id not in the atlas")
    if len(g.x) != n_continuous(g.K, g.r):
        bad.append(f"expected {n_continuous(g.K, g.r)} continuous genes, got {len(g.x)}")
    if np.any(~np.isfinite(g.x)):
        bad.append("non-finite gene")
    p = decode(g)
    if np.abs(p["loadings"]).max() < 1e-6:
        bad.append("all loadings ~zero")
    return bad


# ---- GA operators, kept minimal and separate from the search itself ----
def mutate(g, rng, parcels, p_region=0.12, sigma=0.15):
    h = g.copy()
    for k in range(h.K):
        if rng.random() < p_region:
            avail = np.setdiff1d(parcels, h.regions)
            h.regions[k] = int(rng.choice(avail))
    h.x = np.clip(h.x + rng.normal(0.0, sigma, h.x.shape), 0.0, 1.0)
    return h


def crossover(a, b, rng):
    """Uniform on the continuous part; region sets merged without repeats."""
    m = rng.random(a.x.shape) < 0.5
    x = np.where(m, a.x, b.x)
    pool, regions = list(dict.fromkeys(list(a.regions) + list(b.regions))), []
    order = rng.permutation(len(pool))
    for i in order:
        if len(regions) < a.K and pool[i] not in regions:
            regions.append(int(pool[i]))
    return Genome(np.array(regions, int), x, a.K, a.r)
