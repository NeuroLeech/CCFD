"""A nested family of input processes, from the slow dipole outwards.

Every rung is switched on by its own parameters and collapses to the rung below when
they are set to their off values, so the family provably contains the anticorrelated
cosine (similarity ~0.29) as an interior point. A search over it therefore cannot do
worse than the best hand-made drive, and anything it gains is gain over that.

  rung 0  core dipole on a slow carrier: the cosine, exactly
  rung 1  satellite regions added, with their own gain
  rung 2  coalitions: which regions join is drawn afresh each cycle
  rung 3  per-region phase offsets, so coalition members lead and lag
  rung 4  carrier decoupling: events drift off the carrier towards a point process

Which regions join, and with which sign, is not specified by hand beyond the core. It
comes from geodesic distance to the two poles of the dipole through a kernel with a
near-range and a mid-range term - the spatial-process view of antagonism, where the
layout of one sign is informative about the layout of the other. The kernel's ranges are
searched, so 'near recruits with the same sign, mid-range recruits against it' is a
hypothesis the optimiser can accept or reject rather than an assumption.

  python ladder.py --check          # rung 0 must reproduce the cosine
  python ladder.py --random 8       # score random genomes at full rung
"""
import os, argparse
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra

from paths import CACHE

# the dipole run_v3.py uses: first three positive, last four negative
CORE_POS = [9, 1, 24]
CORE_NEG = [150, 151, 132, 30, 33, 65, 88]             # DMN
SATELLITES = [51, 53, 4, 5, 173, 174, 124, 104]        # Som, Vis, Aud
N_PARAM = 12

# sensory-only: the dipole runs visual against somatomotor, with auditory left for the
# distance kernel to claim for one side or the other rather than being assigned by hand
PRESETS = {
    "default": ([9, 1, 24], [150, 151, 132, 30, 33, 65, 88],
                [51, 53, 4, 5, 173, 174, 124, 104]),
    "sensory": ([4, 5], [51, 53], [173, 174, 124, 104]),
}


def set_groups(pos, neg, sat):
    """Repoint the three region groups. Called in each worker before anything caches
    geodesics, since the distance kernel is defined relative to the poles."""
    global CORE_POS, CORE_NEG, SATELLITES
    CORE_POS, CORE_NEG, SATELLITES = list(pos), list(neg), list(sat)
    return CORE_POS + CORE_NEG + SATELLITES


# ------------------------------------------------------------------ geometry
def _white_graph(cortex):
    """Mesh graph weighted by WHITE-surface edge lengths, plus the white coordinates."""
    import nibabel as nib
    fs = f"/Applications/freesurfer/dev/subjects/{cortex.mesh}/surf/lh.white"
    Vw, _ = nib.freesurfer.read_geometry(fs)
    Vw = Vw[np.asarray(cortex.old, int)]                # submesh, white coordinates
    E = cortex.edges
    w = np.linalg.norm(Vw[E[:, 0]] - Vw[E[:, 1]], axis=1)
    G = sp.coo_matrix((w, (E[:, 0], E[:, 1])), shape=(cortex.nV, cortex.nV))
    return (G + G.T).tocsr(), Vw


def label_geodesic(cortex, labels, n=None, verbose=True):
    """Geodesic distance (mm) between the centroids of an arbitrary vertex labelling.

    Same measure as parcel_geodesic, for sub-parcel pieces: `labels` is the (nV,) array
    subparcels.split_parcels returns, with -1 off the driven set."""
    n = int(labels.max()) + 1 if n is None else n
    G, Vw = _white_graph(cortex)
    centroid = []
    for k in range(n):
        m = np.flatnonzero(labels == k)
        centroid.append(int(m[np.argmin(((Vw[m] - Vw[m].mean(0)) ** 2).sum(1))]))
    centroid = np.array(centroid)
    D = dijkstra(G, indices=centroid)[:, centroid]
    if verbose:
        print(f"  geodesics for {n} pieces (median {np.median(D[D > 0]):.0f} mm, "
              f"max {D.max():.0f} mm)")
    return D, centroid


def parcel_geodesic(cortex, parcels, verbose=True):
    """Geodesic distance (mm) between parcel centroids, along the white surface.

    Distances are measured on the mesh graph with white-surface edge lengths, not on the
    inflated coordinates the solver carries, because inflation is exactly the operation
    that destroys the distances this kernel is about."""
    cache = os.path.join(CACHE, f"geodesic_{cortex.mesh}.npz")
    key = np.asarray(parcels, int)
    if os.path.exists(cache):
        z = np.load(cache)
        if np.array_equal(z["parcels"], key):
            return z["D"], z["centroid"]

    G, Vw = _white_graph(cortex)
    centroid = []
    for p in key:
        m = np.flatnonzero(cortex.lab == p)
        centroid.append(int(m[np.argmin(((Vw[m] - Vw[m].mean(0)) ** 2).sum(1))]))
    centroid = np.array(centroid)
    D = dijkstra(G, indices=centroid)[:, centroid]
    np.savez(cache, D=D, centroid=centroid, parcels=key)
    if verbose:
        print(f"[{cortex.mesh}] geodesic distances for {len(key)} parcels "
              f"(median {np.median(D[D > 0]):.0f} mm, max {D.max():.0f} mm)")
    return D, centroid


def affinity(D, n_pos, n_neg, lam_near, lam_mid, w_mid):
    """Signed affinity of every region to the dipole, from distance to its two poles.

    f(d) = exp(-d/lam_near) - w_mid * exp(-d/lam_mid) is positive nearby and negative at
    mid range when w_mid > 0, so a region close to the positive pole is recruited with
    it while one at mid range is recruited against it. Affinity is the difference between
    what the positive and negative poles offer, so sign(a) is which side a region joins
    and |a| is how strongly it is claimed."""
    f = lambda d: np.exp(-d / lam_near) - w_mid * np.exp(-d / lam_mid)
    to_pos = f(D[:, :n_pos]).mean(1)
    to_neg = f(D[:, n_pos:n_pos + n_neg]).mean(1)
    return to_pos - to_neg


def x0_rung0(period_steps=2000.0):
    """Starting point at the dipole corner: full participation, fixed signs, locked to
    the carrier, no satellites. Values sit just inside the bounds rather than on them,
    because CMA-ES starting on a wall spends its first generations pushing into it."""
    x = np.full(N_PARAM, 0.5)
    x[0] = np.clip(np.log10(period_steps / 200.0) / 1.3, 0.02, 0.98)
    x[1] = 0.98      # duty ~ full, so the bumps sum to a smooth carrier
    x[2] = 0.15      # satellites barely on, free to grow
    x[3] = 0.98      # core participates every cycle
    x[4] = 0.15      # satellites rarely join
    x[5] = 0.02      # no offsets
    x[6] = 0.98      # locked to the carrier
    x[10] = 0.98     # signs coherent
    x[11] = 0.02     # no amplitude heterogeneity
    return x


# ------------------------------------------------------------------ the generator
def decode(x, rung=4, override=None):
    """[0,1]^12 -> named parameters. Higher rungs are pinned to their off values when
    `rung` excludes them, which is what makes the family nested."""
    x = np.clip(np.asarray(x, float), 0.0, 1.0)
    p = dict(
        period=200.0 * (10.0 ** (1.3 * x[0])),      # 200 - 4000 steps
        duty=0.15 + 0.85 * x[1],                    # fraction of a cycle driven
        g_sat=x[2],                                 # satellite gain, 0 = rung 0
        p_core=0.5 + 0.5 * x[3],                    # core participation per cycle
        p_sat=x[4],                                 # satellite participation per cycle
        sigma_off=0.5 * x[5],                       # phase offset sd, in cycles
        kappa=x[6],                                 # 1 = locked to carrier, 0 = Poisson
        lam_near=5.0 + 55.0 * x[7],                 # mm
        lam_mid=20.0 + 180.0 * x[8],                # mm
        w_mid=1.5 * x[9],                           # strength of the mid-range term
        coherence=x[10],                            # 1 = signs fixed, 0 = redrawn
        amp_sd=1.2 * x[11],                         # lognormal sd of event amplitude
    )
    if rung < 4:
        p["kappa"] = 1.0
    if rung < 3:
        p["sigma_off"] = 0.0
    if rung < 2:
        p["p_core"], p["p_sat"], p["coherence"], p["amp_sd"] = 1.0, 1.0, 1.0, 0.0
    if rung < 1:
        p["g_sat"] = 0.0
    if override:
        unknown = set(override) - set(p)
        if unknown:
            raise KeyError(f"unknown ladder parameters: {sorted(unknown)}")
        p.update(override)
    return p


def build(x, nsteps, dt, D, n_pos, n_neg, n_sat, rung=4, seed=0, sharpness=8.0,
          override=None):
    """-> (nsteps, K) region timecourses. K = n_pos + n_neg + n_sat, in that order."""
    p = decode(x, rung, override)
    rng = np.random.default_rng(seed)
    K = n_pos + n_neg + n_sat
    t = np.arange(nsteps) * dt
    period = p["period"] * dt                        # carrier period in time units

    base = np.concatenate([np.ones(n_pos), -np.ones(n_neg),
                           np.sign(affinity(D[n_pos + n_neg:], n_pos, n_neg,
                                            p["lam_near"], p["lam_mid"], p["w_mid"]))])
    claim = np.concatenate([np.ones(n_pos + n_neg),
                            np.abs(affinity(D[n_pos + n_neg:], n_pos, n_neg,
                                            p["lam_near"], p["lam_mid"], p["w_mid"]))])
    claim[n_pos + n_neg:] /= max(claim[n_pos + n_neg:].max(), 1e-12)   # 0..1
    gain = np.concatenate([np.ones(n_pos + n_neg), np.full(n_sat, p["g_sat"])])

    # Events sit half a carrier period apart and alternate polarity, so at full duty,
    # full participation and no jitter the sum of bumps is the cosine itself - that is
    # what puts the reference drive inside the family rather than merely near it.
    # As kappa falls the onsets scatter towards a Poisson process of the same rate.
    half = period / 2.0
    n_ev = max(int(np.ceil(t[-1] / half)) + 2, 1)
    onsets = np.arange(n_ev) * half
    polarity = np.where(np.arange(n_ev) % 2 == 0, 1.0, -1.0)
    if p["kappa"] < 1.0:
        onsets = onsets + rng.normal(0.0, (1.0 - p["kappa"]) * half, n_ev)
    width = p["duty"] * half / 2.0

    A = np.zeros((nsteps, K))
    for e, t0 in enumerate(onsets):
        join = rng.random(K) < np.where(np.arange(K) < n_pos + n_neg,
                                        p["p_core"], p["p_sat"] * claim)
        if not join.any():
            continue
        sign = base * np.where(rng.random(K) < p["coherence"], 1.0,
                               rng.choice([-1.0, 1.0], K))
        amp = np.exp(rng.normal(0.0, p["amp_sd"], K)) if p["amp_sd"] > 0 else np.ones(K)
        off = rng.normal(0.0, p["sigma_off"], K) * period if p["sigma_off"] > 0 \
            else np.zeros(K)
        for k in np.flatnonzero(join):
            # one smooth half-cycle bump per event; sharpness keeps rung 0 sinusoidal
            u = (t - (t0 + off[k])) / max(width, 1e-9)
            A[:, k] += (polarity[e] * sign[k] * gain[k] * amp[k]
                        * np.exp(-0.5 * (u * u) * (sharpness / 8.0)))
    return A


def balance(A, w):
    """Remove the component along the region weights, so every step moves depth between
    regions and never changes total mass - the projection NetworkDrive applies."""
    ww = float(w @ w)
    return A - np.outer(A @ w / ww, w)


def make_drive(cortex, x, nsteps, dt, amp=2e-4, rung=4, seed=0, tapers=None,
               regions=None, D=None, override=None):
    """A ready-to-run RegionDrive carrying the ladder's timecourses."""
    from input2 import RegionDrive
    regions = regions or (CORE_POS + CORE_NEG + SATELLITES)
    if D is None:
        D, _ = parcel_geodesic(cortex, regions, verbose=False)
    A = build(x, nsteps, dt, D, len(CORE_POS), len(CORE_NEG),
              len(regions) - len(CORE_POS) - len(CORE_NEG), rung, seed,
              override=override)
    drive = RegionDrive(cortex, regions, A, amp=amp, nsteps=nsteps, tapers=tapers)
    drive.Aser = balance(drive.Aser, drive.w)
    rms = np.sqrt((drive.Aser ** 2).mean())
    if rms > 1e-12:
        drive.Aser *= amp / rms
    drive.dead = bool(rms <= 1e-12)
    return drive


# ------------------------------------------------------------------ checks
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="rung 0 against the cosine")
    ap.add_argument("--random", type=int, default=0, metavar="N",
                    help="score N random genomes at full rung")
    ap.add_argument("--rung", type=int, default=4)
    ap.add_argument("--nsteps", type=int, default=7000)
    a = ap.parse_args()

    from mesh_cache import load_cortex
    from input2 import parcel_tapers
    from run_ou import run, CFL, C
    from fc_score import FCTarget
    from fc_moran import MoranMatch

    c = load_cortex("fsaverage5", verbose=False)
    dt = CFL * c.d.min() / C
    regions = CORE_POS + CORE_NEG + SATELLITES
    D, _ = parcel_geodesic(c, regions)
    tap = parcel_tapers(c, verbose=False)
    target = FCTarget(c, verbose=False)
    mm = MoranMatch(c, target)

    def score(drive, label):
        fr, _ = run(c, drive, a.nsteps, 25, sponge=True, verbose=False)
        Z, _ = target.model_z(fr)
        sim = float(target._prep(target.model_edges(Z=Z)[0]) @ target.y)
        gap = mm.gap(Z)
        print(f"  {label:44s} sim {sim:+.4f}  gap {gap:.4f}  lam2 {sim-2*gap:+.4f}")
        return sim, gap

    if a.check:
        # the reference: cosine by hand on the core seven, as in run_v3
        from input2 import RegionDrive
        t = np.arange(a.nsteps) * dt
        for period in (1000, 2000):
            s = np.cos(2 * np.pi * t / (period * dt))
            A = np.c_[[s] * len(CORE_POS) + [-s] * len(CORE_NEG)].T
            d = RegionDrive(c, CORE_POS + CORE_NEG, A, amp=2e-4, nsteps=a.nsteps,
                            tapers=tap)
            d.Aser = balance(d.Aser, d.w)
            d.Aser *= 2e-4 / np.sqrt((d.Aser ** 2).mean())
            score(d, f"reference cosine, period {period}")
            x = np.zeros(N_PARAM)
            x[0] = np.log10(period / 200.0) / 1.3
            x[1] = 1.0                                    # duty 1.0 = smooth, sinusoidal
            print(f"    rung 0 target period {decode(x, 0)['period']:.0f} steps")
            score(make_drive(c, x, a.nsteps, dt, rung=0, tapers=tap, regions=regions,
                             D=D), f"ladder rung 0, period {period}")

    if a.random:
        rng = np.random.default_rng(0)
        for i in range(a.random):
            x = rng.random(N_PARAM)
            p = decode(x, a.rung)
            score(make_drive(c, x, a.nsteps, dt, rung=a.rung, seed=i, tapers=tap,
                             regions=regions, D=D),
                  f"rung {a.rung} #{i}: period {p['period']:.0f} kappa {p['kappa']:.2f} "
                  f"p_sat {p['p_sat']:.2f}")


if __name__ == "__main__":
    main()
