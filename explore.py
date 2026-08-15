"""Run a genome through the fluid model and describe what came out.

There is deliberately no fitness function here. The point is to see the spread of
behaviour the input parameterisation actually produces before committing to a target.
The statistics recorded are descriptive - somewhere to hang a target later, not a
target themselves.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from mesh_cache import load_cortex
from input_model import parcel_tapers, NetworkDrive
from genome import random_genome, decode, validate
from swe_rot import RotSWE, sponge_profile
from wave_measures import WaveMeasures

# Fluid parameters are fixed for now. Listed here so they can be promoted into the
# genome later without restructuring anything.
FLUID = dict(c=1.0, H=1.0, g=1.0, Ld=25.0,
             sponge_width=25.0, sponge_strength=0.15,
             cfl=0.35, nsteps=3500, sample_every=10, n_probe=64,
             wave_every=50, exclude_mm=20.0, parcel_every=5, decimate=10)


class Rig:
    """Holds everything reusable across genomes: mesh, operators, tapers, solver."""

    def __init__(self, mesh="fsaverage5", fluid=None, dtype=np.float32):
        self.f = dict(FLUID, **(fluid or {}))
        self.c = load_cortex(mesh)
        self.tapers = parcel_tapers(self.c)
        c = self.c
        s = RotSWE(c.m, self.f["c"]/self.f["Ld"], l=c.l, d=c.d, A=c.A,
                   E=c.edges, bnd_edge=c.bnd)
        if self.f["sponge_strength"] > 0:
            s.set_sponge(sponge_profile(c.V, c.edges, c.bnd,
                                        self.f["sponge_width"], self.f["sponge_strength"]))
        self.dtype = dtype
        if dtype != np.float64:
            s.astype(dtype)
        self.s = s
        self.dt = float(self.f["cfl"]*c.d.min()/self.f["c"])
        rng = np.random.default_rng(0)
        self.probe = rng.choice(c.nV, self.f["n_probe"], replace=False)
        self.wm = WaveMeasures(c, exclude_mm=self.f["exclude_mm"])
        self.wm.baseline()          # mesh anisotropy floor, computed once
        # area-weighted parcel averaging, (nV, 180). Applying this per sampled step
        # keeps 180 numbers instead of a 9374-vector, so the whole parcel timeseries
        # costs a few hundred kB rather than ~100 MB of snapshots.
        Pm = np.zeros((c.nV, 180), np.float32)
        for pid in range(1, 181):
            msk = c.lab == pid
            if msk.any():
                w = c.A[msk]
                Pm[msk, pid-1] = (w/w.sum()).astype(np.float32)
        self.parcel_mat = Pm

    def duration(self):
        return self.f["nsteps"]*self.dt

    def apply_fluid(self, p):
        """Set the per-genome fluid parameters on the existing solver.

        Neither of these needs the operators rebuilt: Ld enters only through the
        per-vertex Coriolis parameter, and the sponge's only costly part is the
        geodesic distance to the medial wall, which depends on neither width nor
        strength. That distance is already computed once by WaveMeasures, so it
        is reused rather than run again. dt is untouched - the run length is
        deliberately the same for every genome.
        """
        s, D = self.s, self.dtype
        s.f[:] = D(self.f["c"]/p["Ld"])
        dist = self.wm.dist_to_edge
        u = np.clip(1.0 - dist/max(p["sponge_width"], 1e-9), 0.0, 1.0)
        s.set_sponge(p["sponge_strength"]*u*u*(3.0 - 2.0*u))
        s.sig_v = s.sig_v.astype(D)
        s.sig_e = s.sig_e.astype(D)


def _length_scale(c, h):
    g = c.m.tri_grad(np.asarray(h, np.float64))
    num = (c.A*h*h).sum()
    den = (c.m.tri_area*(g*g).sum(1)).sum()
    return float(np.sqrt(num/max(den, 1e-30)))


def _var_modulation(E, nwin=12):
    """Modulation depth of the field's spatial variance: standing or travelling.

    A standing pattern collapses everywhere at once, so its spatial variance
    oscillates; a travelling one carries its variance along and does not. On
    synthetic waves at a wavelength that fits the surface this reads 0.09
    (travelling) against 0.71 (standing), where direction-coherence managed only
    0.96 against 0.89.

    Computed in windows and medianed, because the total energy of a driven run
    grows by one to two orders of magnitude over its length and that trend would
    otherwise be indistinguishable from oscillation.
    """
    E = np.asarray(E, float)
    if len(E) < 16:
        return float("nan")
    w = max(8, len(E)//nwin)
    mods = [E[i:i+w].std()/max(E[i:i+w].mean(), 1e-30)
            for i in range(0, len(E)-w+1, w)]
    return float(np.median(mods)) if mods else float("nan")


def _decorr_time(x, dt):
    """1/e decorrelation time of a series, in time units."""
    x = x - x.mean()
    n = len(x)
    if n < 8 or x.std() < 1e-20:
        return float("nan")
    ac = np.correlate(x, x, mode="full")[n-1:]
    ac /= ac[0]
    below = np.flatnonzero(ac < np.exp(-1.0))
    return float(below[0]*dt) if len(below) else float(n*dt)


def run_genome(g, rig, seed=0, keep_snapshots=False, params=None, keep_field=False):
    """Run one genome. `params` bypasses decode() with an explicit parameter dict
    of the same shape, which is how manual exploration (run_input.py) drives the
    fluid genes the genome itself holds frozen. `keep_field` additionally returns
    the sampled vertex x time field, for rendering."""
    p = decode(g) if params is None else params
    rig.apply_fluid(p)
    c, s, f, dt = rig.c, rig.s, rig.f, rig.dt
    D = rig.dtype
    # temporal balance, matching the sweep the frozen regime was chosen from.
    # Under spatial balance the drive can only move depth between the named
    # regions; under temporal balance it injects nothing across the run as a
    # whole but the surface does rise and fall at any instant.
    drive = NetworkDrive(c, p["regions"], p["loadings"], p["silents"], p["taus"],
                         p["amp"], f["nsteps"], dt, seed=seed, tapers=rig.tapers,
                         balance=p.get("balance", "temporal"))
    # Build the source per step rather than materialising (nsteps, nV): at 14000
    # steps that array is 525 MB, which is what made long runs scale worse than
    # linearly. Per step it is a (K,) x (K, nV) product - a few tens of kflops.
    Aser = drive.Aser.astype(D); Pmat = drive.P.astype(D)

    h = np.zeros(c.nV, D); ue = np.zeros(s.nE, D)
    dtD, gD, HD = D(dt), D(f["g"]), D(f["H"])
    probe_ts, snaps, energy, wave_frames, parcel_ts = [], [], [], [], []
    t0 = time.time()
    for n in range(f["nsteps"]):
        h += Aser[n] @ Pmat
        ue, h = s.step(ue, h, dtD, gD, HD)
        if n % f["sample_every"] == 0:
            if not np.isfinite(h).all():          # checking every step cost ~40% of
                return dict(ok=False,             # the runtime for no extra safety
                            reason=f"diverged by step {n}", genome=g)
            probe_ts.append(h[rig.probe].copy())
            energy.append(float(0.5*(c.A*h.astype(np.float64)**2).sum()))
        if n % f["wave_every"] == 0:
            wave_frames.append(h.astype(np.float64).copy())
        if n % f["parcel_every"] == 0:
            parcel_ts.append(h @ rig.parcel_mat)
        if keep_snapshots and n % (f["nsteps"]//6) == 0:
            snaps.append(h.astype(np.float64).copy())
    elapsed = time.time()-t0

    probe_ts = np.array(probe_ts)                                   # (nsample, n_probe)
    hf = h.astype(np.float64)
    dts = [_decorr_time(probe_ts[:, j], dt*f["sample_every"])
           for j in range(probe_ts.shape[1])]
    di = drive.report()
    wv = rig.wm.summarise(np.array(wave_frames), dt*f["wave_every"])
    peak = float(np.abs(hf).max()/f["H"])

    # Fast dynamics: the field's own decorrelation time against the DRIVE's,
    # both from the same estimator so the ratio is meaningful. Measured from the
    # realised drive series rather than the tau genes, because sparsity and
    # thresholding change the effective timescale away from the nominal tau.
    # < 1 means the fluid reorganises faster than it is being told to.
    drive_dts = [_decorr_time(np.asarray(drive.Aser[:, k], np.float64), dt)
                 for k in range(drive.Aser.shape[1])]
    drive_tau = float(np.nanmedian(drive_dts))
    field_tau = float(np.nanmedian(dts))
    tau_ratio = field_tau/drive_tau if drive_tau > 1e-12 else float("nan")
    out = dict(
        ok=True, seconds=elapsed,
        # the model linearises about a flat layer, so a peak that is a large
        # fraction of the layer depth is outside its own assumptions. Recorded
        # rather than rejected, so a fitness function can penalise it explicitly.
        linear_ok=bool(peak < 0.10),
        peak_frac=peak,
        rms_frac=float(np.sqrt((c.A*hf*hf).sum()/c.A.sum())/f["H"]),
        length_scale=_length_scale(c, hf),
        decorr_time=field_tau, drive_tau=drive_tau, tau_ratio=tau_ratio,
        var_mod=_var_modulation(energy),
        energy_final=energy[-1], energy_mean=float(np.mean(energy)),
        energy_drift=float(energy[-1]/max(np.mean(energy[:len(energy)//4]), 1e-30)),
        active_frac=float((np.abs(hf) > 0.2*np.abs(hf).max()).mean()),
        parcels=np.array(parcel_ts, np.float32).T,      # (180, nframes)
        anisotropy=wv["anisotropy"], wave_speed=wv["speed"],
        pattern_tau=wv["pattern_tau"], orient_coherence=wv["orient_coherence"],
        dir_coherence=wv["dir_coherence"],
        corr_agreement=di["corr_agreement"], silent=di["silent"],
        mass_residual=di["mass_residual"], struct_rms=drive.struct_rms,
        regions=[c.names[int(p_)] for p_ in p["regions"]],
        amp=p["amp"], silents=p["silents"], taus=p["taus"],
        Ld=p["Ld"], sponge_strength=p["sponge_strength"],
        sponge_width=p["sponge_width"],
        genome=g,
    )
    if keep_snapshots:
        out["snapshots"] = snaps
    if keep_field:
        # the wave-measure frames, reused rather than sampled a second time
        out["frames"] = np.array(wave_frames, np.float32)
        out["dt_frame"] = dt*f["wave_every"]
        out["drive_series"] = drive.Aser[::f["wave_every"]].astype(np.float32)
    return out


def sample_random_genomes(n, K, r, rig, rng):
    out = []
    while len(out) < n:
        g = random_genome(rng, K, r, rig.c.parcels)
        if not validate(g, rig.c.parcels):
            out.append(g)
    return out


if __name__ == "__main__":
    rig = Rig("fsaverage5")
    print(f"{rig.c}\n  dt={rig.dt:.4f}, {rig.f['nsteps']} steps "
          f"= {rig.duration():.0f} time units (wave crossing ~200)\n")
    rng = np.random.default_rng(1)
    N, r = 20, 3
    hdr = (f"{'K':>3} {'run':>4} {'peak%':>7} {'Lscale':>7} {'anis':>6} {'speed':>7} "
           f"{'patTau':>7} {'decorr':>7} {'sec':>5}")
    for K in (4, 8, 16):
        print(f"--- K = {K} ---"); print(hdr)
        rows = []
        for i, g in enumerate(sample_random_genomes(N, K, r, rig, rng)):
            res = run_genome(g, rig, seed=i)
            if not res["ok"]:
                print(f"{K:3d} {i:4d}   {res['reason']}"); continue
            rows.append(res)
            print(f"{K:3d} {i:4d} {100*res['peak_frac']:7.2f} {res['length_scale']:7.2f} "
                  f"{res['anisotropy']:6.3f} {res['wave_speed']:7.3f} "
                  f"{res['pattern_tau']:7.1f} {res['decorr_time']:7.1f} {res['seconds']:5.1f}")
        nbad = sum(1 for r_ in rows if not r_["linear_ok"])
        print(f"    outside the linear regime (peak > 10% of depth): {nbad}/{len(rows)}")
        if rows:
            import statistics as st
            for key, lab in (("anisotropy", "anisotropy"), ("wave_speed", "wave speed"),
                             ("pattern_tau", "pattern tau"), ("length_scale", "length scale mm")):
                v = [r_[key] for r_ in rows if np.isfinite(r_[key])]
                print(f"    {lab:16s} min {min(v):8.2f}  median {st.median(v):8.2f}  max {max(v):8.2f}")
        print()
