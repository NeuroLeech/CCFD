"""Fluid parameters as searchable fields: wave speed, damping, rotation.

The inverse solve puts a ceiling of ~0.30 on this objective for any drive whatsoever,
with the fluid frozen at Ld 52.4, unit wave speed and damping only at the medial wall.
The limit is that responses ring for the whole run, so frames cannot be set
independently. Damping is the direct control on that, and wave speed sets how far each
injection travels before it does.

Both are fields, not scalars, and the solver already takes them that way: h is advanced
by dt * H * div(u) with H per vertex, so c(x) = sqrt(g H(x)); sig_v damps h per vertex
and sig_e damps velocity per edge. Regions therefore get their own speed and damping
here, over four groups - the two poles of the dipole, the sensory satellites, and the
rest of the sheet - each smoothed across the mesh so the medium has no cliffs in it.

  from fluid import decode, build
  s, dt, g, Hf = build(cortex, decode(x))
"""
import numpy as np

from swe_rot import RotSWE, sponge_profile
from genome import SPONGE_STRENGTH_FIXED, SPONGE_WIDTH_FIXED
import ladder

N_PARAM = 11                       # legacy group mode
CFL = 0.347
GROUPS = ("core+", "DMN", "sensory", "rest")

MAPS_DEFAULT = ("myelin", "thickness", "sulc")
COEF_LIM = 0.45                    # ln units per map sd: +-0.45 is about x0.6 to x1.6
                                   # per sd, so x0.25 to x4 across a +-3 sd map


def n_param_maps(maps=MAPS_DEFAULT):
    return 3 + 2 * len(maps)        # c0, Ld, sig0, then a coefficient per map, twice


def decode_maps(x, maps=MAPS_DEFAULT):
    """[0,1]^(3+2m) -> parameters for a medium graded by cortical maps.

    Speed and damping are log-linear in the z-scored maps,
        c(x)   = c0   * exp(sum_j a_j m_j(x))
        sig(x) = sig0 * exp(sum_j b_j m_j(x))
    so the medium varies smoothly over the whole sheet with cortical structure rather
    than jumping at the boundaries of whichever regions we happen to drive."""
    x = np.clip(np.asarray(x, float), 0.0, 1.0)
    m = len(maps)
    return dict(mode="maps", maps=tuple(maps),
                c0=10.0 ** (-0.5 + 1.0 * x[0]),
                Ld=10.0 ** (1.0 + 1.5 * x[1]),
                sig0=10.0 ** (-4.0 + 3.0 * x[2]),
                a=(2 * x[3:3 + m] - 1) * COEF_LIM,
                b=(2 * x[3 + m:3 + 2 * m] - 1) * COEF_LIM)


def map_fields(cortex, p):
    """-> (wave speed, damping) per vertex, from the map-graded parameters."""
    from cortical_maps import load_maps
    mp = load_maps(cortex, p["maps"], verbose=False)
    M = np.stack([mp[k] for k in p["maps"]])
    c = p["c0"] * np.exp(np.asarray(p["a"]) @ M)
    sig = p["sig0"] * np.exp(np.asarray(p["b"]) @ M)
    return c, sig


def n_param(mode="group", maps=MAPS_DEFAULT):
    return N_PARAM if mode == "group" else n_param_maps(maps)


def decode_mode(x, mode="group", maps=MAPS_DEFAULT):
    return decode(x) if mode == "group" else decode_maps(x, maps)


def decode(x):
    """[0,1]^11 -> fluid parameters. x[0] speed, x[1] rotation, x[2:6] speed by group,
    x[6] damping, x[7:11] damping by group."""
    x = np.clip(np.asarray(x, float), 0.0, 1.0)
    return dict(
        c0=10.0 ** (-0.5 + 1.0 * x[0]),                  # 0.32 - 3.2
        Ld=10.0 ** (1.0 + 1.5 * x[1]),                   # 10 - 316 mm
        c_group=10.0 ** (-0.6 + 1.2 * x[2:6]),           # x0.25 - x4 per group
        sig0=10.0 ** (-4.0 + 3.0 * x[6]),                # 1e-4 - 1e-1
        sig_group=10.0 ** (-0.6 + 1.2 * x[7:11]),
    )


def group_field(cortex, values, smooth=12):
    """Per-vertex field from per-group values, smoothed so the medium has no cliffs."""
    lab = np.asarray(cortex.lab)
    gid = np.full(cortex.nV, 3)
    for g, parcels in enumerate((ladder.CORE_POS, ladder.CORE_NEG, ladder.SATELLITES)):
        gid[np.isin(lab, parcels)] = g
    f = np.asarray(values, float)[gid]

    E = cortex.edges
    deg = np.bincount(E.ravel(), minlength=cortex.nV).astype(float)
    deg[deg == 0] = 1.0
    for _ in range(smooth):                              # Jacobi smoothing on the graph
        acc = np.zeros(cortex.nV)
        np.add.at(acc, E[:, 0], f[E[:, 1]])
        np.add.at(acc, E[:, 1], f[E[:, 0]])
        f = 0.5 * f + 0.5 * acc / deg
    return f


def fields(cortex, p):
    """-> (wave speed, damping) per vertex, whichever parameterisation p came from."""
    if p.get("mode") == "maps":
        return map_fields(cortex, p)
    return (p["c0"] * group_field(cortex, p["c_group"]),
            p["sig0"] * group_field(cortex, p["sig_group"]))


def build(cortex, p, sponge=True, dt=None, coupling=None):
    """-> (solver, dt, g, H field). Wave speed is sqrt(g*H), so a per-vertex H is a
    per-vertex speed; dt follows the CFL condition at the fastest point on the sheet.

    `dt` overrides that. A run whose medium switches between regimes has to integrate all
    of them on one clock, and that clock must be set by the FASTEST regime - a dt chosen
    for a slow medium violates the CFL bound the moment a faster one is entered. Pass the
    common step, and each regime is then simply running below its own limit."""
    c, damp0 = fields(cortex, p)
    Hf = (c ** 2).astype(np.float32)                     # g = 1, so H = c^2
    dt = CFL * cortex.d.min() / float(c.max()) if dt is None else float(dt)

    s = RotSWE(cortex.m, 1.0 / p["Ld"], l=cortex.l, d=cortex.d, A=cortex.A,
               E=cortex.edges, bnd_edge=cortex.bnd)
    damp = damp0
    if sponge:
        # sponge strength is absolute, so its per-step effect dt*sig scales with 1/c;
        # sponge_scale lets it be held fixed in per-step terms when c is varied
        damp = damp + p.get("sponge_scale", 1.0) * sponge_profile(
            cortex.V, cortex.edges, cortex.bnd, SPONGE_WIDTH_FIXED,
            SPONGE_STRENGTH_FIXED)
    s.set_sponge(damp)
    s.astype(np.float32)
    s.sig_v = s.sig_v.astype(np.float32)
    s.sig_e = s.sig_e.astype(np.float32)
    s.coupling = coupling               # long-range term, applied inside RotSWE.step
    return s, dt, np.float32(1.0), Hf


def run(cortex, drive, p, nsteps, save_every=25, sponge=True, dt=None, coupling=None):
    """Integrate with spatially varying speed and damping. -> (frames, dt)."""
    s, dt, g, Hf = build(cortex, p, sponge, dt, coupling)
    Aser = drive.Aser.astype(np.float32)
    P = drive.P.astype(np.float32)
    h = np.zeros(cortex.nV, np.float32)
    ue = np.zeros(s.nE, np.float32)
    dtD = np.float32(dt)
    frames = []
    for n in range(nsteps):
        h += Aser[n] @ P
        ue, h = s.step(ue, h, dtD, g, Hf)
        if n % save_every == 0:
            if not np.isfinite(h).all():
                raise FloatingPointError(f"diverged at step {n}")
            frames.append(h.copy())
    return np.asarray(frames), dt


if __name__ == "__main__":
    import time
    from mesh_cache import load_cortex
    from input2 import parcel_tapers
    from fc_score import FCTarget
    from fc_moran import MoranMatch

    c = load_cortex("fsaverage5", verbose=False)
    t = FCTarget(c, verbose=False); mm = MoranMatch(c, t)
    tap = parcel_tapers(c, verbose=False)
    regions = ladder.CORE_POS + ladder.CORE_NEG + ladder.SATELLITES
    D, _ = ladder.parcel_geodesic(c, regions, verbose=False)
    xl = ladder.x0_rung0()

    rng = np.random.default_rng(0)
    print("ladder rung-0 drive, fluid parameters varied:")
    for tag, xf in (("frozen-equivalent", np.array([0.5, 0.554, .5, .5, .5, .5,
                                                    0.0, .5, .5, .5, .5])),
                    *[(f"random {i}", rng.random(N_PARAM)) for i in range(6)]):
        p = decode(xf)
        dr = ladder.make_drive(c, xl, 7000, CFL * c.d.min() / float(
            (p["c0"] * group_field(c, p["c_group"])).max()), tapers=tap,
            regions=regions, D=D)
        t0 = time.time()
        try:
            fr, dt = run(c, dr, p, 7000)
        except FloatingPointError as e:
            print(f"  {tag:18s} diverged"); continue
        Z, _ = t.model_z(fr)
        sim = float(t._prep(t.model_edges(Z=Z)[0]) @ t.y)
        print(f"  {tag:18s} c0 {p['c0']:.2f} Ld {p['Ld']:6.1f} sig0 {p['sig0']:.1e}  "
              f"sim {sim:+.4f}  gap {mm.gap(Z):.3f}  [{time.time()-t0:.0f}s]")
