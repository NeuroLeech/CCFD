"""A medium that switches between regimes, slowly, inside one run.

Static FC is a second moment, and for a LINEAR TIME-INVARIANT medium that is why every
temporal property of the input is invisible to it: the whole dependence collapses to
C = sum_f H(f) S(f) H(f)^H, so waveform, burstiness and ordering have nothing to act on.
The escape is to make the medium itself time-varying, which is no longer LTI.

This module takes the slow limit of that: R media, switched at epoch boundaries long
compared with the field's own decay. Each regime has its own transfer function, so

    C = sum_r occ_r sum_f w_f 2 Re( H_r(f) S_r(f) H_r(f)^H )

which is a strictly larger model class than any single-H solve can reach - and is still
convex in {S_r}, so the existing projected gradient carries over unchanged. Stacking the
regimes along the region axis, each scaled by sqrt(occ_r), turns it back into exactly the
one-block expression with a block-diagonal constraint; xspec._project does the rest.

What differs between regimes is the MAP GRADING - the a and b coefficients that make speed
and damping log-linear in myelin, thickness and sulcal depth - so the regimes differ in the
spatial PATTERN of the medium rather than its scale. A global speed scalar would be close
to a non-result: bo_step found c0 inert in per-step units, because dt follows c and only
per-step travel distance matters. Each regime's peak speed is renormalised to the base
medium's for the same reason, so that the common dt, and with it every per-step quantity
the medium was tuned in, is the one the incumbent was tuned at. See regime_set.

  python regimes.py --check                 # R=1 must reproduce the single-medium path
"""
import argparse
import numpy as np

import fluid as fl
import xspec

EPOCH_FRAMES = 280               # ~6 field decay times at the incumbent damping


def regime_set(cortex, base_p, deltas, renorm_speed=True):
    """R media from one base, each with its own map coefficients and damping.

    `deltas` is a list of dicts, one per regime, with any of:
      da    (len-3) added to the speed coefficients a
      db    (len-3) added to the damping coefficients b
      sig   multiplier on the global damping sig0
      c     multiplier on the global speed c0
    An empty dict reproduces the base medium, so deltas=[{}] is the single-medium case.

    `renorm_speed` rescales each regime's c0 so that its FASTEST point matches the base
    medium's, and it is on by default because without it the whole per-step
    parameterisation quietly breaks. dt is CFL*d_min/max(c), so a regime whose speed map
    peaks higher drags the common dt down for everybody - by a factor of 4.6 at span
    0.30 - and since the medium is specified in per-step units (damping per step,
    rotation per step), a smaller dt divides all of them by that same factor. The
    regimes would then differ from the incumbent medium in every parameter at once
    rather than in the one thing being varied.

    Renormalising leaves the regimes differing in the spatial PATTERN of speed - which
    is what the map coefficients are for - while the peak speed, and so dt and every
    per-step quantity, stays exactly the base medium's."""
    out = []
    c_base = float(fl.fields(cortex, base_p)[0].max())
    for d in deltas:
        p = dict(base_p)
        p["a"] = np.asarray(base_p.get("a", np.zeros(3)), float) + np.asarray(
            d.get("da", np.zeros(3)), float)
        p["b"] = np.asarray(base_p.get("b", np.zeros(3)), float) + np.asarray(
            d.get("db", np.zeros(3)), float)
        p["sig0"] = base_p["sig0"] * float(d.get("sig", 1.0))
        p["c0"] = base_p["c0"] * float(d.get("c", 1.0))
        if renorm_speed:
            p["c0"] *= c_base / float(fl.fields(cortex, p)[0].max())
        out.append(p)
    return out


def common_dt(cortex, ps):
    """The one timestep every regime is integrated at: the CFL bound of the FASTEST.

    A dt taken from a slower regime is unstable the moment a faster one is entered, and
    the failure is a blow-up mid-run rather than anything subtle."""
    return min(fl.CFL * cortex.d.min() / float(fl.fields(cortex, p)[0].max()) for p in ps)


def schedule(nframes, R, epoch_frames=EPOCH_FRAMES, occ=None, seed=0, order="cycle"):
    """-> (nframes,) regime index per saved frame.

    'cycle' walks the regimes in order, which makes occupancy exact and every regime's
    epochs evenly spread through the run - what the quasi-static approximation wants.
    'random' draws each epoch independently, which lets occupancy wander but does not
    impose a period on the switching."""
    n_ep = int(np.ceil(nframes / epoch_frames))
    if order == "cycle":
        idx = np.arange(n_ep) % R
    else:
        rng = np.random.default_rng(seed)
        p = None if occ is None else np.asarray(occ, float) / np.sum(occ)
        idx = rng.choice(R, n_ep, p=p)
    return np.repeat(idx, epoch_frames)[:nframes]


def occupancy(sched, R):
    return np.array([float((sched == r).mean()) for r in range(R)])


def transfer_stack(cortex, ps, cols, nfreq, pad, nsteps, save, profiles, occ, dt,
                   verbose=True, kernel=None):
    """-> (H stacked over regimes, weights, bin index, per-regime response length).

    Each regime's impulse responses are computed at the COMMON dt, then scaled by
    sqrt(occ_r) and concatenated along the region axis. The sqrt is what makes the
    stacked quadratic form reproduce sum_r occ_r H_r S_r H_r^H exactly."""
    Hs, idx, w, nfr = [], None, None, None
    for r, p in enumerate(ps):
        resp = xspec.impulse_responses(cortex, list(range(len(profiles))), p, nsteps,
                                       save, profiles=profiles, verbose=False, dt=dt)
        R_ = np.pad(resp, ((0, 0), (0, max(0, pad - resp.shape[1])), (0, 0)))
        H, w, idx = xspec.transfer(R_, cols, nfreq, kernel=kernel)
        nfr = R_.shape[1]
        Hs.append(np.sqrt(occ[r]) * H)
        if verbose:
            print(f"    regime {r}: peak response {np.abs(resp).max():.3g}, "
                  f"occupancy {occ[r]:.2f}", flush=True)
    return np.concatenate(Hs, axis=2), w, idx, nfr


def run_switching(cortex, drive, ps, sched_steps, nsteps, save_every, dt, sponge=True):
    """Integrate one run whose medium switches. -> (frames, dt).

    The field state h and the velocity ue are CARRIED ACROSS every switch. That is the
    whole content of the model: a mixture of independent runs would be a mixture of FCs,
    which is what the regime test already looked at and is not this. Only the operator
    changes underneath a continuous trajectory."""
    built = [fl.build(cortex, p, sponge, dt) for p in ps]
    Aser = drive.Aser.astype(np.float32)
    P = drive.P.astype(np.float32)
    h = np.zeros(cortex.nV, np.float32)
    ue = np.zeros(built[0][0].nE, np.float32)
    dtD = np.float32(dt)
    frames = []
    for n in range(nsteps):
        s, _, g, Hf = built[sched_steps[n]]
        h += Aser[n] @ P
        ue, h = s.step(ue, h, dtD, g, Hf)
        if n % save_every == 0:
            if not np.isfinite(h).all():
                raise FloatingPointError(f"diverged at step {n}, regime "
                                         f"{sched_steps[n]}")
            frames.append(h.copy())
    return np.asarray(frames), dt


def block_power(S, w, R):
    """Total input power the solve assigned to each regime block."""
    K = S.shape[1] // R
    return np.array([float(sum(w[f] * np.trace(S[f, r * K:(r + 1) * K,
                                                  r * K:(r + 1) * K]).real
                               for f in range(S.shape[0]))) for r in range(R)])


def epoch_profile(frames, sched, epoch_frames, R, verbose=True):
    """Is the quasi-static assumption actually holding?

    The solve treats the run as an occupancy-weighted mixture of per-regime STATIONARY
    covariances. That is only true if each epoch is long enough for the field to forget
    the previous regime. Two things say whether it is: the field variance per regime
    (which the solve's block powers are supposed to set), and the variance profile within
    an epoch (which should be flat - a large head means every switch is a transient and
    the run is mostly made of them)."""
    X = np.asarray(frames[:, ::7], np.float32)
    v = X.var(1)
    sched = sched[:len(v)]
    per = np.array([float(v[sched == r].mean()) for r in range(R)])
    ep = np.arange(len(v)) % epoch_frames
    prof = np.array([v[ep == k].mean() for k in range(epoch_frames)])
    head, tail = float(prof[:20].mean()), float(prof[-20:].mean())
    if verbose:
        print(f"  field variance per regime: {np.array2string(per, precision=4)}  "
              f"(max/min {per.max()/max(per.min(),1e-30):.2f}x)")
        print(f"  within-epoch variance: head {head:.4g}, tail {tail:.4g}, "
              f"ratio {head/max(tail,1e-30):.1f} "
              f"({'quasi-static' if head/max(tail,1e-30) < 2 else 'TRANSIENT-DOMINATED'})")
    return per, prof


def realise_switching(S, idx, nframes, ref_frames, R, sched, w=None, seed=0):
    """Draw a drive per regime and splice them onto the schedule.

    S is block diagonal, so block r is regime r's own input cross-spectrum. Each block is
    realised over the full run and then read only on the frames that regime occupies,
    which keeps every regime's drive stationary within its own epochs.

    The per-regime amplitudes have to be restored by hand. xspec.realise normalises what
    it returns to unit standard deviation, so realising each block separately throws away
    exactly the thing the solve decided - how much power to give each regime - and leaves
    every regime equally loud. Passing `w` rescales each by the square root of its solved
    block power, which puts that back."""
    K = S.shape[1] // R
    A = np.zeros((nframes, K))
    amp = np.ones(R) if w is None else np.sqrt(np.maximum(block_power(S, w, R), 0.0))
    if w is not None and amp.max() > 0:
        amp = amp / amp.max()
    for r in range(R):
        Sr = S[:, r * K:(r + 1) * K, r * K:(r + 1) * K]
        Ar = xspec.realise(Sr, idx, nframes, ref_frames=ref_frames, seed=seed + 100 * r)
        m = sched == r
        A[m] = amp[r] * Ar[m]
    sd = A.std()
    return A / (sd if sd > 1e-30 else 1.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true",
                    help="R=1 through this path must match the single-medium path")
    a = ap.parse_args()
    if not a.check:
        ap.print_help()
        return

    from mesh_cache import load_cortex
    import bo_step, subparcels
    from best_fit import BEST_X
    c = load_cortex("fsaverage5", verbose=False)
    base, save, _ = bo_step.unpack(BEST_X, c)
    ps = regime_set(c, base, [{}])
    dt = common_dt(c, ps)
    s0, dt0, _, _ = fl.build(c, base)
    print(f"  R=1: common dt {dt:.8g} vs single-medium dt {dt0:.8g}  "
          f"-> {'match' if abs(dt - dt0) < 1e-12 else 'DIFFER'}")

    labels, tags = subparcels.split_parcels(c, subparcels.SENSORY, 50, verbose=False)
    P = subparcels.taper_profiles(c, labels, len(tags))
    sched = schedule(1120, 1)
    print(f"  schedule: {len(sched)} frames, occupancy {occupancy(sched, 1)}")
    ps2 = regime_set(c, base, regime_deltas_demo())
    dt2 = common_dt(c, ps2)
    print(f"  R=3 spread: common dt {dt2:.8g} -> "
          f"{'unchanged' if abs(dt2 - dt0) < 1e-9 else 'DIFFERS from single medium'}")
    print(f"  regime_set([{{}}]) reproduces the base medium: "
          f"{np.allclose(ps[0]['a'], base['a']) and np.allclose(ps[0]['b'], base['b']) and ps[0]['sig0'] == base['sig0']}")


if __name__ == "__main__":
    main()
