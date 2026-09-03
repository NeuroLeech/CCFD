"""Reproduce the current best fit: 47 sensory pieces, solved cross-spectrum, one number.

Everything the result depends on is here rather than spread across scratch scripts:
the medium (per-step units, from the bo_step search), the region split, the zero-padded
transfer function, the rank-transformed target, the realisation and the score.

  python best_fit.py                          # the current best configuration
  python best_fit.py --frames 2240 --draws 3  # longer realisation
  python best_fit.py --rank-iters 4           # iterate the solve on the model's ranks
"""
import os, sys, time, argparse
import numpy as np

from mesh_cache import load_cortex
import fc_score
from fc_moran import MoranMatch
from paths import RESULTS
import xspec, bo_step, subparcels, regimes, units

# the bo_step winner, in per-step units: damping, rotation, boundary absorption, cadence,
# then the six speed/damping map coefficients
BEST_X = np.array([np.log10(6.2e-4), np.log10(1.1e-5), np.log10(1.8e-3), np.log10(33),
                   -0.30, -0.05, 0.01, -0.03, 0.35, 0.35])
PAD, NFREQ, NVERT = 1120, 192, 1000

# The best sensory-only configuration, pinned rather than reconstructed from a shell
# history. Everything here is an OVERRIDE of BEST_X or of an argparse default, because
# the bo_step search that produced BEST_X predates both the damping sweep and the
# whitened solve and has not been re-run against them.
#
# The select grid runs past 400 deliberately. Before whitening the realised score peaked
# around 50 and fell away, so the grid stopped at 400; WITH whitening the curve is not an
# inverted U at all - 10/25/50/100/200/400 gives .5411/.5840/.5905/.5898/.5870/.5987, and
# the pick landed on the top of the grid with the peak possibly beyond it.
#
# impulse_frames is not cosmetic. The field's decay time is 1/(damping per step * save),
# which at 2e-4 and save=33 is ~152 frames, and an impulse window shorter than that
# truncates the response - so H would describe a system nobody simulates. 560 is the
# window diag_residual.py and solver_test.py already use for this damping.
BEST = dict(damp=2e-4, whiten=1e-3, impulse_frames=560, regions="sensory", split=50,
            select_iters="25,50,100,200,400,800,1600", frames=4480, draws=3)


normal_scores = xspec.normal_scores          # lives in xspec so bo_step can use it too
region_set = subparcels.region_set           # lives in subparcels for the same reason


def quantile_match(target_edges, model_edges):
    """Re-express the target in the model's own value scale: rank the target, then read
    off the model's quantile at that rank. Pearson against this tracks Spearman against
    the raw target far more closely, which is what the score actually measures."""
    order = np.argsort(np.argsort(target_edges))
    return np.sort(model_edges)[order]


def regime_deltas(R, span, which="sulc", target="speed", base=None, verbose=True):
    """R regimes differing along ONE map, in ONE of speed or damping.

    Deliberately minimal. The first version moved two maps at once on both speed and
    damping and also scaled global damping, which made every regime differ from the base
    in four ways and left nothing to attribute a result to. Here a regime is the base
    medium with a single coefficient shifted, so R media lie on one line through
    parameter space and the only question is what moving along that line does.

    Offsets are clipped to fluid.COEF_LIM. The base medium sits inside the box bo_step
    searched, and span 0.30 on a coefficient already at -0.30 would put one regime at
    -0.60 - a medium never validated as sensible, and outside the range the incumbent was
    tuned in."""
    import fluid as fl
    if R == 1:
        return [{}]
    j = list(fl.MAPS_DEFAULT).index(which)
    key = "da" if target == "speed" else "db"
    b0 = np.asarray((base or {}).get("a" if target == "speed" else "b", np.zeros(3)), float)
    out, clipped = [], False
    for v in np.linspace(-1.0, 1.0, R):
        off = np.zeros(3)
        want = b0[j] + span * v
        got = float(np.clip(want, -fl.COEF_LIM, fl.COEF_LIM))
        clipped |= abs(got - want) > 1e-12
        off[j] = got - b0[j]
        out.append({key: off})
    if verbose:
        coefs = [b0[j] + d[key][j] for d in out]
        print(f"  regimes vary {target} with {which}: coefficient "
              f"{np.array2string(np.array(coefs), precision=3)} "
              f"(base {b0[j]:+.3f}, limit +-{fl.COEF_LIM})"
              + ("  [CLIPPED]" if clipped else ""))
    return out


def held_out_score(t, frames, val, val_raw_ranks):
    """Spearman of the SIMULATED FC against the target, on vertices the solve never saw.

    Centred the same way the reported score is. The centring is a FULL-matrix operation -
    row means over all 9,217 vertices - so it cannot be done from the held-out block
    alone; the means come from the whole Z, exactly as fc_score.model_edges does it.
    Scoring an un-centred block against a centred target would make the selection
    criterion a different quantity from the thing being selected for."""
    from scipy.stats import rankdata
    Z, _ = t.model_z(frames)
    T, V = Z.shape[1], Z.shape[0]
    Zv = Z[val]
    F = (Zv @ Zv.T) / T
    if t.centre == "double":
        ssum = Z.sum(0)
        diag = (Z * Z).sum(1) / T
        m = ((Z @ ssum) / T - diag) / (V - 1)
        grand = (float(ssum @ ssum) / T - float(diag.sum())) / (V * (V - 1))
        F = F - m[val][:, None] - m[val][None, :] + grand
    iu = np.triu_indices(len(val), 1)
    r = rankdata(F[iu])
    r = (r - r.mean()) / max(r.std(), 1e-30)
    return float(r @ val_raw_ranks / len(r))


def select_iters(a, c, t, p, P, H, w, idx, ref_frames, save, val, val_Ct, Tgt, nb,
                 sched_f, ps, dt, kern, cpl, Lw=None, keep_f=None, spec=None,
                 band=None, frame_s=None):
    """Choose where to stop the solve, by simulating at each candidate and scoring on
    held-out vertices.

    This is more expensive than a criterion evaluated inside the solve, and it is what the
    measurements say is needed. The solve objective rises monotonically for at least 4,000
    steps; the covariance match on held-out vertices peaks around 130; the realised score
    peaks around 25. Two effects compound - the predicted covariance overfits the solve
    vertices, AND the simulation reproduces its own prediction less faithfully as the
    input collapses to fewer modes (fidelity 0.945 down to 0.927 across the sweep). Only a
    realised, held-out number sees both, so only that can be used to pick between
    configurations."""
    from scipy.stats import rankdata
    iu = np.triu_indices(len(val), 1)
    raw_v = np.asarray(val_Ct)[iu]
    yv = rankdata(raw_v)
    yv = (yv - yv.mean()) / max(yv.std(), 1e-30)
    cands = sorted(int(x) for x in a.select_iters.split(",") if x.strip())
    print(f"  selecting the stopping point over {cands}, by realised score on "
          f"{len(val)} held-out vertices at {a.select_frames} frames:")
    scores = {}
    for n in cands:
        S, _ = xspec.solve(H, w, Tgt, iters=n, verbose=False, nblock=nb,
                           share=a.share_input, freq_keep=keep_f, spec=spec)
        if Lw is not None:
            S = xspec.unwhiten(S, Lw)
        if nb == 1:
            A = xspec.realise(S, idx, a.select_frames, ref_frames=ref_frames, seed=7)
            run_fn = None
        else:
            sf = sched_f[:a.select_frames]
            A = regimes.realise_switching(S, idx, a.select_frames, ref_frames, nb, sf,
                                          w=w, seed=7)
            steps = np.repeat(sf, save)
            run_fn = (lambda dr, ns, sv: regimes.run_switching(
                c, dr, ps, steps[:ns], ns, sv, dt))
        if cpl is not None and run_fn is None:
            run_fn = lambda dr, ns, sv: bo_step.fl.run(c, dr, p, ns, sv, coupling=cpl)
        r = xspec.score_realisation(c, t, p, A, save=save, profiles=P, run_fn=run_fn,
                                    kernel=kern, band=band, frame_s=frame_s)
        v = held_out_score(t, r["frames"], val, yv)
        scores[n] = v
        print(f"    {n:5d} iterations: held-out realised {v:+.4f}  "
              f"(rank {r['rank']:.1f})", flush=True)
    best = max(scores.values())
    best_n = min(n for n in cands if scores[n] >= best - a.select_tol)
    note = "" if scores[best_n] == best else (
        f" (cheapest within {a.select_tol} of {max(scores, key=scores.get)}, "
        f"which scored {best:+.4f})")
    print(f"  -> stopping at {best_n} iterations{note}")
    if best_n == cands[-1]:
        print(f"     WARNING: that is the top of the grid, so the peak may lie beyond it")
    return best_n


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--frames", type=int, default=1120, help="realisation length")
    ap.add_argument("--draws", type=int, default=2)
    ap.add_argument("--iters", type=int, default=150, help="solver iterations")
    ap.add_argument("--rank-iters", type=int, default=0, dest="rank_iters",
                    help="fixed-point iterations matching the model's edge distribution")
    ap.add_argument("--nfreq", type=int, default=NFREQ)
    ap.add_argument("--pad", type=int, default=PAD)
    ap.add_argument("--nvert", type=int, default=NVERT)
    ap.add_argument("--select-iters", default="", dest="select_iters",
                    help="comma-separated iteration counts to choose between, by "
                         "REALISED score on held-out vertices (e.g. 10,25,50,100,200). "
                         "The stopping point is a hidden regularisation parameter whose "
                         "best value differs per configuration, and neither the solve "
                         "objective nor a held-out covariance match locates it.")
    ap.add_argument("--select-frames", type=int, default=1120, dest="select_frames",
                    help="realisation length used for selection only")
    ap.add_argument("--select-tol", type=float, default=0.005, dest="select_tol",
                    help="take the CHEAPEST candidate within this of the best, rather "
                         "than the argmax: selection uses one short draw, so differences "
                         "this size are noise, and paying 4x the solve for them is waste")
    ap.add_argument("--val-vert", type=int, default=1000, dest="val_vert",
                    help="held-out vertices for early stopping (0 = fixed iteration "
                         "count, which is a hidden regularisation parameter)")
    ap.add_argument("--whiten", type=float, default=0.0,
                    help="change of variables conditioning the solve (try 1e-3); "
                         "0 = off. See xspec.whiten")
    ap.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 1),
                    help="processes for the impulse responses and for the extra draws; "
                         "both are independent per piece / per draw, so this is near-"
                         "linear. 1 (or 0) = serial. Impulses that hit the cache never "
                         "reach the pool, so the default costs nothing on a rerun")
    ap.add_argument("--damp", type=float, default=None,
                    help="override interior damping per step (BEST_X carries 6.2e-04)")
    ap.add_argument("--impulse-frames", type=int, default=280, dest="impulse_frames",
                    help="impulse window; must exceed the field's decay time, which is "
                         "1/(damping per step). At 6.2e-04 that is ~49 frames; lower "
                         "damping rings longer and needs a longer window")
    ap.add_argument("--split", type=int, default=50,
                    help="pieces to divide the driven parcels into (sets the piece area)")
    ap.add_argument("--regions", default="sensory",
                    choices=("sensory", "dmn", "sensory+dmn", "spread",
                             "subcortical", "subcortical+sensory"),
                    help="which parcels are driven; 'spread' is an even whole-cortex "
                         "sample matched to the sensory driven area")
    ap.add_argument("--coupling", type=float, default=0.0,
                    help="long-range structural coupling strength (0 = off); "
                         "dimensionless, see connectome.load_enigma")
    ap.add_argument("--coupling-mm", type=float, default=60.0, dest="coupling_mm",
                    help="shortest connection counted as long-range")
    ap.add_argument("--coupling-keep", type=float, default=0.15, dest="coupling_keep",
                    help="fraction of long connections kept, by distance residual")
    ap.add_argument("--coupling-lag", type=int, default=0, dest="coupling_lag",
                    help="uniform delay on the long-range term, in STEPS. Delaying a "
                         "linear term keeps the system LTI, so the convex solve is "
                         "unchanged and only H moves. One model step is ~0.2 ms by the "
                         "units.py calibration, so a physiological 10-30 ms delay is "
                         "only 50-150 steps")
    ap.add_argument("--coupling-surrogate", type=int, default=None, metavar="SEED",
                    dest="coupling_surrogate",
                    help="rewire the connectome preserving degree and edge-length "
                         "distribution: the control that separates 'this topology helps' "
                         "from 'any long-range redistribution helps'")
    ap.add_argument("--coupling-raw", action="store_true", dest="coupling_raw",
                    help="use the whole normalised connectome instead of the "
                         "distance-residual long-range filter")
    ap.add_argument("--smooth", type=float, default=0.0,
                    help="temporal FWHM in frames of the filter between field and "
                         "observable (0 = none); units.py says the model frame is "
                         "~1/100 of a TR, so this is where that gap would be closed")
    ap.add_argument("--regimes", type=int, default=1,
                    help="media switched within the run (1 = the single-medium path)")
    ap.add_argument("--epoch", type=int, default=regimes.EPOCH_FRAMES,
                    help="frames per regime epoch; must exceed the field's decay time")
    ap.add_argument("--regime-span", type=float, default=0.30, dest="regime_span",
                    help="spread of the per-regime map coefficient, in ln units")
    ap.add_argument("--regime-map", default="sulc", dest="regime_map",
                    choices=("myelin", "thickness", "sulc"),
                    help="which cortical map the regimes vary along")
    ap.add_argument("--regime-target", default="speed", dest="regime_target",
                    choices=("speed", "damp"), help="whether that map grades speed or damping")
    ap.add_argument("--share-input", action="store_true", dest="share_input",
                    help="constrain every regime to the same input cross-spectrum")
    ap.add_argument("--spread-scale", type=float, default=1.0, dest="spread_scale",
                    help="multiply the 'spread' area budget (1.0 = the sensory area)")
    ap.add_argument("--centre", default="double", choices=("double", "none"),
                    help="'double' centres the target AND the model the same way; "
                         "'none' is the old behaviour, a pre-centred target scored "
                         "against an un-centred model")
    ap.add_argument("--target", default="normal",
                    choices=("normal", "raw"), help="what the solve matches")
    ap.add_argument("--oversample", type=int, default=0, metavar="K",
                    help="put the model on the fMRI clock with frames at TR/K, so it is "
                         "sampled faster than BOLD. Sets save and per-step damping "
                         "together so spread in mm/s and decay in seconds are unchanged, "
                         "and sizes the impulse window to the resulting decay. See "
                         "timescale.py for why the previous anchor put the whole model "
                         "above the resting-state band")
    ap.add_argument("--decay-s", type=float, default=None, dest="decay_s",
                    help="field decay time in SECONDS (default: the measured BOLD 1/e "
                         "time, 9.03 s). Pass 0 to carry the unanchored medium's 97.7 s")
    ap.add_argument("--spread-mm-s", type=float, default=None, dest="spread_mm_s",
                    help="wave spread in mm/s; sets `save`. With the decay pinned by the "
                         "data this is the one free parameter, and reach = spread x decay")
    ap.add_argument("--seconds", type=float, default=577.0,
                    help="realisation length in SECONDS when --oversample is used "
                         "(577 s = the 895-TR NKI run)")
    ap.add_argument("--band", default="",
                    help="restrict the input cross-spectrum to LO,HI in Hz "
                         "(e.g. 0.01,0.1). Needs --oversample to have a clock")
    ap.add_argument("--bandpass", default="",
                    help="passband the TARGET was filtered to, LO,HI in Hz (e.g. "
                         "0.01,0.08 for any RBC/XCP-D target). Multiplies H by the "
                         "filter's |H|^2 and applies the same filter to the realised "
                         "frames, so the model is solved and scored on what survives "
                         "the filter instead of on power the target cannot contain. "
                         "Needs --oversample for a clock. Distinct from --band, which "
                         "restricts where the INPUT may put power rather than what the "
                         "observable keeps")
    ap.add_argument("--match-spectrum", action="store_true", dest="match_spectrum",
                    help="hold the model's OUTPUT power spectrum to the measured BOLD "
                         "spectrum (data/cache/bold_psd.npz). Convex, and a much tighter "
                         "statement than a band: the empirical spectrum falls as f^-2.6 "
                         "with 84%% of its variance in 0.01-0.1 Hz. Needs --oversample")
    ap.add_argument("--bold-smooth", action="store_true", dest="bold_smooth",
                    help="low-pass the field to the fMRI Nyquist before scoring, with "
                         "the same kernel folded into the transfer function")
    ap.add_argument("--profile", default="taper", choices=("taper", "gauss"),
                    help="driven-region shape. 'taper' is the smoothstep whose width is "
                         "set by the piece's own size; 'gauss' is a fixed-width kernel in "
                         "mm, so piece count controls channels only")
    ap.add_argument("--profile-fwhm", type=float, default=10.0, dest="profile_fwhm",
                    help="FWHM in mm for --profile gauss")
    ap.add_argument("--profile-mask", action="store_true", dest="profile_mask",
                    help="restrict gaussian profiles to the driven parcels, so the "
                         "driven AREA matches the taper version and only the shape "
                         "differs. Without it a wide kernel drives far more cortex and "
                         "coverage is confounded with profile shape")
    ap.add_argument("--tag", default="best")
    ap.add_argument("--best", action="store_true",
                    help="the pinned best sensory-only configuration (see BEST). Any "
                         "flag given explicitly still wins, so this is a starting point "
                         "rather than a lock")
    a = ap.parse_args()

    if a.best:
        # only fill in what the caller did not ask for, so --best --damp 1e-4 means what
        # it looks like it means
        given = set(sys.argv[1:])
        for k, v in BEST.items():
            if "--" + k.replace("_", "-") not in given:
                setattr(a, k, v)
        print("  --best: " + ", ".join(f"{k}={getattr(a, k)}" for k in BEST))

    c = load_cortex("fsaverage5", verbose=False)
    t = fc_score.default_target(c, centre=a.centre, verbose=True)
    mm = MoranMatch(c, t)
    parcels, split = subparcels.region_set(c, a.regions, a.split, a.spread_scale)
    labels, tags = subparcels.split_parcels(c, parcels, split, verbose=False)
    P = (subparcels.taper_profiles(c, labels, len(tags)) if a.profile == "taper"
         else subparcels.gauss_profiles(c, labels, len(tags), a.profile_fwhm,
                                        mask=(labels >= 0) if a.profile_mask else None))
    x = BEST_X.copy()
    clock = None
    if a.oversample:
        import timescale
        clock = timescale.plan(a.oversample,
                               decay_s=(None if a.decay_s == 0 else
                                        (a.decay_s or timescale.BOLD_TAU_S)),
                               spread_mm_s=a.spread_mm_s)
        x[3] = np.log10(clock["save"])
        x[0] = np.log10(clock["damp"])
        # the window has to hold the decay, which at a finer `save` is many more FRAMES
        # even though it is the same number of seconds
        decay_fr = 1.0 / (clock["damp"] * clock["save"])
        if a.impulse_frames < 3 * decay_fr:
            a.impulse_frames = int(np.ceil(3 * decay_fr / 64.0) * 64)
        if a.pad < a.impulse_frames:
            a.pad = int(2 ** np.ceil(np.log2(a.impulse_frames)))
        a.frames = timescale.frames_for(a.seconds, clock["frame_s"])
        if a.bold_smooth:
            a.smooth = timescale.bold_fwhm_frames(clock["frame_s"])
        print(f"  clock: frame {clock['frame_s']:.4f}s, impulse window "
              f"{a.impulse_frames} frames ({a.impulse_frames*clock['frame_s']:.0f}s, "
              f"decay {decay_fr:.0f} frames), pad {a.pad}, realise {a.frames} frames "
              f"({a.seconds:.0f}s)")
    if a.damp is not None:
        x[0] = np.log10(a.damp)
    p, save, _ = bo_step.unpack(x, c)
    print(f"  {len(P)} pieces, save {save} steps/frame, per-step damping "
          f"{10**x[0]:.2e}, rotation {10**x[1]:.2e}, sponge {10**x[2]:.2e}"
          + (f", impulse window {a.impulse_frames} frames "
             f"(decay {1.0/(10**x[0]*save):.0f})" if a.impulse_frames != 280 else ""))

    bp = fs_bp = None
    if a.bandpass:
        if clock is None:
            raise SystemExit("  --bandpass needs --oversample: the filter is defined in "
                             "Hz and without a clock there is no mapping from frames")
        bp = tuple(float(v) for v in a.bandpass.split(","))
        if len(bp) != 2:
            raise SystemExit("  --bandpass takes LO,HI in Hz")
        fs_bp = clock["frame_s"]
        print(f"  observable bandpassed {bp[0]}-{bp[1]} Hz, matching the target's filter")

    sub = xspec.medoid_subset(t, a.nvert)
    # held-out vertices serve two mechanisms that must not both run: selecting the
    # stopping point by realised score (--select-iters), or early stopping inside the
    # solve on the predicted covariance. Selection supersedes it, so it wins.
    n_val = a.val_vert or (1000 if a.select_iters else 0)
    val = xspec.validation_subset(t, sub, n_val) if n_val else None
    n = len(sub)
    iu = np.triu_indices(n, 1)
    raw = np.asarray(t.target_fc()[np.ix_(sub, sub)], np.float64)
    raw = raw - raw.mean(0, keepdims=True) - raw.mean(1, keepdims=True) + raw.mean()

    t0 = time.time()
    kern = units.smoothing_kernel(a.smooth) if a.smooth > 0 else None
    cpl = None
    if a.coupling > 0:
        import connectome
        D180 = connectome.parcel_distances(c, verbose=False)
        Wr = connectome.load_enigma(c)
        if not a.coupling_raw:
            Wr = connectome.residual_W(Wr, D180, a.coupling_keep, a.coupling_mm)
        if a.coupling_surrogate is not None:
            Wr = connectome.surrogate_W(Wr, D180, seed=a.coupling_surrogate)
        cpl = connectome.CouplingOperator(c, Wr, a.coupling, a.coupling_lag)
        # The impulse window has to hold the delayed arrival AND the decay that follows
        # it, or the response is truncated and H describes a system nobody simulates -
        # the identity the whole convex solve rests on.
        decay = 1.0 / (10 ** x[0] * save)                      # frames
        need = a.coupling_lag / save + 3.0 * decay
        if a.impulse_frames <= need:
            raise SystemExit(
                f"  impulse window {a.impulse_frames} frames is too short for lag "
                f"{a.coupling_lag} steps ({a.coupling_lag/save:.1f} frames) plus 3 decay "
                f"times ({3*decay:.0f} frames): need > {need:.0f}. Raise "
                f"--impulse-frames.")
        print(f"  coupling lam {a.coupling:g}, lag {a.coupling_lag} steps "
              f"({a.coupling_lag/save:.2f} frames)"
              + ("  [SURROGATE]" if a.coupling_surrogate is not None else "")
              + (", unfiltered W" if a.coupling_raw else "")
              + f": dt*bound {regimes.common_dt(c, [p]) * cpl.spectral_bound():.4g} "
              f"(needs << 1); window {a.impulse_frames} > {need:.0f} frames needed")
    nb = a.regimes
    if nb == 1:
        resp = xspec.impulse_responses(c, list(range(len(P))), p, a.impulse_frames * save, save,
                                       profiles=P, verbose=False, coupling=cpl,
                                       workers=a.workers)
        R = np.pad(resp, ((0, 0), (0, max(0, a.pad - resp.shape[1])), (0, 0)))
        H, w, idx = xspec.transfer(R, t.cols[sub], a.nfreq, kernel=kern)
        Hv = (xspec.transfer(R, t.cols[val], a.nfreq, kernel=kern)[0]
              if val is not None else None)
        ref_frames, ps, dt = R.shape[1], None, None
        if bp is not None:
            # the filter is linear, so like the smoothing kernel it simply multiplies the
            # transfer function - the solve is unchanged and costs nothing extra
            import bandpass
            br = bandpass.transfer_response(idx, ref_frames, clock["frame_s"], *bp)
            H = H * br[:, None, None]
            if Hv is not None:
                Hv = Hv * br[:, None, None]
            print(f"    filter response over the kept bins: {br.min():.3f}-{br.max():.3f}, "
                  f"{int((br > 0.5).sum())} of {len(br)} bins above 0.5")
    else:
        ps = regimes.regime_set(c, p, regime_deltas(
            nb, a.regime_span, a.regime_map, a.regime_target, base=p))
        dt = regimes.common_dt(c, ps)
        sched_f = regimes.schedule(a.frames, nb, a.epoch)
        occ = regimes.occupancy(sched_f, nb)
        dt1 = regimes.common_dt(c, [p])
        print(f"  {nb} regimes, epoch {a.epoch} frames, occupancy "
              f"{np.round(occ, 3)}, common dt {dt:.4g} (single medium {dt1:.4g})")
        H, w, idx, ref_frames = regimes.transfer_stack(
            c, ps, t.cols[sub], a.nfreq, a.pad, a.impulse_frames * save, save, P, occ, dt,
            kernel=kern)
        Hv = (regimes.transfer_stack(c, ps, t.cols[val], a.nfreq, a.pad,
                                     a.impulse_frames * save,
                                     save, P, occ, dt, verbose=False, kernel=kern)[0]
              if val is not None else None)
    print(f"  transfer: {H.shape[0]} frequencies x {H.shape[2]} channels from a "
          f"{ref_frames}-frame window [{time.time()-t0:.0f}s]")

    keep_f = None
    if a.band:
        if clock is None:
            raise SystemExit("  --band needs --oversample: without a clock there is no "
                             "mapping from frequency bins to Hz")
        import timescale
        lo, hi = (float(v) for v in a.band.split(","))
        keep_f = timescale.band_bins(idx, ref_frames, clock["frame_s"], (lo, hi))
        if not keep_f.any():
            raise SystemExit("  no solved frequency lies in that band")

    spec = None
    if a.match_spectrum:
        if clock is None:
            raise SystemExit("  --match-spectrum needs --oversample: without a clock "
                             "there is no mapping from frequency bins to Hz")
        import timescale
        spec = timescale.bold_power(idx, ref_frames, clock["frame_s"], w)
        if spec.sum() <= 0:
            raise SystemExit("  no solved frequency lies in the measured BOLD range")

    Tgt = normal_scores(raw, iu) if a.target == "normal" else raw
    Lw = None
    if a.whiten > 0:
        H, Lw = xspec.whiten(H, a.whiten)
        if Hv is not None:
            Hv = np.stack([np.linalg.solve(Lw[f], Hv[f].conj().T).conj().T
                           for f in range(len(Lw))])
        print(f"  whitened solve (eps {a.whiten:g}): H^H H conditioned to "
              f"{np.linalg.cond(H[0].conj().T @ H[0]):.1f} at the first frequency")

    val_Ct = None if val is None else np.asarray(
        t.target_fc()[np.ix_(val, val)], np.float64)

    if a.select_iters:
        a.iters = select_iters(a, c, t, p, P, H, w, idx, ref_frames, save, val, val_Ct,
                               Tgt, nb, sched_f if nb > 1 else None,
                               ps if nb > 1 else None, dt if nb > 1 else None, kern, cpl,
                               Lw, keep_f, spec, band=bp, frame_s=fs_bp)
    tr = []
    S, C = xspec.solve(H, w, Tgt, iters=a.iters, verbose=False, nblock=nb,
                       share=a.share_input, trace=tr, freq_keep=keep_f, spec=spec,
                       val_H=None if a.select_iters else Hv,
                       val_Ct=None if a.select_iters else val_Ct)
    if Lw is not None:
        S = xspec.unwhiten(S, Lw)
    from scipy.stats import spearmanr
    print(f"  solve: pearson vs raw {np.corrcoef(C[iu], raw[iu])[0,1]:+.4f}, "
          f"spearman vs raw {spearmanr(C[iu], raw[iu]).statistic:+.4f}")
    rep = tr[-1]
    if "stopped_at" in rep:
        print(f"  early stopping on {len(val)} held-out vertices: best spearman "
              f"{rep['held_out']:+.4f} at step {rep['stopped_at']} of {rep['iters']} "
              f"(objective there was still climbing to {rep['final']:.4f})")
    else:
        tail = tr[-6:-1] if len(tr) > 6 else tr[:-1]
        print(f"  convergence: objective {rep['final']:.4f} after {rep['steps']} accepted "
              f"steps of {rep['iters']}"
              + (f", STALLED at {rep['stalled_at']}" if rep['stalled_at'] is not None
                 else "; last 5 gains "
                      + ", ".join(f"{tail[i+1]-tail[i]:+.1e}" for i in range(len(tail)-1))))

    for it in range(a.rank_iters):
        Tm = np.zeros_like(raw)
        Tm[iu] = quantile_match(raw[iu], C[iu])
        Tm = Tm + Tm.T
        Tm = Tm - Tm.mean(0, keepdims=True) - Tm.mean(1, keepdims=True) + Tm.mean()
        S, C = xspec.solve(H, w, Tm, iters=a.iters, verbose=False,
                           nblock=nb, share=a.share_input)
        print(f"  rank iteration {it+1}: spearman vs raw "
              f"{spearmanr(C[iu], raw[iu]).statistic:+.4f}")

    sims, gaps, rks = [], [], []
    # Draws 1.. are scored in a pool while draw 0 runs here. Draw 0 stays in this process
    # because its frames and drive are the ones written to disk, and 134 MB per draw
    # through a pipe would cost more than the realisation does. A run_fn case - switching
    # medium, or coupling - keeps every draw here: the callback is a closure, so it does
    # not pickle.
    pool = res = None
    if a.workers > 1 and a.draws > 1 and nb == 1 and cpl is None:
        rest = [xspec.realise(S, idx, a.frames, ref_frames=ref_frames, seed=1000 + d)
                for d in range(1, a.draws)]
        pool, res = xspec.parallel_scores(c, t, p, rest, save, P, kern, a.workers,
                                          band=bp, frame_s=fs_bp)
        print(f"  draws 2-{a.draws} scored over {min(a.workers, len(rest))} workers",
              flush=True)
    for d in range(1 if pool is not None else a.draws):
        if nb == 1:
            A = xspec.realise(S, idx, a.frames, ref_frames=ref_frames, seed=1000 + d)
            run_fn = None
        else:
            A = regimes.realise_switching(S, idx, a.frames, ref_frames, nb, sched_f,
                                          w=w, seed=1000 + d)
            steps = np.repeat(sched_f, save)
            run_fn = (lambda dr, nsteps, sv: regimes.run_switching(
                c, dr, ps, steps[:nsteps], nsteps, sv, dt))
        if cpl is not None and run_fn is None:
            run_fn = (lambda dr, nsteps, sv: bo_step.fl.run(
                c, dr, p, nsteps, sv, coupling=cpl))
        r = xspec.score_realisation(c, t, p, A, save=save, profiles=P,
                                    run_fn=run_fn, kernel=kern, band=bp, frame_s=fs_bp)
        sims.append(r["sim"]); gaps.append(r["gap"]); rks.append(r["rank"])
        if d == 0:
            if nb > 1:
                regimes.epoch_profile(r["frames"], sched_f, a.epoch, nb)
            np.save(os.path.join(RESULTS, f"frames_{a.tag}.npy"), r["frames"])
            np.save(os.path.join(RESULTS, f"drive_{a.tag}.npy"), r["drive"].Aser)
    if pool is not None:
        for sim, gap, rk in res.get():
            sims.append(sim); gaps.append(gap); rks.append(rk)
        pool.close(); pool.join()
    np.savez(os.path.join(RESULTS, f"xspec_{a.tag}.npz"), S=S, idx=idx, x=x,
             save=save, labels=labels, tags=np.array(tags, dtype=object))
    print(f"\n  realised over {a.frames} frames, {a.draws} draws: "
          f"sim {np.mean(sims):+.4f} +- {np.std(sims):.4f}   "
          f"gap {np.mean(gaps):.3f}   rank {np.mean(rks):.1f}")


if __name__ == "__main__":
    main()
