"""Reproduce the current best fit: 47 sensory pieces, solved cross-spectrum, one number.

Everything the result depends on is here rather than spread across scratch scripts:
the medium (per-step units, from the bo_step search), the region split, the zero-padded
transfer function, the rank-transformed target, the realisation and the score.

  python best_fit.py                          # the current best configuration
  python best_fit.py --frames 2240 --draws 3  # longer realisation
  python best_fit.py --rank-iters 4           # iterate the solve on the model's ranks
"""
import os, time, argparse
import numpy as np

from mesh_cache import load_cortex
from fc_score import FCTarget
from fc_moran import MoranMatch
from paths import RESULTS
import xspec, bo_step, subparcels, regimes, units

# the bo_step winner, in per-step units: damping, rotation, boundary absorption, cadence,
# then the six speed/damping map coefficients
BEST_X = np.array([np.log10(6.2e-4), np.log10(1.1e-5), np.log10(1.8e-3), np.log10(33),
                   -0.30, -0.05, 0.01, -0.03, 0.35, 0.35])
PAD, NFREQ, NVERT = 1120, 192, 1000


normal_scores = xspec.normal_scores          # lives in xspec so bo_step can use it too
region_set = subparcels.region_set           # lives in subparcels for the same reason


def quantile_match(target_edges, model_edges):
    """Re-express the target in the model's own value scale: rank the target, then read
    off the model's quantile at that rank. Pearson against this tracks Spearman against
    the raw target far more closely, which is what the score actually measures."""
    order = np.argsort(np.argsort(target_edges))
    return np.sort(model_edges)[order]


def regime_deltas(R, span):
    """R sets of map-coefficient offsets, spread symmetrically about the base medium.

    The offsets go on a and b - the log-linear grading of speed and damping by myelin,
    thickness and sulcal depth - so the regimes differ in the spatial PATTERN of the
    medium, not merely its overall scale. A global scalar would mostly reproduce the
    inertness bo_step already found in c0."""
    if R == 1:
        return [{}]
    t = np.linspace(-1.0, 1.0, R)
    return [dict(da=span * v * np.array([1.0, -1.0, 0.0]),
                 db=span * v * np.array([-1.0, 0.0, 1.0]),
                 sig=float(10.0 ** (0.5 * span * v))) for v in t]


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
    ap.add_argument("--split", type=int, default=50,
                    help="pieces to divide the driven parcels into (sets the piece area)")
    ap.add_argument("--regions", default="sensory",
                    choices=("sensory", "dmn", "sensory+dmn", "spread"),
                    help="which parcels are driven; 'spread' is an even whole-cortex "
                         "sample matched to the sensory driven area")
    ap.add_argument("--coupling", type=float, default=0.0,
                    help="long-range structural coupling strength (0 = off); "
                         "dimensionless, see connectome.load_enigma")
    ap.add_argument("--coupling-mm", type=float, default=60.0, dest="coupling_mm",
                    help="shortest connection counted as long-range")
    ap.add_argument("--coupling-keep", type=float, default=0.15, dest="coupling_keep",
                    help="fraction of long connections kept, by distance residual")
    ap.add_argument("--smooth", type=float, default=0.0,
                    help="temporal FWHM in frames of the filter between field and "
                         "observable (0 = none); units.py says the model frame is "
                         "~1/100 of a TR, so this is where that gap would be closed")
    ap.add_argument("--regimes", type=int, default=1,
                    help="media switched within the run (1 = the single-medium path)")
    ap.add_argument("--epoch", type=int, default=regimes.EPOCH_FRAMES,
                    help="frames per regime epoch; must exceed the field's decay time")
    ap.add_argument("--regime-span", type=float, default=0.30, dest="regime_span",
                    help="spread of the per-regime map coefficients, in ln units")
    ap.add_argument("--share-input", action="store_true", dest="share_input",
                    help="constrain every regime to the same input cross-spectrum")
    ap.add_argument("--spread-scale", type=float, default=1.0, dest="spread_scale",
                    help="multiply the 'spread' area budget (1.0 = the sensory area)")
    ap.add_argument("--target", default="normal",
                    choices=("normal", "raw"), help="what the solve matches")
    ap.add_argument("--tag", default="best")
    a = ap.parse_args()

    c = load_cortex("fsaverage5", verbose=False)
    t = FCTarget(c, verbose=True)
    mm = MoranMatch(c, t)
    parcels, split = subparcels.region_set(c, a.regions, a.split, a.spread_scale)
    labels, tags = subparcels.split_parcels(c, parcels, split, verbose=False)
    P = subparcels.taper_profiles(c, labels, len(tags))
    p, save, _ = bo_step.unpack(BEST_X, c)
    print(f"  {len(P)} pieces, save {save} steps/frame, per-step damping "
          f"{10**BEST_X[0]:.2e}, rotation {10**BEST_X[1]:.2e}, sponge {10**BEST_X[2]:.2e}")

    sub = xspec.medoid_subset(t, a.nvert)
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
        Wr = connectome.residual_W(connectome.load_enigma(c), D180,
                                   a.coupling_keep, a.coupling_mm)
        cpl = connectome.CouplingOperator(c, Wr, a.coupling)
        print(f"  coupling lam {a.coupling:g}: dt*bound "
              f"{regimes.common_dt(c, [p]) * cpl.spectral_bound():.4g} (needs << 1)")
    nb = a.regimes
    if nb == 1:
        resp = xspec.impulse_responses(c, list(range(len(P))), p, 280 * save, save,
                                       profiles=P, verbose=False, coupling=cpl)
        R = np.pad(resp, ((0, 0), (0, max(0, a.pad - resp.shape[1])), (0, 0)))
        H, w, idx = xspec.transfer(R, t.cols[sub], a.nfreq, kernel=kern)
        ref_frames, ps, dt = R.shape[1], None, None
    else:
        ps = regimes.regime_set(c, p, regime_deltas(nb, a.regime_span))
        dt = regimes.common_dt(c, ps)
        sched_f = regimes.schedule(a.frames, nb, a.epoch)
        occ = regimes.occupancy(sched_f, nb)
        dt1 = regimes.common_dt(c, [p])
        print(f"  {nb} regimes, epoch {a.epoch} frames, occupancy "
              f"{np.round(occ, 3)}, common dt {dt:.4g} (single medium {dt1:.4g})")
        H, w, idx, ref_frames = regimes.transfer_stack(
            c, ps, t.cols[sub], a.nfreq, a.pad, 280 * save, save, P, occ, dt,
            kernel=kern)
    print(f"  transfer: {H.shape[0]} frequencies x {H.shape[2]} channels from a "
          f"{ref_frames}-frame window [{time.time()-t0:.0f}s]")

    Tgt = normal_scores(raw, iu) if a.target == "normal" else raw
    S, C = xspec.solve(H, w, Tgt, iters=a.iters, verbose=False, nblock=nb,
                       share=a.share_input)
    from scipy.stats import spearmanr
    print(f"  solve: pearson vs raw {np.corrcoef(C[iu], raw[iu])[0,1]:+.4f}, "
          f"spearman vs raw {spearmanr(C[iu], raw[iu]).statistic:+.4f}")

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
    for d in range(a.draws):
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
                                    run_fn=run_fn, kernel=kern)
        sims.append(r["sim"]); gaps.append(r["gap"]); rks.append(r["rank"])
        if d == 0:
            if nb > 1:
                regimes.epoch_profile(r["frames"], sched_f, a.epoch, nb)
            np.save(os.path.join(RESULTS, f"frames_{a.tag}.npy"), r["frames"])
            np.save(os.path.join(RESULTS, f"drive_{a.tag}.npy"), r["drive"].Aser)
    np.savez(os.path.join(RESULTS, f"xspec_{a.tag}.npz"), S=S, idx=idx, x=BEST_X,
             save=save, labels=labels, tags=np.array(tags, dtype=object))
    print(f"\n  realised over {a.frames} frames, {a.draws} draws: "
          f"sim {np.mean(sims):+.4f} +- {np.std(sims):.4f}   "
          f"gap {np.mean(gaps):.3f}   rank {np.mean(rks):.1f}")


if __name__ == "__main__":
    main()
