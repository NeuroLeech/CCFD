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
import xspec, bo_step, subparcels

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
    resp = xspec.impulse_responses(c, list(range(len(P))), p, 280 * save, save,
                                   profiles=P, verbose=False)
    R = np.pad(resp, ((0, 0), (0, max(0, a.pad - resp.shape[1])), (0, 0)))
    H, w, idx = xspec.transfer(R, t.cols[sub], a.nfreq)
    print(f"  transfer: {H.shape[0]} frequencies from a {R.shape[1]}-frame window "
          f"[{time.time()-t0:.0f}s]")

    Tgt = normal_scores(raw, iu) if a.target == "normal" else raw
    S, C = xspec.solve(H, w, Tgt, iters=a.iters, verbose=False)
    from scipy.stats import spearmanr
    print(f"  solve: pearson vs raw {np.corrcoef(C[iu], raw[iu])[0,1]:+.4f}, "
          f"spearman vs raw {spearmanr(C[iu], raw[iu]).statistic:+.4f}")

    for it in range(a.rank_iters):
        Tm = np.zeros_like(raw)
        Tm[iu] = quantile_match(raw[iu], C[iu])
        Tm = Tm + Tm.T
        Tm = Tm - Tm.mean(0, keepdims=True) - Tm.mean(1, keepdims=True) + Tm.mean()
        S, C = xspec.solve(H, w, Tm, iters=a.iters, verbose=False)
        print(f"  rank iteration {it+1}: spearman vs raw "
              f"{spearmanr(C[iu], raw[iu]).statistic:+.4f}")

    sims, gaps, rks = [], [], []
    for d in range(a.draws):
        A = xspec.realise(S, idx, a.frames, ref_frames=R.shape[1], seed=1000 + d)
        r = xspec.score_realisation(c, t, p, A, save=save, profiles=P)
        sims.append(r["sim"]); gaps.append(r["gap"]); rks.append(r["rank"])
        if d == 0:
            np.save(os.path.join(RESULTS, f"frames_{a.tag}.npy"), r["frames"])
            np.save(os.path.join(RESULTS, f"drive_{a.tag}.npy"), r["drive"].Aser)
    np.savez(os.path.join(RESULTS, f"xspec_{a.tag}.npz"), S=S, idx=idx, x=BEST_X,
             save=save, labels=labels, tags=np.array(tags, dtype=object))
    print(f"\n  realised over {a.frames} frames, {a.draws} draws: "
          f"sim {np.mean(sims):+.4f} +- {np.std(sims):.4f}   "
          f"gap {np.mean(gaps):.3f}   rank {np.mean(rks):.1f}")


if __name__ == "__main__":
    main()
