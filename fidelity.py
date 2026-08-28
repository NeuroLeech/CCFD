"""Where does the solve's prediction stop describing the simulation?

The solve returns C(S), the covariance its input cross-spectrum implies. The score comes
from simulating a drive drawn with that cross-spectrum and correlating the result with the
target. Those are two different objects and the gap between them is not the same as the
gap between solve vertices and held-out ones - held-out Spearman peaks around step 130
while the realised score peaks around step 25, so most of the loss is downstream of any
vertex-level overfitting.

This measures the downstream part directly: for a saved run, rebuild C(S) and compare it
against the FC the simulation actually produced, on the same vertices.

  python fidelity.py itsweep25 itsweep50 itsweep100 itsweep300
"""
import sys, os
import numpy as np
from scipy.stats import spearmanr, rankdata

from mesh_cache import load_cortex
from fc_score import FCTarget
from paths import RESULTS
import xspec, bo_step, subparcels
from best_fit import BEST_X


def predicted(H, w, S):
    nf, nV, _ = H.shape
    C = np.zeros((nV, nV))
    for f in range(nf):
        C += w[f] * 2.0 * np.real((H[f] @ S[f]) @ H[f].conj().T)
    return C - C.mean(0, keepdims=True) - C.mean(1, keepdims=True) + C.mean()


def main(tags):
    c = load_cortex("fsaverage5", verbose=False)
    t = FCTarget(c, verbose=False)
    labels, tg = subparcels.split_parcels(c, subparcels.SENSORY, 50, verbose=False)
    P = subparcels.taper_profiles(c, labels, len(tg))
    p, save, _ = bo_step.unpack(BEST_X, c)
    sub = xspec.medoid_subset(t, 1000)
    iu = np.triu_indices(len(sub), 1)
    resp = xspec.impulse_responses(c, list(range(len(P))), p, 280 * save, save,
                                   profiles=P, verbose=False)
    R = np.pad(resp, ((0, 0), (0, max(0, 1120 - resp.shape[1])), (0, 0)))
    H, w, idx = xspec.transfer(R, t.cols[sub], 192)
    raw = np.asarray(t.target_fc()[np.ix_(sub, sub)], np.float64)

    print(f"  on the {len(sub)} solve vertices, {len(iu[0])} edges\n")
    print(f"  {'tag':16s} {'pred vs target':>14s} {'sim vs target':>14s} "
          f"{'sim vs pred':>12s} {'rank':>6s}")
    for tag in tags:
        zf = os.path.join(RESULTS, f"xspec_{tag}.npz")
        ff = os.path.join(RESULTS, f"frames_{tag}.npy")
        if not (os.path.exists(zf) and os.path.exists(ff)):
            print(f"  {tag:16s} missing"); continue
        S = np.load(zf, allow_pickle=True)["S"]
        C = predicted(H, w, S)
        frames = np.load(ff, mmap_mode="r")
        Z, _ = t.model_z(np.asarray(frames))
        Zs = Z[sub]
        F = (Zs @ Zs.T) / Zs.shape[1]           # realised FC on the same vertices
        F = F - F.mean(0, keepdims=True) - F.mean(1, keepdims=True) + F.mean()
        rk = t.effective_rank(np.asarray(frames)[t.burn:])
        print(f"  {tag:16s} {spearmanr(C[iu], raw[iu]).statistic:+14.4f} "
              f"{spearmanr(F[iu], raw[iu]).statistic:+14.4f} "
              f"{spearmanr(F[iu], C[iu]).statistic:+12.4f} {rk:6.1f}")


if __name__ == "__main__":
    main(sys.argv[1:])
