"""Is the solver, not the model, what makes more pieces score worse?

The 98-piece parameterisation contains the 47-piece one: each 47-piece is a union of
98-pieces, so setting S98[i,j] = S47[parent(i), parent(j)] drives every child of a parent
in lockstep and reproduces the parent's drive. A blow-up of a PSD matrix is PSD, so that
is a feasible point of the 98-piece problem with (nearly) the 47-piece objective.

If the 98-piece solve from a cold start scores BELOW that feasible point, it has not found
its own optimum, and every comparison made through this solver is a statement about the
optimiser rather than about the medium.
"""
import numpy as np
from mesh_cache import load_cortex
import fc_score, xspec, bo_step, subparcels
from best_fit import BEST_X

DAMP, WINDOW, NF, ITERS = 2e-4, 560, 192, 400
c = load_cortex("fsaverage5", verbose=False)
t = fc_score.default_target(c, verbose=False)
x = BEST_X.copy(); x[0] = np.log10(DAMP)
p, save, _ = bo_step.unpack(x, c)
sub = xspec.medoid_subset(t, 1000); n = len(sub); iu = np.triu_indices(n, 1)
raw = np.asarray(t.target_fc()[np.ix_(sub, sub)], np.float64)
raw = raw - raw.mean(0, keepdims=True) - raw.mean(1, keepdims=True) + raw.mean()
Tgt = xspec.normal_scores(raw); Tgt[np.eye(n, dtype=bool)] = 0.0

def build(split):
    lab, tg = subparcels.split_parcels(c, subparcels.SENSORY, split, verbose=False)
    P = subparcels.taper_profiles(c, lab, len(tg))
    r = xspec.impulse_responses(c, list(range(len(P))), p, WINDOW*save, save,
                                profiles=P, verbose=False)
    R = np.pad(r, ((0,0),(0,max(0,1120-r.shape[1])),(0,0)))
    H, w, idx = xspec.transfer(R, t.cols[sub], NF)
    return lab, tg, H, w

def obj(H, w, S):
    C = np.zeros((n, n))
    for f in range(len(w)):
        C += w[f] * 2.0 * np.real((H[f] @ S[f]) @ H[f].conj().T)
    C = C - C.mean(0, keepdims=True) - C.mean(1, keepdims=True) + C.mean()
    C[np.eye(n, dtype=bool)] = 0.0
    T = Tgt / np.linalg.norm(Tgt)
    return float((C * T).sum() / np.linalg.norm(C))

lab47, tg47, H47, w47 = build(50)
lab98, tg98, H98, w98 = build(100)
print(f"  {len(tg47)} vs {len(tg98)} pieces")

# parent of each 98-piece: the 47-piece it overlaps most
parent = np.array([np.bincount(lab47[lab98 == k][lab47[lab98 == k] >= 0],
                               minlength=len(tg47)).argmax() for k in range(len(tg98))])
print(f"  every 98-piece maps into a 47-piece; parents used: {len(np.unique(parent))}")

S47, _ = xspec.solve(H47, w47, Tgt, iters=ITERS, verbose=False)
print(f"\n  47 pieces, cold start, {ITERS} iters : objective {obj(H47, w47, S47):.4f}")

S98cold, _ = xspec.solve(H98, w98, Tgt, iters=ITERS, verbose=False)
print(f"  98 pieces, cold start, {ITERS} iters : objective {obj(H98, w98, S98cold):.4f}")

blow = np.stack([S47[f][np.ix_(parent, parent)] for f in range(len(w47))])
print(f"  98 pieces, the 47-piece solution embedded: objective {obj(H98, w98, blow):.4f}"
      f"   (PSD: min eig {min(np.linalg.eigvalsh(blow[f]).min() for f in range(len(blow))):+.1e})")

S98warm, _ = xspec.solve(H98, w98, Tgt, iters=ITERS, verbose=False, S0=blow)
print(f"  98 pieces, warm started from it, {ITERS} iters : "
      f"objective {obj(H98, w98, S98warm):.4f}")
