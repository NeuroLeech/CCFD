"""How anisotropic is the reachable span, and what does rotation do to it?

C = sum_f w_f 2 Re(H_f S_f H_f^H), so range(C) lies in the REAL span of the real and
imaginary parts of the columns of H. That span turns out to be numerically full rank, so
there is no geometric ceiling - but its energy is extremely concentrated, and the solve
only ever reaches the strong directions. The bound ||P Ct P|| / ||Ct|| over the top m
directions says what would be reachable if the solve could use m of them.

Rotation is the term that should change this structurally. With Coriolis the dispersion
relation is omega^2 = f^2 + c^2 k^2, so different frequencies give different SPATIAL
patterns rather than rescaled copies of one blob. Without it the medium is close to a
damped diffusion, whose response shape barely changes with frequency - which is what makes
118 frequencies buy so little over one.

  python diag_span.py --rot 1e-3
"""
import argparse
import numpy as np

from mesh_cache import load_cortex
import fc_score, xspec, bo_step, subparcels
from best_fit import BEST_X


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rot", type=float, default=None,
                    help="rotation per step (default: whatever BEST_X carries)")
    ap.add_argument("--damp", type=float, default=None,
                    help="interior damping per step (default: from BEST_X)")
    ap.add_argument("--nvert", type=int, default=1000)
    ap.add_argument("--split", type=int, default=50)
    ap.add_argument("--nfreq", type=int, default=192)
    ap.add_argument("--window", type=int, default=280,
                    help="impulse frames before padding; must exceed the decay time or "
                         "the response is truncated and the transfer function is wrong")
    ap.add_argument("--coupling", type=float, default=0.0,
                    help="long-range structural coupling strength (0 = off)")
    ap.add_argument("--coupling-lag", type=int, default=0, dest="coupling_lag")
    ap.add_argument("--coupling-keep", type=float, default=0.15, dest="coupling_keep")
    ap.add_argument("--coupling-mm", type=float, default=60.0, dest="coupling_mm")
    ap.add_argument("--coupling-surrogate", type=int, default=None,
                    dest="coupling_surrogate", metavar="SEED")
    a = ap.parse_args()

    c = load_cortex("fsaverage5", verbose=False)
    t = fc_score.default_target(c, verbose=False)
    labels, tg = subparcels.split_parcels(c, subparcels.SENSORY, a.split, verbose=False)
    P = subparcels.taper_profiles(c, labels, len(tg))
    x = BEST_X.copy()
    if a.rot is not None:
        x[1] = np.log10(a.rot)
    if a.damp is not None:
        x[0] = np.log10(a.damp)
    p, save, _ = bo_step.unpack(x, c)
    print(f"  rotation per step {10**x[1]:.2e}  (Ld {p['Ld']:.4g}), "
          f"damping {10**x[0]:.2e}, {len(P)} pieces")

    # NOTE: whitening is deliberately absent. It is a change of variables in the CHANNEL
    # space - H L^-H - so it multiplies H on the right by an invertible matrix and leaves
    # the column span, and therefore this ceiling, exactly unchanged. Whitening changes
    # what the solver REACHES, not what is reachable. Coupling is the term that can
    # actually move the span, which is why it is the one exposed here.
    cpl = None
    if a.coupling > 0:
        import connectome
        D180 = connectome.parcel_distances(c, verbose=False)
        Wr = connectome.residual_W(connectome.load_enigma(c, verbose=False), D180,
                                   a.coupling_keep, a.coupling_mm, verbose=False)
        if a.coupling_surrogate is not None:
            Wr = connectome.surrogate_W(Wr, D180, seed=a.coupling_surrogate,
                                        verbose=False)
        cpl = connectome.CouplingOperator(c, Wr, a.coupling, a.coupling_lag)
        print(f"  coupling lam {a.coupling:g}, lag {a.coupling_lag} steps"
              + ("  [SURROGATE]" if a.coupling_surrogate is not None else ""))

    sub = xspec.medoid_subset(t, a.nvert)
    resp = xspec.impulse_responses(c, list(range(len(P))), p, a.window * save, save,
                                   profiles=P, verbose=False, coupling=cpl)
    R = np.pad(resp, ((0, 0), (0, max(0, 1120 - resp.shape[1])), (0, 0)))
    H, w, idx = xspec.transfer(R, t.cols[sub], a.nfreq)
    nf, nV, K = H.shape

    def prank(sv):
        e = np.asarray(sv, float) ** 2
        return float(e.sum() ** 2 / max((e ** 2).sum(), 1e-300))

    one = prank(np.linalg.svd(H[nf // 2], compute_uv=False))
    A = np.concatenate([np.sqrt(w[f]) * H[f] for f in range(nf)], axis=1)
    Ar = np.concatenate([A.real, A.imag], axis=1)
    U, s, _ = np.linalg.svd(Ar, full_matrices=False)
    e = s ** 2
    cum = np.cumsum(e) / e.sum()

    Ct = np.asarray(t.target_fc()[np.ix_(sub, sub)], np.float64)
    Ct = Ct - Ct.mean(0, keepdims=True) - Ct.mean(1, keepdims=True) + Ct.mean()
    Ct = xspec.normal_scores(Ct)
    Ct[np.eye(nV, dtype=bool)] = 0.0
    nrm = np.linalg.norm(Ct)

    print(f"  within one frequency: {one:.1f} effective patterns of {K} pieces")
    print(f"  real span: participation rank {prank(s):.1f}, "
          f"numerical rank {int((s > 1e-10 * s[0]).sum())} of {nV}")
    print(f"  {'dims':>6s} {'energy':>8s} {'ceiling':>9s}")
    for m in (25, 50, 100, 200, 400):
        Pm = U[:, :m]
        Pc = Pm @ (Pm.T @ Ct @ Pm) @ Pm.T
        print(f"  {m:6d} {cum[m-1]:8.4f} {np.linalg.norm(Pc)/nrm:9.3f}")


if __name__ == "__main__":
    main()
