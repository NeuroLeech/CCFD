"""Recover a drive we already know, on a small system, in seconds.

Against real FC a failed fit is uninterpretable: "the solve works and the model is wrong"
and "the model is right and the solve is failing" look identical. Here the target is
generated from a KNOWN cross-spectrum through the same medium and the same transfer
function, so recovery has an answer, and the solver can be separated from the model.

Small enough to iterate on: 10 channels and 100 vertices instead of 100 and 1,000, a short
impulse window, a small pad. The mesh and every code path are the real ones - the point is
to test the actual solve, not a reimplementation of it that can quietly diverge.

WHAT IT IS FOR. `best_fit` multiplies H by the passband response before solving, so above
0.08 Hz H is ~0: a drive placed there produces no observable output, the solve has no
gradient, and the fit is not merely attenuated but annihilated. That is a choice made in
the pipeline, not a fact about the physics, and in a linear medium it decides whether fast
input can be fitted at all. `--observable noband` removes it so the two can be compared on
a problem whose answer is known.

  python fit/toy_recover.py --true-band fast --observable band,noband
"""
import _path  # noqa: F401  - puts the sibling code folders on sys.path
import time, argparse
import numpy as np

from mesh_cache import load_cortex
import xspec, subparcels, bo_step, timescale, bandpass, units


def forward(H, w, S):
    """C(S) = sum_f w_f 2 Re(H_f S_f H_f^H), the covariance the solve matches."""
    out = np.zeros((H.shape[1], H.shape[1]))
    for f in range(H.shape[0]):
        out += w[f] * 2.0 * np.real(H[f] @ S[f] @ H[f].conj().T)
    return out


def psd_truth(nf, K, rank, rng, keep=None):
    """A known S(f) >= 0: G G^H per frequency, low rank. `keep` zeroes it outside a band."""
    G = (rng.normal(size=(nf, K, rank)) + 1j * rng.normal(size=(nf, K, rank)))
    S = np.einsum("fab,fcb->fac", G, G.conj())
    if keep is not None:
        S[~np.asarray(keep, bool)] = 0.0
    tr = sum(np.trace(S[f]).real for f in range(nf))
    return S / max(tr, 1e-300)


def corr(a, b):
    a = np.asarray(a, float).ravel(); b = np.asarray(b, float).ravel()
    a = a - a.mean(); b = b - b.mean()
    return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--channels", type=int, default=10)
    ap.add_argument("--nvert", type=int, default=100)
    ap.add_argument("--oversample", type=int, default=4)
    ap.add_argument("--spread-mm-s", type=float, default=1.5, dest="spread")
    ap.add_argument("--decay-s", type=float, default=25.0, dest="decay_s")
    ap.add_argument("--impulse-decays", type=float, default=3.0, dest="idecays")
    ap.add_argument("--pad", type=int, default=0, help="0 = next power of two above the window")
    ap.add_argument("--nfreq", type=int, default=24)
    ap.add_argument("--band", default="0.01,0.08")
    ap.add_argument("--rank", type=int, default=3, help="rank of the true S(f)")
    ap.add_argument("--true-band", default="all", choices=("all", "slow", "fast"),
                    help="where the TRUE drive puts its power: everywhere, inside the "
                         "passband, or entirely above it. 'fast' is the case the passband "
                         "multiplication decides")
    ap.add_argument("--observable", default="band,noband",
                    help="comma-separated: band (H multiplied by the passband response, "
                         "what best_fit does) and/or noband (H left alone)")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--starts", type=int, default=3,
                    help="random restarts. A convex problem reaches the same objective "
                         "from any of them; spread across starts is the diagnostic")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    lo, hi = (float(v) for v in a.band.split(","))
    rng = np.random.default_rng(a.seed)

    # ---- the system, built with the real code on the real mesh
    t0 = time.time()
    c = load_cortex("fsaverage5", verbose=False)
    cl = timescale.plan(a.oversample, decay_s=a.decay_s, spread_mm_s=a.spread,
                        verbose=False)
    x = np.array(__import__("best_fit").BEST_X, copy=True)
    x[3] = np.log10(cl["save"]); x[0] = np.log10(cl["damp"])
    x[4:10] = 0.0                                        # FLAT medium
    p, save, _ = bo_step.unpack(x, c)
    parcels, split = subparcels.region_set(c, "sensory", a.channels, 1.0)
    labels, tags = subparcels.split_parcels(c, parcels, split, verbose=False)
    P = subparcels.taper_profiles(c, labels, len(tags))
    K = len(tags)

    decay_fr = 1.0 / (cl["damp"] * save)
    imp = int(np.ceil(max(a.idecays * decay_fr, 1.0 / (lo * cl["frame_s"])) / 64) * 64)
    pad = a.pad or int(2 ** np.ceil(np.log2(imp)))
    keep = np.sort(rng.choice(c.nV, a.nvert, replace=False))

    resp = xspec.impulse_responses(c, list(range(K)), p, imp * save, save, profiles=P,
                                   verbose=False, workers=0, keep=keep, cache=False)
    R = np.pad(resp, ((0, 0), (0, max(0, pad - resp.shape[1])), (0, 0)))
    kern = units.smoothing_kernel(timescale.bold_fwhm_frames(cl["frame_s"], verbose=False),
                                  verbose=False)
    H0, w, idx = xspec.transfer(R, np.arange(a.nvert), a.nfreq, kernel=kern)
    br = bandpass.transfer_response(idx, pad, cl["frame_s"], lo, hi)
    f_hz = np.asarray(idx, float) / (pad * cl["frame_s"])
    inband = (f_hz >= lo) & (f_hz <= hi)
    print(f"  {K} channels, {a.nvert} vertices, {len(idx)} frequencies "
          f"({int(inband.sum())} inside {lo}-{hi} Hz), window {imp} frames "
          f"({imp*cl['frame_s']:.0f}s), pad {pad} ({pad*cl['frame_s']:.0f}s)")
    print(f"  flat medium: {cl['spread_mm_s']:.2f} mm/s, decay {cl['decay_s']:.1f}s, "
          f"reach {cl['reach_mm']:.0f} mm   [setup {time.time()-t0:.1f}s]")
    print(f"  passband response over the solved bins: {br.min():.3f}-{br.max():.3f}")

    sel = {"all": None, "slow": inband, "fast": ~inband}[a.true_band]
    S_true = psd_truth(len(idx), K, a.rank, rng, keep=sel)
    print(f"  true S: rank {a.rank}, power {a.true_band}"
          + ("" if sel is None else f" ({int(sel.sum())} of {len(idx)} bins)"))

    for mode in [m.strip() for m in a.observable.split(",") if m.strip()]:
        H = H0 * br[:, None, None] if mode == "band" else H0
        Ct = forward(H, w, S_true)
        iu = np.triu_indices(a.nvert, 1)
        amp = float(np.abs(Ct[iu]).mean())
        print(f"\n  === observable '{mode}' ===")
        print(f"  target from the TRUE drive: mean |edge| {amp:.3e}")
        if amp < 1e-12:
            print("  the target is numerically zero - this observable cannot see this "
                  "drive at all, so there is nothing to recover")
            continue
        objs, sims, ssim, smarg = [], [], [], []
        for st in range(a.starts):
            r2 = np.random.default_rng(1000 + st)
            S0 = psd_truth(len(idx), K, K, r2)
            tr = []
            S, C = xspec.solve(H, w, Ct, iters=a.iters, verbose=False, S0=S0, trace=tr)
            objs.append(tr[-1]["final"] if isinstance(tr[-1], dict) else float(tr[-1]))
            sims.append(corr(C[iu], Ct[iu]))
            ssim.append(corr(np.real(S), np.real(S_true)))
            # C sums over frequency, so any redistribution of power across f that leaves
            # the sum alone is invisible to it. If the frequency MARGINAL is recovered
            # while the per-frequency S is not, the shortfall is that collapse and not
            # the optimiser.
            smarg.append(corr(np.real(S.sum(0)), np.real(S_true.sum(0))))
        print(f"  {a.starts} random starts, {a.iters} iterations each:")
        print(f"    covariance   corr(C(S_hat), C_true)       = "
              f"{np.mean(sims):+.6f} +- {np.std(sims):.1e}")
        print(f"    drive        corr(S_hat, S_true)          = "
              f"{np.mean(ssim):+.6f} +- {np.std(ssim):.1e}")
        print(f"    drive summed corr(sum_f S_hat, sum_f S_true) = "
              f"{np.mean(smarg):+.6f} +- {np.std(smarg):.1e}")
        rel = np.std(objs) / max(abs(np.mean(objs)), 1e-30)
        print(f"    objective {np.mean(objs):.6f}, spread across starts "
              f"{np.std(objs):.1e} ({rel:.1e} relative) "
              + ("- one optimum" if rel < 1e-3 else "- STARTS DISAGREE"))


if __name__ == "__main__":
    main()
