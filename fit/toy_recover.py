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


def field_lagged_cov(Phi, w, idx, ref_frames, lags):
    """C_ij(tau) = sum_f w_f 2 Re(Phi_f e^{i 2 pi f tau}) for each lag, tau in frames.

    The same expression solve_lagged uses, which at tau = 0 reduces to the covariance the
    ordinary solve matches. Summed directly over the solved bins rather than by an inverse
    transform: there are only a few dozen of them and the direct form is checkable."""
    out = np.zeros((len(lags), Phi.shape[1], Phi.shape[1]))
    fb = np.asarray(idx, float)
    for n, L in enumerate(lags):
        ph = np.exp(2j * np.pi * fb * (L / ref_frames))
        out[n] = np.real(np.einsum("f,fij->ij", w * ph, Phi)) * 2.0
    return out


def filter_autocorr(nb, ref_frames, lags, resp):
    """rho(tau) = the autocorrelation of the observable's filter, at those lags.

    The envelope is squared, THEN filtered, so what weights each lag is the filter applied
    twice - its autocorrelation - not the filter itself."""
    rho_full = np.fft.irfft(np.abs(resp) ** 2, n=ref_frames)
    return rho_full[np.asarray(lags) % ref_frames]


def envelope_cov(Clag, rho):
    """cov(u_i, u_j) for u = filter(h^2), in closed form.

    The drive is Gaussian and the medium linear, so the field is Gaussian and Wick gives
    cov(h_i(t)^2, h_j(s)^2) = 2 C_ij(t-s)^2 exactly. Filtering then weights each lag by
    the filter's autocorrelation:

        cov(u_i, u_j) = 2 sum_tau rho(tau) C_ij(tau)^2

    The squaring is what moves power between frequencies: C_ij(tau)^2 in time is the
    cross-spectrum convolved with itself in frequency, so a pair of fast components at f1
    and f2 contributes at |f1 - f2|. That is the down-conversion a linear observable
    cannot produce, and it is why a fast medium can carry slow observable structure."""
    return 2.0 * np.einsum("t,tij->ij", rho, Clag ** 2)


def interp_S(S, idx, ref_frames, nframes):
    """S on every bin of an nframes grid, exactly as xspec.realise interpolates it.

    realise does NOT drive the solved bins as discrete lines - it interpolates S linearly
    in frequency across the whole grid, so the drive's spectrum is a continuum between the
    solved points. A closed form that sums the sampled bins with band weights describes a
    different process: same total power, different shape, and a C(tau) that diverges from
    it at long lags. That is the repo's "interpolation gap", +0.006 for zero-lag FC and
    harmless there, but an envelope observable integrates C(tau)^2 over every lag and is
    far more exposed to it."""
    K = S.shape[1]
    nb = nframes // 2 + 1
    f_src = np.asarray(idx, float) / float(ref_frames)
    f_dst = np.arange(nb) / float(nframes)
    out = np.empty((nb, K, K), complex)
    for i in range(K):
        for j in range(K):
            out[:, i, j] = (np.interp(f_dst, f_src, S[:, i, j].real, left=0.0, right=0.0)
                            + 1j * np.interp(f_dst, f_src, S[:, i, j].imag,
                                             left=0.0, right=0.0))
    return out


def lagged_from_full(Phi, nframes):
    """C_ij(tau) for every lag, from a cross-spectrum defined on EVERY bin.

    C(tau) = sum_k 2 Re(Phi_k e^{i 2 pi k tau / N}) over k >= 1, which is what irfft
    computes once DC is removed - an FFT instead of a sum over sampled bins, so the full
    grid costs no more than the sparse one."""
    Z = np.array(Phi, complex, copy=True)
    Z[0] = 0.0
    return np.fft.irfft(Z, n=nframes, axis=0) * nframes


def env_forward(H, w, S, ph, rho):
    """-> (E, C). The envelope covariance and the lagged field covariances behind it.

    UNCENTRED, unlike lagged.model_lagged: the square has to happen before any centring,
    since centring is a linear operation on the observable and the observable here is
    h^2, not h."""
    nf, nV, _ = H.shape
    # as one gemm rather than nf x nlag Python iterations: at 192 bins and 55 lags the
    # loop is ~10,000 passes per objective evaluation and a line search wants hundreds
    M = np.einsum("fva,fab,fwb->fvw", H, S, H.conj())
    C = 2.0 * np.real((ph * w[None, :]) @ M.reshape(nf, -1)).reshape(-1, nV, nV)
    return 2.0 * np.einsum("t,tij->ij", rho, C ** 2), C


def env_adjoint(H, w, G, C, ph, rho):
    """dJ/dS given dJ/dE = G, chained through E = 2 sum_tau rho(tau) C(tau)^2.

    dE/dC_k = 4 rho_k C_k elementwise, then the same adjoint lagged.adjoint_lagged
    derives for the linear map S -> Phi(tau): w_f 2 exp(-i th) H^H M H, Hermitianised.
    Without the centring adjoint, to match env_forward."""
    Ms = (4.0 * rho[:, None, None]) * C * G[None]
    nf, nV, K = H.shape
    Y = (np.conj(ph).T @ Ms.reshape(ph.shape[0], -1)).reshape(nf, nV, nV)
    out = np.empty((nf, K, K), complex)
    for f in range(nf):
        acc = (w[f] * 2.0) * (H[f].conj().T @ Y[f] @ H[f])
        out[f] = 0.5 * (acc + acc.conj().T)
    return out


def solve_envelope(H, w, ph, rho, T, iters=200, S0=None, verbose=False):
    """Projected gradient for max corr(E(S), T) over S >= 0, E the envelope covariance.

    The same scheme xspec.solve uses on a ratio objective over the PSD cone, with one
    difference that is the whole question: E is QUADRATIC in S, where the FC objective is
    linear in it. Convexity is therefore not given, and random restarts are the test."""
    nf, nV, K = H.shape
    S = (np.stack([np.eye(K, dtype=complex) for _ in range(nf)]) if S0 is None
         else np.array(S0, complex, copy=True))
    iu = np.triu_indices(nV, 1)
    t = np.asarray(T[iu], float); t = (t - t.mean()); t /= max(np.linalg.norm(t), 1e-300)

    def obj(S):
        E, C = env_forward(H, w, S, ph, rho)
        e = E[iu] - E[iu].mean()
        n = max(np.linalg.norm(e), 1e-300)
        return float(e @ t / n), E, C, n

    val, E, C, n = obj(S)
    step = 1.0
    for it in range(iters):
        # d/dE of <e_hat, t>: (t - val * e_hat) / n, scattered back to a full matrix
        g = (t - val * (E[iu] - E[iu].mean()) / n) / n
        # HALF on each of (i,j) and (j,i): the objective reads the upper triangle only,
        # while env_adjoint sums over every entry, so mirroring g unhalved counts each
        # off-diagonal twice. Caught by finite differences as an exact factor of 2.
        G = np.zeros((nV, nV)); G[iu] = g; G = 0.5 * (G + G.T)
        grad = env_adjoint(H, w, G, C, ph, rho)
        moved = False
        for _ in range(40):
            Tn = xspec._project(S + step * grad, 1, False, None)
            tr = sum(np.trace(Tn[f]).real for f in range(nf))
            if tr > 0:
                Tn = Tn / tr
            v2, E2, C2, n2 = obj(Tn)
            if v2 > val:
                S, val, E, C, n = Tn, v2, E2, C2, n2
                step *= 1.6; moved = True
                break
            step *= 0.4
        if not moved:
            break
        if verbose and it % 25 == 0:
            print(f"      iter {it:4d}  corr {val:+.6f}", flush=True)
    return S, val


def corr(a, b):
    a = np.asarray(a, float).ravel(); b = np.asarray(b, float).ravel()
    a = a - a.mean(); b = b - b.mean()
    return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30))


def simulate_envelope(c, p, P, keep, S, idx, pad, save, nframes, kern, frame_s, band,
                      amp=2e-4, seed=0, filtered=True):
    """Realise a drive from S, integrate, and form the envelope observable for real.

    The closed form is only worth anything if it matches this. Squares the field FIRST,
    then applies the same kernel and passband the linear observable applies to the field
    itself - the ordering is the whole point, since filtering before squaring removes the
    carriers before they can beat."""
    import fluid as fl
    from xspec import ProfileDrive
    # The drive realised at `nframes` is periodic with that period, but the field starts
    # from rest, so a single period is a transient and not the stationary response the
    # closed form describes. Driving two periods and keeping the second gives the periodic
    # steady state - which is what C(tau) from a cross-spectrum actually is.
    A1 = xspec.realise(S, idx, nframes, ref_frames=pad, seed=seed)
    A = np.concatenate([A1, A1], axis=0)
    nframes = 2 * nframes
    nsteps = nframes * save
    Aser = np.repeat(A, save, axis=0)[:nsteps] / save
    Aser = (Aser * (amp / np.sqrt((Aser ** 2).mean()))).astype(np.float32)
    frames, _ = fl.run(c, ProfileDrive(c, P, Aser, amp), p, nsteps, save)
    X = np.asarray(frames[:, keep], np.float64)[nframes // 2:]    # second period only
    if not filtered:
        return X, None, None
    lin = bandpass.apply(units.smooth_frames(X, kern), frame_s, *band)
    env = bandpass.apply(units.smooth_frames(X ** 2, kern), frame_s, *band)
    return X, lin, env


def cmat(X, burn=50):
    """Correlation over vertices from a (T, V) series, dropping the filter transient."""
    Z = np.asarray(X[burn:], float)
    Z = Z - Z.mean(0, keepdims=True)
    sd = np.maximum(Z.std(0, keepdims=True), 1e-300)
    return (Z / sd).T @ (Z / sd) / Z.shape[0]


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
    ap.add_argument("--envelope", action="store_true",
                    help="compare the LINEAR observable against the ENVELOPE one "
                         "(square, then filter) on the same known drive, and check the "
                         "closed form against a simulation")
    ap.add_argument("--sim-reps", type=int, default=24, dest="sim_reps")
    ap.add_argument("--grid-sweep", default="", dest="grid_sweep",
                    help="comma-separated nfreq values. Fixes S_true on the FULL rfft "
                         "grid, builds the target from it, then solves at each density "
                         "against that one truth - so only the model's resolution varies, "
                         "where regenerating S_true per grid would change the problem too. "
                         "Runs both objectives: the linear one sees C(0) alone, the "
                         "envelope sees C(tau) at every lag, so the spectrum's fine "
                         "structure enters them differently")
    ap.add_argument("--solve-envelope", action="store_true", dest="solve_env",
                    help="fit S against an ENVELOPE target: finite-difference check the "
                         "gradient, then solve from random restarts. E(S) is quadratic in "
                         "S where the FC objective is linear, so convexity is the question")
    ap.add_argument("--burn", type=int, default=128)
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
    # transfer() folds the kernel INTO H, which is what best_fit wants: its observable is
    # linear in the field, so every filter can be pushed into the transfer function. An
    # envelope observable cannot do that - it is smooth(h^2), not smooth(h)^2, so the
    # square has to act on the RAW field and the filtering has to come after it. H_raw is
    # the same transfer function with no filter applied, on the same bins.
    H_raw = xspec.transfer(R, np.arange(a.nvert), a.nfreq, kernel=None, idx=idx)[0]
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

    if a.grid_sweep:
        import lagged as _lg
        nb = pad // 2 + 1
        iu = np.triu_indices(a.nvert, 1)
        kr = units.kernel_response(kern, nb, pad)
        resp = kr * bandpass.response(np.arange(nb) / (pad * cl["frame_s"]),
                                      cl["frame_s"], lo, hi)
        rho_all = filter_autocorr(nb, pad, np.arange(pad), resp)
        m = np.abs(rho_all) > 1e-8 * np.abs(rho_all).max()
        lags, rho = np.arange(pad)[m], rho_all[m]

        # the TRUTH, on every bin, held fixed across the sweep
        idx_t = np.arange(1, nb)
        H_t = xspec.transfer(R, np.arange(a.nvert), 0, kernel=None, idx=idx_t)[0]
        w_t = np.ones(len(idx_t))
        f_t = idx_t / (pad * cl["frame_s"])
        sel_t = {"all": None, "slow": (f_t >= lo) & (f_t <= hi),
                 "fast": ~((f_t >= lo) & (f_t <= hi))}[a.true_band]
        St = psd_truth(len(idx_t), K, a.rank, np.random.default_rng(a.seed), keep=sel_t)
        ph_t = _lg.phases(idx_t, pad, lags)
        T_lin = forward(H_t * resp[idx_t][:, None, None], w_t, St)
        T_env, _ = env_forward(H_t, w_t, St, ph_t, rho)
        marg_t = np.real(St.sum(0))
        indep = imp // 2
        print(f"\n  truth fixed on all {len(idx_t)} bins; the impulse window is {imp} "
              f"frames, so only ~{indep} of them are independent - above that a denser "
              f"grid interpolates rather than informs, and recovery should flatten")
        print(f"\n  {'nfreq':>6} {'bins':>5} | {'LINEAR':>26} | {'ENVELOPE':>26}")
        print(f"  {'':>6} {'':>5} | {'corr(C,T)':>12} {'corr(sum S)':>13} "
              f"| {'corr(E,T)':>12} {'corr(sum S)':>13}")
        for nq in [int(v) for v in a.grid_sweep.split(",") if v.strip()]:
            ii = np.unique(np.round(np.geomspace(1, nb - 1, nq)).astype(int))
            Hc = xspec.transfer(R, np.arange(a.nvert), 0, kernel=None, idx=ii)[0]
            wc = np.gradient(ii).astype(float)
            Sl, Cl = xspec.solve(Hc * resp[ii][:, None, None], wc, T_lin,
                                 iters=a.iters, verbose=False)
            ph_c = _lg.phases(ii, pad, lags)
            Se, _ = solve_envelope(Hc, wc, ph_c, rho, T_env, iters=a.iters)
            Ee, _ = env_forward(Hc, wc, Se, ph_c, rho)
            print(f"  {nq:>6} {len(ii):>5} | {corr(Cl[iu], T_lin[iu]):>+12.6f} "
                  f"{corr(np.real(Sl.sum(0)), marg_t):>+13.4f} "
                  f"| {corr(Ee[iu], T_env[iu]):>+12.6f} "
                  f"{corr(np.real(Se.sum(0)), marg_t):>+13.4f}", flush=True)
        return

    if a.solve_env:
        import lagged as _lg
        nb = pad // 2 + 1
        kr = units.kernel_response(kern, nb, pad)
        rho_all = filter_autocorr(nb, pad, np.arange(pad), kr)
        m = np.abs(rho_all) > 1e-8 * np.abs(rho_all).max()
        lags, rho = np.arange(pad)[m], rho_all[m]
        ph = _lg.phases(idx, pad, lags)
        iu = np.triu_indices(a.nvert, 1)
        print(f"\n  envelope objective: {len(lags)} lags carry the kernel "
              f"autocorrelation (of {pad})")

        # ---- the adjoint has to be right before anything downstream means anything
        r3 = np.random.default_rng(7)
        Sg = psd_truth(len(idx), K, K, r3)
        Tg, _ = env_forward(H_raw, w, S_true, ph, rho)
        tv = Tg[iu] - Tg[iu].mean(); tv /= max(np.linalg.norm(tv), 1e-300)

        def J(S):
            E, C = env_forward(H_raw, w, S, ph, rho)
            e = E[iu] - E[iu].mean()
            return float(e @ tv / max(np.linalg.norm(e), 1e-300)), E, C

        v0, E0, C0 = J(Sg)
        g0 = (tv - v0 * (E0[iu] - E0[iu].mean())
              / max(np.linalg.norm(E0[iu] - E0[iu].mean()), 1e-300))
        g0 = g0 / max(np.linalg.norm(E0[iu] - E0[iu].mean()), 1e-300)
        G0 = np.zeros((a.nvert, a.nvert)); G0[iu] = g0; G0 = 0.5 * (G0 + G0.T)
        grad = env_adjoint(H_raw, w, G0, C0, ph, rho)
        D = r3.normal(size=(len(idx), K, K)) + 1j * r3.normal(size=(len(idx), K, K))
        D = 0.5 * (D + np.conj(np.transpose(D, (0, 2, 1))))
        D /= np.linalg.norm(D)
        ana = float(sum(np.real(np.trace(grad[f].conj().T @ D[f]))
                        for f in range(len(idx))))
        print("  finite-difference check of the adjoint:")
        for eps in (1e-5, 1e-6):
            num = (J(Sg + eps * D)[0] - J(Sg - eps * D)[0]) / (2 * eps)
            print(f"    eps {eps:.0e}: analytic {ana:+.8f}  numeric {num:+.8f}  "
                  f"rel err {abs(num-ana)/max(abs(ana),1e-30):.2e}")

        # ---- recover a known drive through the envelope observable
        print(f"\n  recovering the true drive from an ENVELOPE target, "
              f"{a.starts} restarts:")
        vals, sims, smarg = [], [], []
        for st in range(a.starts):
            r4 = np.random.default_rng(2000 + st)
            S0 = psd_truth(len(idx), K, K, r4)
            Sh, v = solve_envelope(H_raw, w, ph, rho, Tg, iters=a.iters, S0=S0)
            Eh, _ = env_forward(H_raw, w, Sh, ph, rho)
            vals.append(v)
            sims.append(corr(Eh[iu], Tg[iu]))
            smarg.append(corr(np.real(Sh.sum(0)), np.real(S_true.sum(0))))
            print(f"    start {st}: objective {v:+.6f}  corr(E_hat, T) {sims[-1]:+.6f}  "
                  f"corr(sum_f S) {smarg[-1]:+.4f}", flush=True)
        rel = np.std(vals) / max(abs(np.mean(vals)), 1e-30)
        print(f"  objective {np.mean(vals):+.6f}, spread {np.std(vals):.2e} "
              f"({rel:.1e} relative) "
              + ("- one optimum, behaves convexly" if rel < 1e-3
                 else "- STARTS DISAGREE, not convex on this problem"))
        return

    if a.envelope:
        nb = pad // 2 + 1
        lags = np.arange(pad)
        Phi = np.einsum("fva,fab,fwb->fvw", H_raw, S_true, H_raw.conj())
        Clag = field_lagged_cov(Phi, w, idx, pad, lags)
        iu = np.triu_indices(a.nvert, 1)

        def rmat(M):
            d = np.sqrt(np.maximum(np.diag(M), 1e-300))
            return M / np.outer(d, d)

        # Validate Wick against simulation with the SMOOTHING KERNEL ONLY. The passband is
        # just another linear filter, but its impulse response is ~620 frames at 0.01 Hz,
        # so including it here would test the burn-in rather than the algebra. And the
        # realisations are exactly `pad` frames long, because realise() INTERPOLATES S
        # from the solved grid onto the realisation grid - at any other length the field's
        # spectrum is not the Phi the closed form assumes, and the comparison measures the
        # interpolation instead.
        kr = units.kernel_response(kern, nb, pad)
        # On the FULL grid, interpolated the way realise interpolates, so the closed form
        # describes the process actually simulated rather than one with the same total
        # power in a different shape.
        Hf, wf, idxf = xspec.transfer(R, np.arange(a.nvert), 0, kernel=None,
                                      idx=np.arange(1, nb))
        Sf = interp_S(S_true, idx, pad, pad)[1:]
        Phif = np.einsum("fva,fab,fwb->fvw", Hf, Sf, Hf.conj())
        Pfull = np.zeros((nb,) + Phif.shape[1:], complex)
        Pfull[1:] = Phif
        Clag_f = lagged_from_full(Pfull, pad)
        print(f"    C(0) from the sampled bins vs the full grid: corr "
              f"{corr(Clag[0][iu], Clag_f[0][iu]):+.4f}")
        rho_k = filter_autocorr(nb, pad, lags, kr)
        Cenv_k = envelope_cov(Clag_f, rho_k)
        Clin_k = Clag_f[0] * 0.0
        for n, L in enumerate(lags):
            pass
        Clin_k = np.einsum("t,tij->ij", rho_k, Clag_f)      # linear: filter, do not square

        print(f"\n  === validating the closed form, kernel only, "
              f"{a.sim_reps} x {pad} frames ===", flush=True)
        t1 = time.time()
        Ae = np.zeros((a.nvert, a.nvert)); Al = np.zeros((a.nvert, a.nvert))
        for rep in range(a.sim_reps):
            X, _, _ = simulate_envelope(c, p, P, keep, S_true, idx, pad, save, pad,
                                        kern, cl["frame_s"], (lo, hi), seed=100 + rep,
                                        filtered=False)
            e = units.smooth_frames(X ** 2, kern)[a.burn:]
            l = units.smooth_frames(X, kern)[a.burn:]
            Ae += np.cov(e, rowvar=False); Al += np.cov(l, rowvar=False)
        print(f"    simulated [{time.time()-t1:.0f}s]")
        print(f"    envelope: corr(closed form, simulated) = "
              f"{corr(rmat(Cenv_k)[iu], rmat(Ae)[iu]):+.4f}")
        print(f"    linear:   corr(closed form, simulated) = "
              f"{corr(rmat(Clin_k)[iu], rmat(Al)[iu]):+.4f}")

        # What the two observables carry, with the passband applied to each
        resp = (kr * bandpass.response(np.arange(nb) / (pad * cl["frame_s"]),
                                       cl["frame_s"], lo, hi))
        Cenv = envelope_cov(Clag, filter_autocorr(nb, pad, lags, resp))
        Clin = forward(H_raw * resp[np.asarray(idx)][:, None, None], w, S_true)
        print(f"\n  === through the passband, true drive '{a.true_band}' ===")
        print(f"    linear   observable variance  {np.mean(np.diag(Clin)):.4e}")
        print(f"    envelope observable variance  {np.mean(np.diag(Cenv)):.4e}")
        print(f"    envelope keeps {np.mean(np.diag(Cenv))/max(np.mean(np.diag(Clin)),1e-300):.2e}x "
              f"the in-band variance of the linear one")
        print(f"    corr(envelope FC, linear FC) = {corr(rmat(Cenv)[iu], rmat(Clin)[iu]):+.4f}"
              f"   (1.0 would mean the envelope sees nothing new)")
        return

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
