"""Parameterised input: K Glasser regions driven through r shared latent factors.

    s(t)  in R^r    latent processes: OU, smoothed, soft-thresholded to a TARGET
                    SPARSITY, sign-balanced, unit RMS. Sparsity is specified as the
                    fraction of the run with no drive, and the threshold follows
                    from the matching quantile - so every random draw comes out
                    equally sparse, and no latent can end up entirely dead.

    a(t) = L s(t)   region amplitudes. Regions loading on a shared latent
                    coactivate; a region's sign is the sign of its loading.

    S(x,t) = sum_k a_k(t) * taper_k(x)

Mass balance: with w_k the area-integral of region k's taper, the columns of L are
projected orthogonal to w at construction, so sum_k w_k a_k(t) = 0 for ANY latent
values, exactly and at every step. Nothing is added to or removed from the surface
as a whole - the drive only moves depth between the named regions.
"""
import os
import numpy as np
from scipy.signal import lfilter

from paths import CACHE
SMOOTH_FRAC = 0.2          # Gaussian smoothing of each latent, as a fraction of tau


# ---------------------------------------------------------------- region profiles
def _erode_once(mask, edges, nb_count, nV):
    c = np.zeros(nV)
    np.add.at(c, edges[:, 0], mask[edges[:, 1]].astype(float))
    np.add.at(c, edges[:, 1], mask[edges[:, 0]].astype(float))
    return mask & (c >= nb_count)


def parcel_tapers(c, verbose=True):
    """(nParcels, nV) profiles: peak 1 at each parcel's core, 0 at its boundary,
    support exactly the parcel. Blurring a boxcar instead would spread past the
    boundary and distort small parcels far more than large ones."""
    cache = os.path.join(CACHE, f"tapers_{c.mesh}.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        return z["T"], z["ids"]
    nb = np.zeros(c.nV)
    np.add.at(nb, c.edges[:, 0], 1); np.add.at(nb, c.edges[:, 1], 1)
    T = np.zeros((len(c.parcels), c.nV), np.float32)
    for row, pid in enumerate(c.parcels):
        mask = (c.lab == pid)
        depth = np.zeros(c.nV); cur = mask.copy(); k = 0
        while cur.any():
            k += 1
            depth[cur] = k
            cur = _erode_once(cur, c.edges, nb, c.nV)
        if k:
            u = depth / depth.max()
            T[row] = (u*u*(3.0 - 2.0*u)).astype(np.float32)
        else:
            T[row] = mask.astype(np.float32)
    np.savez(cache, T=T, ids=c.parcels)
    if verbose:
        print(f"[{c.mesh}] built tapers for {len(c.parcels)} parcels")
    return T, c.parcels


# ---------------------------------------------------------------- latents
def _gauss(x, sigma_steps):
    if sigma_steps <= 0:
        return x
    k = np.arange(-int(4*sigma_steps), int(4*sigma_steps)+1)
    w = np.exp(-0.5*(k/sigma_steps)**2); w /= w.sum()
    return np.convolve(x, w, mode="same")


def latent_series(nsteps, dt, tau, silent, seed):
    """One latent: OU -> smooth -> z-score -> soft threshold -> sign balance -> unit RMS.

    `silent` is the TARGET fraction of the run with no drive, and the threshold is
    set to the matching quantile of |x|. Specifying sparsity rather than a raw
    threshold means every random draw yields the same sparsity, which removes one
    source of seed-to-seed variation, and no latent can come out entirely dead."""
    a = np.exp(-dt/max(tau, 1e-9)); sd = np.sqrt(max(1 - a*a, 0.0))
    rng = np.random.default_rng(seed)
    # x[i] = a x[i-1] + sd eps[i] is a first-order recursion, so lfilter runs it in
    # C rather than a Python loop; zi carries the stationary initial condition.
    x0 = rng.standard_normal()
    eps = rng.standard_normal(nsteps)
    x = lfilter([sd], [1.0, -a], eps, zi=np.array([a*x0]))[0]
    x = _gauss(x, SMOOTH_FRAC*tau/dt)
    x = (x - x.mean())/max(x.std(), 1e-12)
    if silent > 0:
        th = np.quantile(np.abs(x), min(max(silent, 0.0), 0.999))
        x = np.sign(x)*np.maximum(np.abs(x) - th, 0.0)
    P = x[x > 0].sum(); N = -x[x < 0].sum()
    if P > 0 and N > 0:                              # match above/below zero
        x[x > 0] *= np.sqrt(N/P); x[x < 0] *= np.sqrt(P/N)
    r = np.sqrt((x*x).mean())
    return x/r if r > 1e-12 else x


# ---------------------------------------------------------------- the input object
class NetworkDrive:
    """Build once per genome; call with a step index to get the source field."""

    def __init__(self, cortex, region_ids, loadings, silents, taus,
                 amp, nsteps, dt, seed=0, tapers=None, balance="spatial"):
        self.c = cortex
        self.region_ids = np.asarray(region_ids, int)
        self.K = len(self.region_ids)
        self.r = np.asarray(loadings).shape[1]
        self.amp = float(amp)

        T, ids = parcel_tapers(cortex, verbose=False) if tapers is None else tapers
        pos = {int(p): i for i, p in enumerate(ids)}
        self.P = np.ascontiguousarray(T[[pos[int(p)] for p in self.region_ids]])  # (K, nV)

        # area-integral of each region's profile: the weight its amplitude carries
        self.w = (self.P.astype(np.float64) * cortex.A[None, :]).sum(1)

        L = np.array(loadings, float).reshape(self.K, self.r)
        ww = float(self.w @ self.w)
        self.L_raw = L.copy()
        self.balance = balance
        if balance == "spatial":
            # sum_k w_k a_k(t) = 0 at EVERY step: the drive only moves depth
            # between the named regions and the surface as a whole is untouched.
            self.L = L - np.outer(self.w, (self.w @ L)/ww)     # columns _|_ w
        elif balance == "temporal":
            # Each latent is already zero-mean and sign-balanced over the run
            # (see latent_series), so the drive injects no net mass ACROSS THE RUN
            # while the surface does rise and fall as a whole at any instant.
            # This is the only option that admits a single driven region: with
            # K = 1 the spatial projection annihilates every loading, because the
            # only vector orthogonal to a 1-vector w is zero.
            self.L = L
        else:
            raise ValueError(f"balance must be 'spatial' or 'temporal', got {balance!r}")
        self.Sigma = self.L @ self.L.T                     # the covariance actually imposed

        self.S = np.stack([latent_series(nsteps, dt, taus[j], silents[j], seed*97 + j)
                           for j in range(self.r)], axis=1)      # (nsteps, r)
        a = self.S @ self.L.T                                    # (nsteps, K)
        # Normalise to unit RMS before applying amp. Without this, scaling L and
        # scaling amp do the same thing, so the search space contains a whole
        # degenerate direction. After it, L carries only structure and amp only size.
        rms = np.sqrt((a*a).mean())
        self.struct_rms = float(rms)
        self.Aser = a*(self.amp/rms) if rms > 1e-12 else a
        self.dead = bool(rms <= 1e-12)

    def source(self, n):
        """Per-vertex height source at step n. Zero net mass by construction."""
        return self.Aser[n] @ self.P

    # ------------------------------------------------------------ diagnostics
    def report(self):
        a = self.Aser
        mass = np.abs(a @ self.w).max()
        corr_real = np.corrcoef(a.T) if self.K > 1 else np.array([[1.0]])
        d = np.sqrt(np.clip(np.diag(self.Sigma), 1e-30, None))
        corr_want = self.Sigma/np.outer(d, d)
        iu = np.triu_indices(self.K, 1)
        agree = (np.corrcoef(corr_real[iu], corr_want[iu])[0, 1]
                 if self.K > 2 else float("nan"))
        return dict(
            mass_residual=float(mass),
            silent=[float((self.S[:, j] == 0).mean()) for j in range(self.r)],
            corr_agreement=float(agree),
            amp_rms=float(np.sqrt((a*a).mean())),
            region_w=self.w.copy(),
        )

    def describe(self):
        d = self.report()
        print(f"  K={self.K} regions, r={self.r} latents, amp={self.amp:.3g}")
        print(f"  mass residual |sum_k w_k a_k| = {d['mass_residual']:.3e} (0 by construction)")
        print(f"  latent silent fractions: " +
              ", ".join(f"{s*100:.0f}%" for s in d["silent"]))
        print(f"  realised vs intended region correlation: r = {d['corr_agreement']:.4f}")
        return d
