"""Input by hand: K Glasser regions, each with a timecourse you supply.

Copy of input_model.py with the latent/loading machinery removed. There are no
latents, no loading matrix, no sparsity or tau genes and no balance projection -
you give one timecourse per region and that is exactly what drives it.

    S(x,t) = sum_k a_k(t) * taper_k(x)

`parcel_tapers` is unchanged from input_model, so the spatial profiles are the
same ones every other part of the pipeline uses.
"""
import os
import numpy as np

from paths import CACHE


# ---------------------------------------------------------------- region profiles
def _erode_once(mask, edges, nb_count, nV):
    c = np.zeros(nV)
    np.add.at(c, edges[:, 0], mask[edges[:, 1]].astype(float))
    np.add.at(c, edges[:, 1], mask[edges[:, 0]].astype(float))
    return mask & (c >= nb_count)


def parcel_tapers(c, verbose=True):
    """(nParcels, nV) profiles: peak 1 at each parcel's core, 0 at its boundary,
    support exactly the parcel. Same cache as input_model."""
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


# ---------------------------------------------------------------- the input object
class RegionDrive:
    """K regions, K timecourses. Interface-compatible with NetworkDrive: .Aser,
    .P, .source(n), .report().

    timecourses  (nsteps, K) or (K, nsteps); a 1-D array is allowed when K == 1.
                 Rows shorter than nsteps are an error, longer are truncated.
    amp          overall scale.
    normalise    True  -> timecourses are scaled to unit RMS as a set before amp,
                         so amp alone sets the size (what the rest of the
                         pipeline assumes, and what keeps runs in the linear
                         regime at the usual amp ~ 2e-4).
                 False -> the numbers you give are used as they are and amp is a
                         plain multiplier.

    Your timecourse is what is ADDED to the fluid at that region each step, so the
    depth you see is its running integral.
    """

    def __init__(self, cortex, region_ids, timecourses, amp=1.0, nsteps=None,
                 tapers=None, normalise=True):
        self.c = cortex
        self.region_ids = np.asarray(region_ids, int)
        self.K = len(self.region_ids)
        self.amp = float(amp)

        T, ids = parcel_tapers(cortex, verbose=False) if tapers is None else tapers
        pos = {int(p): i for i, p in enumerate(ids)}
        self.P = np.ascontiguousarray(T[[pos[int(p)] for p in self.region_ids]])  # (K, nV)

        # area-integral of each region's profile: the weight its amplitude carries
        self.w = (self.P.astype(np.float64) * cortex.A[None, :]).sum(1)

        a = np.atleast_2d(np.asarray(timecourses, float))
        if a.shape[0] == self.K and a.shape[1] != self.K:
            a = a.T                                   # (K, nsteps) -> (nsteps, K)
        if a.shape[1] != self.K:
            raise ValueError(f"timecourses have {a.shape[1]} columns "
                             f"for {self.K} regions")
        if nsteps is not None:
            if len(a) < nsteps:
                raise ValueError(f"timecourses are {len(a)} steps, run is {nsteps}")
            a = a[:nsteps]
        if not np.isfinite(a).all():
            raise ValueError("non-finite value in the timecourses")

        rms = float(np.sqrt((a*a).mean()))
        self.struct_rms = rms
        self.Aser = a*(self.amp/rms) if (normalise and rms > 1e-12) else a*self.amp
        self.dead = bool(rms <= 1e-12)

    def source(self, n):
        """Per-vertex height source at step n."""
        return self.Aser[n] @ self.P

    # ------------------------------------------------------------ diagnostics
    def report(self):
        a = self.Aser
        return dict(
            # NOT zero by construction here: nothing is projected or balanced, so
            # this is however much net depth your timecourses add or remove.
            mass_residual=float(np.abs(a @ self.w).max()),
            mass_residual_mean=float(np.abs((a @ self.w).mean())),
            silent=[float((a[:, k] == 0).mean()) for k in range(self.K)],
            corr_agreement=float("nan"),        # no intended covariance to compare to
            amp_rms=float(np.sqrt((a*a).mean())),
            region_w=self.w.copy(),
        )

    def describe(self):
        d = self.report()
        print(f"  K={self.K} regions, {len(self.Aser)} steps, amp={self.amp:.3g}")
        print(f"  net mass |sum_k w_k a_k|: max {d['mass_residual']:.3e}, "
              f"mean {d['mass_residual_mean']:.3e}")
        print(f"  fraction of steps at exactly zero: " +
              ", ".join(f"{s*100:.0f}%" for s in d["silent"]))
        return d
