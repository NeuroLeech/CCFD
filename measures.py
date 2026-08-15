"""Compute a wide set of candidate measures on the stored vertex x time fields,
so they can be judged by whether they actually separate runs.

Nothing here is a fitness function. The output is a table plus a redundancy
structure: which measures move together, which have any spread at all, and which
survive being computed on half the data.

Every spatial measure uses the same rim exclusion as wave_measures (20 mm from the
medial wall). The sponge sits at that rim, so including it would let the boundary
condition masquerade as a property of the field.

For each per-frame quantity three numbers are kept - time-mean, coefficient of
variation over time, and decorrelation time of that series - because "structure
that is changing" lives in the second and third, not the first.
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra, connected_components

from mesh_cache import load_cortex
from wave_measures import WaveMeasures

from paths import FIELDS
N_SAMPLE = 500          # vertices for distance-based measures
CLUSTER_EVERY = 8       # frames, for connected-component counting
FLOW_EVERY = 2          # frames, for the structure-tensor flow estimate


# ------------------------------------------------------------------ helpers
def _decorr(x, dt, frac=np.exp(-1.0)):
    """1/e decorrelation time of a scalar series, in time units."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 8 or x.std() < 1e-20:
        return float("nan")
    x = x - x.mean()
    ac = np.correlate(x, x, mode="full")[n-1:]
    ac /= ac[0]
    below = np.flatnonzero(ac < frac)
    return float(below[0]*dt) if len(below) else float(n*dt)


def _cv(x):
    """Coefficient of variation, guarded so a near-zero mean cannot explode it."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return float("nan")
    m = np.abs(x.mean())
    return float(x.std()/m) if m > 1e-30 else float("nan")


def _summ(name, series, dt, out):
    out[f"{name}_mean"] = float(np.nanmean(series))
    out[f"{name}_cv"] = _cv(series)
    out[f"{name}_tau"] = _decorr(series, dt)


class Context:
    """Everything reusable across fields: mesh, masks, distances, sampled pairs."""

    def __init__(self, mesh="fsaverage5"):
        self.c = load_cortex(mesh, verbose=False)
        c = self.c
        self.wm = WaveMeasures(c, exclude_mm=20.0)
        self.wm.baseline()
        self.keep = self.wm.keep
        self.kidx = np.flatnonzero(self.keep)
        self.A = c.A[self.keep]
        self.Asum = float(self.A.sum())

        # geodesic distances among a sample of kept vertices
        we = np.linalg.norm(c.V[c.edges[:, 0]] - c.V[c.edges[:, 1]], axis=1)
        self.G = csr_matrix((np.r_[we, we],
                             (np.r_[c.edges[:, 0], c.edges[:, 1]],
                              np.r_[c.edges[:, 1], c.edges[:, 0]])),
                            shape=(c.nV, c.nV))
        rng = np.random.default_rng(0)
        self.samp = np.sort(rng.choice(self.kidx, min(N_SAMPLE, len(self.kidx)),
                                       replace=False))
        D = dijkstra(self.G, indices=self.samp, directed=False)[:, self.samp]
        self.D = D
        iu = np.triu_indices(len(self.samp), 1)
        self.iu = iu
        self.pd = D[iu]
        self.bins = np.arange(0.0, np.nanpercentile(self.pd, 95), 6.0)
        self.bi = np.digitize(self.pd, self.bins) - 1

        # adjacency restricted to kept vertices, for connected components
        ek = c.edges[self.keep[c.edges[:, 0]] & self.keep[c.edges[:, 1]]]
        pos = -np.ones(c.nV, np.int64)
        pos[self.kidx] = np.arange(len(self.kidx))
        e0, e1 = pos[ek[:, 0]], pos[ek[:, 1]]
        n = len(self.kidx)
        self.Adj = csr_matrix((np.ones(len(e0)*2),
                               (np.r_[e0, e1], np.r_[e1, e0])), shape=(n, n))
        self.V = c.V[self.keep]


# ------------------------------------------------------------------ measures
def compute(Hs, drive, dt_frame, ctx, driven_mask):
    """Hs: (nframes, nV) full-surface field. Returns a flat dict of measures."""
    o = {}
    c, keep, A, Asum = ctx.c, ctx.keep, ctx.A, ctx.Asum
    Hk = np.asarray(Hs[:, keep], np.float64)
    nF = len(Hk)

    # area-weighted demean per frame: only the pattern is of interest
    mu = (Hk*A).sum(1)/Asum
    Z = Hk - mu[:, None]
    var = (Z*Z*A).sum(1)/Asum                                   # spatial variance
    sd = np.sqrt(np.maximum(var, 1e-300))

    # ---- 10. energy trend: is the run anywhere near a steady state?
    t = np.arange(nF, dtype=float)
    lv = np.log(np.maximum(var, 1e-300))
    o["energy_logslope"] = float(np.polyfit(t, lv, 1)[0]*nF)     # log change over run
    o["energy_ratio"] = float(var[-nF//4:].mean()/max(var[:nF//4].mean(), 1e-300))

    # ---- 9. variance modulation, windowed so the trend cannot masquerade as it
    w = max(8, nF//12)
    o["var_mod"] = float(np.median([var[i:i+w].std()/max(var[i:i+w].mean(), 1e-300)
                                    for i in range(0, nF-w+1, w)]))

    # normalise each frame: pattern shape only, amplitude divided out
    U = Z/sd[:, None]

    # ---- 4,5,15,16 concentration and shape of each frame
    aw = A/Asum
    m2 = (U*U*aw).sum(1)
    m4 = (U**4*aw).sum(1)
    kurt = m4/np.maximum(m2*m2, 1e-300)
    pr = (m2*m2)/np.maximum(m4, 1e-300)          # participation ratio, area units
    absU = np.abs(U)
    thr = 0.2*absU.max(1, keepdims=True)
    active = ((absU > thr)*aw).sum(1)
    # energy fraction in the top 5% of AREA, ordered by |h|
    top = np.empty(nF)
    for k in range(nF):
        srt = np.argsort(-absU[k])
        ca = np.cumsum(aw[srt])
        cut = np.searchsorted(ca, 0.05) + 1
        e = U[k][srt]**2*aw[srt]
        top[k] = e[:cut].sum()/max(e.sum(), 1e-300)
    _summ("kurtosis", kurt, dt_frame, o)
    _summ("partratio", pr, dt_frame, o)
    _summ("active_frac", active, dt_frame, o)
    _summ("top5_energy", top, dt_frame, o)

    # ---- 2. gradient length scale  sqrt(int h^2 / int |grad h|^2)
    ls = np.empty(nF)
    for k in range(nF):
        hf = np.asarray(Hs[k], np.float64)
        g = c.m.tri_grad(hf)
        num = (c.A*hf*hf).sum()
        den = (c.m.tri_area*(g*g).sum(1)).sum()
        ls[k] = np.sqrt(num/max(den, 1e-30))
    _summ("length_scale", ls, dt_frame, o)

    # ---- 1. anisotropy from the structure tensor, mesh floor divided out
    A0 = ctx.wm.baseline()[keep]
    an = np.empty(nF)
    for k in range(nF):
        Jxx, Jyy, Jxy = ctx.wm.tensor(Hs[k])
        a_, _, tr = ctx.wm.anisotropy(Jxx, Jyy, Jxy)
        a_ = a_[keep]; trk = tr[keep]
        a_ = (a_ - A0)/np.maximum(1.0 - A0, 1e-9)
        wt = trk/max(trk.sum(), 1e-30)
        an[k] = float((a_*wt).sum())
    _summ("anisotropy", an, dt_frame, o)

    # ---- 12. normal-flow speed, subsampled
    sp = []
    for k in range(0, nF-FLOW_EVERY, FLOW_EVERY):
        dh = (np.asarray(Hs[k+FLOW_EVERY], np.float64)
              - np.asarray(Hs[k], np.float64))/(FLOW_EVERY*dt_frame)
        _, _, ns, _ = ctx.wm.flow(0.5*(np.asarray(Hs[k], np.float64)
                                       + np.asarray(Hs[k+FLOW_EVERY], np.float64)), dh)
        v = ns[keep]; v = v[np.isfinite(v)]
        if len(v):
            sp.append(np.median(v))
    o["speed_mean"] = float(np.median(sp)) if sp else float("nan")
    o["speed_cv"] = _cv(sp)

    # ---- 7,11. pattern turnover
    # the field starts flat, so the first frame has zero norm and would give 0/0
    nrm = np.linalg.norm(U, axis=1)
    Un = U[nrm > 1e-30]/nrm[nrm > 1e-30][:, None]
    nF = len(Un)
    o["f2f_r"] = float(np.abs((Un[1:]*Un[:-1]).sum(1)).mean())
    Gm = np.abs(Un @ Un.T)
    np.fill_diagonal(Gm, 0.0)
    o["allpair_r"] = float(Gm.sum()/(nF*(nF-1)))
    acf = np.array([float((Un[:nF-l]*Un[l:]).sum(1).mean())
                    for l in range(min(nF-1, 120)+1)])
    bl = np.flatnonzero(acf < np.exp(-1.0))
    o["pattern_tau"] = float(bl[0]*dt_frame) if len(bl) else float(len(acf)*dt_frame)

    # ---- 8. field timescale against the drive's, same estimator both sides
    ftau = float(np.nanmedian([_decorr(U[:, j], dt_frame)
                               for j in np.linspace(0, U.shape[1]-1, 64).astype(int)]))
    dtau = float(np.nanmedian([_decorr(drive[:, j], dt_frame)
                               for j in range(drive.shape[1])]))
    o["field_tau"], o["drive_tau"] = ftau, dtau
    o["tau_ratio"] = ftau/dtau if dtau > 1e-12 else float("nan")

    # ---- 3. spatial correlation length from the two-point function
    S = U[:, np.searchsorted(ctx.kidx, ctx.samp)]
    S = (S - S.mean(1, keepdims=True))/np.maximum(S.std(1, keepdims=True), 1e-30)
    Cm = (S.T @ S)/nF
    cv_ = Cm[ctx.iu]
    prof = np.array([np.nanmean(cv_[ctx.bi == b]) if (ctx.bi == b).any() else np.nan
                     for b in range(len(ctx.bins))])
    ok = np.isfinite(prof)
    o["corr_length"] = float("nan")
    if ok.sum() > 2:
        pv, bv = prof[ok], ctx.bins[ok]
        under = np.flatnonzero(pv < np.exp(-1.0))
        o["corr_length"] = float(bv[under[0]]) if len(under) else float(bv[-1])

    # ---- 14. propagation speed from lag against distance, no phase needed
    F = np.fft.rfft(S - S.mean(0), axis=0, n=2*nF)
    lags = np.empty(len(ctx.pd))
    step = 64
    for a in range(0, S.shape[1], step):
        blk = np.fft.irfft(F[:, a:a+step, None]*np.conj(F[:, None, :]),
                           axis=0, n=2*nF)
        blk = np.concatenate([blk[-nF+1:], blk[:nF]], axis=0)
        pk = np.argmax(np.abs(blk), axis=0) - (nF-1)
        for jj in range(blk.shape[1]):
            i0 = a + jj
            sel = (ctx.iu[0] == i0)
            if sel.any():
                lags[sel] = pk[jj, ctx.iu[1][sel]]
    al = np.abs(lags)*dt_frame
    med = np.array([np.nanmedian(al[ctx.bi == b]) if (ctx.bi == b).sum() > 4 else np.nan
                    for b in range(len(ctx.bins))])
    ok = np.isfinite(med) & (ctx.bins > 0)
    if ok.sum() > 2:
        sl = np.polyfit(ctx.bins[ok], med[ok], 1)[0]
        o["speed_xcorr"] = float(1.0/sl) if sl > 1e-9 else float("nan")
    else:
        o["speed_xcorr"] = float("nan")

    # ---- 13. centroid path: does the bulk of the field actually go anywhere
    W = (U*U)*aw
    W /= np.maximum(W.sum(1, keepdims=True), 1e-300)
    cen = W @ ctx.V
    steps = np.linalg.norm(np.diff(cen, axis=0), axis=1)
    o["centroid_path"] = float(steps.sum())
    o["centroid_net"] = float(np.linalg.norm(cen[-1] - cen[0]))
    o["centroid_spread"] = float(np.linalg.norm(cen - cen.mean(0), axis=1).mean())
    o["centroid_directedness"] = float(o["centroid_net"]/max(o["centroid_path"], 1e-30))

    # ---- 6. connected active clusters
    nc, big = [], []
    for k in range(0, nF, CLUSTER_EVERY):
        m = absU[k] > 0.2*absU[k].max()
        if m.sum() < 2:
            nc.append(0); big.append(0.0); continue
        sub = ctx.Adj[m][:, m]
        n_, lab_ = connected_components(sub, directed=False)
        sz = np.bincount(lab_)
        nc.append(n_)
        big.append(float(sz.max()/max(m.sum(), 1)))
    o["n_clusters_mean"] = float(np.mean(nc))
    o["n_clusters_cv"] = _cv(nc)
    o["biggest_cluster_frac"] = float(np.mean(big))

    # ---- 17,18. does the drive launch anything, or does it just sit there
    dm = driven_mask[keep]
    ein = (U*U*aw)[:, dm].sum(1)
    o["energy_in_driven"] = float(np.mean(ein))
    o["energy_in_driven_cv"] = _cv(ein)
    dsrc = ctx.dist_from_driven
    e = (U*U*aw).mean(0)
    tot = e.sum()
    order = np.argsort(dsrc)
    ce = np.cumsum(e[order])/max(tot, 1e-300)
    o["reach_50"] = float(dsrc[order][np.searchsorted(ce, 0.5)])
    o["reach_90"] = float(dsrc[order][np.searchsorted(ce, 0.9)])
    return o


def main():
    t00 = time.time()
    ctx = Context()
    man = json.load(open(os.path.join(FIELDS, "manifest.json")))
    from sweep_fields import eroded_taper
    c = ctx.c
    n_ant = int((c.lab == 65).sum() + (c.lab == 88).sum())
    v1t, _, _ = eroded_taper(c, 1, n_ant)
    masks = {"A_10r10v": (c.lab == 65) | (c.lab == 88), "B_V1": v1t > 0}

    rows, names = [], []
    for rec in man:
        z = np.load(os.path.join(FIELDS, rec["name"] + ".npz"))
        Hs, drv, dtf = z["H"], z["drive"], float(z["dt_frame"])
        dm = masks[rec["cond"]]
        ctx.dist_from_driven = dijkstra(ctx.G, indices=np.flatnonzero(dm),
                                        directed=False).min(axis=0)[ctx.keep]
        t0 = time.time()
        full = compute(Hs, drv, dtf, ctx, dm)
        half = len(Hs)//2
        h1 = compute(Hs[:half], drv[:half], dtf, ctx, dm)
        h2 = compute(Hs[half:], drv[half:], dtf, ctx, dm)
        rows.append(dict(full=full, h1=h1, h2=h2, rec=rec))
        names.append(rec["name"])
        print(f"  {rec['name']:16s} {len(full)} measures  [{time.time()-t0:.1f}s]",
              flush=True)

    keys = sorted(rows[0]["full"].keys())
    M = np.array([[r["full"][k] for k in keys] for r in rows])
    H1 = np.array([[r["h1"][k] for k in keys] for r in rows])
    H2 = np.array([[r["h2"][k] for k in keys] for r in rows])
    np.savez(os.path.join(FIELDS, "measures.npz"), M=M, H1=H1, H2=H2,
             keys=np.array(keys), names=np.array(names))
    print(f"\n  {len(keys)} measures x {len(rows)} fields, "
          f"total {time.time()-t00:.0f}s -> measures.npz")


if __name__ == "__main__":
    main()
