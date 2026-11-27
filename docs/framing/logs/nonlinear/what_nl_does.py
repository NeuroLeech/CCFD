"""What the nonlinear rotating medium does to the linear drive, read from saved replays.

The 10 s rotation drive (nobp_sm6_s3_rot10_box), draw 0, replayed at 1x (effectively
linear: field vs linear r 0.9999) and at a higher amplitude M. A linear field scales exactly
with amplitude, so the nonlinear part is D = F_M - M * F_1. Reported:

  1. where D is large, as a share of the scored observable's variance per vertex, split
     below and above 0.1 Hz (the inertial frequency at a 10 s period);
  2. where it costs the fit: per vertex, the agreement of its double-centred FC row with the
     target's (Pearson, on the target's vertices), at 1x and at M;
  3. both against distance from the driven vertices and against the principal gradient;
  4. a surface figure, and frames for a video of D (render_bandpass reads them).
"""
import os, sys
ROOT = "/home/user/CCFD"
sys.path[:0] = [os.path.join(ROOT, d) for d in ("core", "fit", "targets", "analysis", "viz")]
os.chdir(ROOT)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
import torch_fit, fc_score, units, timescale, surface_smooth
from mesh_cache import load_cortex
from paths import RESULTS
from render_regimes import _proj

TAG = "nobp_sm6_s3_rot10_box"
M = int(sys.argv[1]) if len(sys.argv) > 1 else 35
OUT = sys.argv[2] if len(sys.argv) > 2 else "docs/framing/figures"
c = load_cortex("fsaverage5", verbose=False)
ref = torch_fit.load_reference(TAG)
t = fc_score.FCTarget(c, fc_path=ref["fc_path"], centre="double", verbose=False)
fs, burn = ref["frame_s"], t.burn
cols = np.asarray(t.cols)
kern = units.smoothing_kernel(timescale.bold_fwhm_frames(fs, verbose=False), verbose=False)
K = surface_smooth.smoothing_matrix(c, 6.0)

F1 = np.load(os.path.join(RESULTS, f"frames_{TAG}_nl1_raw.npy")).astype(np.float64)
FM = np.load(os.path.join(RESULTS, f"frames_{TAG}_nl{M}_raw.npy")).astype(np.float64)
n = min(len(F1), len(FM))
L, D = M * F1[:n], FM[:n] - M * F1[:n]


def obs(X):
    return np.asarray(surface_smooth.smooth(units.smooth_frames(np.asarray(X, np.float32), kern), K),
                      np.float64)


OL, OD, OM = obs(L), obs(D), obs(FM[:n])
f = np.fft.rfftfreq(n - burn, fs)


def bandvar(X, lo, hi):
    X = X[burn:] - X[burn:].mean(0)
    P = np.abs(np.fft.rfft(X, axis=0)) ** 2
    k = (f > lo) & (f <= hi)
    return P[k].sum(0)


vL, vD = OL[burn:].var(0), OD[burn:].var(0)
share = vD / np.maximum(vL, 1e-30)                       # nonlinear / linear variance, per vertex
slow = bandvar(OD, 0, 0.1) / np.maximum(bandvar(OL, 0, 0.1), 1e-30)
fast = bandvar(OD, 0.1, 0.775) / np.maximum(bandvar(OL, 0.1, 0.775), 1e-30)


def rows_agree(O):
    Z = O[burn:, cols]
    Z = (Z - Z.mean(0)) / np.maximum(Z.std(0), 1e-30)
    C = (Z.T @ Z / len(Z)).astype(np.float32)
    C = fc_score.double_centre(C, inplace=True)
    T = np.asarray(t.target_fc(), np.float32)
    T = fc_score.double_centre(T.copy(), inplace=True)
    np.fill_diagonal(C, 0); np.fill_diagonal(T, 0)
    Cc, Tc = C - C.mean(1, keepdims=True), T - T.mean(1, keepdims=True)
    r = (Cc * Tc).sum(1) / np.sqrt((Cc ** 2).sum(1) * (Tc ** 2).sum(1))
    iu = np.triu_indices(len(cols), 1)
    return r, float(np.corrcoef(C[iu], T[iu])[0, 1])


r1, g1 = rows_agree(OL)
rM, gM = rows_agree(OM)
drow = rM - r1

# distance from the driven vertices, along the surface
P = np.asarray(ref["P"])
driven = np.flatnonzero(P.sum(0) > 0.05 * P.sum(0).max())
w = np.linalg.norm(c.V[c.edges[:, 0]] - c.V[c.edges[:, 1]], axis=1)
G = csr_matrix((np.r_[w, w], (np.r_[c.edges[:, 0], c.edges[:, 1]], np.r_[c.edges[:, 1], c.edges[:, 0]])),
               shape=(c.nV, c.nV))
dist = dijkstra(G, indices=driven, directed=False, min_only=True)
grad = np.load(os.path.join(os.path.dirname(RESULTS), "data", "cache",
                            "cortical_maps_fsaverage5.npz"))["grad1"]

print(f"{TAG}, draw 0, {M}x against 1x")
print(f"  whole-matrix agreement with the target (Pearson, double-centred): 1x {g1:+.4f}, "
      f"{M}x {gM:+.4f}")
print(f"  nonlinear share of the observable's variance, median vertex: {np.median(share[cols]):.1%} "
      f"(below 0.1 Hz {np.median(slow[cols]):.0%} of the linear power there, above "
      f"{np.median(fast[cols]):.0%})")
print(f"  FC-row agreement with the target, median vertex: 1x {np.median(r1):+.3f}, {M}x "
      f"{np.median(rM):+.3f}; rows that lose more than 0.05: {np.mean(drow < -0.05):.0%}, "
      f"gain more than 0.05: {np.mean(drow > 0.05):.0%}")
dc = dist[cols]
for nm, x in (("distance from the inputs", dc), ("principal gradient", grad[cols])):
    q = np.quantile(x, [0, 0.2, 0.4, 0.6, 0.8, 1.0])
    print(f"  by {nm} (fifths, low to high):")
    for lo, hi in zip(q[:-1], q[1:]):
        k = (x >= lo) & (x <= hi)
        print(f"    {lo:7.2f}-{hi:7.2f}: nonlinear share {np.median(share[cols][k]):6.1%}, slow "
              f"{np.median(slow[cols][k]):5.0%}, row agreement 1x {np.median(r1[k]):+.3f} -> "
              f"{M}x {np.median(rM[k]):+.3f} ({np.median(drow[k]):+.3f})")
print(f"  correlation across vertices: nonlinear share vs row change "
      f"{np.corrcoef(np.log(share[cols]), drow)[0, 1]:+.2f}, vs distance "
      f"{np.corrcoef(np.log(share[cols]), dc)[0, 1]:+.2f}, vs gradient "
      f"{np.corrcoef(np.log(share[cols]), grad[cols])[0, 1]:+.2f}")

# figure: nonlinear share, slow nonlinear share, row-agreement change
os.makedirs(OUT, exist_ok=True)
proj = _proj(c.V, c.F)
full_share = np.full(c.nV, np.nan); full_share[cols] = np.log10(share[cols])
full_slow = np.full(c.nV, np.nan); full_slow[cols] = np.log10(slow[cols])
full_d = np.full(c.nV, np.nan); full_d[cols] = drow
rows = [(full_share, "nonlinear / linear variance\nof the scored signal (log10)", "magma", None),
        (full_slow, "the same, below 0.1 Hz\n(log10)", "magma", None),
        (full_d, f"change in FC-row agreement\nwith the target, {M}x - 1x", "RdBu", 0.15)]
fig, ax = plt.subplots(3, len(proj), figsize=(4.0 * len(proj), 9.5), facecolor="white")
for i, (v, lab, cm, lim) in enumerate(rows):
    lo, hi = (np.nanpercentile(v, 2), np.nanpercentile(v, 98)) if lim is None else (-lim, lim)
    for j, (xy, vis, nm) in enumerate(proj):
        a = ax[i, j]
        keep = vis[c.F].all(1)
        tri = Triangulation(xy[:, 0], xy[:, 1], c.F[keep])
        im = a.tripcolor(tri, np.nan_to_num(v, nan=lo), shading="gouraud", cmap=cm, vmin=lo, vmax=hi,
                         rasterized=True)
        a.set_aspect("equal"); a.axis("off")
        if i == 0:
            a.set_title(nm, fontsize=10)
        if j == 0:
            a.text(-0.08, 0.5, lab, rotation=90, transform=a.transAxes, ha="center", va="center",
                   fontsize=9)
    fig.colorbar(im, ax=ax[i, :].tolist(), shrink=0.7, pad=0.01)
fig.suptitle(f"What the nonlinear medium (10 s rotation, {M}x) does to the linear drive", fontsize=11)
path = os.path.join(OUT, f"nonlinear_rot10_{M}x_maps.png")
fig.savefig(path, dpi=150, bbox_inches="tight")
print(f"  wrote {path}")

# frames for a video of the nonlinear part itself (render_bandpass --raw-npy)
vt = f"{TAG}_nl{M}diff"
np.save(os.path.join(RESULTS, f"frames_{vt}.npy"), OD.astype(np.float32))
np.save(os.path.join(RESULTS, f"frames_{vt}_raw.npy"), D.astype(np.float32))
np.save(os.path.join(RESULTS, f"drive_{vt}.npy"), np.load(os.path.join(RESULTS, f"drive_{TAG}_nl{M}.npy")))
print(f"  wrote frames_{vt}.npy and _raw.npy")
