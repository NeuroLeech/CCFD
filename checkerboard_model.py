"""Drive the model with the checkerboard boxcar and compare against the group response.

Every fit so far has been to a second moment. This is the other kind of test: the input is
KNOWN - 20 s blocks in V1 - so what the model does with it can be set against what the
cortex did with it, and the comparison is of propagation rather than of covariance.

Three things make this cheaper and cleaner than it looks:

  LINEARITY. The medium is linear in this regime (interference.py checks the superposition
  against the run's own saved field at r = 0.9990). So the checkerboard response computed
  ALONE is exactly the checkerboard component of a run driven with rest and checkerboard
  together - adding the resting drive would only put resting fluctuation on top as noise.
  Nothing is simulated here either: the response to a boxcar is the piece impulse
  responses convolved with it, and those are the same cached objects H is built from.

  SCALE. The comparison map is a CORRELATION with the boxcar, so the drive amplitude
  cancels out of it entirely. "A small amount of extra drive" and a large amount give the
  same map. Amplitude would only matter if the medium were nonlinear or if absolute
  amplitudes were being compared, and neither is the case.

  UNITS. The empirical map is a correlation against a 0/1 regressor in noisy single-subject
  data averaged over 99 subjects, so it peaks at +0.063; the model's is the same
  correlation in a noiseless field and runs near 1. Only the SHAPE of the two maps is
  comparable - their spatial correlation, their parcel ordering, and how far they reach -
  never the two numbers side by side.

Two things are matched to the data rather than chosen. The model observable is downsampled
to TR before correlating, so both sides are scored on the same grid; and the same XCP-D
passband is applied over the same 154 s length, so both sides carry the same filter edge
effects.

One asymmetry cannot be matched and is reported instead. The empirical lag of 4.52 s is
haemodynamic; the model's BOLD kernel is a symmetric zero-phase gaussian, so its lag comes
only from propagation and from the medium's own 25 s decay. Each side gets its own best
lag on the same grid, and both are printed.

  python checkerboard_model.py --tag g7_s1.5_d25 --parcels 1
"""
import os, argparse
import numpy as np
from scipy.signal import fftconvolve

from paths import RESULTS, CACHE
import timescale

TR = 0.645
BLOCKS = [(20.0, 20.0), (60.0, 20.0), (100.0, 20.0)]     # CHECKER onsets/durations
NT = 239                                                 # frames in one checkerboard run
RUN_S = NT * TR


def piece_weights(z, sel, mode, frame_s, band=(0.01, 0.08), pad=4096):
    """Per-piece drive AMPLITUDE, mean 1.

    `uniform` drives every piece equally, which is a choice and not a neutral one.
    `solve`/`band` instead take the amplitude the resting solve gave each piece:
    diag(S) is that piece's input POWER per bin, so sqrt of it summed over bins (or over
    the passband) is the amplitude. Bin k of the rfft over the run's PADDED window is
    k/(pad*frame_s) Hz - pad, not the impulse length, since best_fit zero-pads before the
    transform and the solved grid indexes that.

    What this cannot carry is the off-diagonal. S's cross-terms are the phase relations
    between pieces, and the fitted input uses them heavily - it is 45% self-cancelling.
    Driving every piece with the SAME boxcar imposes all-in-phase, which is exactly the
    configuration the solve declined. So this varies magnitudes only."""
    if mode == "uniform":
        return np.ones(len(sel))
    import xspec
    p = xspec.piece_power(z["S"], z["idx"], frame_s,
                          band=(None if mode == "solve" else band), pad=pad)[sel]
    a = np.sqrt(np.maximum(p, 0.0))
    return a / a.mean()


def boxcar(t):
    x = np.zeros(len(t), np.float64)
    for on, dur in BLOCKS:
        x[(t >= on) & (t < on + dur)] = 1.0
    return x


def piece_observables(c, tag, parcels=(1,), band=(0.01, 0.08), basis=None,
                      cols=None, verbose=True):
    """Each driven piece's OWN observable on the data's TR grid.

    -> (Y, sel, tags, boxcar) with Y of shape (pieces, drives, 239, vertices).

    The medium is linear and so is every step between the drive and the scored map: the
    convolution with the impulse responses, the BOLD kernel, the passband, the downsample
    to TR. So the field produced by any per-piece amplitude vector w over any mixture of
    the basis time courses is exactly `sum_jb c_jb Y[j,b]`. Driving the pieces one at a
    time is not an approximation of driving them together - it is the same field, kept in
    its parts. That is what makes a search over drive SHAPE cost nothing once these are
    formed: the expensive objects (impulse responses, smoothing, filtering) are built
    once, and every candidate is a weighted sum of them.

    `basis` is (model frames, nB) drive time courses; the default is the single boxcar.
    `cols` restricts the returned vertices, since the comparison only ever uses the
    target's.
    """
    import xspec, bo_step, subparcels, units, bandpass
    from interference import _impulse_frames
    z = np.load(os.path.join(RESULTS, f"xspec_{tag}.npz"), allow_pickle=True)
    x, save, labels, tags = z["x"], int(z["save"]), z["labels"], list(z["tags"])
    p, _, _ = bo_step.unpack(x, c)
    P = subparcels.taper_profiles(c, labels, len(tags))
    frame_s = timescale.TR / 4.0

    sel = [i for i, s in enumerate(tags) if int(s.split("_")[0]) in set(parcels)]
    if not sel:
        raise SystemExit(f"  no pieces in parcels {parcels}")
    if verbose:
        print(f"  driving {len(sel)} pieces: {', '.join(tags[i] for i in sel)}")

    nfr = int(np.ceil(RUN_S / frame_s))
    t = np.arange(nfr) * frame_s
    B = boxcar(t)[:, None] if basis is None else np.asarray(basis, np.float64)
    if B.shape[0] != nfr:
        raise SystemExit(f"  basis has {B.shape[0]} rows, expected {nfr} model frames")

    imp = _impulse_frames(x, save, frame_s)
    R = xspec.impulse_responses(c, sel, p, imp * save, save,
                                profiles=P[sel], verbose=False)
    if verbose:
        print(f"  {nfr} model frames ({nfr*frame_s:.0f} s), impulse window {R.shape[1]}, "
              f"{B.shape[1]} drive time course(s)")

    kern = units.smoothing_kernel(timescale.bold_fwhm_frames(frame_s, verbose=False),
                                  verbose=False)
    keep = slice(None) if cols is None else np.asarray(cols)
    nV = c.nV if cols is None else len(keep)
    Y = np.zeros((len(sel), B.shape[1], NT, nV), np.float32)
    for j in range(len(sel)):
        for b in range(B.shape[1]):
            f = fftconvolve(B[:, b:b + 1], R[j], axes=0)[:nfr]
            f = units.smooth_frames(f, kern)
            f = bandpass.apply(f, frame_s, band[0], band[1])
            Y[j, b] = f[::4][:NT][:, keep]
    return Y, sel, tags, boxcar(np.arange(NT) * TR)


def shift_boxcar(bt, lag):
    """The boxcar delayed by `lag` TRs, mean removed and unit norm."""
    bb = np.roll(np.asarray(bt, np.float64), lag)
    bb[:lag] = 0.0
    bb = bb - bb.mean()
    n = np.linalg.norm(bb)
    return bb / n if n > 0 else bb


def correlate_map(Y, bt, lag):
    """Per-vertex correlation between the observable and the lagged boxcar."""
    y = np.asarray(Y, np.float64)
    y = y - y.mean(0)
    bb = shift_boxcar(bt, lag)
    den = np.linalg.norm(y, axis=0)
    return np.where(den > 0, (y.T @ bb) / np.maximum(den, 1e-30), 0.0)


def best_lag_map(Y, bt, lag_max=16, verbose=True):
    """-> (map at the lag with the highest peak, that lag in frames)."""
    best, rmap = None, None
    for s in range(lag_max + 1):
        g = correlate_map(Y, bt, s)
        if best is None or g.max() > best[1]:
            best, rmap = (s, g.max()), g
        if verbose:
            print(f"    lag {s*TR:5.2f}s  peak r {g.max():+.4f}", flush=True)
    return rmap, best[0]


def model_response(c, tag, parcels=(1,), band=(0.01, 0.08), lag_max=16,
                   weights="uniform", verbose=True):
    """-> (response map on the TR grid, lag frames, the observable, the boxcar)."""
    z = np.load(os.path.join(RESULTS, f"xspec_{tag}.npz"), allow_pickle=True)
    Y, sel, tags, bt = piece_observables(c, tag, parcels=parcels, band=band,
                                         verbose=verbose)
    wk = piece_weights(z, sel, weights, timescale.TR / 4.0, band)
    if verbose:
        print(f"  drive weights ({weights}): " +
              ", ".join(f"{tags[i]} {v:.3f}" for i, v in zip(sel, wk)))
    Yc = np.tensordot(wk.astype(np.float32), Y[:, 0], axes=(0, 0))
    rmap, lag = best_lag_map(Yc, bt, lag_max, verbose)
    return rmap, lag, Yc, bt


EDGES = (0, 5, 10, 15, 20, 30, 40, 50, 65, 80, 100, 120)


def profile(c, resp, cols, seed_full, edges=EDGES):
    """Mean response in geodesic distance bins from a seed vertex, and the distance at
    which it has fallen to half its near value.

    Normalised by the FIRST populated bin, not by bin zero: the seed's own 0-5 mm
    neighbourhood holds a couple of dozen vertices at this resolution and is often empty
    of the 20 a bin needs to be reported at all."""
    import units
    d = units.vertex_geodesic(c, [seed_full])[0][cols]
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (d >= lo) & (d < hi)
        out.append((lo, hi, int(m.sum()),
                    float(resp[m].mean()) if m.sum() > 20 else np.nan))
    ref = next((v for _, _, _, v in out if not np.isnan(v)), np.nan)
    return out, ref


def half_fall(prof, ref):
    """Linear interpolation, on bin centres, of where the profile crosses ref/2."""
    xs = [(lo + hi) / 2.0 for lo, hi, _, v in prof if not np.isnan(v)]
    ys = [v for _, _, _, v in prof if not np.isnan(v)]
    for i in range(1, len(ys)):
        if ys[i] <= ref / 2.0 <= ys[i - 1]:
            f = (ys[i - 1] - ref / 2.0) / max(ys[i - 1] - ys[i], 1e-30)
            return xs[i - 1] + f * (xs[i] - xs[i - 1])
    return np.nan


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", default="g7_s1.5_d25")
    ap.add_argument("--parcels", default="1", help="Glasser ids to drive (1 = V1)")
    ap.add_argument("--lag-max", type=int, default=16, dest="lag_max")
    ap.add_argument("--weights", default="uniform",
                    choices=("uniform", "solve", "band"))
    ap.add_argument("--emp", default=None)
    a = ap.parse_args()

    from mesh_cache import load_cortex
    import fc_score
    from scipy.stats import spearmanr
    c = load_cortex("fsaverage5", verbose=False)
    t = fc_score.default_target(c, verbose=False)
    parcels = tuple(int(v) for v in a.parcels.split(","))

    emp_path = a.emp or os.path.join(CACHE, f"checkerboard_group_100_{t.nV}.npz")
    e = np.load(emp_path, allow_pickle=True)
    emp = e["response"]
    print(f"  empirical: {emp_path}\n    {int(e['n'])} subjects, lag {float(e['lag_s']):.2f} s,"
          f" peak {emp.max():+.4f}")

    rmap, lag, Y, bt = model_response(c, a.tag, parcels=parcels, lag_max=a.lag_max,
                                      weights=a.weights)
    mod = rmap[t.cols]
    print(f"\n  model: lag {lag*TR:.2f} s, peak {mod.max():+.4f}, min {mod.min():+.4f}")

    # ---- the two maps against each other. Shapes only: the scales are not comparable ----
    pr = float(np.corrcoef(mod, emp)[0, 1])
    sr = float(spearmanr(mod, emp).statistic)
    print(f"\n  model vs empirical response map over {len(t.cols)} vertices:"
          f"  Pearson {pr:+.4f}, Spearman {sr:+.4f}")

    lab = c.lab[t.cols]
    def nm(i):
        s = c.names[i]
        return (s.decode() if isinstance(s, bytes) else str(s)
                ).removeprefix("L_").removesuffix("_ROI")
    keep = [int(p) for p in np.unique(lab) if p >= 0 and (lab == p).sum() >= 20]
    sm = {p: float(mod[lab == p].mean()) for p in keep}
    se = {p: float(emp[lab == p].mean()) for p in keep}
    om = sorted(sm, key=lambda p: -sm[p])
    oe = sorted(se, key=lambda p: -se[p])
    print(f"\n  {'':>4s} {'model top 10':<34s}  {'empirical top 10':<34s}")
    for i in range(10):
        print(f"  {i+1:>3d}. {nm(om[i])+' '+format(sm[om[i]],'+.3f'):<34s}  "
              f"{nm(oe[i])+' '+format(se[oe[i]],'+.3f'):<34s}")
    print(f"\n  rank of each area   {'model':>12s} {'empirical':>12s}")
    for k in ("V1", "V2", "V3", "V4", "V3CD", "LO1", "LO2", "MT", "MST", "A1", "4"):
        h = [p for p in keep if nm(p) == k]
        if h:
            print(f"    {k:<6s}          {om.index(h[0])+1:>5d} ({sm[h[0]]:+.3f})"
                  f" {oe.index(h[0])+1:>5d} ({se[h[0]]:+.3f})")

    # ---- how far each side reaches, from the SAME seed so the profiles are comparable ---
    pk_e = int(np.argmax(emp)); pk_m = int(np.argmax(mod))
    print(f"\n  empirical peak vertex in {nm(int(lab[pk_e]))}, "
          f"model peak vertex in {nm(int(lab[pk_m]))}")
    for lbl, seed in (("empirical peak", t.cols[pk_e]), ("model peak", t.cols[pk_m])):
        pe, re_ = profile(c, emp, t.cols, seed)
        pm, rm = profile(c, mod, t.cols, seed)
        print(f"\n  distance from the {lbl}   {'model':>20s} {'empirical':>20s}")
        print(f"    {'':>14s} {'':>7s} {'raw':>9s} {'norm':>6s}  {'raw':>10s} {'norm':>6s}")
        for (lo, hi, n, vm), (_, _, _, ve) in zip(pm, pe):
            if not np.isnan(vm):
                print(f"    {lo:>3d}-{hi:<3d} mm  n={n:>5d}   {vm:>9.4f} {vm/rm:>6.2f}  "
                      f"{ve:>10.4f} {ve/re_:>6.2f}")
        hm, he = half_fall(pm, rm), half_fall(pe, re_)
        print(f"    falls to half by:  model {hm:.0f} mm,  empirical {he:.0f} mm"
              f"   (ratio {hm/he:.1f}x)")

    sfx = "" if a.weights == "uniform" else f"_{a.weights}"
    out = os.path.join(RESULTS, f"checkerboard_model_{a.tag}{sfx}.npz")
    np.savez(out, model=mod, empirical=emp, cols=t.cols, lag=lag,
             parcels=np.array(parcels), Y=Y, boxcar=bt)
    print(f"\n  wrote {out}")
    _plot(a.tag + sfx, c, t, mod, emp, Y, bt, lag)


def _plot(tag, c, t, mod, emp, Y, bt, lag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from render_regimes import _proj
    from plot_fc_map import surface_row
    proj = _proj(c.V, c.F)
    rows = [("model  (V1 boxcar)", mod, "RdBu_r",
             (-abs(mod).max(), abs(mod).max())),
            ("empirical  (99 subjects)", emp, "RdBu_r",
             (-abs(emp).max(), abs(emp).max()))]
    fig = plt.figure(figsize=(3.9 * len(proj), 2.9 * len(rows) + 2.2))
    gs = fig.add_gridspec(len(rows) + 1, len(proj),
                          height_ratios=[2.9] * len(rows) + [1.6],
                          hspace=0.10, wspace=0.02)
    for i, (lab, v, cm, lims) in enumerate(rows):
        surface_row(fig, gs, i, proj, v, c, t.cols, cm, lims, lab)
    ax = fig.add_subplot(gs[len(rows), :])
    tt = np.arange(len(Y)) * TR
    pk = int(np.argmax(mod))
    y = Y[:, t.cols[pk]]
    ax.plot(tt, (y - y.mean()) / y.std(), lw=1.2, color="tab:blue",
            label="model at its peak vertex (z)")
    ax.plot(tt, bt * 2 - 1, lw=1.0, color="0.6", label="checkerboard boxcar")
    ax.set_xlabel("time (s)"); ax.set_xlim(0, tt[-1])
    ax.legend(frameon=False, fontsize=8, ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(f"model lag {lag*TR:.2f} s", fontsize=9)
    out = os.path.join(RESULTS, f"checkerboard_model_{tag}.png")
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
