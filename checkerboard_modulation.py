"""The checkerboard as a MODULATION of the resting drive, not a replacement for it.

The first pass drove V1 with a boxcar and nothing else. That is a different experiment
from the one the scanner ran: in a subject the checkerboard arrives on top of whatever
input V1 was already receiving, and the resting drive does not stop. Two things follow,
and they are not small.

AMPLITUDE STOPS CANCELLING. `checkerboard_model`'s docstring argues that because the map
is a correlation, the drive amplitude divides out. That holds only for drive-alone. With
the resting drive present the denominator of r is dominated by ongoing fluctuation, so
the MODULATION DEPTH sets how far the response rises above the background - it is the
parameter, not a nuisance.

AND IT SUPPLIES THE MISSING DENOMINATOR. Drive-alone, a vertex 100 mm from V1 has a
negligible response that still correlates +-1 with the boxcar, because there is nothing
else in its timeseries; the model's map came out as a whole-hemisphere banded pattern at
+-0.87 against a focal empirical one. In a real vertex that far out the response sits
under the ongoing fluctuation and r falls to ~0. Putting the resting drive back is what
makes the two maps the same quantity.

STILL NO NEW SIMULATION. The medium is linear, so the field is
f = f_rest + f_modulation, and the resting half is already on disk: `frames_{tag}.npy` is
a 2,308 s realisation of the fitted drive, already smoothed and bandpassed. Cut into
154 s windows it gives independent "subjects" to average over, the way the empirical map
averages over 99 people.

TWO FORMS OF MODULATION, and they are not variants of one idea:

  additive        a_k(t) -> a_k(t) + alpha * s_k * b(t).   A mean shift during CHECKER,
                  so it produces an evoked response the boxcar correlation can see.
  multiplicative  a_k(t) -> a_k(t) * (1 + alpha * b(t)).   The solved drive has no DC, so
                  the added term has zero mean: this modulates the VARIANCE of the
                  ongoing input, not its mean. What it does to a boxcar CORRELATION and
                  what it does to the ON-vs-OFF FC are different questions, and both are
                  measured here rather than argued about.

alpha is a fraction of the piece's own resting drive RMS in both cases, so the two are on
one axis and can be swept together.

  python checkerboard_modulation.py --tag g7_s1.5_d25
"""
import os, argparse
import numpy as np
from scipy.signal import fftconvolve

from paths import RESULTS, CACHE
import timescale
import checkerboard_model as cm

TR = cm.TR
NT = cm.NT

# One definition, on the empirical side, used by both: a model and the data have to be
# summarised over the same vertices or the two numbers are not comparable.
from checkerboard import VISUAL


def parcel_names(c):
    return [(s.decode() if isinstance(s, bytes) else str(s)
             ).removeprefix("L_").removesuffix("_ROI") for s in c.names]


def frame_drive(tag, save):
    """The run's drive as TOTAL INJECTED PER FRAME, the units the impulse responses see.

    Exactly interference.contributions' convention - the drive is injected every step and
    the responses are sampled every `save` steps, so a frame's block is what its single
    sampled response stands for. That decomposition was checked against this run's own
    saved field at r = 0.9990, so it is the convention a modulation has to be added in."""
    A = np.asarray(np.load(os.path.join(RESULTS, f"drive_{tag}.npy")), np.float64)
    nfr = len(A) // save
    return A[:nfr * save].reshape(nfr, save, -1).sum(1)


def rest_windows(tag, nfr_run, verbose=True):
    """Non-overlapping runs' worth of the saved resting realisation, on the TR grid.

    Each window is one model 'subject': the same medium and the same solved input, a
    different stretch of its own fluctuation. Already smoothed and bandpassed - these are
    the observable the fit was scored through, not the raw field.

    The filter was applied over the whole 2,308 s rather than per window, where the data
    was filtered over each 154 s run. That difference lives at the window edges."""
    F = np.load(os.path.join(RESULTS, f"frames_{tag}.npy"), mmap_mode="r")
    k = len(F) // nfr_run
    if verbose:
        print(f"  resting background: {len(F)} frames -> {k} windows of {nfr_run} "
              f"({nfr_run * timescale.TR / 4.0:.0f} s each)")
    return F, k


def piece_fields(c, tag, parcels, verbose=True):
    """Each driven piece's own MODEL-RATE response to a unit boxcar, and its drive scale.

    -> (Yb (K, nfr, nV) float32, R (K, imp, nV), sel, tags, a_rest (nfr_tot, K), save)
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
    nfr = int(np.ceil(cm.RUN_S / frame_s))
    b = cm.boxcar(np.arange(nfr) * frame_s)
    imp = _impulse_frames(x, save, frame_s)
    R = xspec.impulse_responses(c, sel, p, imp * save, save, profiles=P[sel],
                                verbose=False)
    kern = units.smoothing_kernel(timescale.bold_fwhm_frames(frame_s, verbose=False),
                                  verbose=False)
    Yb = np.zeros((len(sel), nfr, c.nV), np.float32)
    for j in range(len(sel)):
        f = fftconvolve(b[:, None], R[j], axes=0)[:nfr]
        f = units.smooth_frames(f, kern)
        Yb[j] = bandpass.apply(f, frame_s, 0.01, 0.08)
    if verbose:
        print(f"  {len(sel)} pieces: {', '.join(tags[i] for i in sel)}; "
              f"{nfr} model frames, impulse window {R.shape[1]}")
    return Yb, R, sel, tags, frame_drive(tag, save), save, kern, frame_s


def vertex_profile(c, t, tag, sel, emp, parcel=1, signed=True):
    """A drive profile shaped like the empirical map ITSELF, at vertex resolution.

    The piece shapes approximate "drive only where the data responds" by "drive only the
    pieces whose average response is positive", which is as fine as seven equal-area
    patches allow. This is the un-averaged version: the injected pattern IS the group
    ON-OFF response on the driven parcel's vertices, zero elsewhere.

    It is normalised to carry the same total area-weighted drive magnitude as the
    piece-wise `empirical` shape at the same alpha, so the sweep axis means the same
    thing and the only difference between the two is the spatial resolution of the
    profile. `signed=False` clips it to its positive part.
    """
    import subparcels
    z = np.load(os.path.join(RESULTS, f"xspec_{tag}.npz"), allow_pickle=True)
    labels, tags = z["labels"], [str(s) for s in z["tags"]]
    P = subparcels.taper_profiles(c, labels, len(tags))
    A = np.asarray(c.A, float)

    g = np.zeros(c.nV)
    g[t.cols] = np.where(c.lab[t.cols] == parcel, emp, 0.0)
    if not signed:
        g = np.maximum(g, 0.0)

    # the piece-wise pattern this has to match in total magnitude
    e = empirical_v1(c, t, tag, sel, parcel, emp)
    wpc = e / max(np.abs(e).mean(), 1e-30)
    Qp = (wpc[:, None] * P[sel]).sum(0)
    mass_p = float((np.abs(Qp) * A).sum())
    mass_g = float((np.abs(g) * A).sum())
    return g * (mass_p / max(mass_g, 1e-30))


def empirical_v1(c, t, tag, sel, parcel=1, emp=None):
    """The empirical ON-OFF response averaged over each driven piece.

    The data's own answer to how the drive should be distributed inside V1, rather than a
    profile chosen by hand. Within V1 the group response is graded by distance from the
    occipital pole at r = -0.75: about +0.076 on the most foveal piece and about -0.017 on
    the most peripheral. Returned unnormalised and SIGNED, so a caller can either keep the
    sign - driving the foveal pieces up and the peripheral ones down - or clip it and
    drive only the pieces the data says respond.
    """
    z = np.load(os.path.join(RESULTS, f"xspec_{tag}.npz"), allow_pickle=True)
    labels = z["labels"][t.cols]
    return np.array([float(emp[labels == j].mean()) if (labels == j).any() else 0.0
                     for j in sel])


def realise_field(c, tag, nfr, seed, ref_frames=4096, amp=2e-4, verbose=False):
    """One fresh 154 s resting run: a new draw of the solved input, simulated and made
    into the observable. -> (frames, per-frame drive).

    Fourteen windows of the saved 2,308 s run is fourteen model 'subjects' against the
    data's 99, and the floor of this measurement falls as 1/sqrt(N) - so with too few
    windows the map at small modulation depth is reporting the floor. Independent draws
    are what the data has, so independent draws are what the model gets.

    The drive is normalised to the same RMS `score_realisation` uses, so a window made
    here is on the same scale as one cut out of the saved run.
    """
    import xspec, bo_step, subparcels, units, bandpass
    import fluid as fl
    z = np.load(os.path.join(RESULTS, f"xspec_{tag}.npz"), allow_pickle=True)
    x, save, labels, tags = z["x"], int(z["save"]), z["labels"], list(z["tags"])
    p, _, _ = bo_step.unpack(x, c)
    P = subparcels.taper_profiles(c, labels, len(tags))
    frame_s = timescale.TR / 4.0
    A_frames = xspec.realise(z["S"], z["idx"], nfr, ref_frames=ref_frames, seed=seed)
    nsteps = len(A_frames) * save
    A = np.repeat(A_frames, save, axis=0)[:nsteps] / save
    d = xspec.ProfileDrive(c, P, A, amp)
    d.Aser = (d.Aser * (amp / np.sqrt((d.Aser ** 2).mean()))).astype(np.float32)
    frames, _ = fl.run(c, d, p, nsteps, save)
    kern = units.smoothing_kernel(timescale.bold_fwhm_frames(frame_s, verbose=False),
                                  verbose=False)
    frames = bandpass.apply(units.smooth_frames(frames, kern), frame_s, 0.01, 0.08)
    a = np.asarray(d.Aser, np.float64)[:nfr * save].reshape(nfr, save, -1).sum(1)
    return frames[:nfr], a


def observable(f, kern, frame_s):
    import units, bandpass
    return bandpass.apply(units.smooth_frames(f, kern), frame_s, 0.01, 0.08)


def corr_map(Y, bt, lag):
    """Per-vertex correlation with the lagged boxcar, on the TR grid."""
    y = np.asarray(Y, np.float64)
    y = y - y.mean(0)
    bb = np.roll(np.asarray(bt, np.float64), lag)
    bb[:lag] = 0.0
    bb -= bb.mean()
    bb /= max(np.linalg.norm(bb), 1e-30)
    den = np.linalg.norm(y, axis=0)
    return np.where(den > 0, (y.T @ bb) / np.maximum(den, 1e-30), 0.0)


def on_off_variance(Y):
    """log2 of the variance during CHECKER over the variance during FIXATION.

    A multiplicative modulation of a zero-mean drive has no mean to show up in the boxcar
    correlation; it changes how much the vertex fluctuates. That is this map, and it is
    also what the ON-vs-OFF FC arms of the refit are looking at."""
    import fc_group_rbc as fg
    on = np.concatenate(fg.block_frames("CHECKER", TR, len(Y)))
    off = np.concatenate(fg.block_frames("FIXATION", TR, len(Y)))
    vo = Y[on].var(0)
    vf = Y[off].var(0)
    return np.log2(np.maximum(vo, 1e-30) / np.maximum(vf, 1e-30))


def _pearson(x, y):
    x = np.asarray(x, float) - np.mean(x); y = np.asarray(y, float) - np.mean(y)
    d = np.linalg.norm(x) * np.linalg.norm(y)
    return float(x @ y / d) if d > 0 else 0.0


def agreement(mod, emp, c, t, d_drive):
    from scipy.stats import spearmanr
    lab = c.lab[t.cols]
    names = parcel_names(c)
    vis = np.isin(lab, [i for i, n in enumerate(names) if n in VISUAL])
    out = dict(pearson=_pearson(mod, emp),
               spearman=float(spearmanr(mod, emp).statistic),
               visual=_pearson(mod[vis], emp[vis]))
    for lo, hi in ((0, 30), (30, 60), (60, 300)):
        m = (d_drive >= lo) & (d_drive < hi)
        out[f"d{lo}_{hi}"] = _pearson(mod[m], emp[m]) if m.sum() > 20 else np.nan
    return out


HDR = (f"  {'':<30s} {'peak r':>7s} {'r>60mm':>7s} | {'pearson':>7s} {'spear':>7s} "
       f"{'visual':>7s} {'0-30mm':>7s} {'30-60':>7s} {'>60':>7s}")


def _row(name, mod, ag, far):
    return (f"  {name:<30s} {mod.max():>7.4f} {far:>7.4f} | {ag['pearson']:>+7.4f} "
            f"{ag['spearman']:>+7.4f} {ag['visual']:>+7.4f} {ag['d0_30']:>+7.4f} "
            f"{ag['d30_60']:>+7.4f} {ag['d60_300']:>+7.4f}")


def run_sweep(c, t, tag, parcels, alphas, forms, lag_max, emp, d_drive, nwin=0,
              realise=0, seed=0, emp_piece=None, vprof=None, verbose=True):
    """The modulation depth swept, for each form, averaged over resting windows.

    Every window gets its own correlation map and the maps are averaged, which is what
    the empirical map is - a mean over subjects of per-subject correlations, not a
    correlation of the mean. The scatter over windows is reported with it."""
    Yb, R, sel, tags, a_rest, save, kern, frame_s = piece_fields(c, tag, parcels)
    nfr = Yb.shape[1]
    if realise:
        F, k = None, realise
        print(f"  resting background: {realise} INDEPENDENT {nfr}-frame draws of the "
              f"solved input, simulated fresh")
    else:
        F, k = rest_windows(tag, nfr)
        if nwin:
            k = min(k, nwin)
    rms = np.sqrt((a_rest[:, sel] ** 2).mean(0))
    print(f"  resting drive RMS per driven piece: " +
          ", ".join(f"{tags[i]} {v:.3g}" for i, v in zip(sel, rms)))
    bt = cm.boxcar(np.arange(NT) * TR)
    b_model = cm.boxcar(np.arange(nfr) * frame_s)

    # additive: the field is alpha * sum_j s_j * Yb_j, formed once per shape
    shapes = {"solve": rms / rms.mean(), "flat": np.ones(len(sel))}
    if emp_piece is not None:
        # Two ways to use the data's own within-V1 profile, and they are different
        # experiments. `fovea` drives only the pieces the data says respond and leaves the
        # rest alone. `empirical` keeps the sign, so the peripheral pieces are driven
        # DOWN while the foveal ones are driven up - a redistribution at constant total
        # drive rather than an addition. Both are normalised to the same mean magnitude as
        # the flat shape, so alpha means the same thing across all four.
        e = np.asarray(emp_piece, float)
        pos = np.maximum(e, 0.0)
        shapes["fovea"] = pos / max(np.abs(pos).mean(), 1e-30)
        shapes["empirical"] = e / max(np.abs(e).mean(), 1e-30)
    add = {nm: np.tensordot(s.astype(np.float32) * rms.mean().astype(np.float32),
                            Yb, axes=(0, 0))
           for nm, s in shapes.items()}
    if any(f.endswith(":vertex") for f in forms) and vprof is not None:
        import xspec, bo_step, subparcels
        z = np.load(os.path.join(RESULTS, f"xspec_{tag}.npz"), allow_pickle=True)
        p, _, _ = bo_step.unpack(z["x"], c)
        Rv = xspec.impulse_responses(c, [0], p, R.shape[1] * save, save,
                                     profiles=np.asarray(vprof)[None, :], verbose=False)
        fv = fftconvolve(b_model[:, None], Rv[0], axes=0)[:nfr]
        add["vertex"] = (observable(fv, kern, frame_s)
                         * np.float32(rms.mean())).astype(np.float32)
        print(f"  vertex profile: {int((np.abs(vprof) > 0).sum())} vertices, "
              f"range {vprof.min():+.4g} to {vprof.max():+.4g}")

    # Both forms are LINEAR in alpha, so each window's modulation field is built once
    # at unit depth and scaled. That turns the multiplicative arm from one convolution
    # per piece per depth per window into one per piece per window.
    # Accumulate, do not collect. Holding every window's 25 lag maps for every
    # (form, depth) is nwin x nlags x nV x 16 configurations - about 3 GB at 99 windows,
    # which is enough to have the run killed with no traceback. The sum is all the mean
    # needs, and the per-window peak at each lag is a tiny matrix.
    nlag = lag_max + 1
    nVt = len(t.cols)
    acc = {(f, al): np.zeros((nlag, nVt)) for f in forms for al in alphas}
    vacc = {kk: np.zeros(nVt) for kk in acc}
    peaks = {kk: [] for kk in acc}
    cnt = {kk: 0 for kk in acc}
    # Even and odd windows accumulated apart, at no extra cost. A map built from N draws
    # of a SMOOTH random field is not white: it has spatial structure with few effective
    # degrees of freedom, and such a map can correlate respectably with anything smooth by
    # chance. The only way to tell that from a real one is whether it repeats, so every
    # map here carries the agreement between two disjoint halves of its own windows.
    half = {kk: [np.zeros((nlag, nVt)), np.zeros((nlag, nVt))] for kk in acc}
    hcnt = {kk: [0, 0] for kk in acc}
    for wi in range(k):
        if realise:
            Fw, a_win = realise_field(c, tag, nfr, seed=5000 + wi)
            Fw = np.asarray(Fw, np.float32)
            a_win = a_win[:, sel]
        else:
            Fw = np.asarray(F[wi * nfr:(wi + 1) * nfr], np.float32)
            a_win = a_rest[wi * nfr:(wi + 1) * nfr][:, sel]
        unit = {}
        for form in forms:
            if form.startswith("add"):
                unit[form] = add[form.split(":")[1]]
            else:
                a_w = a_win
                if len(a_w) < nfr:
                    unit[form] = None
                    continue
                ex = np.zeros_like(Fw)
                for j2 in range(len(sel)):
                    m = (b_model * a_w[:, j2])[:, None]
                    ex += observable(fftconvolve(m, R[j2], axes=0)[:nfr], kern, frame_s)
                unit[form] = ex
        for form in forms:
            if unit[form] is None:
                continue
            for alpha in alphas:
                Yt = (Fw + np.float32(alpha) * unit[form])[::4][:NT][:, t.cols]
                # EVERY LAG IS KEPT, and the lag is chosen once at the end on the mean
                # over windows. Picking each window's own best lag is a per-window argmax
                # over 25 maps of a noiseless field, which biases each map upward and
                # survives the average - checkerboard.py refuses the same thing on the
                # data ("a per-subject argmax would bias every vertex upward") and the
                # model side has to refuse it too or the two are not the same estimator.
                gg = np.stack([corr_map(Yt, bt, s) for s in range(nlag)])
                kk = (form, alpha)
                acc[kk] += gg
                half[kk][wi % 2] += gg
                hcnt[kk][wi % 2] += 1
                vacc[kk] += on_off_variance(Yt)
                peaks[kk].append(gg.max(1))
                cnt[kk] += 1
                del gg
        if verbose:
            print(f"    window {wi+1}/{k}", flush=True)

    out = {}
    for form in forms:
        for alpha in alphas:
            kk = (form, alpha)
            if not cnt[kk]:
                continue
            A = acc[kk] / cnt[kk]                      # (lags, vertices)
            lag = int(np.argmax(A.max(1)))             # one lag, on the mean
            M = A[lag]
            V = vacc[kk] / cnt[kk]
            ag = agreement(M, emp, c, t, d_drive)
            nm = f"{form} alpha {alpha:g}"
            pk = np.array([p[lag] for p in peaks[kk]])
            hA = half[kk][0][lag] / max(hcnt[kk][0], 1)
            hB = half[kk][1][lag] / max(hcnt[kk][1], 1)
            ag["split"] = _pearson(hA, hB)
            ring = (d_drive >= 30) & (d_drive < 60)
            ag["split_ring"] = _pearson(hA[ring], hB[ring])
            print(_row(nm, M, ag, float(M[d_drive > 60].mean()))
                  + f"  | lag {lag*TR:4.1f}s  peak/win {pk.mean():.3f}+-{pk.std():.3f}"
                  + f"  var {V[d_drive < 5].mean():+.3f}/{V[d_drive > 60].mean():+.3f}"
                  + f"  split {ag['split']:+.3f} (ring {ag['split_ring']:+.3f})")
            out[nm] = dict(map=M, var=V, ag=ag, peaks=pk, lag=lag)
    return out, sel, tags, k


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", default="g7_s1.5_d25")
    ap.add_argument("--parcels", default="1", help="Glasser ids to modulate (1 = V1)")
    ap.add_argument("--alphas", default="0,0.02,0.05,0.1,0.2,0.5,1,2",
                    help="modulation depth, as a fraction of the piece's own resting "
                         "drive RMS")
    ap.add_argument("--forms", default="add:flat,add:fovea,add:empirical",
                    help="add:flat gives every V1 piece the same increment; add:solve "
                         "keeps the solve's own ratios (which are flat to 4%, so the two "
                         "are the same thing); add:fovea drives only the pieces the "
                         "empirical map says respond; add:empirical keeps the sign, "
                         "driving foveal up and peripheral DOWN; mul is a multiplicative "
                         "gain on the ongoing drive")
    ap.add_argument("--lag-max", type=int, default=24, dest="lag_max")
    ap.add_argument("--windows", type=int, default=0,
                    help="resting windows to average over (0 = all in the saved run)")
    ap.add_argument("--realise", type=int, default=0,
                    help="instead of cutting the saved run, simulate this many "
                         "INDEPENDENT 154 s resting draws. The floor of this measurement "
                         "falls as 1/sqrt(N), and the data averages 99 subjects, so N is "
                         "not a detail")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--emp", default=None)
    a = ap.parse_args()

    from mesh_cache import load_cortex
    import fc_score
    from diag_distance import distance_to_drive
    c = load_cortex("fsaverage5", verbose=False)
    t = fc_score.default_target(c, verbose=False)
    parcels = tuple(int(v) for v in a.parcels.split(","))

    emp_path = a.emp or os.path.join(CACHE, f"checkerboard_group_100_{t.nV}.npz")
    e = np.load(emp_path, allow_pickle=True)
    emp = np.asarray(e["response"], np.float64)
    print(f"  empirical: {int(e['n'])} subjects, lag {float(e['lag_s']):.2f} s, "
          f"peak {emp.max():+.4f}")

    z = np.load(os.path.join(RESULTS, f"xspec_{a.tag}.npz"), allow_pickle=True)
    tagz = list(z["tags"])
    sel0 = [i for i, s in enumerate(tagz) if int(s.split("_")[0]) in set(parcels)]
    d_drive = distance_to_drive(c, np.isin(z["labels"], sel0))[t.cols]

    alphas = [float(v) for v in a.alphas.split(",")]
    forms = a.forms.split(",")
    print(f"\n{HDR}")
    # what the data itself reads on the same axes
    print(_row("EMPIRICAL", emp, agreement(emp, emp, c, t, d_drive),
               float(emp[d_drive > 60].mean())))
    ep = empirical_v1(c, t, a.tag, sel0, parcels[0], emp)
    print(f"  empirical response per driven piece: " +
          ", ".join(f"{v:+.4f}" for v in ep))
    vp = (vertex_profile(c, t, a.tag, sel0, emp, parcels[0])
          if any(f.endswith(":vertex") for f in forms) else None)
    out, sel, tags, k = run_sweep(c, t, a.tag, parcels, alphas, forms, a.lag_max,
                                  emp, d_drive, nwin=a.windows, realise=a.realise,
                                  seed=a.seed, emp_piece=ep, vprof=vp)

    p = os.path.join(RESULTS, f"checkerboard_modulation_{a.tag}.npz")
    np.savez(p, empirical=emp, cols=t.cols, d_drive=d_drive, nwin=k,
             **{f"map_{k2}": v["map"] for k2, v in out.items()},
             **{f"var_{k2}": v["var"] for k2, v in out.items()},
             **{f"peaks_{k2}": v["peaks"] for k2, v in out.items()})
    print(f"\n  wrote {p}")


if __name__ == "__main__":
    main()
