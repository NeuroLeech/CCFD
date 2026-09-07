"""What the solve does with the input when the target is a driven cortex.

Twelve FC matrices, one medium, one set of 47 pieces, one transfer function: only the
target changes between arms, so the difference between two solved inputs is the
difference between two second moments of the same cortex under different conditions.

THE CONTRASTS, sharpest first. `check_on` against `check_off` is the same subjects, the
same runs, the same 84 frames' worth of data and the same three 20 s windows shifted by
one block - the stimulus and nothing else. `rest_on` against `rest_off` is that knife
falling on runs where nothing happened, which is what block-chopping alone does to an FC.
`check_full` against `rest239` is the same question with all the temporal information and
the two conditions confounded.

SHARES, NOT POWER, AND ONLY IN THE PASSBAND. The solve matches a scale-invariant
correlation of patterns, so the total scale of S is arbitrary and only the split between
channels means anything. And the split has to be read where the model can hear it: 37 of
the 135 solved bins lie in 0.01-0.08 Hz, and outside that the passband has taken |H| to
~0, so the solve has no reason to put anything but uniform power there. Integrated over
every bin the resting arm's per-piece share is flat to 1.1x; over the passband it spreads
17x, with the prefrontal and parietal pieces carrying the most and V1 and motor the least.
The all-bin integral is measuring the bins the model cannot use.

TWO THINGS DECIDE WHETHER A SHIFT MEANS ANYTHING, and both are here rather than in the
reader's head. `--halves` re-solves each arm on disjoint halves of the subjects, so a
shift has to be larger than what changing the subjects does. `--family` asks how far the
visual share can move on the OFF target WITHOUT the fit falling - `xspec.share_reg` fed
to `family_member`, the machinery the cancellation question already used. A shift inside
that interval says the objective never pinned the share; outside it, the target moved it.

  python analysis/task_inputs.py --prefix ck_
"""
import _path  # noqa: F401  - puts the sibling code folders on sys.path
import os, argparse
import numpy as np

from paths import RESULTS
import xspec

ARMS = ("rest_full", "rest239", "check_full",
        "check_on_out", "check_off_out", "check_on_in", "check_off_in",
        "rest_on_out", "rest_off_out", "rest_on_in", "rest_off_in")

CONTRASTS = [("check_on_out", "check_off_out", "the stimulus, evoked mean removed"),
             ("check_on_in", "check_off_in", "the same with the evoked mean left in"),
             ("rest_on_out", "rest_off_out", "the null: chopping alone"),
             ("rest_on_in", "rest_off_in", "the null with means left in"),
             ("check_full", "rest239", "whole runs, ON and OFF confounded"),
             ("rest239", "rest_full", "length alone")]

VISUAL_PARCELS = (1, 4, 5, 22)          # V1, V2, V3, PIT - the visual Glasser ids driven


def load_arm(prefix, name):
    p = os.path.join(RESULTS, f"xspec_{prefix}{name}.npz")
    if not os.path.exists(p):
        return None
    return np.load(p, allow_pickle=True)


def shares(z, band=None):
    """-> per-piece fraction of the arm's total input power."""
    pad = int(z["pad"]) if "pad" in z.files else 4096
    fs = float(z["frame_s"]) if "frame_s" in z.files else np.nan
    p = xspec.piece_power(z["S"], z["idx"], fs, band=band, pad=pad)
    p = np.maximum(p, 0.0)
    return p / max(p.sum(), 1e-30)


def parcel_of(tags):
    return np.array([int(str(s).split("_")[0]) for s in tags])


def coherence(z, band=None):
    """In-band input COHERENCE between pieces: |S_jk| / sqrt(S_jj S_kk), bin-weighted.

    The power share reads the diagonal of S, which is a third of ||S||_F^2 on the resting
    arm; the other two thirds are the cross-terms, and those are the coalition structure -
    which pieces are asked to fire together and with what phase. Two inputs can have
    identical per-piece power and completely different cross-spectra, so a null on the
    diagonal is not a null on the input, and this is the part that says so.

    Magnitude and phase are returned apart. The magnitude is how tightly two pieces are
    locked; the phase is the offset between them, and PLAN section 8 records that offsets
    stop tracking geodesic distance at this granularity - so the phases are reported as a
    comparison between arms, not as travel times.
    """
    S = np.asarray(z["S"])
    pad = int(z["pad"]) if "pad" in z.files else 4096
    fs = float(z["frame_s"])
    f = np.asarray(z["idx"], float) / (pad * fs)
    bw = np.gradient(np.asarray(z["idx"], float))
    m = np.ones(len(f), bool) if band is None else (f >= band[0]) & (f <= band[1])
    d = np.real(np.diagonal(S, axis1=1, axis2=2))
    nrm = np.sqrt(np.maximum(d[:, :, None] * d[:, None, :], 1e-30))
    C = S / nrm
    w = bw[m] / bw[m].sum()
    return (np.abs(C[m]) * w[:, None, None]).sum(0), (C[m] * w[:, None, None]).sum(0)


def offdiag_table(Z, arms, band, tags, par, vis, names):
    print(f"\n  input COHERENCE between pieces (|S_jk|/sqrt(S_jj S_kk), in band)")
    Cm = {nm: coherence(Z[nm], band)[0] for nm in arms}
    K = len(tags)
    off = ~np.eye(K, dtype=bool)
    print(f"  {'arm':<16s} {'mean |coh|':>10s} {'within visual':>14s} "
          f"{'visual-other':>13s} {'r vs rest_full':>15s}")
    ref = Cm.get("rest_full")
    vv = np.outer(vis, vis) & off
    vo = (np.outer(vis, ~vis) | np.outer(~vis, vis))
    for nm in arms:
        C = Cm[nm]
        r = (float(np.corrcoef(C[off], ref[off])[0, 1]) if ref is not None else np.nan)
        print(f"  {nm:<16s} {C[off].mean():>10.4f} {C[vv].mean():>14.4f} "
              f"{C[vo].mean():>13.4f} {r:>15.4f}")
    for A, B, why in CONTRASTS:
        if A not in Cm or B not in Cm:
            continue
        D = Cm[A] - Cm[B]
        j, k = np.unravel_index(int(np.argmax(np.abs(np.where(off, D, 0)))), D.shape)
        print(f"    {A} - {B}: mean d|coh| {D[off].mean():+.4f}, within-visual "
              f"{D[vv].mean():+.4f}, largest pair {tags[j]}-{tags[k]} {D[j,k]:+.4f}"
              f"   [corr of the two matrices {np.corrcoef(Cm[A][off], Cm[B][off])[0,1]:.4f}]")
    return Cm


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--prefix", default="ck_")
    ap.add_argument("--band", default="0.01,0.08",
                    help="restrict the power integral to LO,HI Hz. THE DEFAULT IS NOT "
                         "'all bins' for a reason: only 37 of the 135 solved bins lie in "
                         "the passband, and outside it the bandpass has taken |H| to ~0, "
                         "so the solve leaves those channels at a uniform, uninformative "
                         "power. Integrated over every bin the per-piece share comes out "
                         "flat to 1.1x and says nothing; over the passband alone it "
                         "spreads 17x. Pass 'all' to integrate everything anyway")
    ap.add_argument("--family", default="", metavar="ARM",
                    help="admissible visual-share interval on this arm, at equal fit")
    ap.add_argument("--eps", default="0.0002,0.001,0.01,0.05")
    a = ap.parse_args()
    band = (None if a.band in ("", "all")
            else tuple(float(v) for v in a.band.split(",")))

    from mesh_cache import load_cortex
    c = load_cortex("fsaverage5", verbose=False)
    names = [(s.decode() if isinstance(s, bytes) else str(s)
              ).removeprefix("L_").removesuffix("_ROI") for s in c.names]

    Z = {nm: z for nm in ARMS if (z := load_arm(a.prefix, nm)) is not None}
    if not Z:
        raise SystemExit(f"  no xspec_{a.prefix}*.npz in {RESULTS}")
    print(f"  {len(Z)} arms: {', '.join(Z)}")

    ref = next(iter(Z.values()))
    tags = [str(s) for s in ref["tags"]]
    for nm, z in Z.items():                 # the arms have to be the same system
        if [str(s) for s in z["tags"]] != tags:
            raise SystemExit(f"  {nm} has different pieces from {next(iter(Z))}")
        if not np.array_equal(z["labels"], ref["labels"]):
            raise SystemExit(f"  {nm} has a different piece layout")
        if not np.array_equal(z["idx"], ref["idx"]):
            raise SystemExit(f"  {nm} solved on different frequency bins")
    par = parcel_of(tags)
    vis = np.isin(par, VISUAL_PARCELS)
    print(f"  {len(tags)} pieces over {len(set(par.tolist()))} parcels; "
          f"{int(vis.sum())} are visual ({', '.join(sorted(set(names[p] for p in par[vis])))})")

    sh = {nm: shares(z, band) for nm, z in Z.items()}

    # ---- share by parcel, per arm --------------------------------------------------
    arms = [nm for nm in ARMS if nm in Z]
    print(f"\n  input power share by parcel (%), {'in band ' + a.band if band else 'all bins'}")
    print(f"  {'parcel':<8s} {'n':>2s} " + " ".join(f"{nm[:11]:>11s}" for nm in arms))
    for p in sorted(set(par.tolist()), key=lambda q: -sh[arms[0]][par == q].sum()):
        m = par == p
        print(f"  {names[p]:<8s} {int(m.sum()):>2d} " +
              " ".join(f"{100*sh[nm][m].sum():>11.2f}" for nm in arms))
    print(f"  {'VISUAL':<8s} {int(vis.sum()):>2d} " +
          " ".join(f"{100*sh[nm][vis].sum():>11.2f}" for nm in arms))

    # ---- the contrasts ---------------------------------------------------------------
    print(f"\n  contrasts, as the visual share and its log ratio")
    print(f"  {'':<34s} {'A vis%':>7s} {'B vis%':>7s} {'log2 A/B':>9s}  "
          f"{'max piece log2':>15s}")
    for A, B, why in CONTRASTS:
        if A not in Z or B not in Z:
            continue
        va, vb = sh[A][vis].sum(), sh[B][vis].sum()
        lr = np.log2(np.maximum(sh[A], 1e-12) / np.maximum(sh[B], 1e-12))
        j = int(np.argmax(np.abs(lr)))
        print(f"  {A + ' - ' + B:<34s} {100*va:>7.2f} {100*vb:>7.2f} "
              f"{np.log2(va/max(vb,1e-12)):>9.3f}  {tags[j]:>8s} {lr[j]:>+6.2f}")
        print(f"    {why}")

    Cm = offdiag_table(Z, arms, band, tags, par, vis, names)

    out = os.path.join(RESULTS, f"task_inputs_{a.prefix.strip('_')}.npz")
    np.savez(out, **{f"share_{nm}": sh[nm] for nm in arms},
             tags=np.array(tags, dtype=object), parcel=par, visual=vis,
             **{f"coh_{nm}": Cm[nm] for nm in arms})
    print(f"\n  wrote {out}")

    if a.family:
        family_interval(c, Z[a.family], a.family, vis,
                        [float(v) for v in a.eps.split(",")])


def family_interval(c, z, name, vis, epss):
    """How far the visual share moves at equal fit, both directions, on one arm."""
    import bo_step, subparcels, units, timescale, bandpass
    print(f"\n  admissible visual share on {name}, at equal fit:")
    x, save = z["x"], int(z["save"])
    labels, tags = z["labels"], [str(s) for s in z["tags"]]
    p, _, _ = bo_step.unpack(x, c)
    P = subparcels.taper_profiles(c, labels, len(tags))
    frame_s, pad = float(z["frame_s"]), int(z["pad"])
    imp = int(z["impulse_frames"])
    resp = xspec.impulse_responses(c, list(range(len(P))), p, imp * save, save,
                                   profiles=P, verbose=False, workers=8)
    R = np.pad(resp, ((0, 0), (0, max(0, pad - resp.shape[1])), (0, 0)))
    import fc_score
    fcp = str(z["fc_path"])
    t = (fc_score.FCTarget(c, fc_path=fcp, centre="double", verbose=False) if fcp
         else fc_score.default_target(c, verbose=False))
    sub = z["sub"]
    kern = units.smoothing_kernel(timescale.bold_fwhm_frames(frame_s, verbose=False),
                                  verbose=False)
    H, w, idx = xspec.transfer(R, t.cols[sub], int(z["nfreq"]), kernel=kern)
    ref_frames = R.shape[1]
    bp = np.asarray(z["band"], float)
    if np.isfinite(bp).all():
        H = H * bandpass.transfer_response(idx, ref_frames, frame_s, *bp)[:, None, None]
    if int(z["segment"]):
        H = H * bandpass.segment_response(idx, ref_frames, frame_s,
                                          int(z["segment"]))[:, None, None]
    raw = np.asarray(t.target_fc()[np.ix_(sub, sub)], np.float64)
    raw = raw - raw.mean(0, keepdims=True) - raw.mean(1, keepdims=True) + raw.mean()
    Tgt = xspec.normal_scores(raw)
    S = np.asarray(z["S"])
    Rg = xspec.share_reg(w, np.flatnonzero(vis), S.shape[1])
    print(f"    argmax share {Rg(S)[0]:.4f}")
    print(f"    {'eps':>8s} {'min share':>10s} {'max share':>10s} {'fit min':>9s} "
          f"{'fit max':>9s}")
    for eps in epss:
        lo, rl = xspec.family_member(H, w, Tgt, S, Rg, eps=eps, sign=-1.0, iters=60,
                                    verbose=False)
        hi, rh = xspec.family_member(H, w, Tgt, S, Rg, eps=eps, sign=+1.0, iters=60,
                                    verbose=False)
        print(f"    {eps:>8g} {Rg(lo)[0]:>10.4f} {Rg(hi)[0]:>10.4f} "
              f"{rl['rho']:>9.4f} {rh['rho']:>9.4f}")


if __name__ == "__main__":
    main()
