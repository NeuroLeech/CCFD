"""Put the model on the fMRI clock, so the solver can be restricted to what BOLD sees.

The model has no intrinsic second. Distance per step is CFL-locked (c*dt = CFL*d_min
whatever c is, which is why bo_step found c0 inert), so the only geometric knob is `save`,
the steps per saved frame, and the mapping from frames to seconds is a free anchor.

That anchor was set two incompatible ways and nobody reconciled them:

  units.py declared a 300 mm/s cortical spread, which makes one frame 6.54 ms;
  bo_step matched model-frame FC against TR-sampled empirical FC, which implicitly makes
  one frame one TR.

They disagree by 99x. Under the first, a 1,120-frame window is 7.3 s and its lowest
frequency is 0.137 Hz - above the whole resting-state band, so no restriction to the fMRI
range is even expressible. Under the second the window is 722 s and reaches 0.0014 Hz.

This module takes the second reading, because it is the one the fit is actually evidence
for, and then OVERSAMPLES it: frames at TR/k so the model is sampled faster than BOLD, as
any comparison against a sampled observable should be. `save` and the per-step damping are
rescaled together so the physics per SECOND is unchanged - the same spread in mm/s and the
same decay in seconds, simply observed on a finer grid.

The pay-off is that the low-pass to the BOLD band becomes mild. The recorded collapse of
field rank 46.5 -> 7.2 under smoothing was measured with the 6.54 ms anchor, where
matching fMRI needs a ~100x low-pass; at TR/4 it needs ~4x.

  python timescale.py --oversample 4
"""
import argparse
import numpy as np

TR = 0.645                       # NKI enhanced, seconds
MM_PER_FRAME_AT_33 = 1.96        # units.model_speed, measured on the white surface
SAVE_REF = 33
BAND = (0.01, 0.10)              # resting-state band, Hz

MM_PER_STEP = MM_PER_FRAME_AT_33 / SAVE_REF


BOLD_TAU_S = 9.03        # 1/e autocorrelation time of the NKI timeseries, measured
BOLD_TAU_INT_S = 14.88   # integrated autocorrelation time; 38.8 independent samples/577s


def plan(oversample=4, decay_s=BOLD_TAU_S, spread_mm_s=None, damp_ref=2e-4,
         save_ref=SAVE_REF, tr=TR, verbose=True):
    """-> dict describing the model on the fMRI clock at frames of TR/oversample.

    Two knobs, `save` and the per-step damping, set two physical quantities, the spread in
    mm/s and the decay in seconds. Each is now pinned by something outside the fit:

    DECAY comes from the data. The mean BOLD autocorrelation over 20 subjects falls to 1/e
    at 9.0 s and crosses zero at 13.5 s, giving 38.8 effective independent samples in a
    577 s run. Carrying the unanchored medium's decay across instead gives 97.7 s and 5.9
    samples, so its FC estimate is far noisier than the data's over the same duration -
    and 97.7 s was never fitted, it is simply what the medium happened to have when the
    clock was arbitrary.

    SPREAD is then the single free parameter, and it is what has to be searched. It is not
    free of consequences: reach ~ spread x decay, so pinning the decay at 9 s means the
    old configuration's ~290 mm of reach needs about 32 mm/s rather than 3 mm/s.

    `decay_s=None` restores the old behaviour of carrying the reference decay across."""
    frame_s = tr / float(oversample)
    ref_spread = MM_PER_FRAME_AT_33 / tr                   # at one frame = one TR
    ref_decay = (1.0 / (damp_ref * save_ref)) * tr
    if decay_s is None:
        decay_s = ref_decay
    if spread_mm_s is None:
        spread_mm_s = ref_spread
    save = max(1, int(round(spread_mm_s * frame_s / MM_PER_STEP)))
    damp = frame_s / (float(decay_s) * save)
    out = dict(oversample=int(oversample), frame_s=frame_s, save=save, damp=damp,
               spread_mm_s=MM_PER_STEP * save / frame_s,
               decay_s=frame_s / (damp * save), tr=tr)
    out["reach_mm"] = out["spread_mm_s"] * out["decay_s"]
    out["decay_frames"] = 1.0 / (damp * save)
    if verbose:
        print(f"  frames at TR/{oversample} = {frame_s:.4f} s; save {save} steps/frame, "
              f"damping {damp:.4g}/step")
        print(f"    spread {out['spread_mm_s']:.2f} mm/s, decay {out['decay_s']:.1f} s "
              f"({out['decay_frames']:.0f} frames), reach {out['reach_mm']:.0f} mm")
        print(f"    reference medium: {ref_spread:.2f} mm/s, {ref_decay:.1f} s, "
              f"reach {ref_spread*ref_decay:.0f} mm;  BOLD 1/e decay {BOLD_TAU_S:.1f} s")
    return out


def frames_for(seconds, frame_s):
    return int(round(seconds / frame_s))


def band_bins(idx, ref_frames, frame_s, band=BAND, verbose=True):
    """Boolean over the solver's frequency samples: which lie in `band` (Hz)?

    `idx` are bin indices of an rfft over `ref_frames` samples, so bin b is the frequency
    b / (ref_frames * frame_s) Hz."""
    f_hz = np.asarray(idx, float) / (ref_frames * frame_s)
    keep = (f_hz >= band[0]) & (f_hz <= band[1])
    if verbose:
        print(f"  window {ref_frames} frames = {ref_frames*frame_s:.1f} s; "
              f"resolvable {f_hz.min():.4f} - {f_hz.max():.3f} Hz")
        print(f"    band {band[0]}-{band[1]} Hz keeps {int(keep.sum())} of {len(f_hz)} "
              f"solved frequencies"
              + (f" ({f_hz[keep].min():.4f}-{f_hz[keep].max():.4f} Hz)"
                 if keep.any() else "  NONE - the band is outside the window"))
    return keep


def band_grid(ref_frames, frame_s, nfreq, band=BAND, tail=8, tr=TR, verbose=True):
    """Frequency bins concentrated where BOLD actually has power.

    The default grid in xspec.transfer is geometric over the whole rfft range, which was
    the right choice while the model's band was arbitrary. Once the input is held to the
    measured spectrum it is the wrong one: 63% of the target power sits in 0.01-0.03 Hz
    and the geometric grid gives that three samples out of a hundred and seventeen.

    This puts `nfreq` samples geometrically across `band`, and `tail` samples spread over
    the remainder up to the fMRI Nyquist, so the solve keeps some ability to place power
    outside the band rather than being forbidden from it by the grid itself.

    Placement cannot create resolution. The rfft of an `ref_frames` window has bins spaced
    1/(ref_frames*frame_s) Hz, so at 1024 frames of 0.1613 s that is 0.0061 Hz and the
    whole of 0.01-0.03 Hz contains three distinct bins however many samples are asked for.
    Reaching that band properly needs a LONGER window, and since the impulse has decayed
    long before the end, zero-padding buys it for nothing: 4096 frames is 661 s and 0.0015
    Hz. This function warns when the request outruns the resolution."""
    nb = ref_frames // 2 + 1
    lo_b = max(1.0, band[0] * ref_frames * frame_s)
    hi_b = min(nb - 1.0, band[1] * ref_frames * frame_s)
    core = np.geomspace(lo_b, hi_b, nfreq)
    nyq_b = min(nb - 1.0, (1.0 / (2 * tr)) * ref_frames * frame_s)
    rest = np.geomspace(hi_b * 1.05, nyq_b, tail) if nyq_b > hi_b * 1.05 else []
    idx = np.unique(np.round(np.concatenate([[1.0], core, np.asarray(rest)])).astype(int))
    idx = idx[(idx >= 1) & (idx <= nb - 1)]
    df = 1.0 / (ref_frames * frame_s)
    avail = int(np.floor((band[1] - band[0]) / df))
    if avail < nfreq:
        print(f"  NOTE: {ref_frames} frames of {frame_s:.4f}s give {df:.4f} Hz bins, so "
              f"{band[0]}-{band[1]} Hz holds only {avail} distinct bins - asking for "
              f"{nfreq} cannot help. Pad to "
              f"{int(2**np.ceil(np.log2(nfreq/((band[1]-band[0])*frame_s))))} frames "
              f"to resolve them.")
    if verbose:
        f_hz = idx / (ref_frames * frame_s)
        inb = (f_hz >= band[0]) & (f_hz <= band[1])
        print(f"  frequency grid: {len(idx)} bins, {int(inb.sum())} inside "
              f"{band[0]}-{band[1]} Hz (geometric grid would give "
              f"{int(((np.unique(np.round(np.geomspace(1, nb-1, 192)).astype(int)) / (ref_frames*frame_s) >= band[0]) & (np.unique(np.round(np.geomspace(1, nb-1, 192)).astype(int)) / (ref_frames*frame_s) <= band[1])).sum())})")
    return idx


def bold_power(idx, ref_frames, frame_s, w, psd_path="data/cache/bold_psd.npz",
               floor=0.0, verbose=True):
    """Target OUTPUT power per solved frequency, from the measured BOLD spectrum.

    The empirical spectrum is estimated on the fMRI grid (TR spacing, 895 frames) from
    per-vertex unit-variance timeseries, which is the same normalisation the FC uses. Each
    solved sample stands for a band of width w[f], so the target is the empirical spectral
    DENSITY at that frequency times the band width - otherwise a geometric frequency grid
    would systematically under-weight the wide high-frequency bins.

    Frequencies outside the measured range get `floor` (default zero, i.e. forbidden),
    which is the honest reading: above the fMRI Nyquist there is no measurement, and
    below the lowest resolvable frequency of a 577 s run there is none either."""
    import os
    z = np.load(psd_path)
    fe, Pe = z["f"], z["P"]
    f_hz = np.asarray(idx, float) / (ref_frames * frame_s)
    dens = np.interp(f_hz, fe[1:], Pe[1:] / np.gradient(fe)[1:], left=floor, right=floor)
    out = np.clip(dens, 0.0, None) * np.asarray(w, float)
    inside = (f_hz >= fe[1]) & (f_hz <= fe[-1])
    if verbose:
        tot = out.sum()
        print(f"  BOLD spectrum target: {int(inside.sum())} of {len(f_hz)} solved "
              f"frequencies lie in the measured range {fe[1]:.4f}-{fe[-1]:.3f} Hz")
        if tot > 0:
            share = out / tot
            for lo, hi in ((0.0, 0.01), (0.01, 0.03), (0.03, 0.06), (0.06, 0.1),
                           (0.1, 10.0)):
                m = (f_hz > lo) & (f_hz <= hi)
                if m.any():
                    print(f"    {lo:5.3f}-{hi:<5.3f} Hz: {int(m.sum()):>3d} bins, "
                          f"target share {share[m].sum():.4f}")
    return out


def bold_fwhm_frames(frame_s, tr=TR, verbose=True):
    """Low-pass FWHM, in frames, that brings the model to the fMRI observation band.

    Sized so the filter's cutoff sits at the fMRI Nyquist 1/(2 TR): anything faster is not
    something BOLD sampled at TR could have reported, so it must not be scored. This is
    the SAME kernel units.smoothing_kernel builds and units.kernel_response multiplies
    into the transfer function, so both sides of the comparison see one filter."""
    fwhm = (2.0 * tr) / frame_s
    if verbose:
        print(f"  low-pass FWHM {fwhm:.2f} frames (= 2 TR = {2*tr:.3f} s), "
              f"cutting at the fMRI Nyquist {1/(2*tr):.3f} Hz")
    return fwhm


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--oversample", type=int, default=4)
    ap.add_argument("--pad", type=int, default=1120)
    ap.add_argument("--seconds", type=float, default=577.0,
                    help="realisation length in seconds (895 TR = 577 s of NKI)")
    a = ap.parse_args()

    print(f"  reference: the fitted medium read as one frame = one TR "
          f"(save {SAVE_REF}, damping 2e-4)")
    print(f"  the alternative anchor, 300 mm/s, makes one frame {1.96/300*1000:.2f} ms "
          f"and puts the whole model above the BOLD band\n")
    for k in (1, 2, 4, 8):
        p = plan(k, verbose=False)
        keep = band_bins(np.unique(np.round(np.geomspace(1, a.pad // 2, 192)).astype(int)),
                         a.pad, p["frame_s"], verbose=False)
        f_lo = 1.0 / (a.pad * p["frame_s"])
        print(f"  TR/{k}: frame {p['frame_s']:.4f}s  save {p['save']:>2d}  "
              f"damp {p['damp']:.3g}  spread {p['spread_mm_s']:.2f} mm/s  "
              f"decay {p['decay_s']:.0f}s  window {a.pad*p['frame_s']:.0f}s  "
              f"f_lo {f_lo:.4f}Hz  band bins {int(keep.sum())}/{len(keep)}  "
              f"{frames_for(a.seconds, p['frame_s'])} frames for {a.seconds:.0f}s")
    print()
    p = plan(a.oversample)
    band_bins(np.unique(np.round(np.geomspace(1, a.pad // 2, 192)).astype(int)),
              a.pad, p["frame_s"])
    bold_fwhm_frames(p["frame_s"])
    print(f"  realisation: {frames_for(a.seconds, p['frame_s'])} frames "
          f"for {a.seconds:.0f} s (the empirical run length)")


if __name__ == "__main__":
    main()
