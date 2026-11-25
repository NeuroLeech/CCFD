"""Realise a solved cross-spectrum at a different length, without re-solving it.

`--seconds` does not touch the solve. `S(f)` is fitted in frequency space against `H`,
which is built from the impulse responses and the passband; the realisation length enters
only afterwards, when a drive is drawn from `S` and integrated. So two runs that differ
only in `--seconds` share one solve, and comparing them does not need it repeated.

That is not a small effect and it is not part of the model. `asc_first_area_100` scored
**+0.6930 +- 0.0026** realised over 2,308 s and **+0.6186 +- 0.0212** realised over 577 s
from a solve that is identical to four decimals in solve-space Spearman (+0.7291) and to
three in gap (0.047). Seven hundredths of Spearman, and the draw scatter falls 8x, from a
protocol parameter. Any table that mixes realisation lengths is not comparing models.

Everything needed is in the npz: `x` carries the medium, `profiles` the input basis,
`band`/`frame_s`/`segment` the observable, `idx`/`ref_frames` the frequency grid. The one
thing it never stored is the smoothing width, which `--bold-smooth` derives from the
clock - so that is read from the provenance stamp, and a file written before
`core/provenance.py` existed has to be told with `--smooth`.

  python fit/rescore.py --tag graded100_nolag --seconds 2308 --draws 2

A solve fitted with `--smooth-fwhm` (6 mm surface smoothing folded into H) is scored with
the same smoothing matrix applied to its realisations, as best_fit scored it; without it a
rescore measures a different observable from the one the solve was fitted and recorded on.

`--drive` replaces the solved cross-spectrum with one of the controls the wave analysis uses
(`interplay.control_drive`, imported rather than re-implemented): `independent` keeps each
input's power and removes all coherence, `random` keeps the eigenvalues with Haar-random
eigenvectors, `timing` keeps every input's power and every coherence magnitude and gives each
input a random phase per solved bin. They are applied to S on its solved grid and realised
the way every solve is, so between solved bins the interpolation of a phase-scrambled S also
loses some coherence magnitude.

`--jitter-s` keeps the solved S exactly and shifts each input's drawn drive by its own random
delay (normal, sd in seconds; the same delays for every draw of one seed). The drive realise
returns is periodic, so a circular shift is exactly a phase rotation e^{-2 pi i f tau_k} at
every bin: each input's spectrum and every coherence magnitude are untouched, and only the
relative timing between inputs moves. Scored over a range of sd, it says how precisely in
seconds the inputs' timing has to be held for the fit.

  python fit/rescore.py --tag nobp_sm6_s3_box --seconds 577 --draws 4 --drive timing --seed 0
  python fit/rescore.py --tag nobp_sm6_s3_box --seconds 577 --draws 4 --jitter-s 5
"""
import _path  # noqa: F401  - puts the sibling code folders on sys.path
import os, time, json, argparse
import numpy as np

import provenance
from mesh_cache import load_cortex
from paths import RESULTS
import fc_score, xspec, bo_step, units, timescale

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_solution(tag):
    """-> (npz, provenance args). The solve and everything needed to replay it."""
    path = os.path.join(RESULTS, f"xspec_{tag}.npz")
    if not os.path.exists(path):
        raise SystemExit(f"  no {path}")
    z = np.load(path, allow_pickle=True)
    return z, provenance.read(z).get("args", {})


def rescore_torch(tag, ref_tag, seconds, draws, device="mps"):
    """Realise a torch fit at `seconds` through the integrator it was fitted with.

    The convex path below cannot do this. A torch fit may carry nl_flux, nl_adv and a
    grading that is not expressible as map coefficients at all, and xspec.score_realisation
    left to itself integrates with fluid.run - the numpy solver, which has none of them.
    Scoring that way would report the LINEAR score for a nonlinear fit, which torch_fit's
    own run_fn docstring calls the one discrepancy that invalidates a run rather than
    adding noise. So the Fit is rebuilt at the new length, the medium restored from the
    checkpoint, and everything routed through fit.run_fn() exactly as the fit itself
    scored.

    Only `seconds` differs from the fit's own evaluation. G is interpolated onto the
    longer grid the same way torch_fit.drive does at any length."""
    import torch
    from torch_fit import Fit, load_reference, restore_medium

    z = np.load(os.path.join(RESULTS, f"torchfit_{tag}.npz"), allow_pickle=True)
    ref = load_reference(ref_tag)
    c = load_cortex("fsaverage5", verbose=False)
    t = fc_score.default_target(c, verbose=True)
    fit = Fit(ref, c, t, seconds, device=device, amp=float(z["amp"]),
              nl_flux=float(z["nl_flux"]), nl_adv=float(z["nl_adv"]),
              Ld=float(z["Ld"]), maps_scale=float(z["maps_scale"]),
              speed=int(z["speed"]) if "speed" in z.files else 1)
    learned = restore_medium(fit, z)
    with torch.no_grad():
        H = fit.medium()[0] if learned else fit.Hf
    H = np.asarray(H.cpu() if torch.is_tensor(H) else H)
    print(f"  {tag}: ref {ref_tag}, nl_flux {float(z['nl_flux']):g}, "
          f"nl_adv {float(z['nl_adv']):g}, Ld {float(z['Ld']):g}, "
          f"medium {'restored from the checkpoint' if learned else 'fixed'}")
    print(f"  H {H.min():.3f}-{H.max():.3f} ({H.max()/H.min():.1f}x), "
          f"realise {fit.nframes} frames ({seconds:.0f}s) = {fit.nsteps} steps, "
          f"{draws} draws", flush=True)

    cdt = torch.complex64 if fit.dtype == torch.float32 else torch.complex128
    G = torch.tensor(z["G"], dtype=cdt, device=fit.cdev)
    sims, gaps, rks = [], [], []
    for sd in range(draws):
        gen = torch.Generator(device=fit.cdev).manual_seed(10_000 + sd)
        with torch.no_grad():
            A = fit.drive(G, fit.eta(gen))
        Af = A.detach().cpu().numpy()[::fit.save] * fit.save
        r = xspec.score_realisation(
            c, t, fit.p, Af, save=fit.save, amp=fit.amp, profiles=ref["P"],
            kernel=fit.kern, band=ref["band"], frame_s=ref["frame_s"],
            segment=ref["segment"] or None, run_fn=fit.run_fn())
        sims.append(r["sim"]); gaps.append(r["gap"]); rks.append(r["rank"])
        print(f"    draw {sd}: sim {r['sim']:+.4f}", flush=True)
    return sims, gaps, rks, fit.nframes


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", required=True, help="results/xspec_<tag>.npz")
    ap.add_argument("--torch-ref", default=None, dest="torch_ref",
                    help="rescore results/torchfit_<tag>.npz instead, against this "
                         "reference basis, through the integrator it was fitted with")
    ap.add_argument("--seconds", type=float, required=True,
                    help="realisation length. 2308 is the 14,313-frame protocol the "
                         "100-channel runs were originally scored at; 577 is one NKI run")
    ap.add_argument("--draws", type=int, default=2)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--smooth", type=float, default=None,
                    help="BOLD smoothing FWHM in frames. Read from the provenance stamp "
                         "when there is one; required for a file written without it, "
                         "because scoring a smoothed model against the target unsmoothed "
                         "is not the same measurement")
    ap.add_argument("--save-frames", action="store_true", dest="save_frames",
                    help="write frames_/drive_<tag>_<seconds>s.npy for draw 0")
    ap.add_argument("--drive", default="fitted",
                    choices=("fitted", "independent", "random", "timing"),
                    help="score a control cross-spectrum instead of the solved one "
                         "(interplay.control_drive)")
    ap.add_argument("--seed", type=int, default=0,
                    help="seed for the random/timing controls and the jitter delays")
    ap.add_argument("--jitter-s", type=float, default=0.0, dest="jitter_s",
                    help="shift each input's drive by a random delay with this sd (s)")
    a = ap.parse_args()

    if a.torch_ref:
        t0 = time.time()
        sims, gaps, rks, frames = rescore_torch(a.tag, a.torch_ref, a.seconds, a.draws)
        print(f"\n  realised over {frames} frames ({a.seconds:.0f}s), {a.draws} draws: "
              f"sim {np.mean(sims):+.4f} +- {np.std(sims):.4f}   "
              f"gap {np.mean(gaps):.3f}   rank {np.mean(rks):.1f}   "
              f"[{time.time()-t0:.0f}s]")
        sp = os.path.join(RESULTS, f"scores_torchfit_{a.tag}.json")
        rec = {}
        if os.path.exists(sp):
            try:
                rec = json.load(open(sp))
            except (ValueError, OSError):
                rec = {}
        rec[f"{int(round(a.seconds))}s_{a.draws}d"] = dict(
            sim=float(np.mean(sims)), sim_sd=float(np.std(sims)),
            gap=float(np.mean(gaps)), field_rank=float(np.mean(rks)),
            frames=int(frames), draws=int(a.draws), seconds=float(a.seconds),
            ref=a.torch_ref)
        json.dump(rec, open(sp, "w"), indent=2, sort_keys=True)
        print(f"  wrote {os.path.relpath(sp, ROOT)}")
        return

    z, pa = load_solution(a.tag)
    c = load_cortex("fsaverage5", verbose=False)
    fc_path = str(z["fc_path"]) if "fc_path" in z else ""
    centre = pa.get("centre", "double")
    t = (fc_score.FCTarget(c, fc_path=fc_path, centre=centre, verbose=True)
         if fc_path else fc_score.default_target(c, centre=centre, verbose=True))

    p, save, _ = bo_step.unpack(z["x"], c)
    p["map_clip"] = str(z["map_clip"]) if "map_clip" in z else "none"
    p["decimate"] = str(z["decimate"]) if "decimate" in z else "snapshot"
    P = np.asarray(z["profiles"], np.float32)
    idx = z["idx"].astype(int)
    # above 512 channels realise takes the factored path; factor ONCE rather than per draw
    # (135 eigendecompositions of a 1,880-square matrix each time), and never hold the full S
    big = xspec.solution_K(z) > 512
    fac = xspec.load_factors(z) if big else None
    S = None if big else xspec.load_S(z)
    if a.drive != "fitted":
        if big:
            raise SystemExit("  --drive needs the full S; this solve stores only its factor")
        from interplay import control_drive
        S = control_drive(dict(S=S, name=a.tag), a.drive, seed=a.seed)["S"]
    ref_frames = int(z["ref_frames"])
    frame_s = float(z["frame_s"])
    # the solve's own surface smoothing, applied to the realisations as best_fit did
    sm_fwhm = float(z["smooth_fwhm"]) if "smooth_fwhm" in z.files else 0.0
    Kscore = None
    if sm_fwhm > 0:
        import surface_smooth
        Kscore = surface_smooth.smoothing_matrix(c, sm_fwhm)
    shifts = None
    if a.jitter_s > 0:
        rng = np.random.default_rng(a.seed)
        shifts = np.round(rng.normal(0.0, a.jitter_s, P.shape[0]) / frame_s).astype(int)

    def draw(d):
        A = xspec.realise(S, idx, frames, ref_frames=ref_frames, seed=1000 + d,
                          factors=fac)
        if shifts is not None:      # periodic drive: a circular shift is a pure delay
            A = np.stack([np.roll(A[:, k], s) for k, s in enumerate(shifts)], axis=1)
        return A
    band = tuple(z["band"]) if "band" in z and np.isfinite(z["band"]).all() else None
    hrf = bool(z["hrf"]) if "hrf" in z.files else False     # best_fit.py --hrf
    seg = int(z["segment"]) if "segment" in z and z["segment"].size else 0

    smooth = a.smooth
    if smooth is None:
        smooth = pa.get("smooth")
        if smooth is None and pa.get("bold_smooth"):
            smooth = timescale.bold_fwhm_frames(frame_s, verbose=False)
    if smooth is None:
        raise SystemExit(
            f"  xspec_{a.tag}.npz carries no provenance and no smoothing width. Pass "
            f"--smooth explicitly (--bold-smooth on this clock gives "
            f"{timescale.bold_fwhm_frames(frame_s, verbose=False):.0f}), or --smooth 0 "
            f"for a run that had none")
    kern = units.smoothing_kernel(float(smooth), verbose=False) if smooth else None

    frames = timescale.frames_for(a.seconds, frame_s)
    print(f"  {a.tag}: {P.shape[0]} channels, {len(idx)} solved frequencies, save {save}")
    print(f"  drive {a.drive}" + ("" if a.drive == "fitted" else f" (seed {a.seed})")
          + ("" if shifts is None else
             f", delays sd {a.jitter_s:g}s (seed {a.seed}; realised sd "
             f"{np.std(shifts) * frame_s:.2f}s)")
          + (f", surface smoothing {sm_fwhm:g} mm" if sm_fwhm > 0 else ""))
    print(f"  medium from the file: a {np.round(p.get('a', ()), 3)}, "
          f"b {np.round(p.get('b', ()), 3)}, Ld {p['Ld']:.4g}, "
          f"map_clip {p['map_clip']}")
    print(f"  realise {frames} frames ({a.seconds:.0f}s) at {frame_s:.5g}s, "
          f"{a.draws} draws, smoothing FWHM {float(smooth):g} frames"
          + ("" if band is None else f", band {band[0]:g}-{band[1]:g} Hz")
          + (", through the HRF" if hrf else "")
          + ("" if pa else "   [no provenance on this file]"), flush=True)

    t0 = time.time()
    sims, gaps, rks = [], [], []
    pool = res = None
    if a.workers > 1 and a.draws > 1:
        rest = [draw(d) for d in range(1, a.draws)]
        pool, res = xspec.parallel_scores(c, t, p, rest, save, P, kern, a.workers,
                                          band=band,
                                          frame_s=frame_s if (band or hrf) else None,
                                          segment=seg, smooth=Kscore, hrf=hrf)
        print(f"  draws 2-{a.draws} over {min(a.workers, len(rest))} workers", flush=True)
    for d in range(1 if pool is not None else a.draws):
        A = draw(d)
        r = xspec.score_realisation(c, t, p, A, save=save, profiles=P, kernel=kern,
                                    band=band, frame_s=frame_s if (band or hrf) else None,
                                    segment=seg, smooth=Kscore, hrf=hrf)
        sims.append(r["sim"]); gaps.append(r["gap"]); rks.append(r["rank"])
        if d == 0 and a.save_frames:
            sec = int(round(a.seconds))
            # The SAME realisation with nothing applied - no BOLD kernel, no passband - so
            # viz/render_bandpass.py can put the fluid itself on the top row. Its own
            # resimulate_raw would re-integrate from the drive without reading map_clip,
            # rebuilding an UNCLIPPED medium, and would leave the kernel on. Realised here
            # instead, same drive and same medium as the observable beside it.
            raw = xspec.score_realisation(c, t, p, A, save=save, profiles=P, kernel=None,
                                          band=None, frame_s=None, segment=seg)
            np.save(os.path.join(RESULTS, f"frames_{a.tag}_{sec}s.npy"), r["frames"])
            np.save(os.path.join(RESULTS, f"frames_{a.tag}_{sec}s_raw.npy"),
                    np.asarray(raw["frames"], np.float32))
            np.save(os.path.join(RESULTS, f"drive_{a.tag}_{sec}s.npy"), r["drive"].Aser)
            print(f"    wrote frames_{a.tag}_{sec}s.npy and _raw.npy "
                  f"(the bare field: no BOLD kernel, no passband)", flush=True)
    if pool is not None:
        for sim, gap, rk, _sa in res.get():
            sims.append(sim); gaps.append(gap); rks.append(rk)
        pool.close(); pool.join()

    print(f"\n  realised over {frames} frames ({a.seconds:.0f}s), {a.draws} draws: "
          f"sim {np.mean(sims):+.4f} +- {np.std(sims):.4f}   "
          f"gap {np.mean(gaps):.3f}   rank {np.mean(rks):.1f}   "
          f"[{time.time()-t0:.0f}s]")

    # A sidecar keyed by length, merged rather than overwritten: one tag can be scored at
    # several lengths and they are different measurements, not revisions of one.
    sp = os.path.join(RESULTS, f"scores_{a.tag}.json")
    rec = {}
    if os.path.exists(sp):
        try:
            rec = json.load(open(sp))
        except (ValueError, OSError):
            rec = {}
    # keyed by length AND draw count: a one-draw check and a six-draw measurement are two
    # estimates of the same quantity, not revisions of one, and keying by length alone lets
    # the cheaper one silently replace the better.
    # a control is a different drive, not a revision of the solve's own score
    key = f"{int(round(a.seconds))}s_{a.draws}d"
    if a.drive != "fitted":
        key += f"_{a.drive}{a.seed}"
    if a.jitter_s > 0:
        key += f"_jitter{a.jitter_s:g}s{a.seed}"
    rec[key] = dict(
        sim=float(np.mean(sims)), sim_sd=float(np.std(sims)),
        sims=[float(s) for s in sims],
        gap=float(np.mean(gaps)), field_rank=float(np.mean(rks)),
        frames=int(frames), draws=int(a.draws), seconds=float(a.seconds),
        drive=a.drive, seed=int(a.seed), jitter_s=float(a.jitter_s),
        smooth_fwhm_mm=sm_fwhm,
        # which scoring pipeline produced this. Realised scores before bandpass.apply's
        # mirror padding (5be967f) are depressed by an edge artefact that grows with
        # optimisation, so they are not comparable with later ones; runs.py marks them.
        bandpass_pad="mirror", scored_at=time.strftime("%Y-%m-%d %H:%M"))
    json.dump(rec, open(sp, "w"), indent=2, sort_keys=True)
    print(f"  wrote {os.path.relpath(sp, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))}")


if __name__ == "__main__":
    main()
