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
"""
import _path  # noqa: F401  - puts the sibling code folders on sys.path
import os, time, argparse
import numpy as np

import provenance
from mesh_cache import load_cortex
from paths import RESULTS
import fc_score, xspec, bo_step, units, timescale


def load_solution(tag):
    """-> (npz, provenance args). The solve and everything needed to replay it."""
    path = os.path.join(RESULTS, f"xspec_{tag}.npz")
    if not os.path.exists(path):
        raise SystemExit(f"  no {path}")
    z = np.load(path, allow_pickle=True)
    return z, provenance.read(z).get("args", {})


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", required=True, help="results/xspec_<tag>.npz")
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
    a = ap.parse_args()

    z, pa = load_solution(a.tag)
    c = load_cortex("fsaverage5", verbose=False)
    fc_path = str(z["fc_path"]) if "fc_path" in z else ""
    centre = pa.get("centre", "double")
    t = (fc_score.FCTarget(c, fc_path=fc_path, centre=centre, verbose=True)
         if fc_path else fc_score.default_target(c, centre=centre, verbose=True))

    p, save, _ = bo_step.unpack(z["x"], c)
    p["map_clip"] = str(z["map_clip"]) if "map_clip" in z else "none"
    P = np.asarray(z["profiles"], np.float32)
    S, idx = z["S"], z["idx"].astype(int)
    ref_frames = int(z["ref_frames"])
    frame_s = float(z["frame_s"])
    band = tuple(z["band"]) if "band" in z and np.isfinite(z["band"]).all() else None
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
    print(f"  medium from the file: a {np.round(p.get('a', ()), 3)}, "
          f"b {np.round(p.get('b', ()), 3)}, Ld {p['Ld']:.4g}, "
          f"map_clip {p['map_clip']}")
    print(f"  realise {frames} frames ({a.seconds:.0f}s) at {frame_s:.5g}s, "
          f"{a.draws} draws, smoothing FWHM {float(smooth):g} frames"
          + ("" if band is None else f", band {band[0]:g}-{band[1]:g} Hz")
          + ("" if pa else "   [no provenance on this file]"), flush=True)

    t0 = time.time()
    sims, gaps, rks = [], [], []
    pool = res = None
    if a.workers > 1 and a.draws > 1:
        rest = [xspec.realise(S, idx, frames, ref_frames=ref_frames, seed=1000 + d)
                for d in range(1, a.draws)]
        pool, res = xspec.parallel_scores(c, t, p, rest, save, P, kern, a.workers,
                                          band=band, frame_s=frame_s if band else None,
                                          segment=seg)
        print(f"  draws 2-{a.draws} over {min(a.workers, len(rest))} workers", flush=True)
    for d in range(1 if pool is not None else a.draws):
        A = xspec.realise(S, idx, frames, ref_frames=ref_frames, seed=1000 + d)
        r = xspec.score_realisation(c, t, p, A, save=save, profiles=P, kernel=kern,
                                    band=band, frame_s=frame_s if band else None,
                                    segment=seg)
        sims.append(r["sim"]); gaps.append(r["gap"]); rks.append(r["rank"])
        if d == 0 and a.save_frames:
            sec = int(round(a.seconds))
            np.save(os.path.join(RESULTS, f"frames_{a.tag}_{sec}s.npy"), r["frames"])
            np.save(os.path.join(RESULTS, f"drive_{a.tag}_{sec}s.npy"), r["drive"].Aser)
            print(f"    wrote frames_{a.tag}_{sec}s.npy", flush=True)
    if pool is not None:
        for sim, gap, rk, _sa in res.get():
            sims.append(sim); gaps.append(gap); rks.append(rk)
        pool.close(); pool.join()

    print(f"\n  realised over {frames} frames ({a.seconds:.0f}s), {a.draws} draws: "
          f"sim {np.mean(sims):+.4f} +- {np.std(sims):.4f}   "
          f"gap {np.mean(gaps):.3f}   rank {np.mean(rks):.1f}   "
          f"[{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
