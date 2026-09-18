"""Realise a fitted torch solution into the frames and drive analysis/report.py reads.

report.py loads results/frames_<tag>v.npy and results/drive_<tag>v.npy. The only other
writer of that pair is best_fit.py, which realises its OWN convex S through the numpy
solver: it knows nothing of nl_flux, nl_adv or a learned grading, and its one mention of
the maps is p.get("a"). Frames from it would therefore show a different system than the
one report.py rebuilds for its medium, interference and energy-flux sections - the same
linear-frames-for-a-nonlinear-fit mismatch torch_fit.run_fn calls the one discrepancy
that would invalidate a run rather than just add noise.

This realises the saved solution through fit.run_fn(), the torch integrator carrying the
fitted medium, which is the path torch_fit.realised already scores through. The medium is
rebuilt by report.load_all, so the frames and the figures drawn from them come from one
and the same reconstruction.

The reported sim is one draw, not the six torch_fit averages over, so it is a check that
this realisation landed where the fit said - not a new measurement of the fit.

  python fit/render_fit.py --tag nlmap100b --ref asc_first_area_100
"""
import _path  # noqa: F401  - puts the sibling code folders on sys.path
import os, argparse
import numpy as np
import torch

from paths import RESULTS
import xspec
from report import load_all


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", required=True, help="results/torchfit_<tag>.npz")
    ap.add_argument("--ref", default="torchref", help="results/xspec_<ref>.npz")
    ap.add_argument("--vtag", default=None, help="output stem; default <tag>v")
    ap.add_argument("--seed", type=int, default=0,
                    help="which of torch_fit's own eval draws to render (it uses 0-5)")
    a = ap.parse_args()
    vtag = a.vtag or (a.tag + "v")

    z, ref, c, t, fit, rz = load_all(a.tag, a.ref)
    learned = "map_a" in z.files
    print(f"  {a.tag}: nl_flux {float(z['nl_flux']):g}, nl_adv {float(z['nl_adv']):g}, "
          f"medium {'learned' if learned else 'fixed'}, "
          f"best sim {float(z['best_sim']):+.4f} at iteration {int(z['best_iter'])}")
    with torch.no_grad():
        H, sig = fit.medium() if learned else (fit.Hf, None)
    H = np.asarray(H.cpu() if torch.is_tensor(H) else H)
    print(f"  realising {fit.nframes} frames ({float(z['seconds']):.0f}s) in H "
          f"{H.min():.3f}-{H.max():.3f}", flush=True)

    # The drive exactly as torch_fit builds it: same G, same generator stream, then back
    # to per-frame amplitudes, because score_realisation re-expands by `save` itself.
    cdt = torch.complex64 if fit.dtype == torch.float32 else torch.complex128
    G = torch.tensor(z["G"], dtype=cdt, device=fit.cdev)
    gen = torch.Generator(device=fit.cdev).manual_seed(10_000 + a.seed)
    with torch.no_grad():
        A = fit.drive(G, fit.eta(gen))
    Af = A.detach().cpu().numpy()[::fit.save] * fit.save

    r = xspec.score_realisation(
        c, t, fit.p, Af, save=fit.save, amp=fit.amp, profiles=ref["P"],
        kernel=fit.kern, band=ref["band"], frame_s=ref["frame_s"],
        segment=ref["segment"] or None, diagnostics=False, run_fn=fit.run_fn())

    fp = os.path.join(RESULTS, f"frames_{vtag}.npy")
    dp = os.path.join(RESULTS, f"drive_{vtag}.npy")
    np.save(fp, np.asarray(r["frames"], np.float32))
    np.save(dp, np.asarray(r["drive"].Aser, np.float32))
    print(f"  draw {a.seed}: sim {r['sim']:+.4f}   (the fit's six-draw mean was "
          f"{float(z['best_sim']):+.4f})")
    print(f"  wrote {fp}  {np.shape(r['frames'])}")
    print(f"  wrote {dp}  {np.shape(r['drive'].Aser)}")


if __name__ == "__main__":
    main()
