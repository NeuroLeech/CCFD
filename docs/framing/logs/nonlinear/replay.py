"""Step 1 of the nonlinear look: no fitting. The convex 3x drives (no rotation; 30 s
rotation) replayed in the medium with both nonlinear terms on (nl_flux 1: depth update uses
H + h; nl_adv 1: momentum advection), at 1x, 5x and 15x the linear fits' drive amplitude.
Linear FC is amplitude-invariant, so any change with amplitude is the nonlinearity. Each:
2 draws of 577 s scored by score_realisation (the repo's scorer, through torch_fit's own
nonlinear integrator), effective rank, and the shallowest point (H + h)/H the field reaches
in its frames. At 15x, draw 0's frames are kept for a video."""
import os, sys, time
ROOT = "/home/user/CCFD"
sys.path[:0] = [os.path.join(ROOT, d) for d in ("core", "fit", "targets", "analysis")]
os.chdir(ROOT)
import numpy as np, torch
import torch_fit, xspec, fc_score
from mesh_cache import load_cortex
from paths import RESULTS

c = load_cortex("fsaverage5", verbose=False)
REFS = ["nobp_sm6_s3_box", "nobp_sm6_s3_rot30_box"]
AMPS = [(1, 2e-4), (5, 1e-3), (15, 3e-3)]
for tag in REFS:
    ref = torch_fit.load_reference(tag)
    t = fc_score.FCTarget(c, fc_path=ref["fc_path"], centre="double", verbose=False)
    for mult, amp in AMPS:
        fit = torch_fit.Fit(ref, c, t, 577.0, device="cpu", amp=amp, nl_flux=1.0, nl_adv=1.0)
        G = torch_fit.warm_start(ref["S"], fit.dtype, fit.cdev)
        H = np.asarray(fit.Hf, float)
        sims, rks, depth, keep = [], [], [], None
        t0 = time.time()
        for sd in (0, 1):
            gen = torch.Generator(device="cpu").manual_seed(10_000 + sd)
            with torch.no_grad():
                A = fit.drive(G, fit.eta(gen))
            Af = A.detach().cpu().numpy()[::fit.save] * fit.save
            base, cap = fit.run_fn(), {}

            def run_fn(d, nsteps, save, base=base, cap=cap):
                F, dt = base(d, nsteps, save)
                cap["F"] = F
                return F, dt
            r = xspec.score_realisation(
                c, t, fit.p, Af, save=fit.save, amp=amp, profiles=ref["P"],
                kernel=fit.kern, band=ref["band"], frame_s=ref["frame_s"],
                segment=None, diagnostics=True, run_fn=run_fn, smooth=fit.Kscore)
            F = cap["F"]
            ok = bool(np.isfinite(F).all())
            depth.append(float(np.min(1.0 + F[t.burn:] / H[None, :])) if ok else np.nan)
            sims.append(r["sim"] if ok else np.nan); rks.append(r["rank"] if ok else np.nan)
            if sd == 0 and mult == 15 and ok:
                vt = f"{tag}_nl15"
                np.save(os.path.join(RESULTS, f"frames_{vt}.npy"), np.asarray(r["frames"], np.float32))
                np.save(os.path.join(RESULTS, f"frames_{vt}_raw.npy"), np.asarray(F, np.float32))
                np.save(os.path.join(RESULTS, f"drive_{vt}.npy"), np.asarray(r["drive"].Aser, np.float32))
        print(f"{tag:24s} x{mult:<2d} (amp {amp:g}): sim {np.nanmean(sims):+.4f} "
              f"({', '.join(f'{v:+.4f}' for v in sims)}), rank {np.nanmean(rks):.1f}, "
              f"shallowest (H+h)/H {np.nanmin(depth):.3f}  [{time.time()-t0:.0f}s]", flush=True)
print("REPLAY DONE", flush=True)
