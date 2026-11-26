"""Checks before a nonlinear run: (1) torch and numpy integrate the same frames when the
nonlinearity is off, with the solve's mean-over-steps readout; (2) time per descent step."""
import os, sys, time
ROOT = "/home/user/CCFD"
sys.path[:0] = [os.path.join(ROOT, d) for d in ("core", "fit", "targets", "analysis")]
os.chdir(ROOT)
import numpy as np, torch
import torch_fit, torch_swe, xspec, fluid as fl
from mesh_cache import load_cortex
import fc_score

ref = torch_fit.load_reference(sys.argv[1] if len(sys.argv) > 1 else "nobp_sm6_s3_box")
c = load_cortex("fsaverage5", verbose=False)
t = fc_score.FCTarget(c, fc_path=ref["fc_path"], centre="double", verbose=False)
secs = float(sys.argv[2]) if len(sys.argv) > 2 else 60.0
fit = torch_fit.Fit(ref, c, t, secs, device="cpu", nl_flux=0.0, nl_adv=0.0)
print(f"ref: band {ref['band']}, decimate {ref['decimate']}, smooth {ref['smooth_fwhm']} mm, "
      f"save {fit.save}, {fit.nframes} frames = {fit.nsteps} steps, keep {len(fit.keep)} vertices")

# (1) same drive through both integrators
G = torch_fit.warm_start(ref["S"], fit.dtype, fit.cdev)
gen = torch.Generator().manual_seed(1)
with torch.no_grad():
    A = fit.drive(G, fit.eta(gen)).cpu().numpy()
d = xspec.ProfileDrive(c, ref["P"], A, fit.amp)
Fn, _ = fl.run(c, d, fit.p, fit.nsteps, fit.save)
with torch.no_grad():
    Ft, _, _ = torch_swe.run(fit.sw, np.asarray(d.Aser), np.asarray(d.P), fit.dt, fit.g,
                             fit.Hf, save=fit.save, chunk=0, decimate=fit.decimate)
Ft = Ft.numpy()
m = min(len(Fn), len(Ft))
r = np.corrcoef(Fn[50:m].ravel(), Ft[50:m].ravel())[0, 1]
print(f"(1) numpy vs torch frames, nonlinearity off: r = {r:.8f}, "
      f"max rel diff {np.abs(Fn[50:m]-Ft[50:m]).max()/np.abs(Fn[50:m]).max():.2e}")

# (2) one descent step: forward with gradient checkpoints, then backward
G.requires_grad_(True)
t0 = time.time()
obj = fit.score(G, fit.eta())
t1 = time.time()
(-obj).backward()
t2 = time.time()
print(f"(2) {secs:.0f} s realisation: surrogate {obj.item():+.4f}; forward {t1-t0:.1f}s, "
      f"backward {t2-t1:.1f}s; grad finite {bool(torch.isfinite(G.grad).all())}")
