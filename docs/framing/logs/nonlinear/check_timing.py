import os, sys, time
ROOT = "/home/user/CCFD"
sys.path[:0] = [os.path.join(ROOT, d) for d in ("core", "fit", "targets", "analysis")]
os.chdir(ROOT)
import numpy as np, torch
import torch_fit, fc_score
from mesh_cache import load_cortex
ref = torch_fit.load_reference("nobp_sm6_s3_rot10_box")
c = load_cortex("fsaverage5", verbose=False)
t = fc_score.FCTarget(c, fc_path=ref["fc_path"], centre="double", verbose=False)
fit = torch_fit.Fit(ref, c, t, 30.0, device="cpu", amp=7e-3, nl_flux=1.0, nl_adv=1.0)
G = torch_fit.warm_start(ref["S"], fit.dtype, fit.cdev).detach()
gen = torch.Generator().manual_seed(3); eta = fit.eta(gen)
with torch.no_grad():
    A0 = fit.drive(G, eta).numpy()
    fit.tau = torch.zeros(fit.K); fit.lg = torch.zeros(fit.K)
    A1 = fit.drive(G, eta).numpy()
    print(f"(1) zero delays and gains: max |A1 - A0| / max|A0| = {np.abs(A1-A0).max()/np.abs(A0).max():.1e}")
    k, d = 7, 5 * ref["frame_s"]                       # input 7 delayed by 5 frames
    fit.tau = torch.zeros(fit.K); fit.tau[k] = d
    A2 = fit.drive(G, eta).numpy()
    a0, a2 = A0[::fit.save, k], A2[::fit.save, k]
    lags = np.arange(-10, 11)
    xc = [np.corrcoef(np.roll(a0, L), a2)[0, 1] for L in lags]
    print(f"(2) input {k} delayed {d:.3f} s = 5 frames: best match at a shift of {lags[int(np.argmax(xc))]:+d} "
          f"frames (r {max(xc):.4f}); other inputs unchanged: "
          f"{np.allclose(np.delete(A2,k,1), np.delete(A0,k,1), atol=1e-12, rtol=0)}")
fit.tau = torch.zeros(fit.K, requires_grad=True); fit.lg = torch.zeros(fit.K, requires_grad=True)
t0 = time.time()
obj = fit.score(G, fit.eta())
(-obj).backward()
gt, gg = fit.tau.grad.numpy(), fit.lg.grad.numpy()
print(f"(3) objective {obj.item():+.4f}; grad finite {np.isfinite(gt).all() and np.isfinite(gg).all()}, "
      f"nonzero for {np.mean(gt != 0):.0%} of delays and {np.mean(gg != 0):.0%} of gains; "
      f"|grad| delays {np.abs(gt).mean():.2e}, gains {np.abs(gg).mean():.2e}  [{time.time()-t0:.0f}s]")
