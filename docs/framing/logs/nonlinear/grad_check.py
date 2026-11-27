"""Is the gradient through a long nonlinear simulation still a usable direction?
At the starting drive (10 s rotation, 35x, timing parameters at zero), on the same 2 fixed
draws the fit used, for recordings of L seconds: the objective, its gradient g in the 200
timing parameters, and the objective after a step +-h along g/|g|. A usable gradient predicts
the change: (obj(+h) - obj(-h)) / 2h should match |g|, and obj(+h) should exceed obj(-h)."""
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
for L in [float(v) for v in sys.argv[1].split(",")]:
    fit = torch_fit.Fit(ref, c, t, L, device="cpu", amp=7e-3, nl_flux=1.0, nl_adv=1.0)
    G = torch_fit.warm_start(ref["S"], fit.dtype, fit.cdev).detach()
    etas = [fit.eta(torch.Generator(device="cpu").manual_seed(20000 + i)) for i in range(2)]
    tau = torch.zeros(fit.K, requires_grad=True); lg = torch.zeros(fit.K, requires_grad=True)
    fit.tau, fit.lg = tau, lg
    t0 = time.time()
    obj = 0.0
    for e in etas:
        o = fit.score(G, e); (o / 2).backward(); obj += o.item() / 2
    g = torch.cat([tau.grad, lg.grad]).numpy()
    gn = np.linalg.norm(g); d = g / gn
    for h in (0.003, 0.01):
        vals = []
        for sgn in (+1, -1):
            with torch.no_grad():
                fit.tau = torch.as_tensor(sgn * h * d[:fit.K], dtype=torch.float32)
                fit.lg = torch.as_tensor(sgn * h * d[fit.K:], dtype=torch.float32)
                vals.append(np.mean([fit.score(G, e).item() for e in etas]))
        fd = (vals[0] - vals[1]) / (2 * h)
        print(f"{L:.0f} s: obj {obj:+.5f}, |grad| {gn:.4f}; step {h:g} along it: "
              f"obj(+) {vals[0]:+.5f}, obj(-) {vals[1]:+.5f}, measured slope {fd:+.4f} vs "
              f"predicted {gn:+.4f}  [{time.time()-t0:.0f}s]", flush=True)
