"""Where does the linear input break? The convex 3x drive with 30 s rotation replayed in the
nonlinear medium (nl_flux 1, nl_adv 1) at rising amplitude, until the sheet runs dry.
Per amplitude, 2 draws of 577 s: the score (score_realisation, the solve's observable), the
effective rank, the shallowest (H + h)/H, and how far the field itself has moved from
linear - the correlation of the nonlinear field with the linear field for the same drive
(linear frames are amplitude-invariant up to scale, so one linear run per draw serves every
amplitude). Stops at the first amplitude whose field is not finite (a bed strike)."""
import os, sys, time
ROOT = "/home/user/CCFD"
sys.path[:0] = [os.path.join(ROOT, d) for d in ("core", "fit", "targets", "analysis")]
os.chdir(ROOT)
import numpy as np, torch
import torch_fit, torch_swe, xspec, fc_score
from mesh_cache import load_cortex
from paths import RESULTS

tag = sys.argv[1] if len(sys.argv) > 1 else "nobp_sm6_s3_rot30_box"
MULTS = [int(v) for v in (sys.argv[2] if len(sys.argv) > 2 else "30,50,80,120").split(",")]
c = load_cortex("fsaverage5", verbose=False)
ref = torch_fit.load_reference(tag)
t = fc_score.FCTarget(c, fc_path=ref["fc_path"], centre="double", verbose=False)
lin = torch_fit.Fit(ref, c, t, 577.0, device="cpu", amp=2e-4, nl_flux=0.0, nl_adv=0.0)
G = torch_fit.warm_start(ref["S"], lin.dtype, lin.cdev)
H = np.asarray(lin.Hf, float)
drives, linF = [], []
for sd in (0, 1):
    gen = torch.Generator(device="cpu").manual_seed(10_000 + sd)
    with torch.no_grad():
        A = lin.drive(G, lin.eta(gen)).cpu().numpy()
    Af = A[::lin.save] * lin.save
    drives.append(Af)
    d = xspec.ProfileDrive(c, ref["P"], np.repeat(Af, lin.save, axis=0)[:lin.nsteps] / lin.save, 2e-4)
    d.Aser = (d.Aser * (2e-4 / np.sqrt((d.Aser ** 2).mean()))).astype(np.float32)
    with torch.no_grad():
        F, _, _ = torch_swe.run(lin.sw, np.asarray(d.Aser), np.asarray(d.P), lin.dt, lin.g,
                                lin.Hf, save=lin.save, chunk=0, decimate=lin.decimate)
    linF.append(F.numpy())
print(f"{tag}: linear field for 2 draws ready", flush=True)
for mult in MULTS:
    amp = 2e-4 * mult
    fit = torch_fit.Fit(ref, c, t, 577.0, device="cpu", amp=amp, nl_flux=1.0, nl_adv=1.0)
    sims, rks, depth, rfield = [], [], [], []
    t0 = time.time()
    dry = False
    for sd in (0, 1):
        base, cap = fit.run_fn(), {}

        def run_fn(d, nsteps, save, base=base, cap=cap):
            F, dt = base(d, nsteps, save)
            cap["F"] = F
            return F, dt
        try:
            r = xspec.score_realisation(
                c, t, fit.p, drives[sd], save=fit.save, amp=amp, profiles=ref["P"],
                kernel=fit.kern, band=ref["band"], frame_s=ref["frame_s"], segment=None,
                diagnostics=True, run_fn=run_fn, smooth=fit.Kscore)
        except np.linalg.LinAlgError:
            r = None                      # the scorer choked on a non-finite field
        F = cap["F"]
        if r is None or not np.isfinite(F).all():
            dry = True
            break
        depth.append(float(np.min(1.0 + F[t.burn:] / H[None, :])))
        m = min(len(F), len(linF[sd]))
        rfield.append(float(np.corrcoef(F[t.burn:m].ravel(), linF[sd][t.burn:m].ravel())[0, 1]))
        sims.append(r["sim"]); rks.append(r["rank"])
        if sd == 0:
            vt = f"{tag}_nl{mult}"
            np.save(os.path.join(RESULTS, f"frames_{vt}.npy"), np.asarray(r["frames"], np.float32))
            np.save(os.path.join(RESULTS, f"frames_{vt}_raw.npy"), np.asarray(F, np.float32))
            np.save(os.path.join(RESULTS, f"drive_{vt}.npy"), np.asarray(r["drive"].Aser, np.float32))
    if dry:
        print(f"x{mult:<3d} (amp {amp:g}): the field is not finite - the sheet ran dry "
              f"(bed strike)  [{time.time()-t0:.0f}s]", flush=True)
        break
    print(f"x{mult:<3d} (amp {amp:g}): sim {np.mean(sims):+.4f} ({', '.join(f'{v:+.4f}' for v in sims)}), "
          f"rank {np.mean(rks):.1f}, shallowest (H+h)/H {min(depth):.3f}, field vs linear "
          f"r {np.mean(rfield):.4f}  [{time.time()-t0:.0f}s]", flush=True)
print("REPLAY2 DONE", flush=True)
