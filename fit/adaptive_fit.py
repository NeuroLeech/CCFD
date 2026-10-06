"""Refine a drive by descending a gradient through the ADAPTIVE medium, read as local energy.

The nonlinear stage after fit/envelope_fit.py. The medium is a fast one (10x speed, decay set
from its measured front speed) with adaptive damping - one slow state per vertex, a running
average of local h^2 over tau_a, scaling damping by 1 + k a / a_ref (torch_swe.run_adaptive,
fit/adaptive_probe.py) - and BOLD is read from local ENERGY: frame-averaged h^2, then the BOLD
kernel and the passband. There is no closed form, so the drive's factor G is optimised by Adam
through torch_swe.run(adapt=..., observe="energy", decimate="mean"), whose gradient was checked
against central differences to 3e-8 with all three on.

WARM START, and the one mismatch in it. G starts from the envelope solve's S. That closed form
(fit/envelope.py) squares the FRAME-level field of a linear medium; the observable here squares
at every STEP and averages over the frame, which keeps energy above the frame rate that the
closed form never sees, and adaptation is absent from it entirely. So iteration 0 is a start,
not the envelope solve's own score - both are printed.

a_ref is calibrated once, from the warm start with adaptation off: the mean frame energy over
the solve vertices after the burn-in. The burn-in is extended to three tau_a so the slow state
has settled before anything is scored.

The surrogate descended is torch_fit's - Pearson of the double-centred model FC against the
normal-scored target on the solve vertices. The authoritative number is measured separately,
full sheet: run_adaptive with fresh drive seeds, then units.smooth_frames and bandpass.apply,
scored by the repo's Spearman scorer on all 9,310 vertices.

  python fit/adaptive_fit.py --ref env_f10m_d0.87 --k 1 --tau 10 --seconds 300 --iters 30
"""
import _path  # noqa: F401
import os, time, argparse
import numpy as np
import torch

from mesh_cache import load_cortex
from paths import RESULTS
import fc_score, xspec, units, bandpass, provenance
import torch_swe
import torch_fit as tf


def load_envfit(tag):
    z = np.load(os.path.join(RESULTS, f"envfit_{tag}.npz"), allow_pickle=True)
    return dict(S=z["S"], idx=z["idx"].astype(int), x=z["x"], save=int(z["save"]),
                P=z["profiles"], sub=z["sub"], band=tuple(z["band"]),
                frame_s=float(z["frame_s"]), ref_frames=int(z["pad"]), segment=0,
                map_clip=str(z["map_clip"]) if "map_clip" in z.files else "none",
                env_sim=float(z["sim"]) if "sim" in z.files else np.nan)


class AdaptiveFit(tf.Fit):
    def __init__(self, ref, cortex, target, seconds, k, tau_s, device="mps"):
        super().__init__(ref, cortex, target, seconds, device=device)
        self.k = float(k)
        self.tau_steps = float(tau_s) / (ref["frame_s"] / self.save)
        self.burn = max(self.burn, int(np.ceil(3 * tau_s / ref["frame_s"])))
        self.a_ref = 1.0

    def adapt(self, k=None):
        k = self.k if k is None else k
        return None if k == 0.0 else (k, self.tau_steps, self.a_ref)

    def frames(self, G, eta, k=None):
        A = self.drive(G, eta)
        H = self.apply_medium()
        F, _, _ = torch_swe.run(self.sw, A, self.Pt, self.dt, self.g, H, save=self.save,
                                chunk=self.chunk, keep=self.cols, adapt=self.adapt(k),
                                observe="energy", decimate="mean")
        return F

    def calibrate(self, G):
        with torch.no_grad():
            F = self.frames(G, self.eta(torch.Generator().manual_seed(0)), k=0.0)
        self.a_ref = float(F[self.burn:].mean())
        return self.a_ref

    def score(self, G, eta, parts=False):
        Z = self.observable(self.frames(G, eta))
        M = (Z @ Z.T) / Z.shape[1]
        M = M - M.mean(0, keepdim=True) - M.mean(1, keepdim=True) + M.mean()
        e = M[self.iu]
        e = e - e.mean()
        z0 = (e / e.norm().clamp_min(1e-20)) @ self.y
        return (z0, z0, None) if parts else z0

    def realised(self, G, seeds=(0, 1), seconds=None, k=None):
        """Full sheet, fresh drive seeds, the repo's scorer: the authoritative number."""
        from fc_score import _rank_z
        k = self.k if k is None else k
        out = []
        for sd in seeds:
            gen = torch.Generator(device=self.cdev).manual_seed(10_000 + sd)
            with torch.no_grad():
                A = self.drive(G, self.eta(gen)).detach().cpu().numpy()
            o, _ = torch_swe.run_adaptive(self.sw, A, self.ref["P"], self.dt, self.g,
                                          self.Hf, self.save, k=k, tau_steps=self.tau_steps,
                                          a_ref=self.a_ref)
            Y = bandpass.apply(units.smooth_frames(o["energy"], self.kern),
                               self.ref["frame_s"], *self.ref["band"])[self.burn:]
            cols = np.asarray(self.t.cols)
            Z, flat = _rank_z(np.ascontiguousarray(np.asarray(Y[:, cols], np.float64).T),
                              rank=True)
            out.append(float(self.t._prep(self.t.model_edges(Z=Z, flat=flat)[0]) @ self.t.y))
        return float(np.mean(out)), float(np.std(out))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ref", required=True, help="results/envfit_<ref>.npz")
    ap.add_argument("--k", type=float, default=1.0)
    ap.add_argument("--tau", type=float, default=10.0, help="tau_a, seconds")
    ap.add_argument("--seconds", type=float, default=300.0, help="realisation per gradient")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--lr", type=float, default=0.02, help="Adam lr as a fraction of RMS(G)")
    ap.add_argument("--eval-every", type=int, default=5, dest="eval_every")
    ap.add_argument("--eval-draws", type=int, default=2, dest="eval_draws")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    torch.manual_seed(0)
    c = load_cortex("fsaverage5", verbose=False)
    t = fc_score.default_target(c, verbose=False)
    ref = load_envfit(a.ref)
    fit = AdaptiveFit(ref, c, t, a.seconds, a.k, a.tau, device=a.device)
    G = tf.warm_start(ref["S"], fit.dtype, fit.cdev)
    aref = fit.calibrate(G)
    print(f"  {a.ref}: {fit.K} channels, {len(ref['sub'])} solve vertices, save {fit.save}, "
          f"{fit.nsteps} steps per {a.seconds:.0f} s gradient, burn {fit.burn} frames; "
          f"adaptation k {a.k:g}, tau_a {a.tau:g} s, a_ref {aref:.3g}", flush=True)
    print(f"  envelope solve's own realised score (linear medium, frame-level square): "
          f"{ref['env_sim']:+.4f}", flush=True)
    m_lin, s_lin = fit.realised(G, seeds=tuple(range(a.eval_draws)), k=0.0)
    m0, s0 = fit.realised(G, seeds=tuple(range(a.eval_draws)))
    print(f"  iteration 0, step-level energy: adaptation off {m_lin:+.4f} +- {s_lin:.4f}, "
          f"on {m0:+.4f} +- {s0:.4f}", flush=True)

    rms = float(G.detach().abs().pow(2).mean().sqrt())
    opt = torch.optim.Adam([{"params": [G], "lr": a.lr * rms}])
    hist, best, nskip = [], (m0, G.detach().cpu().numpy().copy(), 0), 0
    for it in range(1, a.iters + 1):
        t0 = time.time()
        opt.zero_grad()
        obj = fit.score(G, fit.eta())
        (-obj).backward()
        if (not torch.isfinite(obj)) or G.grad is None or not torch.isfinite(G.grad).all():
            nskip += 1; opt.zero_grad()
            print(f"  {it:4d}  NON-FINITE, update skipped ({nskip} so far)", flush=True)
            continue
        opt.step()
        hist.append(obj.detach().item())
        line = f"  {it:4d}  surrogate {hist[-1]:+.4f}  [{time.time()-t0:.0f}s]"
        if it % a.eval_every == 0 or it == a.iters:
            m, s = fit.realised(G, seeds=tuple(range(a.eval_draws)))
            line += f"   realised {m:+.4f} +- {s:.4f}"
            if np.isfinite(m) and m > best[0]:
                best = (m, G.detach().cpu().numpy().copy(), it); line += "  *"
        print(line, flush=True)

    print(f"\n  best realised {best[0]:+.4f} at iteration {best[2]}")
    if a.tag:
        Gb = best[1]
        np.savez(os.path.join(RESULTS, f"adaptfit_{a.tag}.npz"), G=Gb,
                 S=np.einsum("fab,fcb->fac", Gb, Gb.conj()), hist=np.asarray(hist),
                 best_sim=best[0], best_iter=best[2], ref=a.ref, k=a.k, tau=a.tau,
                 a_ref=aref, seconds=a.seconds, lin0=m_lin, ad0=m0, **provenance.stamp(a))
        print(f"  wrote results/adaptfit_{a.tag}.npz")


if __name__ == "__main__":
    main()
