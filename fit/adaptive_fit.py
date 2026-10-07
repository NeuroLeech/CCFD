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

FIRST RUN, 2026-10-06 (k 1, tau_a 10 s, one draw per gradient, lr 0.02): realised 0.184 ->
0.188 over 30 iterations, the surrogate jumping +-0.03 per iteration. Two faults found after it,
both fixed here: the authoritative score was taken at the DESCENT's 300 s (now --eval-seconds,
577 s), and torch_fit's factor interpolation scrambled the out-of-band drive (now exact; see
drive). At equal length the drive here and realise agree: +0.260 +- 0.017 against
+0.243 +- 0.017 over 6 seeds each.

  python fit/adaptive_fit.py --ref env_f10m_d0.87 --k 1 --tau 10 --seconds 300 --iters 30

LEARNED ADAPTATION (--learn-adapt): k and tau_a are descended with the drive, as log k and
log tau_a with their own Adam lr (--lr-adapt, ~ the relative change per iteration). Their
gradients through torch_swe.run match central differences to 2e-9 - 3e-8 (float64, 240
steps, k 1 and 3, tau 20 and 5 steps), identical with checkpointing. tau_a is clamped at
--tau-max, which fixes the burn-in at 3 tau_max. A continuation starts from its parent's k
and tau_a.
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
    def __init__(self, ref, cortex, target, seconds, k, tau_s, device="mps", tau_max=None):
        super().__init__(ref, cortex, target, seconds, device=device)
        self.sps = ref["frame_s"] / self.save                     # seconds per step
        self.k, self.tau_s = float(k), float(tau_s)
        # k and tau_a as log-parameters, so they stay positive and Adam steps are relative.
        # Fixed unless learn_adapt() turns on their gradient. tau_a is clamped to tau_max,
        # because the burn-in (three tau_a, so the slow state has settled) is fixed when
        # the drive's length is.
        self.tau_max = float(tau_s if tau_max is None else tau_max)
        self.lk = torch.tensor(np.log(max(self.k, 1e-12)), dtype=self.dtype, device=device)
        self.lt = torch.tensor(np.log(self.tau_s), dtype=self.dtype, device=device)
        self.burn = max(self.burn, int(np.ceil(3 * self.tau_max / ref["frame_s"])))
        self.a_ref = 1.0

    def learn_adapt(self):
        self.lk.requires_grad_(True); self.lt.requires_grad_(True)
        return [self.lk, self.lt]

    def set_adapt(self, k, tau_s):
        with torch.no_grad():
            self.lk.fill_(float(np.log(max(k, 1e-12)))); self.lt.fill_(float(np.log(tau_s)))
        self.k, self.tau_s = float(k), float(tau_s)

    def k_tau(self):
        """Current (k, tau_a seconds) as floats."""
        lt = float(self.lt.detach().clamp(max=float(np.log(self.tau_max))))
        return (self.k if not self.lk.requires_grad else float(np.exp(float(self.lk.detach()))),
                float(np.exp(lt)))

    @property
    def tau_steps(self):
        return self.k_tau()[1] / self.sps

    # ---- the drive, with the cross-spectrum interpolated EXACTLY ----
    # torch_fit.Fit.drive interpolates the FACTOR G between solved bins. Each bin's factor
    # comes out of its own eigendecomposition with arbitrary phases, so a blend of two
    # neighbours partly cancels their coherent structure. For a linear observable that was
    # harmless - the solved bins sit densely in the passband, where almost every realisation
    # bin has a solved one beside it. The energy observable reads EVERY frequency, and above
    # the band the solved grid is sparse, so most of the drive there is interpolated across
    # wide gaps. Measured on the envelope warm start at 10x: step-level energy +0.209 with
    # the factor as-is, +0.225 with factors Procrustes-aligned across bins, against
    # +0.259/+0.269 from realise. A blend of two cross-spectra (1-t) S1 + t S2 is EXACTLY the
    # cross-spectrum of sqrt(1-t) G1 eta1 + sqrt(t) G2 eta2 with independent eta1, eta2 -
    # xspec._realise_factored's identity - so each realisation bin draws from both
    # neighbours with its own noise: no eigh in the graph, statistically what realise draws.
    def eta(self, gen=None):
        r = torch.randn(self.nb, 2, self.K, 2, dtype=self.dtype, device=self.cdev,
                        generator=gen) / np.sqrt(2.0)
        return torch.view_as_complex(r.contiguous())

    def drive(self, G, eta):
        wt = self.w[:, :, 0]                                      # (nb, 1), weight on hi
        Z = (torch.sqrt(1.0 - wt) * torch.einsum("fab,fb->fa", G[self.lo], eta[:, 0])
             + torch.sqrt(wt) * torch.einsum("fab,fb->fa", G[self.hi], eta[:, 1]))
        Z = Z * self.inside
        Z = torch.cat([torch.zeros_like(Z[:1]), Z[1:]], 0)        # no DC, as realise
        A = torch.fft.irfft(Z, n=self.nframes, dim=0)
        A = A / A.std().clamp_min(1e-30)
        A = A.repeat_interleave(self.save, dim=0)[:self.nsteps] / self.save
        A = A * (self.amp / A.pow(2).mean().sqrt().clamp_min(1e-30))
        return A.to(self.dev)

    def adapt(self, k=None):
        if k is not None:
            return None if k == 0.0 else (k, self.tau_steps, self.a_ref)
        if self.lk.requires_grad:
            lt = self.lt.clamp(max=float(np.log(self.tau_max)))
            return (torch.exp(self.lk), torch.exp(lt) / self.sps, self.a_ref)
        return None if self.k == 0.0 else (self.k, self.tau_steps, self.a_ref)

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
        k = self.k_tau()[0] if k is None else k
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
    ap.add_argument("--eval-seconds", type=float, default=577.0, dest="eval_seconds",
                    help="realisation length for the authoritative score - its own Fit, since a "
                         "drive's length is fixed at construction; scoring at the descent's "
                         "shorter length understates it against everything else at 577 s")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--lr", type=float, default=0.02, help="Adam lr as a fraction of RMS(G)")
    ap.add_argument("--grad-draws", type=int, default=2, dest="grad_draws",
                    help="drive draws averaged per gradient. One fresh 300 s draw gave a "
                         "surrogate that jumped +-0.03 between iterations, swamping the step")
    ap.add_argument("--eval-every", type=int, default=5, dest="eval_every")
    ap.add_argument("--eval-draws", type=int, default=2, dest="eval_draws")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--init-from", default="", dest="init_from",
                    help="continue from results/adaptfit_<tag>.npz's G (its best) and a_ref")
    ap.add_argument("--seed", type=int, default=0,
                    help="seed of the gradient draws; give a continuation a new one")
    ap.add_argument("--learn-adapt", action="store_true", dest="learn_adapt",
                    help="descend k and tau_a with the drive (as log k, log tau_a)")
    ap.add_argument("--lr-adapt", type=float, default=0.05, dest="lr_adapt",
                    help="Adam lr for log k and log tau_a: ~ the relative change per iteration")
    ap.add_argument("--tau-max", type=float, default=None, dest="tau_max",
                    help="upper clamp on tau_a, seconds; sets the burn-in (3 tau_max). "
                         "Default: --tau with --learn-adapt off, 3x --tau with it on")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    # the seed of the GRADIENT draws. The descent memorises its draws - G multiplies each
    # eta directly, so a step on one realisation shapes the drive to suit that realisation:
    # a continued run re-seeded at 0 replays the draws its parent trained on and scored
    # +0.48-0.55 on them, against +0.27 on fresh ones. A continuation needs a new --seed.
    torch.manual_seed(a.seed)
    c = load_cortex("fsaverage5", verbose=False)
    t = fc_score.default_target(c, verbose=False)
    ref = load_envfit(a.ref)
    prev = None
    if a.init_from:
        prev = np.load(os.path.join(RESULTS, f"adaptfit_{a.init_from}.npz"), allow_pickle=True)
        a.k, a.tau = float(prev["k"]), float(prev["tau"])      # the parent's (best) k, tau_a
    tau_max = a.tau_max or (3 * a.tau if a.learn_adapt else a.tau)
    fit = AdaptiveFit(ref, c, t, a.seconds, a.k, a.tau, device=a.device, tau_max=tau_max)
    G = tf.warm_start(ref["S"], fit.dtype, fit.cdev)
    aref = fit.calibrate(G)
    if a.init_from:
        # a_ref stays the one calibrated from the envelope warm start: it defines k's scale,
        # and recalibrating from the continued G would change the medium being continued
        G = torch.tensor(prev["G"], dtype=G.dtype, device=fit.cdev, requires_grad=True)
        aref = fit.a_ref = float(prev["a_ref"])
        if a.seed == 0:
            raise SystemExit("  --init-from with --seed 0 replays the parent run's gradient "
                             "draws, which the drive has memorised; pass a new --seed")
        print(f"  continuing {a.init_from}: its best realised {float(prev['best_sim']):+.4f} "
              f"at iteration {int(prev['best_iter'])}", flush=True)
    ev = AdaptiveFit(ref, c, t, a.eval_seconds, a.k, a.tau, device=a.device, tau_max=tau_max)
    ev.a_ref = aref
    print(f"  {a.ref}: {fit.K} channels, {len(ref['sub'])} solve vertices, save {fit.save}, "
          f"{fit.nsteps} steps per {a.seconds:.0f} s gradient, scored at {a.eval_seconds:.0f} s, "
          f"burn {fit.burn} frames; "
          f"adaptation k {a.k:g}, tau_a {a.tau:g} s, a_ref {aref:.3g}"
          + (f"; k and tau_a LEARNED (lr {a.lr_adapt:g} in log, tau_a <= {tau_max:g} s)"
             if a.learn_adapt else ""), flush=True)
    print(f"  envelope solve's own realised score (linear medium, frame-level square): "
          f"{ref['env_sim']:+.4f}", flush=True)
    m_lin, s_lin = ev.realised(G, seeds=tuple(range(a.eval_draws)), k=0.0)
    m0, s0 = ev.realised(G, seeds=tuple(range(a.eval_draws)))
    print(f"  iteration 0, step-level energy: adaptation off {m_lin:+.4f} +- {s_lin:.4f}, "
          f"on {m0:+.4f} +- {s0:.4f}", flush=True)

    rms = float(G.detach().abs().pow(2).mean().sqrt())
    groups = [{"params": [G], "lr": a.lr * rms}]
    if a.learn_adapt:
        groups.append({"params": fit.learn_adapt(), "lr": a.lr_adapt})
    opt = torch.optim.Adam(groups)
    hist, khist = [], []
    best, nskip = (m0, G.detach().cpu().numpy().copy(), 0, a.k, a.tau), 0
    for it in range(1, a.iters + 1):
        t0 = time.time()
        opt.zero_grad()
        obj = sum(fit.score(G, fit.eta()) for _ in range(a.grad_draws)) / a.grad_draws
        (-obj).backward()
        bad = [p for p in [G] + ([fit.lk, fit.lt] if a.learn_adapt else [])
               if p.grad is None or not torch.isfinite(p.grad).all()]
        if (not torch.isfinite(obj)) or bad:
            nskip += 1; opt.zero_grad()
            print(f"  {it:4d}  NON-FINITE, update skipped ({nskip} so far)", flush=True)
            continue
        opt.step()
        hist.append(obj.detach().item())
        kt = fit.k_tau(); khist.append(kt)
        line = f"  {it:4d}  surrogate {hist[-1]:+.4f}"
        if a.learn_adapt:
            line += f"  k {kt[0]:.3f} tau_a {kt[1]:.2f} s"
        line += f"  [{time.time()-t0:.0f}s]"
        if it % a.eval_every == 0 or it == a.iters:
            ev.set_adapt(*kt)
            m, s = ev.realised(G, seeds=tuple(range(a.eval_draws)))
            line += f"   realised {m:+.4f} +- {s:.4f}"
            if np.isfinite(m) and m > best[0]:
                best = (m, G.detach().cpu().numpy().copy(), it) + kt; line += "  *"
            if a.tag:                       # checkpoint: a long run is not lost to a stop
                np.savez(os.path.join(RESULTS, f"adaptfit_{a.tag}_ckpt.npz"),
                         G=G.detach().cpu().numpy(), best_G=best[1], best_sim=best[0],
                         best_iter=best[2], it=it, hist=np.asarray(hist), a_ref=aref,
                         k=best[3], tau=best[4], khist=np.asarray(khist), ref=a.ref)
        print(line, flush=True)

    print(f"\n  best realised {best[0]:+.4f} at iteration {best[2]} "
          f"(k {best[3]:.3f}, tau_a {best[4]:.2f} s)")
    if a.tag:
        Gb = best[1]
        np.savez(os.path.join(RESULTS, f"adaptfit_{a.tag}.npz"), G=Gb,
                 S=np.einsum("fab,fcb->fac", Gb, Gb.conj()), hist=np.asarray(hist),
                 best_sim=best[0], best_iter=best[2], ref=a.ref, k=best[3], tau=best[4],
                 khist=np.asarray(khist), k0=a.k, tau0=a.tau, learn_adapt=a.learn_adapt,
                 a_ref=aref, seconds=a.seconds, lin0=m_lin, ad0=m0, **provenance.stamp(a))
        print(f"  wrote results/adaptfit_{a.tag}.npz")


if __name__ == "__main__":
    main()
