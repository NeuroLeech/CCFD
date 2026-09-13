"""Solve for the input by descending a gradient through the simulation, not through H.

`xspec.solve` works because the medium is linear: C = sum_f H S H^H, so the drive enters
only as one Hermitian PSD matrix per frequency and matching FC is convex. Under a
nonlinearity there is no H and no factorisation, and the only thing left is to generate a
drive, simulate it, measure the covariance of what comes out, and differentiate that.

The parameterisation is deliberately the SAME OBJECT the convex solve produces. `realise`
builds a drive from S by taking eigh per bin, forming L = U sqrt(ev), drawing Z = L eta and
irfft-ing; everything from L onward is linear and differentiable, so the free variable here
is G in place of L, with S = G G^H. That makes the warm start literally the convex
solution's factor, and makes the solved S directly comparable, element for element.

Two differences from `realise` and `score_realisation`, both deliberate, both bounded:

  G is interpolated onto the realisation grid, where `realise` interpolates S and
  factorises afterwards. Interpolating S needs an eigh per bin per iteration, and eigh's
  backward pass is ill-conditioned wherever eigenvalues are close. Both give a valid PSD
  spectrum at every bin; they differ only in how the coarse grid is filled in, and the
  size of that difference is measured directly by scoring a warm start at iteration 0
  against the convex solve's own realised score.

  The BOLD kernel and the passband are applied as one multiply on the rfft grid, which is
  what `transfer` does to H, rather than by convolve1d and sosfiltfilt, which is what
  `score_realisation` does to frames. Those agree away from the ends, and the burn-in
  discards the leading end.

What is optimised is the surrogate the convex solve already optimises - Pearson against
the normal-scored target - not the Spearman score, which is not differentiable. The
Spearman score is then measured on the realised frames by `score_realisation` itself, the
repo's own scorer, unchanged. So a gap between this and the convex solve is not a gap
between two objectives.

  python fit/torch_fit.py --ref torchref --seconds 300 --init warm --iters 200
"""
import _path  # noqa: F401  - puts the sibling code folders on sys.path
import os, argparse, time
import numpy as np
import torch

from mesh_cache import load_cortex
from paths import RESULTS
import fc_score, xspec, bo_step, units, timescale, bandpass
import torch_swe


def load_reference(tag):
    z = np.load(os.path.join(RESULTS, f"xspec_{tag}.npz"), allow_pickle=True)
    return dict(S=z["S"], idx=z["idx"].astype(int), x=z["x"], save=int(z["save"]),
                P=z["profiles"], sub=z["sub"], band=tuple(z["band"]),
                frame_s=float(z["frame_s"]), ref_frames=int(z["ref_frames"]),
                segment=int(z["segment"]) if z["segment"].size else 0)


def target_edges(target, sub):
    """The normal-scored target on the solve vertices, as a unit edge vector.

    Identical to what `best_fit` hands `xspec.solve`: double-centre the raw FC on the
    subset, normal-score the edges so Pearson against it is a close surrogate for Spearman
    against the raw one, then centre and normalise so the objective is one dot product."""
    raw = np.asarray(target.target_fc()[np.ix_(sub, sub)], np.float64)
    raw = raw - raw.mean(0, keepdims=True) - raw.mean(1, keepdims=True) + raw.mean()
    Tg = xspec.normal_scores(raw)
    Tg[np.eye(len(sub), dtype=bool)] = 0.0
    iu = np.triu_indices(len(sub), 1)
    y = Tg[iu]
    y = y - y.mean()
    return y / np.linalg.norm(y), iu


def interp_weights(idx, ref_frames, nframes):
    """Linear interpolation from the solved frequency grid onto the realisation's.

    In FREQUENCY, not in bin index, for the reason `realise` gives: bin b of an N-sample
    series is the frequency b/N, so interpolating by index would move the whole spectrum
    whenever the run length changed. Returned as (lo, hi, w) so the interpolation is a
    gather and a lerp, which is cheap to differentiate."""
    nb = nframes // 2 + 1
    f_src = np.asarray(idx, float) / float(ref_frames)
    f_dst = np.arange(nb) / float(nframes)
    pos = np.searchsorted(f_src, f_dst).clip(1, len(f_src) - 1)
    lo, hi = pos - 1, pos
    span = np.maximum(f_src[hi] - f_src[lo], 1e-30)
    w = np.clip((f_dst - f_src[lo]) / span, 0.0, 1.0)
    inside = (f_dst >= f_src[0]) & (f_dst <= f_src[-1])       # no extrapolation, as realise
    return lo, hi, np.where(inside, w, 0.0), inside


class Fit:
    def __init__(self, ref, cortex, target, seconds, device="mps", dtype=torch.float32,
                 amp=2e-4, nl_flux=0.0, chunk=0):
        import fluid as fl
        self.ref, self.c, self.t = ref, cortex, target
        # The SIMULATION runs on `dev`; the complex arithmetic - building the drive from
        # G, and the rfft filters on the observable - runs on CPU, because MPS has no
        # backward for complex ops. Both boundaries are real-valued and small: A is
        # (nsteps, K) and F is (nframes, 1000), against a 28,000-edge state per step.
        self.dev, self.cdev, self.dtype = device, "cpu", dtype
        self.amp, self.save = amp, ref["save"]
        self.p, _, _ = bo_step.unpack(ref["x"], cortex)
        s, dt, g, Hf = fl.build(cortex, self.p)
        self.sw = torch_swe.TorchSWE(s, dtype=dtype, device=device, nl_flux=nl_flux)
        self.np_solver, self.dt, self.g, self.Hf = s, dt, g, Hf
        self.nframes = timescale.frames_for(seconds, ref["frame_s"])
        self.nsteps = self.nframes * self.save
        self.chunk = chunk or max(1, int(np.sqrt(self.nsteps)) // self.save * self.save)

        self.K = ref["P"].shape[0]
        self.nb = self.nframes // 2 + 1
        lo, hi, w, inside = interp_weights(ref["idx"], ref["ref_frames"], self.nframes)
        self.lo = torch.as_tensor(lo, dtype=torch.long, device="cpu")
        self.hi = torch.as_tensor(hi, dtype=torch.long, device="cpu")
        self.w = torch.as_tensor(w, dtype=dtype, device="cpu")[:, None, None]
        self.inside = torch.as_tensor(inside.astype(np.float32), dtype=dtype,
                                      device="cpu")[:, None]

        # the observable, as one rfft-grid multiplier: the BOLD kernel's DFT times the
        # passband's |H|^2, exactly the pair `transfer` multiplies into H
        kern = units.smoothing_kernel(timescale.bold_fwhm_frames(ref["frame_s"]),
                                      verbose=False)
        self.kern = kern
        kr = units.kernel_response(kern, self.nb, self.nframes)
        br = bandpass.response(np.arange(self.nb) / (self.nframes * ref["frame_s"]),
                               ref["frame_s"], ref["band"][0], ref["band"][1])
        self.filt = torch.as_tensor((kr * br).real.copy(), dtype=dtype,
                                    device="cpu")[:, None]

        self.cols = torch.as_tensor(np.asarray(target.cols)[ref["sub"]], dtype=torch.long,
                                    device=device)
        self.Pt = torch.as_tensor(np.asarray(ref["P"]), dtype=dtype, device=device)
        y, iu = target_edges(target, ref["sub"])
        self.y = torch.as_tensor(y, dtype=dtype, device="cpu")
        self.iu = (torch.as_tensor(iu[0], dtype=torch.long, device="cpu"),
                   torch.as_tensor(iu[1], dtype=torch.long, device="cpu"))
        self.burn = target.burn

    # ---------------- the differentiable path ----------------
    def drive(self, G, eta):
        """G (nfreq, K, K) complex -> A (nsteps, K), the per-step amplitudes."""
        Glo = G[self.lo]
        Gi = Glo + (G[self.hi] - Glo) * self.w
        Z = torch.einsum("fab,fb->fa", Gi, eta) * self.inside
        Z = torch.cat([torch.zeros_like(Z[:1]), Z[1:]], 0)        # no DC, as realise
        A = torch.fft.irfft(Z, n=self.nframes, dim=0)
        A = A / A.std().clamp_min(1e-30)
        A = A.repeat_interleave(self.save, dim=0)[:self.nsteps] / self.save
        A = A * (self.amp / A.pow(2).mean().sqrt().clamp_min(1e-30))
        return A.to(self.dev)

    def observable(self, F):
        """(nframes, nsub) field -> z-scored rows, filtered and burned."""
        F = F.to(self.cdev)
        X = torch.fft.irfft(torch.fft.rfft(F, dim=0) * self.filt, n=self.nframes, dim=0)
        X = X[self.burn:]
        X = X - X.mean(0, keepdim=True)
        return (X / X.std(0, keepdim=True).clamp_min(1e-20)).T      # (nsub, T)

    def score(self, G, eta):
        A = self.drive(G, eta)
        F, _, _ = torch_swe.run(self.sw, A, self.Pt, self.dt, self.g, self.Hf,
                                save=self.save, chunk=self.chunk, keep=self.cols)
        Z = self.observable(F)
        M = (Z @ Z.T) / Z.shape[1]
        M = M - M.mean(0, keepdim=True) - M.mean(1, keepdim=True) + M.mean()
        e = M[self.iu]
        e = e - e.mean()
        return (e / e.norm().clamp_min(1e-20)) @ self.y

    def eta(self, gen=None):
        r = torch.randn(self.nb, self.K, 2, dtype=self.dtype, device=self.cdev,
                        generator=gen) / np.sqrt(2.0)
        return torch.view_as_complex(r.contiguous())

    # ---------------- the authoritative score, via the repo's own path ----------------
    def realised(self, G, seeds=(0, 1)):
        """Simulate in numpy and score with `score_realisation`. No torch in the loop."""
        out = []
        for sd in seeds:
            gen = torch.Generator(device=self.cdev).manual_seed(10_000 + sd)
            with torch.no_grad():
                A = self.drive(G, self.eta(gen))
            Af = A.detach().cpu().numpy()[::self.save] * self.save   # back to per-frame
            r = xspec.score_realisation(
                self.c, self.t, self.p, Af, save=self.save, amp=self.amp,
                profiles=self.ref["P"], kernel=self.kern, band=self.ref["band"],
                frame_s=self.ref["frame_s"], segment=self.ref["segment"] or None,
                diagnostics=False)
            out.append(r["sim"])
        return float(np.mean(out)), float(np.std(out))


def warm_start(S, dtype, device):
    """G = U sqrt(ev) per bin: the convex solution's own factor, so S = G G^H exactly."""
    G = np.zeros_like(S)
    for f in range(len(S)):
        ev, U = np.linalg.eigh(0.5 * (S[f] + S[f].conj().T))
        G[f] = U * np.sqrt(np.clip(ev, 0, None))
    cdt = torch.complex64 if dtype == torch.float32 else torch.complex128
    return torch.tensor(G, dtype=cdt, device=device, requires_grad=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ref", default="torchref", help="results/xspec_<ref>.npz")
    ap.add_argument("--seconds", type=float, default=300.0)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--init", default="warm", choices=("warm", "cold"))
    ap.add_argument("--lr", type=float, default=0.02,
                    help="Adam step as a FRACTION of the initial RMS entry of G. Adam's "
                         "step is scale-free while G's entries are ~1/sqrt(nfreq K^2), so "
                         "an absolute lr of 0.02 is ten times a typical entry and wipes "
                         "out a warm start in three steps")
    ap.add_argument("--amp", type=float, default=2e-4)
    ap.add_argument("--nl-flux", type=float, default=0.0, dest="nl_flux")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--eval-every", type=int, default=25, dest="eval_every")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None, help="results/torchfit_<tag>.npz")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    ref = load_reference(a.ref)
    c = load_cortex("fsaverage5", verbose=False)
    t = fc_score.default_target(c, verbose=False)
    fit = Fit(ref, c, t, a.seconds, device=a.device, amp=a.amp, nl_flux=a.nl_flux)
    print(f"  ref {a.ref}: {fit.K} pieces, {len(ref['idx'])} solved frequencies, "
          f"{len(ref['sub'])} vertices")
    print(f"  realise {fit.nframes} frames ({a.seconds:.0f}s) = {fit.nsteps} steps, "
          f"chunk {fit.chunk}, amp {a.amp:g}, nl_flux {a.nl_flux:g}, device {a.device}")

    if a.init == "warm":
        G = warm_start(ref["S"], fit.dtype, fit.cdev)
    else:
        g0 = warm_start(ref["S"], fit.dtype, fit.cdev)
        sc = float(g0.detach().abs().pow(2).sum().sqrt()) / np.sqrt(g0.numel())
        G = (sc * torch.randn_like(g0.detach())).clone().requires_grad_(True)
    rms = float(G.detach().abs().pow(2).mean().sqrt())
    lr = a.lr * rms
    print(f"  init {a.init}: |G|_F {float(G.detach().abs().pow(2).sum().sqrt()):.4g}, "
          f"RMS entry {rms:.3g} -> Adam lr {lr:.3g}")

    m0, s0 = fit.realised(G)
    print(f"  iteration 0 realised: sim {m0:+.4f} +- {s0:.4f}", flush=True)
    seed0 = (m0, G.detach().cpu().numpy().copy(), 0)
    if a.init == "warm":
        # The one thing the G-vs-S interpolation difference could cost. Same S, same
        # length, same seeds, realised the repo's way instead of this module's.
        cv = []
        for sd in (0, 1):
            A = xspec.realise(ref["S"], ref["idx"], fit.nframes,
                              ref_frames=ref["ref_frames"], seed=10_000 + sd)
            cv.append(xspec.score_realisation(
                c, t, fit.p, A, save=fit.save, amp=a.amp, profiles=ref["P"],
                kernel=fit.kern, band=ref["band"], frame_s=ref["frame_s"],
                segment=ref["segment"] or None, diagnostics=False)["sim"])
        print(f"  convex S realised the repo's way, same length: "
              f"sim {np.mean(cv):+.4f} +- {np.std(cv):.4f}   "
              f"(interpolation gap {m0-np.mean(cv):+.4f})", flush=True)

    opt = torch.optim.Adam([G], lr=lr)
    hist, best = [], seed0        # iteration 0 competes, or the first eval always wins
    for it in range(1, a.iters + 1):
        t0 = time.time()
        opt.zero_grad()
        obj = fit.score(G, fit.eta())
        (-obj).backward()
        opt.step()
        hist.append(obj.detach().item())
        line = f"  {it:4d}  surrogate {hist[-1]:+.4f}  [{time.time()-t0:.1f}s]"
        if it % a.eval_every == 0 or it == a.iters:
            m, s = fit.realised(G)
            line += f"   realised sim {m:+.4f} +- {s:.4f}"
            if m > best[0]:
                best = (m, G.detach().cpu().numpy().copy(), it)
                line += "  *"
        print(line, flush=True)

    Gb = best[1] if best[1] is not None else G.detach().cpu().numpy()
    Sg = np.einsum("fab,fcb->fac", Gb, Gb.conj())
    Sr = ref["S"]
    num = np.vdot(Sg.ravel(), Sr.ravel()).real
    den = np.linalg.norm(Sg) * np.linalg.norm(Sr)
    print(f"\n  best realised sim {best[0]:+.4f} at iteration {best[2]}")
    print(f"  solved S vs convex S: corr {num/max(den,1e-30):+.4f}")
    if a.tag:
        np.savez(os.path.join(RESULTS, f"torchfit_{a.tag}.npz"), G=Gb, S=Sg,
                 hist=np.asarray(hist), best_sim=best[0], best_iter=best[2],
                 ref=a.ref, seconds=a.seconds, init=a.init, nl_flux=a.nl_flux, amp=a.amp)
        print(f"  wrote results/torchfit_{a.tag}.npz")


if __name__ == "__main__":
    main()
