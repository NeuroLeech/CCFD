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
                 amp=2e-4, nl_flux=0.0, nl_adv=0.0, Ld=None, maps_scale=1.0,
                 lag_taus=(), wa=1.0, chunk=0):
        import fluid as fl
        self.ref, self.c, self.t = ref, cortex, target
        # The SIMULATION runs on `dev`; the complex arithmetic - building the drive from
        # G, and the rfft filters on the observable - runs on CPU, because MPS has no
        # backward for complex ops. Both boundaries are real-valued and small: A is
        # (nsteps, K) and F is (nframes, 1000), against a 28,000-edge state per step.
        self.dev, self.cdev, self.dtype = device, "cpu", dtype
        self.amp, self.save = amp, ref["save"]
        self.p, _, _ = bo_step.unpack(ref["x"], cortex)
        # bo_step.unpack already takes the speed/damping map coefficients from x[4:10];
        # they are the same a and b best_fit saves as map_a/map_b.
        if maps_scale != 1.0:
            # The depth field grades wave speed AND sets the bed. At full scale H spans
            # 0.119-3.914 and the thinnest point is 9.6x below the mean, so (H + h) turns
            # negative there long before it does anywhere typical - the sheet's amplitude
            # ceiling is set by its weakest patch. Flattening removes that, and also
            # doubles dt, because dt = CFL*d_min/c_max and c_max falls to 1.
            self.p = dict(self.p)
            self.p["a"] = np.asarray(self.p["a"], float) * maps_scale
            self.p["b"] = np.asarray(self.p["b"], float) * maps_scale
        if Ld is not None:
            # Ld sets the Coriolis parameter f = 1/Ld and nothing else in fl.build, so
            # overriding it here changes the rotation rate and leaves the rest of the
            # medium alone. self.p is what score_realisation integrates too, so the
            # evaluation sees the same rotation the descent does.
            self.p = dict(self.p); self.p["Ld"] = float(Ld)
        s, dt, g, Hf = fl.build(cortex, self.p)
        self.sw = torch_swe.TorchSWE(s, dtype=dtype, device=device, nl_flux=nl_flux,
                                     nl_adv=nl_adv, cortex=cortex)
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

        # ---- the lagged term. Zero-lag FC is a second moment and cannot see lead-lag at
        # all: Phi(tau) splits into a symmetric part, which tau=0 already constrains, and
        # an ANTISYMMETRIC part, which it is blind to. Measured on this medium the fits
        # carry 1.5-1.9x the data's antisymmetry at 1-5 TR, reproducible across
        # realisations (seed-seed corr 0.62-0.65), so it is a real and unconstrained
        # mismatch - and the convex solve shows it as strongly as the gradient ones.
        self.wa, self.lag_L = float(wa), []
        self.lagA = None
        if len(lag_taus):
            import lagged as _lg
            over = int(round(timescale.TR / ref["frame_s"]))
            self.lag_L = [int(k) * over for k in lag_taus]
            fs5 = np.asarray(cortex.old)[np.asarray(target.cols)[ref["sub"]]]
            E = _lg.empirical_rbc(list(lag_taus), fs5, gsr=False)
            An = np.stack([0.5 * (E[k] - E[k].T) for k in range(len(lag_taus))])
            An = An / max(np.linalg.norm(An), 1e-30)
            self.lagA = torch.as_tensor(An.ravel().copy(), dtype=dtype, device="cpu")
            print(f"  lagged term: taus {list(lag_taus)} TR = {self.lag_L} model frames, "
                  f"weight {self.wa:g}")

    # ---------------- the medium, when its grading is a free variable ----------------
    def init_medium(self, bound):
        """Make the speed/damping map coefficients differentiable, bounded either side.

        The same family fluid.map_fields builds - c(x) = c0 exp(a.M), sig(x) = sig0 exp(b.M)
        over the z-scored cortical maps - with a and b free instead of fixed, and squashed
        so the multiplier stays inside [1/bound, bound]:

            mult = exp(L tanh((a.M)/L)),  L = ln(bound)

        tanh rather than a clamp so the gradient never dies at the edge of the range.

        THE BOUND IS ON DEPTH, not on speed. What limits how hard the sheet can be driven
        is its thinnest point - (H + h) grounds there first - so bounding c would leave H
        free over bound^2 either way, which at bound 3 is 81x, looser than the incumbent's
        33x and the opposite of the point. Bounded on H, bound 3 gives 9x, and the
        incumbent medium is already outside that: initialising from it and squashing takes
        H_min up from 0.119 to about 0.43, which is the headroom being bought.

        SPEED IS RENORMALISED so its maximum equals the baseline's. dt follows the CFL
        condition at the fastest point on the sheet, so an optimiser free to raise c.max()
        would silently violate the bound dt was chosen for; pinning the maximum keeps dt,
        the clock and the comparability of every earlier run intact. Nothing is lost by it:
        c*dt = CFL*d_min whatever c is, which is why bo_step found c0 inert. What the
        coefficients control is the SHAPE of the grading, which is the part that matters.

        The sponge is not learnable - it is the absorbing boundary, not the medium."""
        import fluid as fl
        from cortical_maps import load_maps
        from swe_rot import sponge_profile
        from genome import SPONGE_STRENGTH_FIXED, SPONGE_WIDTH_FIXED
        mp = load_maps(self.c, self.p["maps"], verbose=False)
        M = np.stack([mp[k] for k in self.p["maps"]])
        self.Mz = torch.as_tensor(M, dtype=self.dtype, device=self.dev)
        self.a_raw = torch.tensor(np.asarray(self.p["a"], float), dtype=self.dtype,
                                  device=self.dev, requires_grad=True)
        self.b_raw = torch.tensor(np.asarray(self.p["b"], float), dtype=self.dtype,
                                  device=self.dev, requires_grad=True)
        self.c0, self.sig0 = float(self.p["c0"]), float(self.p["sig0"])
        cbase, _ = fl.fields(self.c, self.p)
        self.cmax_ref = float(np.max(cbase))
        spg = self.p.get("sponge_scale", 1.0) * sponge_profile(
            self.c.V, self.c.edges, self.c.bnd, SPONGE_WIDTH_FIXED, SPONGE_STRENGTH_FIXED)
        self.sponge = torch.as_tensor(spg, dtype=self.dtype, device=self.dev)
        self.Lb = float(np.log(bound))
        self.Ei = self.sw.Ei
        self.Ej = self.sw.Ej

    def medium(self):
        """-> (H per vertex, damping per vertex), both carrying gradient."""
        L = self.Lb
        hm = torch.exp(L * torch.tanh((2.0 * (self.a_raw @ self.Mz)) / L))
        H = (self.c0 ** 2) * hm
        H = H * ((self.cmax_ref ** 2) / H.max())      # pin the CFL-setting maximum
        sm = torch.exp(L * torch.tanh((self.b_raw @ self.Mz) / L))
        return H, self.sig0 * sm + self.sponge

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

    def lag_score(self, Z):
        """Correlation between the model's antisymmetric Phi(tau) and the data's.

        Double-centred the same way the zero-lag target is, which is what removes the
        global component; the antisymmetric part is then orthogonal to everything tau=0
        constrains, so this adds information rather than re-weighting it."""
        T = Z.shape[1]
        parts = []
        for L in self.lag_L:
            P = (Z[:, :T - L] @ Z[:, L:].T) / (T - L)
            P = P - P.mean(0, keepdim=True) - P.mean(1, keepdim=True) + P.mean()
            parts.append((0.5 * (P - P.T)).reshape(-1))
        v = torch.cat(parts)
        return (v / v.norm().clamp_min(1e-20)) @ self.lagA

    def apply_medium(self):
        """Install the learned damping on the solver and return the learned H.

        Both the descent and the evaluation go through here, so they cannot integrate
        different media - the failure mode that made the first nonlinear run report linear
        scores. Returns the fixed Hf untouched when the medium is not being learned."""
        if getattr(self, "a_raw", None) is None:
            return self.Hf
        H, sig = self.medium()
        self.sw.sig_v = sig
        self.sw.sig_e = 0.5 * (sig[self.Ei] + sig[self.Ej])
        return H

    def score(self, G, eta, parts=False):
        A = self.drive(G, eta)
        H = self.apply_medium()
        F, _, _ = torch_swe.run(self.sw, A, self.Pt, self.dt, self.g, H,
                                save=self.save, chunk=self.chunk, keep=self.cols)
        Z = self.observable(F)
        M = (Z @ Z.T) / Z.shape[1]
        M = M - M.mean(0, keepdim=True) - M.mean(1, keepdim=True) + M.mean()
        e = M[self.iu]
        e = e - e.mean()
        z0 = (e / e.norm().clamp_min(1e-20)) @ self.y
        if self.lagA is None:
            return (z0, z0, None) if parts else z0
        zl = self.lag_score(Z)
        tot = (z0 + self.wa * zl) / (1.0 + self.wa)
        return (tot, z0, zl) if parts else tot

    def eta(self, gen=None):
        r = torch.randn(self.nb, self.K, 2, dtype=self.dtype, device=self.cdev,
                        generator=gen) / np.sqrt(2.0)
        return torch.view_as_complex(r.contiguous())

    # ---------------- the authoritative score, via the repo's own path ----------------
    def run_fn(self):
        """An integrator for `score_realisation`, or None to let it use numpy's.

        The numpy solver has neither nl_flux nor nl_adv, and it cannot express a learned
        medium at all: the squashed grading L*tanh((a.M)/L) is not of the form a'.M, so
        there is no coefficient vector to hand fluid.map_fields that would reproduce it. Under a nonlinearity its field is NOT the field
        being descended, so scoring through it would report the linear score for a
        nonlinear fit - the one discrepancy that would invalidate the whole run rather
        than just add noise. `run_fn` is the hook the module already provides for this.
        At nl_flux 0 it is left alone, so the linear runs keep an independent numpy check
        on the torch path (the two agree to 2e-7)."""
        if (self.sw.nl_flux == 0.0 and self.sw.nl_adv == 0.0
                and getattr(self, "a_raw", None) is None):
            return None

        def _run(d, nsteps, save):
            with torch.no_grad():
                H = self.apply_medium()
                F, _, _ = torch_swe.run(self.sw, np.asarray(d.Aser[:nsteps]),
                                        np.asarray(d.P), self.dt, self.g, H,
                                        save=save, chunk=0)
            return np.asarray(F.cpu().numpy(), np.float32), self.dt
        return _run

    def realised(self, G, seeds=(0, 1, 2, 3, 4, 5)):
        """Simulate and score with `score_realisation`, the repo's own scorer."""
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
                diagnostics=False, run_fn=self.run_fn())
            out.append(r["sim"])
        return float(np.mean(out)), float(np.std(out))


def _medium_state(fit, a):
    """The learned grading, so a saved solution can be rebuilt.

    Without this a run that fits the medium saves only G, and G alone does not define a
    solution: replayed in the baseline medium it is a different system. --learn-maps ran
    once before this existed and that run is not reproducible."""
    if not getattr(a, "learn_maps", 0):
        return {}
    return dict(learn_maps=a.learn_maps,
                map_a=fit.a_raw.detach().cpu().numpy(),
                map_b=fit.b_raw.detach().cpu().numpy())


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
    ap.add_argument("--init", default="warm",
                    help="warm (the convex solve's own factor), cold (random at the same "
                         "scale), or from:<tag> to continue results/torchfit_<tag>.npz - "
                         "the objective does not converge on any budget chosen in "
                         "advance, so a run that is still climbing should be extended "
                         "rather than repeated")
    ap.add_argument("--lr", type=float, default=0.02,
                    help="Adam step as a FRACTION of the initial RMS entry of G. Adam's "
                         "step is scale-free while G's entries are ~1/sqrt(nfreq K^2), so "
                         "an absolute lr of 0.02 is ten times a typical entry and wipes "
                         "out a warm start in three steps")
    ap.add_argument("--amp", type=float, default=2e-4)
    ap.add_argument("--nl-flux", type=float, default=0.0, dest="nl_flux")
    ap.add_argument("--nl-adv", type=float, default=0.0, dest="nl_adv",
                    help="momentum advection, vector-invariant (see core/torch_swe.py)")
    ap.add_argument("--lag-taus", default="", dest="lag_taus",
                    help="comma-separated lags in TR for the lead-lag term, e.g. 1,3,5. "
                         "Empty turns it off and the objective is zero-lag FC alone")
    ap.add_argument("--wa", type=float, default=1.0,
                    help="weight on the lagged term against the zero-lag one")
    ap.add_argument("--learn-maps", type=float, default=0.0, dest="learn_maps",
                    metavar="BOUND",
                    help="also fit the speed/damping map coefficients, with the per-vertex "
                         "multiplier held inside [1/BOUND, BOUND]. 0 (default) leaves the "
                         "medium fixed. Speed is renormalised so its maximum, which sets "
                         "dt through the CFL condition, does not move")
    ap.add_argument("--map-lr", type=float, default=0.02, dest="map_lr",
                    help="Adam step for the map coefficients, as a fraction of their "
                         "initial RMS (they are 6 numbers of order 0.3, not 300k of "
                         "order 0.002, so they need their own scale)")
    ap.add_argument("--maps-scale", type=float, default=1.0, dest="maps_scale",
                    help="scale the speed/damping map coefficients; 0 flattens the medium. "
                         "Flattening also changes dt, because dt = CFL*d_min/c_max and "
                         "c_max falls from 1.978 to 1.0, so the same Ld gives half the "
                         "inertial period in seconds")
    ap.add_argument("--ld", type=float, default=None, dest="Ld",
                    help="override the rotation length scale; f = 1/Ld. The incumbent "
                         "15827.6 gives an inertial period of 23,026 s, ten times the "
                         "longest realisation, so rotation is inert. 52.4 gives 76 s, "
                         "inside the 0.01-0.08 Hz the measurement can see")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--eval-every", type=int, default=25, dest="eval_every")
    ap.add_argument("--eval-draws", type=int, default=6, dest="eval_draws",
                    help="draws per evaluation. `best` takes a MAX over evaluations, so a "
                         "noisy estimate is selected upward: at 2 draws the warm arm's "
                         "best read +0.6497 against +0.6377 +- 0.0083 when the same G was "
                         "re-scored with 6")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None, help="results/torchfit_<tag>.npz")
    ap.add_argument("--dump", default="", dest="dump",
                    help="comma-separated iterations at which to write the CURRENT G to "
                         "results/torchfit_<tag>_it<n>.npz, in the same layout as the "
                         "final file. The trajectory is not otherwise recoverable, and "
                         "what the drive does along the way - whether its pieces start "
                         "out reinforcing and stay, or cross into cancelling - is not "
                         "readable from the endpoint")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    ref = load_reference(a.ref)
    c = load_cortex("fsaverage5", verbose=False)
    t = fc_score.default_target(c, verbose=False)
    fit = Fit(ref, c, t, a.seconds, device=a.device, amp=a.amp, nl_flux=a.nl_flux,
              nl_adv=a.nl_adv, Ld=a.Ld, maps_scale=a.maps_scale, wa=a.wa,
              lag_taus=[int(v) for v in a.lag_taus.split(",") if v.strip()])
    print(f"  ref {a.ref}: {fit.K} pieces, {len(ref['idx'])} solved frequencies, "
          f"{len(ref['sub'])} vertices")
    print(f"  realise {fit.nframes} frames ({a.seconds:.0f}s) = {fit.nsteps} steps, "
          f"chunk {fit.chunk}, amp {a.amp:g}, nl_flux {a.nl_flux:g}, "
          f"nl_adv {a.nl_adv:g}, Ld {fit.p['Ld']:.1f}, maps x{a.maps_scale:g}, "
          f"device {a.device}")
    _Ti = 2 * np.pi * fit.p["Ld"] / fit.dt * (ref["frame_s"] / fit.save)
    print(f"  H {fit.Hf.min():.4f}-{fit.Hf.max():.4f}, dt {fit.dt:.4f}, "
          f"inertial period {_Ti:.0f}s ({_Ti/a.seconds:.2f} of the realisation)")

    cdt = torch.complex64 if fit.dtype == torch.float32 else torch.complex128
    prev = None
    if a.init == "warm":
        G = warm_start(ref["S"], fit.dtype, fit.cdev)
    elif a.init.startswith("from:"):
        prev = np.load(os.path.join(RESULTS, f"torchfit_{a.init[5:]}.npz"),
                       allow_pickle=True)
        G = torch.tensor(prev["G"], dtype=cdt, device=fit.cdev, requires_grad=True)
        print(f"  continuing {a.init[5:]}: best sim {float(prev['best_sim']):+.4f} "
              f"at its iteration {int(prev['best_iter'])}")
    elif a.init == "cold":
        g0 = warm_start(ref["S"], fit.dtype, fit.cdev)
        sc = float(g0.detach().abs().pow(2).sum().sqrt()) / np.sqrt(g0.numel())
        G = (sc * torch.randn_like(g0.detach())).clone().requires_grad_(True)
    else:
        raise SystemExit(f"  --init {a.init}: expected warm, cold or from:<tag>")
    rms = float(G.detach().abs().pow(2).mean().sqrt())
    lr = a.lr * rms
    print(f"  init {a.init}: |G|_F {float(G.detach().abs().pow(2).sum().sqrt()):.4g}, "
          f"RMS entry {rms:.3g} -> Adam lr {lr:.3g}")

    # The medium is built BEFORE the iteration-0 baseline, not after. `realised` routes
    # through the torch integrator and integrates whatever `apply_medium` installs, so
    # building it later measured the baseline on the FIXED medium - whose 0.119 floor is
    # 9.6x below its mean and grounds at any amplitude the bounded medium tolerates. That
    # returned a NaN baseline, and a NaN seeds `best` with a value no later score can
    # beat, because every comparison against NaN is False. The baseline belongs on the
    # medium the descent actually starts from.
    groups = [{"params": [G], "lr": lr}]
    if a.learn_maps:
        fit.init_medium(a.learn_maps)
        # A continuation must resume the MEDIUM as well as G. init_medium always seeds
        # a_raw/b_raw from the incumbent coefficients, so without this the checkpoint's
        # fitted G lands in the baseline grading - a different system, which is the exact
        # failure _medium_state was written to prevent. Restored AFTER init_medium, never
        # by overwriting p["a"]/p["b"] before it: init_medium takes cmax_ref from
        # fl.fields(c, p) to pin the CFL-setting speed maximum, so editing p would move
        # that pin and rebuild a medium that is not the one the checkpoint left.
        if prev is not None and "map_a" in prev:
            if float(prev["learn_maps"]) != float(a.learn_maps):
                raise SystemExit(
                    f"  --learn-maps {a.learn_maps:g} but {a.init[5:]} was fitted at "
                    f"{float(prev['learn_maps']):g}: the tanh bound differs, so its "
                    f"coefficients do not describe the same medium here")
            with torch.no_grad():
                fit.a_raw.copy_(torch.as_tensor(prev["map_a"], dtype=fit.dtype,
                                                device=fit.dev))
                fit.b_raw.copy_(torch.as_tensor(prev["map_b"], dtype=fit.dtype,
                                                device=fit.dev))
            print(f"  resumed the learned grading from {a.init[5:]}")
        mrms = float(np.sqrt(np.mean(np.concatenate(
            [fit.a_raw.detach().cpu().numpy(), fit.b_raw.detach().cpu().numpy()]) ** 2)))
        groups.append({"params": [fit.a_raw, fit.b_raw], "lr": a.map_lr * mrms})
        with torch.no_grad():
            H0, s0f = fit.medium()
        print(f"  medium free, depth multiplier within [1/{a.learn_maps:g}, "
              f"{a.learn_maps:g}]: H {float(H0.min()):.3f}-{float(H0.max()):.3f} "
              f"({float(H0.max()/H0.min()):.1f}x, from {fit.Hf.min():.3f}-"
              f"{fit.Hf.max():.3f} = {fit.Hf.max()/fit.Hf.min():.1f}x fixed), "
              f"damping x{float(s0f.max()/s0f.min()):.1f}, map lr {a.map_lr*mrms:.3g}")

    m0, s0 = fit.realised(G, seeds=tuple(range(a.eval_draws)))
    print(f"  iteration 0 realised: sim {m0:+.4f} +- {s0:.4f}", flush=True)
    # A non-finite baseline must not become `best`: `m > nan` is False for every m, so
    # a NaN seed silently disables best-tracking for the whole run.
    seed0 = ((m0, G.detach().cpu().numpy().copy(), 0) if np.isfinite(m0)
             else (-np.inf, None, 0))
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
                segment=ref["segment"] or None, diagnostics=False,
                run_fn=fit.run_fn())["sim"])
        print(f"  convex S realised the repo's way, same length: "
              f"sim {np.mean(cv):+.4f} +- {np.std(cv):.4f}   "
              f"(interpolation gap {m0-np.mean(cv):+.4f})", flush=True)

    dump_at = {int(v) for v in a.dump.split(",") if v.strip()}

    def dump(it):
        if it not in dump_at or not a.tag:
            return
        Gd = G.detach().cpu().numpy()
        # the whole medium, not just G: a checkpoint that does not say which medium it
        # belongs to cannot be replayed, and these differ in rotation, flattening and
        # which nonlinearities were on
        np.savez(os.path.join(RESULTS, f"torchfit_{a.tag}_it{it}.npz"), G=Gd,
                 S=np.einsum("fab,fcb->fac", Gd, Gd.conj()), best_iter=it, ref=a.ref,
                 best_sim=np.nan, seconds=a.seconds, init=a.init, nl_flux=a.nl_flux,
                 nl_adv=a.nl_adv, Ld=fit.p["Ld"], maps_scale=a.maps_scale,
                 amp=a.amp, hist=np.asarray(hist), **_medium_state(fit, a))
        print(f"        dumped G at iteration {it}", flush=True)

    opt = torch.optim.Adam(groups)
    hist, best = [], seed0        # iteration 0 competes, or the first eval always wins
    dump(0)
    nskip = 0
    for it in range(1, a.iters + 1):
        t0 = time.time()
        opt.zero_grad()
        obj, z0, zl = fit.score(G, fit.eta(), parts=True)
        (-obj).backward()
        # (H + h) goes negative where a trough reaches the bed and the scheme blows up.
        # A single non-finite step would put NaN into G and every later iteration with it,
        # so the update is skipped instead - the run survives and says how often.
        bad = (not torch.isfinite(obj)) or (G.grad is None) or \
              (not torch.isfinite(G.grad).all())
        if bad:
            nskip += 1
            opt.zero_grad()
            print(f"  {it:4d}  NON-FINITE, update skipped ({nskip} so far)", flush=True)
            continue
        opt.step()
        hist.append(obj.detach().item())
        dump(it)
        line = (f"  {it:4d}  surrogate {hist[-1]:+.4f}  [{time.time()-t0:.1f}s]"
                + ("" if zl is None else
                   f"  (fc {z0.detach().item():+.4f}, lag {zl.detach().item():+.4f})"))
        if it % a.eval_every == 0 or it == a.iters:
            if a.learn_maps:
                with torch.no_grad():
                    Hn, sn = fit.medium()
                line += (f"  [H {float(Hn.min()):.2f}-{float(Hn.max()):.2f}, "
                         f"sig x{float(sn.max()/sn.min()):.1f}]")
            m, s = fit.realised(G, seeds=tuple(range(a.eval_draws)))
            line += f"   realised sim {m:+.4f} +- {s:.4f}"
            if np.isfinite(m) and m > best[0]:
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
                 ref=a.ref, seconds=a.seconds, init=a.init, nl_flux=a.nl_flux,
                 nl_adv=a.nl_adv, Ld=fit.p["Ld"], maps_scale=a.maps_scale,
                 amp=a.amp, **_medium_state(fit, a))
        print(f"  wrote results/torchfit_{a.tag}.npz")


if __name__ == "__main__":
    main()
