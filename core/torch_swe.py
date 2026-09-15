"""The rotating shallow-water stepper in torch, so the drive can be differentiated.

The convex solve in `fit/xspec.py` exists because the medium is linear: C = sum_f H S H^H,
so the input enters only through one Hermitian PSD matrix per frequency. Nothing in that
survives a nonlinearity - there is no H, and the covariance no longer factorises. What
replaces it is gradient descent on a drive that is SIMULATED rather than propagated
analytically, which needs the stepper to be differentiable.

This is a port, not a new model. Every operator is built from the numpy `RotSWE` the rest
of the repo uses, so the geometry and the medium have one source of truth, and the
arithmetic mirrors `RotSWE.step` line for line. The sparse matvecs become index_add /
index_select, which are each other's adjoints - so the transpose identity that makes
Coriolis exactly energy-neutral is preserved by construction rather than by a second
sparse matrix that has to be kept in sync - and which, unlike torch.sparse, work on MPS.

`nl_flux` is the amplitude nonlinearity: the depth update uses (H + nl*h) instead of H, so
the local wave speed becomes sqrt(g(H+h)) and superposition goes. It is 0 by default, and
at 0 this integrates exactly the same system as the numpy code.

`nl_adv` is momentum advection, the other half, in VECTOR-INVARIANT form,

    du/dt = -grad(g h + |u|^2/2) - (zeta + f) n x u

rather than as u.grad(u), because ue is one scalar per edge - the component along it - so
there is no local vector to advect. The invariant form is the same equations split into two
pieces this file already has operators for: the kinetic-energy gradient reuses `grad`
exactly as the pressure term does, and the relative vorticity simply ADDS to f inside
`coriolis`. That matters structurally - the term stays a rotation, so the
scatter = transpose(gather) construction that makes Coriolis exactly energy-neutral is
untouched, which a u.grad(u) discretisation would have spent.

zeta is evaluated at the OLD time level and held fixed across the Crank-Nicolson passes.
A rotation at a fixed rate is exactly energy-neutral; recomputing it inside the iteration
would make the rate depend on the iterate and give that up for a higher-order correction
to a term that is already small.

Measured on this mesh (9,374 vertices, 27,983 edges), 2026-09-13: float64 agrees with
the numpy stepper to 3.9e-16 and float32 to 2.0e-7; the gradient matches central
differences to 2e-10 with nl_flux 0 and 9e-11 with nl_flux 1; checkpointing reproduces the
full-tape gradient bit for bit. Per step, forward: numpy 0.66 ms, torch CPU 1.73 ms, torch
CPU compiled 0.56 ms, torch MPS 0.49 ms. Forward+backward on MPS is ~4x forward, so a
57,000-step realisation (2,308 s at oversample 4) costs about two minutes per gradient.
MPS matches CPU to 5e-7 and repeats itself to 1e-7.

  python core/torch_swe.py            # agreement with numpy, and a gradient check
"""
import _path  # noqa: F401  - puts the sibling code folders on sys.path
import numpy as np
import torch
import torch.utils.checkpoint


class TorchSWE:
    """Mirror of RotSWE.step. Build from an already-configured numpy solver.

    The numpy solver is float32 by the time `fluid.build` returns it; `dtype` here only
    sets the arithmetic width, not the values, so a float64 TorchSWE and a float64-promoted
    RotSWE hold bit-identical operators and differ only in summation order."""

    def __init__(self, s, dtype=torch.float64, device="cpu", nl_flux=0.0, nl_adv=0.0,
                 cortex=None):
        if s.coupling is not None:
            raise ValueError("the long-range coupling term is not ported")
        self.dtype, self.device = dtype, device
        self.nV, self.nE = s.nV, s.nE
        self.nl_flux, self.nl_adv = float(nl_flux), float(nl_adv)

        def T(a, d=None):
            return torch.as_tensor(np.asarray(a), dtype=d or dtype, device=device)

        self.Ei, self.Ej = T(s.Ei, torch.long), T(s.Ej, torch.long)
        # g_i/g_j are the only geometry RotSWE.astype leaves alone, because the production
        # path reads them through G. Cast to float32 first so this holds what G holds.
        self.g_i, self.g_j = T(s.g_i.astype(np.float32)), T(s.g_j.astype(np.float32))
        self.n_i, self.f = T(s.n_i), T(s.f)
        self.d, self.l = T(s.d), T(s.l)
        self.invA, self.invAe = T(s._invA), T(s._invAe)
        # `un[bnd_edge] = 0` as a multiply: same result, no in-place write on a tensor
        # autograd still needs
        self.keep = T(~np.asarray(s.bnd_edge, bool))
        self.sig_v = None if s.sig_v is None else T(s.sig_v)
        self.sig_e = None if s.sig_v is None else T(s.sig_e)
        if self.nl_adv != 0.0:
            self._build_vorticity(cortex, T)

    def _build_vorticity(self, c, T):
        """Circulation round each triangle of the INTEGRATED mesh -> relative vorticity.

        The faces are `cortex.F2`, the intrinsic-Delaunay triangulation the solver's edges,
        lengths and dual areas all come from - NOT cortex.m, which is the extrinsic mesh the
        surface arrived with. Building this from m.F instead gave vorticity 1.656 for a
        discrete gradient field, which must telescope to 0 exactly.

        F2 is stored already consistently wound about the surface normal (mesh_cache._orient
        checks it combinatorially), and zeta is taken about that same normal, so adding it
        to f is consistent with whatever sign convention the Coriolis cross product uses.
        Triangle areas are the intrinsic ones: a flipped edge joins vertices that are not
        adjacent on the surface, so its triangle is not the flat one spanned by the three
        points."""
        if c is None or getattr(c, "F2", None) is None:
            raise ValueError("nl_adv needs the intrinsic faces (pass cortex=...)")
        F2 = np.asarray(c.F2)
        eidx = {(int(a), int(b)): n for n, (a, b) in enumerate(np.asarray(c.edges))}
        kk = lambda i, j: (i, j) if i < j else (j, i)
        eL, sL = [], []
        for pq in ((1, 2), (2, 0), (0, 1)):          # the three legs of a positive circuit
            e = np.array([eidx[kk(int(t[pq[0]]), int(t[pq[1]]))] for t in F2])
            eL.append(e)
            sL.append(np.where(np.asarray(c.edges)[e, 0] == F2[:, pq[0]], 1.0, -1.0))
        self.vort_e = torch.as_tensor(np.stack(eL), dtype=torch.long, device=self.device)
        self.vort_s = T(np.stack(sL))                          # (3, nF)
        # self.d, not c.d: `grad` divides by self.d, so the circulation of a discrete
        # gradient telescopes to machine zero only if the same lengths multiply it back
        self.vort_d = self.d[self.vort_e]                      # (3, nF)
        self.tri_area = T(np.asarray(c.tri_area))
        self.tri_v = torch.as_tensor(np.asarray(F2.T, np.int64), dtype=torch.long,
                                     device=self.device)
        wsum = np.zeros(self.nV)
        for k in range(3):
            np.add.at(wsum, F2[:, k], np.asarray(c.tri_area))
        self.vort_wsum = T(np.maximum(wsum, 1e-30))

    def vorticity(self, ue):
        """-> relative vorticity per vertex, area-weighted from the incident triangles."""
        circ = (self.vort_s * self.vort_d * ue[self.vort_e]).sum(0)     # (nF,)
        zt = circ / self.tri_area
        acc = torch.zeros(self.nV, dtype=self.dtype, device=self.device)
        for k in range(3):
            acc = acc.index_add(0, self.tri_v[k], zt * self.tri_area)
        return acc / self.vort_wsum

    # ---- operators: index_add is the scatter, index_select the gather ----
    def gather(self, ue):
        U = torch.zeros(self.nV, 3, dtype=self.dtype, device=self.device)
        U = U.index_add(0, self.Ei, self.g_i * ue[:, None])
        U = U.index_add(0, self.Ej, self.g_j * ue[:, None])
        U = U * self.invA[:, None]
        return U - (U * self.n_i).sum(1)[:, None] * self.n_i

    def coriolis(self, ue, frot=None):
        U = self.gather(ue)
        f = self.f if frot is None else frot
        Vr = f[:, None] * torch.linalg.cross(self.n_i, U)
        acc = (self.g_i * Vr[self.Ei]).sum(1) + (self.g_j * Vr[self.Ej]).sum(1)
        return acc * self.invAe

    def divergence(self, ue):
        acc = torch.zeros(self.nV, dtype=self.dtype, device=self.device)
        acc = acc.index_add(0, self.Ei, self.l * ue)
        acc = acc.index_add(0, self.Ej, -self.l * ue)
        return acc * self.invA

    def grad(self, h):
        return (h[self.Ej] - h[self.Ei]) / self.d

    def step(self, ue, h, dt, g, H, niter=2):
        frot = None
        rhs = ue + dt * (-g * self.grad(h))
        if self.nl_adv != 0.0:
            U = self.gather(ue)
            K = 0.5 * (U * U).sum(1)                       # kinetic energy per vertex
            rhs = rhs - dt * self.nl_adv * self.grad(K)
            frot = self.f + self.nl_adv * self.vorticity(ue)
        Cold = self.coriolis(ue, frot)
        un = rhs + dt * Cold                       # the first pass in closed form, as numpy
        for _ in range(niter - 1):
            un = rhs + 0.5 * dt * (Cold + self.coriolis(un, frot))
        un = un * self.keep
        Hf = H if self.nl_flux == 0.0 else H + self.nl_flux * h
        hn = h - dt * Hf * self.divergence(un)
        if self.sig_v is not None:
            un = un / (1.0 + dt * self.sig_e)
            hn = hn / (1.0 + dt * self.sig_v)
        return un, hn

    def zeros(self):
        return (torch.zeros(self.nE, dtype=self.dtype, device=self.device),
                torch.zeros(self.nV, dtype=self.dtype, device=self.device))


def _as(v, sw):
    """Tensor on the solver's device, passing an existing tensor through so the graph
    survives - np.asarray on an MPS tensor raises rather than moving it."""
    if torch.is_tensor(v):
        return v.to(dtype=sw.dtype, device=sw.device)
    return torch.as_tensor(np.asarray(v), dtype=sw.dtype, device=sw.device)


def run(sw, A, P, dt, g, H, save=1, chunk=0, state=None, keep=None):
    """Integrate, returning saved frames. The drive is `A[n] @ P` added to h per step,
    exactly as in `fluid.run` - held as (nsteps, K) amplitudes and (K, nV) profiles rather
    than a dense (nsteps, nV) field, which at 57,000 steps would be 4 GB on its own.

    `chunk` > 0 wraps every `chunk` steps in a gradient checkpoint: the segment's tape is
    discarded after the forward pass and recomputed during the backward one. A 57,000-step
    realisation holds ~9 GB in (ue, h) alone before any intermediates, so the whole tape
    does not fit; sqrt(nsteps) segments turn that into tens of MB for about 2x the forward
    cost. `keep` is a vertex index: only those columns are kept, which is what the loss
    needs and keeps the saved frames off the graph at full width.
    -> (frames, ue, h), with the final state returned so windows can be chained."""
    nsteps = A.shape[0]
    ue, h = sw.zeros() if state is None else state
    A = _as(A, sw)
    P = _as(P, sw)
    if keep is not None and not torch.is_tensor(keep):
        keep = torch.as_tensor(np.asarray(keep), dtype=torch.long, device=sw.device)
    dt = torch.as_tensor(dt, dtype=sw.dtype, device=sw.device)
    g = torch.as_tensor(g, dtype=sw.dtype, device=sw.device)
    H = _as(H, sw)

    def seg(ue, h, a, off):
        out = []
        dr = a @ P
        for n in range(dr.shape[0]):
            ue, h = sw.step(ue, h + dr[n], dt, g, H)
            if (off + n) % save == 0:
                out.append(h if keep is None else h[keep])
        nc = sw.nV if keep is None else len(keep)
        return ue, h, (torch.stack(out) if out else
                       torch.zeros(0, nc, dtype=sw.dtype, device=sw.device))

    frames, k = [], chunk if chunk > 0 else nsteps
    for a0 in range(0, nsteps, k):
        blk = A[a0:a0 + k]
        if chunk > 0 and torch.is_grad_enabled():
            ue, h, fr = torch.utils.checkpoint.checkpoint(
                seg, ue, h, blk, a0, use_reentrant=False)
        else:
            ue, h, fr = seg(ue, h, blk, a0)
        if fr.shape[0]:
            frames.append(fr)
    return torch.cat(frames) if frames else None, ue, h


# --------------------------------------------------------------------------------------
# Verification. Three questions, in the order that makes a failure interpretable:
# does the port integrate the same system, is its gradient the gradient of what it
# integrates, and does checkpointing change either.
# --------------------------------------------------------------------------------------
def _check_advection(s, cortex, dtype=torch.float64):
    """Three properties the advection terms must have before any run is worth reading."""
    sw = TorchSWE(s, dtype=dtype, nl_adv=1.0, cortex=cortex)
    rng = np.random.default_rng(0)

    # 1. curl of a gradient is zero - exactly, because the directed differences telescope
    phi = torch.as_tensor(rng.standard_normal(sw.nV), dtype=dtype)
    z = sw.vorticity(sw.grad(phi))
    ue = torch.as_tensor(rng.standard_normal(sw.nE) * 1e-3, dtype=dtype)
    scale = float(sw.vorticity(ue).abs().max())
    print(f"  vorticity of a discrete gradient: {float(z.abs().max()):.3e} "
          f"(a generic field gives {scale:.3e})")

    # 2. the rotation term stays energy-neutral once zeta is folded into f, which is the
    #    whole reason for the vector-invariant form
    for lab, frot in (("f alone", None), ("f + zeta", sw.f + sw.vorticity(ue))):
        Cu = sw.coriolis(ue, frot)
        num = float((sw.Ae_t * ue * Cu).sum()) if hasattr(sw, "Ae_t") else \
              float(((1.0 / sw.invAe) * ue * Cu).sum())
        den = float(((1.0 / sw.invAe) * ue * ue).sum())
        print(f"  energy drift of the rotation term, {lab:9s}: "
              f"{abs(num) / max(den, 1e-300):.3e} of its energy")

    return sw


def _setup(K=8, nsteps=60, seed=0, dtype=torch.float64):
    from mesh_cache import load_cortex
    from input2 import parcel_tapers
    import fluid as fl, xspec
    c = load_cortex("fsaverage5", verbose=False)
    p = xspec.medium(1.6125e-3)                  # the oversample-4 / 25 s per-step damping
    s, dt, g, Hf = fl.build(c, p)
    T, ids = parcel_tapers(c, verbose=False)
    pos = {int(q): i for i, q in enumerate(ids)}
    P = np.stack([T[pos[int(k)]] for k in xspec.REGIONS[:K]])
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((nsteps, K)) * 2e-4
    return c, s, dt, g, Hf, P, A


def _numpy_run(s, A, P, dt, g, Hf, nV, dtype):
    s.astype(dtype)                              # promote the float32 operators in place
    A, P = A.astype(dtype), P.astype(dtype)
    h = np.zeros(nV, dtype); ue = np.zeros(s.nE, dtype)
    fr = []
    for n in range(len(A)):
        h = h + A[n] @ P
        ue, h = s.step(ue, h, dtype(dt), dtype(g), Hf.astype(dtype))
        fr.append(h.copy())
    return np.asarray(fr), ue


def _rel(a, b):
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))


def main():
    import time
    torch.manual_seed(0)
    c, s, dt, g, Hf, P, A = _setup()
    nsteps = len(A)

    print("agreement with the numpy stepper "
          f"({nsteps} steps, {c.nV} vertices, {s.nE} edges)")
    for npdt, tdt, nm in ((np.float64, torch.float64, "float64"),
                          (np.float32, torch.float32, "float32")):
        Fn, uen = _numpy_run(s, A, P, dt, g, Hf, c.nV, npdt)
        sw = TorchSWE(s, dtype=tdt)
        Ft, uet, _ = run(sw, A, P, dt, g, Hf, save=1)
        print(f"  {nm}: h {_rel(Ft.numpy(), Fn):.3e}   ue {_rel(uet.numpy(), uen):.3e}  "
              f"(peak |h| {np.abs(Fn).max():.3e})")

    # ---- gradient. Central differences along random directions in A, in float64, on the
    # real mesh - a full gradcheck jacobian at 9,374 vertices is not the useful test, and
    # what has to be right is the gradient of the quantity actually optimised.
    sw = TorchSWE(s, dtype=torch.float64)

    def loss_of(Av, nl=0.0, chunk=0):
        sw.nl_flux = nl
        At = torch.tensor(np.asarray(Av), dtype=torch.float64, requires_grad=True)
        F, _, _ = run(sw, At, P, dt, g, Hf, save=1, chunk=chunk)
        L = (F ** 2).sum() * 1e6
        return L, At

    for nl in (0.0, 1.0):
        L, At = loss_of(A, nl)
        L.backward()
        gA = At.grad.numpy().copy()
        rng = np.random.default_rng(1)
        errs = []
        for _ in range(3):
            V = rng.standard_normal(A.shape); V /= np.linalg.norm(V)
            eps = 1e-8
            with torch.no_grad():
                Lp, _ = loss_of(A + eps * V, nl); Lm, _ = loss_of(A - eps * V, nl)
            fd = (Lp.item() - Lm.item()) / (2 * eps)
            errs.append(abs(fd - float((gA * V).sum())) / max(abs(fd), 1e-300))
        print(f"gradient vs central differences, nl_flux={nl}: "
              f"rel err {max(errs):.2e} (3 directions)")

    # ---- checkpointing must not change the gradient, only where it is stored
    L0, A0 = loss_of(A, 0.0, chunk=0); L0.backward()
    L1, A1 = loss_of(A, 0.0, chunk=13); L1.backward()
    print(f"checkpoint (chunk 13) vs full tape: loss {abs(L0.item()-L1.item()):.2e}, "
          f"grad {_rel(A1.grad.numpy(), A0.grad.numpy()):.2e}")

    # ---- momentum advection
    from mesh_cache import load_cortex
    cc = load_cortex("fsaverage5", verbose=False)
    _check_advection(s, cc)

    swa = TorchSWE(s, dtype=torch.float64, nl_adv=1.0, cortex=cc)
    sw0 = TorchSWE(s, dtype=torch.float64)
    # The ABSOLUTE departure is what must scale as u^2: ||Fa - Fl|| ~ u^2 while ||Fl|| ~ u,
    # so the RELATIVE departure goes as u and halving the amplitude halves it - which any
    # term linear in u would also do. Only the absolute norm tells a quadratic term from a
    # linear one, and it must quarter.
    prev = None
    for mult in (1.0, 0.5, 0.25):
        Fa, _, _ = run(swa, A * mult, P, dt, g, Hf, save=1)
        Fl, _, _ = run(sw0, A * mult, P, dt, g, Hf, save=1)
        ab = float((Fa - Fl).norm())
        rel = float((Fa - Fl).norm() / Fl.norm())
        print(f"  advection at amp x{mult:<5g}: |Fa-Fl| {ab:.4e}  relative {rel:.3e}"
              + (f"   absolute ratio {ab/prev:.4f} (u^2 requires 0.25)" if prev else ""))
        prev = ab

    At = torch.tensor(np.asarray(A), dtype=torch.float64, requires_grad=True)
    F, _, _ = run(swa, At, P, dt, g, Hf, save=1)
    L = (F ** 2).sum() * 1e6
    L.backward()
    gA = At.grad.numpy().copy()
    rng = np.random.default_rng(1); errs = []
    for _ in range(3):
        V = rng.standard_normal(A.shape); V /= np.linalg.norm(V)
        eps = 1e-8
        with torch.no_grad():
            Lp = (run(swa, A + eps*V, P, dt, g, Hf, save=1)[0] ** 2).sum().item() * 1e6
            Lm = (run(swa, A - eps*V, P, dt, g, Hf, save=1)[0] ** 2).sum().item() * 1e6
        fd = (Lp - Lm) / (2 * eps)
        errs.append(abs(fd - float((gA * V).sum())) / max(abs(fd), 1e-300))
    print(f"  gradient vs central differences, nl_adv=1: rel err {max(errs):.2e}")

    # ---- cost, at the precision a fit would use
    sw32 = TorchSWE(s, dtype=torch.float32)
    Al = np.repeat(A, 20, axis=0)                       # 1,200 steps
    t0 = time.time()
    with torch.no_grad():
        run(sw32, Al, P, dt, g, Hf, save=4)
    fwd = time.time() - t0
    At = torch.tensor(Al, dtype=torch.float32, requires_grad=True)
    t0 = time.time()
    F, _, _ = run(sw32, At, P, dt, g, Hf, save=4, chunk=35)
    (F ** 2).sum().backward()
    bwd = time.time() - t0
    print(f"float32, {len(Al)} steps: forward {fwd:.2f}s, "
          f"forward+backward (chunk 35) {bwd:.2f}s  [{bwd/fwd:.1f}x]")


if __name__ == "__main__":
    main()
