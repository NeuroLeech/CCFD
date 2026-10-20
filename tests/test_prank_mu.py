"""Finite-difference the penalised objective solve_factor maximises, and the penalty alone.

Three things can go wrong independently and all three still run and still return a
plausible answer: the gradient of PR with respect to A, the w_f factor that carries it to
each S_f, and the chain rule through S = L L^H. The last is the one in_original_basis warns
about in general - a gradient that maps the wrong way type-checks perfectly.

Also asserts the property the whole design rests on: PR is homogeneous of degree zero, so
Re tr(grad . A) = 0 and scaling L leaves the penalty untouched. If that ever stops holding,
the penalty is being optimised by growing or shrinking L rather than by spreading the input,
which is exactly the failure prank_reg has inside an unnormalised solve.

  pytest tests/test_prank_mu.py -s      # conftest puts the code folders on sys.path
"""
import numpy as np
import xspec


def fd_scalar(f, x, h=1e-6):
    g = np.zeros_like(x)
    for i in range(len(x)):
        xp = x.copy(); xp[i] += h
        xm = x.copy(); xm[i] -= h
        g[i] = (f(xp) - f(xm)) / (2 * h)
    return g


def pack(L):
    return np.concatenate([L.real.ravel(), L.imag.ravel()])


def unpack(v, nf, K, r):
    half = nf * K * r
    return v[:half].reshape(nf, K, r) + 1j * v[half:].reshape(nf, K, r)


def test_prank_mu():
    rng = np.random.default_rng(0)
    nf, nV, K, r = 5, 7, 6, 3
    H = (rng.standard_normal((nf, nV, K)) + 1j * rng.standard_normal((nf, nV, K))) / np.sqrt(K)
    w = rng.random(nf) + 0.2
    Ct = rng.standard_normal((nV, nV)); Ct = Ct + Ct.T
    L = (rng.standard_normal((nf, K, r)) + 1j * rng.standard_normal((nf, K, r))) / np.sqrt(K * r)
    R = xspec.prank_ratio(w)

    # 1. the penalty's own gradient with respect to S, finite-differenced through L
    def pen_of(v):
        Lx = unpack(v, nf, K, r)
        return R(Lx @ np.conj(Lx).transpose(0, 2, 1))[0]

    Gs = R(L @ np.conj(L).transpose(0, 2, 1))[1]
    G = 2.0 * (Gs @ L)
    an = np.concatenate([G.real.ravel(), G.imag.ravel()])
    fd = fd_scalar(pen_of, pack(L))
    e1 = np.abs(an - fd).max() / max(np.abs(fd).max(), 1e-300)
    print(f"  penalty grad through S = L L^H : max rel err {e1:.3e}")

    # 2. degree-zero homogeneity: scaling L must not move the penalty at all
    v0 = R(L @ np.conj(L).transpose(0, 2, 1))[0]
    scaled = [R((a * L) @ np.conj(a * L).transpose(0, 2, 1))[0] for a in (0.1, 10.0, 1000.0)]
    e2 = max(abs(s - v0) for s in scaled) / v0
    A = np.einsum('f,fij->ij', w, L @ np.conj(L).transpose(0, 2, 1), optimize=True)
    euler = abs(float(np.real(np.trace(Gs[0] / w[0] @ A))))   # Re tr(dPR/dA . A) = 0
    print(f"  degree-zero in L               : max rel drift {e2:.3e}")
    print(f"  Euler, Re tr(dPR/dA . A)       : {euler:.3e}  (0 means no radial component)")

    # 3. the whole objective solve_factor differentiates, at mu = 0 and mu != 0
    for mu in (0.0, 0.5, 5.0):
        fg, _, _ = xspec.factor_objective(H, w, Ct, r, reg=R, mu=mu)
        _, grad = fg(pack(L))
        fd = fd_scalar(lambda v: fg(v)[0], pack(L))
        e3 = np.abs(grad - fd).max() / max(np.abs(fd).max(), 1e-300)
        print(f"  full objective at mu {mu:>5.1f}      : max rel err {e3:.3e}")
        assert e3 < 2e-5, (mu, e3)

    assert e1 < 2e-5 and e2 < 1e-12 and euler < 1e-10
    print("  OK")


if __name__ == '__main__':
    import os, sys
    for d in ("core", "fit", "targets"):
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), d))
    globals()['xspec'] = __import__('xspec')
    test_prank_mu()
