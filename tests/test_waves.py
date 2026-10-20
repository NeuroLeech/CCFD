import numpy as np
import pytest

import waves


@pytest.fixture(scope="module")
def surf():
    from mesh_cache import load_cortex
    c = load_cortex("fsaverage5", verbose=False)
    return waves.Surface(c, radius=10.0)


def _seed(S):
    """The vertex farthest from the medial cut, so a radial pattern round it is whole."""
    return int(np.argmax(S.distance_from(np.unique(S.c.edges[S.c.bnd]))))


def _axes(S, seed):
    return dict(anterior=S.axis(S.V[:, 1]), dorsal=S.axis(S.V[:, 2]),
                outward=S.axis(S.distance_from([seed])))


K = 2 * np.pi / 40.0              # 40 mm wavelength
HZ = 0.05


def test_plane_wave_travels_anterior(surf):
    # phase -k y: crests move toward +y, which is anterior
    D = waves.describe(surf, np.exp(-1j * K * surf.V[:, 1]), HZ, _axes(surf, 0))
    assert D["align_anterior"][0] > 0.8
    assert abs(D["align_dorsal"][0]) < 0.3
    assert D["travel"][0] > 0.9
    assert D["planar"][0] > 0.7 and D["coherence"][0] > 0.9
    # speed = omega / k = 0.05 * 40 mm/s on the flat parts; curvature only shortens k
    assert 1.5 < D["speed"][0] < 2.5
    Dr = waves.describe(surf, np.exp(+1j * K * surf.V[:, 1]), HZ, _axes(surf, 0))
    assert Dr["align_anterior"][0] < -0.8


def test_standing_wave_does_not_travel(surf):
    D = waves.describe(surf, np.cos(K * surf.V[:, 1]).astype(complex), HZ,
                       _axes(surf, 0))
    assert D["travel"][0] < 0.05
    assert waves.wave_type(D)[0] == "standing"


def test_source_and_sink(surf):
    s = _seed(surf)
    d = surf.distance_from([s])
    amp = np.exp(-0.5 * (d / 30.0) ** 2)
    ax = _axes(surf, s)
    out = waves.describe(surf, amp * np.exp(-1j * K * d), HZ, ax)
    inn = waves.describe(surf, amp * np.exp(+1j * K * d), HZ, ax)
    assert out["align_outward"][0] > 0.9 and inn["align_outward"][0] < -0.9
    assert out["div"][0] > 0 > inn["div"][0]
    assert waves.wave_type(out)[0] == "source"
    assert waves.wave_type(inn)[0] == "sink"


def test_spiral_has_one_singularity(surf):
    s = _seed(surf)
    n = surf.c.m.vertex_normal[s]
    e1 = np.cross(n, [0, 0, 1.0]); e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)                                  # e1 x e2 = n: counter-clockwise
    # centred inside a face, not on a vertex: at an exact zero the phase is undefined and
    # its charge smears over the whole fan of faces round it
    x = surf.V - (surf.V[s] + 0.31 * e1 + 0.17 * e2)
    d = surf.distance_from([s])
    Phi = (x @ e1 + 1j * (x @ e2)) * np.exp(-0.5 * (d / 15.0) ** 2)
    q = surf.charge(Phi)
    near = np.linalg.norm(surf.V[surf.F].mean(1) - surf.V[s], axis=1) < 5
    assert q[near].sum() == 1
    D = waves.describe(surf, Phi, HZ, _axes(surf, s))
    assert D["n_sing"][0] >= 1 and D["net_charge"][0] == 1
    # phase winds counter-clockwise, so crests travel clockwise: negative curl
    assert D["curl"][0] < 0 and abs(D["curl"][0]) > abs(D["div"][0])
    assert waves.wave_type(D)[0] == "rotating"


def test_invariant_to_global_phase_and_scale(surf):
    rng = np.random.default_rng(0)
    Phi = (rng.standard_normal(surf.nV) + 1j * rng.standard_normal(surf.nV))
    Phi = np.asarray(surf.K @ Phi)                        # something smooth-ish
    a = waves.describe(surf, Phi, HZ, _axes(surf, 0))
    b = waves.describe(surf, 3.7 * np.exp(1.1j) * Phi, HZ, _axes(surf, 0))
    for k in a:
        assert np.allclose(a[k], b[k], atol=1e-4), k


def test_transfer_matches_impulse_responses():
    """The batched, Fourier-summed transfer function is the FFT of the serial impulse
    responses xspec computes, at the frequencies asked for."""
    from mesh_cache import load_cortex
    import xspec, subparcels
    c = load_cortex("fsaverage5", verbose=False)
    p, save, frame_s = waves.standin_medium(c, 25.0, 1.5)
    labels = np.full(c.nV, -1)
    labels[:40], labels[4000:4040] = 0, 1
    P = subparcels.taper_profiles(c, labels, 2)
    run = dict(p=p, save=save, frame_s=frame_s, profiles=P, nimp=48)
    hz = np.array([0.02, 0.07])
    H = waves.transfer_fields(c, run, hz, chunk=7, verbose=False)
    R = xspec.impulse_responses(c, [0, 1], p, 48 * save, save, profiles=P,
                                verbose=False, cache=False)          # (K, 48, nV)
    w = 2 * np.pi * hz * frame_s
    ref = np.einsum("fn,knv->fvk", np.exp(-1j * np.outer(w, np.arange(48))), R)
    assert np.abs(H - ref).max() < 1e-3 * np.abs(ref).max()


def test_radial_is_coherent_but_not_planar(surf):
    s = _seed(surf)
    d = surf.distance_from([s])
    D = waves.describe(surf, np.exp(-0.5 * (d / 30.0) ** 2) * np.exp(-1j * K * d), HZ,
                       _axes(surf, s))
    assert D["coherence"][0] > 0.7 and D["planar"][0] < 0.3
