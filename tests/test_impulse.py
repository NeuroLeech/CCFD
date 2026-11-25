"""xspec.impulse_responses is the system fluid.run integrates: a frame-constant drive
convolved with the responses reproduces the run's own frames, at a coarse clock and in
both readouts. This is the identity the convex solve rests on."""
import numpy as np
import pytest


@pytest.mark.parametrize("decimate", ["snapshot", "mean"])
def test_convolution_reproduces_the_run(decimate):
    from mesh_cache import load_cortex
    import fluid as fl, subparcels, xspec, waves
    c = load_cortex("fsaverage5", verbose=False)
    p, save, _ = waves.standin_medium(c, 2.0, 4.4)                    # save 12, fast decay
    p = dict(p, decimate=decimate)
    labels = np.full(c.nV, -1)
    labels[:40], labels[40:80] = 0, 1
    P = subparcels.taper_profiles(c, labels, 2)
    nfr, nimp = 300, 120
    R = xspec.impulse_responses(c, [0, 1], p, nimp * save, save, profiles=P,
                                verbose=False, cache=False)            # (K, nimp, nV)
    rng = np.random.default_rng(0)
    A = rng.standard_normal((nfr, 2))
    d = xspec.ProfileDrive(c, P, np.repeat(A, save, axis=0) / save, 1.0)
    d.Aser = (np.repeat(A, save, axis=0) / save).astype(np.float32)   # undo the rescale
    frames, _ = fl.run(c, d, p, nfr * save, save)
    pred = sum(np.stack([np.convolve(A[:, k], R[k, :, v])[:nfr]
                         for v in range(c.nV)], 1) for k in range(2))
    sl = slice(nimp, nfr)                                              # past the window
    r = np.corrcoef(pred[sl].ravel(), frames[sl].ravel())[0, 1]
    assert r > 0.9999, r
    assert abs(pred[sl].std() / frames[sl].std() - 1) < 1e-3
