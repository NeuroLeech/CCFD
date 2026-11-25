"""The haemodynamic response, as an observation stage between the field and BOLD.

The measured signal is the field seen through the haemodynamics, which pass little above
about 0.1 Hz. An observable without them lets a fit carry FC structure at frequencies the
data barely contains: the rebuilt 3x solve, fitted on the field low-passed by a 1.29 s
gaussian only, loses most of its score when the inputs' relative timing is jittered by a
few tenths of a second (docs/framing/phaseA_findings.md, A1). `best_fit.py --hrf` puts
this stage into both sides of the fit, the way `--bandpass` puts the passband in:

    transfer_response()   multiplies H by the gain |HRF(f)| on the solved bins
    apply()               convolves the realised frames with the same kernel

The kernel is the SPM canonical double gamma (peak 6 s, undershoot 16 s, ratio 1/6) over
40 s, normalised to unit sum so the gain is 1 at 0 Hz. The same kernel, sampled on the
frame clock, defines both paths, so they describe the same system. H takes the gain alone:
the HRF is identical at every vertex, so its phase is a common delay and cancels in every
cross-spectrum between vertices, and FC cannot see it. The realised frames are filtered
causally and so carry that delay; FC is blind to it for the same reason.

`analysis/interplay.py` reads the same function (`power`, at 0.01 s sampling) for its
"haemodynamic response" reading.

  python core/hrf.py            # self-check: the two paths against each other
"""
import _path  # noqa: F401  - puts the sibling code folders on sys.path
import numpy as np

PEAK_S, UNDER_S, RATIO, LENGTH_S = 6.0, 16.0, 1.0 / 6.0, 40.0


def kernel(dt, length_s=LENGTH_S):
    """The canonical HRF sampled every `dt` seconds over `length_s`, unit sum."""
    from scipy.stats import gamma
    t = np.arange(0.0, length_s, float(dt))
    h = gamma.pdf(t, PEAK_S) - gamma.pdf(t, UNDER_S) * RATIO
    return h / h.sum()


def power(hz, dt=0.01):
    """|HRF(f)|^2 at arbitrary frequencies in Hz, 1 at 0 Hz, of the kernel sampled every
    `dt` seconds (its discrete-time Fourier transform)."""
    h = kernel(dt)
    t = np.arange(len(h)) * float(dt)
    f = np.atleast_1d(np.asarray(hz, float))
    return np.abs(np.exp(-2j * np.pi * np.outer(f, t)) @ h) ** 2


def gain(hz, frame_s):
    """|HRF(f)| of the kernel on the frame clock: what multiplies H."""
    return np.sqrt(power(hz, dt=frame_s))


def transfer_response(idx, ref_frames, frame_s):
    """The gain on the bins `xspec.transfer` keeps (bin k is k/(ref_frames*frame_s) Hz)."""
    f = np.asarray(idx, float) / (float(ref_frames) * float(frame_s))
    return gain(f, frame_s)


def apply(frames, frame_s, chunk=2048):
    """Convolve (T, V) frames along time with the kernel, causally.

    The record is extended before its start by mirroring over one kernel length, as
    bandpass.apply does, so the first 40 s are filtered against the signal's own
    statistics rather than against zeros. Done in column chunks to bound memory."""
    from scipy.signal import fftconvolve
    X = np.asarray(frames)
    h = kernel(frame_s)
    n = X.shape[0]
    P = min(len(h), n)
    shape = (-1,) + (1,) * (X.ndim - 1)
    if X.ndim == 1:
        X = X[:, None]
    out = np.empty(X.shape, np.float32)
    for j in range(0, X.shape[1], chunk):
        Xp = np.pad(X[:, j:j + chunk].astype(np.float64), [(P, 0), (0, 0)],
                    mode="symmetric")
        out[:, j:j + chunk] = fftconvolve(Xp, h[:, None], axes=0)[P:P + n]
    return out.reshape(np.asarray(frames).shape) if np.asarray(frames).ndim == 1 else out


if __name__ == "__main__":
    # the two paths against each other: white noise through apply(), its spectrum against
    # power() on the same grid
    frame_s, T = 0.1613, 2 ** 15
    rng = np.random.default_rng(0)
    x = rng.standard_normal((T, 64))
    y = apply(x, frame_s)
    f = np.fft.rfftfreq(T, frame_s)
    ratio = (np.abs(np.fft.rfft(y, axis=0)) ** 2).mean(1) / \
        (np.abs(np.fft.rfft(x, axis=0)) ** 2).mean(1)
    for hz in (0.01, 0.03, 0.05, 0.1, 0.2, 0.4, 0.7):
        k = slice(max(1, int(hz / f[1]) - 40), int(hz / f[1]) + 40)
        print(f"  {hz:5.2f} Hz: measured {ratio[k].mean():.4f}   "
              f"power() {power(f[k], frame_s).mean():.4f}   "
              f"(0.01 s sampling {power(hz)[0]:.4f})")
