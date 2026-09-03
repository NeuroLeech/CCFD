"""The passband the data was filtered to, applied to the model as well.

XCP-D bandpasses every RBC derivative at 0.01-0.08 Hz, order 2, and the FC target is built
from that. So the model is not being asked to reproduce BOLD's temporal statistics - it
should not be, since BOLD has lost most of them, first to the haemodynamic response and
then to this filter. It is being asked to match what survives INSIDE the passband. The
model keeps its own richer dynamics and the same filter is applied to its observable
before anything is compared.

This is the second stage between the field and the observable, after the BOLD smoothing
kernel in units.py, and it follows the same discipline that kernel does: one object, used
by both sides. units defines a time-domain kernel and derives its DFT; a 0.01 Hz cutoff at
a TR/4 frame would need an FIR of some 2,500 taps, so here the primitive is the analytic
response instead and BOTH sides evaluate it:

    transfer()          multiplies H by response() on the padded rfft grid
    the realised frames are filtered by apply() on their own grid

Those grids differ - 4,096 padded bins against 3,578 realised frames - so the response has
to be a function of frequency in Hz rather than an array, or the two silently describe
different systems. That failure mode is the one units.smooth_frames warns about, and it is
the reason nothing here is written twice.

`response` is |H|^2, not H. XCP-D applies the filter with filtfilt, forward and backward,
which squares the magnitude and cancels the phase; a single-pass |H| would be the wrong
attenuation and would add a delay the data does not have.

  python bandpass.py            # self-check: the two paths against each other
"""
import numpy as np

LO_HZ, HI_HZ, ORDER = 0.01, 0.08, 2          # XCP-D's SoftwareFilters, for RBC


def _sos(frame_s, lo=LO_HZ, hi=HI_HZ, order=ORDER):
    """Second-order sections, not (b, a).

    At a TR/4 frame the Nyquist is 3.1 Hz and the passband edges sit at 0.003 and 0.026 of
    it. A transfer-function form at that ratio loses conditioning badly enough to ring;
    sos is the stable factorisation and scipy recommends it for exactly this case."""
    from scipy.signal import butter
    fs = 1.0 / float(frame_s)
    if hi >= 0.5 * fs:
        raise ValueError(f"passband {hi} Hz is at or above Nyquist {0.5*fs:.3f} Hz")
    return butter(order, [lo, hi], btype="band", fs=fs, output="sos")


def response(freqs_hz, frame_s, lo=LO_HZ, hi=HI_HZ, order=ORDER):
    """|H(f)|^2 of the zero-phase filter, at arbitrary frequencies in Hz.

    Real and non-negative by construction, which is what makes it safe to multiply into a
    transfer function without touching the phase of H."""
    from scipy.signal import sosfreqz
    sos = _sos(frame_s, lo, hi, order)
    fs = 1.0 / float(frame_s)
    _, h = sosfreqz(sos, worN=np.asarray(freqs_hz, float), fs=fs)
    return np.abs(h) ** 2


def apply(frames, frame_s, lo=LO_HZ, hi=HI_HZ, order=ORDER):
    """Filter (T, V) frames along time, zero-phase, the way XCP-D filtered the data."""
    from scipy.signal import sosfiltfilt
    sos = _sos(frame_s, lo, hi, order)
    X = np.asarray(frames, np.float64)
    return np.ascontiguousarray(sosfiltfilt(sos, X, axis=0), dtype=np.float32)


def transfer_response(idx, ref_frames, frame_s, lo=LO_HZ, hi=HI_HZ, order=ORDER):
    """The response on the bins `xspec.transfer` keeps.

    `idx` indexes the rfft of an ref_frames-long window, so bin k is k/(ref_frames*frame_s)
    Hz. Returned in the layout transfer's kernel multiply expects."""
    f = np.asarray(idx, float) / (float(ref_frames) * float(frame_s))
    return response(f, frame_s, lo, hi, order)


def _selfcheck(frame_s=0.16125, n=16384, seed=0):
    """The two paths have to agree, or the solved system is not the scored one."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3))
    td = apply(x, frame_s)
    f = np.fft.rfftfreq(n, frame_s)
    H = response(f, frame_s)
    fd = np.fft.irfft(np.fft.rfft(x, axis=0) * H[:, None], n=n, axis=0)
    # one period of the slowest passband component. sosfiltfilt replicates the edges
    # while the frequency-domain multiply wraps them, so the two can only be compared
    # where neither edge treatment reaches - and the trim must leave a series behind,
    # which at 3 periods and a 3,578-frame run it does not.
    edge = int(round(1.0 / (LO_HZ * frame_s)))
    a, b = td[edge:-edge], fd[edge:-edge]
    r = float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
    rel = float(np.abs(a - b).mean() / np.abs(b).std())
    print(f"  frame {frame_s:.5f}s, {n} frames, passband {LO_HZ}-{HI_HZ} Hz order {ORDER}")
    print(f"  sosfiltfilt against multiply-by-|H|^2, ignoring {edge} edge frames:")
    print(f"    r = {r:.8f}   mean |diff| / sd = {rel:.2e}")
    keep = (f >= LO_HZ) & (f <= HI_HZ)
    print(f"  response: {H[keep].min():.3f}-{H[keep].max():.3f} in band, "
          f"{H[f > 0.2].max():.2e} above 0.2 Hz, {H[(f > 0) & (f < 0.005)].max():.2e} "
          f"below 0.005 Hz")
    P = (np.abs(np.fft.rfft(td, axis=0)) ** 2).mean(1)
    tot = P[1:].sum()
    inb = P[(f >= LO_HZ) & (f <= HI_HZ)].sum() / tot
    print(f"  white noise through it: {inb:.1%} of the surviving power lands in band")
    return r


if __name__ == "__main__":
    _selfcheck()
