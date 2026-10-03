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

  python core/bandpass.py            # self-check: the two paths against each other
"""
import _path  # noqa: F401  - puts the sibling code folders on sys.path
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


# --------------------------------------------------------------------- segmenting
# A block-design run cannot be used whole if the question is about ONE condition: the
# CHECKER frames are three 20 s stretches separated by fixation. Concatenating them and
# demeaning each stretch is how an FC is built from one condition, and it costs the low
# frequencies - a 18 s stretch cannot represent 0.01 Hz. Rather than argue about which
# band survives, the same operation is applied to the model, and the amount lost is a
# number rather than a caveat.


def segment_power(freqs_hz, frame_s, seg_frames):
    """Fraction of the power at each frequency that survives per-segment demeaning.

    Removing the mean of every n-sample segment and then estimating the covariance from
    the concatenation gives, for a stationary process with cross-spectrum P,

        E[cov] = integral P(f) * W(f) df,     W(f) = 1 - |D_n(f)|^2 / n^2

    with D_n the Dirichlet kernel sum_t exp(-2 pi i f t dt). This is exact, not a
    phase-averaged approximation: the estimator itself averages over position within the
    segment, and that average is what turns the two cross terms and the mean-square term
    into a single |D_n|^2/n^2. W is 0 at DC, 1 at every multiple of 1/(n*dt), and in
    between it is what a short window costs.
    """
    f = np.asarray(freqs_hz, float)
    n = int(seg_frames)
    dt = float(frame_s)
    num = np.sin(np.pi * f * n * dt)
    den = np.sin(np.pi * f * dt)
    D = np.where(np.abs(den) < 1e-12, float(n), num / np.where(np.abs(den) < 1e-12, 1.0, den))
    return np.clip(1.0 - (D / n) ** 2, 0.0, 1.0)


def segment_response(idx, ref_frames, frame_s, seg_frames):
    """The SIGNAL-level multiplier for H, on the bins `xspec.transfer` keeps.

    sqrt of segment_power, because H is a signal-level transfer function and the
    covariance carries its square - the same convention `response` follows by returning
    |h|^2 for a filter applied forward and backward."""
    f = np.asarray(idx, float) / (float(ref_frames) * float(frame_s))
    return np.sqrt(segment_power(f, frame_s, seg_frames))


def apply_segments(frames, seg_frames):
    """Chop (T, V) into seg_frames-long segments, demean each along time, concatenate.

    The trailing partial segment is dropped: a shorter segment has a different W and
    would mix two estimators."""
    X = np.asarray(frames)
    n = int(seg_frames)
    k = X.shape[0] // n
    if k < 1:
        raise ValueError(f"{X.shape[0]} frames is shorter than one {n}-frame segment")
    Y = X[:k * n].reshape(k, n, X.shape[1]).astype(np.float32, copy=True)
    Y -= Y.mean(1, keepdims=True)
    return np.ascontiguousarray(Y.reshape(k * n, X.shape[1]))


def _segment_selfcheck(frame_s=0.645, seg=28, ntrial=4000, seed=0):
    """W(f) measured against W(f) predicted, one frequency at a time.

    A sinusoid at f with random phase, demeaned per segment: the retained variance ratio
    IS W(f), so this checks the formula rather than a consequence of it."""
    rng = np.random.default_rng(seed)
    t = np.arange(seg * 40) * frame_s
    print(f"  segment {seg} frames ({seg*frame_s:.1f}s) at frame {frame_s}s")
    print(f"    {'f Hz':>8s} {'predicted':>10s} {'measured':>10s}")
    ok = True
    for f in (0.005, 0.01, 0.02, 0.03, 0.05, 1.0 / (seg * frame_s), 0.08, 0.15):
        ph = rng.uniform(0, 2 * np.pi, ntrial)
        X = np.cos(2 * np.pi * f * t[:, None] + ph[None, :])
        Y = apply_segments(X, seg)
        got = float((Y ** 2).mean() / (X[:Y.shape[0]] ** 2).mean())
        pred = float(segment_power(f, frame_s, seg))
        ok &= abs(got - pred) < 0.01
        print(f"    {f:>8.4f} {pred:>10.4f} {got:>10.4f}")
    print(f"  {'agree' if ok else 'DISAGREE'} to 0.01 at every frequency")
    return ok


def edge_frames(frame_s, lo=LO_HZ):
    """Frames at EACH END that sosfiltfilt's edge replication has corrupted.

    One period of the slowest component the passband keeps: 1/0.01 Hz = 100 s, which is 620
    frames at a TR/4 clock. `apply` runs sosfiltfilt, which invents data past the ends by
    repeating the edge values, so the filtered series is wrong over roughly that span at both
    ends. _selfcheck has always discarded exactly this before comparing its two filter paths -
    "the two can only be compared where neither edge treatment reaches" - and this is that same
    number, lifted out so the SCORING path can apply the rule the self-check states.

    It had not been applied there, and it mattered. fc_score drops burn = 50 frames from the
    START and none from the end, so ~100 s of corrupted signal at each end was entering the
    correlations. Re-scoring saved realisations of the same draw at a range of trims:

        trim each end   converged solve   maxfun 60   maxfun 15
          0 (burn only)     +0.2659        +0.6448     +0.5781
                200         +0.3871        +0.6596     +0.5841
                400         +0.5767        +0.6530     +0.5662
                620         +0.5748        +0.6447     +0.5597
               1200         +0.5282        +0.6176     +0.5270

    The converged solve gains +0.311 and flattens by 400; the early-stopped ones move by less
    than 0.02. The contamination is selective and monotone in optimisation budget (+0.008 at
    maxfun 60, +0.042 at 150, +0.229 at 400, +0.311 at 1500), because a drive concentrated in
    few frequencies and swinging hard gives the edge replication more to get wrong - which is
    what a converged solve produces. Two things say artefact rather than property: trimming
    SHORTENS the record 21%, and a shorter record carries more estimator variance, so the score
    should fall; and the closed form for a single draw's attenuation, derived without reference
    to any of this, predicts the trimmed scores at mean |error| 0.031 against 0.114 untrimmed
    (0.056 against 0.214 on the two runs that moved).

    So any realised score recorded before this was applied is not comparable with one after."""
    return int(round(1.0 / (lo * frame_s)))


def trim_edges(frames, frame_s, lo=LO_HZ):
    """Drop `edge_frames` from both ends. -> the kept frames, which is what gets scored.

    At the 3,578-frame / 577 s realisation this keeps 2,338 frames = 377 s. The score is flat
    from a trim of 400 onward (see edge_frames), so the measured plateau would also be reached
    at 448 s; this uses the derived number rather than the smallest one that works."""
    e = edge_frames(frame_s, lo)
    if 2 * e >= len(frames):
        raise ValueError(f"record of {len(frames)} frames is shorter than the {2*e} frames "
                         f"the {lo} Hz edge costs; realise longer or raise the low cutoff")
    return frames[e:len(frames) - e]


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
    edge = edge_frames(frame_s)
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
    print()
    _segment_selfcheck()
