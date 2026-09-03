"""Temporal autocorrelation and spectrum, empirical against model, on the same clock.

timescale.py measured the empirical side to SET the clock: 9.03 s at 1/e is what the
model's decay was pinned to. That makes the model's own autocorrelation a check rather
than a free result - the question is not whether the number can be fitted but whether
pinning one scalar leaves the rest of the curve in the right place, and whether the
spectrum that comes out matches the f^-2.60 the data has.

Both sides are reduced the same way: demean each vertex, autocorrelate by FFT, average
over vertices, and read 1/e and the integrated time off the mean curve. The model frames
are the OBSERVABLE, not the field - they already carry the BOLD smoothing kernel that
best_fit applied - because that is the thing the empirical data is being compared with.

Sampling differs by construction: the empirical run is at TR = 0.645 s and the model at
TR/4, so lags are reported in seconds and the model curve is read at the empirical lags
when the two are compared.

  python autocorr.py --tag pr_taper
"""
import os, argparse
import numpy as np

from paths import RESULTS
import timescale


def acf(X, nlag):
    """Mean normalised autocorrelation over the rows of a (V, T) array."""
    Xc = np.asarray(X, np.float64)
    Xc = Xc - Xc.mean(1, keepdims=True)
    n = Xc.shape[1]
    nf = 1 << int(np.ceil(np.log2(2 * n)))
    F = np.fft.rfft(Xc, nf, axis=1)
    ac = np.fft.irfft(F * np.conj(F), nf, axis=1)[:, :nlag]
    d = ac[:, :1].copy()
    keep = d[:, 0] > 0
    return (ac[keep] / d[keep]).mean(0)


def one_over_e(ac, dt):
    """First crossing of 1/e, in seconds, linearly interpolated."""
    below = np.flatnonzero(ac < 1.0 / np.e)
    if not len(below):
        return np.nan
    j = int(below[0])
    if j == 0:
        return 0.0
    y0, y1 = ac[j - 1], ac[j]
    return dt * (j - 1 + (y0 - 1.0 / np.e) / max(y0 - y1, 1e-30))


def integrated(ac, dt):
    """Integrated autocorrelation time: sum to the first zero crossing, in seconds."""
    z = np.flatnonzero(ac <= 0)
    end = int(z[0]) if len(z) else len(ac)
    return dt * (0.5 + ac[1:end].sum())


def psd_slope(X, dt, band=(0.01, 0.1)):
    """(slope, band share) of the mean periodogram over the rows of a (V, T) array."""
    Xc = np.asarray(X, np.float64)
    Xc = Xc - Xc.mean(1, keepdims=True)
    n = Xc.shape[1]
    P = (np.abs(np.fft.rfft(Xc, axis=1)) ** 2).mean(0)
    f = np.fft.rfftfreq(n, dt)
    m = (f >= band[0]) & (f <= band[1]) & (P > 0)
    sl = np.polyfit(np.log10(f[m]), np.log10(P[m]), 1)[0]
    share = P[m].sum() / P[1:].sum()
    return float(sl), float(share)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", default="pr_taper")
    ap.add_argument("--subjects", type=int, default=None)
    ap.add_argument("--maxlag-s", type=float, default=60.0, dest="maxlag_s")
    a = ap.parse_args()

    from mesh_cache import load_cortex
    from fc_group_nki import subject_files, load_subject
    import fc_score
    c = load_cortex("fsaverage5", verbose=False)
    t = fc_score.default_target(c, verbose=False)
    tr = timescale.TR
    clock = timescale.plan(4, decay_s=9.03, spread_mm_s=6, verbose=False)
    fs = clock["frame_s"]

    files = subject_files("left")[:a.subjects]
    nlag_e = int(a.maxlag_s / tr)
    acs, sl_e, sh_e = [], [], []
    for i, p in enumerate(files):
        X = load_subject(p)[t.vertices]
        acs.append(acf(X, nlag_e))
        s, sh = psd_slope(X, tr)
        sl_e.append(s); sh_e.append(sh)
        if (i + 1) % 20 == 0:
            print(f"    {i+1}/{len(files)} subjects", flush=True)
    ac_e = np.mean(acs, 0)

    F = np.load(os.path.join(RESULTS, f"frames_{a.tag}.npy"), mmap_mode="r")
    M = np.asarray(F[t.burn:, t.cols], np.float64).T
    ac_m = acf(M, int(a.maxlag_s / fs))
    sl_m, sh_m = psd_slope(M, fs)

    lag_e = np.arange(len(ac_e)) * tr
    lag_m = np.arange(len(ac_m)) * fs
    ac_m_at_e = np.interp(lag_e, lag_m, ac_m)

    print(f"\n  {'':<22s} {'empirical':>12s} {'model':>12s}")
    print(f"  {'sampling (s)':<22s} {tr:>12.4f} {fs:>12.4f}")
    print(f"  {'1/e time (s)':<22s} {one_over_e(ac_e, tr):>12.2f} "
          f"{one_over_e(ac_m, fs):>12.2f}")
    print(f"  {'integrated time (s)':<22s} {integrated(ac_e, tr):>12.2f} "
          f"{integrated(ac_m, fs):>12.2f}")
    print(f"  {'PSD slope 0.01-0.1Hz':<22s} {np.mean(sl_e):>12.2f} {sl_m:>12.2f}")
    print(f"  {'power in 0.01-0.1 Hz':<22s} {np.mean(sh_e):>11.1%} {sh_m:>11.1%}")
    print(f"\n  the two curves at the empirical lags, r = "
          f"{np.corrcoef(ac_e, ac_m_at_e)[0,1]:+.4f} over 0-{a.maxlag_s:.0f} s")
    print(f"  {'lag (s)':>9s} {'empirical':>11s} {'model':>11s}")
    for s in (1.29, 2.58, 5.16, 9.03, 12.9, 19.4, 25.8, 38.7, 51.6):
        if s <= lag_e[-1]:
            print(f"  {s:>9.2f} {np.interp(s, lag_e, ac_e):>11.3f} "
                  f"{np.interp(s, lag_m, ac_m):>11.3f}")

    out = os.path.join(RESULTS, f"autocorr_{a.tag}.npz")
    np.savez(out, ac_emp=ac_e, ac_model=ac_m, lag_emp=lag_e, lag_model=lag_m,
             slope_emp=np.array(sl_e), slope_model=sl_m,
             share_emp=np.array(sh_e), share_model=sh_m)
    print(f"\n  wrote {out}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(lag_e, ac_e, label="empirical (NKI, 99 subjects)")
    ax[0].plot(lag_m, ac_m, label=f"model {a.tag}")
    ax[0].axhline(1 / np.e, color="0.6", lw=0.8, ls="--")
    ax[0].axhline(0, color="0.6", lw=0.8)
    ax[0].set_xlabel("lag (s)"); ax[0].set_ylabel("autocorrelation")
    ax[0].legend(frameon=False, fontsize=9)
    ax[1].plot(lag_e, ac_e, label="empirical")
    ax[1].plot(lag_e, ac_m_at_e, label="model at empirical lags")
    ax[1].set_xlim(0, 20); ax[1].set_xlabel("lag (s)")
    ax[1].legend(frameon=False, fontsize=9)
    for x in ax:
        x.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    png = os.path.join(RESULTS, f"autocorr_{a.tag}.png")
    fig.savefig(png, dpi=120)
    plt.close(fig)
    print(f"  wrote {png}")


if __name__ == "__main__":
    main()
