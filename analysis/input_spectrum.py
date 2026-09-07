"""The spectrum of the input the solve chose, which nothing in the objective constrains.

The convex solve returns one Hermitian PSD matrix S(f) per frequency, and the FC objective
is a zero-lag spatial criterion: it never refers to frequency at all. So the input's
spectrum is whatever fell out, and it is worth looking at directly rather than inferring
it from the field.

`trace(S_f)` is the total input power at bin f, real and non-negative because S is PSD.
Two conventions matter and are both applied:

  BIN WEIGHT. transfer() samples a geometric grid and each kept bin stands for a band of
  width w_f = gradient(idx), which is what the covariance sum weights by. Power PER BIN is
  w_f * trace(S_f); the DENSITY, comparable across a non-uniform grid, is trace(S_f).
  Plotting the un-weighted trace against a log frequency axis would understate the low
  frequencies, where the grid is dense and each bin narrow.

  FREQUENCY. bin k of an rfft over `ref_frames` is k/(ref_frames*frame_s) Hz, so the axis
  needs the pad the run used, not the realisation length.

Per-nucleus power is diag(S) summed over each group's channels, which says which input
system the solve put where in frequency - the same grouping zones.py attributes variance
with. Cross-terms are left out of that split deliberately: they are what makes the groups
sum to more or less than the total, and zones.py already measures them.

  python analysis/input_spectrum.py --tags g7_s1.5_d25,rbc_nobp
"""
import _path  # noqa: F401  - puts the sibling code folders on sys.path
import os, glob, argparse
import numpy as np

from paths import RESULTS
import timescale


def load(tag, pad=4096, frame_s=None):
    """-> (freq Hz, weight, per-bin trace, per-bin diag) for one run's solved S."""
    frame_s = frame_s or timescale.TR / 4.0
    z = np.load(os.path.join(RESULTS, f"xspec_{tag}.npz"), allow_pickle=True)
    S, idx, tags = z["S"], z["idx"], list(z["tags"])
    f = np.asarray(idx, float) / (pad * frame_s)
    w = np.gradient(np.asarray(idx, float))
    dg = np.real(np.diagonal(S, axis1=1, axis2=2))        # (nf, K), real for PSD S
    return f, w, dg.sum(1), dg, tags


def band_share(f, w, tr, lo, hi):
    m = (f >= lo) & (f <= hi)
    return float((w[m] * tr[m]).sum() / max((w * tr).sum(), 1e-300))


def slope(f, tr, lo, hi):
    """log-log slope of the power DENSITY over a band."""
    m = (f >= lo) & (f <= hi) & (tr > 0)
    if m.sum() < 3:
        return np.nan
    return float(np.polyfit(np.log10(f[m]), np.log10(tr[m]), 1)[0])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tags", default="g7_s1.5_d25,rbc_nobp")
    ap.add_argument("--pad", type=int, default=4096)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    from zones import NUCLEI
    frame_s = timescale.TR / 4.0
    tags = [t.strip() for g in a.tags.split(",") for t in sorted(
        os.path.basename(p)[len("xspec_"):-len(".npz")]
        for p in glob.glob(os.path.join(RESULTS, f"xspec_{g.strip()}.npz")))]
    if not tags:
        raise SystemExit(f"  no runs matching xspec_{a.tags}.npz")

    print(f"  bin width {1/(a.pad*frame_s):.5f} Hz, frame {frame_s:.5f} s\n")
    print(f"  {'tag':<18s} {'<0.01':>7s} {'.01-.08':>8s} {'.08-.2':>7s} {'.2-.5':>7s} "
          f"{'>0.5':>7s} {'slope':>7s} {'peak Hz':>8s}")
    keep = {}
    for tag in tags:
        f, w, tr, dg, ptags = load(tag, a.pad, frame_s)
        keep[tag] = (f, w, tr, dg, ptags)
        pk = f[int(np.argmax(tr))]
        print(f"  {tag:<18s} {band_share(f,w,tr,0,0.01):>6.1%} "
              f"{band_share(f,w,tr,0.01,0.08):>7.1%} {band_share(f,w,tr,0.08,0.2):>6.1%} "
              f"{band_share(f,w,tr,0.2,0.5):>6.1%} {band_share(f,w,tr,0.5,99):>6.1%} "
              f"{slope(f,tr,0.01,0.08):>7.2f} {pk:>8.4f}")

    for tag in tags:
        f, w, tr, dg, ptags = keep[tag]
        parcel = np.array([int(s.split("_")[0]) for s in ptags])
        print(f"\n  {tag}: share of input power by nucleus (diagonal only)")
        print(f"  {'nucleus':<11s} {'total':>7s} {'<0.01':>7s} {'.01-.08':>8s} "
              f"{'.08-.2':>7s} {'>0.2':>7s}")
        tot = (w[:, None] * dg).sum()
        for nm, ps, _role in NUCLEI:
            k = np.flatnonzero(np.isin(parcel, ps))
            g = dg[:, k].sum(1)
            row = [(w[(f >= lo) & (f < hi)] * g[(f >= lo) & (f < hi)]).sum() / tot
                   for lo, hi in ((0, 0.01), (0.01, 0.08), (0.08, 0.2), (0.2, 99))]
            print(f"  {nm:<11s} {(w*g).sum()/tot:>6.1%} " +
                  " ".join(f"{v:>6.1%}" for v in row))

    out = a.out or os.path.join(RESULTS, "input_spectrum.png")
    _plot(keep, out, frame_s)


def _plot(keep, out, frame_s):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import bandpass
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    for tag, (f, w, tr, dg, _) in keep.items():
        m = f > 0
        ax[0].loglog(f[m], tr[m], lw=1.4, label=tag)
        ax[1].semilogx(f[m], np.cumsum((w * tr)[m]) / (w * tr)[m].sum(), lw=1.4, label=tag)
    for x in ax:
        x.axvspan(0.01, 0.08, color="0.85", zorder=0)
        x.set_xlabel("frequency (Hz)")
        x.spines[["top", "right"]].set_visible(False)
        x.legend(frameon=False, fontsize=8)
    ax[0].set_ylabel("input power density  trace(S)")
    ax[1].set_ylabel("cumulative share of input power")
    ax[0].set_title("shaded: the 0.01-0.08 Hz passband", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
