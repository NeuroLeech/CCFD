"""Summarise bandpassed fits: what the medium was, and what it produced.

A sweep leaves one xspec_<tag>.npz and one frames_<tag>.npy per run and its numbers in a
log. Reading the log means trusting that the log matches the files; this reads the FILES,
recovering the medium from the saved parameter vector and recomputing everything else from
the realisation, so a mislabelled tag shows up as an inconsistent row rather than a
plausible one.

`x` holds log10(damping per step) and log10(save), which is enough to recover the pair the
sweeps vary:

    spread = MM_PER_STEP * save / frame_s        decay = frame_s / (damp * save)

Four quantities per run, chosen because the sweeps moved them in different directions:

  sim      Spearman against the target over its fixed edge sample, for the ONE saved
           realisation. best_fit reports the mean over its draws and only keeps draw 0's
           frames, so this sits within the draw scatter of the logged number rather than
           equalling it - +0.6192 here against a logged +0.6137 +- 0.0055. The frames it
           reads are already the filtered observable: score_realisation applies the
           bandpass before returning them
  rank     participation ratio of the realisation - the model's in-band dimensionality
  r50      distance at which inter-vertex temporal correlation falls to 0.5, the FC
           correlation length. NOT the field's instantaneous spatial autocorrelation,
           which spatial_scale.py measures and which is a smaller number
  bp       whether the run's OBSERVABLE was bandpassed. Nothing in the saved files
           records it, so it is detected from the realisation's own spectrum. The cut is
           at 50%, where the runs actually separate: filtered ones land at 73-90% and the
           unfiltered control at 12.8%. Not higher - the filter is order 2, not a brick
           wall, so |H|^2 is only 0.5 at the band edges and a filtered run still leaks a
           quarter of its power outside. A 80% cut mislabels a third of the sweep.
           Without this a glob that catches an unfiltered control puts it in the same
           table as the filtered runs, where it tops the ranking for the wrong reason

The empirical row is the same three measures on the cohort's own resting runs, which are
already bandpassed by XCP-D, so it is the like-for-like comparison rather than a reference
value carried over from another preprocessing.

  python bp_sweep.py --tags 'grid_s*_d*'
  python bp_sweep.py --tags 'bpsweep_s*' --empirical 20
"""
import os, glob, argparse
import numpy as np

from paths import RESULTS, CACHE
import timescale

MM_PER_STEP = timescale.MM_PER_STEP


def medium_of(tag):
    """-> (spread mm/s, decay s, reach mm) recovered from the run's saved x."""
    z = np.load(os.path.join(RESULTS, f"xspec_{tag}.npz"), allow_pickle=True)
    x, save = z["x"], int(z["save"])
    damp = 10.0 ** float(x[0])
    frame_s = timescale.TR / 4.0
    spread = MM_PER_STEP * save / frame_s
    decay = frame_s / (damp * save)
    return spread, decay, spread * decay, save


def in_band(X, frame_s, lo=0.01, hi=0.08):
    """Fraction of the realisation's power inside the passband."""
    Xc = np.asarray(X, np.float64)
    Xc = Xc - Xc.mean(0, keepdims=True)
    P = (np.abs(np.fft.rfft(Xc, axis=0)) ** 2).mean(1)
    f = np.fft.rfftfreq(Xc.shape[0], frame_s)
    return float(P[(f >= lo) & (f <= hi)].sum() / max(P[1:].sum(), 1e-300))


def corr_length(X, v, d, iu, edges):
    """Distance at which mean inter-vertex temporal correlation falls to 0.5, in mm."""
    Z = X[:, v].T.astype(np.float64)
    Z -= Z.mean(1, keepdims=True)
    Z /= np.maximum(Z.std(1, keepdims=True), 1e-30)
    r = ((Z @ Z.T) / Z.shape[1])[iu]
    xs, ys = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (d >= a) & (d < b)
        if m.sum() > 50:
            xs.append(0.5 * (a + b)); ys.append(float(r[m].mean()))
    xs, ys = np.array(xs), np.array(ys)
    k = np.flatnonzero(ys < 0.5)
    if not len(k) or k[0] == 0:
        return np.nan
    j = k[0]
    return float(xs[j-1] + (ys[j-1] - 0.5) * (xs[j] - xs[j-1]) / max(ys[j-1] - ys[j], 1e-30))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tags", default="grid_s*_d*",
                    help="comma-separated globs over results/frames_<tag>.npy; a sweep "
                         "run in several batches has several prefixes")
    ap.add_argument("--nvert", type=int, default=900)
    ap.add_argument("--empirical", type=int, default=20,
                    help="subjects for the empirical row (0 to skip)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    from mesh_cache import load_cortex
    import fc_score, units
    c = load_cortex("fsaverage5", verbose=False)
    t = fc_score.default_target(c, verbose=False)
    ER = fc_score.FCTarget.effective_rank
    rng = np.random.default_rng(a.seed)
    v = np.sort(rng.choice(t.nV, a.nvert, replace=False))
    D = units.vertex_geodesic(c, t.cols[v])[:, t.cols[v]]
    iu = np.triu_indices(len(v), 1)
    d = D[iu]
    edges = np.array([0, 5, 10, 15, 20, 30, 40, 60, 80, 120])

    tags = sorted({os.path.basename(p)[len("frames_"):-len(".npy")]
                   for g in a.tags.split(",")
                   for p in glob.glob(os.path.join(RESULTS, f"frames_{g.strip()}.npy"))})
    if not tags:
        raise SystemExit(f"  no runs matching frames_{a.tags}.npy")
    print(f"  {len(tags)} runs, target {os.path.basename(t.fc_path)}\n")
    print(f"  {'tag':<20s} {'spread':>7s} {'decay':>7s} {'reach':>7s} {'sim':>8s} "
          f"{'rank':>6s} {'r50 mm':>7s} {'in band':>8s} {'bp':>3s}")
    rows = []
    for tag in tags:
        try:
            spread, decay, reach, save = medium_of(tag)
        except FileNotFoundError:
            continue
        F = np.asarray(np.load(os.path.join(RESULTS, f"frames_{tag}.npy"),
                               mmap_mode="r")[t.burn:, t.cols], np.float32)
        sim = float(t.score(np.load(os.path.join(RESULTS, f"frames_{tag}.npy"),
                                    mmap_mode="r")))
        ib = in_band(F, timescale.TR / 4.0)
        rows.append((tag, spread, decay, reach, sim, ER(F),
                     corr_length(F, v, d, iu, edges), ib, ib > 0.5))
        print(f"  {tag:<20s} {spread:>7.2f} {decay:>7.1f} {reach:>7.1f} {sim:>+8.4f} "
              f"{rows[-1][5]:>6.1f} {rows[-1][6]:>7.1f} {ib:>7.1%} "
              f"{'yes' if rows[-1][8] else 'NO':>3s}")

    rows = [r for r in rows if r[8]] or rows          # rank among bandpassed runs only
    if rows:
        best = max(rows, key=lambda r: r[4])
        print(f"\n  highest sim: {best[0]}  spread {best[1]:.2f} mm/s, decay {best[2]:.1f} s, "
              f"reach {best[3]:.1f} mm  ->  {best[4]:+.4f}")
        sp = sorted({round(r[1], 2) for r in rows})
        dc = sorted({round(r[2], 1) for r in rows})
        if len(sp) > 1 and len(dc) > 1:
            print(f"\n  sim over the grid (rows spread mm/s, cols decay s):")
            print("        " + "".join(f"{x:>9.1f}" for x in dc))
            for s in sp:
                cells = []
                for D_ in dc:
                    m = [r for r in rows if round(r[1], 2) == s and round(r[2], 1) == D_]
                    cells.append(f"{m[0][4]:>+9.4f}" if m else f"{'-':>9s}")
                print(f"  {s:>6.2f}" + "".join(cells))

    if a.empirical:
        import json, rbc
        subs = json.load(open(os.path.join(
            CACHE, "rbc_cohort_100_seed0.json")))["subjects"][:a.empirical]
        runs = rbc.cohort_runs(subs, specs=(("rest", "645"),))[("rest", "645")]
        er, cl = [], []
        for r in runs:
            X, _, _ = rbc.load(r, verbose=False)
            Xt = X[t.vertices].T
            er.append(ER(Xt)); cl.append(corr_length(Xt, v, d, iu, edges))
        print(f"\n  {'empirical (RBC rest)':<20s} {'-':>7s} {'-':>7s} {'-':>7s} {'-':>8s} "
              f"{np.mean(er):>6.1f} {np.nanmean(cl):>7.1f}   "
              f"({a.empirical} subjects, already 0.01-0.08 Hz)")


if __name__ == "__main__":
    main()
