"""FC within frequency bands: does a solve's FC match the data's in the SLOW band too?

A solve fitted to unfiltered FC matches one correlation matrix, and nothing stops it doing so
with fluctuations faster than BOLD keeps. The question is whether the match also holds band by
band, above all in the slow bands, where BOLD's signal is. Empirical band FC comes from
targets/rbc_nobp.py --build-bands (the same unfiltered runs, each filtered by bandpass.apply).
Each model draw is realised exactly as pool_curve does (Nyquist low-pass, the solve's own
smoothing, no passband), then filtered by the same bandpass.apply per band; the 2M-edge FC
vectors are pooled over draws and scored against every empirical band:

    rows     the model's observable: unfiltered, then each band
    columns  the empirical FC: unfiltered, then each band

The diagonal is the like-for-like match; an off-diagonal row shows which of the data's bands a
model band resembles.

FIRST RUN, 2026-10-09: linear fits to the unfiltered RBC target, 6 mm, reach 37 mm, 8 draws
of 577 s pooled. Like-for-like (model band vs data band), and the unfiltered model:

                  all    0.002-0.01  0.01-0.03  0.03-0.08  0.08-0.2  0.2-0.775
    1x  (25 s)   +0.769   +0.408     +0.334     +0.545     +0.571    +0.116
    3x  (8.3 s)  +0.748   +0.159     +0.239     +0.268     +0.458    +0.453
    5x  (5 s)    +0.700   +0.179     +0.187     +0.235     +0.305    +0.485
    10x (2.5 s)  +0.532   +0.266     +0.302     +0.345     +0.274    +0.519

The data's own bands agree with each other at 0.87-0.98 below 0.2 Hz. The models' do not: each
model's unfiltered score is carried by the band holding its power - 0.03-0.2 Hz at 1x, 0.2-0.775
Hz at 3x and 5x (whose top band scores +0.62/+0.65 against the data's 0.03-0.08 FC) - and the
slow bands score 0.16-0.43. Eight 577 s draws are ~1/20th of the data's 100 x 577 s, so the slow
model bands are the most attenuated by estimation noise; how much is not yet measured.

  python fit/band_fc.py nobp_sm6_s1 nobp_sm6_s3 --draws 8
"""
import _path  # noqa: F401
import os, glob, time, argparse
import numpy as np

from paths import RESULTS

BANDDIR = os.path.join(RESULTS, "fc", "nobp", "bands")
_B = {}


def band_targets(c):
    import fc_score
    out = []
    for p in sorted(glob.glob(os.path.join(BANDDIR, "*_band-*_spearmanfc.npy")),
                    key=lambda p: float(p.split("_band-")[1].split("-")[0])):
        lab = p.split("_band-")[1].split("Hz")[0]
        lo, hi = (float(v) for v in lab.split("-"))
        out.append(((lo, hi), fc_score.FCTarget(c, fc_path=p, centre="double", verbose=False)))
    return out


def _init(tag, seconds):
    import pool_curve
    pool_curve._init(tag, seconds)
    _B["bands"] = [b for b, _ in band_targets(pool_curve._W["c"])]


def _one(seed):
    import xspec, units, bandpass, pool_curve, surface_smooth
    import fluid as fl
    from fc_score import _rank_z
    W = pool_curve._W
    g, c, t = W["g"], W["c"], W["t"]
    nf = W["nframes"]
    A = xspec.realise(W["S"], W["idx"], nf, ref_frames=W["ref"], seed=seed, factors=W["fac"])
    Aser = np.repeat(A, g["save"], axis=0)[:nf * g["save"]] / g["save"]
    fr, _ = fl.run(c, xspec.ProfileDrive(c, g["P"], Aser, 2e-4), g["p"], nf * g["save"],
                   g["save"])
    X = units.smooth_frames(fr, W["kern"])
    del fr
    if np.isfinite(g["lo"]):
        X = bandpass.apply(X, g["frame_s"], g["lo"], g["hi"])
    if g.get("smooth_fwhm", 0.0) > 0:
        X = surface_smooth.smooth(X, surface_smooth.smoothing_matrix(c, g["smooth_fwhm"]))
    out = []
    for b in [None] + _B["bands"]:
        Y = X if b is None else bandpass.apply(X, g["frame_s"], b[0], b[1])
        Z, flat = _rank_z(np.ascontiguousarray(Y[t.burn:, t.cols].T), rank=True)
        out.append(np.asarray(t.model_edges(Z=Z, flat=flat)[0], np.float32))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("tags", nargs="+")
    ap.add_argument("--draws", type=int, default=8)
    ap.add_argument("--seconds", type=float, default=577.0)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    import multiprocessing as mp
    from interp_gap import medium_context
    for tag in a.tags:
        t0 = time.time()
        z = np.load(os.path.join(RESULTS, f"xspec_{tag}.npz"), allow_pickle=True)
        _, c, t, _ = medium_context(z)
        bt = band_targets(c)
        for _, tb in bt:
            if not (np.array_equal(tb.i, t.i) and np.array_equal(tb.j, t.j)):
                raise SystemExit("  band targets sample different edges from the full one")
        cols = [("all", t)] + [(f"{lo:g}-{hi:g}", tb) for (lo, hi), tb in bt]
        rows = ["all"] + [f"{lo:g}-{hi:g}" for (lo, hi), _ in bt]
        if tag == a.tags[0]:
            # the data against itself: how alike the bands' FC already are, the reference
            # every model row is read against
            print("\n  DATA: spearman between the empirical band FCs (double-centred edges)")
            print("    data \\ data    " + "".join(f"{lab:>12s}" for lab, _ in cols))
            for lab, ta in cols:
                print(f"    {lab:>13s}  " + "".join(f"{float(ta.y @ tb.y):+12.4f}"
                                                   for _, tb in cols), flush=True)
        esum = None
        ctx = mp.get_context("spawn")
        with ctx.Pool(a.workers, initializer=_init, initargs=(tag, a.seconds)) as pool:
            for d, es in enumerate(pool.imap(_one, range(1000, 1000 + a.draws))):
                esum = [e.astype(np.float64) for e in es] if esum is None else \
                    [s + e for s, e in zip(esum, es)]
        print(f"\n  {tag}: {a.draws} draws pooled, {a.seconds:.0f} s  [{time.time()-t0:.0f}s]")
        print("    model \\ data   " + "".join(f"{lab:>12s}" for lab, _ in cols))
        M = np.zeros((len(rows), len(cols)))
        for r, e in enumerate(esum):
            for k, (_, tb) in enumerate(cols):
                M[r, k] = float(tb._prep(e / a.draws) @ tb.y)
            print(f"    {rows[r]:>13s}  " + "".join(f"{v:+12.4f}" for v in M[r]), flush=True)
        np.savez(os.path.join(RESULTS, f"bandfc_{tag}.npz"), M=M, rows=rows,
                 cols=[lab for lab, _ in cols], draws=a.draws, seconds=a.seconds)


if __name__ == "__main__":
    main()
