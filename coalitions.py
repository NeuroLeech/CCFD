"""Read the solved cross-spectrum back as coalitions, and check it over longer runs.

S(f) is Hermitian PSD, so at each frequency it factors as sum_i lambda_i u_i u_i^H. Each
term is one coalition: |u_i| are the region amplitudes, arg(u_i) are their relative phases,
and a phase divided by 2 pi f is a time offset. That is the object the thought experiment
started from - signed amplitudes with temporal offsets - recovered from the solution
rather than proposed.

  python coalitions.py                 # read back and re-run the BO winner
"""
import os, pickle, argparse
import numpy as np

from mesh_cache import load_cortex
from input2 import parcel_tapers
from fc_score import FCTarget
from paths import RESULTS
import xspec, ladder

BO = os.path.join(RESULTS, "bo_medium", "bo_medium.pkl")


def coalition_table(S, idx, nframes, regions, top_bands=3, verbose=True):
    """Leading coalitions in the strongest frequency bands."""
    power = np.array([np.trace(S[f]).real for f in range(len(S))])
    order = np.argsort(power)[::-1][:top_bands]
    rows = []
    for b in order:
        ev, U = np.linalg.eigh(0.5 * (S[b] + S[b].conj().T))
        o = np.argsort(ev)[::-1]
        ev, U = np.clip(ev[o], 0, None), U[:, o]
        share = ev / max(ev.sum(), 1e-30)
        f_cycles = idx[b] / nframes                      # cycles per saved frame
        for i in range(2):
            u = U[:, i]
            phase = np.angle(u * np.exp(-1j * np.angle(u[np.argmax(np.abs(u))])))
            offset = phase / (2 * np.pi * max(f_cycles, 1e-30))     # in frames
            rows.append(dict(band=int(idx[b]), period=1.0 / max(f_cycles, 1e-30),
                             mode=i, share=float(share[i]), amp=np.abs(u),
                             offset=offset, phase=phase, f_cycles=float(f_cycles),
                             power=float(power[b])))
    if verbose:
        for r in rows:
            print(f"\n  band {r['band']:3d} (period {r['period']:6.1f} frames), "
                  f"mode {r['mode']}: {100*r['share']:.0f}% of that band's power")
            o = np.argsort(r["amp"])[::-1][:8]
            print("    region   amp   offset(frames)")
            for k in o:
                print(f"    {str(regions[k]):>6s} {r['amp'][k]:5.2f} "
                      f"{r['offset'][k]:+9.1f}")
    return rows


def geometry_test(rows, D, n_show=2, label="piece"):
    """Do the coalitions respect cortical geometry?

    For each pair of driven regions, similarity of amplitude and similarity of time
    offset against -geodesic distance. A positive offset correlation means nearby regions
    fire closer together in time, which is what a travelling disturbance would look like;
    an amplitude correlation would instead mean nearby regions simply share loudness.

    An offset is a phase divided by a frequency, so it only means anything modulo the
    period: at a 16-frame period, +7.4 and -7.4 frames are 1.2 frames apart, not 14.8.
    The lag between two regions is therefore the WRAPPED phase difference, which is what
    is used here."""
    iu = np.triu_indices(D.shape[0], 1)
    for r in rows[:n_show]:
        sim_amp = -np.abs(r["amp"][:, None] - r["amp"][None, :])[iu]
        dphi = np.angle(np.exp(1j * (r["phase"][:, None] - r["phase"][None, :])))
        sim_off = -np.abs(dphi / (2 * np.pi * max(r["f_cycles"], 1e-30)))[iu]
        print(f"  band {r['band']:3d} mode {r['mode']} "
              f"({100*r['share']:2.0f}% of the band) vs geodesic distance over "
              f"{D.shape[0]} {label}s: amplitude r = "
              f"{np.corrcoef(sim_amp, -D[iu])[0,1]:+.2f}, "
              f"offset r = {np.corrcoef(sim_off, -D[iu])[0,1]:+.2f}")


def from_npz(path, nframes, top_bands, verbose=True):
    """Read back a solution saved by best_fit.py and run the same geometry test on it.

    best_fit saves the piece labels alongside S, so the regions here are sub-parcel
    pieces and the distances have to be measured between piece centroids rather than
    parcel centroids."""
    import ladder as ld
    z = np.load(path, allow_pickle=True)
    S, idx, labels = z["S"], z["idx"], z["labels"]
    tags = [str(t) for t in z["tags"]]
    cortex = load_cortex("fsaverage5", verbose=False)
    print(f"{os.path.basename(path)}: {S.shape[1]} pieces, {S.shape[0]} frequencies, "
          f"solved on a {nframes}-frame window")
    ranks = [float((lambda e: e.sum() ** 2 / max((e ** 2).sum(), 1e-30))
                   (np.clip(np.linalg.eigvalsh(S[f]), 0, None))) for f in range(len(S))]
    power = np.array([np.trace(S[f]).real for f in range(len(S))])
    print(f"  cross-spectrum: power spread over "
          f"{power.sum()**2/(power**2).sum():.1f} of {len(power)} bands, "
          f"rank per band min {min(ranks):.1f} median {np.median(ranks):.1f} "
          f"max {max(ranks):.1f} of {S.shape[1]}")
    rows = coalition_table(S, idx, nframes, tags, top_bands=top_bands, verbose=verbose)
    D, _ = ld.label_geodesic(cortex, labels, S.shape[1])
    print()
    geometry_test(rows, D, n_show=len(rows))
    return rows, D


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--npz", default=None,
                    help="a solution saved by best_fit.py (results/xspec_<tag>.npz); "
                         "reads it back and stops after the geometry test")
    ap.add_argument("--nframes", type=int, default=1120,
                    help="window S was solved on, for turning phases into frames")
    ap.add_argument("--top-bands", type=int, default=3, dest="top_bands")
    ap.add_argument("--pkl", default=BO)
    ap.add_argument("--frames", nargs="*", type=int, default=[280, 560, 1120])
    ap.add_argument("--draws", type=int, default=3)
    a = ap.parse_args()
    if a.npz:
        from_npz(a.npz, a.nframes, a.top_bands)
        return

    d = pickle.load(open(a.pkl, "rb"))
    best = d["best"]
    S, idx, x = best["S"], best["idx"], best["x"]
    p = xspec.medium(10.0 ** x[2], 10.0 ** x[0], 10.0 ** x[1])
    print(f"winner: c0 {10**x[0]:.3f}, Ld {10**x[1]:.2f}, sig0 {10**x[2]:.3e}  "
          f"(BO sim {best['sim']:+.4f})")

    cortex = load_cortex("fsaverage5", verbose=False)
    target = FCTarget(cortex, verbose=False)

    # ---- step 2: what the solution says the input is
    ranks = [float((lambda e: e.sum() ** 2 / max((e ** 2).sum(), 1e-30))
                   (np.clip(np.linalg.eigvalsh(S[f]), 0, None))) for f in range(len(S))]
    power = np.array([np.trace(S[f]).real for f in range(len(S))])
    print(f"\n  cross-spectrum: power spread over "
          f"{power.sum()**2/(power**2).sum():.1f} of {len(power)} bands, "
          f"rank per band min {min(ranks):.1f} median {np.median(ranks):.1f} "
          f"max {max(ranks):.1f} of {S.shape[1]}")
    rows = coalition_table(S, idx, 280, xspec.REGIONS)

    D, _ = ladder.parcel_geodesic(cortex, xspec.REGIONS, verbose=False)
    print()
    geometry_test(rows, D, label="parcel")

    # ---- step 3: longer runs
    print(f"\n  realisation length (each {a.draws} draws):")
    for nf in a.frames:
        sims, gaps, ranks_ = [], [], []
        for s in range(a.draws):
            A = xspec.realise(S, idx, nf, ref_frames=280, seed=100 + s)
            r = xspec.score_realisation(cortex, target, p, A)
            sims.append(r["sim"]); gaps.append(r["gap"]); ranks_.append(r["rank"])
            if nf == max(a.frames) and s == 0:
                np.save(os.path.join(RESULTS, "frames_xspec_best.npy"), r["frames"])
                np.save(os.path.join(RESULTS, "drive_xspec_best.npy"), r["drive"].Aser)
        print(f"    {nf:5d} frames ({nf*xspec.SAVE:6d} steps):  sim {np.mean(sims):+.4f} "
              f"+- {np.std(sims):.4f}   gap {np.mean(gaps):.3f}   "
              f"rank {np.mean(ranks_):5.1f}")


if __name__ == "__main__":
    main()
