"""The group checkerboard response: what a KNOWN input actually does to the cortex.

Every other target here is a second moment. Resting FC says which vertices covary, not
what any particular input produces, which is why the input had to be solved for and why
the solved one turned out to be 47 largely cancelling sources. The checkerboard is the
other kind of measurement: the input is known, timed, and the same for every subject, so
the response can be compared against a model driven with that same input rather than
against a covariance.

RBC ships it at acq-645 - the same acquisition, TR and preprocessing as the rest runs the
FC target is built from - so nothing about the clock, the resampling or the passband has
to be re-derived. 239 frames, 6 blocks of 20 s alternating fixation and checkerboard.

Two things are built and cached:

  EVOKED    per vertex, the epoch-average around CHECKER onset, group mean over subjects.
            Epochs are z-scored per vertex WITHIN subject first, so a subject with large
            BOLD amplitude does not dominate the average.
  RESPONSE  per vertex, the correlation with the unconvolved boxcar at a common lag. No
            HRF is assumed: the model has its own path from drive to observable and
            imposing a canonical HRF here would put a second, different haemodynamic
            assumption in front of it. The lag is chosen once, on the group mean, rather
            than per subject - a per-subject argmax would bias every vertex upward.

  python checkerboard.py --subjects 100
"""
import os, json, argparse
import numpy as np

from paths import CACHE, RESULTS
import rbc

TR = 0.645
BLOCK_S = 20.0


def group_response(t, subs, lag_grid=(0, 16), on="CHECKER", verbose=True):
    """-> dict with the evoked epochs, the boxcar response map, and the lag used."""
    runs = rbc.cohort_runs(subs, specs=(("CHECKERBOARD", "645"),))[("CHECKERBOARD", "645")]
    if verbose:
        print(f"  {len(runs)} checkerboard runs")
    ep_len = int(round(2 * BLOCK_S / TR))            # one full on/off cycle
    Z, B, ev0 = [], [], None
    for i, r in enumerate(runs):
        X, ok, tr = rbc.load(r, verbose=False)
        x = X[t.vertices].astype(np.float64)
        x -= x.mean(1, keepdims=True)
        x /= np.maximum(x.std(1, keepdims=True), 1e-12)
        ev = rbc.events(r)
        if ev0 is None:
            ev0 = ev
        onsets = [o for o, d, k in ev if k == on]
        eps = [x[:, s:s + ep_len] for s in (int(round(o / tr)) for o in onsets)
               if s + ep_len <= x.shape[1]]
        if not eps:
            continue
        Z.append(np.mean(eps, 0))
        B.append(rbc.boxcar(r, x.shape[1], tr, on=on))
        if verbose and (i + 1) % 20 == 0:
            print(f"    {i+1}/{len(runs)} subjects", flush=True)
        del X, x
    evoked = np.mean(Z, 0)                            # (nV, ep_len)

    # one lag for everybody, picked on the group mean
    best, rmap = None, None
    for s in range(*lag_grid):
        m = []
        for k, (r, b) in enumerate(zip(runs, B)):
            X, _, _ = rbc.load(r, verbose=False)
            x = X[t.vertices].astype(np.float64)
            x -= x.mean(1, keepdims=True)
            bb = np.roll(b, s).astype(np.float64); bb[:s] = 0.0
            bb -= bb.mean()
            den = np.linalg.norm(x, axis=1) * np.linalg.norm(bb)
            m.append(np.where(den > 0, (x @ bb) / np.maximum(den, 1e-30), 0.0))
            del X, x
        g = np.mean(m, 0)
        if best is None or g.max() > best[1]:
            best, rmap = (s, g.max()), g
        if verbose:
            print(f"    lag {s*TR:5.2f}s  peak r {g.max():+.4f}", flush=True)
    return dict(evoked=evoked, response=rmap, lag_frames=best[0], lag_s=best[0] * TR,
                ep_len=ep_len, tr=TR, n=len(Z), events=ev0)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--subjects", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lag-max", type=int, default=16, dest="lag_max")
    a = ap.parse_args()

    from mesh_cache import load_cortex
    import fc_score
    c = load_cortex("fsaverage5", verbose=False)
    t = fc_score.default_target(c, verbose=False)
    subs = json.load(open(os.path.join(
        CACHE, f"rbc_cohort_100_seed{a.seed}.json")))["subjects"][:a.subjects]

    cache = os.path.join(CACHE, f"checkerboard_group_{len(subs)}_{t.nV}.npz")
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        r = {k: z[k] for k in z.files}
        print(f"  loaded {cache}")
    else:
        r = group_response(t, subs, lag_grid=(0, a.lag_max))
        np.savez(cache, **{k: v for k, v in r.items() if k != "events"})
        print(f"  wrote {cache}")

    resp, evoked = r["response"], r["evoked"]
    print(f"\n  {int(r['n'])} subjects, lag {float(r['lag_s']):.2f} s, "
          f"epoch {int(r['ep_len'])} frames ({int(r['ep_len'])*TR:.0f} s)")
    print(f"  response map: peak {resp.max():+.4f}, min {resp.min():+.4f}, "
          f"mean {resp.mean():+.4f}")

    lab = c.lab[t.cols]
    def nm(i):
        s = c.names[i]
        return (s.decode() if isinstance(s, bytes) else str(s)
                ).removeprefix("L_").removesuffix("_ROI")
    sc = {int(p): float(resp[lab == p].mean())
          for p in np.unique(lab) if p >= 0 and (lab == p).sum() >= 20}
    top = sorted(sc.items(), key=lambda kv: -kv[1])
    print(f"\n  top 10 parcels: " + ", ".join(f"{nm(p)} {v:+.3f}" for p, v in top[:10]))
    print(f"  bottom 5:       " + ", ".join(f"{nm(p)} {v:+.3f}" for p, v in top[-5:]))
    for a_ in ("V1", "V2", "V3", "V4", "MT", "A1", "4"):
        h = [p for p in sc if nm(p) == a_]
        if h:
            rk = sorted(sc.values(), reverse=True).index(sc[h[0]]) + 1
            print(f"    {a_:<4s} {sc[h[0]]:+.3f}  rank {rk}/{len(sc)}")

    # how far the response reaches, as an evoked-amplitude profile from the V1 peak
    import units
    pk = int(np.argmax(resp))
    d = units.vertex_geodesic(c, [t.cols[pk]])[0][t.cols]
    print(f"\n  peak vertex sits in {nm(int(lab[pk]))}; response against distance from it")
    edges = np.array([0, 5, 10, 20, 30, 50, 80, 120])
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (d >= lo) & (d < hi)
        if m.sum() > 20:
            print(f"    {lo:>3d}-{hi:<3d} mm  n={int(m.sum()):>4d}  mean r {resp[m].mean():+.4f}")


if __name__ == "__main__":
    main()
