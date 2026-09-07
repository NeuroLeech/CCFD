"""Does the solved input actually differ between CHECKER and FIXATION, and does the field?

The per-parcel power table asked a question the driven set cannot answer. Only V1 and PIT
are driven here, so V2, V3, V4, LO, MT have no input channel at all - and those are where
the empirical checkerboard response lives. And collapsing V1's seven pieces into one
number throws away the thing the seven pieces exist for: they were cut apart to let the
input make differently-shaped waves over time, so the question is whether the seven
covary with each other the same way under ON and under OFF.

Three readouts, all from the solves already on disk. No new fits, no new simulation.

  1. THE 7x7. S(f) IS the covariance between the piece input time courses, so the zero-lag
     covariance between two pieces is the bin-weighted integral of 2*Re(S_jk), and its
     normalised form is a correlation. Twenty-one numbers per arm. At zero lag only, two
     inputs can be perfectly locked at a 3 s offset and read as uncorrelated - which is
     exactly what a wave-making input would do - so the same integral is taken at several
     lags, with cos and sin weights.

     The sign convention on that sin is the one thing here with a time direction, and
     PLAN section 6 records an inverted conclusion from getting it wrong. So the analytic
     correlation is checked against one drawn from realised input time courses before
     anything is read off it.

  2. THE FIELD, VERTEXWISE. C = sum_f w_f 2 Re(H S H^H) with H shared across every arm,
     so the predicted FC of each arm is available anywhere on the sheet. Compared on the
     1,000 vertices the solves were fitted on and on 1,000 the fit never saw.

     S is trace-normalised inside the solve, so two arms' C differ by an arbitrary scale.
     Everything is compared as a correlation matrix, never as raw covariance.

  3. THE WIDER VISUAL CORTEX. The same C on V2/V3/V4/LO/MT and the rest of the visual
     territory, none of which is driven - so this is purely what arrives there. Two
     readouts: whether the FC pattern differs, and whether the field's variance is
     distributed differently across those vertices.

  python analysis/onoff_fields.py --a check_on_out --b check_off_out
"""
import _path  # noqa: F401  - puts the sibling code folders on sys.path
import os, argparse
import numpy as np

from paths import RESULTS
import xspec

# The visual territory, minus V1 (driven) - PIT is driven too and is marked where it
# appears. These are Glasser ids on the fsaverage5 annotation.
WIDER_VISUAL = ("V2", "V3", "V4", "V6", "V8", "V3A", "V3B", "V3CD", "V4t", "V6A", "V7",
                "VVC", "VMV1", "VMV2", "VMV3", "FFC", "LO1", "LO2", "LO3", "MT", "MST",
                "FST", "PH", "ProS", "DVT", "PIT")


def load(tag):
    return np.load(os.path.join(RESULTS, f"xspec_{tag}.npz"), allow_pickle=True)


def band_mask(z, band):
    pad = int(z["pad"]); fs = float(z["frame_s"])
    f = np.asarray(z["idx"], float) / (pad * fs)
    bw = np.gradient(np.asarray(z["idx"], float))
    m = np.ones(len(f), bool) if band is None else (f >= band[0]) & (f <= band[1])
    return f, bw, m


def input_cov(z, lag_s=0.0, band=(0.01, 0.08)):
    """Lagged covariance between the piece INPUT time courses.

        Phi(tau) = sum_f w_f 2 [ cos(2 pi f tau) Re(S_f) + sin(2 pi f tau) Im(S_f) ]

    the same expression PLAN section 2 writes for the field's lagged covariance, applied
    to S rather than to H S H^H because S is itself a cross-spectrum."""
    S = np.asarray(z["S"])
    f, bw, m = band_mask(z, band)
    th = 2.0 * np.pi * f[m] * float(lag_s)
    W = (bw[m] * 2.0)[:, None, None]
    return (W * (np.cos(th)[:, None, None] * S[m].real
                 + np.sin(th)[:, None, None] * S[m].imag)).sum(0)


def to_corr(C, ref=None):
    """Normalise by the ZERO-LAG diagonals, which are the variances.

    A lagged covariance's own diagonal is the autocovariance at that lag, not a variance -
    it can pass through zero and go negative - so dividing by it produces nonsense. `ref`
    is the zero-lag matrix whose diagonal is the right normaliser."""
    d = np.sqrt(np.maximum(np.diag(C if ref is None else ref), 1e-30))
    return C / np.outer(d, d)


def realised_input_corr(z, lag_frames, nframes=14312, seeds=(0, 1, 2, 3)):
    """The same quantity from drawn input time courses, as a convention check."""
    out = []
    for s in seeds:
        A = xspec.realise(z["S"], z["idx"], nframes,
                          ref_frames=int(z["ref_frames"]), seed=s)
        A = A - A.mean(0)
        n = len(A) - abs(lag_frames)
        x = A[:n] if lag_frames >= 0 else A[-lag_frames:-lag_frames + n]
        y = A[lag_frames:lag_frames + n] if lag_frames >= 0 else A[:n]
        C = (x.T @ y) / n
        out.append(C)
    return np.mean(out, 0)


def rebuild_H(c, z, cols, workers=8):
    """H on `cols`, carrying the kernel, passband and segment weighting the solve used."""
    import bo_step, subparcels, units, timescale, bandpass
    x, save = z["x"], int(z["save"])
    labels, tags = z["labels"], [str(s) for s in z["tags"]]
    p, _, _ = bo_step.unpack(x, c)
    P = subparcels.taper_profiles(c, labels, len(tags))
    frame_s, pad = float(z["frame_s"]), int(z["pad"])
    resp = xspec.impulse_responses(c, list(range(len(P))), p,
                                   int(z["impulse_frames"]) * save, save,
                                   profiles=P, verbose=False, workers=workers)
    R = np.pad(resp, ((0, 0), (0, max(0, pad - resp.shape[1])), (0, 0)))
    kern = units.smoothing_kernel(timescale.bold_fwhm_frames(frame_s, verbose=False),
                                  verbose=False)
    H, w, idx = xspec.transfer(R, cols, int(z["nfreq"]), kernel=kern)
    ref = R.shape[1]
    bp = np.asarray(z["band"], float)
    if np.isfinite(bp).all():
        H = H * bandpass.transfer_response(idx, ref, frame_s, *bp)[:, None, None]
    if int(z["segment"]):
        H = H * bandpass.segment_response(idx, ref, frame_s,
                                          int(z["segment"]))[:, None, None]
    return H, w, idx


def field_cov(H, w, S):
    """C = sum_f w_f 2 Re(H_f S_f H_f^H), as two real gemms.

    Factoring S_f = L_f L_f^H and stacking B_f = sqrt(2 w_f) H_f L_f over frequency turns
    nf complex triple products into one pair of large real matrix multiplies - the same
    rearrangement xspec.solve uses, and the reason this is seconds rather than minutes."""
    nf, nV, K = H.shape
    Z = np.empty((nf * K, nV), np.complex128)
    for f in range(nf):
        ev, V = np.linalg.eigh(S[f])
        L = V * np.sqrt(np.maximum(ev, 0.0))[None, :]
        Z[f * K:(f + 1) * K] = np.sqrt(2.0 * w[f]) * (H[f] @ L).conj().T
    return (Z.real.T @ Z.real) + (Z.imag.T @ Z.imag)


def tri(C):
    iu = np.triu_indices(C.shape[0], 1)
    return C[iu]


def _pearson(x, y):
    x = np.asarray(x, float) - np.mean(x); y = np.asarray(y, float) - np.mean(y)
    d = np.linalg.norm(x) * np.linalg.norm(y)
    return float(x @ y / d) if d > 0 else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--a", default="ck_check_on_out")
    ap.add_argument("--b", default="ck_check_off_out")
    ap.add_argument("--extra", default="ck_check_on_in,ck_check_off_in,"
                                       "ck_rest_on_out,ck_rest_off_out",
                    help="further arms carried through the same tables")
    ap.add_argument("--band", default="0.01,0.08")
    ap.add_argument("--lags", default="-8,-4,-2,0,2,4,8", help="seconds")
    ap.add_argument("--nvert", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    band = tuple(float(v) for v in a.band.split(","))
    lags = [float(v) for v in a.lags.split(",")]

    from mesh_cache import load_cortex
    import fc_score
    c = load_cortex("fsaverage5", verbose=False)
    t = fc_score.default_target(c, verbose=False)
    names = [(s.decode() if isinstance(s, bytes) else str(s)
              ).removeprefix("L_").removesuffix("_ROI") for s in c.names]

    arms = [a.a, a.b] + [s for s in a.extra.split(",") if s]
    Z = {nm: load(nm) for nm in arms}
    zA, zB = Z[a.a], Z[a.b]
    tags = [str(s) for s in zA["tags"]]
    par = np.array([int(s.split("_")[0]) for s in tags])
    v1 = np.flatnonzero(par == 1)
    print(f"  A = {a.a},  B = {a.b}")
    print(f"  {len(tags)} pieces; V1 is {len(v1)}: {', '.join(tags[i] for i in v1)}")

    # ---- 1. the 7x7 within-V1 input structure --------------------------------------
    print(f"\n  (1) INPUT correlation among the {len(v1)} V1 pieces, in "
          f"{band[0]}-{band[1]} Hz")
    fs = float(zA["frame_s"])
    print(f"  {'lag s':>7s}  {'r(A tri, B tri)':>15s}  {'mean |A|':>9s} {'mean |B|':>9s}"
          f"  {'max |A-B|':>10s}  {'pair':>12s}")
    tri_store = {}
    zeroA, zeroB = input_cov(zA, 0.0, band), input_cov(zB, 0.0, band)
    for lg in lags:
        A7 = to_corr(input_cov(zA, lg, band), zeroA)[np.ix_(v1, v1)]
        B7 = to_corr(input_cov(zB, lg, band), zeroB)[np.ix_(v1, v1)]
        ta, tb = tri(A7), tri(B7)
        d = np.abs(A7 - B7); np.fill_diagonal(d, 0)
        j, k = np.unravel_index(int(np.argmax(d)), d.shape)
        tri_store[lg] = (ta, tb)
        print(f"  {lg:>7.1f}  {_pearson(ta, tb):>15.4f}  {np.abs(ta).mean():>9.4f} "
              f"{np.abs(tb).mean():>9.4f}  {d[j,k]:>10.4f}  "
              f"{tags[v1[j]]+'-'+tags[v1[k]]:>12s}")

    print(f"\n  the 21 pairs at lag 0:")
    ta, tb = tri_store[0.0]
    iu = np.triu_indices(len(v1), 1)
    for n, (j, k) in enumerate(zip(*iu)):
        print(f"    {tags[v1[j]]:>4s}-{tags[v1[k]]:<4s}  A {ta[n]:+.4f}  B {tb[n]:+.4f}"
              f"   d {ta[n]-tb[n]:+.4f}")

    # the convention check: the same thing from realised input time courses
    print(f"\n  convention check, analytic against realised input time courses "
          f"(all bins):")
    ref0 = input_cov(zA, 0.0, None)
    rea0 = realised_input_corr(zA, 0)
    for lg in (0.0, 2.0, 4.0):
        lf = int(round(lg / fs))
        an = to_corr(input_cov(zA, lg, None), ref0)[np.ix_(v1, v1)]
        re = to_corr(realised_input_corr(zA, lf), rea0)[np.ix_(v1, v1)]
        print(f"    lag {lg:+.1f}s ({lf:+d} frames): r = {_pearson(tri(an), tri(re)):.4f}"
              f"   max |diff| {np.abs(tri(an)-tri(re)).max():.4f}")

    # ---- 2 and 3. the field ---------------------------------------------------------
    rng = np.random.default_rng(a.seed)
    sub = np.asarray(zA["sub"])
    fresh = np.sort(rng.choice(np.setdiff1d(np.arange(t.nV), sub), a.nvert,
                               replace=False))
    lab = c.lab[t.cols]
    vis_ids = [i for i, n in enumerate(names) if n in WIDER_VISUAL]
    vis_all = np.flatnonzero(np.isin(lab, vis_ids))
    vis = np.sort(rng.choice(vis_all, min(a.nvert, len(vis_all)), replace=False))
    print(f"\n  (2,3) FIELD FC. vertex sets: fitted {len(sub)}, fresh {len(fresh)}, "
          f"wider visual {len(vis)} of {len(vis_all)} "
          f"({len([n for n in WIDER_VISUAL])} parcels, none driven except PIT)")

    out = {}
    for setname, cols in (("fitted", sub), ("fresh", fresh), ("visual", vis)):
        H, w, idx = rebuild_H(c, zA, t.cols[cols], workers=a.workers)
        C = {nm: field_cov(H, w, np.asarray(Z[nm]["S"])) for nm in arms}
        R = {nm: to_corr(C[nm]) for nm in arms}
        print(f"\n  --- {setname} ({len(cols)} vertices) ---")
        print(f"  FC pattern, correlation between arms' upper triangles:")
        print(f"    {a.a} vs {a.b}: {_pearson(tri(R[a.a]), tri(R[a.b])):.6f}")
        for nm in arms[2:]:
            print(f"    {a.a} vs {nm}: {_pearson(tri(R[a.a]), tri(R[nm])):.6f}")
        dv = {nm: np.maximum(np.diag(C[nm]), 1e-30) / np.maximum(np.diag(C[nm]), 1e-30).mean()
              for nm in arms}
        r = np.log2(dv[a.a] / dv[a.b])
        print(f"  per-vertex variance share, log2(A/B): mean {r.mean():+.4f}, "
              f"sd {r.std():.4f}, range {r.min():+.4f} to {r.max():+.4f}, "
              f"corr(A,B) {_pearson(dv[a.a], dv[a.b]):.4f}")
        if setname == "visual":
            for p in sorted(set(lab[cols].tolist())):
                m = lab[cols] == p
                if m.sum() >= 20:
                    print(f"    {names[p]:<6s} n={int(m.sum()):>4d}  "
                          f"var share A {dv[a.a][m].mean():.3f} B {dv[a.b][m].mean():.3f}"
                          f"  log2 {np.log2(dv[a.a][m].mean()/dv[a.b][m].mean()):+.4f}")
        out[setname] = dict(cols=cols, tri_r={nm: _pearson(tri(R[a.a]), tri(R[nm]))
                                             for nm in arms},
                            var_log2=r)
        del H, C, R

    p = os.path.join(RESULTS, f"onoff_fields_{a.a}_{a.b}.npz")
    np.savez(p, v1=v1, tags=np.array(tags, dtype=object),
             **{f"tri_{k2}_{s}": v for s in out for k2, v in
                [("var", out[s]["var_log2"]), ("cols", out[s]["cols"])]})
    print(f"\n  wrote {p}")


if __name__ == "__main__":
    main()
