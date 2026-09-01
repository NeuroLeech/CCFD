"""MSC01: what does 10 sessions per subject buy over NKI's single 9.6-minute run?

NKI gives one 577 s scan per subject, so the only within-subject reliability available is a
split of that run: 0.446, implying a ceiling of 0.783 on fitting one subject. MSC01 has
10 sessions of ~30 min, so reliability can be estimated by holding out SESSIONS.

The original plan used MSC and was abandoned because "the FC values were really low" -
fc_group_nki's docstring records that a single MSC subject's long-range structure sits near
zero. That is measured here rather than recalled: FC is binned by geodesic distance and
compared against the NKI group on the same vertices.

Sessions are resampled fs_LR 32k -> fsaverage5 once and cached, since that is the expensive
step; everything after runs off the cache.

  python msc_reliability.py
"""
import os, argparse, time
import numpy as np

from paths import CACHE
from mesh_cache import load_cortex
import fc_score, xspec


def session_blocks(space="fsaverage5", subject="sub-MSC01", mask_kind="glasser",
                   smooth_fwhm=0.0, verbose=True):
    """Per-session rank-z timecourses on the masked vertices. -> (list of (V,T), vertices)

    `smooth_fwhm` is applied on the TARGET surface after resampling. fc_vertexwise's
    comment says "the MSC data already carries 6 mm FWHM from its own pipeline"; measured,
    it does not. Nearest-neighbour correlation on the native fs_LR mesh is 0.308 mean and
    0.090 median, against 0.910 / 0.926 for the NKI surfaces - MSC as distributed here is
    effectively unsmoothed. That assumption, not the resampling (whose mapping checks out
    to 0.05 degrees), is why the MSC FC came out near zero at every distance."""
    sm = f"_sm{smooth_fwhm:g}" if smooth_fwhm else ""
    cache = os.path.join(CACHE, f"msc_blocks_{subject}_{space}_{mask_kind}{sm}.npz")
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        if verbose:
            print(f"  loaded {cache}")
        return [z[f"b{i}"] for i in range(int(z["n"]))], z["vertices"]
    import shutil, tempfile
    import fc_vertexwise as fv
    if shutil.which("wb_command") is None:
        raise SystemExit("wb_command not on PATH; Connectome Workbench is needed to "
                         "resample fs_LR -> fsaverage5")
    files = fv.atlas_files(space)
    keep = fv.cortex_mask(space, mask_kind)
    paths = fv.session_paths(subject, None)
    blocks = []
    with tempfile.TemporaryDirectory() as tmp:
        for p in paths:
            t0 = time.time()
            X, nodes, nsurf, tr, tmask = fv.load_left(p)
            T = X.shape[1]
            if tmask is not None:
                X = X[:, tmask[:T]]
            dense = np.zeros((nsurf, X.shape[1]), np.float32)
            dense[nodes] = X
            Y = fv.resample(dense, space, files, tmp)
            if smooth_fwhm:
                Y = fv.smooth(Y, space, smooth_fwhm, keep, tmp)
            Y = Y[keep]
            Z, bad = fv.rank_z(Y)
            blocks.append(Z.astype(np.float32))
            if verbose:
                print(f"    {os.path.basename(p)[:30]}  {T} -> {Z.shape[1]} frames "
                      f"[{time.time()-t0:.0f}s]", flush=True)
    var = sum((b ** 2).sum(1) for b in blocks)
    good = var > 0
    blocks = [b[good] for b in blocks]
    vertices = np.flatnonzero(keep)[good]
    np.savez(cache, n=len(blocks), vertices=vertices,
             **{f"b{i}": b for i, b in enumerate(blocks)})
    if verbose:
        print(f"  wrote {cache}")
    return blocks, vertices


def fc_of(Z, iu=None):
    Y = Z - Z.mean(1, keepdims=True)
    sd = Y.std(1, keepdims=True); sd[sd == 0] = 1.0
    Y = Y / sd
    C = (Y @ Y.T) / Y.shape[1]
    C = C - C.mean(0, keepdims=True) - C.mean(1, keepdims=True) + C.mean()
    np.fill_diagonal(C, 1.0)
    return C


def cor_off(A, B, iu):
    a, b = A[iu], B[iu]
    a = a - a.mean(); b = b - b.mean()
    return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30))


def eff_rank(ev):
    e = np.clip(np.asarray(ev, float), 0, None)
    return float(e.sum() ** 2 / max((e ** 2).sum(), 1e-300))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--nvert", type=int, default=800)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smooth", type=float, default=0.0,
                    help="FWHM mm applied on the target surface after resampling")
    a = ap.parse_args()

    c = load_cortex("fsaverage5", verbose=False)
    blocks, verts = session_blocks(smooth_fwhm=a.smooth)
    if a.smooth:
        print(f"  smoothing {a.smooth:g} mm FWHM on the target surface")
    nfr = [b.shape[1] for b in blocks]
    print(f"\n  {len(blocks)} sessions, {blocks[0].shape[0]} vertices, "
          f"frames per session {nfr} (total {sum(nfr)}, "
          f"{sum(nfr)*2.2/60:.0f} min at TR 2.2 s)")

    # common vertices with the NKI target, so the comparison is on one vertex set
    t = fc_score.default_target(c, metric="pearson", verbose=False)
    common = np.intersect1d(verts, t.vertices)
    rng = np.random.default_rng(a.seed)
    pick = np.sort(rng.choice(len(common), min(a.nvert, len(common)), replace=False))
    vsel = common[pick]
    im = np.searchsorted(verts, vsel)
    it = np.searchsorted(t.vertices, vsel)
    n = len(vsel); iu = np.triu_indices(n, 1)
    print(f"  {len(common)} vertices shared with the NKI target; using {n}")

    B = [b[im] for b in blocks]
    G = np.asarray(t.target_fc()[np.ix_(it, it)], np.float64)
    G = G - G.mean(0, keepdims=True) - G.mean(1, keepdims=True) + G.mean()

    old_ids = np.asarray(c.old); Ee = np.asarray(c.edges)
    pf = np.stack([old_ids[Ee[:, 0]], old_ids[Ee[:, 1]]], axis=1)
    pos = -np.ones(10242, np.int64); pos[verts] = np.arange(len(verts))
    ea, eb = pos[pf[:, 0]], pos[pf[:, 1]]
    okp = (ea >= 0) & (eb >= 0)
    W = np.concatenate(blocks, axis=1).astype(np.float64)
    W = W - W.mean(1, keepdims=True)
    W /= np.maximum(np.linalg.norm(W, axis=1, keepdims=True), 1e-12)
    nnr = np.einsum("ij,ij->i", W[ea[okp]], W[eb[okp]])
    print(f"  nearest-neighbour correlation on the mesh: {nnr.mean():+.4f} "
          f"(NKI as shipped: +0.893)")
    del W

    Call = fc_of(np.concatenate(B, axis=1))
    print(f"\n  MSC01 all sessions: eff rank "
          f"{eff_rank(np.abs(np.linalg.eigvalsh(Call))):.1f}, "
          f"vs NKI group {cor_off(Call, G, iu):+.4f}")

    # session-wise reliability: every 5 v 5 split, plus odd/even
    print(f"\n  {'split':<18s} {'frames':>14s} {'reliability':>12s} {'ceiling':>8s}")
    rels = []
    for nm, idx in (("odd vs even", (list(range(0, 10, 2)), list(range(1, 10, 2)))),
                    ("first vs last", (list(range(5)), list(range(5, 10))))):
        A_ = np.concatenate([B[i] for i in idx[0]], axis=1)
        Bb = np.concatenate([B[i] for i in idx[1]], axis=1)
        r = cor_off(fc_of(A_), fc_of(Bb), iu)
        rel = 2 * r / (1 + r)
        rels.append(r)
        print(f"  {nm:<18s} {A_.shape[1]:>6d}/{Bb.shape[1]:<7d} {r:>12.4f} "
              f"{np.sqrt(max(rel,0)):>8.4f}")
    # single session against the other nine, for the short-scan comparison
    r1 = []
    for i in range(len(B)):
        rest = np.concatenate([B[j] for j in range(len(B)) if j != i], axis=1)
        r1.append(cor_off(fc_of(B[i]), fc_of(rest), iu))
    print(f"  {'1 session vs 9':<18s} {np.mean(nfr):>6.0f}/{sum(nfr)-np.mean(nfr):<7.0f} "
          f"{np.mean(r1):>12.4f} {'':>8s}   (mean over the 10 choices)")

    # is long-range FC really near zero?
    import units
    # vertex_geodesic works on the CUT mesh; vsel are FULL-mesh ids
    cols = t.cols[it]
    D = units.vertex_geodesic(c, cols)[:, cols]
    d = D[iu]
    print(f"\n  mean FC by geodesic distance, MSC01 (all sessions) against NKI group:")
    print(f"  {'mm':>12s} {'n':>8s} {'MSC01':>9s} {'NKI group':>10s}")
    for lo, hi in ((0, 10), (10, 20), (20, 40), (40, 60), (60, 100), (100, 250)):
        m = (d >= lo) & (d < hi)
        if m.sum() < 100:
            continue
        print(f"  {lo:5d}-{hi:<6d} {int(m.sum()):>8d} {Call[iu][m].mean():>+9.4f} "
              f"{G[iu][m].mean():>+10.4f}")

    print(f"\n  for comparison, NKI single subject: split-half 0.446, ceiling 0.783, "
          f"eff rank 21.1,\n  vs group 0.515 - from one 9.6 min run")


if __name__ == "__main__":
    main()
