"""Co-activation patterns, empirical against model, clustered by the same rule.

FC is a second moment over the whole run, so a model that visits a few distinct spatial
patterns and a model that holds their average score identically against it. CAPs ask the
question at the level of single frames instead: take the frames where something is
actually happening, cluster them by spatial pattern, and report the patterns, how often
each occurs and how long it stays.

The point here is ALIGNMENT, so both sides go through one function. The empirical side is
99 NKI subjects on fsaverage5, z-scored per vertex within subject so no subject's scale
dominates; the model side is the realised field of one run, z-scored the same way. Frames
are selected by global amplitude (the standard CAP step - the top fraction by RMS over
vertices) and clustered by k-means on a vertex subsample, with each CAP then rebuilt on
every vertex by averaging its own frames. That is the pattern fc_states.py already uses
for windowed FC and it is here for the same reason: clustering 9k-dimensional frames
directly spends its time on vertices that carry no independent information.

Model CAPs are matched to empirical ones by absolute spatial correlation under a Hungarian
assignment, so the reported correspondence is a matching rather than an ordering - ICA and
k-means both return components in an arbitrary order, and pairing them by index would
measure the sort, not the maps.

  python caps.py --k 8
"""
import os, argparse
import numpy as np

from paths import CACHE, RESULTS

TR = 0.645


def select_frames(X, keep, rng=None):
    """-> (selected frames as (n, V), their indices) from a (V, T) z-scored run.

    Selection is by RMS over vertices, which is the amplitude criterion CAP analyses use:
    frames near the mean carry no pattern to cluster and including them pulls every
    centroid towards zero."""
    rms = np.sqrt((X ** 2).mean(0))
    thr = np.quantile(rms, 1.0 - keep)
    idx = np.flatnonzero(rms >= thr)
    return X[:, idx].T, idx


def zscore_rows(X, eps=1e-30):
    X = np.asarray(X, np.float32)
    X = X - X.mean(1, keepdims=True)
    return X / np.maximum(X.std(1, keepdims=True), eps)


def empirical_frames(t, keep, n_subjects=None, verbose=True):
    """Selected high-amplitude frames from every NKI subject, on the target's vertices."""
    from fc_group_nki import subject_files, load_subject
    cache = os.path.join(CACHE, f"caps_emp_keep{keep:.3f}_{n_subjects or 'all'}.npy")
    owner_cache = cache.replace(".npy", "_owner.npy")
    if os.path.exists(cache):
        return np.load(cache), np.load(owner_cache)
    files = subject_files("left")[:n_subjects]
    out, owner = [], []
    for i, p in enumerate(files):
        X = zscore_rows(load_subject(p)[t.vertices])
        sel, _ = select_frames(X, keep)
        out.append(sel.astype(np.float32)); owner.append(np.full(len(sel), i))
        if verbose and (i + 1) % 20 == 0:
            print(f"    {i+1}/{len(files)} subjects", flush=True)
    out, owner = np.concatenate(out), np.concatenate(owner)
    np.save(cache, out); np.save(owner_cache, owner)
    return out, owner


def cluster(frames, k, clust_v, seed=0):
    """k-means on a vertex subsample; centroids rebuilt on every vertex from the members.

    Returns (centroids (k, V), labels, occupancy). Run length is left to the caller,
    which is the only place that knows whether the frames are contiguous."""
    from sklearn.cluster import KMeans
    Z = frames[:, clust_v]
    Z = (Z - Z.mean(1, keepdims=True)) / np.maximum(Z.std(1, keepdims=True), 1e-30)
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Z)
    lab = km.labels_
    cent = np.stack([frames[lab == i].mean(0) for i in range(k)])
    occ = np.array([(lab == i).mean() for i in range(k)])
    return cent, lab, occ


def match(A, B):
    """Hungarian match of B's rows to A's by absolute correlation. -> (order, r, signs)."""
    from scipy.optimize import linear_sum_assignment
    Az = (A - A.mean(1, keepdims=True)) / np.maximum(A.std(1, keepdims=True), 1e-30)
    Bz = (B - B.mean(1, keepdims=True)) / np.maximum(B.std(1, keepdims=True), 1e-30)
    R = (Az @ Bz.T) / A.shape[1]
    r, c = linear_sum_assignment(-np.abs(R))
    return c, R[r, c], np.sign(R[r, c])


def dwell(lab, k):
    """Mean run length of each label over the RETAINED frames.

    Not a dwell time in seconds: frame selection removes the low-amplitude frames between
    events, so consecutive retained frames need not be consecutive in the run. It is
    comparable between the two sides because both are selected the same way, and that is
    all it is used for."""
    runs = [[] for _ in range(k)]
    cur, n = lab[0], 1
    for s in lab[1:]:
        if s == cur:
            n += 1
        else:
            runs[cur].append(n); cur, n = s, 1
    runs[cur].append(n)
    return np.array([np.mean(r) if r else np.nan for r in runs])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", default="pr_taper")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--keep", type=float, default=0.15,
                    help="fraction of frames kept, by RMS over vertices")
    ap.add_argument("--nclust", type=int, default=1500, help="vertices used for k-means")
    ap.add_argument("--subjects", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    from mesh_cache import load_cortex
    import fc_score, xspec
    c = load_cortex("fsaverage5", verbose=False)
    t = fc_score.default_target(c, verbose=False)
    clust_v = np.sort(np.random.default_rng(a.seed).choice(
        xspec.medoid_subset(t, 3000), a.nclust, replace=False))

    print(f"  empirical: loading NKI, keeping the top {a.keep:.0%} of frames by RMS")
    E, owner = empirical_frames(t, a.keep, a.subjects)
    print(f"    {len(E)} frames from {owner.max()+1} subjects, {E.shape[1]} vertices")

    F = np.load(os.path.join(RESULTS, f"frames_{a.tag}.npy"), mmap_mode="r")
    Xm = zscore_rows(np.asarray(F[t.burn:, t.cols], np.float32).T)
    M, midx = select_frames(Xm, a.keep)
    print(f"  model {a.tag}: {Xm.shape[1]} frames, {len(M)} kept")

    Ec, Elab, Eocc = cluster(E, a.k, clust_v, a.seed)
    Mc, Mlab, Mocc = cluster(M, a.k, clust_v, a.seed)

    # dwell: the empirical frames are not contiguous across subjects, so it is measured
    # within subject and averaged; the model is one continuous run
    ed = []
    for i in range(owner.max() + 1):
        m = owner == i
        if m.sum() > 1:
            ed.append(dwell(Elab[m], a.k))
    ed = np.nanmean(np.stack(ed), 0)
    md = dwell(Mlab, a.k)

    order, r, sign = match(Ec, Mc)
    print(f"\n  {a.k} CAPs each side, model matched to empirical by |spatial r|")
    print(f"  {'emp CAP':>8s} {'occ':>7s} {'runlen':>7s} | {'model':>6s} {'occ':>7s} "
          f"{'runlen':>7s} | {'r':>7s}")
    for i in range(a.k):
        j = order[i]
        print(f"  {i:>8d} {Eocc[i]:>7.3f} {ed[i]:>7.2f} | {j:>6d} {Mocc[j]:>7.3f} "
              f"{md[j]:>7.2f} | {r[i]:>+7.3f}")
    print(f"  mean |r| {np.abs(r).mean():.3f}; occupancy correlation "
          f"{np.corrcoef(Eocc, Mocc[order])[0,1]:+.3f}")

    # a matched pair is only meaningful against what an unmatched one would give
    Az = Ec - Ec.mean(1, keepdims=True)
    Rall = (Az / np.linalg.norm(Az, axis=1, keepdims=True)) @ \
           ((Mc - Mc.mean(1, keepdims=True)) /
            np.linalg.norm(Mc - Mc.mean(1, keepdims=True), axis=1, keepdims=True)).T
    off = np.abs(Rall).copy()
    for i in range(a.k):
        off[i, order[i]] = np.nan
    print(f"  matched |r| {np.abs(r).mean():.3f} against unmatched pairs "
          f"{np.nanmean(off):.3f} (max unmatched {np.nanmax(off):.3f})")

    out = os.path.join(RESULTS, f"caps_{a.tag}_k{a.k}.npz")
    np.savez(out, emp=Ec, model=Mc, emp_occ=Eocc, model_occ=Mocc, emp_dwell=ed,
             model_dwell=md, order=order, r=r, sign=sign, cols=t.cols)
    print(f"\n  wrote {out}")
    _plot(a.tag, a.k, c, t, Ec, Mc, order, sign, r, Eocc, Mocc)


def _plot(tag, k, c, t, Ec, Mc, order, sign, r, Eocc, Mocc):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from render_regimes import _proj
    from plot_fc_map import surface_row

    proj = _proj(c.V, c.F)
    rows = []
    for i in range(k):
        j = order[i]
        lim = float(np.percentile(np.abs(Ec[i]), 99))
        rows.append((f"empirical CAP {i}\nocc {Eocc[i]:.2f}", Ec[i], (-lim, lim)))
        lim = float(np.percentile(np.abs(Mc[j]), 99))
        rows.append((f"model {j}  r={r[i]:+.2f}\nocc {Mocc[j]:.2f}",
                     sign[i] * Mc[j], (-lim, lim)))
    fig = plt.figure(figsize=(3.4 * len(proj), 2.2 * len(rows)))
    gs = fig.add_gridspec(len(rows), len(proj), hspace=0.06, wspace=0.02)
    for ri, (lab, vals, lims) in enumerate(rows):
        surface_row(fig, gs, ri, proj, vals, c, t.cols, "RdBu_r", lims, lab)
    out = os.path.join(RESULTS, f"caps_{tag}_k{k}.png")
    fig.savefig(out, dpi=105, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
