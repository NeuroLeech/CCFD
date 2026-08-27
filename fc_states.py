"""Recurrent FC states in the NKI group, from sliding-window connectivity.

Static FC is one number per edge for the whole run, so a model that switches between
regimes and a model that holds their average look identical to it. Finding the regimes
in the data is the first step to scoring them: window each subject's timeseries, cluster
the windows by their connectivity pattern, and report what the states are, how often each
occurs, how long it stays, and how it moves to the next.

Clustering runs on a small vertex set (cheap, and windowed FC on 9k vertices would be
enormous), but each state's FC is then rebuilt on the full solve vertex set by pooling
the windows assigned to it, so the states can be handed straight to the cross-spectrum
solve.

  python fc_states.py --k 5 --window 60 --stride 10
"""
import os, argparse
import numpy as np
from scipy.stats import rankdata

from paths import CACHE, RESULTS


def windowed_fc(X, window, stride, iu):
    """(nwin, nedges) upper-triangle FC of each sliding window of a (V, T) array."""
    V, T = X.shape
    out = []
    for a in range(0, T - window + 1, stride):
        W = X[:, a:a + window]
        Z = W - W.mean(1, keepdims=True)
        Z /= np.maximum(Z.std(1, keepdims=True), 1e-30)
        C = (Z @ Z.T) / window
        out.append(C[iu])
    return np.asarray(out, np.float32)


def build_states(k=5, window=60, stride=10, n_cluster_vertices=150, n_subjects=None,
                 seed=0, verbose=True):
    from mesh_cache import load_cortex
    from fc_score import FCTarget
    from fc_group_nki import subject_files, load_subject
    import xspec

    cache = os.path.join(CACHE, f"fc_states_k{k}_w{window}_s{stride}_"
                                f"{n_cluster_vertices}.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        return {kk: z[kk] for kk in z.files}

    cortex = load_cortex("fsaverage5", verbose=False)
    target = FCTarget(cortex, verbose=False)
    solve_v = xspec.medoid_subset(target, 1000)              # what the solve will use
    rng = np.random.default_rng(seed)
    clust_v = np.sort(rng.choice(solve_v, n_cluster_vertices, replace=False))
    iu = np.triu_indices(len(clust_v), 1)

    files = subject_files("left")[:n_subjects]
    verts_full = target.vertices[solve_v]                    # full-mesh ids
    verts_clu = target.vertices[clust_v]
    if verbose:
        print(f"  {len(files)} subjects, window {window} frames ({window*0.645:.0f} s), "
              f"stride {stride}, clustering on {len(clust_v)} vertices")

    W, owner = [], []
    for i, p in enumerate(files):
        Xs = load_subject(p)
        if Xs.shape[1] < window:
            continue
        W.append(windowed_fc(Xs[verts_clu], window, stride, iu))
        owner.append(np.full(len(W[-1]), i))
    W = np.concatenate(W); owner = np.concatenate(owner)
    if verbose:
        print(f"  {len(W)} windows total, {W.shape[1]} edges each")

    from sklearn.cluster import KMeans
    Wz = (W - W.mean(1, keepdims=True)) / np.maximum(W.std(1, keepdims=True), 1e-30)
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Wz)
    lab = km.labels_

    # rebuild each state's FC on the solve vertices by pooling its windows
    nS = len(solve_v)
    acc = np.zeros((k, nS, nS)); cnt = np.zeros(k)
    for i, p in enumerate(files):
        Xs = load_subject(p)[verts_full]
        mine = np.flatnonzero(owner == i)
        starts = np.arange(0, Xs.shape[1] - window + 1, stride)
        for j, a in zip(mine, starts):
            Wi = Xs[:, a:a + window]
            Z = Wi - Wi.mean(1, keepdims=True)
            Z /= np.maximum(Z.std(1, keepdims=True), 1e-30)
            acc[lab[j]] += (Z @ Z.T) / window
            cnt[lab[j]] += 1
    states = acc / np.maximum(cnt, 1)[:, None, None]

    # occupancy, dwell time, transitions - within subject, windows are consecutive
    occ = cnt / cnt.sum()
    dwell, trans = np.zeros(k), np.zeros((k, k))
    for i in range(len(files)):
        seq = lab[owner == i]
        runs, cur, n = [], seq[0], 1
        for s in seq[1:]:
            trans[cur, s] += 1
            if s == cur:
                n += 1
            else:
                runs.append((cur, n)); cur, n = s, 1
        runs.append((cur, n))
        for st, ln in runs:
            dwell[st] += ln
    counts = np.array([np.sum([1 for i in range(len(files))
                               for st, _ in _runs(lab[owner == i]) if st == s])
                       for s in range(k)])
    dwell = dwell / np.maximum(counts, 1) * stride * 0.645     # seconds
    trans = trans / np.maximum(trans.sum(1, keepdims=True), 1)

    out = dict(states=states, occupancy=occ, dwell=dwell, transitions=trans,
               solve_vertices=solve_v, cluster_vertices=clust_v, labels=lab,
               owner=owner, k=np.array(k), window=np.array(window),
               stride=np.array(stride))
    np.savez(cache, **out)
    return out


def _runs(seq):
    cur, n = seq[0], 1
    for s in seq[1:]:
        if s == cur:
            n += 1
        else:
            yield cur, n; cur, n = s, 1
    yield cur, n


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--window", type=int, default=60)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--subjects", type=int, default=None)
    a = ap.parse_args()

    from mesh_cache import load_cortex
    from fc_score import FCTarget
    st = build_states(a.k, a.window, a.stride, n_subjects=a.subjects)
    S, occ, dwell, tr = st["states"], st["occupancy"], st["dwell"], st["transitions"]
    k, n = len(S), S.shape[1]
    iu = np.triu_indices(n, 1)

    cortex = load_cortex("fsaverage5", verbose=False)
    t = FCTarget(cortex, verbose=False)
    static = np.asarray(t.target_fc()[np.ix_(st["solve_vertices"],
                                             st["solve_vertices"])], np.float64)
    print(f"\n  {k} states on {n} solve vertices")
    print(f"  {'state':>6s} {'occupancy':>10s} {'dwell (s)':>10s} {'r with static':>14s} "
          f"{'mean |edge|':>12s}")
    for i in range(k):
        print(f"  {i:6d} {occ[i]:10.3f} {dwell[i]:10.1f} "
              f"{np.corrcoef(S[i][iu], static[iu])[0,1]:14.3f} "
              f"{np.abs(S[i][iu]).mean():12.3f}")
    R = np.corrcoef(np.stack([S[i][iu] for i in range(k)]))
    print(f"\n  state-to-state FC correlation (how distinct they are):")
    for i in range(k):
        print("   " + " ".join(f"{R[i,j]:+6.2f}" for j in range(k)))
    print(f"\n  transition matrix (rows sum to 1):")
    for i in range(k):
        print("   " + " ".join(f"{tr[i,j]:6.3f}" for j in range(k)))
