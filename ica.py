"""Spatial ICA, empirical against model, decomposed by the same rule.

The gradient decomposition describes the connectome's smooth axes; ICA describes it as a
set of overlapping spatial modes instead, which is the decomposition resting-state work
actually names networks with. Doing it on both sides asks whether the model's field
carries the same modes, not merely the same edge ordering.

Group ICA is the standard two-stage reduction: each subject is reduced to its leading
temporal PCs, those are concatenated, the concatenation is reduced again, and FastICA
rotates the result. The alternative - concatenating 99 x 895 raw frames and decomposing
that - is the same estimate with a much larger intermediate, and the per-subject stage is
what stops one long or one noisy run dominating the group.

Two things are reported, because they answer different questions:

  SUBSPACE. Principal angles between the empirical and model PCA subspaces. ICA's rotation
  within a subspace is not identified by the data, so a component-to-component correlation
  confounds "the model spans the same space" with "the rotation happened to line up".
  The angles measure the first without the second.

  COMPONENTS. Hungarian matching of model components to empirical ones by absolute spatial
  correlation, against the unmatched pairs as the control. Sign is arbitrary in ICA, hence
  the absolute value, and the sign is aligned before anything is drawn.

  python ica.py --nic 20
"""
import os, argparse
import numpy as np

from paths import CACHE, RESULTS


def zscore_rows(X, eps=1e-30):
    X = np.asarray(X, np.float32)
    X = X - X.mean(1, keepdims=True)
    return X / np.maximum(X.std(1, keepdims=True), eps)


def subject_reduction(t, n_pc, n_subjects=None, verbose=True):
    """-> (V, n_subjects * n_pc): each subject's leading temporal PCs, concatenated.

    Each subject contributes U*s rather than U, so a subject whose leading component is
    weak does not enter the group with the same weight as one whose is strong."""
    from fc_group_nki import subject_files, load_subject
    cache = os.path.join(CACHE, f"ica_red_pc{n_pc}_{n_subjects or 'all'}.npy")
    if os.path.exists(cache):
        return np.load(cache)
    files = subject_files("left")[:n_subjects]
    out = []
    for i, p in enumerate(files):
        X = zscore_rows(load_subject(p)[t.vertices])
        U, s, _ = np.linalg.svd(X, full_matrices=False)
        out.append((U[:, :n_pc] * s[:n_pc]).astype(np.float32))
        if verbose and (i + 1) % 20 == 0:
            print(f"    {i+1}/{len(files)} subjects", flush=True)
    out = np.concatenate(out, axis=1)
    np.save(cache, out)
    return out


def decompose(X, n_ic, seed=0, verbose=True):
    """PCA to n_ic then FastICA. -> (maps (V, n_ic), PCA variance share, PCA scores).

    The third return is the PCA subspace expressed over VERTICES. That is the object the
    two sides have in common - `components_` lives in each side's own time basis, which
    is 2,970 group components on one side and 3,528 frames on the other and cannot be
    compared at all.

    FastICA is given the (V, n_ic) reduction with vertices as SAMPLES, which is what makes
    this SPATIAL ICA: the recovered sources are maps over the surface, not timecourses."""
    from sklearn.decomposition import PCA, FastICA
    pca = PCA(n_components=n_ic, svd_solver="randomized", random_state=seed)
    Y = pca.fit_transform(X)                                # (V, n_ic)
    ica = FastICA(n_components=n_ic, random_state=seed, max_iter=2000, tol=1e-4,
                  whiten="unit-variance")
    Sp = ica.fit_transform(Y)                               # (V, n_ic) spatial maps
    if verbose and ica.n_iter_ >= 2000:
        print("    FastICA hit its iteration cap; components are not converged")
    return Sp, pca.explained_variance_ratio_, Y


def principal_angles(A, B):
    """Principal angles (degrees) between the column spans of A and B."""
    Qa, _ = np.linalg.qr(A - A.mean(0, keepdims=True))
    Qb, _ = np.linalg.qr(B - B.mean(0, keepdims=True))
    s = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    return np.degrees(np.arccos(np.clip(s, -1.0, 1.0)))


def match(A, B):
    """Hungarian match of B's columns to A's by |correlation|. -> (order, r, signs)."""
    from scipy.optimize import linear_sum_assignment
    Az = (A - A.mean(0, keepdims=True)) / np.maximum(A.std(0, keepdims=True), 1e-30)
    Bz = (B - B.mean(0, keepdims=True)) / np.maximum(B.std(0, keepdims=True), 1e-30)
    R = (Az.T @ Bz) / A.shape[0]
    r, c = linear_sum_assignment(-np.abs(R))
    return c, R[r, c], np.sign(R[r, c]), R


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", default="pr_taper")
    ap.add_argument("--nic", type=int, default=20)
    ap.add_argument("--npc", type=int, default=30, help="per-subject temporal PCs")
    ap.add_argument("--subjects", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--draw", type=int, default=8, help="components drawn in the figure")
    a = ap.parse_args()

    from mesh_cache import load_cortex
    import fc_score
    c = load_cortex("fsaverage5", verbose=False)
    t = fc_score.default_target(c, verbose=False)

    print(f"  empirical: reducing each subject to {a.npc} temporal PCs")
    E = subject_reduction(t, a.npc, a.subjects)
    print(f"    concatenated {E.shape[1]} components over {E.shape[0]} vertices")
    Es, Evar, Epc = decompose(E, a.nic, a.seed)

    F = np.load(os.path.join(RESULTS, f"frames_{a.tag}.npy"), mmap_mode="r")
    M = zscore_rows(np.asarray(F[t.burn:, t.cols], np.float32).T)
    print(f"  model {a.tag}: {M.shape[1]} frames over {M.shape[0]} vertices")
    Ms, Mvar, Mpc = decompose(M, a.nic, a.seed)

    print(f"\n  PCA before rotation: empirical {Evar.sum():.1%} of its variance in "
          f"{a.nic} components, model {Mvar.sum():.1%}")
    ang = principal_angles(Epc, Mpc)
    print(f"  principal angles between the two {a.nic}-dim subspaces (degrees):")
    print("    " + " ".join(f"{x:.0f}" for x in ang))
    print(f"    median {np.median(ang):.0f} deg; {int((ang < 45).sum())} of {a.nic} "
          f"below 45 deg")

    order, r, sign, R = match(Es, Ms)
    off = np.abs(R).copy()
    for i in range(a.nic):
        off[i, order[i]] = np.nan
    print(f"\n  components, model matched to empirical by |spatial r|")
    print(f"  {'emp IC':>7s} {'model':>6s} {'r':>8s}")
    for i in np.argsort(-np.abs(r)):
        print(f"  {i:>7d} {order[i]:>6d} {r[i]:>+8.3f}")
    print(f"  mean |r| {np.abs(r).mean():.3f}, best {np.abs(r).max():.3f}; "
          f"unmatched pairs {np.nanmean(off):.3f} (max {np.nanmax(off):.3f})")

    out = os.path.join(RESULTS, f"ica_{a.tag}_n{a.nic}.npz")
    np.savez(out, emp=Es, model=Ms, emp_var=Evar, model_var=Mvar, angles=ang,
             order=order, r=r, sign=sign, R=R, cols=t.cols)
    print(f"\n  wrote {out}")
    _plot(a.tag, a.nic, a.draw, c, t, Es, Ms, order, sign, r)


def _plot(tag, nic, ndraw, c, t, Es, Ms, order, sign, r):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from render_regimes import _proj
    from plot_fc_map import surface_row

    proj = _proj(c.V, c.F)
    best = np.argsort(-np.abs(r))[:ndraw]                  # the best-matched components
    rows = []
    for i in best:
        j = order[i]
        lim = float(np.percentile(np.abs(Es[:, i]), 99))
        rows.append((f"empirical IC {i}", Es[:, i], (-lim, lim)))
        lim = float(np.percentile(np.abs(Ms[:, j]), 99))
        rows.append((f"model {j}  r={r[i]:+.2f}", sign[i] * Ms[:, j], (-lim, lim)))
    fig = plt.figure(figsize=(3.4 * len(proj), 2.2 * len(rows)))
    gs = fig.add_gridspec(len(rows), len(proj), hspace=0.06, wspace=0.02)
    for ri, (lab, vals, lims) in enumerate(rows):
        surface_row(fig, gs, ri, proj, vals, c, t.cols, "RdBu_r", lims, lab)
    out = os.path.join(RESULTS, f"ica_{tag}_n{nic}.png")
    fig.savefig(out, dpi=105, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
