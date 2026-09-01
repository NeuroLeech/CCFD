"""Where on the cortex does the fit fail, and is it always the same places?

The failure has been characterised as a set of numbers - accuracy correlates with
distance from the drive at only -0.175, the residual has participation rank 5.4, its
dominant mode loads 3.3x more on undriven vertices than driven - and never once drawn on
the surface. diag_distance and diag_edges produce scatter and line plots only. This is
the missing picture.

Four things, in order of how much they constrain what to do next.

ACCURACY. Per-vertex Spearman between the model's FC profile and the target's, over a
common random partner set so every vertex is scored against the same partners. Computed
for EVERY vertex rather than a subsample, because the point is a map.

STRETCH. The model over-correlates by a roughly constant 1.3-1.5x at every distance, which
is the signature of a field carrying its FC in too few spatial modes. Per vertex that is
the OLS slope of the model's profile on the target's: 1.0 is right, above 1.0 is
over-correlated. Note the row MEAN is not usable for this - double centring sets it to
zero by construction - so it has to be the slope.

RESIDUAL MODES. The residual eigendecomposes; its leading modes are where the miss lives.
They are found on a square block of `nvert` vertices and then extended to the whole sheet
by the residual's own action (mode = Res_rect @ u, the Nystrom extension), so the map is
dense rather than a scatter of sampled points.

REPRODUCIBILITY. All of the above, recomputed against group FC built from one half of the
subjects and then the other. A residual that reproduces across independent halves is
structure the model cannot reach; one that does not has already been claimed by the
reliability ceiling (+0.9679). This is what answers "is it always the same regions"
rather than illustrating it.

  python diag_maps.py --tag whiten
  python diag_maps.py --tag whiten --halves
"""
import os, argparse
import numpy as np
from scipy.stats import rankdata

from mesh_cache import load_cortex
from paths import RESULTS
import fc_score, subparcels


def model_fc_rect(t, Z, part):
    """Double-centred model FC, all vertices x `part`.

    The centring is a FULL-matrix operation - row means over all 9,217 vertices - so it
    cannot be done from the block alone. Same dot-product identity as
    fc_score.model_edges and best_fit.held_out_score; anything else would score an
    un-centred model against a centred target, which is the asymmetry f409535 fixed."""
    T, V = Z.shape[1], Z.shape[0]
    F = (Z @ Z[part].T) / T
    if t.centre == "double":
        ssum = Z.sum(0)
        diag = (Z * Z).sum(1) / T
        m = ((Z @ ssum) / T - diag) / (V - 1)
        grand = (float(ssum @ ssum) / T - float(diag.sum())) / (V * (V - 1))
        F = F - m[:, None] - m[part][None, :] + grand
    return F


def accuracy_and_slope(M, Tt, self_mask):
    """Per-vertex Spearman accuracy, amplitude ratio, and OLS slope.

    Three numbers because the obvious one is confounded. The OLS slope of model on target
    is cov/var, which equals r * sd(M)/sd(T) - so a slope below 1 can mean the model is
    too flat OR merely that it is inaccurate, and the two are not separable from it. The
    AMPLITUDE RATIO sd(M)/sd(T) is the clean version of "does the model over-correlate",
    and it is the quantity the distance-bin table in PLAN.md is really about."""
    n = M.shape[0]
    acc = np.empty(n)
    ratio = np.empty(n)
    slope = np.empty(n)
    for i in range(n):
        k = self_mask[i]
        mi, ti = M[i][k], Tt[i][k]
        rm, rt = rankdata(mi), rankdata(ti)
        rm = (rm - rm.mean()) / max(rm.std(), 1e-12)
        rt = (rt - rt.mean()) / max(rt.std(), 1e-12)
        acc[i] = float(rm @ rt / len(rm))
        sm, st = float(mi.std()), float(ti.std())
        ratio[i] = sm / max(st, 1e-30)
        tc = ti - ti.mean()
        slope[i] = float((tc @ (mi - mi.mean())) / max(float(tc @ tc), 1e-30))
    return acc, ratio, slope


def residual_modes(M, Tt, part, nmode=4):
    """Leading eigenmodes of the residual, extended to every vertex.

    The residual is defined on the square block (target minus its best scalar multiple of
    the model, both z-scored over the block's edges) and eigendecomposed there. Each
    eigenvector is then pushed out to all vertices by the residual's own rectangular
    action, which is the Nystrom extension and is what makes this a map rather than a
    scatter of `nvert` dots."""
    n = len(part)
    blockM, blockT = M[part], Tt[part]
    iu = np.triu_indices(n, 1)
    tv, cv = blockT[iu].astype(float), blockM[iu].astype(float)
    tv = (tv - tv.mean()) / tv.std()
    cv = (cv - cv.mean()) / cv.std()
    beta = float(tv @ cv) / len(tv)
    Res = np.zeros((n, n))
    Res[iu] = tv - beta * cv
    Res = Res + Res.T
    ev, V = np.linalg.eigh(Res)
    o = np.argsort(np.abs(ev))[::-1]
    ev, V = ev[o], V[:, o]
    e = ev ** 2
    prank = float(e.sum() ** 2 / max((e ** 2).sum(), 1e-300))

    # rectangular residual, all vertices x block, on the same scaling
    mu_t, sd_t = blockT.mean(), blockT.std()
    mu_m, sd_m = blockM.mean(), blockM.std()
    Rrect = ((Tt - mu_t) / sd_t) - beta * ((M - mu_m) / sd_m)
    modes = []
    for k in range(nmode):
        v = Rrect @ V[:, k]
        nrm = np.linalg.norm(v)
        modes.append(v / nrm if nrm > 0 else v)
    return np.stack(modes), ev, prank, beta


def parcel_table(cortex, cols, vals, k=12, label="value"):
    """Glasser parcels ranked by the mean of `vals` over their vertices."""
    lab = np.asarray(cortex.lab)[cols]
    out = []
    for p in np.unique(lab):
        if p == 0:
            continue
        m = lab == p
        if m.sum() < 5:
            continue
        nm = cortex.names[p].replace("_ROI", "") if p < len(cortex.names) else str(p)
        out.append((float(vals[m].mean()), int(m.sum()), nm))
    out.sort()
    print(f"\n  {label}: lowest {k} parcels")
    for v, n, nm in out[:k]:
        print(f"    {nm:<12s} {v:+.4f}  ({n} vertices)")
    print(f"  {label}: highest {k} parcels")
    for v, n, nm in out[-k:][::-1]:
        print(f"    {nm:<12s} {v:+.4f}  ({n} vertices)")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", default="whiten", help="a saved run, for its frames")
    ap.add_argument("--nvert", type=int, default=2000,
                    help="square block for the residual eigendecomposition")
    ap.add_argument("--npart", type=int, default=2000,
                    help="common partner set every vertex is scored against")
    ap.add_argument("--nmode", type=int, default=4)
    ap.add_argument("--split", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--halves", action="store_true",
                    help="recompute against each half of the subjects and report how "
                         "much of the failure map reproduces")
    a = ap.parse_args()

    c = load_cortex("fsaverage5", verbose=False)
    t = fc_score.default_target(c, verbose=True)
    labels, tags = subparcels.split_parcels(c, subparcels.SENSORY, a.split, verbose=False)
    driven = (labels >= 0)[t.cols]

    fp = os.path.join(RESULTS, f"frames_{a.tag}.npy")
    frames = np.asarray(np.load(fp, mmap_mode="r"))
    print(f"  {a.tag}: {frames.shape[0]} frames x {frames.shape[1]} vertices")
    Z, _ = t.model_z(frames)
    del frames

    rng = np.random.default_rng(a.seed)
    part = np.sort(rng.choice(t.nV, a.npart, replace=False))
    blk = np.sort(rng.choice(t.nV, a.nvert, replace=False))

    TFC = t.target_fc()
    Tt = np.asarray(TFC[:, part], np.float64)
    Tblk = np.asarray(TFC[:, blk], np.float64)      # rectangular, like the model side
    del TFC
    M = model_fc_rect(t, Z, part)

    # a vertex must not be scored against itself
    self_mask = np.ones((t.nV, len(part)), bool)
    pos = {int(v): i for i, v in enumerate(part)}
    for i in range(t.nV):
        j = pos.get(i)
        if j is not None:
            self_mask[i, j] = False

    print(f"  scoring {t.nV} vertices against {len(part)} common partners")
    acc, ratio, slope = accuracy_and_slope(M, Tt, self_mask)
    print(f"\n  accuracy: mean {acc.mean():+.4f}, sd {acc.std():.4f}, "
          f"range {acc.min():+.3f} to {acc.max():+.3f}")
    print(f"  driven vertices {acc[driven].mean():+.4f} vs undriven "
          f"{acc[~driven].mean():+.4f}")
    print(f"  amplitude ratio sd(model)/sd(target): mean {ratio.mean():.3f}, "
          f"median {np.median(ratio):.3f}, sd {ratio.std():.3f}   "
          f"(>1 = over-correlated)")
    print(f"  OLS slope (= ratio x accuracy, so confounded): "
          f"median {np.median(slope):.3f}")

    # residual modes: eigen-decomposed on the square block M[blk], Tblk[blk], then
    # extended to every vertex through the rectangular residual
    Mblk = model_fc_rect(t, Z, blk)
    modes, ev, prank, beta = residual_modes(Mblk, Tblk, blk, a.nmode)
    print(f"\n  model explains {beta:+.4f} of the target on the {len(blk)}-vertex block")
    print(f"  residual participation rank {prank:.1f} of {len(blk)}; "
          f"leading eigenvalues " + ", ".join(f"{v:+.1f}" for v in ev[:6]))
    for k in range(a.nmode):
        v = np.abs(modes[k])
        print(f"    mode {k+1}: |loading| driven {v[driven].mean():.4f}, "
              f"undriven {v[~driven].mean():.4f}  "
              f"(ratio {v[~driven].mean()/max(v[driven].mean(),1e-12):.2f})")

    parcel_table(c, t.cols, acc, label="accuracy")
    parcel_table(c, t.cols, ratio, k=8, label="amplitude ratio")

    np.savez(os.path.join(RESULTS, f"diag_maps_{a.tag}.npz"),
             acc=acc, ratio=ratio, slope=slope, modes=modes, ev=ev, part=part, blk=blk,
             cols=t.cols, driven=driven)

    _figure(c, t, a, acc, ratio, modes, ev, driven)

    if a.halves:
        _halves(c, t, a, Z, acc)


def _figure(c, t, a, acc, ratio, modes, ev, driven):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from render_regimes import _proj
    from plot_fc_map import surface_row

    proj = _proj(c.V, c.F)
    rows = [("accuracy", acc, "viridis",
             (float(np.percentile(acc, 2)), float(np.percentile(acc, 98)))),
            ("amplitude ratio\n(>1 = over-corr)", ratio, "RdBu_r",
             (2.0 - float(np.percentile(ratio, 98)), float(np.percentile(ratio, 98))))]
    for k in range(len(modes)):
        s = float(np.percentile(np.abs(modes[k]), 99))
        rows.append((f"residual mode {k+1}\n(eig {ev[k]:+.0f})", modes[k], "RdBu_r",
                     (-s, s)))

    fig = plt.figure(figsize=(3.6 * len(proj), 2.5 * len(rows)))
    gs = fig.add_gridspec(len(rows), len(proj), hspace=0.04, wspace=0.02)
    for r, (label, vals, cmap, lims) in enumerate(rows):
        surface_row(fig, gs, r, proj, vals, c, t.cols, cmap, lims, label)
    fig.suptitle(f"{a.tag}: where the sensory model fails", fontsize=10)
    p = os.path.join(RESULTS, f"diag_maps_{a.tag}.png")
    fig.savefig(p, dpi=135, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  wrote {p}")


def _halves(c, t, a, Z, acc_full):
    """Does the failure map reproduce across independent halves of the subjects?"""
    import holdout
    print(f"\n  --- reproducibility across subject halves ---")
    _, blocks, sub = holdout.half_targets(t, seed=a.seed, verbose=False)
    n = len(sub)
    self_mask = ~np.eye(n, dtype=bool)
    Msub = model_fc_rect(t, Z, sub)[sub]
    accs = []
    for name, B in zip(("half A", "half B"), blocks):
        B = np.asarray(B, np.float64)
        ac, _, _ = accuracy_and_slope(Msub, B, self_mask)
        accs.append(ac)
        print(f"    {name}: mean per-vertex accuracy {ac.mean():+.4f}")
    r = float(np.corrcoef(accs[0], accs[1])[0, 1])
    print(f"    accuracy map correlates {r:+.4f} between halves over {n} vertices")
    verdict = ("the same vertices fail in both halves, so this is structure the model "
               "cannot reach" if r > 0.5 else
               "the failure map does not reproduce across halves, so much of it is "
               "target noise the reliability ceiling has already claimed")
    print(f"    -> {verdict}")
    np.savez(os.path.join(RESULTS, f"diag_maps_halves_{a.tag}.npz"),
             accA=accs[0], accB=accs[1], sub=sub, r=r)


if __name__ == "__main__":
    main()
