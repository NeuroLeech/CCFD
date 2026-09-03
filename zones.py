"""Zones of influence: which thalamic input system accounts for which part of the field.

The model is LTI, so its covariance is exactly C = sum_f w_f 2 Re(H_f S_f H_f^H), and the
variance of every vertex splits between named groups of input channels. That is a zone of
influence in the strict sense rather than a proxy for one. The data has no counterpart -
there is no measured input to attribute anything to - so this keeps two things apart:

  MODEL-INTERNAL. Group the 47 driven pieces by the thalamic nucleus that named them and
  split diag(C) between the groups. diag(C) is a quadratic form in the channels, so the
  three obvious attributions differ ONLY in how much of the cross term they carry:

      own-block       G_gg                        cross terms dropped
      Shapley         G_gg +   sum_{h!=g} G_gh    cross split evenly; sums to the total
      leave-one-out   G_gg + 2 sum_{h!=g} G_gh    cross counted whole

  (the Shapley value of a quadratic set function is exactly the own block plus half of
  each pairwise interaction, so it is closed-form here and needs no subset enumeration).
  Their disagreement IS the interference between systems, vertex by vertex, and reporting
  all three costs one extra sum.

  AGAINST THE DATA. Attribution has no empirical counterpart, so the comparison runs
  through a rule that uses no model internals and can be applied to both sides: seed FC
  from each group's driven vertices, winner-take-all across groups. Same rule on the
  empirical target and on the model FC, then Dice per zone. The model's attributed zone
  can then be checked against the model's OWN seed-FC territory - a consistency check
  between "influence" and "connectivity territory" that the data cannot supply.

S = I separates the medium from the input: with the solved S a zone is where the fit puts
that system's power, with S = I it is the medium's own footprint from those pieces.

  python zones.py --tag pr_taper
"""
import os, argparse
import numpy as np

from mesh_cache import load_cortex
from paths import RESULTS
import fc_score, subparcels, bo_step, xspec, units, timescale
from diag_distance import distance_to_drive

# The grouping is the one that built the region set (subparcels.SUBCORTICAL, whose comment
# names the nucleus for each parcel). Sensory systems are LGN / MGN / VPL-VPM; the rest
# are kept as their own groups rather than folded into an "other" bin, because 18 of the
# 47 pieces sit outside any sensory nucleus and hiding them would make the sensory
# fractions sum to something meaningless.
NUCLEI = [
    ("LGN",      [1],            "visual"),
    ("MGN",      [24],           "auditory"),
    ("VPL/VPM",  [9, 51],        "somatosensory"),
    ("VA/VL",    [8, 96, 55],    "motor"),
    ("pulvinar", [145, 42, 22],  "associative visual"),
    ("mediodors", [86, 84, 62],  "prefrontal"),
    ("limbic",   [164, 93, 118], "limbic"),
    ("insular",  [112],          "insular"),
]


def rebuild(tag, c, t, verbose=True):
    """Reconstruct H and the solved S for a best_fit run, from what it saved.

    best_fit stores S, the frequency bins, the medium vector x and `save`, which is
    everything except the impulse responses themselves - and those are cached by medium,
    so this is a cache hit rather than a re-simulation whenever the run's own cache
    survives. The bins are taken from the file instead of regenerated: transfer's default
    grid depends on the pad, and a silently different grid would pair S with an H that
    describes another system."""
    z = np.load(os.path.join(RESULTS, f"xspec_{tag}.npz"), allow_pickle=True)
    S, idx, x, save = z["S"], z["idx"], z["x"], int(z["save"])
    labels, tags = z["labels"], list(z["tags"])

    p, save_x, _ = bo_step.unpack(x, c)
    assert save_x == save, (save_x, save)
    P = subparcels.taper_profiles(c, labels, len(tags))

    # the clock only enters here through the BOLD smoothing kernel; p and save come from x
    clock = timescale.plan(4, decay_s=9.03, spread_mm_s=6, verbose=False)
    kern = units.smoothing_kernel(timescale.bold_fwhm_frames(clock["frame_s"]))
    decay_fr = 1.0 / (10.0 ** x[0] * save)
    imp = 224
    if imp < 3 * decay_fr:
        imp = int(np.ceil(3 * decay_fr / 64.0) * 64)
    pad = 4096
    if verbose:
        print(f"  {tag}: {len(tags)} pieces, save {save}, damping {10**x[0]:.2e}, "
              f"decay {decay_fr:.0f} frames, impulse window {imp}, pad {pad}, "
              f"{len(idx)} frequency bins")

    resp = xspec.impulse_responses(c, list(range(len(P))), p, imp * save, save,
                                   profiles=P, verbose=False)
    R = np.pad(resp, ((0, 0), (0, max(0, pad - resp.shape[1])), (0, 0)))
    H, w, idx2 = xspec.transfer(R, t.cols, len(idx), kernel=kern, idx=idx)
    assert np.array_equal(idx2, idx), "frequency grid does not match the saved solve"
    return dict(H=H, w=w, S=S, tags=tags, labels=labels, P=P, p=p, save=save)


def group_index(tags):
    """-> (group names, list of channel-index arrays, piece-count per group).

    Piece tags are '<parcel>_<n>', so the parcel each channel came from is its prefix."""
    parcel = np.array([int(s.split("_")[0]) for s in tags])
    names, idxs = [], []
    for nm, ps, _role in NUCLEI:
        k = np.flatnonzero(np.isin(parcel, ps))
        names.append(nm)
        idxs.append(k)
    seen = np.concatenate(idxs)
    assert len(seen) == len(tags) and len(set(seen.tolist())) == len(tags), \
        "the nucleus grouping does not partition the driven pieces"
    return names, idxs


def group_quadratic(H, w, S, gidx, chunk=2048):
    """-> G (nV, nG, nG): the vertexwise quadratic form of diag(C), grouped.

    G[v, g, h] = sum_f w_f 2 Re( sum_{k in g, l in h} H[f,v,k] S[f,k,l] conj(H[f,v,l]) ),
    so diag(C)[v] = G[v].sum() and any subset of groups is a masked sum. Built by group
    pair rather than by forming the (nV, K, K) array, which would be 20M entries per
    frequency for nothing."""
    nf, nV, _K = H.shape
    nG = len(gidx)
    G = np.zeros((nV, nG, nG))
    for a in range(0, nV, chunk):
        b = min(a + chunk, nV)
        acc = np.zeros((b - a, nG, nG))
        for f in range(nf):
            A = H[f, a:b]                                  # (nv, K)
            for g in range(nG):
                if len(gidx[g]) == 0:
                    continue
                # T = A_g @ S[g, :], the group's contribution to every channel's partner
                T = A[:, gidx[g]] @ S[f][np.ix_(gidx[g], np.arange(A.shape[1]))]
                for h in range(nG):
                    if len(gidx[h]) == 0:
                        continue
                    acc[:, g, h] += w[f] * 2.0 * np.real(
                        (T[:, gidx[h]] * np.conj(A[:, gidx[h]])).sum(1))
        G[a:b] = acc
    return G


def attributions(G):
    """-> dict of (nV, nG) attributions, all three, plus the total and the cross term."""
    own = np.diagonal(G, axis1=1, axis2=2).copy()          # (nV, nG)
    off = G.sum(2) - own                                   # sum_{h!=g} G_gh
    return dict(own=own, shapley=own + off, loo=own + 2.0 * off,
                total=G.sum((1, 2)), cross=off)


def fractions(A, floor=1e-12):
    """Per-vertex fractions of a (nV, nG) attribution, and the winner-take-all label.

    Attributions can go negative where a system's cross terms cancel its own power, and a
    negative share is not a fraction. Fractions are therefore taken over the POSITIVE part
    and vertices whose positive part is negligible are left unlabelled (-1) instead of
    being assigned to whichever group is least negative."""
    Ap = np.clip(A, 0.0, None)
    tot = Ap.sum(1)
    ok = tot > floor * max(float(tot.max()), floor)
    f = np.zeros_like(Ap)
    f[ok] = Ap[ok] / tot[ok, None]
    lab = np.where(ok, np.argmax(Ap, 1), -1)
    return f, lab, ok


def half_max_radius(share, d, step=5.0, minn=25):
    """Distance at which a group's mean share has fallen to half its peak, in mm.

    Taken on the distance PROFILE rather than on individual vertices: the largest distance
    at which some vertex still exceeds half the peak is an outlier statistic, and it reads
    0 for any group whose peak sits on a single vertex. Bins with fewer than `minn`
    vertices are dropped so the tail does not turn on one or two of them. Linear
    interpolation between the straddling bins; nan if the profile never falls that far."""
    edges = np.arange(0, np.nanmax(d) + step, step)
    ctr, mean = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (d >= lo) & (d < hi)
        if m.sum() >= minn:
            ctr.append(0.5 * (lo + hi)); mean.append(float(share[m].mean()))
    if len(mean) < 2:
        return np.nan
    ctr, mean = np.asarray(ctr), np.asarray(mean)
    k = int(np.argmax(mean))
    half = 0.5 * mean[k]
    below = np.flatnonzero(mean[k:] < half)
    if not len(below):
        return np.nan
    j = k + int(below[0])
    x0, x1, y0, y1 = ctr[j - 1], ctr[j], mean[j - 1], mean[j]
    return float(x0 + (y0 - half) * (x1 - x0) / max(y0 - y1, 1e-30))


def seed_territory(FCrows, gidx_vert, nV):
    """Winner-take-all territory from seed FC: for each group, the mean FC row over its
    driven vertices; the label is the group with the largest mean.

    This is the one rule that can be applied to the model and to the data alike - it uses
    only an FC matrix and a set of seed vertices, and nothing the model knows about its
    own input."""
    M = np.zeros((len(gidx_vert), nV))
    for g, vs in enumerate(gidx_vert):
        M[g] = FCrows[g] if FCrows[g].ndim == 1 else FCrows[g].mean(0)
    return M, np.argmax(M, 0)


def dice(a, b, nG):
    """Per-label Dice between two label vectors, and the mean over labels present."""
    out = []
    for g in range(nG):
        A, B = a == g, b == g
        d = 2.0 * float((A & B).sum()) / max(int(A.sum()) + int(B.sum()), 1)
        out.append(d if (A.sum() or B.sum()) else np.nan)
    return np.array(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", default="pr_taper")
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--margin", type=float, default=0.10,
                    help="a vertex is 'contested' when the top two shares differ by less")
    a = ap.parse_args()

    c = load_cortex("fsaverage5", verbose=False)
    t = fc_score.default_target(c, verbose=False)
    R = rebuild(a.tag, c, t)
    H, w, S, tags, labels = R["H"], R["w"], R["S"], R["tags"], R["labels"]
    names, gidx = group_index(tags)
    nG = len(names)

    # ---- the rebuild has to be checked before anything is attributed to it -----------
    # against the RAW frames, not target.model_z: for a Spearman target model_z rank
    # transforms each row, which gives every vertex the same variance and would turn this
    # check into a correlation between diag(C) and a constant.
    F = np.load(os.path.join(RESULTS, f"frames_{a.tag}.npy"), mmap_mode="r")
    raw = np.asarray(F[t.burn:, t.cols], np.float64)
    realised_var = raw.var(0)
    G = group_quadratic(H, w, S, gidx, a.chunk)
    att = attributions(G)
    r = float(np.corrcoef(att["total"], realised_var)[0, 1])
    rl = float(np.corrcoef(np.log(np.abs(att["total"]) + 1e-30),
                           np.log(realised_var + 1e-30))[0, 1])
    print(f"\n  rebuild check: predicted diag(C) against the realised field's variance "
          f"over {len(realised_var)} vertices, r = {r:+.4f} (log-log {rl:+.4f})")
    if r < 0.9:
        raise SystemExit("  the rebuilt H does not pair with the saved S; stopping "
                         "rather than attributing variance the model never produced")

    # ---- how much of the field is interference between systems ----------------------
    tot = att["total"].sum()
    own_sum = att["own"].sum()
    print(f"  total variance {tot:.4g}; own blocks {own_sum/tot:+.1%} of it, "
          f"cross terms {(tot-own_sum)/tot:+.1%}")

    # ---- the three attributions, as whole-cortex shares ------------------------------
    print(f"\n  {'group':<10s} {'role':<19s} {'pieces':>6s} {'mm2':>7s}"
          f" {'own':>8s} {'shapley':>8s} {'loo':>8s}")
    area = np.asarray(c.A, float)
    for g, (nm, ps, role) in enumerate(NUCLEI):
        mm2 = float(sum(area[c.lab == q].sum() for q in ps))
        print(f"  {nm:<10s} {role:<19s} {len(gidx[g]):>6d} {mm2:>7.0f}"
              f" {att['own'][:, g].sum()/tot:>7.1%} {att['shapley'][:, g].sum()/tot:>7.1%}"
              f" {att['loo'][:, g].sum()/tot:>7.1%}")

    # ---- S = I: the medium's own footprint, with the fit switched off -----------------
    SI = np.zeros_like(S)
    for f in range(S.shape[0]):
        np.fill_diagonal(SI[f], 1.0)
    G0 = group_quadratic(H, w, SI, gidx, a.chunk)
    att0 = attributions(G0)

    f_fit, lab_fit, ok_fit = fractions(att["shapley"])
    f_geo, lab_geo, _ = fractions(att0["shapley"])

    # ---- zone size, and how far each system reaches ----------------------------------
    print(f"\n  zone (winner-take-all on the Shapley share), and half-max radius")
    print(f"  {'group':<10s} {'fit mm2':>9s} {'S=I mm2':>9s} {'r50 fit':>8s} "
          f"{'r50 S=I':>8s} {'contested':>10s}")
    srt = np.sort(f_fit, 1)
    contested = ok_fit & ((srt[:, -1] - srt[:, -2]) < a.margin)
    piece_of = np.array([int(s.split("_")[0]) for s in tags])
    dist, radii = {}, {}
    for g, (nm, ps, _role) in enumerate(NUCLEI):
        driven = np.isin(labels, np.flatnonzero(np.isin(piece_of, ps)))
        d = distance_to_drive(c, driven)[t.cols]
        dist[nm] = d
        rr = [half_max_radius(fr, d) for fr in (f_fit[:, g], f_geo[:, g])]
        radii[nm] = rr
        av = area[t.cols]
        print(f"  {nm:<10s} {av[lab_fit == g].sum():>9.0f} {av[lab_geo == g].sum():>9.0f}"
              f" {rr[0]:>8.0f} {rr[1]:>8.0f}"
              f" {float((contested & (lab_fit == g)).sum()) / max((lab_fit==g).sum(),1):>9.1%}")
    print(f"  contested overall: {contested.sum()} of {ok_fit.sum()} vertices "
          f"({contested.sum()/max(ok_fit.sum(),1):.1%}) within {a.margin:.0%}")

    # ---- the rule that both sides can answer: seed FC territory ----------------------
    Tfc = t.target_fc()
    seeds = []
    for g, (nm, ps, _role) in enumerate(NUCLEI):
        driven = np.isin(labels, np.flatnonzero(np.isin(piece_of, ps)))
        seeds.append(np.flatnonzero(driven[t.cols]))
    # the model side has to be the SAME object as the target side - Spearman FC, double
    # centred - or the two winner-take-all maps are answering different questions
    Mfc = t.model_fc(F)
    Emp = np.stack([np.asarray(Tfc[s]).mean(0) for s in seeds])
    Mod = np.stack([np.asarray(Mfc[s]).mean(0) for s in seeds])
    del Mfc
    lab_emp, lab_mod = np.argmax(Emp, 0), np.argmax(Mod, 0)

    d_em = dice(lab_emp, lab_mod, nG)
    d_af = dice(lab_fit, lab_emp, nG)
    d_am = dice(lab_fit, lab_mod, nG)
    print(f"\n  seed-FC territory, same rule both sides; and the attributed zone against each")
    print(f"  {'group':<10s} {'emp mm2':>9s} {'mod mm2':>9s} {'Dice e|m':>9s} "
          f"{'att|emp':>8s} {'att|mod':>8s}")
    av = area[t.cols]
    for g, nm in enumerate(names):
        print(f"  {nm:<10s} {av[lab_emp == g].sum():>9.0f} {av[lab_mod == g].sum():>9.0f}"
              f" {d_em[g]:>9.3f} {d_af[g]:>8.3f} {d_am[g]:>8.3f}")
    print(f"  {'mean':<10s} {'':>9s} {'':>9s} {np.nanmean(d_em):>9.3f} "
          f"{np.nanmean(d_af):>8.3f} {np.nanmean(d_am):>8.3f}")

    out = os.path.join(RESULTS, f"zones_{a.tag}.npz")
    np.savez(out, names=np.array(names, dtype=object), G=G, G0=G0,
             own=att["own"], shapley=att["shapley"], loo=att["loo"],
             total=att["total"], cross=att["cross"], shapley_SI=att0["shapley"],
             f_fit=f_fit, f_geo=f_geo, lab_fit=lab_fit, lab_geo=lab_geo,
             lab_emp=lab_emp, lab_mod=lab_mod, contested=contested,
             emp_seed=Emp, mod_seed=Mod, cols=t.cols,
             dist=np.stack([dist[n] for n in names]))
    print(f"\n  wrote {out}")

    _plot(a.tag, c, t, names, f_fit, lab_fit, lab_emp, lab_mod, contested)


def _plot(tag, c, t, names, f_fit, lab_fit, lab_emp, lab_mod, contested):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from render_regimes import _proj
    from plot_fc_map import surface_row

    proj = _proj(c.V, c.F)
    nG = len(names)
    cols = plt.cm.tab10(np.linspace(0, 1, 10))[:nG]
    cmap = ListedColormap(cols)

    rows = [("attributed zone\n(Shapley share)", lab_fit.astype(float), cmap, (-0.5, nG - 0.5)),
            ("model seed-FC\nterritory", lab_mod.astype(float), cmap, (-0.5, nG - 0.5)),
            ("empirical seed-FC\nterritory", lab_emp.astype(float), cmap, (-0.5, nG - 0.5)),
            ("contested\n(top two < 10%)", contested.astype(float), "Greys", (0, 1))]
    rows += [(f"share: {names[g]}", f_fit[:, g], "magma", (0, 1)) for g in range(nG)]

    fig = plt.figure(figsize=(3.4 * len(proj), 2.3 * len(rows)))
    gs = fig.add_gridspec(len(rows), len(proj), hspace=0.06, wspace=0.02)
    for r, (lab, vals, cm, lims) in enumerate(rows):
        surface_row(fig, gs, r, proj, vals, c, t.cols, cm, lims, lab)
    handles = [plt.Line2D([], [], marker="s", ls="", color=cols[g], label=names[g])
               for g in range(nG)]
    fig.legend(handles=handles, loc="lower center", ncol=nG, frameon=False, fontsize=9)
    out = os.path.join(RESULTS, f"zones_{tag}.png")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
