"""Why the fitted inputs reinforce below about half a cycle per decay time and cancel above.

The medium damps depth and velocity at the same rate, sigma = 1/tau. With uniform damping
the damped field is exactly the undamped one times e^{-t/tau}, so in the frequency domain
the wavenumber is k = (omega + i/tau)/c. The medium is non-dispersive; a response falls
off as e^{-r/(c tau)} with distance while its phase advances by omega r / c. Over one
attenuation length c tau the phase advances by omega tau = 2 pi f tau, which reaches pi -
half a cycle - at f tau = 1/2.

That predicts the switch, from the medium alone and without the speed c: below f tau ~ 1/2
everything within reach of a patch arrives within half a cycle of everything else, so one
timing pattern can make arrivals add at many patches at once; above it, aligning arrivals
at one patch misaligns them at its neighbours. It also predicts where the argument stops
holding. When the reach c tau is longer than the sheet, the spread of distances among the
inputs that reach a patch is set by the sheet, not by c tau, and the switch should move to
higher f tau. The four media analysed in docs/waves-findings.md were built with the reach
held at about the size of the hemisphere, which is exactly where the two readings meet.

Three measurements, none of which uses a fitted drive:

  dispersion   for a few pieces, the phase gained per mm against the log-amplitude lost
               per mm, in 2.5 mm annuli around the piece. The derivation says their ratio
               is 2 pi f tau, whatever c is.
  arrival      at each of the 150 whole-cortex patches of interplay.py combine, every input
               driven at unit amplitude and in phase: R = |sum_k h_k| / sum_k |h_k|. R = 1
               when every contribution arrives in phase; it falls as the propagation
               phases spread over half a cycle and more.
  alignability G = mean over patches of h_hat h_hat^H, h_hat each patch's normalised
               response over the inputs. The participation ratio of G counts how many
               distinct input timing patterns the patches ask for. Divided by the same
               count with the phases removed (|h_hat|), it is the number of patterns the
               propagation phases alone add.

Media are the recorded solves' (only the medium is read; their S is not used), plus
stand-ins at a chosen (decay, speed) with the same flat construction (waves.standin_medium),
which is how decay and reach are separated.

RESULT, 2026-11-25 (docs/framing/phaseA_findings.md has the full account):

  The stand-in at 8.3333 s, 4.5 reproduces nobp_sm6_s3 to every printed digit.

  The wave speed on the sheet is 5.9x the spread label (the label is an older time-to-peak
  spread measured on the white surface, timescale.py). Measured from the high-frequency
  phase speed, the four recorded media all have a true reach c tau of ~220 mm; the sheet's
  patch-to-patch geodesic distances have median 119 mm and maximum 257 mm.

  Median arrival R falls through 0.5 (interpolated, `crossing`) at, in cycles per decay:
      recorded media (reach ~220 mm)   1x 0.51, 3x 0.51, 5x 0.51, 3x map-weighted 0.45
      decay 8.33 s, reach  74 mm       0.36   (R hovers at 0.50-0.52 from 0.29 to 0.46)
      decay 8.33 s, reach 664 mm       1.27
      infinite flat plane (plane_R)    0.19
  So the frequency where in-phase inputs stop adding is not set by the medium alone: it
  rises with reach relative to the sheet, and "half a cycle per decay" is where it falls
  when reach ~ sheet. The four media's fitted switches (0.54, 0.53, 0.51, 0.42; interplay.py
  media) sit just below their in-phase crossings, map-weighted lowest in both.

  The dispersion check is not informative for the recorded media: at a 220 mm reach the
  15-60 mm annuli lie in the near field, where even the flat-plane Green's function does
  not give the far-field ratio. For the 73 mm stand-in at high f tau it does, and the mesh
  matches it (ratio 31.7 against 2 pi f tau 31.4 at f tau 5; L 74 mm against c tau 73 mm).

  python analysis/phase_reach.py --tags nobp_sm6_s1,nobp_sm6_s3,nobp_sm6_s5,nobp_sm6_mapsmm_s3
  python analysis/phase_reach.py --standin 8.3333,1.5 --standin 8.3333,13.5
"""
import _path  # noqa: F401  - puts the sibling code folders on sys.path
import os, argparse, time
import numpy as np

from mesh_cache import load_cortex
from paths import RESULTS, CACHE
import provenance
import waves
from interplay import cortex_patches

BASIS = os.path.join(RESULTS, "xspec_asc_first_area_100.npz")


def medium(c, spec):
    """-> (run, decay_s, label). `spec` is a solve tag or 'decay,speed' for a stand-in."""
    if "," in spec:
        dec, spr = (float(v) for v in spec.split(","))
        run = waves.load_run(c, standin=spec, basis=BASIS)
        return run, dec, f"stand-in {dec:g} s, {spr:g} mm/s label"
    z = np.load(os.path.join(RESULTS, f"xspec_{spec}.npz"), allow_pickle=True)
    args = provenance.read(z).get("args", {})
    run = waves.load_run(c, spec)
    return run, float(args["decay_s"]), spec


def plane_R(ftau):
    """Arrival R on an infinite flat plane driven everywhere in phase, as a function of
    cycles per decay time alone (with c = tau = 1 the plane has no other scale).

    The response to a point source is G = (i/4) H0(kr), k = omega + i; the in-phase sum is
    the uniform response |1/k^2|, the incoherent sum the integral of |G| over the plane."""
    from scipy.special import hankel1
    from scipy.integrate import quad
    out = []
    for ft in np.atleast_1d(ftau):
        k = 2 * np.pi * ft + 1j
        f = lambda r: abs(0.25 * hankel1(0, k * r)) * 2 * np.pi * r
        den = quad(f, 0, 1, limit=400)[0] + quad(f, 1, 60, limit=800)[0]
        out.append(1.0 / abs(k) ** 2 / den)
    return np.array(out)


def rim(S):
    """Vertices on the sheet's boundary (edges used by one face)."""
    F = np.asarray(S.F)
    e = np.sort(np.r_[F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], axis=1)
    u, n = np.unique(e, axis=0, return_counts=True)
    return np.unique(u[n == 1])


def dispersion(Hk, hz, dist, rlo, rhi, width=2.5):
    """Per frequency: phase slope (rad/mm) and amplitude slope (1/mm) of one piece's
    response against distance from it, in annuli. -> (phase_slope, amp_slope) (nhz,)"""
    edges = np.arange(0.0, rhi + width, width)
    cen = 0.5 * (edges[1:] + edges[:-1])
    which = np.digitize(dist, edges) - 1
    ok = (which >= 0) & (which < len(cen))
    ps, as_ = np.full(len(hz), np.nan), np.full(len(hz), np.nan)
    for f in range(len(hz)):
        m = np.zeros(len(cen), complex)
        a = np.zeros(len(cen))
        cnt = np.bincount(which[ok], minlength=len(cen))
        np.add.at(m, which[ok], Hk[f][ok])
        np.add.at(a, which[ok], np.abs(Hk[f][ok]))
        good = cnt > 3
        m[good] /= cnt[good]; a[good] /= cnt[good]
        ph = np.unwrap(np.angle(np.where(good, m, 1.0)))
        sel = good & (cen >= rlo) & (cen <= rhi) & (a > 1e-4 * a[good].max())
        if sel.sum() < 4:
            continue
        ps[f] = np.polyfit(cen[sel], ph[sel], 1)[0]
        as_[f] = np.polyfit(cen[sel], np.log(a[sel]) + 0.5 * np.log(cen[sel]), 1)[0]
    return ps, as_


def patch_measures(H, patches):
    """-> arrival R (nhz, P), alignability PR and PR with phases removed (nhz,)."""
    nhz, _, K = H.shape
    R = np.empty((nhz, len(patches)))
    pr, pr_abs, l1, l1_abs = (np.empty(nhz) for _ in range(4))
    for f in range(nhz):
        h = np.stack([H[f][P["pv"]].mean(0) for P in patches])          # (P, K)
        R[f] = np.abs(h.sum(1)) / np.abs(h).sum(1)
        hh = h / np.linalg.norm(h, axis=1, keepdims=True)
        for G, out_pr, out_l1, i in ((hh.T @ hh.conj() / len(h), pr, l1, f),
                                     (np.abs(hh).T @ np.abs(hh) / len(h), pr_abs, l1_abs, f)):
            ev = np.clip(np.linalg.eigvalsh(0.5 * (G + G.conj().T)), 0, None)
            ev = ev / ev.sum()
            out_pr[i] = 1.0 / np.sum(ev ** 2)
            out_l1[i] = ev.max()
    return R, pr, pr_abs, l1, l1_abs


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tags", default="", help="comma-separated solve tags (medium only)")
    ap.add_argument("--standin", action="append", default=[],
                    help="decay_s,spread_mm_s for a flat stand-in medium (repeatable)")
    ap.add_argument("--nhz", type=int, default=24)
    ap.add_argument("--fmin", type=float, default=0.004)
    ap.add_argument("--fmax", type=float, default=0.6)
    ap.add_argument("--pieces", type=int, default=3,
                    help="pieces used for the dispersion check, farthest from the rim")
    ap.add_argument("--rlo", type=float, default=15.0)
    ap.add_argument("--rhi", type=float, default=60.0)
    ap.add_argument("--figure", default="",
                    help="only redraw the findings figure from this saved .npz")
    a = ap.parse_args()
    if a.figure:
        reach_figure(a.figure, a.figure.replace(".npz", "_reach.png"))
        return

    c = load_cortex("fsaverage5", verbose=False)
    S = waves.Surface(c)
    g1 = np.load(os.path.join(CACHE, f"cortical_maps_{c.mesh}.npz"))["grad1"]
    patches = cortex_patches(c, S, 15.0, 10.0, g1)
    hz = np.geomspace(a.fmin, a.fmax, a.nhz)
    edge = rim(S)
    specs = [t for t in a.tags.split(",") if t] + a.standin
    out = {}
    for spec in specs:
        t0 = time.time()
        run, tau, label = medium(c, spec)
        H = waves.transfer_fields(c, run, hz, verbose=False)               # (nhz, nV, K)
        lab = np.asarray(run["labels"])
        K = H.shape[2]
        d_rim = S.distance_from(edge)
        cen_rim = np.array([np.median(d_rim[lab == k]) for k in range(K)])
        pick = np.argsort(cen_rim)[::-1][:a.pieces]
        ratios, cph, L = [], [], []
        for k in pick:
            dist = S.distance_from(np.flatnonzero(lab == k))
            ps, as_ = dispersion(H[:, :, k], hz, dist, a.rlo, a.rhi)
            ratios.append(np.abs(ps) / np.abs(as_))
            cph.append(2 * np.pi * hz / np.abs(ps))
            L.append(1.0 / np.abs(as_))
        ratio = np.nanmedian(ratios, 0)
        R, pr, pr_abs, l1, l1_abs = patch_measures(H, patches)
        ft = hz * tau
        out[spec] = dict(label=label, tau=tau, hz=hz, ftau=ft, ratio=ratio,
                         c_phase=np.nanmedian(cph, 0), L=np.nanmedian(L, 0),
                         R_med=np.median(R, 1), R_q=np.percentile(R, [25, 75], 1),
                         pr=pr, pr_abs=pr_abs, l1=l1, l1_abs=l1_abs)
        print(f"\n  {label}: decay {tau:g} s, {K} pieces, {len(patches)} patches  "
              f"[{time.time()-t0:.0f}s]")
        print("      Hz     f*tau   phase/amp slope  2pi f tau   c_phase mm/s   L mm   "
              "arrival R   patterns  (no phase)")
        for i in range(len(hz)):
            print(f"    {hz[i]:.4f}  {ft[i]:6.3f}   {ratio[i]:9.3f}      {2*np.pi*ft[i]:8.3f}"
                  f"   {out[spec]['c_phase'][i]:9.1f}   {out[spec]['L'][i]:6.0f}   "
                  f"{out[spec]['R_med'][i]:7.3f}    {pr[i]:6.1f}   ({pr_abs[i]:5.1f})")
        hit = np.flatnonzero(out[spec]["R_med"] < 0.5)
        if len(hit):
            print(f"    median arrival R first falls below 0.5 at f*tau = {ft[hit[0]]:.3f}")
    stem = os.path.join(RESULTS, "phase_reach_" + "_".join(s.replace(",", "-") for s in specs))
    np.savez(stem + ".npz", out=np.array(out, object))
    print(f"\n  wrote {stem}.npz")
    _plot(stem + ".png", out)


def crossing(ft, R, level=0.5):
    """First f tau at which R falls through `level`, interpolated in log f tau."""
    i = np.flatnonzero((R[:-1] >= level) & (R[1:] < level))
    if not len(i):
        return np.nan
    i = i[0]
    w = (R[i] - level) / (R[i] - R[i + 1])
    return float(np.exp(np.log(ft[i]) + w * np.log(ft[i + 1] / ft[i])))


# where the fitted drives' reinforcing share gives way to cancelling, in cycles per decay
# (interplay.py media on the impulse-fixed solves, docs/waves-findings.md). To be replaced
# by the rebuilt solves' values once the Phase A queue has run `interplay.py bold` on them.
FITTED_SWITCH = {"nobp_sm6_s1": 0.54, "nobp_sm6_s3": 0.53, "nobp_sm6_s5": 0.51,
                 "nobp_sm6_mapsmm_s3": 0.42}
NAMES = {"nobp_sm6_s1": "1x", "nobp_sm6_s3": "3x", "nobp_sm6_s5": "5x",
         "nobp_sm6_mapsmm_s3": "3x map-weighted"}
C_PER_LABEL = 5.9        # wave speed on the sheet per mm/s of spread label, measured here


def reach_figure(npz, path):
    """Two panels for the findings note: (a) arrival R against f tau at a fixed decay with
    the reach varied, against the infinite plane; (b) each recorded medium's in-phase
    crossing against its fitted switch."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = np.load(npz, allow_pickle=True)["out"].item()
    ink, ink2, grid = "#191919", "#666666", "#e5e5e5"
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": ink2, "axes.labelcolor": ink,
                         "xtick.color": ink2, "ytick.color": ink2, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw=dict(width_ratios=[1.6, 1]))

    a = ax[0]
    ftp = np.geomspace(0.02, 5, 60)
    a.plot(ftp, plane_R(ftp), color="#8c8c8c", lw=1.5, ls=(0, (4, 3)))
    a.text(0.045, 0.80, "infinite flat plane", color=ink2, fontsize=8, ha="right")
    series = [("8.3333,1.5", "#257de0"), ("8.3333,4.5", "#f47230"), ("8.3333,13.5", "#00b894")]
    for spec, col in series:
        if spec not in out:
            continue
        r = out[spec]
        spr = float(spec.split(",")[1])
        reach = C_PER_LABEL * spr * r["tau"]
        a.plot(r["ftau"], r["R_med"], color=col, lw=2, marker="o", ms=4,
               markeredgecolor="white", markeredgewidth=0.8)
        lab = f"reach {reach:.0f} mm" + (" = 3x" if spr == 4.5 else "")
        # direct labels, each placed in a clear part of the panel
        i, dx, dy, ha = {1.5: (int(np.flatnonzero(r["R_med"] < 0.5)[0]), 0, -16, "center"),
                         4.5: (int(np.flatnonzero(r["R_med"] < 0.6)[0]), 8, 0, "left"),
                         13.5: (int(np.flatnonzero(r["R_med"] < 0.8)[0]), 8, 0, "left")
                         }.get(spr, (0, 8, 0, "left"))
        a.annotate(lab, (r["ftau"][i], r["R_med"][i]), xytext=(dx, dy),
                   textcoords="offset points", color=ink, fontsize=8, va="center", ha=ha)
    a.axhline(0.5, color=grid, lw=1, zorder=0)
    a.axvline(0.5, color=ink2, lw=0.8, ls=":")
    a.text(0.52, 1.02, "half a cycle per decay", color=ink2, fontsize=8)
    a.set_xscale("log")
    a.set_ylim(0, 1.06)
    a.set_xlabel("cycles per decay time (f tau), decay held at 8.33 s")
    a.set_ylabel("arrival R: in-phase sum / sum of sizes\n(median over 150 patches)")
    a.set_title("(a) Where in-phase inputs stop adding moves with reach", loc="left",
                fontsize=10, color=ink)
    a.grid(axis="y", color=grid, lw=0.6)

    b = ax[1]
    xs, ys, labs = [], [], []
    for tag, fs in FITTED_SWITCH.items():
        if tag in out:
            xs.append(crossing(out[tag]["ftau"], out[tag]["R_med"]))
            ys.append(fs)
            labs.append(NAMES[tag])
    lim = (0.38, 0.6)
    b.plot(lim, lim, color=ink2, lw=0.8, ls=":", zorder=0)
    b.scatter(xs, ys, s=40, color="#f47230", edgecolor="white", linewidth=1, zorder=3)
    off = {"1x": (7, 2), "3x": (7, -9), "5x": (7, -3), "3x map-weighted": (7, -3)}
    for x, y, l in zip(xs, ys, labs):
        b.annotate(l, (x, y), xytext=off.get(l, (7, -3)), textcoords="offset points",
                   color=ink, fontsize=8)
    b.text(0.385, 0.59, "dotted: equal", color=ink2, fontsize=8)
    b.set_xlim(lim); b.set_ylim(lim)
    b.set_xlabel("in-phase crossing, medium only (f tau)")
    b.set_ylabel("fitted switch, reinforcing to cancelling (f tau)")
    b.set_title("(b) Fitted switch against the medium's crossing", loc="left",
                fontsize=10, color=ink)
    b.grid(color=grid, lw=0.6)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor="white")
    print(f"  wrote {path}")
    for tag in out:
        print(f"    {out[tag]['label']:40s} crossing {crossing(out[tag]['ftau'], out[tag]['R_med']):.3f}")
    print(f"    {'infinite flat plane':40s} crossing {crossing(ftp, plane_R(ftp)):.3f}")


def _plot(path, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    for spec, r in out.items():
        ax[0].plot(r["ftau"], r["ratio"] / (2 * np.pi * r["ftau"]), "o-", ms=3, label=r["label"])
        ax[1].plot(r["ftau"], r["R_med"], "o-", ms=3, label=r["label"])
        ax[2].plot(r["ftau"], r["pr"] / r["pr_abs"], "o-", ms=3, label=r["label"])
    ax[0].axhline(1, color="0.5", lw=0.8)
    ax[0].set_ylabel("measured phase/amplitude slope  /  2 pi f tau")
    ax[1].set_ylabel("arrival R (median over patches)")
    ax[2].set_ylabel("timing patterns asked for  /  same without phase")
    for x in ax:
        x.set_xscale("log"); x.axvline(0.5, color="0.5", ls=":", lw=0.8)
        x.set_xlabel("cycles per decay time (f tau)")
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"  wrote {path}")


if __name__ == "__main__":
    main()
