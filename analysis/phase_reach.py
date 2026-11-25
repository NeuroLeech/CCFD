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
    a = ap.parse_args()

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
