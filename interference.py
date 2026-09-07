"""Where the driven pieces add and where they cancel.

The medium is linear, so the field is a superposition of one contribution per driven
piece: f(v,t) = sum_k (R_k * a_k)(v,t), with R_k the piece's impulse response and a_k its
amplitude series. Nothing about that needs a new simulation - the responses are the same
cached objects the transfer function is built from - so each piece's contribution can be
formed separately and the total rebuilt from them.

That makes interference measurable rather than inferred. Per vertex:

    coherent   = var_t( sum_k f_k )        what the field actually does
    incoherent = sum_k var_t( f_k )        what it would do if the pieces never met

Their ratio is 1 where contributions are unrelated, above 1 where they reinforce and below
1 where they cancel. Instantaneously the same split gives a movie: (sum_k f_k)^2 minus
sum_k f_k^2 is positive on a constructive front and negative on a destructive one.

Filtering commutes with the superposition, so the smoothing kernel and the passband are
applied to each contribution separately and the sum is still the observable that was
scored. That matters because interference in the raw field and interference in the
OBSERVABLE are different questions, and only the second one bears on the fit.

The decomposition is checked against the run's own saved frames before anything is read
off it - r = 0.9990 with an sd ratio of 1.007 on the run this was built for. It is not
exact: the drive is injected every STEP and the responses are sampled every `save` steps,
so treating a frame's whole block as one impulse is an approximation, and that residual is
what the check measures.

  python interference.py --tag g7_s1.5_d25
"""
import os, argparse
import numpy as np
from scipy.signal import fftconvolve

from paths import RESULTS
import timescale


def contributions(c, tag, imp_frames=None, window=None, band=(0.01, 0.08), verbose=True):
    """Accumulate per-piece contributions without ever holding them all at once.

    -> dict with the total field, each piece's variance per vertex, the incoherent sum,
    and - over `window` frames - the instantaneous coherent and incoherent energy."""
    import xspec, bo_step, subparcels, units, bandpass
    z = np.load(os.path.join(RESULTS, f"xspec_{tag}.npz"), allow_pickle=True)
    x, save, labels, tags = z["x"], int(z["save"]), z["labels"], list(z["tags"])
    p, _, _ = bo_step.unpack(x, c)
    P = subparcels.taper_profiles(c, labels, len(tags))
    frame_s = timescale.TR / 4.0
    K = len(tags)

    A = np.asarray(np.load(os.path.join(RESULTS, f"drive_{tag}.npy")), np.float64)
    nfr = len(A) // save
    # total injected per frame, not the per-step value: the block is what the frame's
    # single sampled impulse response has to stand for
    a = A[:nfr * save].reshape(nfr, save, -1).sum(1)

    if imp_frames is None:
        imp_frames = _impulse_frames(x, save, frame_s)
    R = xspec.impulse_responses(c, list(range(K)), p, imp_frames * save, save,
                                profiles=P, verbose=False)
    kern = units.smoothing_kernel(timescale.bold_fwhm_frames(frame_s, verbose=False),
                                  verbose=False)
    if verbose:
        print(f"  {tag}: {K} pieces, {nfr} frames, impulse window {R.shape[1]}")

    win = window if window is not None else (0, min(600, nfr))
    w0, w1 = win
    nV = c.nV
    tot = np.zeros((nfr, nV), np.float32)
    var_k = np.zeros((K, nV), np.float32)
    win_sq = np.zeros((w1 - w0, nV), np.float32)          # sum_k f_k^2, instantaneous
    for k in range(K):
        f = fftconvolve(a[:, k:k + 1], R[k], axes=0)[:nfr]
        f = units.smooth_frames(f, kern)
        f = bandpass.apply(f, frame_s, band[0], band[1])
        tot += f
        var_k[k] = f.var(0)
        win_sq += f[w0:w1] ** 2
        if verbose and (k + 1) % 10 == 0:
            print(f"    {k+1}/{K} pieces", flush=True)
        del f
    return dict(tot=tot, var_k=var_k, win_sq=win_sq, win=(w0, w1), tags=tags,
                labels=labels, frame_s=frame_s, nfr=nfr)


def _impulse_frames(x, save, frame_s, decays=7.0, lo=0.01):
    """The window best_fit would have used, from the run's own damping."""
    decay_fr = 1.0 / (10.0 ** float(x[0]) * save)
    need = max(decays * decay_fr, 1.0 / (lo * frame_s))
    return int(np.ceil(need / 64.0) * 64)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", default="g7_s1.5_d25")
    ap.add_argument("--start", type=int, default=200)
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--edge", type=int, default=600,
                    help="frames ignored at each end when checking against the run")
    a = ap.parse_args()

    from mesh_cache import load_cortex
    import fc_score
    c = load_cortex("fsaverage5", verbose=False)
    t = fc_score.default_target(c, verbose=False)
    r = contributions(c, a.tag, window=(a.start, a.start + a.n))
    tot, var_k, nfr = r["tot"], r["var_k"], r["nfr"]

    # ---- the decomposition has to reproduce the run before anything is read off it ----
    F = np.asarray(np.load(os.path.join(RESULTS, f"frames_{a.tag}.npy"), mmap_mode="r"),
                   np.float32)
    n, e = min(len(tot), len(F)), a.edge
    rr = float(np.corrcoef(tot[e:n-e].ravel(), F[e:n-e].ravel())[0, 1])
    print(f"\n  superposition against the run's saved field: r = {rr:.6f}, "
          f"sd ratio {tot[e:n-e].std()/F[e:n-e].std():.4f}")
    if rr < 0.99:
        raise SystemExit("  the decomposition does not reproduce the run")

    coh = tot.var(0)
    inc = var_k.sum(0)
    ok = inc > 0
    ratio = np.where(ok, coh / np.maximum(inc, 1e-30), np.nan)
    cols = t.cols
    print(f"\n  coherent / incoherent power per vertex, over {nfr} frames")
    print(f"    mean {np.nanmean(ratio[cols]):.3f}  median {np.nanmedian(ratio[cols]):.3f}"
          f"  range {np.nanmin(ratio[cols]):.3f}-{np.nanmax(ratio[cols]):.3f}")
    print(f"    vertices below 1 (net cancelling): "
          f"{np.nanmean(ratio[cols] < 1):.1%}")
    tot_coh, tot_inc = coh[cols].sum(), inc[cols].sum()
    print(f"    summed over cortex: coherent {tot_coh:.4g}, incoherent {tot_inc:.4g}, "
          f"ratio {tot_coh/tot_inc:.3f}")

    # How many pieces actually reach a vertex. Only where the vertex receives something:
    # at a vertex every piece leaves at ~0 the shares are 0/0, the guard makes them all
    # tiny, and 1/sum(share^2) then returns ~1e30 rather than a participation ratio.
    vk = var_k[:, cols]
    recv = vk.sum(0)
    live = recv > 1e-6 * np.median(recv[recv > 0])
    eff = np.full(len(cols), np.nan)
    share = vk[:, live] / recv[live]
    eff[live] = 1.0 / (share ** 2).sum(0)
    print(f"\n  pieces contributing to a vertex (participation ratio of the variance "
          f"share), over the {int(live.sum())} of {len(cols)} vertices that receive "
          f"anything: median {np.nanmedian(eff):.1f}, "
          f"range {np.nanmin(eff):.1f}-{np.nanmax(eff):.1f}")

    out = os.path.join(RESULTS, f"interference_{a.tag}.npz")
    np.savez(out, ratio=ratio, coherent=coh, incoherent=inc, var_k=var_k,
             eff_pieces=eff, cols=cols, tags=np.array(r["tags"], dtype=object))
    print(f"\n  wrote {out}")

    _plot(a.tag, c, t, ratio, eff, r)


def _plot(tag, c, t, ratio, eff, r):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from render_regimes import _proj
    from plot_fc_map import surface_row
    proj = _proj(c.V, c.F)
    full_eff = np.full(c.nV, np.nan); full_eff[t.cols] = eff
    rows = [("coherent / incoherent", np.clip(ratio[t.cols], 0, 2), "RdBu_r", (0, 2)),
            ("pieces reaching the vertex", eff, "viridis",
             (float(np.percentile(eff, 2)), float(np.percentile(eff, 98))))]
    fig = plt.figure(figsize=(3.9 * len(proj), 2.9 * len(rows)))
    gs = fig.add_gridspec(len(rows), len(proj), hspace=0.06, wspace=0.02)
    for i, (lab, v, cm, lims) in enumerate(rows):
        surface_row(fig, gs, i, proj, v, c, t.cols, cm, lims, lab)
    out = os.path.join(RESULTS, f"interference_{tag}.png")
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
