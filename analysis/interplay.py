"""How pairs of driven pieces shape the waves between them: steer, expand, reinforce, cancel.

The medium is linear, so at one frequency each piece contributes a fixed complex depth
pattern H_k(v) and velocity pattern U_k(v) per unit of drive, and the field is their sum
weighted by the pieces' complex amplitudes z_k. Two readouts per vertex:

    intensity   |h|^2, the depth power
    flux        depth x h x u time-averaged, = depth/2 Re(conj(h) u): the medium's energy
                flux, pointing where the waves carry their energy. Velocity alone sloshes
                back and forth with every crest and averages to ~0 even under a steady
                travelling wave; the flux does not.

Both are QUADRATIC in the inputs, so both split exactly into one term per piece (its own
emission) plus one term per PAIR of pieces (their interference). That is what makes
"which pairs shape the waves" a decomposition rather than a regression.

  A. adjacent pairs, controlled
     Every pair of pieces sharing a border, driven equally with the second lagging by
     lambda, read in the patch within `--radius` mm of either:
       gain     patch power over the sum of the two driven alone. Exactly
                1 + m cos(lambda - lambda*), so each pair has a modulation depth m (how
                much their timing matters for intensity) and a constructive lag lambda*.
       along    flux along the first -> second axis, as a mean cosine: +1 all energy
                moving from the first piece toward the second
       side     the same across that axis
       expand   flux along the direction away from the pair's midpoint: +1 radiating out
       crest    the PHASE direction along the axis - where crests appear to move, which
                near the inputs need not be where the energy goes
  B. the fitted drive, decomposed
     Under the solved cross-spectrum S the time-averaged flux and power are
         sum_k S_kk (own emission)  +  sum_{k<l} 2 Re(S_kl ...)  (pair interference)
     exactly. Per frequency: how much of the net flux and power is interference, how far
     interference turns the flux away from the direction emission alone would give it,
     and which pairs carry it - by spacing, coherence |S_kl| / sqrt(S_kk S_ll) and phase.

Transfer functions come from waves.transfer_fields with velocity=True; they are checked
against a directly driven run in tests/test_waves.py.

  python analysis/interplay.py pairs  --tag nobp_sm6_s3 --hz 0.0106,0.0212,0.0409,0.0848
  python analysis/interplay.py decomp --tag nobp_sm6_s3
"""
import _path  # noqa: F401  - puts the sibling code folders on sys.path
import os, argparse
import numpy as np

from paths import RESULTS
import waves


def _tangent(X, n):
    X = X - (X * n).sum(-1, keepdims=True) * n
    return X / np.maximum(np.linalg.norm(X, axis=-1, keepdims=True), 1e-12)


def geometry(c, S, labels, K):
    """Per-piece distance maps, centroids, and the pairs that share a border."""
    lab = np.asarray(labels)
    D = np.stack([S.distance_from(np.flatnonzero(lab == k)) for k in range(K)])
    A = np.asarray(c.A)
    cen = np.stack([(S.V[lab == k] * A[lab == k, None]).sum(0) / A[lab == k].sum()
                    for k in range(K)])
    E = np.asarray(c.edges)
    a, b = lab[E[:, 0]], lab[E[:, 1]]
    ok = (a != b) & (a >= 0) & (b >= 0)
    pairs = sorted({(int(min(p, q)), int(max(p, q))) for p, q in zip(a[ok], b[ok])})
    return D, cen, pairs


def flux(depth, h, u):
    """Time-averaged energy flux of complex depth h (..., nV) and velocity u (..., nV, 3)."""
    return depth[:, None] * 0.5 * np.real(np.conj(h)[..., None] * u)


# ============================================================ A: adjacent pairs

def pair_sweep(c, S, H, U, depth, i, j, D, cen, radius, lags, hz):
    """One pair at one frequency, every lag. -> dict of (nlag,) arrays and scalars."""
    nrm = np.asarray(c.m.vertex_normal)
    A = np.asarray(c.A)
    pv = np.flatnonzero(np.minimum(D[i], D[j]) <= radius)
    inpatch = np.zeros(c.nV, bool); inpatch[pv] = True
    w = A[pv]
    Hi, Hj, Ui, Uj = H[pv, i], H[pv, j], U[pv, :, i], U[pv, :, j]
    rot = np.exp(-1j * lags)                                          # second lags
    h = Hi[None] + rot[:, None] * Hj[None]                            # (L, nv)
    u = Ui[None] + rot[:, None, None] * Uj[None]                      # (L, nv, 3)
    Pi, Pj = (w * np.abs(Hi) ** 2).sum(), (w * np.abs(Hj) ** 2).sum()
    C = (w * np.conj(Hi) * Hj).sum()
    F = flux(depth[pv], h, u)                                         # (L, nv, 3)
    n = nrm[pv]
    ax = _tangent(np.broadcast_to(cen[j] - cen[i], (len(pv), 3)), n)
    sd = np.cross(n, ax)
    out = _tangent(S.V[pv] - 0.5 * (cen[i] + cen[j]), n)
    Fm = np.linalg.norm(F, axis=-1)
    norm = np.maximum((w * Fm).sum(1), 1e-30)
    proj = lambda a: (w * (F * a).sum(-1)).sum(1) / norm
    # crest direction: phase gradient on the faces wholly inside the patch
    fm = inpatch[S.F].all(1)
    Fp = S.F[fm]
    loc = -np.ones(c.nV, int); loc[pv] = np.arange(len(pv))
    a0, a1, a2 = (h[:, loc[Fp[:, k]]] for k in range(3))
    dph = np.stack([np.angle(a1 * a0.conj()), np.angle(a2 * a0.conj())], -1)
    g = np.einsum("fij,lfj->lfi", S.B[fm], dph)
    kh = -g / np.maximum(np.linalg.norm(g, axis=-1, keepdims=True), 1e-12)
    axf = ax[loc[Fp]].mean(1)
    wf = S.area[fm] * (np.abs(a0) ** 2 + np.abs(a1) ** 2 + np.abs(a2) ** 2)
    crest = (wf * (kh * axf).sum(-1)).sum(1) / np.maximum(wf.sum(1), 1e-30)
    gain = (w * np.abs(h) ** 2).sum(1) / (Pi + Pj)
    return dict(gain=gain, along=proj(ax), side=proj(sd), expand=proj(out), crest=crest,
                flux=norm, m=2 * abs(C) / (Pi + Pj), lam_c=float(np.angle(C)),
                balance=float(min(Pi, Pj) / max(Pi, Pj)),
                spacing=float(np.linalg.norm(cen[i] - cen[j])), nv=len(pv), hz=hz)


def cmd_pairs(a, c, S, run, H, U, depth, hz):
    K = len(run["tags"])
    D, cen, pairs = geometry(c, S, run["labels"], K)
    lags = np.linspace(-np.pi, np.pi, a.nlag, endpoint=False)
    print(f"\n  {len(pairs)} adjacent pairs, patch = within {a.radius:g} mm of either piece; "
          f"+lag = second piece lags")
    res = {}
    for fi, f in enumerate(hz):
        rows = [pair_sweep(c, S, H[fi], U[fi], depth, i, j, D, cen, a.radius, lags, f)
                for i, j in pairs]
        res[f] = rows
        m = np.array([r["m"] for r in rows])
        lc = np.array([r["lam_c"] for r in rows])
        al = np.stack([r["along"] for r in rows])
        cr = np.stack([r["crest"] for r in rows])
        ex = np.stack([r["expand"] for r in rows])
        sw = lambda X: 0.5 * (X.max(1) - X.min(1))
        print(f"\n  {f:.4f} Hz (period {1/f:.0f} s)")
        print(f"    intensity: modulation depth m median {np.median(m):.2f} "
              f"(IQR {np.percentile(m,25):.2f}-{np.percentile(m,75):.2f}); constructive "
              f"lag median |{np.degrees(np.median(np.abs(lc))):.0f}| deg")
        print(f"    flux along the pair axis: swing over lag median {np.median(sw(al)):.2f}; "
              f"mean curve " + " ".join(f"{np.degrees(L):+.0f}:{v:+.2f}"
                                       for L, v in zip(lags[::a.nlag // 8], al.mean(0)[::a.nlag // 8])))
        print(f"    crests along the axis: swing median {np.median(sw(cr)):.2f}; mean curve "
              + " ".join(f"{np.degrees(L):+.0f}:{v:+.2f}"
                         for L, v in zip(lags[::a.nlag // 8], cr.mean(0)[::a.nlag // 8])))
        print(f"    expansion: median {np.median(ex):+.2f}, swing median {np.median(sw(ex)):.2f}")
        sp = np.array([r["spacing"] for r in rows])
        print(f"    swing of flux direction vs spacing: r = "
              f"{np.corrcoef(sp, sw(al))[0,1]:+.2f}; vs balance of the two: r = "
              f"{np.corrcoef([r['balance'] for r in rows], sw(al))[0,1]:+.2f}")
    path = os.path.join(RESULTS, f"interplay_pairs_{run['name']}.npz")
    np.savez(path, res=np.array([res], object), pairs=np.array(pairs), lags=lags, hz=hz,
             tags=np.array(run["tags"], object), radius=a.radius)
    print(f"\n  wrote {path}")
    _plot_pairs(path.replace(".npz", ".png"), res, lags, hz)


# ============================================================ B: the fitted drive

def decompose(c, H, U, depth, Sf, outward):
    """Exact split of time-averaged flux and power under cross-spectrum Sf, one frequency.

    -> dict: per-vertex total / own-emission flux and power, and per-pair signed
    contributions to the cortex-integrated net flux (projected on the total's direction
    at each vertex) and to the integrated power. E[z z^H] = Sf, so
        E[conj(h) u] = sum_kl conj(H_k) U_l Sf[l, k]
        E[|h|^2]     = sum_kl H_k conj(H_l) Sf[k, l]"""
    A = np.asarray(c.A)
    a_ = np.conj(H) @ Sf.T                                           # (nV, K): sum_k conj(H_k) S_lk
    Ft = depth[:, None] * 0.5 * np.real(np.einsum("vdl,vl->vd", U, a_))
    d = np.real(np.diag(Sf))
    Fs = depth[:, None] * 0.5 * np.real(np.einsum("vdk,vk->vd", U, np.conj(H) * d))
    Pt = np.real(np.einsum("vk,kl,vl->v", H, Sf, np.conj(H)))
    Ps = (np.abs(H) ** 2 * d).sum(1)
    # per-pair contributions to the integrated net flux along the total's own direction:
    # sum_v A F_tot . Fhat = sum_v A |F_tot|, split exactly over k, l
    Fh = Ft / np.maximum(np.linalg.norm(Ft, axis=1, keepdims=True), 1e-30)
    Ut = np.einsum("vdk,vd->vk", U, Fh)
    M = (np.conj(H) * (A * depth * 0.5)[:, None]).T @ Ut             # (K, K): M[k,l]
    CF = np.real(M * Sf.T)                                            # k,l term, Re(M_kl S_lk)
    G = (H * A[:, None]).T @ np.conj(H)                               # sum_v A H_k conj(H_l)
    CP = np.real(G * Sf)
    return dict(Ft=Ft, Fs=Fs, Pt=Pt, Ps=Ps, CF=CF + CF.T, CP=CP + CP.T,
                out_t=(Ft * outward).sum(1), out_s=(Fs * outward).sum(1))


def cmd_decomp(a, c, S, run, H, U, depth, hz, Sb):
    K = len(run["tags"])
    D, cen, adj = geometry(c, S, run["labels"], K)
    A = np.asarray(c.A)
    lab = np.asarray(run["labels"])
    driven = np.flatnonzero(lab >= 0)
    dterr = S.distance_from(driven)
    nrm = np.asarray(c.m.vertex_normal)
    outward = _tangent(np.asarray(S.FtoV @ S.axis(dterr)), nrm)
    near = dterr <= a.radius
    iu = np.triu_indices(K, 1)
    dist = np.linalg.norm(cen[:, None] - cen[None], axis=-1)[iu]
    isadj = np.zeros((K, K), bool)
    for i, j in adj:
        isadj[i, j] = True
    isadj = isadj[iu]
    out = []
    print(f"\n  {len(hz)} solved bins; 'near' = within {a.radius:g} mm of the driven territory")
    for fi, f in enumerate(hz):
        Sf = 0.5 * (Sb[fi] + Sb[fi].conj().T)
        r = decompose(c, H[fi], U[fi], depth, Sf, outward)
        Ft, Fs = r["Ft"], r["Fs"]
        mt, ms = np.linalg.norm(Ft, axis=1), np.linalg.norm(Fs, axis=1)
        cosang = (Ft * Fs).sum(1) / np.maximum(mt * ms, 1e-30)
        P_t, P_s = (A * r["Pt"]).sum(), (A * r["Ps"]).sum()
        net = (A * mt).sum()
        own = np.real(np.diag(r["CF"])) / 2                               # diagonal double counted
        cross = r["CF"][iu]
        coh = np.abs(Sf[iu]) / np.sqrt(np.outer(np.real(np.diag(Sf)), np.real(np.diag(Sf)))[iu] + 1e-30)
        cp = r["CP"][iu]
        # the split is exact by construction; check it rather than trust it
        assert abs(own.sum() + cross.sum() - net) < 1e-3 * net, "flux split does not close"
        assert abs(np.real(np.diag(r["CP"])).sum() / 2 + cp.sum() - P_t) < 1e-3 * P_t

        def zone(m):
            w = A[m] * mt[m]
            return (np.sum(w * cosang[m]) / max(w.sum(), 1e-30),
                    np.sum(A[m] * r["out_t"][m]) / max(np.sum(A[m] * mt[m]), 1e-30),
                    np.sum(A[m] * r["out_s"][m]) / max(np.sum(A[m] * ms[m]), 1e-30),
                    np.sum(A[m] * mt[m]) / max(np.sum(A[m] * ms[m]), 1e-30))
        zn, zf = zone(near), zone(~near)
        print(f"\n  {f:.4f} Hz (period {1/f:.0f} s): field power {P_t:.3g}")
        print(f"    power: interference {100*(P_t-P_s)/max(P_t,1e-30):+.0f}% of the total "
              f"(coherent / incoherent {P_t/max(P_s,1e-30):.2f})")
        print(f"    net flux: own emission {100*own.sum()/max(net,1e-30):.0f}%, pair "
              f"interference {100*cross.sum()/max(net,1e-30):.0f}%")
        for nm, z in (("near", zn), ("far", zf)):
            print(f"    {nm:4s}: flux turned by interference, mean cos(total, own) "
                  f"{z[0]:+.2f}; outward alignment total {z[1]:+.2f} vs own {z[2]:+.2f}; "
                  f"|flux| total / own {z[3]:.2f}")
        bins = [("adjacent", isadj), ("<20 mm, not adj", (~isadj) & (dist < 20)),
                ("20-40 mm", (dist >= 20) & (dist < 40)), (">40 mm", dist >= 40)]
        print("    interference by pair class: share of net flux | of power | mean coherence"
              " | n pairs")
        for nm, m in bins:
            print(f"      {nm:16s} {100*cross[m].sum()/max(net,1e-30):+6.1f}%  "
                  f"{100*cp[m].sum()/max(P_t,1e-30):+6.1f}%  {coh[m].mean():.2f}  {m.sum()}")
        o = np.argsort(-np.abs(cross))[:a.top]
        print("    pairs carrying most net flux (signed share, spacing, coherence, phase of "
              "S_kl):")
        for q in o:
            k, l = iu[0][q], iu[1][q]
            print(f"      {run['tags'][k]:>12s} - {run['tags'][l]:<12s} "
                  f"{100*cross[q]/net:+5.1f}%  {dist[q]:5.1f} mm  coh {coh[q]:.2f}  "
                  f"{np.degrees(np.angle(Sf[k, l])):+4.0f} deg")
        out.append(dict(hz=f, Ft=Ft.astype(np.float32), Fs=Fs.astype(np.float32),
                        Pt=r["Pt"], Ps=r["Ps"], cross=cross, own=own, cp=cp, coh=coh,
                        phase=np.angle(Sf[iu]), dist=dist, isadj=isadj, net=net, P_t=P_t,
                        P_s=P_s, cosang=cosang))
    path = os.path.join(RESULTS, f"interplay_decomp_{run['name']}.npz")
    np.savez(path, res=np.array(out, object), tags=np.array(run["tags"], object),
             radius=a.radius)
    print(f"\n  wrote {path}")
    _plot_decomp(path.replace(".npz", ".png"), out, c, near)


# ============================================================ pictures

def _plot_pairs(path, res, lags, hz):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = (("gain", "intensity gain"), ("along", "flux along first->second"),
            ("crest", "crests along first->second"), ("expand", "flux outward"))
    fig, axs = plt.subplots(1, len(keys), figsize=(3.4 * len(keys), 2.9))
    for ax, (k, t) in zip(axs, keys):
        for f in hz:
            X = np.stack([r[k] for r in res[f]])
            ax.plot(np.degrees(lags), X.mean(0), label=f"{f:.3f} Hz")
            ax.fill_between(np.degrees(lags), np.percentile(X, 25, 0),
                            np.percentile(X, 75, 0), alpha=0.12)
        ax.axvline(0, color="0.7", lw=0.6); ax.axhline(1 if k == "gain" else 0,
                                                       color="0.7", lw=0.6)
        ax.set_title(t, fontsize=9); ax.set_xlabel("lag of second piece (deg)", fontsize=8)
        ax.tick_params(labelsize=7)
    axs[0].legend(fontsize=6)
    fig.suptitle("adjacent piece pairs: mean and IQR over pairs", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor="white")
    plt.close(fig)
    print(f"  wrote {path}")


def _plot_decomp(path, out, c, near):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from render_regimes import _proj
    from plot_fc_map import surface_row
    hz = np.array([r["hz"] for r in out])
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.0))
    axs[0].semilogx(hz, [100 * (r["P_t"] - r["P_s"]) / r["P_t"] for r in out], "o-",
                    label="power")
    axs[0].semilogx(hz, [100 * r["cross"].sum() / r["net"] for r in out], "s-",
                    label="net flux")
    axs[0].axhline(0, color="0.7", lw=0.6)
    axs[0].set_title("share from pair interference (%)", fontsize=9); axs[0].legend(fontsize=7)
    for r in out:
        d = r["dist"]; bins = np.arange(0, 80, 5)
        s = [r["cross"][(d >= b) & (d < b + 5)].sum() / r["net"] * 100 for b in bins]
        axs[1].plot(bins + 2.5, s, label=f"{r['hz']:.3f}")
    axs[1].set_title("interference share of net flux by pair spacing (mm)", fontsize=9)
    axs[1].legend(fontsize=5, ncol=2)
    w = [np.sum(np.linalg.norm(r["Ft"], axis=1)[near] * r["cosang"][near]) /
         np.sum(np.linalg.norm(r["Ft"], axis=1)[near]) for r in out]
    wf = [np.sum(np.linalg.norm(r["Ft"], axis=1)[~near] * r["cosang"][~near]) /
          np.sum(np.linalg.norm(r["Ft"], axis=1)[~near]) for r in out]
    axs[2].semilogx(hz, w, "o-", label="near"); axs[2].semilogx(hz, wf, "s-", label="far")
    axs[2].set_title("mean cos(total flux, own emission)", fontsize=9)
    axs[2].legend(fontsize=7); axs[2].set_ylim(-1, 1)
    for ax in axs:
        ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor="white")
    plt.close(fig)
    print(f"  wrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=("pairs", "decomp"))
    ap.add_argument("--tag", default="nobp_sm6_s3")
    ap.add_argument("--hz", default="",
                    help="analysis frequencies; snapped to the nearest solved bins. Default: "
                         "14 bins log-spaced 0.005-0.5 Hz")
    ap.add_argument("--radius", type=float, default=10.0, help="mm")
    ap.add_argument("--nlag", type=int, default=24)
    ap.add_argument("--top", type=int, default=10)
    a = ap.parse_args()
    from mesh_cache import load_cortex
    c = load_cortex("fsaverage5", verbose=False)
    run = waves.load_run(c, a.tag)
    hzb = run["idx"] / run["ref_frames"] / run["frame_s"]
    want = ([float(v) for v in a.hz.split(",")] if a.hz else np.geomspace(0.005, 0.5, 14))
    pick = np.unique([int(np.argmin(np.abs(np.log(hzb / f)))) for f in want])
    # one transfer computation serves both commands: always ask for the default grid plus
    # whatever was requested, so the cache is shared
    grid = np.unique(np.r_[[int(np.argmin(np.abs(np.log(hzb / f))))
                            for f in np.geomspace(0.005, 0.5, 14)], pick])
    H, U, depth = waves.transfer_fields(c, run, hzb[grid], velocity=True)
    sel = np.searchsorted(grid, pick)
    S = waves.Surface(c)
    print(f"  {run['name']}: {len(run['tags'])} pieces, bins "
          + ", ".join(f"{hzb[b]:.4f}" for b in pick) + " Hz")
    if a.cmd == "pairs":
        cmd_pairs(a, c, S, run, H[sel], U[sel], depth, hzb[pick])
    else:
        cmd_decomp(a, c, S, run, H[sel], U[sel], depth, hzb[pick], run["S"][pick])


if __name__ == "__main__":
    main()
