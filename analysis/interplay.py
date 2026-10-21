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
from scipy.sparse.csgraph import dijkstra
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


# ============================================================ C: frame by frame

def record_run(c, run, A_frames, keep, verbose=True):
    """Integrate a frame-constant drive and keep depth and gathered velocity at `keep`,
    read as fluid.run reads frames (snapshot, or the frame mean under decimate="mean").
    -> h (T, nk), u (T, nk, 3). The run itself, not a reconstruction: the exact total
    that the per-piece split below must add back up to."""
    import fluid as fl
    p, save = run["p"], run["save"]
    s, dt, g, Hd = fl.build(c, p, sponge=True)
    P = np.asarray(run["profiles"], np.float32)
    mean = p.get("decimate", "snapshot") == "mean"
    T, nk = len(A_frames), len(keep)
    hs = np.empty((T, nk), np.float32)
    us = np.empty((T, nk, 3), np.float32)
    h = np.zeros(c.nV, np.float32); ue = np.zeros(s.nE, np.float32)
    ah = np.zeros(nk, np.float32); au = np.zeros((nk, 3), np.float32)
    src = (np.asarray(A_frames, np.float32) / np.float32(save)) @ P   # (T, nV) per step
    dt = np.float32(dt)
    for n in range(T * save):
        h += src[n // save]
        ue, h = s.step(ue, h, dt, g, Hd)
        if mean:
            ah += h[keep]; au += s.gather(ue)[keep]
        if n % save == 0:
            fi = n // save
            if mean:
                d = 1 if n == 0 else save
                hs[fi], us[fi] = ah / d, au / d
                ah[:] = 0; au[:] = 0
            else:
                hs[fi], us[fi] = h[keep], s.gather(ue)[keep]
        if verbose and n % (save * 2000) == 0:
            print(f"    run: frame {n // save}/{T}", flush=True)
    return hs, us


def impulse_hu(c, run, keep, verbose=True):
    """Depth and velocity responses of every piece at `keep`, to one frame of drive,
    read as record_run reads: -> Rh (nimp, nk, K), Ru (nimp, nk, 3, K).

    The same experiment as xspec.impulse_series, all K pieces integrated at once through
    RotSWE.step_b, with velocity kept as well."""
    import fluid as fl
    p, save, nimp = run["p"], run["save"], run["nimp"]
    s, dt, g, Hd = fl.build(c, p, sponge=True)
    P = np.asarray(run["profiles"], np.float32)
    K, nk = len(P), len(keep)
    mean = p.get("decimate", "snapshot") == "mean"
    Pt = np.ascontiguousarray(P.T) / np.float32(save)
    h = np.zeros((c.nV, K), np.float32); ue = np.zeros((s.nE, K), np.float32)
    Rh = np.empty((nimp, nk, K), np.float32)
    Ru = np.empty((nimp, nk, 3, K), np.float32)
    ah = np.zeros((nk, K), np.float32); au = np.zeros((nk, 3, K), np.float32)
    dt = np.float32(dt); Hd = Hd[:, None]
    for n in range(nimp * save):
        if n < save:
            h += Pt
        ue, h = s.step_b(ue, h, dt, g, Hd)
        if mean:
            ah += h[keep]; au += s.gather_b(ue)[keep]
        if n % save == 0:
            f = n // save
            if mean:
                Rh[f], Ru[f] = ah / save, au / save     # shift-invariant, see impulse_series
                ah[:] = 0; au[:] = 0
            else:
                Rh[f], Ru[f] = h[keep], s.gather_b(ue)[keep]
        if verbose and n % (save * 100) == 0:
            print(f"    impulses: frame {n // save}/{nimp}", flush=True)
    return Rh, Ru


def make_patches(c, S, run, D, cen, pairs, radius, grid_spacing, reach):
    """Pair patches (every adjacent pair, within `radius` of either piece) and grid
    patches (centres spaced `grid_spacing` apart over the territory and `reach` mm round
    it, each the vertices within `radius`). -> list of dicts."""
    nrm = np.asarray(c.m.vertex_normal)
    lab = np.asarray(run["labels"])
    out = []
    for i, j in pairs:
        pv = np.flatnonzero(np.minimum(D[i], D[j]) <= radius)
        out.append(dict(kind="pair", i=i, j=j, pv=pv, axis=cen[j] - cen[i],
                        centre=0.5 * (cen[i] + cen[j])))
    dterr = np.min(D, 0)
    cand = np.flatnonzero(dterr <= reach)
    # farthest-point sampling at the requested spacing
    cent = [cand[np.argmin(dterr[cand])]]
    dmin = np.linalg.norm(S.V[cand] - S.V[cent[0]], axis=1)
    while dmin.max() > grid_spacing:
        q = cand[np.argmax(dmin)]
        cent.append(q)
        dmin = np.minimum(dmin, np.linalg.norm(S.V[cand] - S.V[q], axis=1))
    G = S.graph()
    for q in cent:
        d = dijkstra(G, indices=[q], limit=radius)[0]
        pv = np.flatnonzero(np.isfinite(d))
        out.append(dict(kind="grid", pv=pv, axis=np.array([0.0, 1.0, 0.0]),
                        centre=S.V[q], vertex=int(q),
                        inside=bool(lab[q] >= 0)))
    for P in out:
        n = nrm[P["pv"]].mean(0); n /= np.linalg.norm(n)
        e1 = P["axis"] - (P["axis"] @ n) * n; e1 /= np.linalg.norm(e1)
        P["e1"], P["e2"], P["n"] = e1, np.cross(n, e1), n
    return out


def _smooth(x, n):
    """Moving average over n frames along axis -1 (linear, so splits stay exact)."""
    from scipy.ndimage import uniform_filter1d
    return uniform_filter1d(x, size=max(1, int(n)), axis=-1, mode="nearest")


def patch_terms(P, loc, hs, us, Rh, Ru, A, depth, area, nrm, V, band, frame_s, cover,
                maxp, smooth_n):
    """Split one patch's flux and power, frame by frame, into own and pair terms.

    -> dict with the term labels (piece indices, -1 for the rest) and, per term pair
    (k <= l), series (T,) of: flux along e1, along e2, outward; and power."""
    import bandpass
    pv = loc[P["pv"]]
    w = area[P["pv"]]
    # which pieces reach the patch: impulse energy here, largest first, to `cover`
    en = np.einsum("tvk,v->k", Rh[:, pv] ** 2, w)
    o = np.argsort(en)[::-1]
    cum = np.cumsum(en[o]) / en.sum()
    ks = list(o[:max(2, min(maxp, int(np.searchsorted(cum, cover)) + 1))])
    for q in ("i", "j"):
        if q in P and P[q] not in ks:
            ks.append(P[q])
    T = len(A)
    nfft = 1 << int(np.ceil(np.log2(T + Rh.shape[0])))
    Ah = np.fft.rfft(A[:, ks], nfft, axis=0)                          # (nb, m)
    # per-vertex projections of velocity: on e1, e2 and away from the patch centre
    out_dir = V[P["pv"]] - P["centre"]
    n_v = nrm[P["pv"]]
    out_dir = out_dir - (out_dir * n_v).sum(1, keepdims=True) * n_v
    out_dir /= np.maximum(np.linalg.norm(out_dir, axis=1, keepdims=True), 1e-9)
    proj = np.stack([np.broadcast_to(P["e1"], out_dir.shape),
                     np.broadcast_to(P["e2"], out_dir.shape), out_dir], 1)   # (nv, 3dir, 3)
    def conv(R):
        Rf = np.fft.rfft(R, nfft, axis=0)
        return np.fft.irfft(Rf * Ah.reshape((Ah.shape[0],) + (1,) * (R.ndim - 2) + (-1,)),
                            nfft, axis=0)[:T]
    hk = conv(Rh[:, pv][:, :, ks])                                    # (T, nv, m)
    uk = conv(Ru[:, pv][:, :, :, ks])                                 # (T, nv, 3, m)
    uk = np.einsum("tvcm,vdc->tvdm", uk, proj)                        # (T, nv, 3dir, m)
    ht = hs[:, pv]
    ut = np.einsum("tvc,vdc->tvd", us[:, pv], proj)
    resid = float(np.linalg.norm(ht - hk.sum(-1)) / np.linalg.norm(ht))
    # the rest: everything the named pieces do not account for, so the split is exact
    hk = np.concatenate([hk, (ht - hk.sum(-1))[..., None]], -1)
    uk = np.concatenate([uk, (ut - uk.sum(-1))[..., None]], -1)
    if band is not None:
        hk = bandpass.apply(hk.reshape(T, -1), frame_s, *band).reshape(hk.shape)
        uk = bandpass.apply(uk.reshape(T, -1), frame_s, *band).reshape(uk.shape)
    m = hk.shape[-1]
    wd = (w * depth[P["pv"]]).astype(np.float32)
    # F[k, l, d, t] = sum_v w depth h_k u_l(d): one batched product per direction
    F = np.stack([np.einsum("tvk,tvl->klt", hk * wd[None, :, None], uk[:, :, d])
                  for d in range(3)], 2)                              # (m, m, 3, T)
    Pw = np.einsum("tvk,tvl->klt", hk * w[None, :, None], hk)         # (m, m, T)
    iu = np.triu_indices(m, 1)
    own = np.stack([F[k, k] for k in range(m)])                       # (m, 3, T)
    pair = F[iu] + F[(iu[1], iu[0])]                                  # (npair, 3, T)
    pown = np.stack([Pw[k, k] for k in range(m)])
    ppair = 2 * Pw[iu]
    own, pair = _smooth(own, smooth_n), _smooth(pair, smooth_n)
    pown, ppair = _smooth(pown, smooth_n), _smooth(ppair, smooth_n)
    lab = np.array(list(ks) + [-1])
    return dict(own=own, pair=pair, pown=pown, ppair=ppair, labels=lab,
                pairs=np.stack([lab[iu[0]], lab[iu[1]]], 1), resid=resid, ks=ks)


def summarise(P, r, A_an, edge):
    """Per patch: how the flux turns over time, how much of it pairs carry, and - for a
    pair patch - whether the two pieces' instantaneous lag sets its direction."""
    sl = slice(edge, r["own"].shape[-1] - edge)
    tot = r["own"].sum(0) + r["pair"].sum(0)                          # (3, T)
    own = r["own"].sum(0)
    v = tot[:2, sl]
    mag = np.linalg.norm(v, axis=0)
    th = np.arctan2(v[1], v[0])
    fh = v / np.maximum(mag, 1e-30)
    c_own = (r["own"][:, :2, sl] * fh).sum(1)                         # (m, T)
    c_pair = (r["pair"][:, :2, sl] * fh).sum(1)                       # (np, T)
    w = mag / mag.sum()
    o = own[:2, sl]
    cos_turn = (o * v).sum(0) / np.maximum(np.linalg.norm(o, axis=0) * mag, 1e-30)
    pw_tot = r["pown"].sum(0)[sl] + r["ppair"].sum(0)[sl]
    dom = np.argmax(np.abs(c_pair), 0)
    out = dict(kind=P["kind"], resid=r["resid"], n_terms=len(r["labels"]),
               # how much the direction wanders: 1 - |mean unit vector|, flux-weighted
               wander=1 - float(np.abs((w * np.exp(1j * th)).sum())),
               # signed: the pair terms' net push along the flux, as a share of it
               pair_share=float(c_pair.sum() / max(mag.sum(), 1e-30)),
               pair_share_abs=float(np.abs(c_pair).sum() / max(np.abs(c_pair).sum()
                                                             + np.abs(c_own).sum(), 1e-30)),
               turn=float((w * cos_turn).sum()),
               power_pair=float(r["ppair"].sum(0)[sl].sum() / max(pw_tot.sum(), 1e-30)),
               expand=float((tot[2, sl]).sum() / max(np.abs(tot[2, sl]).sum(), 1e-30)),
               pair_contrib=np.abs(c_pair).sum(1) / max(np.abs(c_pair).sum(), 1e-30),
               dominant=np.bincount(dom, minlength=len(r["pairs"])) / len(dom),
               pairs=r["pairs"], theta=th.astype(np.float32), mag=mag.astype(np.float32))
    if P["kind"] == "pair":
        i, j = P["i"], P["j"]
        # +lag = second piece lags, as in A: z_j = z_i exp(-i lag)
        lag = -np.angle(A_an[sl, j] * np.conj(A_an[sl, i]))
        # the pair term is symmetric in its two pieces; e1 already points i -> j
        pp = r["pairs"]
        q = int(np.flatnonzero(((pp[:, 0] == i) & (pp[:, 1] == j)) |
                               ((pp[:, 0] == j) & (pp[:, 1] == i)))[0])
        prn = r["pair"][q, 0, sl]
        corr = lambda a, b: float(np.corrcoef(a, b)[0, 1]) if np.std(b) > 0 else np.nan
        out.update(lag=lag.astype(np.float32),
                   r_pairterm=corr(np.sin(lag), prn),
                   r_total=corr(np.sin(lag), v[0] / np.maximum(mag, 1e-30)),
                   r_power=corr(np.cos(lag), pw_tot))
    return out


def cmd_frames(a, c, S, run):
    from scipy.signal import hilbert
    import xspec, bandpass
    K = len(run["tags"])
    D, cen, pairs = geometry(c, S, run["labels"], K)
    patches = make_patches(c, S, run, D, cen, pairs, a.radius, a.grid, a.reach)
    keep = np.unique(np.concatenate([P["pv"] for P in patches]))
    loc = -np.ones(c.nV, int); loc[keep] = np.arange(len(keep))
    fs = run["frame_s"]
    T = int(round(a.seconds / fs))
    band = None if a.band == "none" else tuple(float(v) for v in a.band.split(","))
    print(f"\n  {sum(P['kind'] == 'pair' for P in patches)} pair patches, "
          f"{sum(P['kind'] == 'grid' for P in patches)} grid patches, {len(keep)} vertices "
          f"read; {a.seconds:g} s realisation ({T} frames), band {band}")
    A = xspec.realise(run["S"], run["idx"], T, ref_frames=run["ref_frames"], seed=a.seed)
    # the run and the responses are the slow part; keep them between passes
    import hashlib
    ck = hashlib.md5(keep.tobytes()).hexdigest()[:10]
    cdir = a.cache or os.path.join(RESULTS, "interplay_cache")
    os.makedirs(cdir, exist_ok=True)
    fr = os.path.join(cdir, f"run_{run['name']}_{a.seconds:g}s_seed{a.seed}_{ck}.npz")
    fi = os.path.join(cdir, f"imp_{run['name']}_{run['nimp']}_{ck}.npz")
    if os.path.exists(fr):
        z = np.load(fr); hs, us = z["hs"], z["us"]
    else:
        print("  simulating the realisation once (the exact total) ...", flush=True)
        hs, us = record_run(c, run, A, keep)
        np.savez(fr, hs=hs, us=us)
    if os.path.exists(fi):
        z = np.load(fi); Rh, Ru = z["Rh"], z["Ru"]
    else:
        print("  impulse responses with velocity ...", flush=True)
        Rh, Ru = impulse_hu(c, run, keep)
        np.savez(fi, Rh=Rh, Ru=Ru)
    depth = np.asarray(__import__("fluid").build(c, run["p"], sponge=True)[3], float)
    # the split rests on convolution reproducing the run: check it on one patch with
    # every piece named, before reading anything off it
    pv0 = loc[patches[0]["pv"]]
    nfft = 1 << int(np.ceil(np.log2(T + Rh.shape[0])))
    rec = np.fft.irfft((np.fft.rfft(Rh[:, pv0], nfft, axis=0)
                        * np.fft.rfft(A, nfft, axis=0)[:, None, :]).sum(-1), nfft, axis=0)[:T]
    e0 = Rh.shape[0]
    rr = float(np.corrcoef(rec[e0:].ravel(), hs[e0:, pv0].ravel())[0, 1])
    print(f"  convolution against the run, all {K} pieces, one patch: r = {rr:.6f}, "
          f"sd ratio {rec[e0:].std() / hs[e0:, pv0].std():.4f}")
    if rr < 0.999:
        raise SystemExit("  the responses do not reproduce the run")
    Ab = bandpass.apply(A, fs, *band) if band else A
    A_an = hilbert(Ab, axis=0)
    smooth_n = int(round(1.0 / (fs * (np.mean(band) if band else 0.1))))
    edge = max(Rh.shape[0], int(60 / fs))
    nrm = np.asarray(c.m.vertex_normal)
    res = []
    for n, P in enumerate(patches):
        r = patch_terms(P, loc, hs, us, Rh, Ru, A, depth, np.asarray(c.A), nrm, S.V,
                        band, fs, a.cover, a.maxp, smooth_n)
        sm = summarise(P, r, A_an, edge)
        sm.update({k: P[k] for k in ("i", "j", "vertex", "inside") if k in P})
        sm["ks"] = r["ks"]
        res.append(sm)
        if (n + 1) % 25 == 0:
            print(f"    {n+1}/{len(patches)} patches", flush=True)
    rs = np.array([s["resid"] for s in res])
    print(f"\n  named pieces account for the field to within {np.median(rs):.1%} median "
          f"(max {rs.max():.1%}); the rest term carries the remainder exactly")
    for kind in ("pair", "grid"):
        R = [s for s in res if s["kind"] == kind]
        g = lambda k: np.array([s[k] for s in R], float)
        print(f"\n  {kind} patches ({len(R)}):")
        print(f"    direction wanders (1 - |mean unit flux|): median {np.median(g('wander')):.2f}")
        print(f"    pair terms' share of |flux| along its direction: median "
              f"{np.median(g('pair_share_abs')):.2f}; turn (cos total vs own) median "
              f"{np.median(g('turn')):+.2f}")
        print(f"    power from pair interference: median {np.median(g('power_pair')):+.2f}")
        if kind == "pair":
            print(f"    sin(lag) of the two pieces vs their pair term along the axis: r "
                  f"median {np.nanmedian(g('r_pairterm')):+.2f} "
                  f"(IQR {np.nanpercentile(g('r_pairterm'),25):+.2f} to "
                  f"{np.nanpercentile(g('r_pairterm'),75):+.2f})")
            print(f"    ... vs the patch's TOTAL flux along the axis: r median "
                  f"{np.nanmedian(g('r_total')):+.2f}; cos(lag) vs patch power: r median "
                  f"{np.nanmedian(g('r_power')):+.2f}")
    path = os.path.join(RESULTS, f"interplay_frames_{run['name']}.npz")
    np.savez(path, res=np.array(res, object), tags=np.array(run["tags"], object),
             band=np.array(band if band else (np.nan, np.nan)), seconds=a.seconds,
             seed=a.seed, radius=a.radius)
    print(f"\n  wrote {path}")
    _plot_frames(path.replace(".npz", ".png"), res, run, fs, edge)


# ============================================================ D: pattern types

TYPES = ("+axis", "+side", "-axis", "-side", "radiate", "converge", "mixed")


def band_fields(Rh, Ru, A, fs, band, block=160, verbose=True):
    """Total depth and velocity in `band` at every kept vertex, for drive A, without
    integrating anything: per frequency the field is sum_k H_k(v, f) A_k(f), and a drive
    from xspec.realise is periodic by construction (an irfft), so multiplying spectra is
    the steady state exactly, with no ends to trim.

    -> X (T, nk, 4) float32 [h, u_x, u_y, u_z], E (nk, K) each piece's in-band depth
    power at each vertex under this drive, and the analytic in-band drive Z (T, K).
    The band is the zero-phase filter's own gain, bandpass.response, so this equals
    bandpass.apply on the simulated run away from its ends."""
    import bandpass
    T, K = A.shape
    nimp, nk = Rh.shape[0], Rh.shape[1]
    f = np.fft.rfftfreq(T, d=fs)
    g = bandpass.response(f, fs, *band)
    sel = np.flatnonzero(g > 1e-4)
    Ah = (np.fft.rfft(A, axis=0)[sel] * g[sel, None]).astype(np.complex64)   # (nb, K)
    Ef = np.exp(-2j * np.pi * np.outer(f[sel] * fs, np.arange(nimp))).astype(np.complex64)
    X = np.empty((T, nk, 4), np.float32)
    En = np.empty((nk, K), np.float32)
    for b0 in range(0, nk, block):
        b1 = min(nk, b0 + block)
        R = np.concatenate([Rh[:, b0:b1, None, :], Ru[:, b0:b1]], 2)          # (nimp, nb, 4, K)
        Hf = (Ef @ R.reshape(nimp, -1)).reshape(len(sel), b1 - b0, 4, K)
        HA = Hf * Ah[:, None, None, :]
        En[b0:b1] = (np.abs(HA[:, :, 0]) ** 2).sum(0)
        spec = np.zeros((len(f), b1 - b0, 4), np.complex64)
        spec[sel] = HA.sum(-1)
        X[:, b0:b1] = np.fft.irfft(spec, n=T, axis=0)
        if verbose and (b0 // block) % 8 == 0:
            print(f"    band fields: vertex {b0}/{nk}", flush=True)
    full = np.zeros((T, K), np.complex64)
    full[sel] = 2 * Ah                                                # one-sided: analytic
    Z = np.fft.ifft(full, axis=0)
    return X, En, Z


def patch_state(P, loc, X, depth, area, nrm, V, smooth_n, every):
    """Directedness, direction, expansion and intensity of a patch's flux, per frame.

    Directedness is |net flux| over the total flux magnitude: 1 when every vertex carries
    energy the same way, ~0 when it radiates or converges or is disordered. Expansion is
    the flux along the direction away from the patch centre over the same total: +1 all
    leaving, -1 all arriving. Smoothed over one period of the band's centre so the
    twice-per-cycle ripple of a product of two oscillations does not set the state."""
    pv = loc[P["pv"]]
    wd = (area[P["pv"]] * depth[P["pv"]]).astype(np.float32)
    h, u = X[:, pv, 0], X[:, pv, 1:]
    n_v = nrm[P["pv"]]
    od = V[P["pv"]] - P["centre"]
    od = od - (od * n_v).sum(1, keepdims=True) * n_v
    od /= np.maximum(np.linalg.norm(od, axis=1, keepdims=True), 1e-9)
    Fv = (h * wd)[..., None] * u                                      # (T, nv, 3)
    F1, F2 = Fv @ P["e1"], Fv @ P["e2"]
    Fo = np.einsum("tvc,vc->tv", Fv, od)
    mag = np.linalg.norm(Fv, axis=-1).sum(1)
    S = [_smooth(x, smooth_n) for x in (F1.sum(1), F2.sum(1), Fo.sum(1), mag,
                                         (h ** 2 * area[P["pv"]]).sum(1))]
    F1, F2, Fo, mag, I = (x[::every] for x in S)
    m = np.maximum(mag, 1e-30)
    return dict(dirn=np.hypot(F1, F2) / m, theta=np.arctan2(F2, F1), expand=Fo / m,
                power=I)


def wave_types(st, floor=0.3):
    """One label per frame. Provisional thresholds: below `floor` on both directedness and
    |expansion| the patch is 'mixed'; otherwise whichever is larger names it, translating
    states by the quarter of the compass their direction falls in, relative to the patch
    axis (the pair axis, or anterior for a grid patch)."""
    d, x, th = st["dirn"], st["expand"], np.degrees(st["theta"])
    lab = np.full(len(d), "mixed", dtype=object)
    tr = (d >= floor) & (d >= np.abs(x))
    rad = (np.abs(x) >= floor) & (np.abs(x) > d)
    sector = np.where(np.abs(th) <= 45, "+axis", np.where(th > 135, "-axis",
                      np.where(th < -135, "-axis", np.where(th > 0, "+side", "-side"))))
    lab[tr] = sector[tr]
    lab[rad & (x > 0)] = "radiate"
    lab[rad & (x < 0)] = "converge"
    return lab


def input_features(Z, ks):
    """What a patch's pieces are doing, as the flux actually depends on it.

    In a narrow band the flux is a Hermitian form in the pieces' complex amplitudes z,
    Re(z^H M z): a sum over PAIRS of |z_k| |z_l| cos(phase_k - phase_l + c_kl). A linear
    model in each piece's own phase and amplitude cannot form those products - that was
    the first version, and on 40 patches it reached median R2 0.26 for the along-axis
    direction where these reach 0.46. So the features are pairwise:

      timing  cos and sin of every pairwise phase difference, amplitudes ignored
      loud    each piece's share of the patch's input power, phases ignored
      both    the normalised cross-products z_k conj(z_l) / sum |z|^2 and the shares:
              timing weighted by loudness, which is the form the flux takes

    Also returns each piece's phase against the first, for pattern-triggered averages."""
    z = Z[:, ks]
    m = len(ks)
    iu = np.triu_indices(m, 1)
    tot = np.maximum((np.abs(z) ** 2).sum(1, keepdims=True), 1e-30)
    cr = z[:, iu[0]] * np.conj(z[:, iu[1]])
    dph = np.angle(cr)
    share = np.abs(z) ** 2 / tot
    timing = np.c_[np.cos(dph), np.sin(dph)]
    both = np.c_[share, cr.real / tot, cr.imag / tot]
    ph = np.angle(z[:, 1:] * np.conj(z[:, :1]))
    return timing, share, both, ph


def classify(Xf, y, folds=5):
    """Blocked cross-validated balanced accuracy of a multinomial logistic fit."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict, KFold
    from sklearn.metrics import balanced_accuracy_score
    if len(np.unique(y)) < 2:
        return np.nan
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    # standardised first: the pairwise products are collinear and badly scaled, and
    # unscaled the solver took ~50 s a patch to converge
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    pred = cross_val_predict(model, Xf, y, cv=KFold(folds))
    return float(balanced_accuracy_score(y, pred))


def cmd_types(a, c, S, run):
    import xspec, hashlib
    K = len(run["tags"])
    D, cen, pairs = geometry(c, S, run["labels"], K)
    patches = make_patches(c, S, run, D, cen, pairs, a.radius, a.grid, a.reach)
    keep = np.unique(np.concatenate([P["pv"] for P in patches]))
    loc = -np.ones(c.nV, int); loc[keep] = np.arange(len(keep))
    fs = run["frame_s"]
    band = tuple(float(v) for v in a.band.split(","))
    ck = hashlib.md5(keep.tobytes()).hexdigest()[:10]
    cdir = a.cache or os.path.join(RESULTS, "interplay_cache")
    os.makedirs(cdir, exist_ok=True)
    fi = os.path.join(cdir, f"imp_{run['name']}_{run['nimp']}_{ck}.npz")
    if os.path.exists(fi):
        z = np.load(fi); Rh, Ru = z["Rh"], z["Ru"]
    else:
        print("  impulse responses with velocity ...", flush=True)
        Rh, Ru = impulse_hu(c, run, keep)
        np.savez(fi, Rh=Rh, Ru=Ru)
    depth = np.asarray(__import__("fluid").build(c, run["p"], sponge=True)[3], float)
    # the spectral construction has to equal the simulated, bandpassed field: check it on
    # a run that was actually integrated, when one is cached
    fr = os.path.join(cdir, f"run_{run['name']}_1154s_seed0_{ck}.npz")
    if os.path.exists(fr):
        import bandpass
        T0 = int(round(1154 / fs))
        A0 = xspec.realise(run["S"], run["idx"], T0, ref_frames=run["ref_frames"], seed=0)
        v0 = loc[patches[0]["pv"]]
        X0, _, _ = band_fields(Rh[:, v0], Ru[:, v0], A0, fs, band, verbose=False)
        hs = np.load(fr)["hs"][:, v0]
        ref = bandpass.apply(hs, fs, *band)
        e = int(200 / fs)
        rr = float(np.corrcoef(X0[e:-e, :, 0].ravel(), ref[e:-e].ravel())[0, 1])
        print(f"  spectral field against the integrated run, bandpassed, one patch: "
              f"r = {rr:.5f}, sd ratio {X0[e:-e, :, 0].std() / ref[e:-e].std():.4f}")
        if rr < 0.995:
            raise SystemExit("  the spectral construction does not reproduce the run")
    T = int(round(a.seconds / fs))
    A = xspec.realise(run["S"], run["idx"], T, ref_frames=run["ref_frames"], seed=a.seed)
    fb = os.path.join(cdir, f"band_{run['name']}_{a.seconds:g}s_seed{a.seed}_"
                      f"{band[0]:g}-{band[1]:g}_{ck}.npz")
    if os.path.exists(fb):
        z = np.load(fb); X, En, Z = z["X"], z["En"], z["Z"]
    else:
        print(f"  {a.seconds:g} s realisation ({T} frames), band {band}: building the "
              f"field at {len(keep)} vertices ...", flush=True)
        X, En, Z = band_fields(Rh, Ru, A, fs, band)
        np.savez(fb, X=X, En=En, Z=Z)
    del Rh, Ru
    smooth_n = int(round(1.0 / (fs * np.mean(band))))
    every = max(1, int(round(a.every_s / fs)))
    nrm = np.asarray(c.m.vertex_normal)
    area = np.asarray(c.A)
    n_eff = a.seconds * (band[1] - band[0])
    print(f"  ~{n_eff:.0f} independent states in band; states read every {every} frames "
          f"({every * fs:.2f} s), smoothed over {smooth_n}")
    res = []
    for n, P in enumerate(patches):
        st = patch_state(P, loc, X, depth, area, nrm, S.V, smooth_n, every)
        lab = wave_types(st, a.floor)
        en = (En[loc[P["pv"]]] * area[P["pv"], None]).sum(0)
        ks = list(np.argsort(en)[::-1][:a.ninputs])
        Tf, Af, Bf, ph = input_features(Z[::every][:len(lab)], ks)
        # classes rarer than 3% are left out of the classifier, not merged
        cls, cnt = np.unique(lab, return_counts=True)
        ok = np.isin(lab, cls[cnt >= 0.03 * len(lab)])
        r = dict(kind=P["kind"], ks=ks, occ={t: float(np.mean(lab == t)) for t in TYPES},
                 n_cls=int(len(np.unique(lab[ok]))))
        r["acc_t"] = classify(Tf[ok], lab[ok])
        r["acc_a"] = classify(Af[ok], lab[ok])
        r["acc_b"] = classify(Bf[ok], lab[ok])
        cont = dict(along=st["dirn"] * np.cos(st["theta"]),
                    across=st["dirn"] * np.sin(st["theta"]), expand=st["expand"],
                    power=np.log(np.maximum(st["power"], 1e-30)))
        for nm, y in cont.items():
            r[f"r2t_{nm}"] = waves.explained(Tf, y, blocked=True)
            r[f"r2a_{nm}"] = waves.explained(Af, y, blocked=True)
            r[f"r2b_{nm}"] = waves.explained(Bf, y, blocked=True)
        # pattern-triggered relative phases: per type, circular mean and consistency of
        # each piece's phase against the first
        trig = {}
        for t in TYPES:
            m = lab == t
            if m.sum() >= 20:
                zz = np.exp(1j * ph[m]).mean(0)
                trig[t] = (np.angle(zz), np.abs(zz))
        r["trig"] = trig
        r.update({k: P[k] for k in ("i", "j", "vertex", "inside") if k in P})
        r["state"] = {k: v.astype(np.float32) for k, v in st.items()}
        r["labels"] = lab
        res.append(r)
        if (n + 1) % 50 == 0:
            print(f"    {n+1}/{len(patches)} patches", flush=True)
    for kind in ("grid", "pair"):
        R = [r for r in res if r["kind"] == kind]
        g = lambda k: np.array([r[k] for r in R], float)
        occ = {t: np.mean([r["occ"][t] for r in R]) for t in TYPES}
        chance = np.array([1.0 / max(r["n_cls"], 1) for r in R])
        print(f"\n  {kind} patches ({len(R)}): mean occupancy "
              + ", ".join(f"{t} {100*occ[t]:.0f}%" for t in TYPES))
        print(f"    type predicted from (balanced accuracy, chance median "
              f"{np.median(chance):.2f}): timing {np.nanmedian(g('acc_t')):.2f}, "
              f"loudness {np.nanmedian(g('acc_a')):.2f}, both {np.nanmedian(g('acc_b')):.2f}; "
              f"timing beats loudness in {np.mean(g('acc_t') > g('acc_a')):.0%} of patches")
        for nm in ("along", "across", "expand", "power"):
            print(f"    {nm:7s} R2 timing {np.median(g('r2t_'+nm)):.2f}  loudness "
                  f"{np.median(g('r2a_'+nm)):.2f}  both {np.median(g('r2b_'+nm)):.2f}")
    path = os.path.join(RESULTS, f"interplay_types_{run['name']}.npz")
    np.savez(path, res=np.array(res, object), tags=np.array(run["tags"], object),
             band=np.array(band), seconds=a.seconds, seed=a.seed, floor=a.floor,
             every=every)
    print(f"\n  wrote {path}")
    _plot_types(path.replace(".npz", ".png"), res, run)


# ============================================================ E: steering modes

READOUTS = ("along", "across", "expand")


def steering_matrices(P, H, U, depth, area, nrm, V):
    """Per readout, the Hermitian K x K matrix M with time-averaged patch flux
    = Re(z^H M z) exactly, for complex input amplitudes z at this frequency; and G with
    patch power = z^H G z.

    The flux at a vertex is depth/2 Re(conj(h) u), h = H z and u = U z, so along a
    direction d it is z^H B z / 2 with B = sum_v w depth conj(H_v)^T (U_v . d); its real
    part is the Hermitian part of B. The readouts are the patch axis e1, the side axis e2,
    and the direction away from the patch centre at each vertex."""
    pv = P["pv"]
    wd = area[pv] * depth[pv]
    Hp, Up = H[pv], U[pv]                                            # (nv, K), (nv, 3, K)
    od = V[pv] - P["centre"]
    n_v = nrm[pv]
    od = od - (od * n_v).sum(1, keepdims=True) * n_v
    od /= np.maximum(np.linalg.norm(od, axis=1, keepdims=True), 1e-9)
    dirs = dict(along=np.broadcast_to(P["e1"], od.shape),
                across=np.broadcast_to(P["e2"], od.shape), expand=od)
    out = {}
    Hw = np.conj(Hp) * wd[:, None]
    for nm, d in dirs.items():
        Ud = np.einsum("vck,vc->vk", Up, d)
        B = Hw.T @ Ud
        out[nm] = 0.25 * (B + B.conj().T)                            # 1/2 flux, 1/2 Hermitian
    out["G"] = (np.conj(Hp) * area[pv][:, None]).T @ Hp
    return out


def mode_shape(x, cen, P, hz, readout="along"):
    """What an input combination is: how many pieces, how far away, and whether it is a
    SWEEP in the readout's own direction - the downstream pieces lagging.

    sweep: sum over pairs of |x_k|^2 |x_l|^2 sign(s_l - s_k) sin(phi_k - phi_l), over the
    same with |sin|. s is position along e1 for `along`, along e2 for `across`, and
    distance from the patch centre for `expand`. +1 when every piece further along lags
    every one before it - an input wave moving the way the readout counts as positive,
    or for `expand` the centre leading and the surround following; -1 the reverse."""
    w = np.abs(x) ** 2
    pr = 1.0 / np.sum(w ** 2)
    rel = cen - P["centre"]
    s = {"along": rel @ P["e1"], "across": rel @ P["e2"],
         "expand": np.linalg.norm(rel, axis=1)}[readout]
    dist = np.linalg.norm(cen - P["centre"], axis=1)
    phi = np.angle(x)
    top = np.argsort(w)[::-1][:12]
    num = den = 0.0
    for a_ in range(len(top)):
        for b_ in range(a_ + 1, len(top)):
            k, l = top[a_], top[b_]
            if abs(s[l] - s[k]) < 3:
                continue
            ww = w[k] * w[l]
            sn = np.sin(phi[k] - phi[l])
            num += ww * np.sign(s[l] - s[k]) * sn
            den += ww * abs(sn)
    return dict(pr=float(pr), reach=float((w * dist).sum()), sweep=float(num / den) if den
                else np.nan, top=top[:6], amp=np.abs(x[top[:6]]),
                lag_s=(np.angle(x[top[:6]] * np.conj(x[top[0]])) / (2 * np.pi * hz)))


def cmd_steer(a, c, S, run):
    import glob
    K = len(run["tags"])
    D, cen, pairs = geometry(c, S, run["labels"], K)
    patches = make_patches(c, S, run, D, cen, pairs, a.radius, a.grid, a.reach)
    hzb = run["idx"] / run["ref_frames"] / run["frame_s"]
    grid = np.unique([int(np.argmin(np.abs(np.log(hzb / f)))) for f in np.geomspace(0.005, 0.5, 14)])
    H, U, depth = waves.transfer_fields(c, run, hzb[grid], velocity=True)
    want = [float(v) for v in a.hz.split(",")] if a.hz else [0.059, 0.0848, 0.1226]
    sel = [int(np.argmin(np.abs(np.log(hzb[grid] / f)))) for f in want]
    area, nrm = np.asarray(c.A), np.asarray(c.m.vertex_normal)
    res = []
    for fi in sel:
        hz, b = hzb[grid][fi], grid[fi]
        Sf = 0.5 * (run["S"][b] + run["S"][b].conj().T)
        iso = np.trace(Sf).real / K
        rows = []
        for P in patches:
            M = steering_matrices(P, H[fi], U[fi], depth, area, nrm, S.V)
            r = dict(kind=P["kind"])
            for nm in READOUTS:
                ev, X = np.linalg.eigh(M[nm])
                for sgn, j in (("+", -1), ("-", 0)):
                    x = X[:, j]
                    sh = mode_shape(x, cen, P, hz, nm)
                    key = nm + sgn
                    r[key] = dict(lam=float(ev[j]), use=float((x.conj() @ Sf @ x).real / iso),
                                  **sh)
                # how the fitted drive pushes this readout, and how hard it could
                r[nm + "_drive"] = float(np.trace(M[nm] @ Sf).real)
                r[nm + "_range"] = (float(ev[0]), float(ev[-1]))
            r["M"] = {nm: M[nm].astype(np.complex64) for nm in READOUTS}
            r.update({k: P[k] for k in ("i", "j", "vertex") if k in P})
            rows.append(r)
        res.append(dict(hz=float(hz), rows=rows))
        print(f"\n  {hz:.4f} Hz (period {1/hz:.0f} s)")
        for kind in ("pair", "grid"):
            R = [r for r in rows if r["kind"] == kind]
            for nm in READOUTS:
                sw = np.array([r[nm + "+"]["sweep"] for r in R])
                swm = np.array([r[nm + "-"]["sweep"] for r in R])
                pr = np.array([r[nm + "+"]["pr"] for r in R])
                reach = np.array([r[nm + "+"]["reach"] for r in R])
                up = np.array([r[nm + "+"]["use"] for r in R])
                um = np.array([r[nm + "-"]["use"] for r in R])
                asym = np.array([r[nm + "_range"][1] / max(-r[nm + "_range"][0], 1e-30)
                                 for r in R])
                print(f"    {kind} {nm:6s}: + mode {np.median(pr):4.1f} pieces, centred "
                      f"{np.median(reach):4.1f} mm away; sweep + {np.nanmedian(sw):+.2f} / "
                      f"- {np.nanmedian(swm):+.2f}; |+|/|-| {np.median(asym):.2f}; fitted "
                      f"drive uses + x{np.median(up):.2f}, - x{np.median(um):.2f}")
    for q in res[:len(sel) // 2] + res[len(sel) // 2 + 1:]:
        for r in q["rows"]:
            r.pop("M", None)                       # kept only where they are used
    # the exact quadratic form against the realised states: no fitting at all
    tf = glob.glob(os.path.join(RESULTS, f"interplay_types_{run['name']}.npz"))
    cdir = a.cache or os.path.join(RESULTS, "interplay_cache")
    bf = glob.glob(os.path.join(cdir, f"band_{run['name']}_*s_seed*_*.npz"))
    pred = None
    if tf and bf:
        zt = np.load(tf[0], allow_pickle=True)
        every = int(zt["every"])
        Z = np.load(bf[0], mmap_mode="r")["Z"]
        smooth_n = int(round(1.0 / (run["frame_s"] * np.mean(zt["band"]))))
        mid = sel[len(sel) // 2]
        rows = res[len(sel) // 2]["rows"]
        pred = []
        for r, t in zip(rows, zt["res"]):
            Fp = [_smooth(np.real(np.einsum("tk,tk->t", np.conj(Z), Z @ r["M"][nm].T)),
                          smooth_n)[::every] for nm in ("along", "across")]
            n = len(t["state"]["theta"])
            thp = np.arctan2(Fp[1][:n], Fp[0][:n])
            th = t["state"]["theta"]
            w = t["state"]["dirn"]
            agree = float((w * np.cos(thp - th)).sum() / w.sum())
            rc = float(np.corrcoef(np.cos(thp), np.cos(th))[0, 1])
            pred.append(dict(kind=r["kind"], agree=agree, r_cos=rc))
        for kind in ("pair", "grid"):
            P_ = [p for p in pred if p["kind"] == kind]
            print(f"\n  {kind}: flux direction predicted from all {K} inputs by the "
                  f"{hzb[grid][mid]:.4f} Hz form, no fitting: mean cos(predicted - actual) "
                  f"median {np.median([p['agree'] for p in P_]):+.2f}; r(cos) median "
                  f"{np.median([p['r_cos'] for p in P_]):+.2f}")
    path = os.path.join(RESULTS, f"interplay_steer_{run['name']}.npz")
    np.savez(path, res=np.array(res, object), pred=np.array(pred, object),
             tags=np.array(run["tags"], object), cen=cen)
    print(f"\n  wrote {path}")
    _plot_steer(path.replace(".npz", ".png"), res, pred, c, run, cen, patches)


# ============================================================ F: steering by pathway

# the validated sequential blue ramp (dataviz reference palette), light -> dark
BLUES = ("#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
         "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b")
STEER = {"direction": ("along+", "along-", "across+", "across-"),
         "radiate / converge": ("expand+", "expand-")}


def pathway_shares(M, groups, names):
    """Share of each steering mode's power carried by each pathway's pieces.
    -> dict mode -> (len(names),) shares summing to 1."""
    out = {}
    G = np.stack([np.asarray([g == nm for g in groups], float) for nm in names])
    for nm in READOUTS:
        ev, X = np.linalg.eigh(np.asarray(M[nm], complex))
        for sgn, j in (("+", -1), ("-", 0)):
            w = np.abs(X[:, j]) ** 2
            out[nm + sgn] = G @ w / w.sum()
    return out


def cross_sweep(M, groups, cen, P):
    """For the along+ mode: the share of its pairwise sweep weight that comes from pairs
    of pieces in DIFFERENT pathways - how much steering needs timing across pathways."""
    ev, X = np.linalg.eigh(np.asarray(M["along"], complex))
    x = X[:, -1]
    w = np.abs(x) ** 2
    top = np.argsort(w)[::-1][:12]
    s = (cen - P["centre"]) @ P["e1"]
    tot = cross = 0.0
    for a_ in range(len(top)):
        for b_ in range(a_ + 1, len(top)):
            k, l = top[a_], top[b_]
            if abs(s[l] - s[k]) < 3:
                continue
            ww = w[k] * w[l] * abs(np.sin(np.angle(x[k]) - np.angle(x[l])))
            tot += ww
            cross += ww * (groups[k] != groups[l])
    return cross / tot if tot else np.nan


def cmd_steermap(a, c, S, run):
    K = len(run["tags"])
    z = np.load(os.path.join(RESULTS, f"interplay_steer_{run['name']}.npz"),
                allow_pickle=True)
    q = next(r for r in z["res"] if "M" in r["rows"][0])
    rows, hz = q["rows"], q["hz"]
    D, cen, pairs = geometry(c, S, run["labels"], K)
    patches = make_patches(c, S, run, D, cen, pairs, a.radius, a.grid, a.reach)
    groups = waves.pathway_groups(c, run["labels"], K)
    names = sorted(set(groups) - {"?"})
    # each patch's home: the pathway of the nearest driven piece to its centre
    home = []
    for P in patches:
        v = int(np.argmin(np.linalg.norm(S.V - P["centre"], axis=1)))
        home.append(groups[int(np.argmin(D[:, v]))])
    shares = [pathway_shares(r["M"], groups, names) for r in rows]
    xs = np.array([cross_sweep(r["M"], groups, cen, P) for r, P in zip(rows, patches)])
    print(f"\n  {len(patches)} patches, steering modes at {hz:.4f} Hz, {len(names)} pathways")
    print(f"  along+ sweeps: share of their timing weight between DIFFERENT pathways, "
          f"median {np.nanmedian(xs):.2f} (IQR {np.nanpercentile(xs, 25):.2f}-"
          f"{np.nanpercentile(xs, 75):.2f})")
    mats = {}
    for title, modes in STEER.items():
        Mx = np.zeros((len(names), len(names)))
        for hi, h in enumerate(names):
            idx = [n for n, hh in enumerate(home) if hh == h]
            if idx:
                Mx[hi] = np.mean([np.mean([shares[n][m] for m in modes], 0) for n in idx], 0)
        mats[title] = Mx
        print(f"\n  {title}: share of the steering modes by pathway (rows: where the patch "
              f"is; columns: whose inputs steer it)")
        print("  " + " " * 15 + " ".join(f"{nm[:9]:>9s}" for nm in names) + "   own")
        for hi, h in enumerate(names):
            n = sum(hh == h for hh in home)
            print(f"  {h[:13]:>13s} " + " ".join(f"{v:9.2f}" for v in Mx[hi])
                  + f"   {Mx[hi, hi]:.2f}  ({n} patches)")
    path = os.path.join(RESULTS, f"interplay_steermap_{run['name']}.npz")
    np.savez(path, shares=np.array(shares, object), home=np.array(home, object),
             names=np.array(names, object), mats=np.array([mats], object),
             cross_sweep=xs, centres=np.array([P["centre"] for P in patches]), hz=hz)
    print(f"\n  wrote {path}")
    _plot_steermap(path.replace(".npz", ".png"), c, run, patches, shares, names, groups,
                   mats, hz)


# ============================================================ G: steering as input sweeps

def area_metric(c, P):
    """W[k, l] = sum_v A_v P_k(v) P_l(v): injected power per mm^2 of cortex, so a mode
    normalised in it is a timing pattern per unit of injection over the surface, whatever
    the number and size of the pieces that carry it."""
    Pa = np.asarray(P, float) * np.sqrt(np.asarray(c.A))[None, :]
    return Pa @ Pa.T


def area_modes(M, W):
    """Extreme generalised eigenvectors of (M, W): the per-area steering modes.
    -> {"+": (lam, x), "-": (lam, x)}."""
    from scipy.linalg import eigh
    ev, X = eigh(np.asarray(M, complex), W + 1e-9 * np.trace(W) / len(W) * np.eye(len(W)))
    return {"+": (float(ev[-1]), X[:, -1]), "-": (float(ev[0]), X[:, 0])}


def medium_phase_speed(S, H, lab, hz, nsamp=25, seed=0):
    """The medium's own wave phase speed and wavelength at this frequency: |grad arg H|
    of single pieces' responses, 5-30 mm from the piece, median over pieces."""
    rng = np.random.default_rng(seed)
    K = H.shape[1]
    g = []
    for k in rng.choice(K, min(nsamp, K), replace=False):
        d = S.distance_from(np.flatnonzero(lab == k))
        gr = np.linalg.norm(S.phase_grad(H[:, k][None])[0], axis=1)
        dfc = d[S.F].mean(1)
        g.append(np.median(gr[(dfc > 5) & (dfc < 30)]))
    k_ = float(np.median(g))
    return 2 * np.pi * hz / k_, 2 * np.pi / k_


def cmd_sweeps(a, c, S, run):
    import glob
    from scipy.sparse.csgraph import dijkstra as _dj
    K = len(run["tags"])
    z = np.load(os.path.join(RESULTS, f"interplay_steer_{run['name']}.npz"), allow_pickle=True)
    q = next(r for r in z["res"] if "M" in r["rows"][0])
    rows, hz = q["rows"], float(q["hz"])
    D, cen, pairs = geometry(c, S, run["labels"], K)
    patches = make_patches(c, S, run, D, cen, pairs, a.radius, a.grid, a.reach)
    Pf = np.asarray(run["profiles"], np.float32)
    W = area_metric(c, Pf)
    lab = np.asarray(run["labels"])
    hzb = run["idx"] / run["ref_frames"] / run["frame_s"]
    grid = np.unique([int(np.argmin(np.abs(np.log(hzb / f)))) for f in np.geomspace(0.005, 0.5, 14)])
    H, _, _ = waves.transfer_fields(c, run, hzb[grid], velocity=True)
    fi = int(np.argmin(np.abs(hzb[grid] - hz)))
    c_med, lam_med = medium_phase_speed(S, H[fi], lab, hz)
    print(f"\n  {hz:.4f} Hz: the medium's own waves travel at {c_med:.1f} mm/s, "
          f"wavelength {lam_med:.0f} mm (inflated-surface mm)")
    smooth = {sg: waves.Surface(c, radius=sg).K for sg in sorted({a.sigma, 4.0, 9.0})}
    G = S.graph()
    res = []
    for n, (P, r) in enumerate(zip(patches, rows)):
        cvx = int(np.argmin(np.linalg.norm(S.V - P["centre"], axis=1)))
        dpc = _dj(G, indices=[cvx])[0]
        ax = dict(e1=S.axis(S.V @ P["e1"]), e2=S.axis(S.V @ P["e2"]), out=S.axis(dpc))
        out = dict(kind=P["kind"])
        for nm in READOUTS:
            md = area_modes(r["M"][nm], W)
            for sgn in ("+", "-"):
                lam, x = md[sgn]
                qv = (x @ Pf).astype(np.complex64)                      # injection map
                for sg, Ks in smooth.items():
                    qs = np.asarray(Ks @ qv)
                    d = waves.describe(S, qs[None], hz, ax)
                    a2 = np.abs(qs) ** 2 * np.asarray(c.A)
                    w = a2 / a2.sum()
                    rec = dict(lam=lam, speed=float(d["speed"][0]),
                               planar=float(d["planar"][0]),
                               a_e1=float(d["align_e1"][0]), a_e2=float(d["align_e2"][0]),
                               a_out=float(d["align_out"][0]),
                               extent=float(np.sqrt((w * dpc ** 2).sum())),
                               centroid=float((w * dpc).sum()))
                    out[f"{nm}{sgn}@{sg:g}"] = rec
                    if sg == a.sigma:
                        out[f"{nm}{sgn}"] = rec
                        if nm == "along" and sgn == "+":
                            out["q_along+"] = qs.astype(np.complex64)
                        if sgn == "+":
                            out[f"w_{nm}"] = (np.abs(qs[S.F]) ** 2).mean(1).astype(np.float32)
        out.update({k: P[k] for k in ("i", "j", "vertex") if k in P})
        res.append(out)
        if (n + 1) % 50 == 0:
            print(f"    {n+1}/{len(patches)} patches", flush=True)
    # what the modes are, as sweeps
    want = {"along": "a_e1", "across": "a_e2", "expand": "a_out"}
    for kind in ("pair", "grid"):
        R = [r for r in res if r["kind"] == kind]
        print(f"\n  {kind} patches ({len(R)}), per-area modes smoothed at {a.sigma:g} mm:")
        for nm in READOUTS:
            for sgn in ("+", "-"):
                k = f"{nm}{sgn}"
                g = lambda f: np.array([r[k][f] for r in R])
                print(f"    {k:8s} sweep along its own direction {np.median(g(want[nm])):+.2f}; "
                      f"speed {np.median(g('speed')):5.1f} mm/s "
                      f"(x{np.median(g('speed')) / c_med:.2f} the medium's); planar "
                      f"{np.median(g('planar')):.2f}; aimed at the patch "
                      f"{-np.median(g('a_out')):+.2f}; extent {np.median(g('extent')):4.0f} mm, "
                      f"centred {np.median(g('centroid')):4.0f} mm away")
    for sg in sorted(smooth):
        R = res
        print(f"  smoothing {sg:g} mm: along+ sweep {np.median([r[f'along+@{sg:g}']['a_e1'] for r in R]):+.2f}, "
              f"speed {np.median([r[f'along+@{sg:g}']['speed'] for r in R]):.1f} mm/s, "
              f"planar {np.median([r[f'along+@{sg:g}']['planar'] for r in R]):.2f}")
    # the fitted drive's own sweeps, in the region that can steer each patch
    pred = None
    tf = os.path.join(RESULTS, f"interplay_types_{run['name']}.npz")
    cdir = a.cache or os.path.join(RESULTS, "interplay_cache")
    bf = glob.glob(os.path.join(cdir, f"band_{run['name']}_*s_seed*_*.npz"))
    if os.path.exists(tf) and bf:
        zt = np.load(tf, allow_pickle=True)
        every = int(zt["every"])
        Z = np.load(bf[0], mmap_mode="r")["Z"][::every]
        Ks = smooth[a.sigma]
        n_t = min(len(Z), len(zt["res"][0]["state"]["theta"]))
        Wm = {nm: np.stack([r[f"w_{nm}"] for r in res]) for nm in READOUTS}
        Wm = {nm: v / v.sum(1, keepdims=True) for nm, v in Wm.items()}
        E1 = np.stack([S.axis(S.V @ P["e1"]) for P in patches])          # (np, nF, 3)
        E2 = np.stack([S.axis(S.V @ P["e2"]) for P in patches])
        num = {nm: np.zeros((len(patches), n_t)) for nm in READOUTS}
        den = {nm: np.zeros((len(patches), n_t)) for nm in READOUTS}
        outf = []
        for P in patches:
            cvx = int(np.argmin(np.linalg.norm(S.V - P["centre"], axis=1)))
            outf.append(S.axis(_dj(G, indices=[cvx])[0]))
        outf = np.stack(outf)
        E1, E2, outf = (x.astype(np.float32) for x in (E1, E2, outf))
        Wm = {k: v.astype(np.float32) for k, v in Wm.items()}
        for t0 in range(0, n_t, 16):                     # (patches x frames x faces) per step
            t1 = min(n_t, t0 + 16)
            qd = np.asarray(Ks @ (np.asarray(Z[t0:t1]) @ Pf).T).T               # (n, nV)
            g = S.phase_grad(qd).astype(np.float32)
            kh = -g / np.maximum(np.linalg.norm(g, axis=-1, keepdims=True), 1e-12)
            a2 = (np.abs(qd[:, S.F]) ** 2).mean(-1)                              # (n, nF)
            for nm, Ax in (("along", E1), ("across", E2), ("expand", outf)):
                proj = np.einsum("nfc,pfc->pnf", kh, Ax)                         # (np, n, nF)
                wgt = Wm[nm][:, None, :] * a2[None]
                num[nm][:, t0:t1] = (wgt * proj).sum(-1)
                den[nm][:, t0:t1] = wgt.sum(-1)
        sm = int(round(1.0 / (run["frame_s"] * np.mean(zt["band"])) / every))
        pred = []
        for p, (r, t) in enumerate(zip(res, zt["res"])):
            st = t["state"]
            act = dict(along=(st["dirn"] * np.cos(st["theta"]))[:n_t],
                       across=(st["dirn"] * np.sin(st["theta"]))[:n_t],
                       expand=st["expand"][:n_t])
            rec = dict(kind=r["kind"])
            for nm in READOUTS:
                sw = _smooth(num[nm][p] / np.maximum(den[nm][p], 1e-30), sm)
                rec[nm] = float(np.corrcoef(sw, act[nm])[0, 1])
            pred.append(rec)
        for kind in ("pair", "grid"):
            P_ = [p for p in pred if p["kind"] == kind]
            print(f"\n  {kind}: the fitted drive's input sweep, in the region that can steer "
                  f"the patch, against the patch's actual waves (r, median): "
                  + ", ".join(f"{nm} {np.nanmedian([p[nm] for p in P_]):+.2f}"
                              for nm in READOUTS))
    path = os.path.join(RESULTS, f"interplay_sweeps_{run['name']}.npz")
    np.savez(path, res=np.array(res, object), pred=np.array(pred, object), hz=hz,
             c_med=c_med, lam_med=lam_med, sigma=a.sigma)
    print(f"\n  wrote {path}")
    _plot_sweeps(path.replace(".npz", ".png"), res, pred, c, S, patches, hz, c_med)


# ============================================================ H: remote cortex

def cortex_patches(c, S, spacing, radius, axis_map):
    """Patches over the WHOLE cortex: centres by farthest-point sampling `spacing` mm
    apart, each the vertices within `radius`. The patch axis e1 is the direction up
    `axis_map` (the principal gradient), averaged over the patch, so flux along +e1 is
    flux climbing the gradient."""
    from scipy.sparse.csgraph import dijkstra as _dj
    nrm = np.asarray(c.m.vertex_normal)
    gface = S.scalar_grad(axis_map)
    cent = [int(np.argmin(axis_map))]
    dmin = np.linalg.norm(S.V - S.V[cent[0]], axis=1)
    while dmin.max() > spacing:
        q = int(np.argmax(dmin)); cent.append(q)
        dmin = np.minimum(dmin, np.linalg.norm(S.V - S.V[q], axis=1))
    G = S.graph()
    out = []
    for q in cent:
        d = _dj(G, indices=[q], limit=radius)[0]
        pv = np.flatnonzero(np.isfinite(d))
        inside = np.isin(S.F, pv).all(1)
        n = nrm[pv].mean(0); n /= np.linalg.norm(n)
        ax = gface[inside].mean(0) if inside.any() else np.array([0.0, 1.0, 0.0])
        ax = ax - (ax @ n) * n
        steep = float(np.linalg.norm(ax))
        if steep < 1e-6:
            ax = np.array([0.0, 1.0, 0.0]) - n[1] * n
        e1 = ax / np.linalg.norm(ax)
        out.append(dict(kind="cortex", pv=pv, centre=S.V[q], vertex=int(q), e1=e1,
                        e2=np.cross(n, e1), n=n, steep=steep))
    return out


def source_map(M, Sf, n_hat):
    """Exact split of a patch's net time-averaged flux over the inputs, under drive Sf.

    C[k, l] = Re(M_net[k, l] Sf[l, k]) with M_net the flux matrix along the net flux
    direction, so sum C = |net flux|. Each input gets its own term and half of every
    pair term it is in: c_k = (C[k, :].sum() + C[:, k].sum()) / 2, sum c_k = |net flux|.
    -> c (K,), own (K,), pair share (pair terms over the net)."""
    Mn = n_hat[0] * M["along"] + n_hat[1] * M["across"]
    C = np.real(Mn * Sf.T)
    c = 0.5 * (C.sum(1) + C.sum(0))
    own = np.real(np.diag(C))
    net = C.sum()
    return c, own, float((net - own.sum()) / net) if net else np.nan


def source_regions(c, adj, frac=0.15):
    """Contiguous regions of inputs that carry a patch's flux, found from its own map:
    inputs with |c_k| >= frac * max|c|, joined when their pieces share a border.
    -> list of (members, summed contribution), largest |contribution| first."""
    K = len(c)
    big = np.abs(c) >= frac * np.abs(c).max()
    parent = list(range(K))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for i, j in adj:
        if big[i] and big[j] and np.sign(c[i]) == np.sign(c[j]):
            parent[find(i)] = find(j)
    groups = {}
    for k in np.flatnonzero(big):
        groups.setdefault(find(k), []).append(int(k))
    out = [(m, float(c[m].sum())) for m in groups.values()]
    return sorted(out, key=lambda t: -abs(t[1]))


def cmd_remote(a, c, S, run):
    K = len(run["tags"])
    D, cen, adj = geometry(c, S, run["labels"], K)
    g1 = np.load(os.path.join(os.path.dirname(RESULTS), "data", "cache",
                              f"cortical_maps_{c.mesh}.npz"))["grad1"]
    patches = cortex_patches(c, S, a.grid, a.radius, g1)
    lab = np.asarray(run["labels"])
    dterr = S.distance_from(np.flatnonzero(lab >= 0))
    names = [str(v).removeprefix("L_").removesuffix("_ROI") for v in c.names]
    for P in patches:
        P["grad"] = float(g1[P["pv"]].mean())
        P["dist"] = float(dterr[P["vertex"]])
        P["name"] = names[int(c.lab[P["vertex"]])]
    hzb = run["idx"] / run["ref_frames"] / run["frame_s"]
    grid = np.unique([int(np.argmin(np.abs(np.log(hzb / f)))) for f in np.geomspace(0.005, 0.5, 14)])
    H, U, depth = waves.transfer_fields(c, run, hzb[grid], velocity=True)
    want = [float(v) for v in a.hz.split(",")] if a.hz else [0.0212, 0.0409, 0.059, 0.0848, 0.1226]
    sel = [int(np.argmin(np.abs(np.log(hzb[grid] / f)))) for f in want]
    area, nrm = np.asarray(c.A), np.asarray(c.m.vertex_normal)
    gr = np.array([P["grad"] for P in patches]); dist = np.array([P["dist"] for P in patches])
    print(f"\n  {len(patches)} patches over the whole cortex; principal gradient "
          f"{gr.min():+.2f} to {gr.max():+.2f}; distance from input median {np.median(dist):.0f} mm, "
          f"max {dist.max():.0f} mm")
    res = []
    for fi in sel:
        hz, b = float(hzb[grid][fi]), grid[fi]
        Sf = 0.5 * (run["S"][b] + run["S"][b].conj().T)
        rows = []
        for P in patches:
            M = steering_matrices(P, H[fi], U[fi], depth, area, nrm, S.V)
            Fa, Fc = (float(np.real(np.trace(M[k] @ Sf))) for k in ("along", "across"))
            Fx = float(np.real(np.trace(M["expand"] @ Sf)))
            power = float(np.real(np.trace(M["G"] @ Sf)))
            # magnitude of the expected flux vertex by vertex, for directedness
            pv = P["pv"]
            HS = np.conj(H[fi][pv]) @ Sf.T                                # (nv, K)
            Fv = depth[pv, None] * 0.5 * np.real(np.einsum("vck,vk->vc", U[fi][pv], HS))
            tot = float((area[pv] * np.linalg.norm(Fv, axis=1)).sum())
            mag = float(np.hypot(Fa, Fc))
            n_hat = np.array([Fa, Fc]) / max(mag, 1e-30)
            cmap_, own, pshare = source_map(M, Sf, n_hat)
            regs = source_regions(cmap_, adj)
            w = np.abs(cmap_) / max(np.abs(cmap_).sum(), 1e-30)
            # source distances from the patch along the surface
            dsrc = D[:, P["vertex"]]
            rows.append(dict(
                up=Fa / max(mag, 1e-30), theta=float(np.arctan2(Fc, Fa)),
                directed=mag / max(tot, 1e-30), expand=Fx / max(tot, 1e-30),
                power=power, pair_share=pshare,
                fight=float(np.abs(cmap_).sum() / max(abs(cmap_.sum()), 1e-30)),
                src_dist=float((w * dsrc).sum()), n_src=len(regs),
                main_src=float(abs(regs[0][1]) / max(np.abs(cmap_).sum(), 1e-30)) if regs else np.nan,
                c=cmap_.astype(np.float32),
                regions=[(m, v) for m, v in regs[:4]]))
        res.append(dict(hz=hz, rows=rows))
        R = rows
        g = lambda k: np.array([r[k] for r in R], float)
        hi, lo, far = gr >= np.percentile(gr, 80), gr <= np.percentile(gr, 20), dist > 30
        print(f"\n  {hz:.4f} Hz (period {1/hz:.0f} s)")
        for nm, m in (("all", np.ones(len(R), bool)), ("top 20% of gradient", hi),
                      ("bottom 20%", lo), ("> 30 mm from input", far)):
            print(f"    {nm:20s} up-gradient {np.median(g('up')[m]):+.2f}  directed "
                  f"{np.median(g('directed')[m]):.2f}  interference share "
                  f"{np.median(g('pair_share')[m]):+.2f}  sources fight x{np.median(g('fight')[m]):.1f}"
                  f"  source regions {np.median(g('n_src')[m]):.0f}, main carries "
                  f"{np.median(g('main_src')[m]):.2f}, sources {np.median(g('src_dist')[m]):.0f} mm away")
        print(f"    across patches: r(gradient, up-gradient flow) {np.corrcoef(gr, g('up'))[0,1]:+.2f}; "
              f"r(distance, interference share) {np.corrcoef(dist, g('pair_share'))[0,1]:+.2f}; "
              f"r(distance, source regions) {np.corrcoef(dist, g('n_src'))[0,1]:+.2f}")
    # named remote sites at the band's centre frequency
    mid = res[min(range(len(res)), key=lambda n: abs(res[n]["hz"] - 0.0848))]
    order = np.argsort(-gr)[:a.top]
    print(f"\n  the top of the gradient at {mid['hz']:.4f} Hz: where each patch's waves come from")
    for n in order:
        P, r = patches[n], mid["rows"][n]
        srcs = "; ".join(f"{len(m)} pieces centred {np.mean(D[m, P['vertex']]):.0f} mm away "
                         f"({100 * v / max(np.abs(r['c']).sum(), 1e-30):+.0f}%)"
                         for m, v in r["regions"][:3])
        print(f"    {P['name']:>7s} grad {P['grad']:+.2f}, {P['dist']:4.0f} mm from input: "
              f"up-gradient {r['up']:+.2f}, interference {r['pair_share']:+.2f}; {srcs}")
    path = os.path.join(RESULTS, f"interplay_remote_{run['name']}.npz")
    np.savez(path, res=np.array(res, object),
             patches=np.array([{k: P[k] for k in ("vertex", "grad", "dist", "name", "e1",
                                                  "e2", "centre")} for P in patches], object),
             tags=np.array(run["tags"], object))
    print(f"\n  wrote {path}")
    _plot_remote(path.replace(".npz", ""), res, patches, c, S, run, g1, cen, D)


# ============================================================ I: single source or cancellation

CATS = ("single source", "reinforcing", "cancelling", "other")


def flow_alone(Mn, Sf, mask):
    """The patch's net flow (along the net direction) if only the inputs in `mask` were
    driven, with the fitted drive's own timing and loudness among them: the cross-spectrum
    restricted to mask x mask. Exact: the medium is linear, and switching inputs off
    removes their rows and columns of S."""
    m = np.flatnonzero(mask)
    return float(np.real(np.sum(Mn[np.ix_(m, m)] * Sf[np.ix_(m, m)].T)))


def classify_flow(parts, total, interf):
    """How a patch's net flow is made from its input regions (each alone) and their
    interference, as shares of the net. Provisional thresholds, round numbers:

      cancelling     the parts alone would push at least twice the net: it is a remainder
      reinforcing    interference between regions adds at least 40% of the net
      single source  one region alone gives 60-150% of the net, the rest little
      other          none of these"""
    kappa = sum(abs(p) for p in parts) / max(abs(total), 1e-30)
    top = max(parts, key=abs) / total if total else np.nan
    it = interf / total if total else np.nan
    if kappa >= 2.0:
        return "cancelling", kappa, top
    if it >= 0.4:
        return "reinforcing", kappa, top
    if 0.6 <= top <= 1.5 and abs(it) < 0.4:
        return "single source", kappa, top
    return "other", kappa, top


def control_drive(run, mode, seed=0):
    """The fitted cross-spectrum, or a control with something taken out of it.

      fitted        as solved
      independent   each input keeps its fitted power, all coherence between inputs
                    removed (S -> diag S): interference averages to zero by construction,
                    so this shows what the inputs' coherence is responsible for
      random        at every frequency the fitted eigenvalues - the same total power and
                    the same spread of it over independent patterns - with eigenvectors
                    drawn at random (Haar unitary): coherent like the fit, but with timing
                    patterns that were never fitted to anything"""
    S = np.asarray(run["S"])
    if mode == "fitted":
        return run
    if mode == "independent":
        d = np.real(np.einsum("fkk->fk", S))
        return dict(run, S=np.einsum("fk,kl->fkl", d, np.eye(S.shape[1])).astype(complex),
                    name=f"{run['name']}_independent")
    if mode == "random":
        rng = np.random.default_rng(seed)
        K = S.shape[1]
        out = np.empty_like(S)
        for f in range(len(S)):
            ev = np.clip(np.linalg.eigvalsh(0.5 * (S[f] + S[f].conj().T)), 0, None)
            Z = (rng.standard_normal((K, K)) + 1j * rng.standard_normal((K, K))) / np.sqrt(2)
            Q, R = np.linalg.qr(Z)
            Q = Q * (np.diag(R) / np.abs(np.diag(R)))[None, :]
            out[f] = (Q * ev) @ Q.conj().T
        return dict(run, S=out, name=f"{run['name']}_random{seed}")
    raise ValueError(mode)


def combine_at(run, patches, Hf, Uf, depth, area, nrm, S, D, adj, b, nregions=3):
    """One frequency: for every patch, its net flow under the fitted drive set against
    each main input region alone, everything else alone, and their interference."""
    K = len(run["tags"])
    Sf = 0.5 * (run["S"][b] + run["S"][b].conj().T)
    rows = []
    for P in patches:
        M = steering_matrices(P, Hf, Uf, depth, area, nrm, S.V)
        Fa, Fc = (float(np.real(np.trace(M[k] @ Sf))) for k in ("along", "across"))
        mag = float(np.hypot(Fa, Fc))
        n_hat = np.array([Fa, Fc]) / max(mag, 1e-30)
        Mn = n_hat[0] * M["along"] + n_hat[1] * M["across"]
        ck, _, _ = source_map(M, Sf, n_hat)
        regs = source_regions(ck, adj)[:nregions]
        used = np.zeros(K, bool)
        parts = []
        for mem, _ in regs:
            msk = np.zeros(K, bool); msk[mem] = True; used |= msk
            parts.append(flow_alone(Mn, Sf, msk))
        rest = flow_alone(Mn, Sf, ~used)
        total = flow_alone(Mn, Sf, np.ones(K, bool))
        interf = total - sum(parts) - rest
        cat, kappa, top = classify_flow(parts + [rest], total, interf)
        power = float(np.real(np.trace(M["G"] @ Sf)))
        rows.append(dict(cat=cat, kappa=kappa, top=top, interf=interf / total,
                         parts=[p / total for p in parts], rest=rest / total,
                         n_regions=len(regs), up=Fa / max(mag, 1e-30), power=power,
                         src_dist=[float(np.mean(D[m, P["vertex"]])) for m, _ in regs]))
    return rows


def cmd_combine(a, c, S, run):
    run = control_drive(run, a.drive, a.seed)
    K = len(run["tags"])
    D, cen, adj = geometry(c, S, run["labels"], K)
    g1 = np.load(os.path.join(os.path.dirname(RESULTS), "data", "cache",
                              f"cortical_maps_{c.mesh}.npz"))["grad1"]
    patches = cortex_patches(c, S, a.grid, a.radius, g1)
    lab = np.asarray(run["labels"])
    dterr = S.distance_from(np.flatnonzero(lab >= 0))
    names = [str(v).removeprefix("L_").removesuffix("_ROI") for v in c.names]
    for P in patches:
        P["grad"] = float(g1[P["pv"]].mean())
        P["dist"] = float(dterr[P["vertex"]])
        P["name"] = names[int(c.lab[P["vertex"]])]
    gr = np.array([P["grad"] for P in patches])
    hzb = run["idx"] / run["ref_frames"] / run["frame_s"]
    grid = np.unique([int(np.argmin(np.abs(np.log(hzb / f)))) for f in np.geomspace(0.005, 0.5, 14)])
    H, U, depth = waves.transfer_fields(c, run, hzb[grid], velocity=True)
    want = [float(v) for v in a.hz.split(",")] if a.hz else [0.0212, 0.0409, 0.0848, 0.2513, 0.4936]
    sel = [int(np.argmin(np.abs(np.log(hzb[grid] / f)))) for f in want]
    area, nrm = np.asarray(c.A), np.asarray(c.m.vertex_normal)
    print(f"\n  {len(patches)} patches over the whole cortex, under the fitted drive of "
          f"{run['name']}")
    res = []
    for fi in sel:
        hz = float(hzb[grid][fi])
        rows = combine_at(run, patches, H[fi], U[fi], depth, area, nrm, S, D, adj,
                          grid[fi], a.nregions)
        res.append(dict(hz=hz, rows=rows))
        cats = np.array([r["cat"] for r in rows])
        kap = np.array([r["kappa"] for r in rows])
        print(f"\n  {hz:.4f} Hz (period {1/hz:.0f} s): "
              + ", ".join(f"{k} {np.mean(cats == k):.0%}" for k in CATS)
              + f"; parts overshoot the net x{np.median(kap):.1f} (median)")
        for nm, m in (("bottom 20% of gradient", gr <= np.percentile(gr, 20)),
                      ("top 20% of gradient", gr >= np.percentile(gr, 80)),
                      ("> 30 mm from input", np.array([P["dist"] > 30 for P in patches]))):
            print(f"    {nm:24s}: " + ", ".join(f"{k} {np.mean(cats[m] == k):.0%}" for k in CATS)
                  + f"; overshoot x{np.median(kap[m]):.1f}")
        print(f"    r(gradient, log overshoot) {np.corrcoef(gr, np.log(kap))[0, 1]:+.2f}")
    path = os.path.join(RESULTS, f"interplay_combine_{run['name']}.npz")
    np.savez(path, res=np.array(res, object), grad=gr,
             dist=np.array([P["dist"] for P in patches]),
             vertex=np.array([P["vertex"] for P in patches]),
             names=np.array([P["name"] for P in patches], object))
    print(f"\n  wrote {path}")
    _plot_combine(path.replace(".npz", ""), res, patches, c, run, g1)


# ============================================================ J: weighted by what BOLD sees

def hrf_power(hz):
    """|HRF(f)|^2 of the SPM canonical double gamma (peak 6 s, undershoot 16 s, 1/6),
    normalised to 1 at 0 Hz."""
    from scipy.stats import gamma
    t = np.arange(0, 40, 0.01)
    h = gamma.pdf(t, 6) - gamma.pdf(t, 16) / 6.0
    Hf = np.array([np.sum(h * np.exp(-2j * np.pi * f * t)) for f in np.atleast_1d(hz)])
    H0 = np.sum(h)
    return np.abs(Hf / H0) ** 2


def observables(hz, frame_s, smooth_frames):
    """Gain^2 per frequency of four readings of what is observed. Temporal filters act on
    each frequency by a gain alone, so they reweight frequencies and change no wave's
    geometry at any one of them."""
    sd_t = smooth_frames / 2.3548 * frame_s
    import bandpass
    model = np.exp(-(2 * np.pi * hz * sd_t) ** 2)          # the gaussian kernel, squared
    hrf = hrf_power(hz)
    return {"raw field": np.ones_like(hz),
            "this model's observable": model,
            "haemodynamic response": hrf,
            "HRF + 0.01-0.08 Hz": hrf * bandpass.response(hz, frame_s, 0.01, 0.08)}


def cmd_bold(a, c, S, run):
    run = control_drive(run, a.drive, a.seed)
    import json
    K = len(run["tags"])
    D, cen, adj = geometry(c, S, run["labels"], K)
    g1 = np.load(os.path.join(os.path.dirname(RESULTS), "data", "cache",
                              f"cortical_maps_{c.mesh}.npz"))["grad1"]
    patches = cortex_patches(c, S, a.grid, a.radius, g1)
    gr = np.array([float(g1[P["pv"]].mean()) for P in patches])
    hzb = run["idx"] / run["ref_frames"] / run["frame_s"]
    grid = np.unique([int(np.argmin(np.abs(np.log(hzb / f)))) for f in np.geomspace(0.005, 0.5, 14)])
    hz = hzb[grid]
    cf = os.path.join(RESULTS, f"interplay_combine_all_{run['name']}.npz")
    if os.path.exists(cf):
        allrows = list(np.load(cf, allow_pickle=True)["rows"])
    else:
        H, U, depth = waves.transfer_fields(c, run, hz, velocity=True)
        area, nrm = np.asarray(c.A), np.asarray(c.m.vertex_normal)
        allrows = []
        for fi in range(len(grid)):
            allrows.append(combine_at(run, patches, H[fi], U[fi], depth, area, nrm, S, D,
                                      adj, grid[fi], a.nregions))
            print(f"    {hz[fi]:.4f} Hz done", flush=True)
        np.savez(cf, rows=np.array(allrows, object), hz=hz)
    z = np.load(os.path.join(RESULTS, f"xspec_{a.tag}.npz"), allow_pickle=True)
    smooth = float(json.loads(str(z["args_json"])).get("smooth", 8.0))
    obs = observables(hz, run["frame_s"], smooth)
    width = np.gradient(hz)                                  # each bin stands for a band
    power = np.array([[r["power"] for r in rows] for rows in allrows])     # (nf, np)
    cats = np.array([[r["cat"] for r in rows] for rows in allrows])
    interf = np.array([[r["interf"] for r in rows] for rows in allrows])
    out = {}
    print(f"\n  {len(patches)} patches, {len(hz)} frequencies {hz[0]:.4f}-{hz[-1]:.3f} Hz, "
          f"under the fitted drive of {run['name']}")
    for nm, g in obs.items():
        w = g[:, None] * power * width[:, None]
        w = w / w.sum(0, keepdims=True)                      # per patch, over frequency
        share = {k: (w * (cats == k)).sum(0) for k in CATS}  # per patch
        f_med = np.array([np.interp(0.5, np.cumsum(w[:, p]), hz) for p in range(w.shape[1])])
        out[nm] = dict(w=w, share=share, f_med=f_med,
                       interf=(w * np.clip(interf, -10, 10)).sum(0))
        lo, hi = gr <= np.percentile(gr, 20), gr >= np.percentile(gr, 80)
        print(f"\n  {nm}: half of what is seen lies below {np.median(f_med):.3f} Hz (median "
              f"patch)")
        for lab_, m in (("all", np.ones(len(gr), bool)), ("sensorimotor fifth", lo),
                        ("transmodal fifth", hi)):
            print(f"    {lab_:19s}: " + ", ".join(f"{k} {np.mean(share[k][m]):.0%}"
                                                  for k in CATS)
                  + f"; interference adds {np.median(out[nm]['interf'][m]):+.0%} of the net "
                    f"(median)")
    path = os.path.join(RESULTS, f"interplay_bold_{run['name']}.npz")
    np.savez(path, hz=hz, grad=gr, out=np.array([out], object),
             obs=np.array([obs], object), vertex=np.array([P["vertex"] for P in patches]))
    print(f"\n  wrote {path}")
    _plot_bold(path.replace(".npz", ""), hz, obs, out, power, width, gr, patches, c, run, g1)


def cmd_media(a):
    """The whole-cortex categories of `bold` side by side across media, against frequency
    and against frequency x the medium's decay time (cycles per decay). Reads only the
    cached per-bin categories, so each medium needs `bold` run first, with its controls
    for the grey band."""
    import json
    tags = [t for t in a.media.split(",") if t.strip()]
    med = []
    for t in tags:
        cf = os.path.join(RESULTS, f"interplay_combine_all_{t}.npz")
        if not os.path.exists(cf):
            raise SystemExit(f"  no {cf}: run `bold --tag {t}` first")
        z = np.load(cf, allow_pickle=True)
        args = json.loads(str(np.load(os.path.join(RESULTS, f"xspec_{t}.npz"),
                                      allow_pickle=True)["args_json"]))

        def frac(rows):
            cats = np.array([[r["cat"] for r in rr] for rr in rows])
            return {k: (cats == k).mean(1) for k in CATS}
        ctl = [frac(np.load(f, allow_pickle=True)["rows"]) for f in
               (os.path.join(RESULTS, f"interplay_combine_all_{t}_random{s}.npz")
                for s in range(3)) if os.path.exists(f)]
        med.append(dict(tag=t, hz=z["hz"], tau=float(args["decay_s"]),
                        speed=float(args["spread_mm_s"]), frac=frac(z["rows"]), ctl=ctl))
    print(f"\n  share of 150 patches by category, fitted drive (random coherent drive in "
          f"brackets, mean of {len(med[0]['ctl'])} seeds)")
    for m in med:
        print(f"\n  {m['tag']}: {m['speed']:g} mm/s, decay {m['tau']:.3g} s")
        for i, f in enumerate(m["hz"]):
            print(f"    {f:.4f} Hz = {f * m['tau']:5.2f} cyc/decay: " + ", ".join(
                f"{k[:6]} {m['frac'][k][i]:4.0%}"
                + (f" ({np.mean([c[k][i] for c in m['ctl']]):3.0%})" if m["ctl"] else "")
                for k in CATS[:3]))
        d = m["frac"]["reinforcing"] - m["frac"]["cancelling"]
        x = np.flatnonzero((d[:-1] > 0) & (d[1:] <= 0))
        if len(x):
            i = x[0]
            lf = np.log(m["hz"][i]) + d[i] / (d[i] - d[i + 1]) * np.log(m["hz"][i + 1]
                                                                    / m["hz"][i])
            m["cross"] = float(np.exp(lf))
            print(f"    reinforcing gives way to cancelling at {m['cross']:.3f} Hz = "
                  f"{m['cross'] * m['tau']:.2f} cycles per decay")
    _plot_media(os.path.join(RESULTS, "interplay_media"), med)


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


def _plot_frames(path, res, run, fs, edge):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pr = [r for r in res if r["kind"] == "pair"]
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.1))
    a = np.array([r["r_pairterm"] for r in pr]); b = np.array([r["r_total"] for r in pr])
    axs[0].hist(a[np.isfinite(a)], 30, alpha=0.6, label="their pair term")
    axs[0].hist(b[np.isfinite(b)], 30, alpha=0.6, label="patch total")
    axs[0].set_title("r(sin lag, flux along the pair axis)", fontsize=9)
    axs[0].legend(fontsize=7)
    for kind, mk in (("pair", "o"), ("grid", "s")):
        R = [r for r in res if r["kind"] == kind]
        axs[1].scatter([r["pair_share_abs"] for r in R], [r["wander"] for r in R], s=8,
                       marker=mk, label=kind, alpha=0.7)
    axs[1].set_xlabel("pair terms' share of |flux|", fontsize=8)
    axs[1].set_ylabel("direction wander", fontsize=8); axs[1].legend(fontsize=7)
    # the most timing-driven pair patch, as a time series
    best = max(pr, key=lambda r: (r["r_total"] if np.isfinite(r["r_total"]) else -9))
    t = (np.arange(len(best["theta"])) + edge) * fs
    seg = slice(0, min(len(t), int(240 / fs)))
    axs[2].plot(t[seg], np.degrees(best["theta"][seg]), lw=0.8, label="flux direction")
    axs[2].plot(t[seg], np.degrees(best["lag"][seg]), lw=0.8, label="lag of second")
    axs[2].set_title(f"{run['tags'][best['i']]} -> {run['tags'][best['j']]} "
                     f"(r total {best['r_total']:+.2f})", fontsize=9)
    axs[2].set_xlabel("s", fontsize=8); axs[2].legend(fontsize=7)
    axs[3].scatter(np.degrees(best["lag"]), np.cos(best["theta"]), s=2, alpha=0.3)
    axs[3].set_xlabel("lag of second piece (deg)", fontsize=8)
    axs[3].set_ylabel("cos(flux direction vs pair axis)", fontsize=8)
    for ax in axs:
        ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor="white")
    plt.close(fig)
    print(f"  wrote {path}")


def _plot_types(path, res, run):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(16, 3.4))
    ax = fig.add_subplot(1, 4, 1)
    for n, kind in enumerate(("grid", "pair")):
        R = [r for r in res if r["kind"] == kind]
        bottom = 0
        for t, col in zip(TYPES, plt.cm.tab10.colors):
            v = np.mean([r["occ"][t] for r in R])
            ax.bar(n, v, bottom=bottom, color=col, label=t if n == 0 else None)
            bottom += v
    ax.set_xticks([0, 1]); ax.set_xticklabels(["grid", "pair"])
    ax.set_title("pattern type occupancy", fontsize=9); ax.legend(fontsize=6)
    ax = fig.add_subplot(1, 4, 2)
    for kind, mk in (("grid", "s"), ("pair", "o")):
        R = [r for r in res if r["kind"] == kind]
        ax.scatter([r["acc_a"] for r in R], [r["acc_t"] for r in R], s=8, marker=mk,
                   label=kind, alpha=0.7)
    lim = [0, 1]; ax.plot(lim, lim, color="0.6", lw=0.6)
    ax.set_xlabel("balanced accuracy from loudness", fontsize=8)
    ax.set_ylabel("from timing", fontsize=8); ax.legend(fontsize=7)
    ax.set_title("which inputs' property predicts the type", fontsize=9)
    ax = fig.add_subplot(1, 4, 3)
    nm = ("along", "across", "expand", "power")
    for n, kind in enumerate(("grid", "pair")):
        R = [r for r in res if r["kind"] == kind]
        x = np.arange(len(nm)) + 0.2 * n
        ax.bar(x, [np.median([r["r2t_" + k] for r in R]) for k in nm], 0.2,
               label=f"{kind} timing")
        ax.bar(x + 0.4, [np.median([r["r2a_" + k] for r in R]) for k in nm], 0.2,
               label=f"{kind} loudness", alpha=0.5)
    ax.set_xticks(np.arange(len(nm)) + 0.3); ax.set_xticklabels(nm, fontsize=8)
    ax.set_title("median cross-validated R2", fontsize=9); ax.legend(fontsize=6)
    # the patch whose type timing predicts best: its pieces' phases in two opposite states
    best = max(res, key=lambda r: (r["acc_t"] if np.isfinite(r["acc_t"]) else -1))
    ax = fig.add_subplot(1, 4, 4, projection="polar")
    for t, col in zip(TYPES, plt.cm.tab10.colors):
        if t in best["trig"]:
            mu, R = best["trig"][t]
            ax.scatter(mu, R, color=col, s=18, label=t)
            for k, (m_, r_) in enumerate(zip(mu, R)):
                ax.annotate(str(k + 1), (m_, r_), fontsize=5, color=col)
    ax.set_title(f"best patch ({best['kind']}): pieces 2.. phase vs piece 1, by type",
                 fontsize=8)
    ax.legend(fontsize=5, loc="lower left", bbox_to_anchor=(1.0, 0.0))
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor="white")
    plt.close(fig)
    print(f"  wrote {path}")


def _plot_steer(path, res, pred, c, run, cen, patches):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.tri import Triangulation
    from render_regimes import _proj
    rows = res[len(res) // 2]["rows"]
    hz = res[len(res) // 2]["hz"]
    # the pair patch whose along-axis flux the inputs can swing hardest, both ways
    best = max(range(len(rows)), key=lambda n: (rows[n]["kind"] == "pair") *
               min(rows[n]["along_range"][1], -rows[n]["along_range"][0]))
    P, r = patches[best], rows[best]
    lab = np.asarray(run["labels"])
    cv = int(np.argmin(np.linalg.norm(c.V - P["centre"], axis=1)))
    views = _proj(c.V, c.F)
    xy, vis, _ = next((v for v in views if v[1][cv]), views[0])     # one that shows it
    keep = vis[c.F].all(1)
    tri = Triangulation(xy[:, 0], xy[:, 1], c.F[keep])
    ctr = xy[np.argmin(np.linalg.norm(c.V - P["centre"], axis=1))]
    fig = plt.figure(figsize=(16, 7))
    for n, key in enumerate(("along+", "along-", "expand+", "expand-")):
        ax = fig.add_subplot(2, 4, n + 1)
        ev, X = np.linalg.eigh(r["M"][key[:-1]].astype(complex))
        x = X[:, -1] if key.endswith("+") else X[:, 0]
        x = x * np.exp(-1j * np.angle(x[np.argmax(np.abs(x))]))
        ph = np.where(lab >= 0, np.angle(x[np.clip(lab, 0, None)]), np.nan)
        am = np.where(lab >= 0, np.abs(x[np.clip(lab, 0, None)]), 0.0)
        am = am / max(am.max(), 1e-30)
        ax.tripcolor(tri, np.zeros(c.nV), cmap="Greys", vmin=0, vmax=4, shading="gouraud")
        col = plt.cm.twilight((np.nan_to_num(ph) + np.pi) / (2 * np.pi))
        col[:, 3] = am
        ax.scatter(xy[vis, 0], xy[vis, 1], c=col[vis], s=7, linewidths=0)
        e1 = P["e1"]
        p0 = ctr
        p1 = xy[np.argmin(np.linalg.norm(c.V - (P["centre"] + 12 * e1), axis=1))]
        ax.annotate("", p1, p0, arrowprops=dict(arrowstyle="->", lw=1.5))
        ax.set_xlim(ctr[0] - 45, ctr[0] + 45); ax.set_ylim(ctr[1] - 35, ctr[1] + 35)
        ax.set_aspect("equal"); ax.axis("off")
        nm = {"along+": "pushes waves along the arrow", "along-": "pushes them back",
              "expand+": "makes the patch radiate", "expand-": "makes it converge"}[key]
        ax.set_title(f"{nm}\nsweep {r[key]['sweep']:+.2f}, {r[key]['pr']:.1f} pieces",
                     fontsize=8)
    ax = fig.add_subplot(2, 4, 5)
    for nm, colr in zip(READOUTS, ("C0", "C1", "C2")):
        v = np.array([q[nm + "+"]["sweep"] for q in rows])
        ax.hist(v[np.isfinite(v)], 30, alpha=0.5, color=colr, label=f"{nm} +")
    ax.set_title(f"sweep of the + modes ({hz:.3f} Hz)", fontsize=9); ax.legend(fontsize=7)
    ax = fig.add_subplot(2, 4, 6)
    for nm, colr in zip(READOUTS, ("C0", "C1", "C2")):
        ax.scatter([q[nm + "+"]["use"] for q in rows], [q[nm + "-"]["use"] for q in rows],
                   s=5, color=colr, label=nm, alpha=0.6)
    ax.plot([0.1, 10], [0.1, 10], color="0.6", lw=0.6)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("fitted drive's power in + mode / isotropic", fontsize=8)
    ax.set_ylabel("in - mode / isotropic", fontsize=8); ax.legend(fontsize=7)
    ax = fig.add_subplot(2, 4, 7)
    ax.hist([q["along+"]["pr"] for q in rows], 30, alpha=0.6, label="pieces in mode")
    ax.hist([q["along+"]["reach"] for q in rows], 30, alpha=0.6, label="mm from patch")
    ax.legend(fontsize=7); ax.set_title("size of the along+ modes", fontsize=9)
    if pred:
        ax = fig.add_subplot(2, 4, 8)
        for kind in ("pair", "grid"):
            ax.hist([p["agree"] for p in pred if p["kind"] == kind], 30, alpha=0.6,
                    label=kind)
        ax.set_title("direction from all inputs, no fitting: mean cos(error)", fontsize=9)
        ax.legend(fontsize=7)
    fig.suptitle(f"steering modes at {hz:.3f} Hz: colour = input phase, opacity = amplitude; "
                 f"example patch {run['tags'][P['i']]}-{run['tags'][P['j']]}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor="white")
    plt.close(fig)
    print(f"  wrote {path}")


def _plot_steermap(path, c, run, patches, shares, names, groups, mats, hz):
    """Small multiples, one row per pathway: where on cortex its inputs carry the
    steering. One sequential hue, so no two of eight categorical colours ever have to be
    told apart on a map - which the validated palette cannot guarantee past three."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.tri import Triangulation
    from render_regimes import _proj
    cmap = LinearSegmentedColormap.from_list("blues", BLUES)
    ink, muted, surf, terr = "#1a1a19", "#6b6a64", "#f4f3ef", "#d8d6cf"
    views = _proj(c.V, c.F, views=((8, 180, "lateral"), (8, 0, "medial"),
                                   (-90, 180, "ventral")))
    lab = np.asarray(run["labels"])
    cv = [int(np.argmin(np.linalg.norm(c.V - P["centre"], axis=1))) for P in patches]
    cols = list(STEER)
    fig = plt.figure(figsize=(3.1 * len(views) * len(cols) + 1, 1.9 * len(names) + 1.2),
                     facecolor="white")
    gs = fig.add_gridspec(len(names), len(views) * len(cols), wspace=0.02, hspace=0.08)
    for gi, g in enumerate(names):
        own = np.isin(lab, [k for k, gg in enumerate(groups) if gg == g])
        for ci, title in enumerate(cols):
            v = np.array([np.mean([sh[m][gi] for m in STEER[title]]) for sh in shares])
            for vi, (xy, vis, vname) in enumerate(views):
                ax = fig.add_subplot(gs[gi, ci * len(views) + vi])
                keep = vis[c.F].all(1)
                tri = Triangulation(xy[:, 0], xy[:, 1], c.F[keep])
                ax.tripcolor(tri, own.astype(float), cmap=LinearSegmentedColormap.from_list(
                    "t", (surf, terr)), vmin=0, vmax=1, shading="flat", rasterized=True)
                m = np.array([vis[q] for q in cv])
                o = np.argsort(v[m])
                ax.scatter(xy[np.array(cv)[m][o], 0], xy[np.array(cv)[m][o], 1],
                           c=v[m][o], cmap=cmap, vmin=0, vmax=1, s=14,
                           edgecolors="white", linewidths=0.6)
                ax.set_xlim(xy[:, 0].min(), xy[:, 0].max())
                ax.set_ylim(xy[:, 1].min(), xy[:, 1].max())
                ax.set_aspect("equal"); ax.axis("off")
                if gi == 0:
                    ax.set_title(f"{title}\n{vname}" if vi == 1 else vname, fontsize=8,
                                 color=ink)
                if ci == 0 and vi == 0:
                    ax.text(-0.04, 0.5, g, transform=ax.transAxes, ha="right",
                            va="center", fontsize=8, color=ink)
    cax = fig.add_axes([0.35, 0.03, 0.3, 0.008])
    sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=matplotlib.colors.Normalize(0, 1))
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label("share of the patch's steering-mode power carried by this pathway "
                 "(grey: its own territory)", fontsize=8, color=muted)
    cb.ax.tick_params(labelsize=7, colors=muted)
    fig.suptitle(f"who steers where, {hz:.3f} Hz", fontsize=10, color=ink)
    fig.savefig(path, dpi=130, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")
    # the matrix, annotated: rows where the patch is, columns whose inputs steer it
    fig, axs = plt.subplots(1, len(mats), figsize=(5.6 * len(mats), 5.0), facecolor="white")
    for ax, (title, Mx) in zip(np.atleast_1d(axs), mats.items()):
        ax.imshow(Mx, cmap=cmap, vmin=0, vmax=max(0.5, Mx.max()))
        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(j, i, f"{Mx[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if Mx[i, j] > 0.3 else ink)
        ax.set_xticks(range(len(names))); ax.set_yticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7, color=ink)
        ax.set_yticklabels(names, fontsize=7, color=ink)
        ax.set_xlabel("whose inputs carry the steering", fontsize=8, color=muted)
        ax.set_ylabel("where the patch is (nearest pathway)", fontsize=8, color=muted)
        ax.set_title(title, fontsize=9, color=ink)
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.tight_layout()
    p2 = path.replace(".png", "_matrix.png")
    fig.savefig(p2, dpi=130, facecolor="white")
    plt.close(fig)
    print(f"  wrote {p2}")


def _plot_sweeps(path, res, pred, c, S, patches, hz, c_med):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.tri import Triangulation
    from render_regimes import _proj
    ink, muted = "#1a1a19", "#6b6a64"
    blue, orange, aqua = "#2a78d6", "#eb6834", "#1baf7a"
    fig = plt.figure(figsize=(17, 7.6), facecolor="white")
    gs = fig.add_gridspec(2, 4, hspace=0.35, wspace=0.28)
    # one patch's along+ mode, as the smoothed input field it is: phase as hue, amplitude
    # as opacity, the sweep direction as arrows, the patch axis as the black arrow
    best = max(range(len(res)), key=lambda n: (res[n]["kind"] == "grid") *
               res[n]["along+"]["a_e1"] * (res[n]["along+"]["planar"] > 0.3))
    r, P = res[best], patches[best]
    qs = r["q_along+"].astype(complex)
    cv = int(np.argmin(np.linalg.norm(c.V - P["centre"], axis=1)))
    views = _proj(c.V, c.F)
    xy, vis, vname = next((v for v in views if v[1][cv]), views[0])
    ax = fig.add_subplot(gs[:, 0:2])
    keep = vis[c.F].all(1)
    tri = Triangulation(xy[:, 0], xy[:, 1], c.F[keep])
    amp = np.clip(np.abs(qs) / np.percentile(np.abs(qs)[np.abs(qs) > 0], 95), 0, 1)
    ph = np.angle(qs * np.exp(-1j * np.angle(qs[np.argmax(np.abs(qs))])))
    ax.tripcolor(tri, np.zeros(c.nV), cmap="Greys", vmin=0, vmax=8, shading="gouraud")
    col = plt.cm.twilight_shifted((ph + np.pi) / (2 * np.pi))
    col[:, 3] = 0.08 + 0.92 * amp
    ax.scatter(xy[vis, 0], xy[vis, 1], c=col[vis], s=16, linewidths=0)
    g = S.phase_grad(qs[None])[0]
    kh = -g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-12)
    fa = np.abs(qs[S.F]).mean(1) / np.percentile(np.abs(qs)[np.abs(qs) > 0], 95)
    fc = S.V[S.F].mean(1)
    sel = np.flatnonzero((fa > 0.25) & vis[S.F].all(1))[::6]
    # project face centres and directions into the view
    def to2(X):
        return np.array([[np.argmin(np.linalg.norm(c.V - x, axis=1))] for x in X]).ravel()
    vi = to2(fc[sel])
    tip = to2(fc[sel] + 6 * kh[sel])
    ax.quiver(xy[vi, 0], xy[vi, 1], xy[tip, 0] - xy[vi, 0], xy[tip, 1] - xy[vi, 1],
              angles="xy", scale_units="xy", scale=1, width=0.0025, color=ink, alpha=0.75)
    p1 = xy[np.argmin(np.linalg.norm(c.V - (P["centre"] + 15 * P["e1"]), axis=1))]
    ax.annotate("", p1, xy[cv], arrowprops=dict(arrowstyle="-|>", lw=2.2, color="black"))
    ax.scatter(*xy[cv], s=60, facecolors="none", edgecolors="black", linewidths=1.5)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(f"the input timing that pushes waves along the black arrow ({hz:.3f} Hz, "
                 f"{vname})\nhue = when each place peaks in the cycle, opacity = how hard it "
                 f"is driven, arrows = where the peak moves next", fontsize=9, color=ink)
    def hist(axh, key, field, title, sign=1):
        for kind, colr in (("grid", blue), ("pair", orange)):
            v = sign * np.array([q[key][field] for q in res if q["kind"] == kind])
            axh.hist(v, 30, color=colr, alpha=0.65, label=kind)
        axh.set_title(title, fontsize=9, color=ink)
        axh.tick_params(labelsize=7, colors=muted)
        for sp in ("top", "right"):
            axh.spines[sp].set_visible(False)
    ax = fig.add_subplot(gs[0, 2])
    for nm, fld, colr in (("along+", "a_e1", blue), ("across+", "a_e2", orange),
                          ("expand+", "a_out", aqua)):
        ax.hist([q[nm][fld] for q in res], 30, color=colr, alpha=0.6, label=nm)
    ax.axvline(0, color=muted, lw=0.6)
    ax.set_title("does the input sweep point the way the waves are pushed?", fontsize=9,
                 color=ink)
    ax.set_xlabel("alignment of input sweep with the pushed direction", fontsize=8,
                  color=muted)
    ax.legend(fontsize=7, frameon=False); ax.tick_params(labelsize=7, colors=muted)
    ax = fig.add_subplot(gs[0, 3])
    hist(ax, "along+", "speed", "apparent speed of the steering sweep (mm/s)")
    ax.axvline(c_med, color=ink, lw=1.2, ls="--")
    ax.text(c_med, ax.get_ylim()[1] * 0.92, " medium's own\n wave speed", fontsize=7,
            color=ink, va="top")
    ax.set_xscale("log")
    from matplotlib.ticker import FixedLocator, NullLocator, FixedFormatter
    tk = [20, 30, 50, 100, 200]
    ax.xaxis.set_major_locator(FixedLocator(tk))
    ax.xaxis.set_major_formatter(FixedFormatter([str(t) for t in tk]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.legend(fontsize=7, frameon=False)
    ax = fig.add_subplot(gs[1, 2])
    for kind, colr in (("grid", blue), ("pair", orange)):
        R = [q for q in res if q["kind"] == kind]
        ax.scatter([q["along+"]["planar"] for q in R], [-q["along+"]["a_out"] for q in R],
                   s=10, color=colr, alpha=0.7, label=kind, edgecolors="white",
                   linewidths=0.4)
    ax.axhline(0, color=muted, lw=0.6)
    ax.set_xlabel("planarity of the along+ sweep", fontsize=8, color=muted)
    ax.set_ylabel("aimed at the patch (+) or away (-)", fontsize=8, color=muted)
    ax.set_title("straight sweep, or focused on the patch?", fontsize=9, color=ink)
    ax.legend(fontsize=7, frameon=False); ax.tick_params(labelsize=7, colors=muted)
    if pred:
        ax = fig.add_subplot(gs[1, 3])
        for nm, colr in zip(READOUTS, (blue, orange, aqua)):
            ax.hist([p[nm] for p in pred if np.isfinite(p[nm])], 30, color=colr, alpha=0.6,
                    label=nm)
        ax.axvline(0, color=muted, lw=0.6)
        ax.set_title("fitted drive's sweep vs the patch's actual waves (r)", fontsize=9,
                     color=ink)
        ax.legend(fontsize=7, frameon=False); ax.tick_params(labelsize=7, colors=muted)
    fig.savefig(path, dpi=130, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def _plot_remote(stem, res, patches, c, S, run, g1, cen, D):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    from matplotlib.tri import Triangulation
    from render_regimes import _proj
    ink, muted = "#1a1a19", "#6b6a64"
    greys = LinearSegmentedColormap.from_list("g", ("#f4f3ef", "#a9a79f"))
    # diverging blue <-> red, neutral grey midpoint (dataviz reference pair)
    div = LinearSegmentedColormap.from_list("d", ("#184f95", "#6da7ec", "#f0efec",
                                                  "#ec8a88", "#a12a29"))
    lab = np.asarray(run["labels"])
    views = _proj(c.V, c.F, views=((8, 180, "lateral"), (8, 0, "medial"),
                                   (-90, 180, "ventral")))
    gr = np.array([P["grad"] for P in patches])
    pick = [min(res, key=lambda q: abs(q["hz"] - f)) for f in (0.0212, 0.0848)]

    # 1. the expected energy flow over the whole cortex
    fig = plt.figure(figsize=(16, 9.6), facecolor="white")
    gs = fig.add_gridspec(len(pick), len(views), hspace=0.08, wspace=0.02)
    for ri, q in enumerate(pick):
        for vi, (xy, vis, vname) in enumerate(views):
            ax = fig.add_subplot(gs[ri, vi])
            keep = vis[c.F].all(1)
            tri = Triangulation(xy[:, 0], xy[:, 1], c.F[keep])
            ax.tripcolor(tri, g1, cmap=greys, shading="gouraud", rasterized=True)
            ax.tricontour(tri, (lab >= 0).astype(float), levels=[0.5], colors=ink,
                          linewidths=0.6)
            P0, U0, C0 = [], [], []
            for P, r in zip(patches, q["rows"]):
                v = P["vertex"]
                if not vis[v]:
                    continue
                dvec = np.cos(r["theta"]) * P["e1"] + np.sin(r["theta"]) * P["e2"]
                tip = int(np.argmin(np.linalg.norm(c.V - (P["centre"] + 7 * dvec), axis=1)))
                L = min(1.0, r["directed"] * 2.5)
                P0.append(xy[v]); U0.append((xy[tip] - xy[v]) * L); C0.append(r["pair_share"])
            P0, U0 = np.array(P0), np.array(U0)
            qv = ax.quiver(P0[:, 0], P0[:, 1], U0[:, 0], U0[:, 1], np.clip(C0, -1, 1),
                           cmap=div, norm=TwoSlopeNorm(0, -1, 1), angles="xy",
                           scale_units="xy", scale=1, width=0.004, headwidth=3.5)
            ax.set_aspect("equal"); ax.axis("off")
            if ri == 0:
                ax.set_title(vname, fontsize=9, color=ink)
            if vi == 0:
                ax.text(-0.02, 0.5, f"{q['hz']:.3f} Hz", transform=ax.transAxes, ha="right",
                        va="center", fontsize=9, color=ink)
    cax = fig.add_axes([0.36, 0.05, 0.28, 0.01])
    cb = fig.colorbar(qv, cax=cax, orientation="horizontal")
    cb.set_label("share of the patch's net flow from interference between inputs "
                 "(blue: interference opposes it, red: adds to it)", fontsize=8, color=muted)
    cb.ax.tick_params(labelsize=7, colors=muted)
    fig.suptitle("expected energy flow under the fitted drive; arrow length = how directed; "
                 "background = principal gradient (dark = transmodal); outline = inputs",
                 fontsize=10, color=ink)
    fig.savefig(stem + "_flow.png", dpi=130, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {stem}_flow.png")

    # 2. against the gradient
    fig, axs = plt.subplots(1, 4, figsize=(16, 3.4), facecolor="white")
    cols = ("#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#e34948")
    for q, col in zip(res, cols):
        R = q["rows"]
        for ax, k, t in zip(axs, ("up", "pair_share", "n_src", "src_dist"),
                            ("flow up the gradient (cos)", "interference share of net flow",
                             "separate source regions", "where the sources are (mm)")):
            y = np.array([r[k] for r in R], float)
            o = np.argsort(gr)
            # running median over 25 patches, so trends read through the scatter
            from scipy.ndimage import median_filter
            ax.scatter(gr, y, s=4, color=col, alpha=0.25, linewidths=0)
            ax.plot(gr[o], median_filter(y[o], 25, mode="nearest"), color=col, lw=2,
                    label=f"{q['hz']:.3f} Hz")
            ax.set_title(t, fontsize=9, color=ink)
            ax.set_xlabel("principal gradient (sensorimotor -> transmodal)", fontsize=8,
                          color=muted)
            ax.tick_params(labelsize=7, colors=muted)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
    axs[0].axhline(0, color=muted, lw=0.6); axs[1].axhline(0, color=muted, lw=0.6)
    axs[1].set_ylim(-3, 3)
    axs[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(stem + "_gradient.png", dpi=130, facecolor="white")
    plt.close(fig)
    print(f"  wrote {stem}_gradient.png")

    # 3. source maps of four remote, high-gradient patches
    q = pick[-1]
    far = [n for n in np.argsort(-gr) if patches[n]["dist"] > 25]
    ex = []
    for n in far:
        if all(np.linalg.norm(patches[n]["centre"] - patches[m]["centre"]) > 40 for m in ex):
            ex.append(n)
        if len(ex) == 4:
            break
    fig = plt.figure(figsize=(16, 4.4), facecolor="white")
    for k, n in enumerate(ex):
        P, r = patches[n], q["rows"][n]
        xy, vis, vname = next((v for v in views if v[1][P["vertex"]]), views[0])
        ax = fig.add_subplot(1, 4, k + 1)
        keep = vis[c.F].all(1)
        tri = Triangulation(xy[:, 0], xy[:, 1], c.F[keep])
        cc = r["c"] / max(np.abs(r["c"]).max(), 1e-30)
        field = np.where(lab >= 0, cc[np.clip(lab, 0, None)], 0.0)
        ax.tripcolor(tri, field, cmap=div.reversed(), norm=TwoSlopeNorm(0, -1, 1),
                     shading="gouraud", rasterized=True)
        ax.tricontour(tri, (lab >= 0).astype(float), levels=[0.5], colors=muted,
                      linewidths=0.4)
        v = P["vertex"]
        dvec = np.cos(r["theta"]) * P["e1"] + np.sin(r["theta"]) * P["e2"]
        tip = int(np.argmin(np.linalg.norm(c.V - (P["centre"] + 14 * dvec), axis=1)))
        ax.annotate("", xy[tip], xy[v], arrowprops=dict(arrowstyle="-|>", lw=2, color="black"))
        ax.scatter(*xy[v], s=70, facecolors="none", edgecolors="black", linewidths=1.6)
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(f"{P['name']} (gradient {P['grad']:+.2f}, {P['dist']:.0f} mm from input, "
                     f"{vname})\n{r['n_src']} source regions; interference "
                     f"{r['pair_share']:+.0%} of the flow", fontsize=8, color=ink)
    fig.suptitle(f"where remote waves come from ({q['hz']:.3f} Hz): each input coloured by "
                 f"how much it drives the patch's net flow (red: along the arrow, blue: "
                 f"against)", fontsize=10, color=ink)
    fig.savefig(stem + "_sources.png", dpi=130, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {stem}_sources.png")


def _plot_combine(stem, res, patches, c, run, g1):
    """Maps per frequency - one dot per patch, coloured by how its flow is made (three
    categorical slots, the most the validated palette separates on a map) - and the
    category mix and overshoot against the principal gradient."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.tri import Triangulation
    from render_regimes import _proj
    ink, muted = "#1a1a19", "#6b6a64"
    # three categorical slots (all-pairs safe on a map) and neutral grey for "other"
    col = dict(zip(CATS, ("#2a78d6", "#1baf7a", "#eb6834", "#b5b3ab")))
    greys = LinearSegmentedColormap.from_list("g", ("#f4f3ef", "#b9b7af"))
    lab = np.asarray(run["labels"])
    views = _proj(c.V, c.F, views=((8, 180, "lateral"), (8, 0, "medial")))
    vx = np.array([P["vertex"] for P in patches])
    fig = plt.figure(figsize=(4.6 * len(views) + 0.4, 3.1 * len(res) + 0.8), facecolor="white")
    gs = fig.add_gridspec(len(res), len(views), hspace=0.05, wspace=0.02,
                          top=1 - 0.7 / (3.1 * len(res) + 0.8))
    for ri, q in enumerate(res):
        for vi, (xy, vis, vname) in enumerate(views):
            ax = fig.add_subplot(gs[ri, vi])
            keep = vis[c.F].all(1)
            tri = Triangulation(xy[:, 0], xy[:, 1], c.F[keep])
            ax.tripcolor(tri, g1, cmap=greys, shading="gouraud", rasterized=True)
            ax.tricontour(tri, (lab >= 0).astype(float), levels=[0.5], colors=muted,
                          linewidths=0.6)
            m = vis[vx]
            for k in CATS:
                mm = m & np.array([r["cat"] == k for r in q["rows"]])
                ax.scatter(xy[vx[mm], 0], xy[vx[mm], 1], s=42, color=col[k],
                           edgecolors="white", linewidths=1.0, label=k)
            ax.set_aspect("equal"); ax.axis("off")
            if ri == 0:
                ax.set_title(vname, fontsize=10, color=ink)
            if vi == 0:
                ax.text(-0.02, 0.5, f"{q['hz']:.3f} Hz\n({1 / q['hz']:.0f} s cycle)",
                        transform=ax.transAxes, ha="right", va="center", fontsize=9,
                        color=ink)
    h, l = fig.axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 1 - 0.25 / (3.1 * len(res) + 0.8)))
    fig.suptitle("how each patch's waves are made under the fitted drive: one input region, "
                 "several reinforcing, or the remainder of several cancelling\n"
                 "(background: principal gradient, dark = transmodal; outline: inputs)",
                 fontsize=10, color=ink, y=1.0)
    fig.savefig(stem + "_maps.png", dpi=130, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {stem}_maps.png")

    gr = np.array([P["grad"] for P in patches])
    edges = np.percentile(gr, [0, 20, 40, 60, 80, 100])
    bins = np.clip(np.digitize(gr, edges[1:-1]), 0, 4)
    fig, axs = plt.subplots(1, len(res) + 1, figsize=(3.2 * (len(res) + 1), 3.3),
                            facecolor="white")
    for ax, q in zip(axs, res):
        cats = np.array([r["cat"] for r in q["rows"]])
        bottom = np.zeros(5)
        for k in CATS:
            v = np.array([np.mean(cats[bins == i] == k) for i in range(5)])
            ax.bar(range(5), v, bottom=bottom, color=col[k], edgecolor="white",
                   linewidth=2, label=k)
            bottom += v
        ax.set_xticks(range(5))
        ax.set_xticklabels(["sensori-\nmotor", "", "", "", "trans-\nmodal"], fontsize=7,
                           color=muted)
        ax.set_title(f"{q['hz']:.3f} Hz", fontsize=9, color=ink)
        ax.tick_params(labelsize=7, colors=muted)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axs[0].set_ylabel("share of patches", fontsize=8, color=muted)
    h_, l_ = axs[0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="upper center", ncol=4, frameon=False, fontsize=8,
               bbox_to_anchor=(0.42, 0.93))
    ax = axs[-1]
    # frequency is ordered: one hue, light (slow) to dark (fast) - never the category colours
    pal = [BLUES[i] for i in np.linspace(3, len(BLUES) - 1, len(res)).astype(int)]
    for q, cl in zip(res, pal):
        k = np.array([r["kappa"] for r in q["rows"]])
        ax.plot(range(5), [np.median(k[bins == i]) for i in range(5)], "o-", color=cl,
                lw=2, ms=4, label=f"{q['hz']:.3f} Hz")
    ax.set_yscale("log"); ax.axhline(1, color=muted, lw=0.6)
    ax.set_xticks(range(5))
    ax.set_xticklabels(["sensori-\nmotor", "", "", "", "trans-\nmodal"], fontsize=7,
                       color=muted)
    ax.set_title("parts alone / net (median)", fontsize=9, color=ink)
    ax.legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.tick_params(labelsize=7, colors=muted)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.suptitle("against the principal gradient (fifths of the patches)", fontsize=10,
                 color=ink)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(stem + "_gradient.png", dpi=130, facecolor="white")
    plt.close(fig)
    print(f"  wrote {stem}_gradient.png")


def _plot_media(stem, med):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, muted, band = "#1a1a19", "#6b6a64", "#e4e2dc"
    flat = sorted({m["speed"] for m in med if "mapsmm" not in m["tag"]})
    shade = dict(zip(flat, [BLUES[i] for i in np.linspace(4, len(BLUES) - 1,
                                                          len(flat)).astype(int)]))
    show = ("reinforcing", "cancelling", "single source")
    fig, axs = plt.subplots(2, 3, figsize=(13.5, 7.4), facecolor="white")
    for r, unit in enumerate(("hz", "cyc")):
        for k, ct in enumerate(show):
            ax = axs[r, k]
            # what a random coherent drive gives in the same media: the spread over media
            # and seeds, drawn once per panel as a band
            if all(m["ctl"] for m in med):
                xs, ys = [], []
                for m in med:
                    x = m["hz"] * (m["tau"] if unit == "cyc" else 1)
                    for cc in m["ctl"]:
                        xs.append(x); ys.append(cc[ct])
                gx = np.geomspace(min(x.min() for x in xs), max(x.max() for x in xs), 60)
                Y = np.array([np.where((gx >= x[0]) & (gx <= x[-1]),
                                       np.interp(np.log(gx), np.log(x), y), np.nan)
                              for x, y in zip(xs, ys)])
                ax.fill_between(gx, np.nanmin(Y, 0), np.nanmax(Y, 0), color=band, lw=0,
                                label="random coherent drive (range)")
            if unit == "hz":
                ax.axvspan(0.01, 0.08, color="#f2efe6", lw=0, zorder=0)
            for m in med:
                dash = "mapsmm" in m["tag"]
                x = m["hz"] * (m["tau"] if unit == "cyc" else 1)
                lab = (f"3x map-weighted ({m['tau']:.3g} s decay)" if dash else
                       f"{m['speed'] / 1.5:g}x flat ({m['tau']:.3g} s decay)")
                ax.plot(x, m["frac"][ct], "--" if dash else "-", marker="o", ms=3, lw=2,
                        color=shade.get(m["speed"], BLUES[8]), label=lab)
            ax.set_xscale("log"); ax.set_ylim(0, 1)
            if unit == "cyc":
                ax.axvline(1, color=muted, lw=0.8, ls=":")
            ax.set_xlabel("Hz" if unit == "hz" else "cycles per decay time (Hz x decay s)",
                          fontsize=8, color=muted)
            if k == 0:
                ax.set_ylabel("share of patches", fontsize=8, color=muted)
            if r == 0:
                ax.set_title(ct, fontsize=10, color=ink)
            ax.tick_params(labelsize=7, colors=muted)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
    axs[0, 0].text(0.0105, 0.97, "0.01-0.08 Hz", fontsize=7, color=muted, va="top")
    h_, l_ = axs[0, 0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="upper center", ncol=len(l_), frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, 0.995))
    fig.suptitle("How patches make their waves, across media (fitted drive, whole cortex)",
                 fontsize=11, color=ink, y=1.04)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(stem + ".png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  wrote {stem}.png")


def _plot_bold(stem, hz, obs, out, power, width, gr, patches, c, run, g1):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.tri import Triangulation
    from render_regimes import _proj
    ink, muted = "#1a1a19", "#6b6a64"
    col = dict(zip(CATS, ("#2a78d6", "#1baf7a", "#eb6834", "#b5b3ab")))
    ocol = dict(zip(obs, ("#b5b3ab", "#6250d6", "#e87ba4", "#008300")))
    names = list(obs)
    fig = plt.figure(figsize=(17, 8.6), facecolor="white")
    gs = fig.add_gridspec(2, 5, width_ratios=(1.25, 1, 1, 1, 1), hspace=0.38, wspace=0.28)
    # what each observable keeps: the median patch's weight per frequency
    ax = fig.add_subplot(gs[0, 0])
    for nm in names:
        w = out[nm]["w"] / width[:, None]
        ax.plot(hz, np.median(w, 1) / np.median(w, 1).max(), "o-", ms=3, lw=2,
                color=ocol[nm], label=nm)
    ax.set_xscale("log"); ax.set_xlabel("Hz", fontsize=8, color=muted)
    ax.set_title("what each observable sees of the field\n(median patch, peak = 1)",
                 fontsize=9, color=ink)
    ax.legend(fontsize=7, frameon=False); ax.tick_params(labelsize=7, colors=muted)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    # the category mix along the gradient, per observable
    edges = np.percentile(gr, [0, 20, 40, 60, 80, 100])
    bins = np.clip(np.digitize(gr, edges[1:-1]), 0, 4)
    for k, nm in enumerate(names):
        ax = fig.add_subplot(gs[0, k + 1])
        bottom = np.zeros(5)
        for ct in CATS:
            v = np.array([np.mean(out[nm]["share"][ct][bins == i]) for i in range(5)])
            ax.bar(range(5), v, bottom=bottom, color=col[ct], edgecolor="white",
                   linewidth=2, label=ct)
            bottom += v
        ax.set_xticks(range(5))
        ax.set_xticklabels(["sensori-\nmotor", "", "", "", "trans-\nmodal"], fontsize=7,
                           color=muted)
        ax.set_ylim(0, 1)
        ax.set_title(nm, fontsize=9, color=ocol[nm] if nm != "raw field" else ink)
        ax.tick_params(labelsize=7, colors=muted)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        if k == 0:
            ax.set_ylabel("share of observed wave power", fontsize=8, color=muted)
    h_, l_ = fig.axes[1].get_legend_handles_labels()
    fig.legend(h_, l_, loc="upper center", ncol=4, frameon=False, fontsize=9,
               bbox_to_anchor=(0.6, 0.995))
    # maps: each patch's dominant way of making its observed waves
    greys = LinearSegmentedColormap.from_list("g", ("#f4f3ef", "#b9b7af"))
    lab = np.asarray(run["labels"])
    xy, vis, _ = _proj(c.V, c.F, views=((8, 180, "lateral"),))[0]
    keep = vis[c.F].all(1)
    tri = Triangulation(xy[:, 0], xy[:, 1], c.F[keep])
    vx = np.array([P["vertex"] for P in patches])
    for k, nm in enumerate(names):
        ax = fig.add_subplot(gs[1, k + 1])
        ax.tripcolor(tri, g1, cmap=greys, shading="gouraud", rasterized=True)
        ax.tricontour(tri, (lab >= 0).astype(float), levels=[0.5], colors=muted,
                      linewidths=0.5)
        sh = np.stack([out[nm]["share"][ct] for ct in CATS])
        dom = np.array(CATS)[np.argmax(sh, 0)]
        m = vis[vx]
        for ct in CATS:
            mm = m & (dom == ct)
            ax.scatter(xy[vx[mm], 0], xy[vx[mm], 1], s=30, color=col[ct],
                       edgecolors="white", linewidths=0.8)
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(f"{nm}: dominant, lateral", fontsize=8, color=ink)
    ax = fig.add_subplot(gs[1, 0]); ax.axis("off")
    ax.text(0, 0.95, "each patch's observed waves are weighted\nover frequency by "
            "(observable gain)^2 x\n(the patch's power under the fitted drive)\nx (bin "
            "width); the category shares are\nthe fractions of that weight in frequencies\n"
            "where the patch's flow is single source,\nreinforcing, cancelling or other.\n\n"
            "filters in time change no wave's\ngeometry at a given frequency; they only\n"
            "decide which frequencies are seen.", va="top", fontsize=8.5, color=ink,
            transform=ax.transAxes)
    fig.suptitle("Which way of making waves dominates what is OBSERVED, by reading of "
                 "the observable", fontsize=12, color=ink, y=1.04)
    fig.savefig(stem + ".png", dpi=130, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {stem}.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=("pairs", "decomp", "frames", "types", "steer",
                                    "steermap", "sweeps", "remote", "combine", "bold",
                                    "media"))
    ap.add_argument("--tag", default="nobp_sm6_s3_box")
    ap.add_argument("--hz", default="",
                    help="analysis frequencies; snapped to the nearest solved bins. Default: "
                         "14 bins log-spaced 0.005-0.5 Hz")
    ap.add_argument("--radius", type=float, default=10.0, help="mm")
    ap.add_argument("--nlag", type=int, default=24)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--seconds", type=float, default=1154.0,
                    help="frames: realisation length")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--band", default="0.06,0.12",
                    help="frames: lo,hi Hz the split and the lags are read in, or none")
    ap.add_argument("--grid", type=float, default=15.0,
                    help="frames: spacing of grid patch centres, mm")
    ap.add_argument("--reach", type=float, default=20.0,
                    help="frames: grid patches cover the territory and this far round it")
    ap.add_argument("--cover", type=float, default=0.95,
                    help="frames: name the pieces carrying this share of a patch's "
                         "impulse energy; the rest go into one term")
    ap.add_argument("--maxp", type=int, default=12, help="frames: at most this many named")
    ap.add_argument("--cache", default="",
                    help="frames: where to keep the run and responses (several GB); "
                         "default results/interplay_cache")
    ap.add_argument("--floor", type=float, default=0.3,
                    help="types: below this on both directedness and |expansion|: mixed")
    ap.add_argument("--ninputs", type=int, default=8,
                    help="types: the pieces per patch whose timing and loudness are read")
    ap.add_argument("--drive", default="fitted", choices=("fitted", "independent", "random"),
                    help="combine/bold: the fitted drive, or a control (see control_drive)")
    ap.add_argument("--nregions", type=int, default=3,
                    help="combine: the input regions split out per patch; the rest is one")
    ap.add_argument("--sigma", type=float, default=6.0,
                    help="sweeps: mm, the smoothing that turns the piecewise input map "
                         "into a field (also read at 4 and 9 mm)")
    ap.add_argument("--every-s", type=float, default=1.0, dest="every_s",
                    help="types: seconds between the states read")
    ap.add_argument("--media", default="nobp_sm6_s1_box,nobp_sm6_s3_box,nobp_sm6_s5_box,"
                                       "nobp_sm6_mapsmm_s3_box",
                    help="media: the tags compared; a 'mapsmm' tag is drawn dashed")
    a = ap.parse_args()
    if a.cmd == "media":
        cmd_media(a)
        return
    from mesh_cache import load_cortex
    c = load_cortex("fsaverage5", verbose=False)
    run = waves.load_run(c, a.tag)
    S = waves.Surface(c)
    if a.cmd in ("frames", "types", "steer", "steermap", "sweeps", "remote", "combine",
                 "bold"):
        print(f"  {run['name']}: {len(run['tags'])} pieces")
        dict(frames=cmd_frames, types=cmd_types, steer=cmd_steer, steermap=cmd_steermap,
             sweeps=cmd_sweeps, remote=cmd_remote, combine=cmd_combine,
             bold=cmd_bold)[a.cmd](a, c, S, run)
        return
    hzb = run["idx"] / run["ref_frames"] / run["frame_s"]
    want = ([float(v) for v in a.hz.split(",")] if a.hz else np.geomspace(0.005, 0.5, 14))
    pick = np.unique([int(np.argmin(np.abs(np.log(hzb / f)))) for f in want])
    # one transfer computation serves both commands: always ask for the default grid plus
    # whatever was requested, so the cache is shared
    grid = np.unique(np.r_[[int(np.argmin(np.abs(np.log(hzb / f))))
                            for f in np.geomspace(0.005, 0.5, 14)], pick])
    H, U, depth = waves.transfer_fields(c, run, hzb[grid], velocity=True)
    sel = np.searchsorted(grid, pick)
    print(f"  {run['name']}: {len(run['tags'])} pieces, bins "
          + ", ".join(f"{hzb[b]:.4f}" for b in pick) + " Hz")
    if a.cmd == "pairs":
        cmd_pairs(a, c, S, run, H[sel], U[sel], depth, hzb[pick])
    else:
        cmd_decomp(a, c, S, run, H[sel], U[sel], depth, hzb[pick], run["S"][pick])


if __name__ == "__main__":
    main()
