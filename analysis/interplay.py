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


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=("pairs", "decomp", "frames"))
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
    a = ap.parse_args()
    from mesh_cache import load_cortex
    c = load_cortex("fsaverage5", verbose=False)
    run = waves.load_run(c, a.tag)
    S = waves.Surface(c)
    if a.cmd == "frames":
        print(f"  {run['name']}: {len(run['tags'])} pieces")
        cmd_frames(a, c, S, run)
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
