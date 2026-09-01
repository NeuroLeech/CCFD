"""Does per-subject PCA truncation make the FC more reproducible?

A single subject's FC has split-half reliability 0.446 and effective rank 21, so most of
the 895-dimensional record is noise. Truncating to the leading K principal components
should remove some of it. But PCA keeps the highest-VARIANCE directions, which is not the
same as the most reliable ones - global signal and motion are high variance - so whether
it helps has to be measured rather than assumed.

The measurement must not let the denoising see both halves. PCA is fitted SEPARATELY on
each half of the run and each half truncated to K, then the two half-FCs are correlated.
Fitting PCA on the whole run and then splitting would let the components carry information
across the split and inflate the reliability for free.

Reported per K: split-half FC reliability, the Spearman-Brown ceiling it implies, the
effective rank of the truncated FC, and its correlation with the group.

  python denoise.py --nsub 8
"""
import argparse
import numpy as np

from mesh_cache import load_cortex
import fc_score, xspec, fc_group_nki as nki


def eff_rank(ev):
    e = np.clip(np.asarray(ev, float), 0, None)
    return float(e.sum() ** 2 / max((e ** 2).sum(), 1e-300))


def zs(X):
    Z = X - X.mean(1, keepdims=True)
    sd = Z.std(1, keepdims=True); sd[sd == 0] = 1.0
    return Z / sd


def truncate(X, K):
    """Rank-K PCA reconstruction of a (V, T) block, components over time."""
    if K is None or K >= min(X.shape):
        return X
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    return (U[:, :K] * s[:K]) @ Vt[:K]


TR = 0.645


def bandpass(X, lo=0.01, hi=0.10, tr=TR):
    """Keep only the resting-state band, by zeroing rfft bins outside it.

    Principled rather than variance-based: 84% of BOLD variance sits in 0.01-0.1 Hz
    (measured), and the model is already low-passed to the fMRI Nyquist, so band-limiting
    the data makes the two sides of the comparison the same kind of object. Unlike PCA
    truncation this discards by FREQUENCY, and the noise that PCA could not reach - high
    variance, spread through the leading components - is largely out of band."""
    T = X.shape[1]
    F = np.fft.rfft(X, axis=1)
    f = np.fft.rfftfreq(T, d=tr)
    F[:, (f < lo) | (f > hi)] = 0.0
    return np.fft.irfft(F, n=T, axis=1)


def gsr(X):
    """Regress out the mean timecourse over vertices."""
    g = X.mean(0); g = g - g.mean()
    d = float(g @ g)
    return X if d <= 0 else X - np.outer((X @ g) / d, g)


def fc(X, centre=True):
    Z = zs(X)
    C = (Z @ Z.T) / Z.shape[1]
    if centre:
        C = C - C.mean(0, keepdims=True) - C.mean(1, keepdims=True) + C.mean()
        np.fill_diagonal(C, 1.0)
    return C


def cor_off(A, B, iu):
    a, b = A[iu], B[iu]
    a = a - a.mean(); b = b - b.mean()
    return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--nsub", type=int, default=8)
    ap.add_argument("--nvert", type=int, default=1000)
    ap.add_argument("--ks", default="5,10,20,50,100,200,0",
                    help="components to keep; 0 = no truncation")
    a = ap.parse_args()
    Ks = [int(v) for v in a.ks.split(",")]

    c = load_cortex("fsaverage5", verbose=False)
    t = fc_score.default_target(c, metric="pearson", verbose=False)
    sub = xspec.medoid_subset(t, a.nvert)
    n = len(sub); iu = np.triu_indices(n, 1)
    verts = t.vertices[sub]
    G = np.asarray(t.target_fc()[np.ix_(sub, sub)], np.float64)
    G = G - G.mean(0, keepdims=True) - G.mean(1, keepdims=True) + G.mean()

    files = nki.subject_files("left")[:a.nsub]
    X = [zs(nki.load_subject(p)[verts].astype(np.float64)) for p in files]
    T = X[0].shape[1]
    print(f"  {len(X)} subjects, {n} vertices, {T} frames; PCA fitted per half "
          f"({T//2} frames each)\n")
    print(f"  {'cleaning':<22s} {'split-half':>11s} {'ceiling':>8s} {'eff rank':>9s} "
          f"{'vs group':>9s}")
    for nm, fn in (("none", lambda Z: Z),
                   ("bandpass 0.01-0.1", bandpass),
                   ("GSR", gsr),
                   ("GSR + bandpass", lambda Z: bandpass(gsr(Z))),
                   ("bandpass + GSR", lambda Z: gsr(bandpass(Z)))):
        sh, er, vg = [], [], []
        for Z in X:
            h1, h2 = fn(Z[:, :T // 2]), fn(Z[:, T // 2:])
            sh.append(cor_off(fc(h1), fc(h2), iu))
            C = fc(fn(Z))
            er.append(eff_rank(np.abs(np.linalg.eigvalsh(C))))
            vg.append(cor_off(C, G, iu))
        m = float(np.mean(sh))
        rel = 2 * m / (1 + m) if m > -1 else np.nan
        print(f"  {nm:<22s} {m:>11.4f} {np.sqrt(max(rel,0)):>8.4f} "
              f"{np.mean(er):>9.1f} {np.mean(vg):>9.4f}", flush=True)
    print()
    print(f"  {'K':>6s} {'split-half':>11s} {'ceiling':>8s} {'eff rank':>9s} "
          f"{'vs group':>9s} {'var kept':>9s}")
    for K in Ks:
        kk = None if K == 0 else K
        sh, er, vg, vk = [], [], [], []
        for Z in X:
            h1, h2 = Z[:, :T // 2], Z[:, T // 2:]
            a1, a2 = truncate(h1, kk), truncate(h2, kk)
            sh.append(cor_off(fc(a1), fc(a2), iu))
            full = truncate(Z, kk)
            C = fc(full)
            er.append(eff_rank(np.abs(np.linalg.eigvalsh(C))))
            vg.append(cor_off(C, G, iu))
            vk.append(float((full ** 2).sum() / max((Z ** 2).sum(), 1e-300)))
        m = float(np.mean(sh))
        rel = 2 * m / (1 + m) if m > -1 else np.nan
        lbl = "all" if K == 0 else str(K)
        print(f"  {lbl:>6s} {m:>11.4f} {np.sqrt(max(rel,0)):>8.4f} "
              f"{np.mean(er):>9.1f} {np.mean(vg):>9.4f} {np.mean(vk):>9.3f}",
              flush=True)
    print(f"\n  'ceiling' is Spearman-Brown from the half reliability: the bound on "
          f"fitting\n  that subject's full-run FC. 'vs group' is how much truncation "
          f"moves the\n  subject toward the group average.")


if __name__ == "__main__":
    main()
