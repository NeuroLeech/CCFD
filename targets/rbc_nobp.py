"""The RBC resting runs denoised as XCP-D did it, WITHOUT the 0.01-0.08 Hz bandpass.

XCP-D's `denoised` output is bandpassed (2nd order, 0.01-0.08 Hz, per its sidecar), so every
FC target built from it is too. This rebuilds the same 100 runs from the step before: fMRIPrep's
fsLR-32k timeseries (ReproBrainChart NKI_fMRIPrep-Func, public, fcp-indi S3) regressed on the
EXACT design matrix XCP-D used (its `_design.tsv`: 36P - motion, CSF, white matter and global
signal, with derivatives and squares), then 6 mm FWHM on the fsLR midthickness as XCP-D's
`denoisedSmoothed`, then the same resampling to fsaverage5 as rbc.load. The only step left out
is the bandpass. Linear detrending stays: it is part of the regression (XCP-D detrends BOLD and
confounds before regressing), and without it scanner drift dominates FC.

VALIDATION FIRST (--validate): one run processed both ways, WITH the bandpass and without; the
bandpassed version is compared, per vertex, against XCP-D's own denoisedSmoothed for that run,
which is already local. Agreement there is what licenses trusting the unfiltered version.

STORAGE. The fMRIPrep dtseries are ~310 MB each, ~31 GB for the cohort. Each is fetched,
processed and dropped (`datalad drop`) in turn, keeping only the ~37 MB fsaverage5 result in
data/cache, so the disk never holds more than one.

The group FC is written under results/fc/nobp/, NOT results/fc/: fc_score.default_fc and
raw_fc take the NEWEST file matching results/fc/*fc.npy as the target, so writing it beside
the current target would silently replace it for every script in the repo.

BUILT 2026-10-09, all 100 runs (0-3 dummy volumes dropped per run, as XCP-D did), 9,305
vertices. Against the bandpassed target on the 9,303 both cover: edge spearman +0.966 raw,
+0.972 double-centred; per-vertex FC-row correlation median 0.979 (5% 0.913, min 0.781). The
unfiltered FC is weaker - double-centred sd 0.094 against 0.130 (0.72x) - but ranks its edges
almost the same way.

  python targets/rbc_nobp.py --validate
  python targets/rbc_nobp.py --process
  python targets/rbc_nobp.py --build
"""
import _path  # noqa: F401
import os, re, json, time, argparse, subprocess, tempfile
import numpy as np

import rbc
from paths import CACHE, RESULTS

FMRIPREP = os.path.join(rbc.RBC_ROOT, "NKI_fMRIPrep-Func")
FCDIR = os.path.join(RESULTS, "fc", "nobp")


def paths(run):
    """-> dict of the files this run needs, from its XCP-D denoisedSmoothed path."""
    rel = os.path.relpath(run.bold, rbc.XCPD)
    stem = rel.split("_space-fsLR")[0]
    return dict(
        xcpd_smoothed=run.bold,
        design=os.path.join(rbc.XCPD, stem + "_design.tsv"),
        bold=os.path.join(FMRIPREP, stem + "_space-fsLR_den-91k_bold.dtseries.nii"),
        confounds=os.path.join(FMRIPREP, stem + "_desc-confounds_timeseries.tsv"))


def _get(path):
    ds = FMRIPREP if path.startswith(FMRIPREP) else rbc.XCPD
    r = subprocess.run(["datalad", "get", os.path.relpath(path, ds)], cwd=ds, env=rbc._ENV,
                       capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"datalad get failed for {path}\n{r.stdout}\n{r.stderr}")


def _drop(path):
    ds = FMRIPREP if path.startswith(FMRIPREP) else rbc.XCPD
    subprocess.run(["datalad", "drop", "--reckless", "availability", os.path.relpath(path, ds)],
                   cwd=ds, env=rbc._ENV, capture_output=True, text=True)


def read_tsv(path):
    with open(path) as fh:
        head = fh.readline().rstrip("\n").split("\t")
        rows = [ln.rstrip("\n").split("\t") for ln in fh if ln.strip()]
    A = np.array([[np.nan if v in ("n/a", "") else float(v) for v in r] for r in rows])
    return head, A


def n_dummy(confounds):
    """fMRIPrep's non-steady-state volumes, which XCP-D drops (dummy scans 'auto')."""
    head, A = read_tsv(confounds)
    cols = [i for i, h in enumerate(head) if h.startswith("non_steady_state_outlier")]
    if not cols:
        return 0
    hit = np.flatnonzero(np.nansum(A[:, cols], 1) > 0)
    return int(hit.max() + 1) if len(hit) else 0


def detrend(X):
    """Remove mean and linear trend along axis 0."""
    T = X.shape[0]
    B = np.c_[np.ones(T), np.linspace(-1, 1, T)]
    return X - B @ np.linalg.lstsq(B, X, rcond=None)[0]


def regress(Y, D):
    """(T, V) BOLD on (T, k) design, both detrended first; residuals."""
    Yd, Dd = detrend(Y), detrend(np.nan_to_num(D))
    beta = np.linalg.lstsq(Dd, Yd, rcond=None)[0]
    return Yd - Dd @ beta


def bandpass(Y, tr, lo=0.01, hi=0.08, order=2):
    from scipy.signal import butter, filtfilt
    b, a = butter(order, [lo, hi], btype="bandpass", fs=1.0 / tr)
    return filtfilt(b, a, Y, axis=0, padtype="constant")


def smooth_fslr(X, fwhm=6.0):
    """(32492, T) on the fsLR midthickness, within the cortex ROI, as XCP-D smooths."""
    import nibabel as nib
    from neuromaps.datasets import get_atlas_dir
    d = get_atlas_dir("fsLR")
    surf = f"{d}/tpl-fsLR_den-32k_hemi-L_midthickness.surf.gii"
    roi_lab = f"{d}/tpl-fsLR_den-32k_hemi-L_desc-nomedialwall_dparc.label.gii"
    roi = np.asarray(nib.load(roi_lab).darrays[0].data) > 0
    with tempfile.TemporaryDirectory(prefix="nobp_") as td:
        src, rf, out = (os.path.join(td, n) for n in ("in.func.gii", "roi.shape.gii",
                                                       "out.func.gii"))
        g = nib.gifti.GiftiImage()
        for t in range(X.shape[1]):
            g.add_gifti_data_array(nib.gifti.GiftiDataArray(
                np.ascontiguousarray(X[:, t], np.float32), intent="NIFTI_INTENT_NORMAL"))
        nib.save(g, src)
        r = nib.gifti.GiftiImage()
        r.add_gifti_data_array(nib.gifti.GiftiDataArray(roi.astype(np.float32),
                                                        intent="NIFTI_INTENT_NORMAL"))
        nib.save(r, rf)
        p = subprocess.run(["wb_command", "-metric-smoothing", surf, src, str(fwhm), out,
                            "-fwhm", "-roi", rf], capture_output=True, text=True)
        if p.returncode:
            raise RuntimeError(p.stderr)
        return np.stack([a.data for a in nib.load(out).darrays], 1).astype(np.float32)


def process(run, band=False, fetch=True, drop=True, verbose=True, order=2,
            filter_first=True):
    """-> (10242, T) fsaverage5, ok mask, TR, dummies dropped. With `band`, the filter is
    applied to BOLD and design BEFORE the regression (filter_first, XCP-D's order) or to the
    residuals after it; without it the two are the same thing."""
    f = paths(run)
    if fetch:
        for k in ("design", "bold", "confounds"):
            _get(f[k])
    X, valid, tr = rbc.cifti_left_cortex(f["bold"])               # (32492, T)
    _, D = read_tsv(f["design"])
    # XCP-D's design covers only the volumes it kept, so its dummy count is the difference;
    # fMRIPrep's non-steady-state flags are the cross-check
    nd = X.shape[1] - D.shape[0]
    nd_f = n_dummy(f["confounds"])
    if nd < 0 or (nd != nd_f and verbose):
        print(f"    note: {run.sub} XCP-D dropped {nd} volumes, fMRIPrep flags {nd_f}", flush=True)
    if nd < 0:
        raise RuntimeError(f"design {D.shape[0]} rows, bold {X.shape[1]} frames")
    Y = X[:, nd:].T.astype(np.float64)                            # (T, 32492)
    m = valid
    R = np.zeros_like(Y)
    Dn = np.nan_to_num(D)
    if band and filter_first:
        Yf, Df = bandpass(detrend(Y[:, m]), tr, order=order), bandpass(detrend(Dn), tr, order=order)
        R[:, m] = Yf - Df @ np.linalg.lstsq(Df, Yf, rcond=None)[0]
    else:
        R[:, m] = regress(Y[:, m], Dn)
        if band:
            R[:, m] = bandpass(R[:, m], tr, order=order)
    S = smooth_fslr(R.T.astype(np.float32))
    F, ok = rbc.fslr_to_fsaverage5(S, valid)
    if drop:
        _drop(f["bold"])
    if verbose:
        print(f"    {run.sub}: {X.shape[1]} frames, {nd} dummies dropped, TR {tr:.3f}, "
              f"design {D.shape[1]} columns{', bandpassed' if band else ''}", flush=True)
    return F, ok, tr, nd


def cache_path(run):
    return os.path.join(CACHE, f"rbcnobp_{run.key}_fsaverage5.npz")


def cohort():
    import fc_group_rbc as fg
    subs, pinned = fg.cohort_subjects(100, 0)
    runs = rbc.cohort_runs(subs, specs=(("rest", "645"),))[("rest", "645")]
    return runs, pinned


def validate():
    runs, _ = cohort()
    run = runs[0]
    print(f"  validating on {run.sub}: our pipeline WITH the bandpass against XCP-D's "
          f"denoisedSmoothed", flush=True)
    Xc, okc, _ = rbc.load(run, verbose=False)                    # XCP-D, already resampled
    first = True
    for ff in (True, False):
        for order in (1, 2):
            F, ok, tr, nd = process(run, band=True, fetch=first, drop=False, order=order,
                                    filter_first=ff, verbose=first)
            first = False
            T = min(F.shape[1], Xc.shape[1])
            a, b = F[:, -T:], Xc[:, -T:]
            use = ok & okc & (a.std(1) > 0) & (b.std(1) > 0)
            az = (a[use] - a[use].mean(1, keepdims=True)) / a[use].std(1, keepdims=True)
            bz = (b[use] - b[use].mean(1, keepdims=True)) / b[use].std(1, keepdims=True)
            r = (az * bz).mean(1)
            print(f"    {'filter then regress' if ff else 'regress then filter'}, butter N="
                  f"{order}: per-vertex r with XCP-D over {use.sum()} vertices: median "
                  f"{np.median(r):.4f}, 5% {np.percentile(r, 5):.4f}", flush=True)
    G, okg, _, _ = process(run, band=False, fetch=False, drop=True)
    pw = lambda Z: np.abs(np.fft.rfft(Z - Z.mean(1, keepdims=True), axis=1)) ** 2
    fr = np.fft.rfftfreq(G.shape[1], tr)
    P = pw(G[use]).sum(0)
    inb = P[(fr >= 0.01) & (fr <= 0.08)].sum() / P[1:].sum()
    print(f"    without the bandpass: {inb:.1%} of the power is in 0.01-0.08 Hz", flush=True)
    np.savez(cache_path(run), X=G, ok=okg, tr=tr, nd=nd)


def process_all(limit=None):
    runs, _ = cohort()
    t0 = time.time()
    for k, r in enumerate(runs[:limit], 1):
        cp = cache_path(r)
        if os.path.exists(cp):
            continue
        F, ok, tr, nd = process(r, band=False)
        np.savez(cp, X=F, ok=ok, tr=tr, nd=nd)
        print(f"  {k}/{len(runs)} done  [{(time.time()-t0)/60:.0f} min]", flush=True)


def build(metric="spearman", mask_kind="glasser"):
    """Group FC as fc_group_rbc.build does it - same vertices, Fisher z mean - from the
    unfiltered runs."""
    from scipy.stats import rankdata
    import fc_group_rbc as fg
    runs, pinned = cohort()
    keep = fg.cortex_mask(fg.SPACE, mask_kind)
    cover = np.ones(fg.NVERT[fg.SPACE], bool)
    for r in runs:
        z = np.load(cache_path(r))
        cover &= z["ok"] & (z["X"].std(1) > 0)
    idx = np.flatnonzero(keep & cover)
    nV = len(idx)
    acc = np.zeros((nV, nV), np.float32)
    nfr = []
    for k, r in enumerate(runs, 1):
        X = np.load(cache_path(r))["X"]
        nfr.append(X.shape[1])
        Z = rankdata(X[idx], axis=1).astype(np.float32) if metric == "spearman" \
            else X[idx].astype(np.float32)
        Z -= Z.mean(1, keepdims=True)
        Z /= np.maximum(Z.std(1, keepdims=True), 1e-30)
        S = (Z @ Z.T) / Z.shape[1]
        np.clip(S, -0.9999, 0.9999, out=S)
        acc += np.arctanh(S)
    FC = np.tanh(acc / len(runs)); np.fill_diagonal(FC, 1.0)
    os.makedirs(FCDIR, exist_ok=True)
    tag = f"group-RBCNKI{len(runs)}nobp_hemi-L_space-{fg.SPACE}_mask-{mask_kind}"
    np.save(os.path.join(FCDIR, f"{tag}_{metric}fc.npy"), FC.astype(np.float32))
    np.save(os.path.join(FCDIR, f"{tag}_vertices.npy"), idx)
    json.dump(dict(source="ReproBrainChart NKI_fMRIPrep-Func (fMRIPrep 24.1.1), regressed on "
                          "XCP-D 0.10.6's own design.tsv (36P), linear detrend, NO bandpass, "
                          "6 mm FWHM on the fsLR midthickness, resampled as rbc.load",
                   n_subjects=len(runs), frames_per_run=[min(nfr), max(nfr)],
                   cohort_file=os.path.basename(pinned), metric=metric),
              open(os.path.join(FCDIR, f"{tag}_meta.json"), "w"), indent=1)
    print(f"  wrote {FCDIR}/{tag}_{metric}fc.npy  ({nV} vertices)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--process", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--build", action="store_true")
    a = ap.parse_args()
    if a.validate:
        validate()
    if a.process:
        process_all(a.limit)
    if a.build:
        build()
