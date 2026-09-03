"""Group-average vertexwise FC on fsaverage5, from the RBC cohort's resting runs.

fc_group_nki builds the same object from nilearn's release. This one exists because that
release ships rest and nothing else, so the target and any task scan would have come from
different preprocessing. Here rest, checkerboard and breath-hold are the same 100 people,
one session, fMRIPrep 24.1.1 + XCP-D 0.10.6, and one resampling into fsaverage5 - the
same wb_command call the MSC data goes through.

The construction is otherwise deliberately identical to fc_group_nki: per subject a
Spearman FC over the kept vertices, Fisher z, accumulate, tanh of the mean. Keeping that
fixed is what makes the two targets comparable as targets, so that a score difference is
attributable to the data rather than to how the matrix was assembled.

WRITING THIS CHANGES THE DEFAULT TARGET. fc_score.default_fc and raw_fc both take the
NEWEST matching file in the FC directory, so from the moment this lands every script that
calls default_target scores against it instead of against group-NKI99. That is the point,
but it also means the numbers PLAN.md records are no longer being reproduced by a re-run:
they were measured against the nilearn target, on a clock anchored to the nilearn
preprocessing. Both anchors - the 9.03 s decay and the f^-2.60 spectrum - have to be
re-derived here before a fit against this target means anything.

  python fc_group_rbc.py --cohort 100                # build from the pinned cohort
  python fc_group_rbc.py --cohort 100 --dry-run      # coverage and sizes, no FC
"""
import os, json, time, argparse
import numpy as np
from scipy.stats import rankdata

from fc_vertexwise import cortex_mask, NVERT, FCDIR
from paths import CACHE
import rbc

SPACE = "fsaverage5"


def cohort_subjects(n, seed=0):
    """The pinned cohort if rbc.py wrote one, otherwise selected fresh and pinned now."""
    p = os.path.join(CACHE, f"rbc_cohort_{n}_seed{seed}.json")
    if os.path.exists(p):
        return json.load(open(p))["subjects"], p
    subs, _ = rbc.cohort(n, seed=seed)
    with open(p, "w") as f:
        json.dump(dict(subjects=subs, specs=[list(s) for s in rbc.COHORT_SPECS],
                       seed=seed), f, indent=1)
    return subs, p


def build(n=100, seed=0, mask_kind="glasser", metric="spearman", dtype="float32",
          outdir=FCDIR, dry_run=False, verbose=True):
    subs, pinned = cohort_subjects(n, seed)
    runs = rbc.cohort_runs(subs, specs=(("rest", "645"),))[("rest", "645")]
    if len(runs) != len(subs):
        raise SystemExit(f"  {len(runs)} rest runs for {len(subs)} subjects - "
                         f"the cohort and the clone disagree")
    if verbose:
        print(f"RBC group FC: {len(runs)} subjects, {SPACE}, {mask_kind} mask")
        print(f"  cohort pinned in {os.path.relpath(pinned)}")

    # coverage first, in one pass: a vertex is usable only where EVERY subject has both
    # resampling coverage and a non-flat timeseries. Done before any FC so a subject that
    # would silently drop a region is visible rather than folded into the average.
    keep = cortex_mask(SPACE, mask_kind)
    cover = np.ones(NVERT[SPACE], bool)
    trs, nframes = [], []
    t0 = time.time()
    for k, r in enumerate(runs, 1):
        X, ok, tr = rbc.load(r, verbose=False)
        cover &= ok & (X.std(1) > 0)
        trs.append(tr); nframes.append(X.shape[1])
        if verbose and (k % 10 == 0 or k == len(runs)):
            print(f"  coverage {k:3d}/{len(runs)}  [{time.time()-t0:.0f}s]", flush=True)
        del X
    idx = np.flatnonzero(keep & cover)
    nV = len(idx)
    if verbose:
        lost = int((keep & ~cover).sum())
        print(f"  {nV} vertices ({lost} of the {mask_kind} mask are outside coverage)")
        print(f"  frames per run {min(nframes)}-{max(nframes)}, "
              f"TR {min(trs):.4f}-{max(trs):.4f} s")
    if dry_run:
        return None, idx, None

    acc = np.zeros((nV, nV), np.float32)
    t0 = time.time()
    for k, r in enumerate(runs, 1):
        X, _, _ = rbc.load(r, verbose=False)
        Z = rankdata(X[idx], axis=1).astype(np.float32) if metric == "spearman" \
            else X[idx].astype(np.float32)
        Z -= Z.mean(1, keepdims=True)
        Z /= np.maximum(Z.std(1, keepdims=True), 1e-30)
        S = (Z @ Z.T) / Z.shape[1]
        np.clip(S, -0.9999, 0.9999, out=S)
        acc += np.arctanh(S)
        if verbose and (k % 10 == 0 or k == len(runs)):
            print(f"  FC {k:3d}/{len(runs)}  [{time.time()-t0:.0f}s]", flush=True)
        del X, Z, S
    acc /= len(runs)
    FC = np.tanh(acc)
    np.fill_diagonal(FC, 1.0)
    FC = FC.astype(dtype, copy=False)
    del acc

    os.makedirs(outdir, exist_ok=True)
    tag = (f"group-RBCNKI{len(runs)}_hemi-L_space-{SPACE}_mask-{mask_kind}")
    fc_path = os.path.join(outdir, f"{tag}_{metric}fc.npy")
    np.save(fc_path, FC)
    np.save(os.path.join(outdir, f"{tag}_vertices.npy"), idx)
    with open(os.path.join(outdir, f"{tag}_meta.json"), "w") as fh:
        json.dump(dict(
            source="ReproBrainChart NKI_XCP-D (fMRIPrep 24.1.1 + XCP-D 0.10.6)",
            derivative="space-fsLR_den-91k_desc-denoisedSmoothed_bold.dtseries.nii",
            task="rest", acq="645", session="ses-BAS1",
            n_subjects=len(runs), space=SPACE, hemi="left", mask=mask_kind,
            n_vertices=int(nV), frames_per_run=[int(min(nframes)), int(max(nframes))],
            tr=float(np.median(trs)), correlation=metric,
            averaging="Fisher z across subjects, tanh of the mean",
            resampling=("fsLR-32k -> fsaverage5, wb_command -metric-resample "
                        "ADAP_BARY_AREA with area metrics and -current-roi, via "
                        "fc_vertexwise.resample"),
            selection=("XCP-D motion_exclude==0 on rest-645, CHECKERBOARD-645 and "
                       "BREATHHOLD-1400; random draw from those passing"),
            cohort_file=os.path.basename(pinned), seed=seed,
            subjects=list(subs), dtype=dtype, fc_file=os.path.basename(fc_path)), fh,
            indent=2)
    if verbose:
        print(f"  wrote {fc_path}  {FC.shape} {FC.dtype}")
        print(f"  this is now the newest FC matrix, so default_target resolves to it")
    return FC, idx, fc_path


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cohort", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mask", default="glasser", choices=("fsaverage", "glasser"),
                    dest="mask_kind")
    ap.add_argument("--metric", default="spearman", choices=("spearman", "pearson"))
    ap.add_argument("--dtype", default="float32", choices=("float32", "float16"))
    ap.add_argument("--dry-run", action="store_true", dest="dry_run")
    a = ap.parse_args()
    build(n=a.cohort, seed=a.seed, mask_kind=a.mask_kind, metric=a.metric,
          dtype=a.dtype, dry_run=a.dry_run)


if __name__ == "__main__":
    main()
