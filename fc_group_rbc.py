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


# The checkerboard's event table, which is identical in every run: three CHECKER blocks
# and three FIXATION blocks, 20 s each, alternating from t = 0. It is written here rather
# than read per run because the REST runs are cut at the SAME clock times - that is the
# null for what block-chopping alone does to an FC, and it only means anything if the
# knife falls in the same place.
BLOCK_S = 20.0
ONSETS = {"CHECKER": (20.0, 60.0, 100.0), "FIXATION": (0.0, 40.0, 80.0)}
SHIFT_S, DROP_TR = 5.0, 3


def block_frames(cond, tr, n_frames, shift_s=SHIFT_S, drop_tr=DROP_TR, dur=BLOCK_S,
                 onsets=None):
    """Frame indices of one condition's blocks: shifted for haemodynamics, head trimmed.

    The shift is the group response lag (4.52 s, rounded to 5) and the trim drops the
    rise. At TR 0.645 that leaves 28 frames a block, 84 for three - and CHECKER and
    FIXATION come out the same length by construction, which is the point: two FC
    matrices differing in the stimulus and in nothing else, not even in how much data
    they were estimated from.
    """
    out = []
    for on in (ONSETS[cond] if onsets is None else onsets):
        s = int(round((on + shift_s) / tr)) + drop_tr
        e = int(round((on + dur + shift_s) / tr))
        if e <= n_frames and s < e:
            out.append(np.arange(s, e))
    return out


def window(x, spec, tr):
    """Apply one window spec to a (V, T) run -> (V, T') to be correlated.

    `demean='segment'` removes each block's own mean before concatenating, which is what
    separates a change in COUPLING from the evoked response's shared mean shift. It also
    costs the low frequencies, and exactly how much is bandpass.segment_power - the same
    weighting is put on the model's transfer function so both sides are estimated the
    same way.
    """
    kind = spec.get("kind", "all")
    if kind == "all":
        return x
    if kind == "first":
        return x[:, :spec["n"]]
    if kind == "filter":
        # An FC matrix has no frequency axis - it is one number per edge with every
        # frequency already summed - so a solve against it cannot see a spectral line
        # however fine its own frequency grid is. Filtering the DATA to a narrow band
        # before forming the FC is how a frequency-resolved target gets made out of the
        # machinery that exists: the 20 s on / 20 s off design puts a deterministic,
        # phase-locked line at 0.025 Hz, so an FC built from 0.020-0.030 Hz carries it and
        # one built from 0.033-0.043 Hz does not. The model side matches with --bandpass.
        import bandpass
        lo, hi = spec["band"]
        return np.ascontiguousarray(bandpass.apply(x.T, tr, lo, hi).T)
    segs = block_frames(spec["cond"], tr, x.shape[1],
                        spec.get("shift_s", SHIFT_S), spec.get("drop_tr", DROP_TR))
    if not segs:
        return None
    parts = [x[:, s] for s in segs]
    if spec.get("demean") == "segment":
        parts = [p - p.mean(1, keepdims=True) for p in parts]
    return np.concatenate(parts, axis=1)


def build_many(specs, task="rest", acq="645", n=100, seed=0, mask_kind="glasser",
               metric="spearman", dtype="float32", outdir=None, idx=None,
               subs=None, verbose=True):
    """Every FC this task can give, in ONE pass over the subjects.

    A separate `build` per target would read the same 35 MB per subject twelve times over
    for nothing. Each run is loaded once and every accumulator is fed from it, which also
    guarantees the matrices are built from exactly the same frames of the same runs.

    `idx` pins the vertex set. It must be pinned: coverage is the intersection over the
    runs that went into a matrix, so letting each target derive its own would give the
    twelve matrices twelve different vertex sets and nothing could be differenced.
    """
    outdir = outdir or os.path.join(FCDIR, "task")
    if subs is None:
        subs, _ = cohort_subjects(n, seed)
    runs = rbc.cohort_runs(subs, specs=((task, acq),))[(task, acq)]
    if verbose:
        print(f"  {task}-{acq}: {len(runs)} runs for {len(subs)} subjects, "
              f"{len(idx)} pinned vertices")
    nV = len(idx)
    acc = {s["name"]: np.zeros((nV, nV), np.float32) for s in specs}
    cnt = {s["name"]: 0 for s in specs}
    shape = {}
    t0 = time.time()
    for k, r in enumerate(runs, 1):
        X, _, tr = rbc.load(r, verbose=False)
        x = X[idx]
        for s in specs:
            xw = window(x, s, tr)
            if xw is None or xw.shape[1] < 8:
                continue
            Z = rankdata(xw, axis=1).astype(np.float32) if metric == "spearman" \
                else xw.astype(np.float32)
            Z -= Z.mean(1, keepdims=True)
            Z /= np.maximum(Z.std(1, keepdims=True), 1e-30)
            C = (Z @ Z.T) / Z.shape[1]
            np.clip(C, -0.9999, 0.9999, out=C)
            acc[s["name"]] += np.arctanh(C)
            cnt[s["name"]] += 1
            shape[s["name"]] = xw.shape[1]
            del Z, C
        del X, x
        if verbose and (k % 10 == 0 or k == len(runs)):
            print(f"  {k:3d}/{len(runs)}  [{time.time()-t0:.0f}s]", flush=True)

    os.makedirs(outdir, exist_ok=True)
    paths = {}
    for s in specs:
        nm = s["name"]
        FC = np.tanh(acc[nm] / max(cnt[nm], 1))
        np.fill_diagonal(FC, 1.0)
        tag = f"group-{nm}_hemi-L_space-{SPACE}_mask-{mask_kind}"
        p = os.path.join(outdir, f"{tag}_{metric}fc.npy")
        np.save(p, FC.astype(dtype, copy=False))
        np.save(os.path.join(outdir, f"{tag}_vertices.npy"), idx)
        with open(os.path.join(outdir, f"{tag}_meta.json"), "w") as fh:
            json.dump(dict(
                source="ReproBrainChart NKI_XCP-D (fMRIPrep 24.1.1 + XCP-D 0.10.6)",
                task=task, acq=acq, session="ses-BAS1", window=s,
                n_runs=cnt[nm], frames_per_run=shape.get(nm), space=SPACE,
                mask=mask_kind, n_vertices=int(nV), correlation=metric,
                averaging="Fisher z across subjects, tanh of the mean",
                seed=seed, subjects=list(subs)), fh, indent=2)
        paths[nm] = p
        if verbose:
            iu = np.triu_indices(min(nV, 1500), 1)
            print(f"  wrote {os.path.basename(p)}  {cnt[nm]} runs, "
                  f"{shape.get(nm)} frames, edge sd "
                  f"{FC[:1500, :1500][iu].std():.4f}")
        del FC
    return paths


def shared_vertices(subs, tasks=(("rest", "645"), ("CHECKERBOARD", "645")),
                    mask_kind="glasser", verbose=True):
    """Vertices covered in EVERY run of EVERY task, intersected with the mask.

    One vertex set for every matrix built here. Without it the rest and checkerboard
    matrices would sit on different vertices and no per-piece or per-edge comparison
    between their solved inputs would be a comparison."""
    keep = cortex_mask(SPACE, mask_kind)
    cover = np.ones(NVERT[SPACE], bool)
    for task, acq in tasks:
        runs = rbc.cohort_runs(subs, specs=((task, acq),))[(task, acq)]
        if verbose:
            print(f"  coverage over {len(runs)} {task}-{acq} runs")
        for k, r in enumerate(runs, 1):
            X, ok, _ = rbc.load(r, verbose=False)
            cover &= ok & (X.std(1) > 0)
            del X
            if verbose and k % 20 == 0:
                print(f"    {k}/{len(runs)}", flush=True)
    idx = np.flatnonzero(keep & cover)
    if verbose:
        print(f"  {len(idx)} vertices ({int((keep & ~cover).sum())} of the "
              f"{mask_kind} mask are outside coverage on some run)")
    return idx


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


# The twelve matrices the visual-drive comparison needs, and what each one is for.
# Every chopped window is built twice - with the block means left in and taken out -
# because the first asks whether the stimulus adds a shared evoked component and the
# second whether the coupling changed underneath it, and those are different questions.
# The task fundamental and a control band between it and the second harmonic. Equal
# width, both clear of 0.05 and 0.075 Hz.
TASK_HZ, OFF_HZ = (0.020, 0.030), (0.033, 0.043)

NARROW_TARGETS = {
    "CHECKERBOARD": [
        dict(name="check_f025", kind="filter", band=TASK_HZ),
        dict(name="check_f038", kind="filter", band=OFF_HZ),
    ],
    "rest": [
        dict(name="rest_f025", kind="filter", band=TASK_HZ),
        dict(name="rest_f038", kind="filter", band=OFF_HZ),
    ],
}

TASK_TARGETS = {
    "CHECKERBOARD": [
        dict(name="check_full", kind="all"),
        dict(name="check_on_in", kind="blocks", cond="CHECKER", demean="run"),
        dict(name="check_on_out", kind="blocks", cond="CHECKER", demean="segment"),
        dict(name="check_off_in", kind="blocks", cond="FIXATION", demean="run"),
        dict(name="check_off_out", kind="blocks", cond="FIXATION", demean="segment"),
    ],
    "rest": [
        dict(name="rest_full", kind="all"),
        dict(name="rest239", kind="first", n=239),
        dict(name="rest_on_in", kind="blocks", cond="CHECKER", demean="run"),
        dict(name="rest_on_out", kind="blocks", cond="CHECKER", demean="segment"),
        dict(name="rest_off_in", kind="blocks", cond="FIXATION", demean="run"),
        dict(name="rest_off_out", kind="blocks", cond="FIXATION", demean="segment"),
    ],
}


def build_task_targets(n=100, seed=0, mask_kind="glasser", outdir=None, narrow=False,
                       verbose=True):
    """Both passes, one shared vertex set, into results/fc/task.

    NOT into results/fc. `fc_score.default_fc` and `raw_fc` take the NEWEST matrix
    matching their glob, so a target written beside the resting one silently becomes what
    every script in the project scores against - the hazard this module's own docstring
    records. A sibling directory leaves the default where it is, and everything here
    passes its target by path.
    """
    outdir = outdir or os.path.join(FCDIR, "task")
    subs, pinned = cohort_subjects(n, seed)
    # only subjects with BOTH scans; the whole design is one subject set across conditions
    have = rbc.cohort_runs(subs, specs=(("rest", "645"), ("CHECKERBOARD", "645")))
    keep = sorted(set(r.sub.replace("sub-", "") for r in have[("rest", "645")]) &
                  set(r.sub.replace("sub-", "") for r in have[("CHECKERBOARD", "645")]))
    if verbose:
        print(f"  {len(keep)} of {len(subs)} subjects have both rest-645 and "
              f"CHECKERBOARD-645")
    idx_path = os.path.join(CACHE, f"rbc_task_vertices_{len(keep)}_{mask_kind}.npy")
    if os.path.exists(idx_path):
        idx = np.load(idx_path)
        print(f"  loaded {os.path.basename(idx_path)}: {len(idx)} vertices")
    else:
        idx = shared_vertices(keep, mask_kind=mask_kind, verbose=verbose)
        np.save(idx_path, idx)
        print(f"  wrote {os.path.basename(idx_path)}")
    out = {}
    for task, specs in (NARROW_TARGETS if narrow else TASK_TARGETS).items():
        out.update(build_many(specs, task=task, subs=keep, idx=idx, outdir=outdir,
                              seed=seed, mask_kind=mask_kind, verbose=verbose))
    return out, idx


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--targets", action="store_true",
                    help="build the eleven task/rest matrices on one shared vertex set")
    ap.add_argument("--narrow", action="store_true",
                    help="build the four NARROWBAND matrices instead: the whole runs "
                         "filtered to the task fundamental and to a control band")
    ap.add_argument("--cohort", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mask", default="glasser", choices=("fsaverage", "glasser"),
                    dest="mask_kind")
    ap.add_argument("--metric", default="spearman", choices=("spearman", "pearson"))
    ap.add_argument("--dtype", default="float32", choices=("float32", "float16"))
    ap.add_argument("--dry-run", action="store_true", dest="dry_run")
    a = ap.parse_args()
    if a.targets or a.narrow:
        build_task_targets(n=a.cohort, seed=a.seed, mask_kind=a.mask_kind,
                           narrow=a.narrow)
        return
    build(n=a.cohort, seed=a.seed, mask_kind=a.mask_kind, metric=a.metric,
          dtype=a.dtype, dry_run=a.dry_run)


if __name__ == "__main__":
    main()
