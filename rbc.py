"""NKI runs from ReproBrainChart, on the fsaverage5 vertices everything else here uses.

RBC pools five cohorts; only NKI is touched by this module, and only through the two
NKI_* DataLad datasets, so nothing from BHRC, dCCNP, HBN or PNC can enter by accident.
Those four are child and adolescent samples and would put an age difference inside any
comparison against the adult rest target.

What this buys over the raw Rockland release: the runs are already preprocessed
(fMRIPrep 24.1.1 + XCP-D 0.10.6) and, unlike nilearn's rest fetch, the TASK runs are
included - CHECKERBOARD at both TRs and BREATHHOLD at 1400. The checkerboard at acq-645
is the same acquisition as the rest scan the FC target is built from, 239 frames at
TR 0.645 s, and its events are a 20 s on / 20 s off block design: a fundamental at
0.025 Hz, inside the 0.01-0.1 Hz band the model's clock was anchored to.

TWO SPACES HAVE TO BE RECONCILED. XCP-D writes fsLR-32k (`space-fsLR_den-91k`); this
project is fsaverage5 throughout - the mesh, the geodesics, the Glasser annotation and
the 9,217-vertex target. Runs are therefore resampled fsLR-32k -> fsaverage5 through the
standard registration spheres, and the result is returned as (10242, T) with the medial
wall included, which is exactly the layout `fc_group_nki.load_subject` returns. So

    X = rbc.load(run)[target.vertices]

is the same expression that already works for the rest data, and every downstream
consumer - FCTarget, zones, caps, ica - takes it unchanged.

The medial wall needs care in that resampling. XCP-D drops it (29,696 of 32,492 vertices
carry data), and a barycentric interpolation that treats the hole as zeros drags the rim
towards zero. The fix is to resample an all-ones validity mask alongside the data and
divide by it, which is what makes the interpolation a weighted mean over the vertices
that actually have data rather than over all of them.

  python rbc.py --list                      # what is available, no downloads
  python rbc.py --subjects 20 --fetch       # pull and cache 20 checkerboard runs
"""
import os, re, glob, json, argparse, subprocess, tempfile
import numpy as np

from paths import CACHE

RBC_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "rbc")
XCPD = os.path.join(RBC_ROOT, "NKI_XCP-D")
BIDS = os.path.join(RBC_ROOT, "NKI_BIDS")
FSAVG5_NV = 10242                                  # left hemisphere, medial wall included
FSLR32K_NV = 32492

# datalad writes a commit for some operations and refuses without an identity; the repo's
# own identity is reused rather than setting anything global on the machine
_ENV = dict(os.environ,
            GIT_AUTHOR_NAME="NeuroLeech", GIT_COMMITTER_NAME="NeuroLeech",
            GIT_AUTHOR_EMAIL="robertleech6+github@gmail.com",
            GIT_COMMITTER_EMAIL="robertleech6+github@gmail.com")


class Run:
    """One functional run: where its files are, and what it is."""
    __slots__ = ("sub", "ses", "task", "acq", "desc", "bold", "events")

    def __init__(self, sub, ses, task, acq, desc, bold, events):
        self.sub, self.ses, self.task, self.acq = sub, ses, task, acq
        self.desc, self.bold, self.events = desc, bold, events

    @property
    def key(self):
        return f"{self.sub}_{self.ses}_{self.task}_{self.acq}_{self.desc}"

    def __repr__(self):
        return f"<Run {self.sub} {self.ses} {self.task} acq-{self.acq} {self.desc}>"


def _acq_matches(name, acq):
    """acq-645 has to match acq-645VARIANT... too.

    RBC appends a VARIANT suffix naming every protocol field a run deviated on, so an
    exact match on 'acq-645' silently drops most of the dataset - for the subject used to
    develop this module it drops ALL of it."""
    m = re.search(r"_acq-([A-Za-z0-9]+?)(VARIANT[A-Za-z]*)?_", name)
    return m is not None and m.group(1) == acq


def runs(task="CHECKERBOARD", acq="645", desc="denoisedSmoothed", session=None,
         limit=None, root=XCPD, bids=BIDS):
    """-> [Run] discovered from the clone. No content is downloaded."""
    if not os.path.isdir(root):
        raise FileNotFoundError(
            f"no NKI_XCP-D clone at {root}. Get it with:\n"
            f"  cd data/rbc && datalad clone "
            f"https://github.com/ReproBrainChart/NKI_XCP-D.git NKI_XCP-D")
    pat = os.path.join(root, "sub-*", "ses-*", "func",
                       f"*_task-{task}_acq-*_space-fsLR_den-91k_desc-{desc}_bold.dtseries.nii")
    out = []
    for p in sorted(glob.glob(pat)):
        nm = os.path.basename(p)
        if not _acq_matches(nm, acq):
            continue
        sub, ses = nm.split("_")[0], nm.split("_")[1]
        if session and ses != session:
            continue
        stem = nm.split("_space-")[0]
        ev = os.path.join(bids, sub, ses, "func", stem + "_events.tsv")
        # lexists, not exists: an annexed file whose content has not been fetched is a
        # BROKEN symlink, and exists() calls it absent. events() fetches on demand, so the
        # path is what matters here - not whether the bytes are already local.
        out.append(Run(sub, ses, task, acq, desc, p, ev if os.path.lexists(ev) else None))
        if limit and len(out) >= limit:
            break
    return out


def _datalad_get(path, verbose=True):
    """Fetch one annexed file. Content sits on the public fcp-indi S3 bucket, so this
    needs no credentials - the UPenn RIA store in the remote list is unreachable and
    git-annex falls through to S3 on its own."""
    if os.path.exists(path) and os.path.getsize(path) > 0:
        try:
            os.stat(path)
            if not os.path.islink(path) or os.path.exists(os.path.realpath(path)):
                return path
        except OSError:
            pass
    ds = XCPD if path.startswith(XCPD) else BIDS
    rel = os.path.relpath(path, ds)
    r = subprocess.run(["datalad", "get", rel], cwd=ds, env=_ENV,
                       capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(os.path.realpath(path)):
        raise RuntimeError(f"datalad get failed for {rel}\n{r.stdout}\n{r.stderr}")
    return path


def cifti_left_cortex(path):
    """-> (32492, T) left-hemisphere dense array, medial wall zero-filled, and the TR.

    The CIFTI stores only the ~29,696 vertices that carry data, with their vertex ids, so
    the dense array is rebuilt from the BrainModelAxis rather than assumed. A `valid` mask
    comes back too: which of the 32,492 the file actually covers."""
    import nibabel as nib
    img = nib.load(path)
    data = np.asarray(img.get_fdata(dtype=np.float32))          # (T, grayordinates)
    ax_t, ax_b = img.header.get_axis(0), img.header.get_axis(1)
    tr = float(getattr(ax_t, "step", np.nan))
    for name, sl, bm in ax_b.iter_structures():
        if "CORTEX_LEFT" in name:
            nv = int(ax_b.nvertices[name])
            X = np.zeros((nv, data.shape[0]), np.float32)
            valid = np.zeros(nv, bool)
            X[bm.vertex] = data[:, sl].T
            valid[bm.vertex] = True
            return X, valid, tr
    raise ValueError(f"no CORTEX_LEFT structure in {path}")


def _gifti(arr):
    import nibabel as nib
    arr = np.atleast_2d(np.asarray(arr, np.float32).T).T if arr.ndim == 1 else arr
    da = [nib.gifti.GiftiDataArray(np.ascontiguousarray(arr[:, i], np.float32),
                                   intent="NIFTI_INTENT_NORMAL",
                                   datatype="NIFTI_TYPE_FLOAT32")
          for i in range(arr.shape[1])]
    return nib.gifti.GiftiImage(darrays=da)


def fslr_to_fsaverage5(X, valid, mask_floor=0.5):
    """(32492, T) fsLR-32k -> (10242, T) fsaverage5, corrected for the medial wall.

    The validity mask is resampled with the data and divided out, so each target vertex is
    a weighted MEAN over the source vertices that carry data instead of a weighted sum
    that counts the medial wall as zeros. Target vertices whose resampled mask falls below
    `mask_floor` are left at zero and reported as uncovered rather than filled with a value
    interpolated mostly from nothing."""
    import nibabel as nib
    from neuromaps import transforms
    tmp = tempfile.mkdtemp(prefix="rbc_")
    try:
        pd = os.path.join(tmp, "d.L.func.gii")
        pm = os.path.join(tmp, "m.L.func.gii")
        nib.save(_gifti(X), pd)
        nib.save(_gifti(valid.astype(np.float32)[:, None]), pm)
        D = transforms.fslr_to_fsaverage(pd, target_density="10k", hemi="L",
                                         method="linear")[0]
        M = transforms.fslr_to_fsaverage(pm, target_density="10k", hemi="L",
                                         method="linear")[0]
        R = np.stack([d.data for d in D.darrays], axis=1).astype(np.float32)
        w = np.asarray(M.darrays[0].data, np.float32)
    finally:
        for f in glob.glob(os.path.join(tmp, "*")):
            os.remove(f)
        os.rmdir(tmp)
    ok = w >= mask_floor
    R[ok] /= w[ok, None]
    R[~ok] = 0.0
    return R, ok


def load(run, fetch=True, cache=True, verbose=True):
    """-> (10242, T) fsaverage5 left-hemisphere timeseries, medial wall included.

    Same layout as fc_group_nki.load_subject, so `load(run)[target.vertices]` is the
    expression the rest of the project already uses."""
    cp = os.path.join(CACHE, f"rbc_{run.key}_fsaverage5.npz")
    if cache and os.path.exists(cp):
        z = np.load(cp)
        return z["X"], z["ok"], float(z["tr"])
    if fetch:
        _datalad_get(run.bold, verbose)
    X, valid, tr = cifti_left_cortex(run.bold)
    R, ok = fslr_to_fsaverage5(X, valid)
    if verbose:
        print(f"    {run.sub} {run.ses}: {X.shape[1]} frames, TR {tr:.3f}s, "
              f"{int(valid.sum())}/{FSLR32K_NV} fsLR vertices with data -> "
              f"{int(ok.sum())}/{FSAVG5_NV} fsaverage5 covered")
    if cache:
        np.savez(cp, X=R, ok=ok, tr=tr)
    return R, ok, tr


def events(run, fetch=True):
    """-> list of (onset_s, duration_s, trial_type) from the BIDS events file."""
    if run.events is None:
        return None
    if fetch:
        _datalad_get(run.events, verbose=False)
    rows = []
    with open(run.events) as f:
        hdr = f.readline().rstrip("\n").split("\t")
        io, id_, it = hdr.index("onset"), hdr.index("duration"), hdr.index("trial_type")
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) > max(io, id_, it):
                rows.append((float(p[io]), float(p[id_]), p[it]))
    return rows


def boxcar(run, n_frames, tr, on="CHECKER"):
    """-> (n_frames,) 0/1 regressor for the named condition, unconvolved.

    Deliberately not convolved with an HRF: the model has its own path from drive to
    observable (the BOLD kernel in units.py), and imposing a canonical HRF here would put
    a second, different haemodynamic assumption in front of it."""
    ev = events(run)
    if ev is None:
        return None
    t = np.arange(n_frames) * tr
    x = np.zeros(n_frames, np.float32)
    for onset, dur, kind in ev:
        if kind == on:
            x[(t >= onset) & (t < onset + dur)] = 1.0
    return x


QC_TSV = os.path.join(XCPD, "study-NKI_desc-func_qc.tsv")

# rest and checkerboard at acq-645 share the acquisition the FC target is built from;
# breathhold only exists at 1400. Ordered cheapest-last so a partial fetch still leaves a
# usable pair.
COHORT_SPECS = (("rest", "645"), ("CHECKERBOARD", "645"), ("BREATHHOLD", "1400"))


def qc_table(path=QC_TSV):
    """-> list of dicts, XCP-D's per-run QC for the whole study.

    `acq` is normalised by stripping the VARIANT suffix, the same rule `_acq_matches`
    applies to filenames, so a run that deviated on some protocol field is still counted
    against the acquisition it belongs to."""
    if not os.path.exists(path):
        _datalad_get(path, verbose=False)
    import csv
    with open(path) as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    for r in rows:
        r["acq_base"] = re.sub(r"VARIANT.*$", "", str(r.get("acq", "")))
    return rows


def cohort(n=100, specs=COHORT_SPECS, session="BAS1", seed=0, exclude_motion=True,
           verbose=True):
    """-> (subject ids, per-spec QC summary) for n subjects holding EVERY spec.

    One subject set across scan types is the whole point: rest and task then differ by the
    stimulus and nothing else - not by cohort, not by pipeline, not by the resampling.

    Subjects are drawn at random (seeded) from those passing XCP-D's motion_exclude on
    every requested run, rather than taking the lowest-motion n. Taking the cleanest tail
    would select a calmer, older subsample, and the target this builds is meant to stand
    for the population rather than for its stillest members."""
    rows = qc_table()
    passing, summary = [], {}
    for task, acq in specs:
        sel = [r for r in rows if r["ses"] == session and r["task"] == task
               and r["acq_base"] == acq]
        ok = [r for r in sel if not exclude_motion or r["motion_exclude"] in ("0", "0.0", "False")]
        fd = np.array([float(r["mean_fd"]) for r in ok if r["mean_fd"] not in ("", "n/a")])
        summary[(task, acq)] = dict(runs=len(sel), passing=len(ok),
                                    fd_median=float(np.median(fd)) if len(fd) else np.nan)
        passing.append({r["sub"] for r in ok})
    common = sorted(set.intersection(*passing))
    if verbose:
        for (task, acq), s in summary.items():
            print(f"    {task:<13s} acq-{acq:<5s} {s['runs']:>5d} runs, "
                  f"{s['passing']:>5d} pass motion QC, median FD {s['fd_median']:.3f}")
        print(f"    {len(common)} subjects pass on all {len(specs)} scans")
    if n and n < len(common):
        rng = np.random.default_rng(seed)
        common = sorted(rng.choice(common, n, replace=False).tolist())
    return common, summary


def cohort_runs(subs, specs=COHORT_SPECS, session="ses-BAS1", desc="denoisedSmoothed"):
    """-> {(task, acq): [Run]} restricted to `subs`, in the order `subs` gives."""
    want = set(subs) if not str(next(iter(subs))).startswith("sub-") else \
           {s.replace("sub-", "") for s in subs}
    out = {}
    for task, acq in specs:
        rr = [r for r in runs(task, acq, desc, session)
              if r.sub.replace("sub-", "") in want]
        out[(task, acq)] = rr
    return out


def fetch_cohort(by_spec, batch=8, verbose=True):
    """datalad get every run, in batches. One call per batch rather than per file: each
    `datalad get` pays a fixed start-up cost, and these are ~90-330 MB files."""
    paths = [r.bold for rr in by_spec.values() for r in rr]
    todo = [p for p in paths if not os.path.exists(os.path.realpath(p))]
    if verbose:
        print(f"  {len(paths)} runs, {len(todo)} still to fetch")
    for i in range(0, len(todo), batch):
        chunk = [os.path.relpath(p, XCPD) for p in todo[i:i + batch]]
        r = subprocess.run(["datalad", "get"] + chunk, cwd=XCPD, env=_ENV,
                           capture_output=True, text=True)
        if verbose:
            done = min(i + batch, len(todo))
            print(f"    {done}/{len(todo)}" + ("" if r.returncode == 0 else "  [ERRORS]"),
                  flush=True)
    return paths


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--task", default="CHECKERBOARD",
                    choices=("CHECKERBOARD", "BREATHHOLD", "rest"))
    ap.add_argument("--acq", default="645")
    ap.add_argument("--desc", default="denoisedSmoothed",
                    choices=("denoised", "denoisedSmoothed"))
    ap.add_argument("--session", default="ses-BAS1")
    ap.add_argument("--subjects", type=int, default=0,
                    help="how many runs to fetch and cache (0 = none, just report)")
    ap.add_argument("--list", action="store_true", help="inventory only")
    ap.add_argument("--cohort", type=int, default=0,
                    help="select N subjects holding rest-645, CHECKERBOARD-645 and "
                         "BREATHHOLD-1400, and fetch all three for each")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    if a.cohort:
        print(f"  selecting {a.cohort} subjects with all of "
              f"{', '.join(t+'-'+q for t, q in COHORT_SPECS)}")
        subs, _ = cohort(a.cohort, seed=a.seed)
        print(f"  cohort: {len(subs)} subjects, seed {a.seed}")
        by = cohort_runs(subs)
        for k, rr in by.items():
            print(f"    {k[0]}-{k[1]}: {len(rr)} runs")
        gb = sum(int(os.readlink(r.bold).split("SHA256E-s")[1].split("--")[0])
                 for rr in by.values() for r in rr) / 1e9
        print(f"  {gb:.1f} GB total")
        with open(os.path.join(CACHE, f"rbc_cohort_{a.cohort}_seed{a.seed}.json"), "w") as f:
            json.dump(dict(subjects=subs, specs=[list(s_) for s_ in COHORT_SPECS],
                           seed=a.seed), f, indent=1)
        fetch_cohort(by)
        return

    rr = runs(a.task, a.acq, a.desc, a.session)
    print(f"  NKI only, from {os.path.relpath(XCPD)}")
    print(f"  {len(rr)} runs: task-{a.task} acq-{a.acq} {a.desc} {a.session or 'any session'}")
    with_ev = sum(r.events is not None for r in rr)
    print(f"  {with_ev} of them have a BIDS events file alongside")
    if a.list or not a.subjects:
        for r in rr[:5]:
            print(f"    {r}")
        if len(rr) > 5:
            print(f"    ... and {len(rr)-5} more")
        return

    from mesh_cache import load_cortex
    import fc_score
    c = load_cortex("fsaverage5", verbose=False)
    t = fc_score.default_target(c, verbose=False)
    print(f"\n  fetching {min(a.subjects, len(rr))} runs")
    covered = []
    for r in rr[:a.subjects]:
        X, ok, tr = load(r)
        covered.append(ok[t.vertices].mean())
    cov = np.array(covered)
    print(f"\n  target-vertex coverage over {len(cov)} runs: "
          f"mean {cov.mean():.4%}, worst {cov.min():.4%} of {t.nV} vertices")
    ev = events(rr[0])
    if ev:
        blocks = [e for e in ev if e[2] == "CHECKER"]
        print(f"  design: {len(ev)} blocks, {len(blocks)} of them {ev[1][2] if len(ev)>1 else ''}"
              f"; block {ev[0][1]:.0f}s, run {ev[-1][0]+ev[-1][1]:.0f}s")


if __name__ == "__main__":
    main()
