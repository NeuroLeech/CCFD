"""Group-average vertexwise FC on fsaverage5, from the NKI enhanced surface release.

Nothing here is resampled: nilearn ships these subjects already on fsaverage5, already
smoothed at 6 mm FWHM, 895 frames each. That removes the fs_LR -> fsaverage step and its
medial wall edge effects entirely, and averaging ~100 subjects buys back the per-vertex
SNR that makes a single MSC subject's long-range structure sit near zero.

Per subject: Spearman FC over the kept vertices, Fisher z, accumulate; the group matrix
is tanh of the mean z. Output matches fc_vertexwise's layout (matrix, vertices, meta) so
FCTarget reads it without changes.

  python fc_group_nki.py                       # all subjects on disk, glasser mask
  python fc_group_nki.py --n-subjects 20 --metric pearson
"""
import os, glob, json, time, argparse
import numpy as np
import nibabel as nib
from scipy.stats import rankdata

from fc_vertexwise import cortex_mask, NVERT, FCDIR

NKI = os.path.expanduser("~/nilearn_data/nki_enhanced_surface")


def subject_files(hemi="left", n=None, root=NKI, min_frames=100):
    """Subjects with a usable run. One release file (A00051882) holds 5 frames rather
    than 895; correlations from 5 samples sit near +-1 and, after the Fisher-z clip,
    enter the group average at +-4.95 against typical values near 0.2."""
    fs = sorted(glob.glob(os.path.join(root, "A*", f"A*_{hemi}_preprocessed_fwhm6.gii")))
    if min_frames:
        keep = []
        for f in fs:
            n_fr = len(nib.load(f).darrays)
            if n_fr >= min_frames:
                keep.append(f)
            else:
                print(f"  skipping {os.path.basename(os.path.dirname(f))}: "
                      f"{n_fr} frames")
        fs = keep
    if not fs:
        raise FileNotFoundError(
            f"no NKI surface data in {root} - fetch it with\n"
            f"  python -c \"from nilearn.datasets import fetch_surf_nki_enhanced;"
            f" fetch_surf_nki_enhanced(n_subjects=100)\"")
    return fs[:n] if n else fs


def load_subject(path):
    return np.stack([d.data for d in nib.load(path).darrays], axis=1).astype(np.float32)


def build(n_subjects=None, mask_kind="glasser", metric="spearman", hemi="left",
          dtype="float32", outdir=FCDIR, verbose=True):
    space = "fsaverage5"
    files = subject_files(hemi, n_subjects)
    keep = cortex_mask(space, mask_kind)

    # a vertex is only usable if every subject has signal there; the NKI medial wall is
    # the FreeSurfer one, so a glasser mask asks for a few vertices nobody covers
    if verbose:
        print(f"NKI group FC: {len(files)} subjects, {space}, {mask_kind} mask")
    cover = np.ones(NVERT[space], bool)
    for p in files:
        cover &= load_subject(p).std(1) > 0
    keep = keep & cover
    idx = np.flatnonzero(keep)
    nV = len(idx)
    if verbose:
        print(f"  {nV} vertices ({int((~cover & cortex_mask(space, mask_kind)).sum())} "
              f"of the {mask_kind} mask are outside NKI coverage)")

    # accumulate in float32: 100 z-values of magnitude < 5 lose nothing to rounding here,
    # and a float64 accumulator would cost another 685 MB and a copy per subject
    acc = np.zeros((nV, nV), np.float32)
    t0 = time.time()
    for k, p in enumerate(files, 1):
        X = load_subject(p)[idx]
        Z = rankdata(X, axis=1).astype(np.float32) if metric == "spearman" else X.copy()
        Z -= Z.mean(1, keepdims=True)
        Z /= Z.std(1, keepdims=True)
        S = (Z @ Z.T) / Z.shape[1]
        np.clip(S, -0.9999, 0.9999, out=S)      # the diagonal is overwritten at the end
        acc += np.arctanh(S)
        if verbose and (k % 10 == 0 or k == len(files)):
            print(f"  {k:3d}/{len(files)} subjects  [{time.time()-t0:.0f}s]", flush=True)
        del X, Z, S

    acc /= len(files)
    FC = np.tanh(acc)
    np.fill_diagonal(FC, 1.0)
    FC = FC.astype(dtype, copy=False)
    del acc

    os.makedirs(outdir, exist_ok=True)
    tag = f"group-NKI{len(files)}_hemi-{'L' if hemi == 'left' else 'R'}_space-{space}_mask-{mask_kind}"
    fc_path = os.path.join(outdir, f"{tag}_{'spearman' if metric == 'spearman' else 'pearson'}fc.npy")
    np.save(fc_path, FC)
    np.save(os.path.join(outdir, f"{tag}_vertices.npy"), idx)
    with open(os.path.join(outdir, f"{tag}_meta.json"), "w") as fh:
        json.dump(dict(source="NKI enhanced surface (nilearn fetch_surf_nki_enhanced)",
                       n_subjects=len(files), space=space, hemi=hemi, mask=mask_kind,
                       n_vertices=int(nV), frames_per_subject=895, tr=0.645,
                       smoothing_fwhm_native=6.0, correlation=metric,
                       averaging="Fisher z across subjects, tanh of the mean",
                       resampling="none - the release is already on fsaverage5",
                       subjects=[os.path.basename(os.path.dirname(p)) for p in files],
                       dtype=dtype, fc_file=os.path.basename(fc_path)), fh, indent=2)
    if verbose:
        print(f"  wrote {fc_path}  {FC.shape} {FC.dtype}")
    return FC, idx, fc_path


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-subjects", type=int, default=None, dest="n_subjects")
    ap.add_argument("--mask", default="glasser", choices=("fsaverage", "glasser"),
                    dest="mask_kind")
    ap.add_argument("--metric", default="spearman", choices=("spearman", "pearson"))
    ap.add_argument("--dtype", default="float32", choices=("float32", "float16"))
    ap.add_argument("--outdir", default=FCDIR)
    a = ap.parse_args()
    build(n_subjects=a.n_subjects, mask_kind=a.mask_kind, metric=a.metric,
          dtype=a.dtype, outdir=a.outdir)


if __name__ == "__main__":
    main()
