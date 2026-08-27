"""Vertexwise Spearman functional connectivity for MSC left cortex.

The MSC dtseries are fs_LR 32k (CIFTI-1, 29,696 left vertices, medial wall absent).
The model meshes are fsaverage5/6, so the fMRI has to be resampled. That is done with
wb_command -metric-resample ADAP_BARY_AREA using the registration spheres and vertex
area metrics that neuromaps distributes - the same call neuromaps' fslr_to_fsaverage
makes, run once per session on the whole frame block instead of frame by frame.

Vertex ordering is checked, not assumed: neuromaps' fsaverage 10k/41k spheres are
vertex-for-vertex identical to FreeSurfer's fsaverage5/6 and to data/surf/*/infl_left,
so a row of the FC matrix indexes the same vertex as a row of Cortex.

Per session: motion frames dropped with the tmask, then each vertex's timecourse is
rank transformed and z-scored *within* the session. Correlating the concatenation of
those z-scored ranks gives a frame-weighted mean of the per-session Spearman matrices
and never compares ranks across sessions, which would fold session-level offsets into
the correlation.

  python fc_vertexwise.py --space fsaverage5
  python fc_vertexwise.py --space fsaverage6 --sessions 1 2 3 --save-timeseries
"""
import os, sys, glob, json, time, shutil, argparse, subprocess, tempfile
import xml.etree.ElementTree as ET
import numpy as np
import nibabel as nib
from scipy.stats import rankdata

from paths import MSC, RESULTS, ANNOT, CACHE

DENSITY = {"fsaverage5": "10k", "fsaverage6": "41k"}
NVERT = {"fsaverage5": 10242, "fsaverage6": 40962}
FCDIR = os.path.join(RESULTS, "fc")


# ------------------------------------------------------------------ atlas files
def atlas_files(space):
    """Registration spheres, vertex areas and medial wall labels, fetched once."""
    from neuromaps.datasets import fetch_atlas, get_atlas_dir
    den = DENSITY[space]
    fetch_atlas("fsLR", "32k")
    fetch_atlas("fsaverage", den)
    src, trg = get_atlas_dir("fsLR"), get_atlas_dir("fsaverage")
    f = dict(
        src_sphere=f"{src}/tpl-fsLR_space-fsaverage_den-32k_hemi-L_sphere.surf.gii",
        trg_sphere=f"{trg}/tpl-fsaverage_den-{den}_hemi-L_sphere.surf.gii",
        src_area=f"{src}/tpl-fsLR_den-32k_hemi-L_desc-vaavg_midthickness.shape.gii",
        trg_area=f"{trg}/tpl-fsaverage_den-{den}_hemi-L_desc-vaavg_midthickness.shape.gii",
        src_mask=f"{src}/tpl-fsLR_den-32k_hemi-L_desc-nomedialwall_dparc.label.gii",
        trg_mask=f"{trg}/tpl-fsaverage_den-{den}_hemi-L_desc-nomedialwall_dparc.label.gii",
    )
    missing = [k for k, v in f.items() if not os.path.exists(v)]
    if missing:
        raise FileNotFoundError(f"neuromaps atlas files missing: {missing}")
    return f


def midthickness(space):
    """fsaverage midthickness, built once as the mean of white and pial. neuromaps ships
    both but not their average, and smoothing kernels should follow the mid-cortical
    sheet rather than either surface."""
    out = os.path.join(CACHE, f"tpl-fsaverage_den-{DENSITY[space]}_hemi-L_midthickness.surf.gii")
    if not os.path.exists(out):
        from neuromaps.datasets import get_atlas_dir
        d, den = get_atlas_dir("fsaverage"), DENSITY[space]
        w = nib.load(f"{d}/tpl-fsaverage_den-{den}_hemi-L_white.surf.gii")
        p = nib.load(f"{d}/tpl-fsaverage_den-{den}_hemi-L_pial.surf.gii")
        w.darrays[0].data = ((w.darrays[0].data + p.darrays[0].data) / 2).astype(np.float32)
        nib.save(w, out)
    return out


def smooth(Y, space, fwhm, keep, tmpdir):
    """Geodesic Gaussian smoothing on the target surface, restricted to `keep` so nothing
    is pulled across the medial wall. The MSC data already carries 6 mm FWHM from its own
    pipeline; kernels add in quadrature, so 5 mm here gives about 7.8 mm in total."""
    src = os.path.join(tmpdir, "pre.func.gii")
    roi = os.path.join(tmpdir, "roi.shape.gii")
    out = os.path.join(tmpdir, "post.func.gii")
    gii = nib.gifti.GiftiImage()
    for t in range(Y.shape[1]):
        gii.add_gifti_data_array(nib.gifti.GiftiDataArray(
            np.ascontiguousarray(Y[:, t], np.float32), intent="NIFTI_INTENT_NORMAL"))
    nib.save(gii, src)
    r = nib.gifti.GiftiImage()
    r.add_gifti_data_array(nib.gifti.GiftiDataArray(
        np.asarray(keep, np.float32), intent="NIFTI_INTENT_NORMAL"))
    nib.save(r, roi)

    cmd = ["wb_command", "-metric-smoothing", midthickness(space), src, str(fwhm), out,
           "-fwhm", "-roi", roi]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError(f"wb_command failed:\n{p.stderr}")
    S = np.stack([d.data for d in nib.load(out).darrays], axis=1).astype(np.float32)
    for f in (src, roi, out):
        os.remove(f)
    return S


def cortex_mask(space, kind="fsaverage"):
    """Vertices to keep. 'fsaverage' is FreeSurfer's medial wall (9354 / 37476),
    'glasser' is label > 0 in lh.HCP-MMP1.annot, which is what mesh_cache.load_cortex
    uses - pick that one if the FC rows have to line up with Cortex vertices."""
    if kind == "fsaverage":
        m = nib.load(atlas_files(space)["trg_mask"]).darrays[0].data > 0
    elif kind == "glasser":
        from surf_ops import load_glasser
        lab, _ = load_glasser(ANNOT, NVERT[space])
        m = lab > 0
    else:
        raise ValueError(kind)
    return np.asarray(m, bool)


# ------------------------------------------------------------------ MSC loading
def _cifti_xml(img):
    raw = img.header.extensions[0]._raw          # CIFTI-1: nibabel's parser refuses
    return raw.decode("utf-8", errors="replace").rstrip("\x00")


def load_left(path, structure="CIFTI_STRUCTURE_CORTEX_LEFT"):
    """-> (data (nUsed, T) float32, node indices into the 32k mesh, n_surf, TR, tmask).

    These files are CIFTI-1, where the vertex list is <NodeIndices>; CIFTI-2 renamed it
    <VertexIndices>. Both are read, because reading neither and falling back to
    arange(29696) silently mislabels every vertex."""
    img = nib.Nifti2Image.from_filename(path)
    arr = np.asarray(img.dataobj)[0, 0, 0, 0]
    root = ET.fromstring(_cifti_xml(img))

    tr = None
    for mat in root.iter("MatrixIndicesMap"):
        if mat.get("IndicesMapToDataType") == "CIFTI_INDEX_TYPE_TIME_POINTS":
            tr = float(mat.get("TimeStep"))
            if (mat.get("TimeStepUnits") or "").endswith("MSEC"):
                tr /= 1000.0

    sel = None
    for bm in root.iter("BrainModel"):
        if bm.get("BrainStructure") != structure:
            continue
        off, n = int(bm.get("IndexOffset")), int(bm.get("IndexCount"))
        nsurf = int(bm.get("SurfaceNumberOfNodes"))
        node = bm.find("NodeIndices")
        if node is None:
            node = bm.find("VertexIndices")
        if node is None:
            raise ValueError(f"{path}: no NodeIndices/VertexIndices for {structure}")
        idx = np.fromstring(node.text, sep=" ", dtype=np.int64)
        if len(idx) != n:
            raise ValueError(f"{path}: {len(idx)} node indices for {n} rows")
        sel = (arr[off:off + n], idx, nsurf)
    if sel is None:
        raise KeyError(structure)

    tm = path.replace("_rest.dtseries.nii", "_tmask.txt")
    tmask = np.loadtxt(tm).astype(bool) if os.path.exists(tm) else None
    return sel[0], sel[1], sel[2], tr, tmask


def session_paths(subject="sub-MSC01", which=None, data=MSC):
    ps = sorted(glob.glob(os.path.join(data, f"{subject}_ses-*_rest.dtseries.nii")))
    if which:
        want = {f"ses-func{int(w):02d}" for w in which}
        ps = [p for p in ps if any(w in os.path.basename(p) for w in want)]
        if len(ps) != len(want):
            raise FileNotFoundError(f"asked for {sorted(want)}, found {len(ps)} files")
    return ps


# ------------------------------------------------------------------ resampling
def resample(dense, space, files, tmpdir):
    """(32492, T) fs_LR 32k -> (nVert, T) fsaverage5/6, barycentric, area adjusted."""
    src = os.path.join(tmpdir, "in.func.gii")
    out = os.path.join(tmpdir, "out.func.gii")
    gii = nib.gifti.GiftiImage()
    for t in range(dense.shape[1]):
        gii.add_gifti_data_array(nib.gifti.GiftiDataArray(
            np.ascontiguousarray(dense[:, t], np.float32), intent="NIFTI_INTENT_NORMAL"))
    nib.save(gii, src)

    cmd = ["wb_command", "-metric-resample", src, files["src_sphere"],
           files["trg_sphere"], "ADAP_BARY_AREA", out,
           "-area-metrics", files["src_area"], files["trg_area"],
           "-current-roi", files["src_mask"]]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"wb_command failed:\n{r.stderr}")
    Y = np.stack([d.data for d in nib.load(out).darrays], axis=1).astype(np.float32)
    os.remove(src); os.remove(out)
    return Y


def rank_z(X):
    """Rank each row over time (ties averaged), then z-score. Pearson on these is
    Spearman; z-scoring per session keeps sessions from being compared to each other."""
    R = rankdata(X, axis=1).astype(np.float32)
    R -= R.mean(1, keepdims=True)
    sd = R.std(1, keepdims=True)
    bad = (sd == 0).ravel()
    sd[sd == 0] = 1.0
    R /= sd
    return R, bad


# ------------------------------------------------------------------ main
def build(space="fsaverage5", subject="sub-MSC01", which=None, mask_kind="fsaverage",
          smooth_fwhm=0.0, save_timeseries=False, dtype="float32", outdir=FCDIR,
          verbose=True):
    if shutil.which("wb_command") is None:
        sys.exit("wb_command not on PATH (Connectome Workbench is needed to resample)")
    files = atlas_files(space)
    keep = cortex_mask(space, mask_kind)
    paths = session_paths(subject, which)
    if not paths:
        sys.exit(f"no sessions found for {subject} in {MSC}")
    os.makedirs(outdir, exist_ok=True)

    if verbose:
        print(f"{subject}: {len(paths)} session(s) -> {space} "
              f"({keep.sum()} of {len(keep)} vertices, {mask_kind} mask)"
              + (f", +{smooth_fwhm:g} mm FWHM on top of the 6 mm the data already has"
                 if smooth_fwhm else ""))

    blocks, kept_frames, tr = [], [], None
    with tempfile.TemporaryDirectory() as tmp:
        for p in paths:
            t0 = time.time()
            X, nodes, nsurf, tr, tmask = load_left(p)
            if nsurf != 32492:
                raise ValueError(f"{p}: expected fs_LR 32k, got {nsurf} nodes")
            T = X.shape[1]
            if tmask is not None:
                X = X[:, tmask[:T]]
            dense = np.zeros((nsurf, X.shape[1]), np.float32)
            dense[nodes] = X                          # medial wall stays 0, masked below
            Y = resample(dense, space, files, tmp)
            if smooth_fwhm:
                Y = smooth(Y, space, smooth_fwhm, keep, tmp)
            Y = Y[keep]
            Z, bad = rank_z(Y)
            blocks.append(Z)
            kept_frames.append(Z.shape[1])
            if verbose:
                print(f"  {os.path.basename(p)[:28]}  {T:4d} frames -> {Z.shape[1]:4d} kept "
                      f"({100*Z.shape[1]/T:3.0f}%)  {Z.shape[0]} vertices"
                      f"{f'  [{bad.sum()} flat]' if bad.any() else ''}  {time.time()-t0:.0f}s")

    Z = np.concatenate(blocks, axis=1)
    del blocks
    nT = Z.shape[1]

    # a vertex with no variance in some session cannot be ranked there; drop it entirely
    var = (Z ** 2).sum(1)
    good = var > 0
    if not good.all():
        if verbose:
            print(f"  dropping {int((~good).sum())} vertices with no variance")
        Z = Z[good]
    vertices = np.flatnonzero(keep)[good]

    if verbose:
        print(f"  correlating {Z.shape[0]} vertices over {nT} frames "
              f"({nT*(tr or 0)/60:.0f} min) -> {Z.shape[0]**2*np.dtype(dtype).itemsize/1e9:.1f} GB")
    t0 = time.time()
    FC = (Z @ Z.T) / float(nT)
    np.clip(FC, -1.0, 1.0, out=FC)
    np.fill_diagonal(FC, 1.0)
    FC = FC.astype(dtype, copy=False)
    if verbose:
        print(f"  done in {time.time()-t0:.0f}s")

    ses = "-".join(os.path.basename(p).split("_")[1].replace("ses-func", "")
                   for p in paths)
    sm = f"_fwhm-{smooth_fwhm:g}" if smooth_fwhm else ""
    tag = f"{subject}_hemi-L_space-{space}_mask-{mask_kind}{sm}_ses-{ses}"
    fc_path = os.path.join(outdir, f"{tag}_spearmanfc.npy")
    np.save(fc_path, FC)
    np.save(os.path.join(outdir, f"{tag}_vertices.npy"), vertices)
    meta = dict(subject=subject, space=space, hemi="L", density=DENSITY[space],
                n_surface_vertices=NVERT[space], mask=mask_kind,
                smoothing_fwhm_added=smooth_fwhm,
                smoothing_fwhm_native=6.0,   # 2.55 mm sigma in the MSC pipeline
                n_vertices=int(Z.shape[0]), sessions=[os.path.basename(p) for p in paths],
                frames_kept=[int(k) for k in kept_frames], frames_total=int(nT), tr=tr,
                correlation="spearman (ranks within session, z-scored, frame-weighted mean)",
                resampling="wb_command -metric-resample ADAP_BARY_AREA, "
                           "neuromaps fsLR->fsaverage registration spheres",
                dtype=dtype, fc_file=os.path.basename(fc_path))
    with open(os.path.join(outdir, f"{tag}_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    if save_timeseries:
        ts = os.path.join(outdir, f"{tag}_rankz.npy")
        np.save(ts, Z)
        if verbose:
            print(f"  wrote {ts}  {Z.shape}")

    if verbose:
        print(f"  wrote {fc_path}  {FC.shape} {FC.dtype}")
        iu = np.triu_indices(min(2000, FC.shape[0]), 1)
        print(f"  off-diagonal (first {min(2000, FC.shape[0])} vertices): "
              f"mean {FC[:2000, :2000][iu].mean():+.3f}, "
              f"range {FC[:2000, :2000][iu].min():+.3f} to {FC[:2000, :2000][iu].max():+.3f}")
    return FC, vertices, meta


def align_to_cortex(FC, vertices, cortex):
    """Line an FC matrix up with model output on the same surface.

    Model frames are (nsteps, cortex.nV) and column i is full-mesh vertex cortex.old[i];
    FC row j is full-mesh vertex vertices[j]. The two vertex sets are not identical even
    with mask='glasser' (vertices at the medial wall edge get no data when resampling and
    are dropped), so intersect rather than assume.

    -> (FC restricted to the shared vertices, columns to take from frames, shared ids)
    Use as:  FCa, cols, ids = align_to_cortex(FC, vertices, cortex); H = frames[:, cols]
    """
    old = np.asarray(cortex.old)
    shared, i_fc, i_model = np.intersect1d(vertices, old, return_indices=True)
    return FC[np.ix_(i_fc, i_fc)], i_model, shared


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--space", default="fsaverage5", choices=sorted(DENSITY))
    ap.add_argument("--subject", default="sub-MSC01")
    ap.add_argument("--sessions", nargs="*", default=None,
                    help="session numbers, e.g. --sessions 1 2 3 (default: all)")
    ap.add_argument("--mask", default="fsaverage", choices=("fsaverage", "glasser"),
                    dest="mask_kind", help="which non-medial-wall definition to keep")
    ap.add_argument("--smooth-fwhm", type=float, default=0.0, dest="smooth_fwhm",
                    help="extra geodesic smoothing on the target surface, mm FWHM "
                         "(the data already carries ~6 mm from the MSC pipeline)")
    ap.add_argument("--dtype", default="float32", choices=("float32", "float16"))
    ap.add_argument("--save-timeseries", action="store_true",
                    help="also save the concatenated rank z-scored timeseries")
    ap.add_argument("--outdir", default=FCDIR)
    a = ap.parse_args()
    build(space=a.space, subject=a.subject, which=a.sessions, mask_kind=a.mask_kind,
          smooth_fwhm=a.smooth_fwhm, save_timeseries=a.save_timeseries, dtype=a.dtype,
          outdir=a.outdir)


if __name__ == "__main__":
    main()
