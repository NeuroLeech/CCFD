"""Read MSC subject-01 resting-state surface data.

ds000224 derivatives/surface_pipeline, fs_LR 32k, TR 2.2 s, 10 sessions x 30 min.

The files are CIFTI-1, which nibabel refuses (CIFTI-2 only). Underneath they are
plain NIfTI-2, so the array loads directly; the grayordinate -> vertex mapping is
recovered by parsing the XML extension by hand.

Cortex is stored without the medial wall: 29,696 of 32,492 left vertices carry data.
"""
import os, glob, re
import xml.etree.ElementTree as ET
import numpy as np
import nibabel as nib

from paths import MSC as DATA


def sessions(data=DATA):
    return sorted(glob.glob(os.path.join(data, "*_rest.dtseries.nii")))


def _cifti1_xml(img):
    raw = img.header.extensions[0]._raw          # bypass the CIFTI-2-only parser
    return raw.decode("utf-8", errors="replace").rstrip("\x00")


def load_session(path, structure="CIFTI_STRUCTURE_CORTEX_LEFT"):
    """-> dict with data (nVertexUsed, nFrames), vertex indices, surface size, TR, tmask."""
    img = nib.Nifti2Image.from_filename(path)
    arr = np.asarray(img.dataobj)[0, 0, 0, 0]            # (grayordinates, frames)
    root = ET.fromstring(_cifti1_xml(img))

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
        vi = bm.find("VertexIndices")
        verts = np.fromstring(vi.text, sep=" ", dtype=np.int64) if vi is not None \
            else np.arange(n)
        sel = dict(data=arr[off:off+n], verts=verts, n_surf=nsurf)
    if sel is None:
        raise KeyError(structure)

    tm = path.replace("_rest.dtseries.nii", "_tmask.txt")
    sel["tmask"] = np.loadtxt(tm).astype(bool) if os.path.exists(tm) else None
    sel["tr"] = tr
    sel["path"] = path
    return sel


def load_all(structure="CIFTI_STRUCTURE_CORTEX_LEFT", censor=True, data=DATA):
    """Concatenate every session, optionally dropping motion-censored frames."""
    chunks, verts, tr, nsurf = [], None, None, None
    for p in sessions(data):
        s = load_session(p, structure)
        d = s["data"]
        if censor and s["tmask"] is not None:
            d = d[:, s["tmask"][:d.shape[1]]]
        chunks.append(d)
        verts, tr, nsurf = s["verts"], s["tr"], s["n_surf"]
    return np.concatenate(chunks, axis=1), verts, nsurf, tr


if __name__ == "__main__":
    ss = sessions()
    print(f"{len(ss)} sessions in {DATA}")
    tot, keep = 0, 0
    for p in ss:
        s = load_session(p)
        n = s["data"].shape[1]
        k = int(s["tmask"].sum()) if s["tmask"] is not None else n
        tot += n; keep += k
        print(f"  {os.path.basename(p)[:32]}  {s['data'].shape[0]:5d} verts x {n:4d} frames"
              f"   kept {k:4d} ({100*k/n:3.0f}%)")
    s0 = load_session(ss[0])
    print(f"\nleft cortex: {s0['data'].shape[0]} vertices with data "
          f"of {s0['n_surf']} on the fs_LR 32k surface (medial wall excluded)")
    print(f"TR {s0['tr']} s")
    print(f"total {tot} frames, {keep} after censoring ({100*keep/tot:.0f}%) "
          f"= {keep*s0['tr']/60:.0f} minutes of usable rest")
    X, verts, nsurf, tr = load_all()
    print(f"concatenated: {X.shape} (vertices x frames), "
          f"{X.nbytes/1e9:.2f} GB as float32")
