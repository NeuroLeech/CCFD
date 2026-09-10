# Ascending Input Basis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a tractography-defined ascending-pathway input basis (nucleus masks → group-dMRI endpoint fields on fsaverage5 → per-field Laplace eigenmode channels) and plumb it into the existing convex solve, with driven-area coverage as a first-class control.

**Architecture:** A new plain-file module `fit/ascending.py` produces the (K, nV) profile matrix P and per-channel tags. P replaces the taper/gauss/shell profiles everywhere downstream via the existing `profiles=` argument of `xspec.impulse_responses`, so the solve, realisation, and scoring are untouched. The solver changes are diagnostic only (whitening stays off; `--solver factor` is the fallback).

**Tech Stack:** Python 3.13 (numpy, scipy, nibabel, templateflow, hcp_utils), DSI Studio CLI (HCP1021 group FIB), Connectome Workbench `wb_command` (already at `~/Downloads/workbench/bin_macosxub/wb_command`), pytest 9.

**Spec:** `docs/superpowers/specs/2026-09-10-ascending-input-design.md`

## Global Constraints

- The medium is pinned and must not change: `python fit/best_fit.py --oversample 4 --decay-s 25 --spread-mm-s 1.47 --bold-smooth --regions subcortical --split 40 --profile taper --bandpass 0.01,0.08 --impulse-decays 7 --pad 4096 --iters 400 --val-vert 0 --seconds 2308 --draws 2` (spread 1.47 mm/s, decay 25 s, `BEST_X` maps). Same for rest and task.
- No HRF. No whitening. `--iters 400` is the convention, not a convergence defect.
- The passband 0.01–0.08 Hz is a property of the measurement.
- The medium has no endogenous noise; all variability comes from the input.
- Coverage sweep limited to {6, 8, 10, 12, 14}% driven cortex; nothing above ~20% is in scope.
- Repo conventions: no package, no install; every runnable file imports `_path` first and runs as a plain script from the repo root (`python fit/ascending.py ...`). Results are keyed by `--tag`; never reuse a tag. Watch memory: scoring paths allocate multi-GB transients — never run them at full pool width; the BLAS-thread throttling in `xspec.parallel_scores` already handles the pool case.
- Do not overwrite existing results; `data/cache` regenerates on demand.
- HCP1021 FIB cohort is accepted (not the on-disk NKI DWI) for this round.

## File Structure

- Create: `fit/ascending.py` — masks, tractography fields, modal channels, coverage.
- Create: `data/atlases/ascending_manifest.json` — nucleus name → source file / label / role.
- Create: `tests/conftest.py`, `tests/test_ascending.py` — pytest path setup and unit tests.
- Create: `analysis/ascending_distance.py` — score binned by geodesic distance from the driven set.
- Create: `analysis/ascending_readout.py` — per-nucleus power / coordination / mode content + family brackets.
- Modify: `fit/xspec.py` — `profile_tag()` helper + hashed profile key in `impulse_responses`.
- Modify: `fit/best_fit.py` — `--regions ascending` + `--ascending-*` flags; save `profiles` in the npz.
- Modify: `analysis/onoff_fields.py`, `analysis/interference.py`, `analysis/zones.py` — read `z["profiles"]` when present; `zones.group_index` handles `nucleus:mode` tags.

---

### Task 1: Acquire and verify the external data

**Files:**
- Create: `data/atlases/` (downloaded atlas files, untracked)
- Create: `data/hcp1021.fib.gz` (downloaded template, untracked)
- Create: `data/atlases/ascending_manifest.json`

**Interfaces:**
- Consumes: nothing (pure data task).
- Produces: `data/atlases/ascending_manifest.json` with variant `ascending_v1`; `FIB_PATH = data/hcp1021.fib.gz`; `data/atlases/fib_source.txt` recording the resolved download URL. Task 2's loader and Task 3's tracker read these.

- [ ] **Step 1: Install the toolchain**

```bash
python3 -m pip install templateflow
python3 -c "import templateflow.api as tf; print(tf.get('MNI152NLin6Asym', resolution='01', suffix='T1w'))"
```

Expected: templateflow installs and prints an existing `MNI152NLin6Asym_res-01_T1w.nii.gz` path (it downloads on first use; ~50 MB).

Download DSI Studio for macOS from https://github.com/frankyeh/DSI-Studio/releases (latest release zip), unzip, move `DSI Studio.app` to `/Applications`. Verify:

```bash
"/Applications/DSI Studio.app/Contents/MacOS/dsi_studio" --version
```

Expected: prints a version line. If the binary path differs, export `DSI_STUDIO` to it for this session.

- [ ] **Step 2: Download the group dMRI template**

Fetch https://brain.labsolver.org/hcp_template.html and extract the download link for the **HCP1021 1 mm template FIB** (prefer 1 mm; if the link is dead or > 12 GB, take the 2 mm QA variant). Download with `curl -L -o data/hcp1021.fib.gz "<resolved-url>"` (use the link found on the page, not a guess). Verify:

```bash
ls -l data/hcp1021.fib.gz
python3 - <<'EOF'
import nibabel as nib
img = nib.load("data/hcp1021.fib.gz")
print("shape", img.shape[:3], "affine diag", img.affine[0,0], img.affine[1,1], img.affine[2,2])
EOF
```

Expected: file ≥ 1 GB; prints a 3-D shape and unit diagonals (±1 mm). If nibabel cannot read the FIB header, note it — the module falls back to the ICBM152 1 mm constants, but record the actual grid here:

```bash
python3 - <<'EOF'
import nibabel as nib
img = nib.load("data/hcp1021.fib.gz")
print(repr(img.shape), repr(img.affine.tolist()))
EOF
```

Write the resolved URL to `data/atlases/fib_source.txt`.

- [ ] **Step 3: Download the nucleus atlases**

Thalamus (Morel, MNI space): clone or download the zip of https://github.com/thalamicseg/atlas into `data/atlases/thalamicseg/`.

Brainstem ascending arousal network (Harvard AAN, 1 mm MNI): download the atlas from https://www.nmr.mgh.harvard.edu/resources/aan-atlas into `data/atlases/aan/`. If the page is down, search for the mirror of the "Harvard Ascending Arousal Network Atlas" and record the source.

Amygdala: fetch via templateflow:

```bash
python3 - <<'EOF'
import templateflow.api as tf, shutil, os
src = tf.get("MNI152NLin6Asym", resolution="01", atlas="HarvardOxford", desc="sub", suffix="probseg")
os.makedirs("data/atlases/harvardoxford", exist_ok=True)
shutil.copy(src, "data/atlases/harvardoxford/HarvardOxford-sub-prob-1mm.nii.gz")
print("copied", src)
EOF
```

Expected: each command leaves files under `data/atlases/`; no missing-directory errors.

- [ ] **Step 4: Write the manifest with verified labels**

List what each atlas actually contains before writing the manifest:

```bash
python3 - <<'EOF'
import nibabel as nib, numpy as np, glob, os
for pat in ("data/atlases/aan/*.nii*", "data/atlases/thalamicseg/**/*.nii*", "data/atlases/thalamicseg/**/*.nii.gz"):
    for f in glob.glob(pat, recursive=True):
        d = np.asarray(nib.load(f).dataobj)
        labs = np.unique(d)
        print(os.path.relpath(f, "data/atlases"), "shape", d.shape, "labels:", labs[:20].tolist())
EOF
```

Read the README/LUT files inside `thalamicseg/` to map Morel label values to nucleus names (the Morel atlas is a single labelled volume; the AAN atlas ships one probability file per nucleus).

Write `data/atlases/ascending_manifest.json` with this exact structure (fill `file` paths and `label` values from the output above; `label: null` means the file is already a probability map for one nucleus):

```json
{
  "ascending_v1": [
    {"name": "LGN",  "file": "thalamicseg/Morel_MNI/Morel_MNI_2mm.nii.gz", "label": 1, "role": "visual"},
    {"name": "MGN",  "file": "thalamicseg/Morel_MNI/Morel_MNI_2mm.nii.gz", "label": 2, "role": "auditory"},
    {"name": "LC",   "file": "aan/LC.nii.gz", "label": null, "role": "arousal"},
    {"name": "VTA",  "file": "aan/VTA.nii.gz", "label": null, "role": "arousal"},
    {"name": "DRN",  "file": "aan/DRN.nii.gz", "label": null, "role": "arousal"},
    {"name": "MRN",  "file": "aan/MRN.nii.gz", "label": null, "role": "arousal"},
    {"name": "PPN",  "file": "aan/PPN.nii.gz", "label": null, "role": "arousal"},
    {"name": "LDT",  "file": "aan/LDT.nii.gz", "label": null, "role": "arousal"},
    {"name": "PB",   "file": "aan/PB.nii.gz", "label": null, "role": "arousal"},
    {"name": "amygdala", "file": "harvardoxford/HarvardOxford-sub-prob-1mm.nii.gz", "label": 17, "role": "limbic"}
  ]
}
```

Rules: use the actual filenames found on disk; the two Morel rows must have the label values the atlas LUT assigns to LGN and MGN; add rows for any further nuclei the AAN zip ships (e.g. `BF`, `TMN`) with `label: null` and the role from the atlas README; omit nuclei whose file is absent — the manifest must only reference files that exist. Validate:

```bash
python3 - <<'EOF'
import json, os
man = json.load(open("data/atlases/ascending_manifest.json"))["ascending_v1"]
names = [e["name"] for e in man]
assert len(names) == len(set(names)), "duplicate names"
for e in man:
    p = os.path.join("data/atlases", e["file"])
    assert os.path.exists(p), f"missing {p}"
print("manifest ok:", len(man), "nuclei:", ", ".join(names))
EOF
```

Expected: `manifest ok: <N> nuclei: ...` listing every nucleus.

- [ ] **Step 5: Commit**

```bash
git add data/atlases/ascending_manifest.json
git commit -m "Add the ascending-nucleus atlas manifest"
```

(Only the manifest JSON is tracked; the downloaded volumes are data and stay untracked via the existing `data/cache`-style ignores — add `data/atlases/*.nii*`, `data/atlases/*.gz`, `data/*.fib.gz` to `.gitignore` first if `git status` lists them.)

---

### Task 2: `fit/ascending.py` — masks and the module skeleton

**Files:**
- Create: `fit/ascending.py`
- Create: `tests/conftest.py`
- Create: `tests/test_ascending.py`

**Interfaces:**
- Consumes: `data/atlases/ascending_manifest.json` (Task 1).
- Produces: `ascending.load_manifest(variant="ascending_v1") -> list[dict]`; `ascending.nucleus_masks(entries, fib=FIB_PATH, verbose=True) -> (names: list[str], masks: np.ndarray (M, nx, ny, nz) float32)`; `ascending._grid(fib=FIB_PATH) -> (shape, affine)`; module constants `ATLAS_DIR`, `FIB_PATH`, `MANIFEST`, `WB_COMMAND`, `DSI_STUDIO`. Tasks 3–6 import these.

- [ ] **Step 1: Write the failing test**

`tests/conftest.py`:

```python
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for d in ("core", "fit", "targets", "analysis", "viz"):
    sys.path.insert(0, os.path.join(ROOT, d))
```

`tests/test_ascending.py` (Task 2 portion):

```python
import os
import numpy as np
import pytest

from paths import DATA
import ascending


@pytest.mark.skipif(not os.path.exists(ascending.MANIFEST), reason="atlas manifest missing")
def test_manifest_entries_exist():
    entries = ascending.load_manifest()
    assert entries, "empty manifest"
    names = [e["name"] for e in entries]
    assert len(names) == len(set(names)), "duplicate nucleus names"


@pytest.mark.skipif(not os.path.exists(ascending.FIB_PATH), reason="FIB missing")
def test_masks_load_on_fib_grid():
    entries = ascending.load_manifest()
    names, masks = ascending.nucleus_masks(entries, verbose=False)
    assert len(names) == len(entries)
    assert masks.shape[0] == len(entries)
    assert masks.ndim == 4
    for nm, m in zip(names, masks):
        assert m.max() > 0, f"{nm} mask is empty after resampling"
        assert m.min() >= -1e-6, f"{nm} mask has negative values"
        assert m.max() <= 1.0 + 1e-6, f"{nm} mask exceeds 1"


def test_grid_fallback_shape():
    shape, affine = ascending._grid(os.path.join(DATA, "nonexistent.fib.gz"))
    assert shape == (182, 218, 182)
    assert affine.shape == (4, 4)
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
python3 -m pytest tests/test_ascending.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'ascending'`.

- [ ] **Step 3: Write `fit/ascending.py` (masks + skeleton)**

```python
"""Ascending-pathway input basis: nucleus masks -> tractography fields -> modal channels.

The input regions are defined by the ascending systems into cortex rather than by the
Glasser atlas: each nucleus mask (thalamic nuclei, brainstem arousal nuclei, basal
forebrain, hypothalamus, amygdala) is tracked through a group dMRI template in MNI space
and the streamline endpoints become a graded projection field on fsaverage5. Each field is
then decomposed into weighted-Laplacian eigenmodes on its support; channels are
(nucleus, mode) pairs whose amplitudes and phases the cross-spectrum solve sets.

  python fit/ascending.py --fields                    # run tracking + mapping (once)
  python fit/ascending.py --channels --tau 0.2        # show the channel basis
"""
import _path  # noqa: F401  - puts the sibling code folders on sys.path
import os, json, hashlib, subprocess, tempfile, argparse
import numpy as np

from paths import DATA, CACHE

ATLAS_DIR = os.path.join(DATA, "atlases")
FIB_PATH = os.path.join(DATA, "hcp1021.fib.gz")
MANIFEST = os.path.join(ATLAS_DIR, "ascending_manifest.json")

WB_COMMAND = os.environ.get(
    "WB_COMMAND", os.path.expanduser("~/Downloads/workbench/bin_macosxub/wb_command"))
DSI_STUDIO = os.environ.get(
    "DSI_STUDIO", "/Applications/DSI Studio.app/Contents/MacOS/dsi_studio")

# ICBM152 1 mm constants, used only when the FIB header cannot be read
_FIB_FALLBACK = ((182, 218, 182),
                 np.array([[-1, 0, 0, 90], [0, 1, 0, -126], [0, 0, 1, -72],
                           [0, 0, 0, 1]], float))


def load_manifest(variant="ascending_v1"):
    with open(MANIFEST) as f:
        return json.load(f)[variant]


def _grid(fib=FIB_PATH):
    """-> (shape, affine) of the FIB volume. The FIB header is NIFTI-compatible."""
    import nibabel as nib
    try:
        img = nib.load(fib)
        return img.shape[:3], np.asarray(img.affine, float)
    except Exception:
        return _FIB_FALLBACK


def nucleus_masks(entries, fib=FIB_PATH, verbose=True):
    """-> (names, masks): each mask resampled onto the FIB grid, normalised to max 1.

    `label` entries are one-hot from a labelled volume (nearest neighbour); `label: null`
    entries are already probability maps (linear interpolation)."""
    import nibabel as nib
    from scipy.ndimage import affine_transform
    shape, aff = _grid(fib)
    names, out = [], []
    for e in entries:
        img = nib.load(os.path.join(ATLAS_DIR, e["file"]))
        d = np.asarray(img.dataobj).astype(np.float32)
        if e.get("label") is not None:
            m = (d == int(e["label"])).astype(np.float32)
            order = 0
        else:
            m = d
            order = 1
        T = np.linalg.inv(aff) @ np.asarray(img.affine, float)
        m = affine_transform(m, T, output_shape=shape, order=order)
        mx = float(m.max())
        if mx > 0:
            m = m / mx
        names.append(e["name"])
        out.append(m.astype(np.float32))
        if verbose:
            print(f"  {e['name']:<12s} mass {m.sum():.0f} voxels on the FIB grid")
    return names, np.asarray(out)


def wb(*args):
    """Run wb_command; raises on failure."""
    subprocess.run([WB_COMMAND] + [str(a) for a in args], check=True,
                   capture_output=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fields", action="store_true",
                    help="track every nucleus through the FIB and map endpoints to "
                         "fsaverage5 (cached)")
    ap.add_argument("--channels", action="store_true",
                    help="print the modal channel basis for the cached fields")
    ap.add_argument("--seed-count", type=int, default=20000, dest="seed_count")
    ap.add_argument("--tau", type=float, default=0.2)
    ap.add_argument("--budget", type=int, default=80)
    ap.add_argument("--coverage", type=float, default=0.0,
                    help="pick tau for this driven-area fraction (overrides --tau)")
    a = ap.parse_args()

    entries = load_manifest()
    from mesh_cache import load_cortex
    c = load_cortex("fsaverage5", verbose=False)
    if a.fields:
        names, fields = projection_fields(entries, seed_count=a.seed_count)
        print(f"  {len(names)} projection fields on {fields.shape[1]} vertices")
        return
    if a.channels:
        names, fields = load_fields(c)
        tau = a.tau
        if a.coverage > 0:
            tau = tau_for_coverage(c, names, fields, a.coverage)
        P, tags, frac, meta = modal_channels(c, names, fields, tau, a.budget)
        print(f"  tau {tau:g}: {frac[0]:.2%} of cortex ({frac[1]:.0f} mm2), "
              f"{len(tags)} channels")
        for nm, k in meta["k_n"].items():
            print(f"    {nm:<12s} {k} modes")
        return
    ap.print_help()


if __name__ == "__main__":
    main()
```

Note: `projection_fields`, `load_fields`, `tau_for_coverage`, `modal_channels` are defined in Tasks 3–4; until then the `main()` references fail only when the flags are used — the Task 2 tests do not exercise them.

- [ ] **Step 4: Run the tests to verify they pass**

```bash
python3 -m pytest tests/test_ascending.py -v
```

Expected: PASS (skips are allowed if the manifest/FIB are absent; on this machine they were created in Task 1, so the non-skip tests must pass).

- [ ] **Step 5: Commit**

```bash
git add fit/ascending.py tests/conftest.py tests/test_ascending.py
git commit -m "Add ascending input module skeleton and mask loading"
```

---

### Task 3: Projection fields — tracking, endpoints, surface mapping

**Files:**
- Modify: `fit/ascending.py`
- Modify: `tests/test_ascending.py`

**Interfaces:**
- Consumes: `nucleus_masks`, `_grid`, `wb`, `DSI_STUDIO`, `FIB_PATH` (Task 2); `load_cortex("fsaverage5")` gives `c.old` (submesh → full-fsaverage5 index) and `c.lab` (Glasser ids).
- Produces: `ascending.projection_fields(entries, seed_count=20000, cache=True, verbose=True) -> (names, fields)` with `fields` shape (M, c.nV) float32; `ascending.load_fields(c, verbose=True) -> (names, fields)` (cache loader used by Task 4+); `ascending._endpoint_density(trk_path, fib=FIB_PATH) -> volume`; `ascending._to_surface(name, volume, tmpdir) -> np.ndarray (10242,)`; cache file `data/cache/ascending_fields_<hash12>_<seed_count>.npz` with keys `names`, `fields`.

- [ ] **Step 1: Write the failing test (surface mapping only, no tracking needed)**

Add to `tests/test_ascending.py`:

```python
def test_endpoint_density_rasterizes_endpoints():
    shape, affine = ascending._grid()
    trk_path = os.path.join(DATA, "cache", "_test_endpoints.trk.gz")
    import nibabel as nib
    vox = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])
    ras = nib.affines.apply_affine(affine, vox)
    streamlines = [np.stack([ras[0], (ras[0] + ras[1]) / 2]), np.stack([ras[1]])]
    tfile = nib.streamlines.Tractogram(
        streamlines, affine_to_rasmm=np.eye(4))
    nib.streamlines.save(tfile, trk_path)
    V = ascending._endpoint_density(trk_path)
    assert V[tuple(vox[0].astype(int))] == 2
    assert V[tuple(vox[1].astype(int))] == 1
    assert float(V.sum()) == 3
    os.remove(trk_path)
```

Expected failure: `AttributeError: module 'ascending' has no attribute '_endpoint_density'`.

- [ ] **Step 2: Implement tracking, endpoint density, and surface mapping**

Append to `fit/ascending.py` (before `main()`):

```python
def _run_dsi(seed_nii, out_trk, seed_count, fib=FIB_PATH, threads=8):
    """Deterministic tracking seeded from one nucleus mask through the group FIB."""
    if not os.path.exists(DSI_STUDIO):
        raise SystemExit(f"  dsi_studio not found at {DSI_STUDIO}; export DSI_STUDIO")
    cmd = [DSI_STUDIO, "--action=trk", f"--source={fib}", f"--seed={seed_nii}",
           f"--seed_count={seed_count}", "--method=0", "--turn_angle=45",
           "--min_length=10", "--max_length=300", f"--thread_count={threads}",
           f"--output={out_trk}"]
    subprocess.run(cmd, check=True, capture_output=True)
    return out_trk


def _endpoint_density(trk_path, fib=FIB_PATH):
    """Rasterise streamline ENDPOINTS onto the FIB grid. Both ends count."""
    import nibabel as nib
    trk = nib.streamlines.load(trk_path)
    shape, aff = _grid(fib)
    inv = np.linalg.inv(aff)
    pts = [p for s in trk.streamlines for p in (s[0], s[-1])]
    xyz = np.round(nib.affines.apply_affine(inv, np.asarray(pts))).astype(np.int64)
    ok = ((xyz[:, 0] >= 0) & (xyz[:, 0] < shape[0]) &
          (xyz[:, 1] >= 0) & (xyz[:, 1] < shape[1]) &
          (xyz[:, 2] >= 0) & (xyz[:, 2] < shape[2]))
    xyz = xyz[ok]
    V = np.zeros(shape, np.float32)
    if len(xyz):
        np.add.at(V, (xyz[:, 0], xyz[:, 1], xyz[:, 2]), 1.0)
    return V


def _to_surface(name, V, tmpdir):
    """Endpoint-density volume -> metric on the full fsaverage5 left (10242 vertices).

    The density volume lives on the FIB (ICBM152 1 mm) grid; it is resampled to
    MNI152NLin6Asym and ribbon-constrained-mapped with templateflow's fsaverage5
    surfaces, which live in that space. If templateflow lacks fsaverage5 surfaces the
    mapping runs at 164k and is resampled with the standard FreeSurfer spheres."""
    import nibabel as nib
    import templateflow.api as tf
    dens = os.path.join(tmpdir, f"{name}_density.nii.gz")
    nib.save(nib.Nifti1Image(V, _grid()[1]), dens)
    mni = tf.get("MNI152NLin6Asym", resolution="01", suffix="T1w")
    res = os.path.join(tmpdir, f"{name}_mni.nii.gz")
    wb("-volume-resample", dens, mni, "CUBIC", res)
    try:
        mid = tf.get("MNI152NLin6Asym", density="fsaverage5", hemi="L",
                     suffix="midthickness")
        white = tf.get("MNI152NLin6Asym", density="fsaverage5", hemi="L",
                       suffix="white")
        pial = tf.get("MNI152NLin6Asym", density="fsaverage5", hemi="L", suffix="pial")
        out = os.path.join(tmpdir, f"{name}.func.gii")
        wb("-volume-to-surface-mapping", res, mid, out, "-ribbon-constrained",
           white, pial)
        return np.asarray(nib.load(out).darrays[0].data, np.float32)
    except Exception:
        # 164k mapping + standard-sphere resample to fsaverage5
        import nilearn.datasets as nd
        mid = tf.get("MNI152NLin6Asym", density="fsaverage", hemi="L",
                     suffix="midthickness")
        white = tf.get("MNI152NLin6Asym", density="fsaverage", hemi="L", suffix="white")
        pial = tf.get("MNI152NLin6Asym", density="fsaverage", hemi="L", suffix="pial")
        out164 = os.path.join(tmpdir, f"{name}_164k.func.gii")
        wb("-volume-to-surface-mapping", res, mid, out164, "-ribbon-constrained",
           white, pial)
        sph164 = nd.fetch_surf_fsaverage("fsaverage")["sphere_left"]
        sph5 = nd.fetch_surf_fsaverage("fsaverage5")["sphere_left"]
        out = os.path.join(tmpdir, f"{name}.func.gii")
        wb("-metric-resample", out164, sph164, sph5, "ADAP_BARY_AREA", out)
        return np.asarray(nib.load(out).darrays[0].data, np.float32)


def projection_fields(entries, seed_count=20000, cache=True, verbose=True):
    """-> (names, fields): endpoint-density fields on the cortex submesh (c.nV)."""
    from mesh_cache import load_cortex
    import nibabel as nib
    c = load_cortex("fsaverage5", verbose=False)
    key = (hashlib.sha1(
        json.dumps([e["name"] for e in entries]).encode()).hexdigest()[:12])
    cache_f = os.path.join(CACHE, f"ascending_fields_{key}_{seed_count}.npz")
    if cache and os.path.exists(cache_f):
        z = np.load(cache_f, allow_pickle=True)
        if verbose:
            print(f"  loaded {cache_f}")
        return list(z["names"]), np.asarray(z["fields"])
    names, masks = nucleus_masks(entries, verbose=verbose)
    fields = []
    for nm, mk in zip(names, masks):
        with tempfile.TemporaryDirectory() as td:
            seed = os.path.join(td, f"{nm}_seed.nii.gz")
            trk = os.path.join(td, f"{nm}.trk.gz")
            nib.save(nib.Nifti1Image(mk, _grid()[1]), seed)
            _run_dsi(seed, trk, seed_count)
            V = _endpoint_density(trk)
            f_full = _to_surface(nm, V, td)
            fields.append(f_full[c.old].astype(np.float32))
            if verbose:
                print(f"  {nm:<12s} {int((f_full > 0).sum())} vertices reached, "
                      f"field sum {f_full.sum():.0f}")
    F = np.stack(fields)
    np.savez(cache_f, names=np.array(names, dtype=object), fields=F)
    if verbose:
        print(f"  wrote {cache_f}")
    return names, F


def load_fields(c, verbose=True):
    """Cached fields for the manifest, on the given cortex. -> (names, fields)."""
    entries = load_manifest()
    names, F = projection_fields(entries, verbose=verbose)
    return names, np.asarray(F[:, :c.nV], np.float32)
```

- [ ] **Step 3: Add the anatomy test (runs once fields exist)**

Add to `tests/test_ascending.py`:

```python
@pytest.mark.skipif(not os.path.exists(ascending.FIB_PATH), reason="FIB missing")
def test_fields_anatomy_lgn_is_visual():
    from mesh_cache import load_cortex
    c = load_cortex("fsaverage5", verbose=False)
    names, fields = ascending.load_fields(c, verbose=False)
    i = names.index("LGN")
    f = fields[i]
    assert f.max() > 0, "LGN field is empty"
    v1 = np.flatnonzero(c.lab == 1)          # V1 in the Glasser annotation
    assert v1.size > 0
    assert f[v1].mean() > 2.0 * f.mean(), \
        f"LGN field does not concentrate on V1 ({f[v1].mean():.3f} vs {f.mean():.3f})"
    for nm, g in zip(names, fields):
        assert g.max() > 0, f"{nm} field is empty"
```

- [ ] **Step 4: Run the tracking pipeline once**

```bash
python fit/ascending.py --fields --seed-count 20000
```

Expected: one DSI Studio invocation per nucleus (several minutes each on 16 cores), then a summary line per nucleus and `wrote data/cache/ascending_fields_<hash>_20000.npz`. If any nucleus produces an empty field (0 vertices), rerun that nucleus with `--seed-count 40000` and note it; an empty field is not acceptable for LGN/MGN (the visual/auditory anchors).

- [ ] **Step 5: Run the tests**

```bash
python3 -m pytest tests/test_ascending.py -v
```

Expected: all PASS (including `test_fields_anatomy_lgn_is_visual`).

- [ ] **Step 6: Commit**

```bash
git add fit/ascending.py tests/test_ascending.py
git commit -m "Track nuclei through the group FIB and map endpoints to fsaverage5"
```

---

### Task 4: Modal channels and coverage control

**Files:**
- Modify: `fit/ascending.py`
- Modify: `tests/test_ascending.py`

**Interfaces:**
- Consumes: `load_fields` (Task 3); `c.edges`, `c.A`, `c.nV` from `load_cortex`.
- Produces: `ascending.support(c, f, tau) -> np.ndarray` (vertices of the largest connected component of `f > tau`); `ascending._field_modes(c, f, verts, k) -> np.ndarray (n, k)`; `ascending.modal_channels(c, names, fields, tau, budget=80, k_max=4, verbose=True) -> (P (K, nV) float32, tags list[str] of "nucleus:j", frac (coverage_fraction, mm2), meta dict with k_n)`; `ascending.coverage(c, fields, tau) -> (fraction, mm2)`; `ascending.tau_for_coverage(c, names, fields, frac) -> float`. Task 6 (best_fit) consumes `modal_channels`, `coverage`, `tau_for_coverage`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_ascending.py`:

```python
def _bfs_dist(c, seed):
    import scipy.sparse as sp
    from scipy.sparse.csgraph import dijkstra
    E = c.edges
    A = sp.coo_matrix((np.ones(len(E)), (E[:, 0], E[:, 1])),
                      shape=(c.nV, c.nV))
    return dijkstra((A + A.T).tocsr(), indices=[seed])[0]


def _synthetic_fields(c, n=3, scale=8.0, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for s in rng.choice(c.nV, n, replace=False):
        d = _bfs_dist(c, s)
        out.append(np.exp(-0.5 * (d / scale) ** 2))
    return np.stack(out)


def _cortex():
    from mesh_cache import load_cortex
    return load_cortex("fsaverage5", verbose=False)


def test_mode0_is_plain_field():
    c = _cortex()
    fields = _synthetic_fields(c)
    names = [f"nuc{i}" for i in range(len(fields))]
    P, tags, frac, meta = ascending.modal_channels(c, names, fields, tau=0.3,
                                                   budget=12, verbose=False)
    for nm in names:
        rows = [k for k, t in enumerate(tags) if t.startswith(nm + ":")]
        i = names.index(nm)
        f = fields[i]
        on = f > 0.3
        r = np.corrcoef(P[rows[0]][on], f[on])[0, 1]
        assert r > 0.95, f"{nm} mode 0 does not match the plain field (r={r:.3f})"


def test_field_modes_orthonormal():
    c = _cortex()
    f = _synthetic_fields(c, n=1)[0]
    verts = ascending.support(c, f, 0.3)
    V = ascending._field_modes(c, f, verts, 3)
    err = np.abs(V.T @ V - np.eye(V.shape[1])).max()
    assert err < 1e-8, f"modes not orthonormal ({err:.2e})"


def test_budget_and_tags():
    c = _cortex()
    fields = _synthetic_fields(c, n=4)
    names = [f"nuc{i}" for i in range(len(fields))]
    P, tags, frac, meta = ascending.modal_channels(c, names, fields, tau=0.3,
                                                   budget=12, verbose=False)
    K = len(tags)
    assert K <= 12 + len(names) * 4, f"channel budget blown: {K}"
    assert K >= 3, "too few channels"
    for t in tags:
        nm, j = t.split(":")
        assert nm in names and j.isdigit()
    assert P.shape == (K, c.nV)
    assert P.dtype == np.float32
    assert np.isfinite(P).all()
    assert ((np.abs(P) @ c.A) > 0).all(), "a channel has zero area-weighted mass"
    # higher modes are signed, so only mode-0 rows are non-negative
    for nm in names:
        rows = [k for k, t in enumerate(tags) if t == f"{nm}:0"]
        assert rows and (P[rows[0]] >= -1e-6).all(), f"{nm} mode 0 has negative lobes"
        assert meta["k_n"][nm] <= 4


def test_coverage_monotonic_and_tau_bisection():
    c = _cortex()
    fields = _synthetic_fields(c, n=8, scale=40.0)
    names = [f"nuc{i}" for i in range(len(fields))]
    c1 = ascending.coverage(c, fields, 0.1)[0]
    c2 = ascending.coverage(c, fields, 0.5)[0]
    assert c1 >= c2, "coverage must shrink as tau rises"
    maxcov = ascending.coverage(c, fields, 0.001)[0]
    assert maxcov > 0.15, f"fields too small for the bisection test ({maxcov:.3f})"
    tau = ascending.tau_for_coverage(c, names, fields, 0.5 * maxcov)
    got = ascending.coverage(c, fields, tau)[0]
    assert abs(got - 0.5 * maxcov) < 0.005, f"tau_for_coverage hit {got:.3f}, " \
        f"wanted {0.5 * maxcov:.3f}"
```

Expected failure: `module 'ascending' has no attribute 'modal_channels'` (or `support`).

- [ ] **Step 2: Implement support, modes, coverage, channels**

Append to `fit/ascending.py` (before `main()`):

```python
def support(c, f, tau):
    """Vertices of the largest connected component of f > tau, in mesh-adjacency order."""
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components
    m = np.asarray(f) > tau
    if m.sum() == 0:
        return np.zeros(0, np.int64)
    pos = -np.ones(c.nV, np.int64)
    pos[m] = np.arange(m.sum())
    E = c.edges
    keep = (pos[E[:, 0]] >= 0) & (pos[E[:, 1]] >= 0)
    n, lab = connected_components(
        sp.coo_matrix((np.ones(keep.sum()), (pos[E[keep, 0]], pos[E[keep, 1]])),
                      shape=(m.sum(),) * 2), directed=False)
    big = np.argmax(np.bincount(lab))
    return np.flatnonzero(m)[lab == big]


def _field_modes(c, f, verts, k):
    """The k smallest eigenmodes of the graph Laplacian on `verts`.

    Edges are weighted by 1 - 0.5*(f_i + f_j), floored at 1e-3 so the graph cannot
    disconnect; the periphery (low f) is therefore cheap to cut. Mode 0 is near-constant
    on the support, so channel (nucleus, 0) is approximately the plain projection field."""
    import scipy.sparse as sp
    from scipy.sparse.linalg import eigsh
    pos = -np.ones(c.nV, np.int64)
    pos[verts] = np.arange(len(verts))
    E = c.edges
    keep = (pos[E[:, 0]] >= 0) & (pos[E[:, 1]] >= 0)
    a, b = pos[E[keep, 0]], pos[E[keep, 1]]
    wgt = np.maximum(1.0 - 0.5 * (np.asarray(f)[E[keep, 0]] + np.asarray(f)[E[keep, 1]]),
                     1e-3)
    A = sp.coo_matrix((wgt, (a, b)), shape=(len(verts),) * 2)
    A = A + A.T
    L = sp.diags(np.asarray(A.sum(1)).ravel()) - A
    _, V = eigsh(L.astype(float), k=min(k, len(verts)), sigma=-1e-6, which="LM")
    return V


def coverage(c, fields, tau):
    """-> (fraction, mm2) of cortex where max_n f_n > tau."""
    m = np.max(np.stack([np.asarray(f) for f in fields]), axis=0)
    on = m > tau
    area = np.asarray(c.A, float)
    return float(area[on].sum() / area.sum()), float(area[on].sum())


def tau_for_coverage(c, names, fields, frac):
    """The smallest tau whose coverage is at most `frac` (bisection over densities)."""
    m = np.max(np.stack([np.asarray(f) for f in fields]), axis=0)
    area = np.asarray(c.A, float)
    tot = float(area.sum())
    vals = np.sort(np.unique(m[m > 0]))
    lo, hi = 0, len(vals) - 1

    def cov(i):
        return float(area[m > vals[i]].sum() / tot)

    while lo < hi:
        mid = (lo + hi) // 2
        if cov(mid) <= frac:
            hi = mid
        else:
            lo = mid + 1
    return float(vals[lo])


def modal_channels(c, names, fields, tau, budget=80, k_max=4, verbose=True):
    """-> (P, tags, (coverage_frac, coverage_mm2), meta).

    k_n = clamp(round(area_n / target), 1, k_max) with target = total support area /
    budget - the same area-proportional logic split_parcels uses. Profiles are
    envelope x mode: P[nuc, j] = f_n * psi_j on the support, 0 elsewhere."""
    area = np.asarray(c.A, float)
    sups = [support(c, f, tau) for f in fields]
    areas = np.array([area[v].sum() for v in sups])
    target = float(areas.sum() / float(budget))
    P, tags, k_n = [], [], {}
    for i, (nm, f, v) in enumerate(zip(names, fields, sups)):
        if len(v) == 0:
            k_n[nm] = 0
            continue
        k = int(np.clip(round(areas[i] / target), 1, k_max))
        V = _field_modes(c, f, v, k)
        k_n[nm] = V.shape[1]
        for j in range(V.shape[1]):
            row = np.zeros(c.nV, np.float32)
            row[v] = np.asarray(f)[v] * V[:, j]
            P.append(row)
            tags.append(f"{nm}:{j}")
    frac = coverage(c, fields, tau)
    if verbose:
        print(f"  tau {tau:g}: coverage {frac[0]:.2%} ({frac[1]:.0f} mm2), "
              f"{len(tags)} channels from {len(names)} nuclei")
        for i, nm in enumerate(names):
            print(f"    {nm:<14s} {k_n[nm]} modes ({areas[i]:.0f} mm2)")
    return np.asarray(P, np.float32), tags, frac, dict(k_n=k_n)
```

- [ ] **Step 3: Run the tests**

```bash
python3 -m pytest tests/test_ascending.py -v
```

Expected: the five new tests PASS; previous tests still PASS.

- [ ] **Step 4: Inspect the real basis once**

```bash
python fit/ascending.py --channels --coverage 0.085 --budget 80
```

Expected: prints coverage ≈ 8.5% with ~60–100 channels, per-nucleus mode counts, LGN/MGN with the most modes (largest fields).

- [ ] **Step 5: Commit**

```bash
git add fit/ascending.py tests/test_ascending.py
git commit -m "Decompose projection fields into Laplace eigenmode channels"
```

---

### Task 5: Profile-aware impulse cache key

**Files:**
- Modify: `fit/xspec.py`
- Modify: `tests/test_ascending.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `xspec.profile_tag(profiles) -> str`; `xspec.impulse_responses` uses it. Tasks 6+ rely on distinct profile sets never sharing a cache entry.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_ascending.py`:

```python
import xspec


def test_profile_tag_distinguishes_profiles():
    rng = np.random.default_rng(0)
    P1 = rng.standard_normal((5, 200)).astype(np.float32)
    P2 = P1.copy()
    P2[0, 0] += 1e-3
    assert xspec.profile_tag(P1) == xspec.profile_tag(P1)
    assert xspec.profile_tag(P1) != xspec.profile_tag(P2)
    assert xspec.profile_tag(P1) == xspec.profile_tag(np.ascontiguousarray(P1, np.float32))
```

Expected failure: `AttributeError: module 'xspec' has no attribute 'profile_tag'`.

- [ ] **Step 2: Implement `profile_tag` and use it**

In `fit/xspec.py`, add next to `_stack_conj` (before `impulse_responses` so it is defined above its use):

```python
def profile_tag(profiles):
    """Short content hash of a profile matrix, for cache keys.

    The old key carried only (K, sum) - two different bases with the same channel count
    and total drive collided silently. The hash makes the key content-addressed."""
    import hashlib
    h = hashlib.sha1(
        np.ascontiguousarray(profiles, np.float32).tobytes()).hexdigest()[:12]
    return f"prof{profiles.shape[0]}x{float(profiles.sum()):.3f}h{h}"
```

Replace the `ptag` line in `impulse_responses`:

```python
    ptag = "" if profiles is None else f"_prof{profiles.shape[0]}x{float(profiles.sum()):.3f}"
```

with:

```python
    ptag = "" if profiles is None else "_" + profile_tag(profiles)
```

Note: this changes the cache key of every existing taper run, so the first rerun of the pinned incumbent recomputes its impulse responses once (minutes with `--workers 8`). That is intended — old entries were ambiguous.

- [ ] **Step 3: Run the tests**

```bash
python3 -m pytest tests/test_ascending.py -v
```

Expected: `test_profile_tag_distinguishes_profiles` PASS; all others still PASS.

- [ ] **Step 4: Commit**

```bash
git add fit/xspec.py tests/test_ascending.py
git commit -m "Content-hash the profile matrix into the impulse cache key"
```

---

### Task 6: Wire the ascending basis into `best_fit.py`

**Files:**
- Modify: `fit/best_fit.py`

**Interfaces:**
- Consumes: `modal_channels`, `tau_for_coverage`, `load_fields` (Task 4); `profile_tag` (Task 5).
- Produces: `--regions ascending` plus `--ascending-atlas` (default `ascending_v1`), `--ascending-coverage` (default 0.085), `--ascending-tau` (default 0 = auto), `--ascending-modes` (budget, default 80). The npz gains key `profiles` (K, nV) float32 — every `--regions` value saves it, not just ascending.

- [ ] **Step 1: Add the arguments**

In `fit/best_fit.py` `main()`, change the `--regions` line to:

```python
    ap.add_argument("--regions", default="sensory",
                    choices=("sensory", "dmn", "sensory+dmn", "spread",
                             "subcortical", "subcortical+sensory", "ascending"),
                    help="which parcels are driven; 'spread' is an even whole-cortex "
                         "sample matched to the sensory driven area; 'ascending' drives "
                         "tractography-defined nucleus projection fields")
```

and add after `--spread-scale`:

```python
    ap.add_argument("--ascending-coverage", type=float, default=0.085,
                    dest="ascending_coverage",
                    help="driven-area fraction for --regions ascending when "
                         "--ascending-tau is 0 (0.085 ~= the incumbent driven area)")
    ap.add_argument("--ascending-tau", type=float, default=0.0, dest="ascending_tau",
                    help="explicit density threshold; 0 = pick it from --ascending-coverage")
    ap.add_argument("--ascending-modes", type=int, default=80, dest="ascending_modes",
                    help="channel budget for the modal decomposition")
```

- [ ] **Step 2: Build the ascending P/tags instead of the parcel split**

In `main()`, wrap the existing region/profiles block. Currently:

```python
    parcels, split = subparcels.region_set(c, a.regions, a.split, a.spread_scale)
    fc_split = None
    if a.fc_split:
        ...
    labels, tags = subparcels.split_parcels(...)
    if a.profile == "taper":
        P = subparcels.shell_profiles(c, labels, len(tags), a.shells)
        if a.shells > 1:
            ...
    else:
        P = subparcels.gauss_profiles(...)
```

Replace with:

```python
    if a.regions == "ascending":
        import ascending
        names, fields = ascending.load_fields(c)
        tau = (a.ascending_tau if a.ascending_tau > 0
               else ascending.tau_for_coverage(c, names, fields,
                                               a.ascending_coverage))
        P, tags, frac, meta = ascending.modal_channels(
            c, names, fields, tau, a.ascending_modes)
        labels = np.full(c.nV, -1, np.int64)
        print(f"  ascending input: {len(tags)} channels from {len(names)} nuclei, "
              f"tau {tau:g}, coverage {frac[0]:.2%} ({frac[1]:.0f} mm2)")
    else:
        parcels, split = subparcels.region_set(c, a.regions, a.split, a.spread_scale)
        fc_split = None
        if a.fc_split:
            fc_split = subparcels.fc_profiles(c, t, npart=a.fc_split)
            print(f"  splitting pieces by FC profile ({a.fc_split} common partners) rather "
                  f"than by mesh geometry alone")
        labels, tags = subparcels.split_parcels(
            c, parcels, split, verbose=False, fc=fc_split,
            fc_key=f"_fc{a.fc_split}" if a.fc_split else "")
        if a.profile == "taper":
            P = subparcels.shell_profiles(c, labels, len(tags), a.shells)
            if a.shells > 1:
                tags = [f"{t}s{j}" for t in tags for j in range(a.shells)]
                print(f"  {a.shells} concentric shells per piece: {len(P)} channels over the "
                      f"same {len(labels[labels >= 0])} driven vertices. The shells of a piece "
                      f"sum to the profile --shells 1 would have given it")
        else:
            P = subparcels.gauss_profiles(c, labels, len(tags), a.profile_fwhm,
                                          mask=(labels >= 0) if a.profile_mask else None)
```

(All existing branches keep their exact behaviour; only the indentation changes.)

- [ ] **Step 3: Save P in the npz**

In the `np.savez` call at the end of `main()`, add `profiles=P,` after `sub=sub,` so downstream tools can rebuild H without re-deriving the basis:

```python
    np.savez(os.path.join(RESULTS, f"xspec_{a.tag}.npz"), S=S, idx=idx, x=x,
             save=save, labels=labels, tags=np.array(tags, dtype=object),
             pad=a.pad, ref_frames=ref_frames,
             frame_s=(clock["frame_s"] if clock else np.nan),
             band=np.array(bp if bp else (np.nan, np.nan)), segment=seg_fr,
             fc_path=(a.fc_path or ""), H_w=w, regions=a.regions, split=a.split,
             sub=sub, impulse_frames=a.impulse_frames, nfreq=a.nfreq,
             medoid_from=(a.medoid_from or ""), profiles=P,
             maps=np.array(p.get("maps", ()), dtype=object),
             map_a=np.asarray(p.get("a", ()), float),
             map_b=np.asarray(p.get("b", ()), float))
```

- [ ] **Step 4: Smoke run (reduced settings, unique tag)**

```bash
python fit/best_fit.py --oversample 4 --decay-s 25 --spread-mm-s 1.47 --bold-smooth \
  --regions ascending --ascending-coverage 0.085 --bandpass 0.01,0.08 \
  --impulse-decays 3 --pad 4096 --iters 5 --val-vert 0 --seconds 120 --draws 1 \
  --workers 6 --tag ascend_smoke
```

Expected: prints the ascending summary (coverage ≈ 8.5%, channel count), solves 5 iterations, realises 120 s, writes `results/xspec_ascend_smoke.npz`, `results/frames_ascend_smoke.npy`, `results/drive_ascend_smoke.npy`, and prints the realised score line. (This run is a plumbing check only — 5 iterations is not a solve.) Then verify the npz carries the profiles:

```bash
python3 - <<'EOF'
import numpy as np
z = np.load("results/xspec_ascend_smoke.npz", allow_pickle=True)
print("profiles", z["profiles"].shape, z["profiles"].dtype)
print("tags:", len(z["tags"]))
print("regions:", str(z["regions"]), "split:", int(z["split"]))
EOF
```

Expected: `profiles (<K>, 9374) float32`, `tags: <K>`, `regions: ascending`.

- [ ] **Step 5: Commit**

```bash
git add fit/best_fit.py
git commit -m "Add --regions ascending: tractography fields decomposed into modal channels"
```

---

### Task 7: Downstream tools read the saved profiles

**Files:**
- Modify: `analysis/onoff_fields.py`
- Modify: `analysis/interference.py`
- Modify: `analysis/zones.py`

**Interfaces:**
- Consumes: `z["profiles"]` (Task 6); existing `z["labels"]`/`z["tags"]` for the fallback.
- Produces: every H rebuild in these three tools uses the same P the solve used, for ascending and taper runs alike. `zones.group_index` partitions `nucleus:j` tags by nucleus name (roles from the manifest when available).

- [ ] **Step 1: `rebuild_H` and `contributions` prefer `z["profiles"]`**

In `analysis/onoff_fields.py` `rebuild_H`, replace:

```python
    labels, tags = z["labels"], [str(s) for s in z["tags"]]
    p, _, _ = bo_step.unpack(x, c)
    P = subparcels.taper_profiles(c, labels, len(tags))
```

with:

```python
    labels, tags = z["labels"], [str(s) for s in z["tags"]]
    p, _, _ = bo_step.unpack(x, c)
    P = (np.asarray(z["profiles"], np.float32) if "profiles" in z
         else subparcels.taper_profiles(c, labels, len(tags)))
```

In `analysis/interference.py` `contributions`, replace the same `P = subparcels.taper_profiles(...)` line:

```python
    P = (np.asarray(z["profiles"], np.float32) if "profiles" in z
         else subparcels.taper_profiles(c, labels, len(tags)))
```

- [ ] **Step 2: `zones.rebuild` uses saved profiles, window and pad**

In `analysis/zones.py` `rebuild`, replace:

```python
    P = subparcels.taper_profiles(c, labels, len(tags))

    # the clock only enters here through the BOLD smoothing kernel; p and save come from x
    clock = timescale.plan(4, decay_s=9.03, spread_mm_s=6, verbose=False)
    kern = units.smoothing_kernel(timescale.bold_fwhm_frames(clock["frame_s"]))
    decay_fr = 1.0 / (10.0 ** x[0] * save)
    imp = 224
    if imp < 3 * decay_fr:
        imp = int(np.ceil(3 * decay_fr / 64.0) * 64)
    pad = 4096
```

with:

```python
    P = (np.asarray(z["profiles"], np.float32) if "profiles" in z
         else subparcels.taper_profiles(c, labels, len(tags)))

    # the clock only enters here through the BOLD smoothing kernel; p and save come from x
    frame_s = float(z["frame_s"]) if np.isfinite(float(z["frame_s"])) else timescale.TR / 4.0
    kern = units.smoothing_kernel(timescale.bold_fwhm_frames(frame_s))
    decay_fr = 1.0 / (10.0 ** x[0] * save)
    imp = int(z.get("impulse_frames", 224) or 224)
    if imp < 3 * decay_fr:
        imp = int(np.ceil(3 * decay_fr / 64.0) * 64)
    pad = int(z.get("pad", 4096) or 4096)
```

(Old npz files without those keys keep the previous behaviour; ascending runs carry them.)

- [ ] **Step 3: `group_index` handles `nucleus:j` tags**

In `analysis/zones.py`, replace `group_index` with `group_table` (which additionally knows each group's driven vertices — the legacy path derives them from parcel labels, the ascending path from the profile rows):

```python
def group_table(tags, labels, P):
    """-> [(name, role, channel indices, driven mask over cortex vertices)].

    Legacy tags are '<parcel>_<n>' and map through NUCLEI; driven = parcels in the group.
    Ascending tags are '<nucleus>:<mode>' and group by nucleus name directly; driven =
    vertices touched by any of the group's profile rows. P is the (K, nV) profile matrix
    from the run file."""
    tags = [str(s) for s in tags]
    if tags and ":" in tags[0] and not tags[0].split(":")[0].isdigit():
        names = sorted({t.split(":")[0] for t in tags})
        return [(nm, _ascending_role(nm),
                 np.flatnonzero([t.split(":")[0] == nm for t in tags]),
                 np.abs(np.asarray(P)[
                     [i for i, t in enumerate(tags) if t.split(":")[0] == nm]]).sum(0) > 0)
                for nm in names]
    parcel = np.array([int(s.split("_")[0]) for s in tags])
    out = []
    for nm, ps, role in NUCLEI:
        g = np.flatnonzero(np.isin(parcel, ps))
        driven = np.isin(labels, np.flatnonzero(np.isin(parcel, ps)))
        out.append((nm, role, g, driven))
    seen = np.concatenate([o[2] for o in out])
    assert len(seen) == len(tags) and len(set(seen.tolist())) == len(tags), \
        "the nucleus grouping does not partition the driven pieces"
    return out


def _ascending_role(nm):
    import ascending
    for e in ascending.load_manifest():
        if e["name"] == nm:
            return e.get("role", "")
    return ""
```

Then rework `main()` to use it. Replace:

```python
    H, w, S, tags, labels = R["H"], R["w"], R["S"], R["tags"], R["labels"]
    names, gidx = group_index(tags)
    nG = len(names)
```

with:

```python
    H, w, S, tags, labels, P = R["H"], R["w"], R["S"], R["tags"], R["labels"], R["P"]
    groups = group_table(tags, labels, P)
    names = [g[0] for g in groups]
    gidx = [g[2] for g in groups]
    nG = len(names)
```

Replace the attribution table loop:

```python
    print(f"\n  {'group':<10s} {'role':<19s} {'pieces':>6s} {'mm2':>7s}"
          f" {'own':>8s} {'shapley':>8s} {'loo':>8s}")
    area = np.asarray(c.A, float)
    for g, (nm, ps, role) in enumerate(NUCLEI):
        mm2 = float(sum(area[c.lab == q].sum() for q in ps))
        print(f"  {nm:<10s} {role:<19s} {len(gidx[g]):>6d} {mm2:>7.0f}"
              f" {att['own'][:, g].sum()/tot:>7.1%} {att['shapley'][:, g].sum()/tot:>7.1%}"
              f" {att['loo'][:, g].sum()/tot:>7.1%}")
```

with:

```python
    print(f"\n  {'group':<10s} {'role':<19s} {'pieces':>6s} {'mm2':>7s}"
          f" {'own':>8s} {'shapley':>8s} {'loo':>8s}")
    area = np.asarray(c.A, float)
    for g, (nm, role, gidx_g, driven_g) in enumerate(groups):
        mm2 = float(area[driven_g].sum())
        print(f"  {nm:<10s} {role:<19s} {len(gidx_g):>6d} {mm2:>7.0f}"
              f" {att['own'][:, g].sum()/tot:>7.1%} {att['shapley'][:, g].sum()/tot:>7.1%}"
              f" {att['loo'][:, g].sum()/tot:>7.1%}")
```

Replace the zone-size loop's opening:

```python
    srt = np.sort(f_fit, 1)
    contested = ok_fit & ((srt[:, -1] - srt[:, -2]) < a.margin)
    piece_of = np.array([int(s.split("_")[0]) for s in tags])
    dist, radii = {}, {}
    for g, (nm, ps, _role) in enumerate(NUCLEI):
        driven = np.isin(labels, np.flatnonzero(np.isin(piece_of, ps)))
        d = distance_to_drive(c, driven)[t.cols]
```

with:

```python
    srt = np.sort(f_fit, 1)
    contested = ok_fit & ((srt[:, -1] - srt[:, -2]) < a.margin)
    dist, radii = {}, {}
    for g, (nm, _role, _gidx_g, driven_g) in enumerate(groups):
        d = distance_to_drive(c, driven_g)[t.cols]
```

Replace the seed-territory loop:

```python
    Tfc = t.target_fc()
    seeds = []
    for g, (nm, ps, _role) in enumerate(NUCLEI):
        driven = np.isin(labels, np.flatnonzero(np.isin(piece_of, ps)))
        seeds.append(np.flatnonzero(driven[t.cols]))
```

with:

```python
    Tfc = t.target_fc()
    seeds = []
    for g, (nm, _role, _gidx_g, driven_g) in enumerate(groups):
        seeds.append(np.flatnonzero(driven_g[t.cols]))
```

Everything after (`Mfc = t.model_fc(F)` onward) is unchanged — it already iterates `names`.

- [ ] **Step 4: Verify on the smoke run**

```bash
python analysis/zones.py --tag ascend_smoke
```

Expected: runs to completion, grouping the smoke tag's `nucleus:j` channels. The 5-iteration smoke S is underfit, so the rebuild check (`r >= 0.9` against realised variance) may legitimately fail and stop the run — if it does, that is expected here and not a Task 7 failure; the definitive check of the rebuild is the full-run zones invocation in Task 8 Step 6. What this step must show is the group table building without the parcel-path (`split("_")`, NUCLEI) errors — i.e. any failure must come from the `r >= 0.9` guard, not from a tag parse.

- [ ] **Step 5: Commit**

```bash
git add analysis/onoff_fields.py analysis/interference.py analysis/zones.py
git commit -m "Read solved profiles from the run file in the downstream tools"
```

---

### Task 8: Resting-state evaluation

**Files:**
- Create: `analysis/ascending_distance.py`
- Create: `analysis/ascending_readout.py`

**Interfaces:**
- Consumes: `results/xspec_<tag>.npz` (with `profiles`), `results/frames_<tag>.npy`, `fc_score.FCTarget`, `diag_distance.distance_to_drive`, `xspec.piece_power`, `xspec.family_member`, `xspec.share_reg`, `onoff_fields.input_cov`.
- Produces: `results/ascending_distance_<tag>.npz`, `results/ascending_readout_<tag>.npz` + printed tables.

- [ ] **Step 1: The incumbent-comparable solve**

Run the pinned incumbent flags with the ascending basis at the incumbent coverage (this is the headline run; ~1–2 h on 16 cores):

```bash
python fit/best_fit.py --oversample 4 --decay-s 25 --spread-mm-s 1.47 --bold-smooth \
  --regions ascending --ascending-coverage 0.085 --bandpass 0.01,0.08 \
  --impulse-decays 7 --pad 4096 --iters 400 --val-vert 0 --seconds 2308 --draws 2 \
  --workers 8 --tag ascend_085
```

Expected: completes and prints `sim <x> +- <y>` and `rank <z>`. Record the realised sim against the incumbent ~+0.72 and against the split-half ceiling (`data/cache/band_ceiling_*.npz`; read `targets/reliability.py` for how the ceiling is quoted). If the impulse responses for the ascending profiles were not yet cached, this run pays the ~100-channel impulse computation once.

- [ ] **Step 2: Check the stopping point is adequate (empirical gate)**

```bash
python fit/best_fit.py --oversample 4 --decay-s 25 --spread-mm-s 1.47 --bold-smooth \
  --regions ascending --ascending-coverage 0.085 --bandpass 0.01,0.08 \
  --impulse-decays 7 --pad 4096 --val-vert 0 --select-iters 25,50,100,200,400,800 \
  --select-frames 1120 --seconds 2308 --draws 1 --workers 8 --tag ascend_sel
```

Expected: prints the realised held-out score per candidate stopping point and the chosen `--iters`. If the peak is at the top of the grid (800), the gradient solve is the wrong tool at this K — fall back to the factor solver for the sweep:

```bash
python fit/best_fit.py --oversample 4 --decay-s 25 --spread-mm-s 1.47 --bold-smooth \
  --regions ascending --ascending-coverage 0.085 --bandpass 0.01,0.08 \
  --impulse-decays 7 --pad 4096 --solver factor --rank 16 --maxfun 2000 \
  --seconds 2308 --draws 2 --workers 8 --tag ascend_fac
```

Use whichever solver the gate selects for Steps 3–4 and record the choice in the commit message.

Also measure the conditioning the spec calls for (diagnostic only; whitening stays off):

```bash
python3 - <<'EOF'
import numpy as np
from mesh_cache import load_cortex
import fc_score
from zones import rebuild
c = load_cortex("fsaverage5", verbose=False)
t = fc_score.default_target(c, verbose=False)
R = rebuild("ascend_085", c, t)
H = R["H"]
cond = np.array([np.linalg.cond(H[f].conj().T @ H[f]) for f in range(H.shape[0])])
print(f"cond(H^H H) over {len(cond)} frequencies: min {cond.min():.3g}, "
      f"median {np.median(cond):.3g}, max {cond.max():.3g}")
EOF
```

Expected: prints the condition numbers. A median above ~1e6 with the select-iters gate peaking at the grid top together mean the gradient solve is the wrong tool at this K — that is the trigger for the `--solver factor` fallback above, not a reason to re-enable whitening.

- [ ] **Step 3: The coverage sweep (coarse, five points)**

```bash
for c in 0.06 0.08 0.10 0.12 0.14; do
  python fit/best_fit.py --oversample 4 --decay-s 25 --spread-mm-s 1.47 --bold-smooth \
    --regions ascending --ascending-coverage $c --bandpass 0.01,0.08 \
    --impulse-decays 7 --pad 4096 --iters 400 --val-vert 0 --seconds 2308 --draws 2 \
    --workers 8 --tag ascend_c${c/./}
done
```

Expected: five completed runs with tags `ascend_c006` … `ascend_c014`; print the five realised sims side by side. (The impulse responses differ per coverage — each tau changes P — so each run pays its own impulse computation; the profile content hash in Task 5 is what keeps them from colliding.)

- [ ] **Step 4: `analysis/ascending_distance.py` — where the prediction lives**

```python
"""Score binned by geodesic distance from the driven set.

For every vertex, correlate its FC row (model against target) and bin the mean by the
vertex's distance to the nearest driven vertex. This measures whether waves actually
account for FC far from the input or only near it.

  python analysis/ascending_distance.py --tag ascend_085
"""
import _path  # noqa: F401
import os, argparse
import numpy as np

from mesh_cache import load_cortex
from paths import RESULTS
import fc_score
from diag_distance import distance_to_drive


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", default="ascend_085")
    ap.add_argument("--nbins", type=int, default=6)
    a = ap.parse_args()

    c = load_cortex("fsaverage5", verbose=False)
    t = fc_score.default_target(c, verbose=False)
    z = np.load(os.path.join(RESULTS, f"xspec_{a.tag}.npz"), allow_pickle=True)
    P = np.asarray(z["profiles"], np.float32)
    driven = np.abs(P).sum(0) > 0
    print(f"  {a.tag}: {int(driven.sum())} driven vertices "
          f"({c.A[driven].sum()/c.A.sum():.2%} of area)")

    F = np.asarray(np.load(os.path.join(RESULTS, f"frames_{a.tag}.npy")), np.float32)
    Z, _ = t.model_z(F)
    T = Z.shape[1]
    M = (Z @ Z.T) / T
    ssum = Z.sum(0)
    diag = (Z * Z).sum(1) / T
    V = Z.shape[0]
    m = ((Z @ ssum) / T - diag) / (V - 1)
    grand = (float(ssum @ ssum) / T - float(diag.sum())) / (V * (V - 1))
    M = M - m[:, None] - m[None, :] + grand
    Tf = np.asarray(t.target_fc(), np.float64)
    Tf = Tf - Tf.mean(1, keepdims=True)
    M = M - M.mean(1, keepdims=True)
    denom = np.sqrt((M ** 2).sum(1) * (Tf ** 2).sum(1))
    r = np.divide((M * Tf).sum(1), denom, out=np.zeros_like(denom), where=denom > 0)

    d = distance_to_drive(c, driven)
    edges = np.percentile(d, np.linspace(0, 100, a.nbins + 1))
    print(f"\n  distance bins (mm): {np.round(edges, 1)}")
    out = {}
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        m = (d >= lo) & (d < hi)
        out[f"mean_r_{i}"] = float(r[m].mean())
        out[f"n_{i}"] = int(m.sum())
        print(f"    {lo:6.1f}-{hi:6.1f} mm  n={int(m.sum()):>5d}  mean row corr "
              f"{r[m].mean():+.4f}")
    np.savez(os.path.join(RESULTS, f"ascending_distance_{a.tag}.npz"),
             edges=edges, r=r, d=d, driven=driven, **out)
    print(f"  wrote results/ascending_distance_{a.tag}.npz")


if __name__ == "__main__":
    main()
```

Run it: `python analysis/ascending_distance.py --tag ascend_085`. Expected: a table of mean row correlations by distance; the bin means are what get reported (no claim about them beyond the table).

- [ ] **Step 5: `analysis/ascending_readout.py` — the scientific reading**

```python
"""Read the solved S(f) back as ascending systems: per-nucleus power, coordination,
mode content, and how far each nucleus's power can move at matched fit.

  python analysis/ascending_readout.py --tag ascend_085
"""
import _path  # noqa: F401
import os, argparse
import numpy as np

from mesh_cache import load_cortex
from paths import RESULTS
import fc_score, xspec
import onoff_fields


def nuclei(tags):
    return sorted({str(t).split(":")[0] for t in tags})


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", default="ascend_085")
    ap.add_argument("--eps", default="0.002,0.01")
    ap.add_argument("--band", default="0.01,0.08")
    a = ap.parse_args()

    c = load_cortex("fsaverage5", verbose=False)
    t = fc_score.default_target(c, verbose=False)
    z = np.load(os.path.join(RESULTS, f"xspec_{a.tag}.npz"), allow_pickle=True)
    S = np.asarray(z["S"])
    tags = [str(s) for s in z["tags"]]
    nms = nuclei(tags)
    band = tuple(float(v) for v in a.band.split(","))

    # --- per-nucleus input power, mode content, and lag-0 coordination ---------------
    pw = xspec.piece_power(S, z["idx"], float(z["frame_s"]), band=band,
                           pad=int(z["pad"]))
    print(f"\n  input power in {a.band} Hz, by nucleus (and mode split):")
    for nm in nms:
        k = [i for i, tg in enumerate(tags) if tg.split(":")[0] == nm]
        share = pw[k] / max(pw[k].sum(), 1e-30)
        print(f"    {nm:<12s} {pw[k].sum():9.4g}   modes "
              + " ".join(f"{tags[i].split(':')[1]}={share[j]:.2f}"
                         for j, i in enumerate(k)))

    C0 = onoff_fields.to_corr(onoff_fields.input_cov(z, 0.0, band))
    print(f"\n  lag-0 input correlation between nuclei:")
    for i, a_nm in enumerate(nms):
        ka = [k for k, tg in enumerate(tags) if tg.split(":")[0] == a_nm]
        row = []
        for j, b_nm in enumerate(nms):
            if j <= i:
                continue
            kb = [k for k, tg in enumerate(tags) if tg.split(":")[0] == b_nm]
            row.append(f"{b_nm}={C0[np.ix_(ka, kb)].max():+.3f}")
        if row:
            print(f"    {a_nm:<12s} " + "  ".join(row))

    # --- family brackets: how far can each nucleus's power move at matched fit --------
    from zones import rebuild
    R = rebuild(a.tag, c, t)
    H, w = R["H"], R["w"]
    raw = np.asarray(t.target_fc(), np.float64)
    raw = raw - raw.mean(0, keepdims=True) - raw.mean(1, keepdims=True) + raw.mean()
    Ct = xspec.normal_scores(raw)
    print(f"\n  family brackets at matched fit (share of total input power):")
    for nm in nms:
        k = [i for i, tg in enumerate(tags) if tg.split(":")[0] == nm]
        share0 = float(pw[k].sum() / pw.sum())
        lo, hi = share0, share0
        for sign in (-1, 1):
            Rg = xspec.share_reg(w, k, H.shape[2])
            S2, rep = xspec.family_member(H, w, Ct, S, Rg, eps=0.01, iters=200,
                                          sign=sign, verbose=False)
            pw2 = xspec.piece_power(S2, z["idx"], float(z["frame_s"]), band=band,
                                    pad=int(z["pad"]))
            v = float(pw2[k].sum() / max(pw2.sum(), 1e-30))
            lo, hi = min(lo, v), max(hi, v)
        print(f"    {nm:<12s} share {share0:.3f} in [{lo:.3f}, {hi:.3f}]")
```

Run: `python analysis/ascending_readout.py --tag ascend_085 --eps 0.002,0.01`. Expected: the power/mode table, the nucleus coordination table, and the brackets. This is the most expensive step (family_member at full vertex width); run it serially and do not parallelise anything alongside it.

- [ ] **Step 6: Interference on the best run**

```bash
python analysis/interference.py --tag ascend_085
python analysis/zones.py --tag ascend_085
```

Expected: `interference.py` prints the superposition check (`r` ≥ 0.99 against the saved frames — if below, STOP and debug before reading any numbers) then the coherent/incoherent ratios; `zones.py` prints the per-nucleus attributions.

- [ ] **Step 7: Commit the analysis scripts and results summary**

```bash
git add analysis/ascending_distance.py analysis/ascending_readout.py
git commit -m "Add distance-binned scoring and ascending-system readout"
```

Record the headline numbers (sim vs 0.72 and the ceiling, the coverage table, the distance table) in the commit message body — measured values only, no interpretation.

---

### Task 9: ON−OFF with the ascending basis

**Files:**
- No new code (existing `analysis/onoff_solve.py` / `analysis/onoff_fields.py`).
- (If `onoff_solve.ascend` needs the difference ascent unchanged, this task is entirely runs.)

**Interfaces:**
- Consumes: Task 6 (`--regions ascending`, `--fc-path`, `--medoid-from`), Task 7 (`rebuild_H` reads `z["profiles"]`), Task 1/8 target machinery.

- [ ] **Step 1: Build the task targets if absent**

```bash
ls results/fc/task 2>/dev/null || python targets/fc_group_rbc.py --targets
```

Expected: `results/fc/task/` exists with matrices including the CHECKER (`on`) and FIXATION (`off`) segment-demeaned FC files (`*on_out*` / `*off_out*` style names; list them to confirm exact paths). If the folder already exists and contains them, skip the build.

- [ ] **Step 2: Solve the two arms on the ascending basis, same vertices**

From the listing in Step 1, set `ON_FC` and `OFF_FC` to the full paths of the CHECKER and FIXATION segment-demeaned matrices (`*on_out*` / `*off_out*` style names) and substitute them into:

```bash
python fit/best_fit.py --oversample 4 --decay-s 25 --spread-mm-s 1.47 --bold-smooth \
  --regions ascending --ascending-coverage 0.085 --bandpass 0.01,0.08 \
  --impulse-decays 7 --pad 4096 --iters 400 --val-vert 0 --seconds 2308 --draws 2 \
  --workers 8 --fc-path "$ON_FC" --medoid-from "$ON_FC" --tag ascend_on

python fit/best_fit.py --oversample 4 --decay-s 25 --spread-mm-s 1.47 --bold-smooth \
  --regions ascending --ascending-coverage 0.085 --bandpass 0.01,0.08 \
  --impulse-decays 7 --pad 4096 --iters 400 --val-vert 0 --seconds 2308 --draws 2 \
  --workers 8 --fc-path "$OFF_FC" --medoid-from "$ON_FC" --tag ascend_off
```

`--medoid-from` is the ON matrix in BOTH runs so the two arms solve on identical vertices (the difference ascent requires it). Expected: two completed runs.

- [ ] **Step 3: The difference solve**

```bash
python analysis/onoff_solve.py --a ascend_on --b ascend_off --iters 200
python analysis/onoff_fields.py --a ascend_on --b ascend_off
```

Expected: `onoff_solve.py` prints the baseline (the two independent solves' difference against the target difference), the free ascent from three starts, and the eps-constrained runs; `onoff_fields.py` prints the input/field comparisons. The question, per the spec: does the solved ON−OFF input differ from rest in amplitude, phase, or location.

- [ ] **Step 4: Commit**

```bash
git add -A results/ascending* 2>/dev/null || true
git commit -m "ON-OFF solves on the ascending basis" --allow-empty
```

(Only commit small result tables if the repo already tracks any; `results/` files are otherwise untracked by convention — check `git status` and follow what previous sessions committed. If nothing is tracked, use the empty commit only if the branch history needs the marker, else skip.)

---

## Self-review notes (done, fixed inline)

- Spec coverage: sections 1.1–1.4 → Tasks 1–4; section 2 → Tasks 5–6; section 3 → Task 8 Step 2 (gate) with `--solver factor` fallback; section 4 items 1–5 → Task 8 Steps 1, 3, 4, 5–6 and Task 9; section 5 (testing) → pytest tasks 2–5 and the anatomy test in Task 3; section 6 (out of scope) → respected (medium/whitening/HRF untouched).
- Placeholder scan: removed all TBD-style steps; external URLs are resolved by reading the source page in Task 1; manifest labels are verified against the atlas output before use.
- Type consistency: `modal_channels` returns `(P, tags, frac, meta)` everywhere it is called; `projection_fields`/`load_fields` return `(names, fields)`; `z["profiles"]` is (K, nV) float32 in best_fit, onoff_fields, interference, zones.
