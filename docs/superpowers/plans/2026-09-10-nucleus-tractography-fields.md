# Vertex-level thalamocortical input field, by tractography

Date: 2026-09-10
Status: plan for a fresh session (this session degenerated; do not reuse its conclusions)

## Goal

Build the **vertex-level cortical projection field of the THALAMUS** on fsaverage5-left:
seed the left thalamus in the HCP1065 group dMRI template, track to cortex, rasterise
streamline endpoints onto cortical vertices. This field — decomposed into modes — is the
input basis for the rotating shallow-water model.

**Per-nucleus separation is a bonus, not the requirement.** The primary deliverable is ONE
vertex-level cortical field: where the thalamus reaches cortex.

## Premises (binding)

- **Objective: whole-cortex FC fit. Unchanged.**
- **Input: NOT whole-cortex.** Localized anatomical streams entering cortex at fixed
  places. The solve moves their amplitudes/coherence/phase — never their spatial spread.
- **Vertex-level at the cortex.** The cortical side must be a per-vertex weight map, not
  an ROI. That is the entire reason to do tractography.
- **Seed from the thalamus, never from cortical ROIs.** Seeding a big cortical region
  destroys the point: the cortical side becomes the ROI instead of vertex-level endpoint
  density.
- Never put whole-brain maps (neuromodulatory PET transporter maps etc.) into the input
  basis. That mistake killed the previous session — those maps are diffuse by biology and
  are the Hansen 2022 "neurotransmitter systems" paper. Do not repeat.
- The medium is pinned. No HRF. No whitening. Coverage stays ~8-15%.

## Prior art (this exact thing has been done)

- Behrens et al. 2003, Nat Neurosci: whole-thalamus probabilistic tractography -> voxel-wise
  thalamus-cortex connectivity.
- Johansen-Berg et al. 2005, Nat Neurosci (FSL Thalamic Connectivity Atlas).
- Najdenovska et al. 2018 (THOMAS): connectivity-based thalamic parcellation.
- Nature Neuroscience 2025: Morel-guided nuclei -> cortex with a tractography atlas.

Our recipe = Behrens 2003, with vertex-level (not zone-level) cortical output.

## Data & tools already in place

- `data/hcp1021.fib.gz` — HCP1065 group FIB, 2 mm, ICBM152 2009a space (78x94x68).
  `data/atlases/fib_source.txt` documents it.
- DSI Studio CLI at `/Applications/DSI Studio.app/Contents/MacOS/dsi_studio`.
- `data/atlases/dsistudio/THOMAS.nii.gz` (24 nucleus labels, left = odd labels 1-23) and
  `JulichBrain.nii.gz` (basal forebrain, BG subregions), both in the FIB's own space.
- Connectome Workbench `wb_command` at `~/Downloads/workbench/bin_macosxub/wb_command`.
- neuromaps fsaverage5 white/pial surfaces in MNI152 space (vertex-identical to the repo's
  fsaverage5; already used for volume->surface mapping).

## Tested building blocks (in fit/ascending.py, keep these)

- `_parse_tt(path)` — decodes DSI Studio's `.tt.gz` tract format (reverse-engineered from
  DSI Studio source; unit-tested). Streamlines come back in FIB voxel coordinates.
- `_endpoint_density(tt_path)` — rasterises streamline endpoints onto the FIB grid.
- `_vol_to_surface(V, affine, name, tmpdir)` — maps an MNI-mm volume to fsaverage5 (10,242)
  via wb_command ribbon-constrained with neuromaps surfaces; restrict to the cortex
  submesh with `c.old`.
- Modal decomposition + coverage control (`support`, `_field_modes`, `modal_channels`,
  `coverage`, `tau_for_coverage`) — unit-tested.

## Steps

1. **Quality gate.** Seed the whole LEFT thalamus (union of THOMAS left labels) with
   ~100-200k seeds in the 2 mm FIB. Report: fraction of streamlines whose far endpoint
   lands in cortical gray matter; length distribution. The previous failure mode (tiny LGN
   seed) was 84% of endpoints staying in the thalamus — gray-matter seeds terminate before
   reaching white matter. The whole thalamus is a much larger seed, but if the same
   problem appears, in order:
   a. probabilistic tracking (`--method=1`), turn angle 60,
   b. lower the tracking QA threshold (`--otsu_threshold` ~0.3-0.5; gray matter needs a
      lower threshold than the default 0.6),
   c. find a **1 mm** FIB (search Zenodo for HCP842/HCP1065 1mm; the labsolver 1 mm links
      were dead). The gate exists for exactly this decision.

2. **The thalamic cortical field.** Rasterise all streamline endpoints into the FIB-grid
   density volume -> `_vol_to_surface` -> ONE vertex-level field on fsaverage5-left,
   normalised to unit max.

3. **Validation gate (hard).** Using the repo's Glasser annotation (validation only, not
   for definition): the field must be strong over primary sensory/motor cortex (V1, A1,
   S1, M1) and graded elsewhere; it must not be diffuse noise. Show the map on the
   surface. Do not proceed to the model until this passes.

4. **Bonus — per-nucleus sub-fields.** Attribute each streamline to the THOMAS label at
   its FIRST point (= its seed location); build per-nucleus vertex-level fields.
   Validate: LGN->V1, MGN->A1, VPL->S1, MD->PFC. If clean, they can serve as separate
   streams (sharper scientific reading); if not, the whole-thalamus field plus modal
   decomposition is the deliverable and nothing is lost.

5. **Basal ganglia (later).** Same recipe seeded from GPi/SNr (JulichBrain/Melbourne
   labels); their cortical endpoints arrive via the thalamus and land in motor/PFC.

## Then the model (machinery mostly exists)

6. The vertex-level field(s) replace the input basis. Discard the "published_fields /
   neuromodulatory" path in fit/ascending.py — that was the wrong turn.
7. Laplace eigenmode decomposition + coverage control (already built and tested).
8. Solve: restricted input basis, whole-cortex FC objective. Evaluate: coverage sweep
   (coarse), interference/zones, ON-OFF.

## Pitfalls that cost the last session (do not repeat)

- DSI Studio flags: `--turning_angle` (NOT `--turn_angle`); `--output` is a DIRECTORY;
  the tract file is `<source>.tt.gz` inside it (parse directly, do not convert to TRK).
- Seeds must be written in their native MNI space with their own sform and "mni" in the
  filename — DSI Studio registers them. Pre-resampling seeds onto the FIB grid misplaces
  them (observed: seed read at (46,21,10) instead of (50,51,22)).
- `neuromaps.datasets.fetch_annotation` returns a path string, not a list.
- The 2 mm FIB may be too coarse for the small nuclei — resolve via the quality gate, not
  by shrugging.
- Whole-brain input maps are banned. Localized streams only.
