# Ascending input basis: tractography-defined regions and modal channels

Date: 2026-09-10
Branch: `explore-input-solver`
Status: approved design, awaiting implementation plan

## Goal

Replace the Glasser-parcel input basis with one defined by the ascending pathways into
cortex, and give the solve more expressive power to shape waves inside each pathway's
projection field. The picture is streams flowing into a lake: input enters cortex only at
the ascending projection zones, waves propagate from there through the (unchanged) medium,
and the question is how much of the whole-cortex FC the propagating waves account for.
Driven-area coverage is a first-class control, not an incidental output.

## Fixed premises (do not revisit)

From `NEXT_SESSION.md`, unchanged:

- The medium is settled: spread 1.47 mm/s, decay 25 s, `BEST_X` map coefficients. Same for
  rest and task.
- No HRF.
- No whitening. (It was tried against the incumbent and did not help; it is off here too.)
- `--iters 400` is the convention. The solve does not converge and does not stall; the
  stopping point is part of the configuration.
- The passband (0.01–0.08 Hz) is a property of the measurement.
- The medium has no arbitrary endogenous noise; all variability comes from the input.
- A refit of the medium is deferred: if this basis matches or beats the incumbent, the
  medium becomes its own follow-up exploration, not part of this work.

## 1. Input construction: nuclei → fields → modes → channels

### 1.1 Nucleus masks (MNI 1 mm)

Atlas stack, stored under `data/atlases/`:

- Thalamus: THOMAS (preferred) or Morel, ~14 nuclei.
- Brainstem ascending arousal: Harvard Ascending Arousal Network atlas — LC, DRN, MRN,
  VTA, PPN, LDT, PB.
- Basal forebrain: Ch1–4 maps.
- Hypothalamus: TMN, LH.
- Amygdala (Harvard–Oxford or equivalent).

Each entry is a probability mask in MNI152 1 mm space.

### 1.2 Projection fields via group tractography (route B)

- Track from each nucleus mask through the DSI Studio HCP1021 group FIB (single download,
  ~2–8 GB, stored under `data/`) with `dsi_studio --action=trk`, seed-count per nucleus
  sized by mask volume.
- Voxelate streamline endpoints into a per-nucleus endpoint-density volume in MNI space.
- Map to fsaverage5-left with `wb_command -volume-to-surface-mapping`, ribbon-constrained,
  using templateflow's fsaverage surfaces in MNI152NLin6Asym. Output: per-nucleus graded
  density field φₙ over vertices.
- Normalise each φₙ to unit max. The drive keeps the existing area weighting and net
  injection removal.

The cohort is HCP, not NKI — accepted trade-off: one download, track directly in MNI, no
per-subject pipeline. (The NKI DWI on disk was considered and rejected for this round
because it would need per-subject processing and warps the XCP-D derivatives do not ship.)

### 1.3 Coverage control

- A global density threshold τ: a vertex is *driven* if maxₙ φₙ(v) > τ.
- Coverage is reported as driven mm² and % of cortex.
- The exploration sweep is coarse and small: τ chosen to give 6, 8, 10, 12, 14% driven
  cortex. The incumbent subcortical configuration (≈8–9%) is one point on the sweep, not
  the target. No more than ~20% is in scope.

### 1.4 Modal channels

- On each nucleus's support (φₙ > τ), build the mesh-graph Laplacian weighted by
  1 − φₙ/φₙmax (weakly connected periphery cheap to cut). Take the kₙ smallest eigenmodes.
- Channels = (nucleus, mode); profile = φₙ × ψⱼ. Mode 0 approximates the plain projection
  field, so the incumbent taper behaviour is a special case; higher modes give
  radial/azimuthal shape freedom whose phases the solve sets — wave shaping comes from
  S(f)'s cross terms, as before, now between modal channels.
- kₙ = clamp(round(support areaₙ / target piece area), 1, k_max), the same area-proportional
  logic `split_parcels` uses, with `target piece area` chosen so the global channel budget
  lands at ≈60–120; k_max = 4–5. Bigger fields get more modes, tiny fields stay at one.
- Modes are signed and overlap. Signed drive is already legal in the model (timecourses
  are a source, not a depth).

## 2. Pipeline integration

- New module `fit/ascending.py` (plain file, `_path.py` convention, runnable from root):
  - `nucleus_masks(atlas)` → named MNI masks
  - `projection_fields(masks, fib)` → per-nucleus φₙ on fsaverage5-left
  - `modal_channels(fields, tau, budget)` → P (K, nV) and per-row `(nucleus, mode)` tags
- `fit/best_fit.py`: new `--regions ascending` choice plus `--ascending-tau`,
  `--ascending-modes`, `--ascending-atlas`. P plumbs into `impulse_responses(profiles=P)`
  unchanged. Solve, realise, score paths unchanged.
- Caches: fields and channels cached under `data/cache/` keyed by atlas + τ + budget. The
  impulse-response cache key must distinguish field/mode sets — fold a short hash of P
  into `ptag` (the current `prof{K}x{sum}` key can collide). At K≈120 the impulse array is
  ~4–5 GB per medium configuration; exploration runs should avoid filling the cache
  indiscriminately (94 GB free, 26 GB already used). Existing BLAS-throttling rules for
  parallel scoring still apply.
- `core/`, `fit/xspec.py` and the onoff scripts change only where they read profiles.

## 3. Solver handling

- The solve stays the convex projected-gradient ascent over PSD S(f) — `C(S)` is still
  linear in S, so the mechanism, trace normalisation and the ON−OFF difference objective
  all apply verbatim; only H changes.
- Whitening stays off. The condition of H^H H per frequency is *measured* as a diagnostic.
  If the gradient solve stalls or the realised score collapses early at larger K, the
  response is not whitening:
  - fall back to `--solver factor` (S = L L^H, declared rank, L-BFGS to tolerance), and/or
  - drop near-redundant channels (modes nearly spanned by their neighbours).
- The 400-iteration convention stays for comparability but is not assumed adequate at
  larger K; the select-iters sweep (realised, held-out score over candidate stopping
  points) is the empirical gate.
- Overfitting guard: the existing held-out early stopping, select-iters selection, and the
  rank diagnostic. A more expressive basis losing generalisation is a finding to report,
  not a bug to patch — no new regulariser is added.

## 4. Evaluation and what counts as progress

1. **Incumbent-comparable solve.** New basis at ≈8–9% coverage, resting target, everything
   else pinned (medium, bandpass, bold-smooth, iters convention). Realised held-out
   Spearman against the incumbent ≈+0.72 and against the split-half reliability ceiling.
2. **Coverage curve (coarse).** τ set for 6, 8, 10, 12, 14% driven cortex — five points,
   one full solve + realise + score each.
3. **Where the prediction lives.** Score restricted to vertices binned by geodesic
   distance from the driven set — does the model account for FC far from the input?
4. **Interference and zones** (the interesting part):
   - `analysis/interference.py`, adapted to read the new P/tags instead of rebuilding
     taper profiles: per-vertex coherent/incoherent power ratio, instantaneous
     constructive/destructive fronts — the interference patterns underlying the FC.
   - `analysis/zones.py`, now keyed by real nucleus names (channels are (nucleus, mode),
     no parcel→nucleus mapping needed): own-block / Shapley / leave-one-out variance
     attribution over diag(C) per ascending system; their disagreement is the
     between-system interference, vertex by vertex.
   - Solved S(f) readout: per-nucleus input power (diagonal), nucleus–nucleus
     coordination (cross terms, phase), per-nucleus mode content. `family_member` with the
     per-channel share regulariser brackets how far each nucleus's power can move at
     matched fit — which ascending systems the FC pins and which it cannot decide.
5. **ON−OFF.** Rebuild ON/OFF targets as before (`fc_group_rbc.build_task_targets`), solve
   both arms on the new basis (`--medoid-from` convention for shared vertices), run the
   difference ascent free and eps-constrained (`analysis/onoff_solve.py`, logic unchanged).
   The question stays: does the ON−OFF input differ from rest in amplitude, phase, or
   location?

## 5. Testing

Unit-testable pieces: field mapping lands vertices inside the anatomical projection
zones; envelope normalisation and area weighting; mode orthonormality on the weighted
Laplacian; mode 0 ≈ plain field; channel budget allocation; cache-key uniqueness across
τ/mode sets; net injection removal. Solve/score paths are verified by comparison against
the pinned incumbent configuration, not by new tests.

## 6. Out of scope

- Refitting the medium (deferred follow-up).
- Whitening, HRF, passband re-derivation (fixed list).
- Per-subject NKI tractography (rejected for this round; revisit only if the HCP-template
  fields prove unusable).
- Right-hemisphere fields; the model is left-hemisphere only.
