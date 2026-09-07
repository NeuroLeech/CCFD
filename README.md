# CCFD

Rotating shallow-water flow on a cortical surface, driven at named Glasser
parcels, compared against resting-state fMRI.

The surface is the left fsaverage5 inflated mesh with the medial wall cut out
(9,374 vertices). Fluid depth `h` and edge-normal velocity `u` evolve under the
linear rotating shallow-water equations, discretised with a C-grid / DEC scheme
whose Coriolis operator is energy-neutral by construction.

The input is **solved for, not searched**. Because the medium is linear, the
field covariance depends on the drive only through its cross-spectral density
`S(f)`, so matching functional connectivity is a convex problem over one
Hermitian PSD matrix per frequency — it has an optimum instead of a plateau.
The pipeline is: impulse response per driven piece → `H(f)` by FFT → solve
`max corr(sum_f H S H^H, target)` over `S(f) >= 0` → draw a drive with that
cross-spectrum → simulate → score.

Scoring is against the 99-subject NKI group FC on fsaverage5, double-centred,
by Spearman over a fixed 2M edge sample. A solve correlation is not a score:
every candidate is realised and simulated before any number is quoted.

## What it looks like

[![the best fit, on the surface](docs/best_field.gif)](docs/best_field.mp4)

The loop above is the first 15 s, surfaces only; the full clip is
[`docs/best_field.mp4`](docs/best_field.mp4) — 30 s, with the drive traced underneath.

Both show depth `h` on the left fsaverage5 inflated surface, lateral / medial / dorsal,
on a colour scale held fixed across the clip so amplitude stays comparable frame to frame
instead of being renormalised. The mp4 adds the six loudest drive channels underneath,
with a cursor on the current frame. It covers 600 saved frames of the 3,578-frame
realisation — 97 s of 577 s — at 20 fps, about 3.2x real time.

The run is the `Current best` of `PLAN.md`: 47 subcortically driven pieces over
8,542 mm², spread 6 mm/s, decay 9.03 s pinned to the NKI autocorrelation, scoring
**+0.7204 ± 0.0009** Spearman over 2M edges. To rebuild it:

```bash
python fit/best_fit.py --oversample 4 --decay-s 9.03 --spread-mm-s 6 --bold-smooth \
  --pad 4096 --impulse-frames 224 --iters 400 --val-vert 0 --draws 2 \
  --regions subcortical --split 40 --profile taper
python viz/render_frames.py --tag pr_taper --start 200 --n 600 --save 16 --fps 20
```

## Layout

The tree is five folders and no package: there is no install step and every script is run
as a plain file from the repository root, `python fit/best_fit.py --tag ...`, importing
what it needs by plain name. Each folder carries a copy of `_path.py`, which every runnable
file imports first — Python puts only the script's own directory on `sys.path`, so without
it a script in `analysis/` could not find `fit/xspec.py`.

| folder | what lives there |
|---|---|
| `core/` | the mesh, the medium, the clock and the observable — `paths`, `mesh_cache`, `surf_ops`, `fluid`, `swe_rot`, `units`, `timescale`, `bandpass`, `ladder`, `regimes` |
| `targets/` | reading scans and building the FC targets they are scored against — `rbc`, `fc_score`, `fc_group_nki`, `fc_group_rbc`, `fc_vertexwise`, `reliability`, `holdout`, `lagged`, `checkerboard` |
| `fit/` | the input model and the convex solve — `xspec`, `best_fit`, `bo_step`, `subparcels`, `connectome`, `segments`, `family` |
| `analysis/` | diagnostics and experiments — `diag_*`, `band_*`, `interference`, `zones`, `checkerboard_model`, `checkerboard_modulation`, `onoff_*`, `task_inputs` |
| `viz/` | `render_*`, `plot_*`, `surface_plots` |

`data/` holds the datasets and `results/` everything written; both are addressed through
`core/paths.py`, which derives them from the repository root rather than from a working
directory.

| file | what it does |
|---|---|
| `core/paths.py` | every path, derived from the repo location |
| `core/surf_ops.py` | surface loading, primal/dual (DEC) operators |
| `core/intrinsic_delaunay.py` | metric repair; without it the scheme blows up in ~200 steps |
| `core/mesh_cache.py` | builds and caches the `Cortex` object: mesh + atlas + repaired metric |
| `core/swe_rot.py` | the solver: `RotSWE.step`, plus the absorbing rim sponge |
| `core/fluid.py` | the medium: speed and damping graded by cortical maps, integration |
| **the fit** | |
| `fit/best_fit.py` | reproduce the current fit; the entry point |
| `fit/xspec.py` | transfer function, the convex solve, realisation, scoring |
| `fit/subparcels.py` | equal-area splitting; which parcels are driven |
| `fit/bo_step.py` | Bayesian optimisation over the medium, in per-step units |
| `fit/coalitions.py` | read the solved `S(f)` back as amplitudes and time offsets |
| **targets and controls** | |
| `targets/fc_vertexwise.py` | vertexwise FC from MSC surface data |
| `targets/fc_group_nki.py` | the NKI group target (99 usable subjects) |
| `targets/fc_centre.py` | double-centring: the linear analogue of global signal regression |
| `targets/fc_score.py` | `FCTarget`: alignment, edge sample, Spearman score |
| `targets/fc_moran.py` | spatial autocorrelation match, as a diagnostic |
| `fit/family.py` | the family of inputs consistent with the target, not just the argmax |
| `targets/fc_states.py` | windowed-FC states, occupancy, dwell, transitions |
| `targets/reliability.py` | split-half reliability of the target, and the ceiling it implies |
| `targets/holdout.py` | solve on one half of the subjects, score on the other |
| `analysis/reach.py` | can this fluid produce the target's patterns at all? |
| `analysis/diag_maps.py` | where the fit fails, drawn on the surface |
| **input, by hand** | |
| `fit/input2.py` | input as K regions with timecourses supplied directly |
| `fit/input_model.py` | input as K regions driven through r shared latent factors |
| `core/ladder.py` | a nested family of input processes; parcel geodesics |
| `core/run_ou.py` | Ornstein-Uhlenbeck drive |
| `core/run2.py` | minimal script: regions + your own timecourses + mp4 |
| `core/play_fluid.py` | hand-tune the medium with the input held fixed |
| **pictures** | |
| `viz/render_frames.py` | video of a field a `best_fit` run already wrote to disk |
| `viz/surface_plots.py` | the movie / latent-map / medium-map helpers |
| `viz/render_regimes.py` | surface projections, videos of swept regimes |
| `viz/plot_fc_map.py`, `core/cortical_maps.py` | surface maps and the cortical map stack |
| `targets/get_msc.py` | reads MSC CIFTI-1 files (nibabel refuses these directly) |

## Running

```bash
python core/mesh_cache.py                          # build and cache the mesh
python targets/fc_group_nki.py                        # build the group FC target
python fit/best_fit.py --frames 4480 --draws 3    # solve, realise, score
python viz/render_frames.py --tag best --n 500    # watch what it produced
```

`best_fit.py --regions {sensory,spread,sensory+dmn,dmn}` selects which parcels
are driven; `PLAN.md` records what each of those returned and why the answer is
less obvious than it looks.

`core/run2.py` is the smaller entry point: set `REGIONS`, build an `(nsteps, K)`
array of timecourses, run, write an mp4.

Two things about the drive worth knowing, since neither is obvious from the code:

- **Timecourses are a source, not a depth.** `h += Aser[n] @ P` adds to depth
  every step, so the field follows the running integral of what you supply. A
  series whose integral never crosses zero (a sine started at zero) leaves that
  region one-signed for the whole run.
- **Regions are weighted by area.** Tapers peak at 1 per vertex regardless of
  parcel size, so a large parcel injects proportionally more. `drive.w` holds
  the area integral per region; `A -= np.outer(A @ w, w)/(w @ w)` removes any
  net injection. `input_model.NetworkDrive` does this internally as its
  `balance="spatial"` mode; `input2.RegionDrive` leaves it to the caller.

## Data

Included: the HCP-MMP1 left-hemisphere annotation (`data/annot`) and fsaverage5
/ fsaverage6 inflated left surfaces (`data/surf`). Both carry their upstream
licences.

Not included: MSC subject-01 resting-state scans (~2 GB, above GitHub's file
size limit). Only the MSC path in `targets/fc_vertexwise.py` needs them; the NKI group
target that everything is now fitted to is fetched by nilearn. Get them from [OpenNeuro ds000224](https://openneuro.org/datasets/ds000224),
`derivatives/surface_pipeline`, and put the `*_rest.dtseries.nii` and
`*_tmask.txt` files in `data/msc/`.

Caches under `data/cache/` and output under `results/` are regenerated on
demand and are not tracked.

## Requirements

numpy, scipy, scikit-learn, nibabel, nilearn, matplotlib, hcp_utils,
scikit-optimize (for `fit/bo_step.py`). ffmpeg for video.
