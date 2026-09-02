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

<video src="https://github.com/NeuroLeech/CCFD/raw/main/docs/best_field.mp4" controls muted loop playsinline width="900"></video>

[`docs/best_field.mp4`](docs/best_field.mp4) — the best fit to date. Depth `h` on the
left fsaverage5 inflated surface (lateral, medial, dorsal), with the six loudest drive
channels traced underneath and a cursor on the current frame. The colour scale is fixed
across the clip, so amplitude is comparable frame to frame rather than renormalised.
600 saved frames — 97 s of the 577 s realisation — at 20 fps, about 3.2x real time.

The run is the `Current best` of `PLAN.md`: 47 subcortically driven pieces over
8,542 mm², spread 6 mm/s, decay 9.03 s pinned to the NKI autocorrelation, scoring
**+0.7204 ± 0.0009** Spearman over 2M edges. To rebuild it:

```bash
python best_fit.py --oversample 4 --decay-s 9.03 --spread-mm-s 6 --bold-smooth \
  --pad 4096 --impulse-frames 224 --iters 400 --val-vert 0 --draws 2 \
  --regions subcortical --split 40 --profile taper
python render_frames.py --tag pr_taper --start 200 --n 600 --save 16 --fps 20
```

## Layout

| file | what it does |
|---|---|
| `paths.py` | every path, derived from the repo location |
| `surf_ops.py` | surface loading, primal/dual (DEC) operators |
| `intrinsic_delaunay.py` | metric repair; without it the scheme blows up in ~200 steps |
| `mesh_cache.py` | builds and caches the `Cortex` object: mesh + atlas + repaired metric |
| `swe_rot.py` | the solver: `RotSWE.step`, plus the absorbing rim sponge |
| `fluid.py` | the medium: speed and damping graded by cortical maps, integration |
| **the fit** | |
| `best_fit.py` | reproduce the current fit; the entry point |
| `xspec.py` | transfer function, the convex solve, realisation, scoring |
| `subparcels.py` | equal-area splitting; which parcels are driven |
| `bo_step.py` | Bayesian optimisation over the medium, in per-step units |
| `coalitions.py` | read the solved `S(f)` back as amplitudes and time offsets |
| **targets and controls** | |
| `fc_vertexwise.py` | vertexwise FC from MSC surface data |
| `fc_group_nki.py` | the NKI group target (99 usable subjects) |
| `fc_centre.py` | double-centring: the linear analogue of global signal regression |
| `fc_score.py` | `FCTarget`: alignment, edge sample, Spearman score |
| `fc_moran.py` | spatial autocorrelation match, as a diagnostic |
| `family.py` | the family of inputs consistent with the target, not just the argmax |
| `fc_states.py` | windowed-FC states, occupancy, dwell, transitions |
| `reliability.py` | split-half reliability of the target, and the ceiling it implies |
| `holdout.py` | solve on one half of the subjects, score on the other |
| `reach.py` | can this fluid produce the target's patterns at all? |
| `diag_maps.py` | where the fit fails, drawn on the surface |
| **input, by hand** | |
| `input2.py` | input as K regions with timecourses supplied directly |
| `input_model.py` | input as K regions driven through r shared latent factors |
| `ladder.py` | a nested family of input processes; parcel geodesics |
| `run_ou.py` | Ornstein-Uhlenbeck drive |
| `run2.py` | minimal script: regions + your own timecourses + mp4 |
| `play_fluid.py` | hand-tune the medium with the input held fixed |
| **pictures** | |
| `render_frames.py` | video of a field a `best_fit` run already wrote to disk |
| `surface_plots.py` | the movie / latent-map / medium-map helpers |
| `render_regimes.py` | surface projections, videos of swept regimes |
| `plot_fc_map.py`, `cortical_maps.py` | surface maps and the cortical map stack |
| `get_msc.py` | reads MSC CIFTI-1 files (nibabel refuses these directly) |

## Running

```bash
python mesh_cache.py                          # build and cache the mesh
python fc_group_nki.py                        # build the group FC target
python best_fit.py --frames 4480 --draws 3    # solve, realise, score
python render_frames.py --tag best --n 500    # watch what it produced
```

`best_fit.py --regions {sensory,spread,sensory+dmn,dmn}` selects which parcels
are driven; `PLAN.md` records what each of those returned and why the answer is
less obvious than it looks.

`run2.py` is the smaller entry point: set `REGIONS`, build an `(nsteps, K)`
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
size limit). Only the MSC path in `fc_vertexwise.py` needs them; the NKI group
target that everything is now fitted to is fetched by nilearn. Get them from [OpenNeuro ds000224](https://openneuro.org/datasets/ds000224),
`derivatives/surface_pipeline`, and put the `*_rest.dtseries.nii` and
`*_tmask.txt` files in `data/msc/`.

Caches under `data/cache/` and output under `results/` are regenerated on
demand and are not tracked.

## Requirements

numpy, scipy, scikit-learn, nibabel, nilearn, matplotlib, hcp_utils,
scikit-optimize (for `bo_step.py`). ffmpeg for video.
