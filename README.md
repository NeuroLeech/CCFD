# CCFD

Rotating shallow-water flow on a cortical surface, driven at the cortical
territory of the primary ascending pathways, compared against resting-state fMRI.

The surface is the left fsaverage5 inflated mesh with the medial wall cut out
(9,374 vertices). Fluid depth `h` and edge-normal velocity `u` evolve under the
rotating shallow-water equations, discretised with a C-grid / DEC scheme whose
Coriolis operator is energy-neutral by construction.

## Two stages: solve linear, then descend into nonlinear

**The linear solve is convex and has an optimum.** With a linear medium the field
covariance depends on the drive only through its cross-spectral density `S(f)`, so
matching functional connectivity is a convex problem over one Hermitian PSD matrix per
frequency — an optimum rather than a plateau. The pipeline is: impulse response per
driven piece → `H(f)` by FFT → solve `max corr(sum_f H S H^H, target)` over `S(f) >= 0`
→ draw a drive with that cross-spectrum → simulate → score. That is `fit/best_fit.py`
and `fit/xspec.py`.

**A nonlinear medium destroys that.** Put `(H + h)` in the depth update, or add momentum
advection, and there is no transfer function to factorise: the covariance no longer
splits into `H S H^H` and the problem stops being convex. The only route left is to
generate a drive, simulate it, measure the covariance of what comes out, and
differentiate through the whole realisation. That is `core/torch_swe.py`, the stepper
ported to torch, and `fit/torch_fit.py`, which descends on the same object the convex
solve produces — `S = G G^H`, with `G` the free variable — so the convex solution is
literally the warm start and the two are comparable element for element.

So the working order is: **solve the linear model in closed form, then warm-start a torch
fit and turn the nonlinearity on.** The medium's map coefficients can be freed at the same
time, bounded to the box the numpy search used, so the optimiser cannot walk them
somewhere the anatomy does not go.

Scoring is against a 100-subject NKI group FC from ReproBrainChart (fMRIPrep + XCP-D,
denoised and bandpassed to 0.01–0.08 Hz, resampled to fsaverage5, 9,310 vertices),
double-centred, by Spearman over a fixed 2M edge sample. A solve correlation is not a
score: every candidate is realised and simulated before any number is quoted, and the
realisation length matters — the same solve scores **+0.6799** over 2,308 s and
**+0.6339** over 577 s.

## What it looks like

![the fluid and the observable](docs/nonlinear_pair.gif)

The loop is 10 s of a 3,578-frame realisation, depth `h` on the left fsaverage5 inflated
surface, lateral / medial / dorsal; the full 30 s at 20 fps is
[`docs/nonlinear_pair.mp4`](docs/nonlinear_pair.mp4). **Top row is the fluid itself**: no BOLD kernel, no passband,
nothing applied. **Bottom row is the observable** — the same realisation BOLD-smoothed and
then bandpassed to 0.01–0.08 Hz, which is what the model is scored on and what a scanner
would report. Same drive, same medium, same frames; only the observable differs. Colour
scales are per row and printed, because the observable is 0.29x the fluid's amplitude and
a shared scale would leave the bottom row nearly blank.

The fluid carries **14.2%** of its power in the 0.01–0.08 Hz band the observable keeps.

This is a rotating, advecting, nonlinear medium — `(H + h)` in the depth update,
momentum advection in vector-invariant form, and a 113 s inertial period against a 25 s
decay. It is warm-started from the convex solve in the same medium and fitted by gradient
descent through the simulation, which is what a nonlinearity requires: there is no `H` to
factorise, so the cross-spectrum is no longer a convex problem. 200 iterations, converged
(the last 25 gained 0.0001), no bed strikes. Scored **+0.6818 ± 0.0041** over 2,308 s,
against the linear solve's **+0.6799 ± 0.0035** in the same medium.

```bash
python fit/best_fit.py --oversample 4 --decay-s 25 --spread-mm-s 1.5 --impulse-decays 7 \
  --bandpass 0.01,0.08 --pad 4096 --nfreq 192 --bold-smooth --regions hybrid \
  --hybrid-file results/xspec_asc_first_area_100.npz --split 50 --maps-scale 1 \
  --map-clip iqr --seconds 2308 --iters 400 --draws 2 --val-vert 0 --tag grclip100_nolag
python fit/torch_fit.py --ref grclip100_nolag --init warm --seconds 577 --iters 200 \
  --coef-lim 0.45 --nl-flux 1 --nl-adv 1 --ld 100 --amp 3e-3 --tag nl100_ld100_200
python fit/render_fit.py --tag nl100_ld100_200 --ref grclip100_nolag
python viz/render_bandpass.py --tag nl100_ld100_200v \
  --raw-npy results/frames_nl100_ld100_200v_raw.npy --start 200 --n 600 --save 4
```

### The linear fit it starts from

![the linear solve, fluid and observable](docs/linear_pair.gif)

The nonlinear run above is warm-started from a convex solve in the **same medium**, and
that solve is what the loop above shows — same two rows, same 577 s realisation, same
window. It scores **+0.6799 ± 0.0035** over 2,308 s, against the nonlinear fit's
**+0.6818 ± 0.0041**. Its fluid carries **11.9%** of its power in band where the
nonlinear one carries 14.2%, and it runs about a ninth of the amplitude, since the
convex drive sits at x1 where the nonlinear fit is driven at x15.

The loop is cut to 10 s at 10 fps; the full 30 s clip lands in `results/`, which is
gitignored, so it is rebuilt rather than fetched:

```bash
python fit/rescore.py --tag grclip100_nolag --seconds 577 --draws 1 --save-frames
python viz/render_bandpass.py --tag grclip100_nolag_577s \
  --raw-npy results/frames_grclip100_nolag_577s_raw.npy --start 200 --n 600 --save 4
```

[`docs/best_field.mp4`](docs/best_field.mp4) is an older single-row surface clip of
`pr_taper` — a different basis and clock, 47 subcortically driven pieces over 8,542 mm²
at 6 mm/s and a 9.03 s decay, scoring **+0.7204 ± 0.0009**:

```bash
python fit/best_fit.py --oversample 4 --decay-s 9.03 --spread-mm-s 6 --bold-smooth \
  --pad 4096 --impulse-frames 224 --iters 400 --val-vert 0 --draws 2 \
  --regions subcortical --split 40 --profile taper
python viz/render_frames.py --tag pr_taper --start 200 --n 600 --save 16 --fps 20
```

`RUNS.md` indexes every solve and fit on disk with its parameters and its score.

## The 100 driven regions

The drive is not spread over the whole cortex and it is not a parcellation chosen for
convenience. The territory is the cortical footprint of the **primary ascending
pathways**, taken from the anatomical literature: whole Glasser parcels grouped by the
pathway that drives them, with motor excluded, because VA/VL is the cerebellar and
pallidal relay rather than an ascending sensory route.

| group | parcels |
|---|---|
| LGN → V1 | `V1` |
| MGv → A1 | `A1` |
| VPL/VPM → S1 | `3a` `3b` `1` `2` |
| VPI/VMpo → S2, insula | `OP1` `OP2-3` `OP4` `Ig` `PoI1` `PoI2` `PI` |
| VPMpc → taste | `AAIC` `AVI` |
| olfactory (non-thalamic) | `Pir` |
| AV/AM → retrosplenial | `RSC` `ProS` `PreS` |
| amygdalocortical | `OFC` `pOFC` `25` `s32` `EC` `PeEc` `TGv` `TGd` |

27 parcels, **9,409 mm², 17.3% of cortex, 1,880 driven vertices**. These are the
*first-order* relays — nuclei whose driving afferents come from a subcortical source —
plus the two ascending routes that do not pass through thalamus at all. `fit/ascending_basis.py`
also carries the higher-order set (pulvinar, MGdm, MD, LD), whose driving input is
cortical layer 5 rather than ascending, and which is not used here.

Inside that territory the **channels** are cut by anatomy, not by a grid. Sereno
CsurfMaps1 areas take their natural extent, so `V1` splits into lower and upper visual
quadrant and the postcentral strip into face, hand and foot; territory with no CsurfMaps1
area falls back to its Glasser parcel; the amygdalocortical group is never subdivided.
That gives 38 tiles, which Fiedler-vector bisection cuts to **100**, always splitting the
largest-area piece next, with a 12-vertex floor. Per channel: median area 83 mm², median
18 vertices, median diameter 13.5 mm. Profiles are smoothstep tapers, 1 at each piece's
core and 0 at its border.

```bash
python fit/ascending_basis.py tiles --set first --out tiles_first.npz
python fit/ascending_basis.py subdivide --in tiles_first.npz --by area --count 100 \
  --out sub_first_area_100.npz
```

`best_fit.py --regions hybrid --hybrid-file <that npz>` reads it. The other `--regions`
choices — `sensory`, `dmn`, `spread`, `subcortical`, `ascending` — are earlier bases kept
for comparison; `RUNS.md` records what each returned.

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
| `fit/` | the input model, the convex solve and the torch descent — `xspec`, `best_fit`, `torch_fit`, `rescore`, `ascending_basis`, `bo_step`, `subparcels`, `connectome`, `family` |
| `analysis/` | diagnostics and experiments — `diag_*`, `band_*`, `interference`, `zones`, `checkerboard_model`, `checkerboard_modulation`, `onoff_*`, `task_inputs` |
| `viz/` | `render_*`, `plot_*`, `surface_plots` |

`data/` holds the datasets and `results/` everything written; both are addressed through
`core/paths.py`, which derives them from the repository root rather than from a working
directory.

| file | what it does |
|---|---|
| `core/paths.py` | every path, derived from the repo location |
| `core/provenance.py` | argv, the full namespace and the commit, saved into every checkpoint |
| `analysis/runs.py` | regenerate `RUNS.md`, the index of every solve and fit on disk |
| `core/surf_ops.py` | surface loading, primal/dual (DEC) operators |
| `core/intrinsic_delaunay.py` | metric repair; without it the scheme blows up in ~200 steps |
| `core/mesh_cache.py` | builds and caches the `Cortex` object: mesh + atlas + repaired metric |
| `core/swe_rot.py` | the solver: `RotSWE.step`, plus the absorbing rim sponge |
| `core/torch_swe.py` | the same stepper in torch, differentiable, with `(H+h)` flux and momentum advection |
| `core/fluid.py` | the medium: speed and damping graded by cortical maps, integration |
| `core/cortical_maps.py` | the map stack, and `clip_maps` — trimming the tails that otherwise set the medium's whole range |
| **the fit** | |
| `fit/best_fit.py` | the convex solve end to end; the entry point |
| `fit/xspec.py` | transfer function, the convex solve, realisation, scoring |
| `fit/torch_fit.py` | gradient descent through the simulation, for media the convex solve cannot express |
| `fit/rescore.py` | replay a saved solve at another realisation length, without re-solving |
| `fit/render_fit.py` | realise a torch fit into frames, filtered and bare, for the videos |
| `fit/ascending_basis.py` | build the ascending-pathway input basis, four ways |
| `targets/lagged.py` | lagged covariance targets; `xspec.solve_lagged` matches their antisymmetric part |
| `fit/subparcels.py` | equal-area splitting; which parcels are driven |
| `fit/bo_step.py` | Bayesian optimisation over the medium, in per-step units |
| `fit/coalitions.py` | read the solved `S(f)` back as amplitudes and time offsets |
| **targets and controls** | |
| `targets/fc_vertexwise.py` | vertexwise FC from MSC surface data |
| `targets/fc_group_rbc.py` | the RBC NKI group target, XCP-D denoised and bandpassed — what everything is scored against |
| `targets/fc_group_nki.py` | the older raw-NKI group target, unfiltered |
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
python core/mesh_cache.py                        # build and cache the mesh
python targets/fc_group_rbc.py --cohort 100      # build the group FC target
python fit/best_fit.py --oversample 4 --seconds 2308 ...   # solve, realise, score
python fit/torch_fit.py --ref <tag> --init warm ...        # then descend, nonlinear
python fit/rescore.py --tag <tag> --seconds 2308           # rescore without re-solving
python analysis/runs.py                          # rewrite RUNS.md
```

Two things worth knowing about scoring:

- **`--seconds` never touches the solve.** `S(f)` is fitted in frequency space against
  `H`; the realisation length only enters afterwards, when a drive is drawn and
  integrated. So a solve scored at two lengths does not need repeating — `fit/rescore.py`
  replays it, and `--torch-ref` does the same for a torch fit through the integrator it
  was fitted with. It is worth 0.07 of Spearman between 577 s and 2,308 s, so a table
  mixing lengths is not comparing models.
- **`RUNS.md` indexes every solve and fit on disk** with its parameters, its realised
  score at each length, and its command line. It is generated from the npz files by
  `python analysis/runs.py` rather than kept by hand, so it cannot drift from them.

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

numpy, scipy, scikit-learn, nibabel, nilearn, matplotlib, hcp_utils.
**torch** for `fit/torch_fit.py` and `core/torch_swe.py` — the nonlinear stage; it runs on
MPS, CUDA or CPU. scikit-optimize for `fit/bo_step.py`, neuromaps for the cortical maps.
ffmpeg for video.
