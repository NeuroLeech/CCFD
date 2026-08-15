# CCFD

Rotating shallow-water flow on a cortical surface, driven at named Glasser
parcels, compared against resting-state fMRI.

The surface is the left fsaverage5 inflated mesh with the medial wall cut out
(9,374 vertices). Fluid depth `h` and edge-normal velocity `u` evolve under the
linear rotating shallow-water equations, discretised with a C-grid / DEC scheme
whose Coriolis operator is energy-neutral by construction. Input is injected at
chosen parcels; the resulting spatiotemporal field is summarised by wave
measures and, optionally, scored against MSC resting-state data reduced to the
same 180 parcels.

## Layout

| file | what it does |
|---|---|
| `paths.py` | every path, derived from the repo location |
| `surf_ops.py` | surface loading, primal/dual (DEC) operators |
| `intrinsic_delaunay.py` | metric repair; without it the scheme blows up in ~200 steps |
| `mesh_cache.py` | builds and caches the `Cortex` object: mesh + atlas + repaired metric |
| `swe_rot.py` | the solver: `RotSWE.step`, plus the absorbing rim sponge |
| `input_model.py` | input as K regions driven through r shared latent factors |
| `input2.py` | input as K regions with timecourses supplied directly |
| `explore.py` | run a genome, compute descriptive measures |
| `run_input.py` | CLI: run one hand-specified input, print measures, render video |
| `run2.py` | minimal script: regions + your own timecourses + mp4 |
| `wave_measures.py` | anisotropy, transport speed, pattern timescale, coherence |
| `measures.py` | measure development against known-different runs |
| `genome.py` | what the GA searches over |
| `ga.py` | genetic algorithm over the input arrangement |
| `stage2.py` | parcel-space comparison with fMRI; similarity and richness |
| `get_msc.py` | reads MSC CIFTI-1 files (nibabel refuses these directly) |
| `sweep_fields.py` | fixed input, varied dynamical regime; saves full fields |
| `render_regimes.py` | videos of the swept regimes |
| `plot_axes.py` | measure-space plots |

## Running

```bash
python mesh_cache.py                       # build and cache the mesh
python run_input.py --find all             # list the 180 parcels
python run_input.py --regions V1,10r --video
```

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
size limit). `stage2.py`, `ga.py` and `run_input.py --fmri` need them. Get
them from [OpenNeuro ds000224](https://openneuro.org/datasets/ds000224),
`derivatives/surface_pipeline`, and put the `*_rest.dtseries.nii` and
`*_tmask.txt` files in `data/msc/`.

Caches under `data/cache/` and output under `results/` are regenerated on
demand and are not tracked.

## Requirements

numpy, scipy, nibabel, matplotlib, hcp_utils. ffmpeg for video.
