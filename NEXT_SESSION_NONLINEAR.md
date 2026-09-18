# Next session: nonlinear media on the ascending 100-region basis

## The model to fit

```
python fit/torch_fit.py --ref asc_first_area_100 \
  --init cold --seconds 577 --iters 400 --lr 0.05 --seed 0 \
  --nl-flux 1 --nl-adv 1 --amp 6e-3 \
  --learn-maps 3 --map-lr 0.02 \
  --lag-taus 1,3,5 --wa 0.1 \
  --eval-every 50 --eval-draws 6 --dump 0,100,200,400 --tag <name>
```

Four things, all required together:

- **the 100-region ascending basis**, `results/xspec_asc_first_area_100.npz`. NOT
  `torchref`. torchref is the old 47-piece `subcortical` set - 17 Glasser parcels from
  AlternateListOfInputs.json - which `ascending_basis.py` superseded and 14 of whose
  pieces (areas 4, 6a, 6mp) are the cerebellar/pallidal relay, not ascending at all.
- **a nonlinear medium**. `--nl-flux` puts (H+h) in the depth update; `--nl-adv` adds
  momentum advection in vector-invariant form. The point of the torch solver is media the
  convex solve cannot express. A linear run is a slower way to get an answer `fit/xspec.py`
  already gives in closed form.
- **the map weights in the optimiser**, `--learn-maps 3`: the speed and damping
  coefficients over myelin/thickness/sulc become free, bounded to a 3x multiplier on
  DEPTH (see below).
- **the lag term at wa 0.1**, `--lag-taus 1,3,5 --wa 0.1`.

## Amplitude

Measured on the ascending basis with the medium free, cold drive, 577 s:

| amp | peak h/H | |
|---|---|---|
| x15 (3e-3) | 0.529 | fine |
| x30 (6e-3) | 1.059 | fine |
| x50 (1e-2) | - | diverges |

Same ceiling for flux alone, flux+advection, and flux+advection+rotation. x30 is the
working point; the guard skips non-finite updates and reports how often.

Divergence is the bed strike, (H + h) < 0, not CFL. It happens first at the thinnest
point, and the fixed medium's H spans 33x (0.119-3.914) with its thinnest point 9.6x below
the mean - which is why --learn-maps matters for this: bounded at 3x on depth it comes out
near 0.61-3.91, and the amplitude ceiling rises 3-5x with it.

## What is already built and checked

`core/torch_swe.py` - the stepper. float64 matches the numpy one to 3.9e-16, gradients
match central differences to 1e-10 with either nonlinearity, checkpointing is bitwise
exact, MPS matches CPU to 5e-7. Advection is verified separately: vorticity of a discrete
gradient 1.9e-16, the rotation term still exactly energy-neutral with zeta folded into f
(4.6e-20), and the departure from linear quarters when the amplitude halves, twice.

`core/mesh_cache.py` carries `F2`, the intrinsic-Delaunay faces the solver actually
integrates - advection needs them and `cortex.m` is the wrong triangulation.

`fit/torch_fit.py` - the fit. Evaluation routes through the torch integrator whenever a
nonlinearity or a learned medium is on, because the numpy solver has neither and a learned
grading is not expressible as map coefficients at all. Checkpoints store the whole medium.

`targets/lagged.py:empirical_rbc` - the lagged target from RBC (bandpassed), cached. The
1000 solve vertices are identical between torchref and asc_first_area_100, so the cache
applies to both.

`analysis/report.py` - collects one fit into `results/<tag>/`: parameters, both videos,
edge and parcel scatters, input regions (flat shaded) and per-piece drive, interference and
pieces-per-vertex, drive share by thalamic nucleus, energy-flux arrows, the three maps and
the speed/damping they make, and the first five FC gradients against the empirical ones.
Runs on any tag; rebuilds whatever medium the run used.

## Reference numbers on this basis

The convex solve on `asc_first_area_100`: **+0.6964 +- 0.0039** at 2308 s,
**+0.6238 +- 0.0217** at 577 s. That is what a fit on this basis is measured against.

## How to work

- **Rank configurations at 50 iterations.** 50-iteration runs took ~25 min and gave the
  whole wa trade-off curve; the 400-iteration versions cost 3.5 h each and changed no
  conclusion. Spend 400 iterations only on the model being reported.
- **Check the amplitude ceiling before launching**, with the cold drive and the medium the
  fit will start from. A ceiling measured on the convex drive is not the same number.
- **Compare like with like.** The saved frames are BOLD-kernel-smoothed; a raw field
  against them correlates ~0.95 however right the decomposition is.
