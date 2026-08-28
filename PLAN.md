# CCFD — plan for the next session

Fitting a rotating shallow-water field on the cortical surface to group resting-state FC,
by solving for the input rather than searching for it.

## Where things stand

**Best fit: +0.6276 ± 0.0005** (47 sensory pieces, 4,480 frames, 3 draws), gap 0.058,
field rank 53.6:

```bash
python best_fit.py --select-iters 10,25,50,100,200 --frames 4480 --draws 3
```

### The target was centred and the model was not

The FC file the target loads from is `...spearmandcfc.npy` - already double-centred by
`fc_centre.py` - and `FCTarget` was constructed with `centre="none"`. So the global
component had been removed from the target and left in the model. `fc_score` has always
had the symmetric path (`centre="double"` double-centres the model too, exactly, on the
sampled-edge path via a dot-product identity); it was simply never switched on.
`fc_score.default_target` now loads the raw matrix with `centre="double"`, and `best_fit`,
`bo_step`, `holdout` and `reliability` all use it.

Double-centring the raw matrix reproduces the pre-centred file to **0.000e+00**, so the
target is the same object either way and every medoid cache, subject-half and edge sample
stays valid. Only the model side moves.

Worth +0.02 of score on identical frames (+0.6165 -> +0.6380). What it changes more is the
diagnostics, which had been badly misleading:

| edge length | accuracy asym | accuracy sym | target mean FC | model mean FC, sym |
|---|---|---|---|---|
| 0-10 mm | +0.569 | +0.684 | +0.543 | +0.644 |
| 10-20 | +0.191 | +0.356 | +0.252 | +0.367 |
| 20-30 | +0.206 | +0.376 | +0.135 | +0.203 |
| 30-40 | +0.325 | +0.484 | +0.075 | +0.112 |
| 40-60 | +0.524 | +0.628 | +0.030 | +0.047 |
| 80-100 | +0.579 | +0.596 | -0.007 | -0.012 |
| 160-250 | +0.435 | +0.436 | -0.028 | -0.041 |

**"The model cannot produce anticorrelation" was an artefact of the asymmetry.** Centred
symmetrically it crosses zero at the same distance as the target and is 56.4% negative
against the target's 60.9%, minimum -0.338 against -0.379. The modality heat plot showing
the sensory systems failing to anticorrelate has the same flaw and should be discarded.

### The stopping point was a hidden parameter, and it was set wrong

The convex solve's objective is anti-correlated with the score past about 25 steps. A
4,000-step probe: the objective goes 0.2428 (white input) to 0.6903 by step 25 to 0.7298
at step 4,000, still gaining 5e-05 per 100 steps and **never once stalling** - there is no
convergence criterion in `xspec.solve`, only an iteration cap. The first 25 steps buy 92%
of the total objective gain; the remaining 8% costs 0.11 of realised score.

Sensory 47 pieces, 4,480 frames, nothing else varied:

| solver iterations | solve rho | sim | gap | field rank |
|---|---|---|---|---|
| 10 | 0.6349 | +0.5863 ± 0.0017 | 0.063 | 48.1 |
| 25 | 0.6750 | +0.6158 ± 0.0000 | 0.058 | 53.6 |
| **50** | 0.6894 | **+0.6143 ± 0.0021** | 0.048 | 55.1 |
| 100 | 0.7016 | +0.6031 ± 0.0052 | 0.048 | 51.9 |
| 150 (the old default) | 0.7077 | +0.5875 ± 0.0011 | 0.043 | 46.5 |
| 300 | 0.7135 | +0.5569 ± 0.0085 | 0.049 | 36.9 |
| 600 | 0.7164 | +0.5307 ± 0.0080 | 0.052 | 31.3 |

Solve rho climbs monotonically; the score peaks around 25-50 and falls monotonically after.
Field rank tracks the score (48, 54, 55, 52, 46, 37, 31) - the diagnostic was in the output
all along and was never used to decide when to stop.

**Two effects compound, which is why the obvious fix is not enough.** Measured on the 1,000
solve vertices, the SIMULATED FC against the target improves monotonically with iterations
(+0.6053 at 10 steps to +0.6747 at 600) - so on the vertices it fits, more solving is
strictly better. The loss is entirely off them. On top of that the simulation reproduces
its own prediction less faithfully as the input collapses to fewer modes: Spearman between
realised and predicted FC falls 0.9452 to 0.9271 across the sweep (`fidelity.py`).

So a criterion has to see both, and only a realised, held-out number does:

| criterion | stops at | resulting sim |
|---|---|---|
| fixed count (what everything used) | 150 | +0.5875 |
| held-out COVARIANCE match, inside the solve | 130 | +0.5947 |
| held-out REALISED score (`--select-iters`) | 50 | **+0.6135** |

Cheap in-solve proxies do not work: the effective rank of the predicted `C(S)` rises
monotonically (57.6 to 101.1) and the mean rank of `S(f)` falls monotonically (7.84 to
6.04). Neither turns over where the score does. `--select-iters` therefore solves and
simulates once per candidate at 1,120 frames, scores on 1,000 vertices the solve never
saw, and picks - about 10 minutes per configuration, and the selection curve is a clean
inverted U (+0.5473, +0.5856, +0.5871, +0.5839, +0.5482 over 10/25/50/100/200).

### What this reaches backwards into

Solve correlation has been the main progress signal in this project and it points the
wrong way. Every "the solve reaches 0.71, we realise 0.57, where does the 0.14 go" framing
had the gap backwards: the solve was not a ceiling being approached but a different
objective being pursued past the point where it helped.

**Every result below was taken at 150 iterations and none has been re-run.** Configurations
with more channels converge more slowly at a fixed count, and being less converged now
looks like an advantage, so the coverage table in particular is confounded - the spread
advantage may be partly an artefact of that, a mechanism independent of dispersion or of
degrees of freedom. Treat every number below as "at 150 iterations", not as a property of
the configuration.

### The model

Sensory is the constrained, falsifiable model - the dominant patterned input to cortex
enters through sensory areas. `--regions spread` relaxes that and scored far higher
(+0.7653 at 100 pieces, at 150 iterations), but 24 farthest-point parcels including pOFC
and TGd are the input prior switched off rather than an alternative hypothesis about it.

**The pipeline.** The medium is linear, so the FC depends on the input only through its
cross-spectral density `S(f)`, which is solved convexly:

1. impulse response per driven piece, FFT along time -> `H(f)`
2. solve `max corr(sum_f H S H^H, target)` over `S(f) >= 0` by projected gradient
3. realise a drive by drawing `z_f = L_f eta` per frequency, `S = L L^H`; inverse FFT
4. simulate, score with Spearman over a fixed 2M edge sample

**Current configuration** (all in `best_fit.py:BEST_X`): 47 equal-area sensory pieces of
~145 mm2; per-step damping 6.2e-04, rotation 1.1e-05, boundary absorption 1.8e-03, cadence
33 steps/frame; impulse window 280 frames zero-padded to 1,120, 118 usable frequencies;
solved against the normal-scored target on 1,000 medoid vertices.

## What the diagnostics say the problem is

Two things survive the centring fix, from `diag_distance.py` and `diag_edges.py` on 1,000
random vertices.

**The failure is not spatial.** Per-vertex accuracy runs +0.62 near the drive to +0.54 at
120 mm, and correlates with distance to the nearest driven vertex at only **-0.175**. It is
+0.634 at the driven vertices themselves, against +0.604 overall - the model is barely
better at the regions it injects into directly. Target |FC| is flat-to-rising with
distance, so there is signal to predict everywhere. Whatever is missing is missing
uniformly. This also undercuts the transport-range reading of the coverage result: if
reach were the problem, accuracy would fall off sharply with distance, and it does not.

**Most of the overall score is the distance trend.** Within an edge-length bin, where the
decay is removed, accuracy is far lower than the +0.630 overall - down to +0.356 at
10-20 mm and +0.376 at 20-30 mm, against +0.684 for the shortest edges. A diffusing wave
medium gets the shape of the distance decay nearly for free; the mid-range, where FC has
structure that is not just decay, is where it is weakest.

**And the model over-correlates by a roughly constant factor at every distance**: 0.644
against 0.543, 0.367 against 0.252, 0.203 against 0.135, 0.112 against 0.075, -0.041
against -0.028. The shape of the distance profile is right and the amplitude is stretched
by about 1.3-1.5x throughout. That is consistent with the field expressing its FC through
too few spatial modes - effective rank ~54 over 9,217 vertices - so vertices share more
variance than they should.

## Priority experiments

1. **Mode count.** Rank ~54 is the recurring number: it tracks the score across the
   iteration sweep, and it is the natural explanation for uniform over-correlation. How
   well can ANY rank-54 approximation of the target score? That bounds what the current
   architecture can reach without changing the mode repertoire, and it is a few lines
   against the target's eigendecomposition.
2. **Mid-range structure.** 10-40 mm is the weak band. Whether that is the medium's
   characteristic scale, the piece size (~145 mm2, about 14 mm across), or the taper
   profile is testable by varying the piece size at fixed driven area.
3. `reach.py` on the current medium, as before.

## Things to keep reporting

Every result: multi-draw mean **and** scatter; solve correlation (Pearson and Spearman
against the raw target); Moran gap; field rank; and the realisation length, which is not a
detail. Quote 4,480 frames rather than 1,120 unless there is a reason not to.

## Theory

**Coalitions do not respect geometry after all.** For the 19-parcel fit, coalition time
offsets correlated with geodesic distance at r = 0.45 — the closest thing to a testable
claim about the input. It does not survive the finer split:

| solution | pieces | offset r vs geodesic distance |
|---|---|---|
| 19 whole parcels | 19 | +0.39 to +0.45 |
| sensory | 47 | +0.01 to +0.28 |
| spread | 51 | +0.03 to +0.17 |
| sensory+DMN | 63 | +0.03 to +0.28 |

Amplitudes never correlated, on any of them. This is not a coverage effect — spread and
sensory+DMN both span the cortex — it is granularity: with only 19 channels the medium's
own travel times pin the phases, and with 50+ the solve has enough freedom to put them
anywhere. Read the r = 0.45 as a property of a coarse parcellation, not of the fit.

Note `coalitions.py` was comparing offsets linearly, when an offset is a phase over a
frequency and only means anything modulo the period. It now uses the wrapped phase
difference. The fix does move numbers (one band went 0.00 → 0.28) but changes no
conclusion: the 19-parcel result survives it, the fine ones still fail it.

**Why is the Moran gap so easy now?** Still unanswered, and now more striking: it tracks
coverage almost perfectly (0.064 at 16 pieces down to 0.015 at 100), with no penalty
applied. Driving more of the sheet apparently makes the spatial autocorrelation come out
right for free.

**Dynamics.** Unchanged: static FC is a second moment, so a linear medium makes it depend
only on `S(f)`, and waveform, burstiness and regime-switching are invisible to it. Any
payoff from temporal structure needs a lagged or dynamic term in the objective — FCD
distribution, or occupancy and dwell-time match against the states in
`data/cache/fc_states_k5_w60_s10_150.npz`.

**Rotation still wants to be near zero.** Unchanged, and the re-run BO will say whether
that survives the new target and region set.

## Code map

| file | role |
|---|---|
| `best_fit.py` | reproduce the current best; `--regions` picks the driven set |
| `xspec.py` | transfer function, convex solve, `normal_scores`, realisation, scoring |
| `subparcels.py` | equal-area splitting; `region_set`, `spread_sample` |
| `holdout.py` | subject-split control: solve on one half, score on the other |
| `regimes.py` | a medium that switches between regimes inside one run |
| `connectome.py` | long-range structural coupling; the ENIGMA HCP connectome |
| `units.py` | approximate mm and seconds; the temporal filter |
| `reliability.py` | split-half reliability of the target, and the ceiling it implies |
| `fluid.py` | medium: speed/damping fields, map grading, integration |
| `bo_step.py` | BO over the medium in per-step units; checkpoints, `--resume` |
| `coalitions.py` | read `S` back as coalitions; `--npz` reads a `best_fit` solution |
| `fc_score.py` | `FCTarget`: alignment, edge sample, Spearman score, rank/Moran terms |
| `fc_group_nki.py` | build the NKI group target (99 usable subjects) |
| `fc_states.py` | windowed-FC states, occupancy, dwell, transitions |
| `fc_moran.py`, `fc_vertexwise.py`, `cortical_maps.py` | diagnostics, MSC path, maps |
| `render_frames.py`, `surface_plots.py` | video of a saved run; the plotting helpers |
| `plot_fc_map.py`, `play_fluid.py` | surface maps, manual sweeps |
