# CCFD — plan for the next session

Fitting a rotating shallow-water field on the cortical surface to group resting-state FC,
by solving for the input rather than searching for it.

## Where things stand

**Read this before any other number in this file.** The convex solve's objective is
ANTI-CORRELATED with the score it is supposed to serve, past about 25 projected-gradient
steps. Every result recorded here was taken at `--iters 150`, which sits well down the
descending limb, and different configurations converge at different rates - so the
comparisons between them are confounded by how far each was solved, not only by what was
varied.

Sensory 47 pieces, 4,480 frames, everything else at the incumbent configuration:

| solver iterations | solve rho | sim | gap | field rank |
|---|---|---|---|---|
| 10 | 0.6349 | +0.5863 ± 0.0017 | 0.063 | 48.1 |
| **25** | 0.6750 | **+0.6158 ± 0.0000** | 0.058 | 53.6 |
| 50 | 0.6894 | +0.6143 ± 0.0021 | 0.048 | 55.1 |
| 100 | 0.7016 | +0.6031 ± 0.0052 | 0.048 | 51.9 |
| 150 (what everything below used) | 0.7077 | +0.5875 ± 0.0011 | 0.043 | 46.5 |
| 300 | 0.7135 | +0.5569 ± 0.0085 | 0.049 | 36.9 |
| 600 | 0.7164 | +0.5307 ± 0.0080 | 0.052 | 31.3 |

Solve rho climbs monotonically; the realised score peaks at 25 steps and falls
monotonically after. A 4,000-step probe shows where the objective actually goes: 0.2428 at
the white-input start, 0.6903 by step 25, 0.7298 at step 4,000, still gaining 5e-05 per
100 steps and never once stalling - there is no convergence criterion, only an iteration
cap. **The first 25 steps buy 92% of the total objective gain.** The remaining 8% is what
costs 0.11 of realised score. **Field rank tracks the score almost exactly** (48, 54, 55, 52, 46,
37, 31): as the solve converges it concentrates the input into fewer modes, and the
low-rank solution generalises worse from the 1,000 solve vertices to all 9,217. The
diagnostic was in the output all along; it was never used to decide when to stop.

Three consequences.

`--iters 150` was doing implicit regularisation, badly tuned. The current configuration
reaches **+0.6158 at 25 iterations**, +0.028 above the long-standing headline, for a sixth
of the compute.

The solve correlation has been the project's main progress signal and it points the wrong
way. Every "the solve reaches 0.71 and we realise 0.57, where does the 0.14 go" framing
had it backwards: the solve was not a ceiling being approached but a different objective
being pursued past the point where it helped.

**The coverage table is confounded.** More channels converge more slowly, so 150 steps is
a different degree of convergence at K=47 than at K=100 - and being less converged is now
known to help. The spread advantage may be partly an artefact of that, which is a
mechanism independent of dispersion or of degrees of freedom.

**Nothing below has been re-run at a defensible stopping rule.** Treat every number in the
rest of this file as "at 150 iterations", not as a property of the configuration.

### The model, as last measured

Sensory 47 pieces at 150 iterations: **+0.5875 ± 0.0011**, gap 0.046, rank 46.5.

```bash
python best_fit.py --frames 4480 --draws 3            # as recorded below
python best_fit.py --frames 4480 --draws 3 --iters 25 # +0.6158
```

Sensory is the constrained, falsifiable model - the dominant patterned input to cortex
enters through sensory areas. `--regions spread` relaxes that and scores far higher
(+0.7653 at 100 pieces), but 24 farthest-point parcels including pOFC and TGd are the
input prior switched off rather than an alternative hypothesis about it. Read that gap as
an upper bound on the machinery with nothing constraining where the drive is applied.

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

## Priority experiments

0. **Switching, resolved.** Simplified to one map on speed alone (`--regime-map sulc
   --regime-target speed`), the epochs are quasi-static (within-epoch ratio 1.2, against
   23-25x for the first four-things-at-once construction) and R=3 matches R=1 at each
   one's peak: +0.6103 ± 0.0114 against +0.6158, inside R=3's own scatter. R=3 also
   reaches a strictly higher solve objective at every matched iteration count (0.7725
   against 0.7164 at 600), so the larger model class is real and the original comparison
   was measuring convergence. Switching is now NEUTRAL rather than harmful, and needs a
   reason to help rather than a bug to fix.

   The transient was the damping variation, not the speed jump. Speed is the H whose
   discontinuity was blamed for it, and speed-only switching has no transient; damping
   sets the field's stationary amplitude, so switching damping forces a re-equilibration
   at every boundary. Damping regimes therefore need a ramped switch before they can be
   tested - and damping is the more plausible physiological modulation of the two.

1. **Fix the stopping rule, then re-run everything.** Options: early stopping selected on
   held-out vertices (the solve uses 1,000 medoids, 8,217 are unused and free); an explicit
   rank floor or nuclear-norm penalty, since rank collapse is the visible mechanism; or
   solving on far more vertices, which `--nvert 2500` tested at 150 iterations - on the
   descending limb, so that null needs re-running. Until this is settled, comparisons
   between configurations are comparisons of convergence.
2. **Re-run the coverage table** at whatever rule comes out of 1, since that result carries
   the reading of where input enters and is the one most exposed to the confound.
3. `reach.py` on the current medium.

## What last session's five experiments returned

### 1. Longer realisations — real, but not monotone

Sensory 47 pieces, solve ρ = 0.7077 throughout, so only the realisation changes:

| frames | sim | sd | gap | rank |
|---|---|---|---|---|
| 1,120 | +0.5661 | 0.0198 | 0.043 | 45.6 |
| 1,680 | +0.5477 | 0.0143 | 0.064 | 37.9 |
| 2,240 | +0.5467 | 0.0056 | 0.062 | 38.3 |
| 3,360 | +0.5806 | 0.0034 | 0.054 | 44.7 |
| 4,480 | +0.5875 | 0.0011 | 0.046 | 46.5 |

It dips before it climbs, and the scatter falls twentyfold, so the dip is real rather than
draw luck. The likely cause is that 1,120 is the length `S` was solved at: only there does
the drawn drive land exactly on the solved frequency bins, and `xspec.realise` has to
interpolate everywhere else. 1,120 gets a fidelity bonus that intermediate lengths lose
before extra length pays it back. **Anything quoted at 1,120 frames is therefore quoted at
a favourable point** — worth checking whether `realise` should draw on the solved grid and
tile, rather than interpolate.

Spread at 4,480 → 8,960 goes +0.6972 → +0.7020, so realisation has converged by ~4,480.

### 2. Iterated rank solve — closed, it fails

The fixed point walks away from the target, monotonically, and does not oscillate:
solve ρ 0.7077 → 0.6898 → 0.6770 → 0.6666 → 0.6597 over four iterations, and the realised
score falls to +0.4837 ± 0.0054. Interestingly the field gets *better* by the other two
diagnostics (gap 0.019, rank 109) while scoring far worse. Do not revisit without a
different scheme.

### 3. More vertices in the solve — no gain

2,500 medoids, 120 iterations: realised +0.5747 ± 0.0261 against +0.5661 ± 0.0198, i.e.
inside the scatter, and the field degraded (gap 0.075, rank 25.7 — the collapse signature).
The promised ~0.02 is not there.

### 4. Region coverage — this is where everything was

All at 1,120 frames, 3 draws, piece area held at ~145 mm² so only *where* and *how much*
changes:

| set | pieces | driven area | solve ρ | sim | gap | rank |
|---|---|---|---|---|---|---|
| DMN only | 16 | 2,349 mm² | 0.5268 | +0.4727 ± 0.0083 | 0.064 | 32.1 |
| sensory | 47 | 7,271 mm² | 0.7077 | +0.5661 ± 0.0198 | 0.043 | 45.6 |
| spread | 51 | 7,476 mm² | 0.7643 | +0.6552 ± 0.0073 | 0.030 | 57.7 |
| sensory+DMN | 63 | 9,620 mm² | 0.7873 | +0.6370 ± 0.0189 | 0.034 | 70.2 |
| spread ×2 | 100 | 14,616 mm² | 0.8755 | +0.7194 ± 0.0053 | 0.026 | 93.5 |
| spread ×3 | 148 | 21,866 mm² | 0.9249 | +0.7380 ± 0.0070 | 0.043 | 105.0 |

At matched area and matched piece count, spreading beats sensory by +0.089, and every
diagnostic moves the right way at once. Doubling the driven area buys +0.064 — but
trebling it buys only +0.019 more, and the Moran gap gets *worse* (0.043 against 0.026),
so coverage saturates somewhere around twice the sensory area.

**What this is not.** It is not a finding that the input is spread over the cortex. The
dominant patterned input to cortex enters through sensory areas; a farthest-point sample
that includes pOFC and TGd is the input prior switched off, not an alternative hypothesis
about it. The scores in the lower rows are what the machinery can do when nothing
constrains where the drive is applied.

**What it probably is.** Note the ordering: spread at 51 pieces beats sensory+DMN at 63
pieces and 2,000 mm² more. More channels and more area, worse score — so this is not
degrees of freedom, or that would reverse. What pays is *dispersion*: having a source near
every part of the sheet. The natural reading is a statement about the medium rather than
the input. If the fluid transported far enough, sources in sensory cortex could generate
distant FC structure by propagating there; that it helps instead to put an injector nearby
says the effective reach is too short, and the solve is buying with geography what the
dynamics will not deliver.

The ordering survives at 4,480 frames, where the scatter is an order of magnitude
smaller: spread +0.6972 ± 0.0022 against sensory+DMN +0.6863 ± 0.0065, still with 12
fewer pieces and 2,144 mm² less area. But the margin is only ~0.011, about 1.5 pooled sd,
so treat "dispersion, not count" as supported and not yet nailed down. Experiment 4 in the
list below is what would settle it.

### 5. Re-run the medium BO — wired up, then deliberately stopped

`bo_step` now solves against the normal-scored target (`xspec.normal_scores`, moved there
so `best_fit` and `bo_step` share one implementation), reports Pearson *and* Spearman, and
takes `--region-set`. It was pointed at the **spread** set, which on the reading above was
the wrong choice — if the medium is to be re-tuned it should be re-tuned on `sensory`, the
set the model actually claims:

```bash
python bo_step.py --region-set spread --split 50 --pad 1120 --realise 1120 --nfreq 192 \
  --calls 50 --initial 18 --workers 6 --target normal --resume --tag bo_spread_rank
```

~7 min a call. It was stopped after 10 evaluations: a many-hour search over the medium is
exploitation, and the coverage result says the exploratory questions are not exhausted.
Best of the 10 was +0.6345 against the incumbent medium's +0.6552 on the same set, so the
search had not yet beaten the medium it started from — but 10 draws is only the random
initialisation, so that says nothing either way. `results/bo_step/bo_spread_rank.pkl`
holds them; `--resume` picks them up.

Two operational notes for whenever it is run for real. It now checkpoints every call,
because one earlier run deadlocked when a pool worker died and sat blocked for three
hours with nothing written. And watch memory: `np.pad`ing the impulse responses to 1,120
frames costs 4.1 GB at 100 pieces and 6.1 GB at 148, which is what killed that worker.

## Two controls that changed how the numbers should be read

**The target is not the limit.** Split-half across the 99 NKI subjects, built the same way
the target was — including the double centring, which matters because it removes the
global component and so the most reliable part of the matrix:

```bash
python reliability.py --splits 3
```

Halves agree at ρ = 0.8801 ± 0.0026; Spearman-Brown gives the 99-subject target a
reliability of 0.9369, so the ceiling on the score is **0.968**. Target noise accounts for
essentially none of the gap. (A first pass that skipped the double centring gave 0.8896
and is the wrong quantity — it describes a matrix nobody fits.)

**The extra freedom is not what is fitting the target's noise.** Note this is a narrower
claim than it looks: fitting real structure through a mechanism that is not the real one
would pass this test. At 100 pieces the solve carries
595,900 real parameters against 499,500 solve edges, which on its own would make a better
score meaningless. `holdout.py` solves against a group FC from one half of the subjects
and scores against the other half, so nothing about the target's noise is shared:

| | solve, in sample | solve, held out | realised, in sample | realised, held out |
|---|---|---|---|---|
| sensory, K=47 | 0.6999 | 0.6617 | +0.5852 | +0.5584 |
| spread ×2, K=100 | 0.8668 | 0.8106 | +0.7518 | +0.7252 |

More channels overfit more at the solve stage (0.056 against 0.038) — but that extra
freedom does *not* survive into the realisation, where both sets lose the same +0.027.
Held out, 100 spread pieces still beat 47 sensory ones by +0.167. So the spread advantage
is not target noise — which leaves the mechanism question open, and that is the one that
matters.

## Three extensions built this session

Machinery in `regimes.py`, `connectome.py`, `units.py`; all wired into `best_fit.py`
behind flags that default to the existing path. **R=1 and lam=0 reproduce +0.5875 ±
0.0011 exactly**, so everything below is a comparison against an unchanged baseline.

### Switching medium — implemented, and it makes the fit worse

R media switched at epoch boundaries inside one run, field state carried across. Regimes
differ in the map grading of speed and damping (`--regime-span`), not a global scalar,
since `bo_step` found `c0` inert in per-step units.

| | solve rho | sim @4480 | gap | rank |
|---|---|---|---|---|
| R=1 baseline | 0.7077 | +0.5875 ± 0.0011 | 0.046 | 46.5 |
| R=2, identical media (null) | 0.7022 | +0.5941 (1 draw) | 0.060 | 54.1 |
| R=2, epoch 280 | 0.6540 | +0.4241 ± 0.0055 | 0.050 | 121.5 |
| R=3, epoch 280 | 0.7022 | +0.4703 ± 0.0324 | 0.043 | 76.1 |

**The quasi-static assumption fails, and that is the finding.** `regimes.epoch_profile`
reports the field variance 23-25x higher in the first 20 frames of an epoch than the last
20, and it is not the drive: measured per frame, drive power is flat to 1.00 across the
epoch and equal between regimes. The switch itself is injecting the transient. The state
`(h, ue)` is carried across continuously, but the energy `1/2 int (H|u|^2 + g h^2)`
depends on `H`, so a discontinuous change in the speed field is a discontinuous change in
energy. The run is then a sequence of kicks and ring-downs rather than a mixture of
stationary regimes - which is exactly what the solve assumes, and the score reflects the
mismatch.

**The null control settles it.** `--regime-span 0` switches between two IDENTICAL media,
so the machinery runs unchanged while the medium does not actually change: within-epoch
ratio **1.1 (quasi-static)**, regime variance ratio 1.04, sim +0.5941. The switching
machinery is therefore sound and costs nothing; the entire 23-25x transient, and the
entire loss from +0.59 to +0.42, comes from the medium changing.

Two other explanations are ruled out. The solve is not collapsing onto one regime: block
powers are 0.42/0.58 at R=2 and 0.18/0.57/0.26 at R=3. And it is not the drive amplitudes -
`realise_switching` originally lost the solved per-regime power, because `xspec.realise`
normalises each block to unit variance; fixing that moved the score by 0.003 (+0.4215 to
+0.4241), confirming the bug was real but not the cause.

**Next:** ramp the medium between regimes rather than jumping it - a ramp over a few decay
times removes the energy discontinuity and is closer to what arousal or neuromodulation
would actually do - then lengthen the epochs.

### Long-range structural coupling — implemented, and it also makes the fit worse

HCP-derived group-normative Glasser-360 connectome from ENIGMA Toolbox; the term is
linear, instantaneous and low rank, so the system stays LTI and the convex solve is
untouched.

| lam | solve rho | sim @4480 | gap | rank |
|---|---|---|---|---|
| 0 | 0.7077 | +0.5875 ± 0.0011 | 0.046 | 46.5 |
| 1 | 0.6509 | +0.5363 ± 0.0056 | 0.037 | 53.2 |
| 5 | 0.5527 | +0.4245 ± 0.0064 | 0.028 | 74.4 |

Monotone in lam, and already at the solve stage, before any realisation. Note the Moran
gap moves the OTHER way - 0.046 to 0.028 - so the coupling improves the spatial
autocorrelation match while making the edge ordering worse. Whatever it is adding is the
right kind of long-range structure by one measure and the wrong kind by the one being
scored. Adding transport that the
coverage result suggested was missing makes the transfer function a *worse* basis for the
target - which is worth understanding rather than tuning away, since it is the opposite of
what motivated the term.

Three traps were found and fixed on the way in, all silent:
- Stripping the `L_`/`R_` prefix when matching parcel names makes `L_V1` collide with
  `R_V1`; the right-hemisphere entry wins and **the wrong hemisphere loads**, looking
  entirely normal. The ordering check caught it.
- `"L_7AL_ROI".replace("L_", "")` also eats the `L_` inside `7AL_`, mangling every area
  whose name ends in L (PSL, SFL, 5L, 7AL, 7PL).
- Tractography weights have arbitrary scale, so `lam` meant nothing until `W` was
  normalised by its Laplacian's largest eigenvalue. Before that, lam=0.05 was already past
  the stability bound.

Filtering on distance residual alone does not give long-range connections - the top
residuals sit at the median distance of all connected pairs. `residual_W` now takes a
distance floor as well: beyond 60 mm, top 15% by residual, 82 edges at median 98 mm
against 41 mm for all connected pairs.

### Approximate units — the model runs ~100x faster than the data

`units.py` measures spread against **white-surface** geodesics (already millimetres, via
`ladder._white_graph`) while the dynamics run on the inflated mesh.

Spread is **1.96 mm/frame** (IQR 1.55-2.54 over the 47 sensory pieces). So:

- at a plausible cortical 300 mm/s, one frame is **6.5 ms** - about **1/100 of a TR**
- equivalently, if one frame *were* one TR, the implied spread would be **3.0 mm/s**,
  about 100x slower than cortical travelling waves

Either way the model sits two orders of magnitude off the data it is fitted to. The two
global calibrations (white/inflated area ratio 0.73 mm per unit, mean edge ratio 1.09)
differ by 33%, which is the width of the approximation and nowhere near enough to close a
factor of 100.

**And closing that gap by filtering costs the fit.** A temporal low-pass between field and
observable is linear, so it multiplies into `H` and the solve is unchanged
(`--smooth`, FWHM in frames; `xspec.transfer` and `score_realisation` share one kernel).
At FWHM 8 frames: solve rho 0.7077 -> 0.5967, sim +0.5875 -> +0.5171, gap 0.046 -> 0.126,
and **field rank collapses from 46.5 to 7.2**. The structure the fit depends on lives at
frequencies far above anything fMRI could observe.

## Priority experiments

1. **Ramp the switch, then lengthen the epochs.** The null control shows the machinery is
   clean and the transient is the medium change itself, so the switching result so far is a
   measurement of switching transients rather than of a time-varying medium. Ramp the
   medium over a few decay times to remove the energy discontinuity, then epochs of 1120+.
2. **Understand why coupling hurts** before tuning it. It is monotone in lam at the solve
   stage, so it is a statement about the transfer function, not about realisation noise.
   Candidates: the parcel-mean projection is too coarse (180 parcels against 47 driven
   pieces), the Laplacian form removes as much as it adds, or long-range transport genuinely
   does not help this objective - which would undercut the reading of the coverage result.
3. **Take the 100x seriously.** It is the largest unexplained number in the project. Either
   the medium is far too fast for the timescale it is fitted at, or the frame is not the
   observable and a haemodynamic stage is missing - and the smoothing result says that
   stage costs most of the fit when added naively.
4. `reach.py` on the current medium, as before.

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
