# CCFD — plan for the next session

Fitting a rotating shallow-water field on the cortical surface to group resting-state FC,
by solving for the input rather than searching for it.

## Where things stand

**The model is still the sensory one: sim +0.5875 ± 0.0011** (3 draws, 4,480 frames),
Moran gap 0.046, field rank 46.5.

```bash
python best_fit.py --frames 4480 --draws 3
```

That is the number to quote. It is the constrained, falsifiable model — sensory cortex is
where the dominant patterned input actually enters, so a fit that drives it is making a
claim, and last session's +0.5661 was the same configuration at a shorter (and, see below,
flattering) realisation length.

**Relaxing where the input enters scores much higher, and that is a measurement, not a
result.** Driving 100 pieces spread over the whole cortex reaches +0.7653 ± 0.0015. But 24
farthest-point parcels including pOFC and TGd are not a hypothesis about input; they are
the input prior switched off. Read the gap 0.588 → 0.765 as an upper bound on what the
medium and solve can do when nothing constrains where the drive is applied — and as a
diagnostic of the medium, for the reason in section 4.

**The pipeline** is unchanged and still the right one. The medium is linear, so the FC
depends on the input only through its cross-spectral density `S(f)`, which is solved
convexly:

1. impulse response per driven piece, FFT along time → `H(f)`
2. solve `max corr(Σ_f H S H^H, target)` over `S(f) ⪰ 0` by projected gradient
3. realise a drive by drawing `z_f = L_f η` per frequency, `S = L L^H`; inverse FFT
4. simulate, score with Spearman over a fixed 2M edge sample

**Current configuration** (all in `best_fit.py:BEST_X`): 47 equal-area sensory pieces of
~145 mm²; per-step damping 6.2e-04, rotation 1.1e-05, boundary absorption 1.8e-03, cadence
33 steps/frame; impulse window 280 frames zero-padded to 1,120, 118 usable frequencies;
solved against the normal-scored target on 1,000 medoid vertices. `--regions` selects
other driven sets (`spread`, `sensory+dmn`, `dmn`) for the diagnostic runs.

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

## Priority experiments

1. **Does the medium reach far enough?** This is the top question, and `reach.py` already
   asks it — and has been run once before, on an older medium and the whole-parcel region
   set, where it found the target's eigenpatterns *inside* the reachable span (recorded in
   the docstring of the now-deleted `inverse.py`, which existed because of that result).
   So the question is not whether the span is adequate in principle but whether it still
   is for the current per-step medium and the 47 sensory pieces, and whether "inside the
   span" survives being asked about the leading eigenpatterns rather than all of them: the model is linear, so the fields obtainable from a region set are the span
   of its impulse responses, and if the target's leading FC eigenpatterns lie outside that
   span, no input whatsoever can succeed and the limit is the fluid rather than the drive.
   Run it on the current per-step medium with the 47 sensory pieces, and again with
   `spread` — if sensory-only is structurally capped and spread is not, that confirms the
   reading above and points the fix at the medium's reach, damping and speed rather than
   at adding injectors. `reach.py` still hardcodes the old `run_ou` constants and whole
   parcels; it needs the current medium and the piece profiles wired in first.
2. **A control for the medium itself.** Replace the fluid with something trivial (pure
   diffusion, or geodesic smoothing of the drive) at matched K and driven area, on both
   `sensory` and `spread`. If the score barely moves, the model is a flexible spatial
   basis rather than a claim about dynamics — and the spread advantage would then be
   basis size, not transport.
3. **Fix the realisation grid.** Draw on the solved bins and tile to length instead of
   interpolating, and re-run the length sweep. If the 1,120 bonus disappears and the curve
   becomes monotone, every number in the tables above shifts and the comparisons get
   cleaner. Cheap, and it touches everything.
4. **Alternative spread samples.** The sampler is deterministic, so both "spread beats
   sensory" and the sharper "dispersion beats count" rest on a single draw of the set.
   Several alternatives (different starting parcel, or random with a minimum separation)
   at matched area would turn a 1.5-sd ordering into a distribution — worth doing before
   the diagnostic is leaned on.
5. **Re-tune the medium on `sensory`**, not on spread, if the BO is run at all. It is
   exploitation, so it should wait until 1 and 2 have said what the medium ought to be
   doing differently.

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
